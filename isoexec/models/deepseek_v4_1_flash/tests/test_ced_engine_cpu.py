"""The engine binding's decoder-row plan (``engine_ced``): a function of the prompt length only, whatever the
scheduler does (chunk sizes, which requests share a step, preemption and recompute); the CPU reference engine
runs on these plans and equals the trainer's per-segment scoring."""
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from isoexec.models.deepseek_v4_1_flash import reference as R
from isoexec.models.deepseek_v4_1_flash.config import BOUNDED_REPLAY, FULL_DECODER
from isoexec.models.deepseek_v4_1_flash.engine_ced import plan_from_runner, plan_step, refuse_request
from isoexec.models.deepseek_v4_1_flash.program import Turn


def _simulate(prompts, outputs, W, mode, seed, max_tokens=37, preempt=0.05):
    """A scheduler: each step takes a random subset of live requests, prefill chunks of random size within a
    token budget, one token per decoding request; a request may be preempted and recomputed from scratch with
    its outputs so far (prefill_len > prompt_len). Returns each request's decoder positions in its last life."""
    rng = random.Random(seed)
    n = len(prompts)
    total = [p + o for p, o in zip(prompts, outputs)]
    computed, rows = [0] * n, [[] for _ in range(n)]
    while any(c < t for c, t in zip(computed, total)):
        live = [i for i in range(n) if computed[i] < total[i]]
        step = sorted(rng.sample(live, rng.randint(1, len(live))))
        budget, sched = max_tokens, []
        for i in step:
            if rng.random() < preempt and computed[i] > prompts[i]:
                computed[i], rows[i] = 0, []                 # preempted: recompute prompt + outputs so far
            target = max(prompts[i], computed[i] + 1) if computed[i] < prompts[i] else computed[i] + 1
            want = min(target - computed[i], budget, total[i] - computed[i])
            if computed[i] < prompts[i]:
                want = rng.randint(1, max(1, want))
            sched.append(max(want, 0))
            budget -= sched[-1]
        keep = [k for k, q in enumerate(sched) if q > 0]
        step, sched = [step[k] for k in keep], [sched[k] for k in keep]
        if not step:
            continue
        plan = plan_step([prompts[i] for i in step], [computed[i] for i in step], sched, window=W, mode=mode)
        q0 = np.concatenate(([0], np.cumsum(sched)))
        for k, i in enumerate(step):
            mine = plan.rows[plan.cu_rows[k]:plan.cu_rows[k + 1]]
            assert ((mine >= q0[k]) & (mine < q0[k + 1])).all()
            rows[i] += [computed[i] + int(r - q0[k]) for r in mine]
            assert plan.segment_start[k] == (max(0, prompts[i] - W) if mode == BOUNDED_REPLAY else 0)
        if all(computed[i] >= prompts[i] for i in step) and all(q == 1 for q in sched):
            assert plan.all_rows                               # decode-only steps: every token
        for k, i in enumerate(step):
            computed[i] += sched[k]
    return rows


@pytest.mark.parametrize("mode", [BOUNDED_REPLAY, FULL_DECODER])
@pytest.mark.parametrize("seed", range(6))
def test_decoder_rows_are_the_segment_under_any_schedule(mode, seed):
    W = 128
    rng = random.Random(100 + seed)
    prompts = [rng.choice([1, 5, 127, 128, 129, 300, 1000]) for _ in range(5)]
    outputs = [rng.randint(1, 20) for _ in prompts]
    rows = _simulate(prompts, outputs, W, mode, seed, max_tokens=rng.choice([16, 64, 512]))
    for p, o, got in zip(prompts, outputs, rows):
        s = max(0, p - W) if mode == BOUNDED_REPLAY else 0
        assert got == list(range(s, p + o))


def test_runner_arrays_feed_the_plan():
    batch = SimpleNamespace(num_reqs=3, idx_mapping_np=np.array([4, 0, 2]),
                            num_computed_tokens_np=np.array([0, 200, 7]), num_scheduled_tokens=np.array([300, 1, 5]),
                            prefill_len_np=np.array([300, 150, 9]))
    states = SimpleNamespace(prompt_len=SimpleNamespace(np=np.array([150, 0, 9, 0, 300])))
    plan = plan_from_runner(batch, states, window=128, mode=BOUNDED_REPLAY)
    assert plan.segment_start.tolist() == [172, 22, 0]
    assert plan.rows.tolist() == list(range(172, 300)) + [300] + list(range(301, 306))
    assert plan.cu_rows.tolist() == [0, 128, 129, 134] and not plan.all_rows


def test_prompt_logprobs_and_routed_experts_are_refused_under_replay():
    for kw, code in ((dict(prompt_logprobs=True), "CED_PROMPT_LOGPROBS"),
                     (dict(prompt_logprob_token_ids=True), "CED_PROMPT_LOGPROBS"),
                     (dict(return_routed_experts=True), "CED_ROUTED_EXPERTS")):
        args = dict(prompt_logprobs=False, prompt_logprob_token_ids=False, return_routed_experts=False) | kw
        with pytest.raises(ValueError, match=code):
            refuse_request(BOUNDED_REPLAY, **args)
        refuse_request(FULL_DECODER, **args)


def test_preempted_and_recomputed_request_generates_the_same_bits():
    ref = R.tiny(seed=0, window=8)
    prompt, gen = 19, 7
    seq = torch.randint(0, 64, (prompt + gen,), generator=torch.Generator().manual_seed(5))
    engine = R.ReferenceEngine(ref, BOUNDED_REPLAY)
    whole, _ = R.rollout(engine, seq[:prompt], seq[prompt:], [7, 12])
    # preempted after 3 generated tokens: a fresh request recomputes prompt + 3 outputs, then decodes on
    st = engine.new_request(prompt)
    resumed = [engine.prefill(st, seq[:prompt + 3], [10, 12])]
    for t in seq[prompt + 3:-1].tolist():
        resumed.append(engine.decode(st, t))
    assert torch.equal(torch.stack(resumed), whole[3:])
    tl, _, _, _ = R.trainer_logits(ref, [seq], [[Turn(prompt, prompt + gen)]], BOUNDED_REPLAY)
    assert torch.equal(tl, whole)


def _runner_objects(max_reqs, max_tokens):
    """vLLM 0.33 V2's own objects, as the runner builds them (CPU device)."""
    pytest.importorskip("vllm")
    from vllm.v1.worker.gpu.input_batch import InputBuffers
    from vllm.v1.worker.gpu.states import RequestState
    buffers = InputBuffers(max_reqs, max_tokens, torch.device("cpu"))
    try:
        states = RequestState(max_reqs, 4096, max_tokens, 0, 64, torch.device("cpu"))
    except Exception:            # UVA-backed fields need CUDA: the dummy path never reads them
        states = SimpleNamespace(prompt_len=SimpleNamespace(np=np.zeros(max_reqs, dtype=np.int32)))
    return buffers, states


@pytest.mark.parametrize("num_tokens,num_reqs,context", [
    (2048, 64, 0),      # profile_run: max_num_batched_tokens over min(tokens, max_num_seqs) dummy requests
    (1, 1, 0),          # the smallest warm-up / capture size
    (64, 64, 0),        # uniform-decode warm-up / capture at the largest size
    (64, 64, 1000),     # a decode-like dummy with fabricated context (set_dummy_context)
    (37, 5, 0),         # an uneven split (extra tokens on the last requests)
])
def test_vllm_dummy_steps_take_the_dummy_plan(num_tokens, num_reqs, context):
    from vllm.v1.worker.gpu.input_batch import InputBatch

    from isoexec.models.deepseek_v4_1_flash.engine_ced import dummy_split, is_dummy_batch
    buffers, states = _runner_objects(64, 2048)
    batch = InputBatch.make_dummy(num_reqs, num_tokens, buffers)
    if context:
        batch.num_computed_tokens_np.fill(context)           # what set_dummy_context does on the host
    assert is_dummy_batch(batch)
    plan = plan_from_runner(batch, states, window=128, mode=BOUNDED_REPLAY)
    assert plan.dummy and plan.all_rows and plan.tokens == num_tokens
    assert plan.rows.tolist() == list(range(num_tokens)) and not plan.segment_start.any()
    assert plan.cu_rows.tolist() == batch.query_start_loc_np.tolist()
    assert dummy_split(num_reqs, num_tokens) == batch.num_scheduled_tokens.tolist()   # prepare_dummy_inputs


def test_a_real_request_with_an_empty_prompt_is_still_refused():
    batch = SimpleNamespace(num_reqs=2, idx_mapping_np=np.array([0, 1]), num_computed_tokens_np=np.array([0, 0]),
                            num_scheduled_tokens=np.array([3, 4]), prefill_len_np=np.array([3, 0]))
    states = SimpleNamespace(prompt_len=SimpleNamespace(np=np.array([3, 0])))
    with pytest.raises(ValueError, match="non-empty prompt"):
        plan_from_runner(batch, states, window=128, mode=BOUNDED_REPLAY)


def test_vllm_warmup_requests_are_not_refused_but_real_ones_are():
    """vLLM's graph-mode kernel warm-up submits requests ``_warmup_{i}_`` with ``SamplingParams.for_sampler_warmup()``
    (prompt logprobs on): smoke3 g2048 died on the prompt-logprob refusal (defect 6)."""
    import inspect
    pytest.importorskip("vllm")
    from vllm.sampling_params import SamplingParams
    from vllm.v1.worker.gpu import warmup

    from isoexec.models.deepseek_v4_1_flash.engine_ced import VLLM_WARMUP_PREFIXES
    source = inspect.getsource(warmup)
    assert 'f"_warmup_{i}_"' in source and '"_v2_mixed_warmup"' in source      # the ids vLLM 0.33 builds
    assert all(p in ("_warmup_", "_v2_mixed_warmup") for p in VLLM_WARMUP_PREFIXES)
    params = SamplingParams.for_sampler_warmup()
    assert params.prompt_logprobs is not None
    kw = dict(prompt_logprobs=True, prompt_logprob_token_ids=False, return_routed_experts=False)
    refuse_request(BOUNDED_REPLAY, req_id="_warmup_3_", **kw)
    refuse_request(BOUNDED_REPLAY, req_id="_v2_mixed_warmup_prefill_", **kw)
    with pytest.raises(ValueError, match="CED_PROMPT_LOGPROBS"):
        refuse_request(BOUNDED_REPLAY, req_id="dapo-000-00", **kw)


def _engine_embed():
    from isoexec.compiler.source import ModelSource
    from isoexec.models.deepseek_v4_1_flash.tests.test_deepseek_v4_1_flash_cpu import placement, raw
    model = ModelSource("isoexec.models.deepseek_v4_1_flash").build_meta(raw("debug_config.json"))
    embed = model.embed
    embed.apply_placement(placement(side="engine"), site="embed")
    return embed


def test_engine_embed_refuses_to_hash_without_the_step_sequence_context():
    """smoke5: every real engine step hashed the packed batch as ONE sequence (seq was None), so Engram n-grams
    crossed requests and chunks (defects 7 and 9). An engine forward without seq is now a named refusal."""
    embed = _engine_embed()
    with pytest.raises(ValueError, match="ENGRAM_NO_SEQUENCE_CONTEXT"):
        embed(torch.zeros((1, 4), dtype=torch.long))


def _hash_standin(ids, token_map, primes, offsets, multipliers, pad, *, state=None, validate=True):
    """CPU stand-in of engram_sm100.ngram_hash (CUDA-only): a deterministic function of each position's window of
    N compressed ids (the N-1 history ids oldest first, -1 / before the start = pad, then the token)."""
    L, Nm1, H = primes.shape
    B, T = ids.shape
    comp = token_map[ids]
    hist = torch.full((B, Nm1), -1, dtype=torch.int64) if state is None else state.clone()
    out = torch.empty((B, T, L, Nm1 * H), dtype=torch.int64)
    for b in range(B):
        window = [int(v) for v in hist[b]]
        for t in range(T):
            window = window[1:] + [int(comp[b, t])]
            full = [pad if v < 0 else v for v in window]
            for c in range(L * Nm1 * H):
                out[b, t].view(-1)[c] = sum((v + 1) * (c + 7) ** k for k, v in enumerate(full)) % 1000003
    return out


def test_engine_engram_history_is_per_request_across_chunks_and_batch_orders(monkeypatch):
    """The binding's per-request hash (``embed._engine_hashes`` over vLLM's lookback window: the ``N - 1`` ids before
    each request's chunk, newest first, -1 none) equals hashing each whole sequence, for any packing / chunking."""
    from isoexec.ops.embed.engram_sm100 import ops as engram_ops
    monkeypatch.setattr(engram_ops, "ngram_hash", _hash_standin)
    embed = _engine_embed()
    cfg = embed.cfg
    token_map = torch.arange(cfg.vocab_size) % cfg.engram_compressed_vocab_size
    primes = torch.ones((1, cfg.engram_max_ngram_size - 1, 2), dtype=torch.int64)
    embed._hash = {"cpu": (token_map, primes, primes.flatten(1), primes, int(token_map[cfg.engram_pad_token_id]))}
    depth = cfg.engram_max_ngram_size - 1
    seqs = [torch.randint(0, cfg.vocab_size, (n,), generator=torch.Generator().manual_seed(n)) for n in (9, 4, 13)]
    whole = [_hash_standin(s.view(1, -1), *embed._hash["cpu"][:4], embed._hash["cpu"][4])[0] for s in seqs]

    def step(parts):
        """parts: (sequence index, start, length) packed in this order -> per-part hashes from _engine_hashes."""
        ids = torch.cat([seqs[i][a:a + n] for i, a, n in parts]).view(1, -1)
        lookback = torch.full((len(parts), depth), -1, dtype=torch.int64)
        for r, (i, a, _) in enumerate(parts):
            prev = seqs[i][max(0, a - depth):a].flip(0)               # newest first
            lookback[r, :prev.numel()] = prev
        starts = [0]
        for _, _, n in parts:
            starts.append(starts[-1] + n)
        got = embed._engine_hashes(ids, lookback, starts)[0]
        return [got[starts[r]:starts[r + 1]] for r in range(len(parts))]

    for schedule in ([[(0, 0, 9), (1, 0, 4), (2, 0, 13)]],                       # one step, order 0 1 2
                     [[(2, 0, 5), (0, 0, 3)], [(1, 0, 4), (2, 5, 8), (0, 3, 6)]],  # chunks, other order
                     [[(0, 0, 1)], [(0, 1, 1)], [(2, 0, 13), (0, 2, 7), (1, 0, 4)]]):
        for parts in schedule:
            for (i, a, n), h in zip(parts, step(parts)):
                assert torch.equal(h, whole[i][a:a + n]), (schedule, i, a)


def test_piecewise_capture_makes_each_component_an_eager_break_into_captured_outputs(monkeypatch):
    """FULL_AND_PIECEWISE: while a breakable segment records, a component call becomes ONE eager break that writes
    into outputs of its abstract shapes (the addresses the next segment captures) and reruns on replay with the
    model state's inputs it saw at capture (the Engram lookback buffer); outside a capture it runs inline."""
    pytest.importorskip("vllm")
    from isoexec.models.deepseek_v4_1_flash import piecewise
    from isoexec.runtimes.vllm import _inputs, graph_breaks
    replays = []
    monkeypatch.setattr(graph_breaks, "piecewise_capturing", lambda: True)
    monkeypatch.setattr(graph_breaks, "_eager_break", lambda fn: (lambda *a, **k: (replays.append((a, k)), fn(*a, **k))[1]))
    monkeypatch.setattr(piecewise, "_BREAK", None)
    seen = []

    class Comp:
        def abstract_output(self, x, seq=None):
            return (torch.empty_like(x), torch.empty((x.shape[0], 2)))

        def impl(self, x, seq=None):
            seen.append(dict(_inputs.current_engine_inputs() or {}))
            return x * 2, torch.ones((x.shape[0], 2))
    comp, x = Comp(), torch.arange(6.0).view(3, 2)
    with _inputs.engine_inputs({"lookback_token_ids": torch.zeros(1)}):
        a, b = piecewise.run(comp, comp.impl, (x,), {"seq": None})
    assert torch.equal(a, x * 2) and torch.equal(b, torch.ones((3, 2))) and len(replays) == 1
    impl, args, kwargs, inputs, outs = replays[0][0]
    assert outs[0] is a and outs[1] is b and "lookback_token_ids" in inputs      # replay writes the same tensors
    x.add_(1)
    piecewise._run_into(impl, args, kwargs, inputs, outs)                        # a replay, outside the context
    assert torch.equal(a, x * 2) and "lookback_token_ids" in seen[-1]
    monkeypatch.setattr(graph_breaks, "piecewise_capturing", lambda: False)
    assert piecewise.run(comp, comp.impl, (x,), {})[0] is not a                  # inline outside a capture


def test_piecewise_core_break_writes_the_callers_outputs_and_records_host_time(monkeypatch):
    """``core`` structure: an attention-core break writes into tensors the caller allocated while recording (the
    addresses the next segment reads), reruns on replay, and records its host time per break body."""
    pytest.importorskip("vllm")
    from vllm.utils import torch_utils
    from isoexec.models.deepseek_v4_1_flash import piecewise
    from isoexec.runtimes.vllm import graph_breaks
    replays = []
    monkeypatch.setattr(graph_breaks, "_eager_break", lambda fn: (lambda *a, **k: (replays.append((a, k)), fn(*a, **k))[1]))
    monkeypatch.setattr(torch_utils, "weak_ref_tensor", lambda t: t)
    monkeypatch.setattr(piecewise, "_BREAK", None)
    monkeypatch.setattr(piecewise, "BREAK_HOST", {})

    def body(x):
        return x * 2
    x = torch.arange(6.0).view(3, 2)
    out = torch.empty_like(x)
    assert piecewise.core(body, (x,), (out,))[0] is out and torch.equal(out, x * 2)
    impl, args, kwargs, inputs, outs = replays[0][0]
    x.add_(1)
    piecewise._run_into(impl, args, kwargs, inputs, outs)                       # a replay
    assert torch.equal(out, x * 2)
    assert piecewise.BREAK_HOST[body.__qualname__][0] == 2
    piecewise.core(lambda: None, (), ())                                         # a break that only writes pages


def test_piecewise_break_mode_is_agreed_and_refuses_unknown_values(monkeypatch):
    from isoexec.models.deepseek_v4_1_flash import piecewise
    monkeypatch.setattr(piecewise, "_MODE", {})
    monkeypatch.setenv(piecewise.BREAKS_ENV, "core")
    assert piecewise.agree_mode(None) == "core" and piecewise.mode() == "core"
    monkeypatch.setattr(piecewise, "_MODE", {})
    monkeypatch.setenv(piecewise.BREAKS_ENV, "layers")
    with pytest.raises(ValueError, match="ISOEXEC_V41_PIECEWISE_BREAKS"):
        piecewise.agree_mode(None)
    monkeypatch.setattr(piecewise, "_MODE", {})
    monkeypatch.delenv(piecewise.BREAKS_ENV)
    assert piecewise.agree_mode(None) == "component"


def test_decoder_rows_from_plan_carry_the_plan_values():
    """``DecoderRows.from_plan``: the plan's rows, positions / sequence gathered at them, the segment start per row."""
    from isoexec.models.deepseek_v4_1_flash.program import DecoderRows
    plan = plan_step([5, 3, 9], [0, 3, 2], [5, 1, 4], window=2, mode=BOUNDED_REPLAY)
    T = 10
    positions = torch.arange(100, 100 + T)
    tok = torch.tensor([0] * 5 + [1] + [2] * 4, dtype=torch.int32)
    cu = torch.tensor([0, 5, 6, 10])
    a = DecoderRows.from_plan(plan, positions, tok, cu)
    idx = torch.from_numpy(plan.rows)
    assert torch.equal(a.index, idx) and torch.equal(a.positions, positions[idx])
    assert torch.equal(a.sequence, tok[idx].long()) and a.cu_seqlens is cu
    assert torch.equal(a.window_floor, torch.from_numpy(plan.segment_start)[tok[idx].long()])
