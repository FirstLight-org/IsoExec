"""Production bindings, real convolution, and stateful engine/trainer composition.

Random weights qualify the numerical composition without a checkpoint. These are
single-device component tests, not a vLLM scheduler or full Qwen model launch.
"""

from __future__ import annotations

import pytest
import torch

from isoexec.ops._testing import bitwise_equal
from isoexec.ops.gdn.gdn_cpr_sm90 import engine_state  # deliberately before conv install
from isoexec.ops.gdn.gdn_cpr_sm90.tests._fixtures import SM90

pytestmark = SM90
HK, HV, D = 2, 4, 128


@pytest.fixture(params=[None, "two_pass", "fused", "prepared"], ids=["auto", "two_pass", "fused", "prepared"])
def production(request, monkeypatch):
    from isoexec.ops import _dispatch, _resolve
    from isoexec.ops.gdn import trainer
    from isoexec.ops.gdn.conv_sm90 import integration as conv
    from isoexec.ops.gdn.gdn_cpr_sm90 import ops

    monkeypatch.setitem(_dispatch.FAMILIES["gdn"].archs, "sm90", "gdn.gdn_cpr_sm90")
    _resolve._registry.cache_clear()
    trainer._conv_kernel.cache_clear()
    for flag in ("KERNEL", "NATIVE_KERNELS", "CHUNKED_PREFILL"):
        monkeypatch.setenv(f"ISOEXEC_GDN_{flag}", "cpr" if flag == "KERNEL" else "1")
    for flag in ("NATIVE_STATE", "CPR_APC", "CPR_SLEEP"):
        monkeypatch.delenv(f"ISOEXEC_GDN_{flag}", raising=False)
    installed = bool(conv._INSTALLED)
    layer_count = len(engine_state.CPR_LAYERS)
    conv.uninstall()
    # engine_state has already been imported: rebinding the module must still work.
    conv.install()
    assert engine_state.conv.causal_conv1d_fn is conv.causal_conv1d_fn
    assert engine_state.conv.causal_conv1d_update is conv.causal_conv1d_update
    try:
        with ops._prefill_mode(request.param):
            yield conv
    finally:
        del engine_state.CPR_LAYERS[layer_count:]
        if not installed:
            conv.uninstall()
        _resolve._registry.cache_clear()
        trainer._conv_kernel.cache_clear()


def _pool(w, A, dt, capacity=4):
    from isoexec.ops.gdn import build_state_pool

    pool = build_state_pool(max_num_seqs=capacity, chunk_size=64, conv_weight=w, conv_bias=None,
                            A_log=A, dt_bias=dt, num_k_heads=HK, head_k_dim=D,
                            num_v_heads=HV, head_v_dim=D, activation="silu",
                            dtype=torch.bfloat16, device="cuda")
    assert isinstance(pool, engine_state.CprSm90GDN)
    pool.note_slot_positions(None)  # requests are lifecycle-owned, as in the engine driver
    return pool


def _random(seed=73):
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def rand(*shape, scale=0.4):
        return (torch.randn(*shape, generator=gen, device="cuda") * scale).bfloat16()

    return rand


def _cu(lengths):
    return [0, *torch.tensor(lengths).cumsum(0).tolist()]


def test_real_conv_trainer_vs_chunked_engine_graph_and_reused_slots(production):
    from isoexec.ops.gdn.gdn_cpr_sm90 import ops
    from isoexec.ops.table import OpTable

    witness = ops.census()
    table = OpTable()
    conv = table.bind("conv.short", site="qualification.conv")
    core = table.bind("gdn.core", site="qualification.core")
    assert conv.impl == "sm90_cuda_v1"
    assert core.impl == "cuda_cpr_v1" and core.version == 1
    assert core.provider == "ops.gdn.gdn_cpr_sm90"
    rand = _random()
    lengths, splits, ndecode = [137, 89, 261], [37, 64, 129], 5
    cu = _cu(lengths)
    T = cu[-1]
    x = rand(T, (2 * HK + HV) * D).requires_grad_()
    w = rand(x.shape[1], 4).requires_grad_()
    a, b = rand(T, HV), rand(T, HV)
    A, dt = torch.full((HV,), -4., device="cuda"), torch.full((HV,), -1., device="cuda")
    cut = torch.tensor(cu, device="cuda", dtype=torch.int32)
    y = conv.fn(x, w[:, None], cut)
    assert not bitwise_equal(y, x)  # a real history-dependent filter and SiLU
    q, k, v = y.split([HK * D, HK * D, HV * D], dim=-1)
    want = core.fn(q.reshape(T, HK, D), k.reshape(T, HK, D), v.reshape(T, HV, D), a, b, A, dt, cut)
    assert want.requires_grad  # exercises the trainer save path, not no-grad prefill
    want = want.detach()
    counts = production.stats()
    for graph_decode in (False, True):
        pool = _pool(w.detach(), A, dt)
        slots_cpu = [11, 3, 19]
        slots = torch.tensor(slots_cpu, device="cuda", dtype=torch.int32)
        got = torch.empty_like(want)
        taken = [0] * 3
        for ends in (splits, [L - ndecode for L in lengths]):
            spans = [torch.arange(cu[i] + taken[i], cu[i] + ends[i]) for i in range(3)]
            idx = torch.cat(spans).cuda()
            qsl = _cu([len(s) for s in spans])
            got[idx] = pool.prefill(slots, slots_cpu, x.detach()[idx], a[idx], b[idx], qsl,
                                    has_initial_state=[t > 0 for t in taken])
            taken = ends
        # Capture with mutable slot ids and padded lanes. Replay resolves device
        # slot maps, not a captured host order. Warmup/capture state is restored.
        static_x, static_a, static_b = rand(5, x.shape[1]), rand(5, HV), rand(5, HV)
        static_slots = torch.tensor([11, -1, 3, 19, 999], device="cuda", dtype=torch.int32)
        state0, conv0 = pool.ssm_state.clone(), pool.conv_state.clone()
        if graph_decode:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                pool.decode(static_slots, static_x, static_a, static_b)
            torch.cuda.current_stream().wait_stream(stream)
            pool.ssm_state.copy_(state0)
            pool.conv_state.copy_(conv0)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_out = pool.decode(static_slots, static_x, static_a, static_b)
            pool.ssm_state.copy_(state0)
            pool.conv_state.copy_(conv0)
        for step in range(ndecode):
            order = [(step + i) % 3 for i in range(3)]
            idx = torch.tensor([cu[i] + lengths[i] - ndecode + step for i in order], device="cuda")
            lanes = torch.tensor([0, 2, 3], device="cuda")
            static_slots.copy_(torch.tensor([slots_cpu[order[0]], -1, slots_cpu[order[1]],
                                             slots_cpu[order[2]], 999], device="cuda", dtype=torch.int32))
            static_x.zero_()
            static_x[lanes], static_a[lanes], static_b[lanes] = x.detach()[idx], a[idx], b[idx]
            if graph_decode:
                graph.replay()
                out = static_out
            else:
                out = pool.decode(static_slots, static_x, static_a, static_b)
            got[idx] = out[lanes]
        assert bitwise_equal(got, want)
        assert bitwise_equal(pool.ssm_state[0], state0[0])
        assert bitwise_equal(pool.conv_state[0], conv0[0])
        # A released row retains dirty bytes, but a new request must start fresh
        # in both the conv window and CPR record.
        old_row = pool.release_slot(slots_cpu[0])
        pool.flush_slot_map()
        new_slot = 23
        sub = torch.arange(65, device="cuda")
        reuse = pool.prefill(torch.tensor([new_slot], device="cuda", dtype=torch.int32), [new_slot],
                            x.detach()[sub], a[sub], b[sub], [0, len(sub)], has_initial_state=[False])
        assert pool._slot2row[new_slot] == old_row
        assert bitwise_equal(reuse, want[sub])
        static_slots.copy_(torch.tensor([new_slot, -1, -1, -1, 999], device="cuda", dtype=torch.int32))
        static_x.zero_()
        static_x[0], static_a[0], static_b[0] = x.detach()[65], a[65], b[65]
        if graph_decode:
            graph.replay()
            out = static_out
        else:
            out = pool.decode(static_slots, static_x, static_a, static_b)
        assert bitwise_equal(out[0], want[65])
    after = production.stats()
    assert after["varlen"] > counts["varlen"]
    assert after["update"] > counts["update"]
    calls = ops.census()
    for name in ("trainer_forward", "prefill", "decode"):
        assert calls[name] > witness[name], name


def test_nn_gdn_and_sampled_logprobs_match_chunked_graph_engine(production):
    """Full GDN mixer with random weights, real host core adapter and score ops.

    The tiny driver supplies scheduler decisions directly; it does not emulate
    attention metadata or claim to exercise either scheduler/materialization path.
    """
    from isoexec.nn.gdn import GDN
    from isoexec.ops.collectives.pik_bootstrap import ensure_pik
    from isoexec.ops.table import OpTable
    from isoexec.schema import Axis, Placement, make_sequence_context

    from isoexec.runtimes.engine_ops import gdn_core_resolution

    ensure_pik()  # the runtime normally installs the canonical vendored module at startup
    table = OpTable()
    one = Axis(1, 0, None)

    def component(side):
        model = GDN(hidden=128, key_heads=HK, value_heads=HV, key_head_dim=D,
                    value_head_dim=D, conv_kernel=4, device="cuda")
        placement = Placement(tp=one, ep=one, dp=one, pp=one, etp=one, cp=one, side=side, leaves=8)
        model.apply_placement(placement, site="qualification.mixer", op_table=table)
        return model

    rand = _random(941)
    trainer, engine = component("trainer"), component("engine")
    with torch.no_grad():
        for name, p in trainer.named_parameters():
            p.copy_(rand(*p.shape, scale=0.04 if p.ndim == 2 else 0.2))
        trainer.A_log.fill_(-4)
        trainer.dt_bias.fill_(-1)
        engine.load_state_dict(trainer.state_dict())
    pool = _pool(engine.conv_w[:, 0].detach(), engine.A_log.detach(), engine.dt_bias.detach())

    class Driver:
        decode = False
        slots_cpu = [11, 3]
        slots = torch.tensor(slots_cpu, device="cuda", dtype=torch.int32)
        qsl = None
        has = False

        def core(self, x, a, b, facts=None):
            assert facts is None
            if self.decode:
                return pool.decode(self.slots, x, a, b)
            return pool.prefill(self.slots, self.slots_cpu, x, a, b, self.qsl,
                                has_initial_state=[self.has] * len(self.slots_cpu))

    driver = Driver()

    def stage(q, k, v, alpha, beta, A_log, dt_bias, cu_seqlens, *, cu_seqlens_cpu=None):
        # The native state layer's composition (runtimes/vllm/_gdn.py), with the driver standing in
        # for vLLM's GDN metadata.
        del A_log, dt_bias, cu_seqlens, cu_seqlens_cpu
        return driver.core(torch.cat((q.flatten(1), k.flatten(1), v.flatten(1)), dim=-1), alpha, beta)

    bound = gdn_core_resolution("qualification.mixer", stage, table=table, provider="runtimes.vllm")
    engine.attach_core(bound)
    assert bound.impl == trainer.slots.resolved()["core"].impl == "cuda_cpr_v1"
    assert bound.owns["conv.short"] == trainer.slots.resolved()["conv"].impl == "sm90_cuda_v1"
    head = table.bind("logprobs.head", site="qualification.head").fn
    logprobs = table.bind("logprobs.log_softmax", site="qualification.logprobs").fn
    vocab_weight = rand(1024, 128)
    lengths, splits = [130, 194], [37, 65]
    cu, T = _cu(lengths), sum(lengths)
    x = rand(T, 128).requires_grad_()
    targets = torch.arange(T, device="cuda", dtype=torch.int64) % 1024

    def sequence(lens, positions=None):
        pos = torch.cat([torch.arange(n) for n in lens]).cuda() if positions is None else positions
        return make_sequence_context(pos, torch.tensor(_cu(lens), device="cuda", dtype=torch.int32),
                                     max_q=max(lens), cu_seqlens_cpu=torch.tensor(_cu(lens)))

    def scores(y, tg):
        return logprobs(head(y, vocab_weight).float(), tg, 0.7)

    want = trainer(x, sequence(lengths))
    assert want.requires_grad
    want_scores = scores(want, targets).detach()
    want = want.detach()
    got, got_scores = torch.empty_like(want), torch.empty_like(want_scores)
    with torch.no_grad():
        taken = [0, 0]
        for ends in (splits, [n - 5 for n in lengths]):
            spans = [torch.arange(cu[i] + taken[i], cu[i] + ends[i]) for i in range(2)]
            idx = torch.cat(spans).cuda()
            lens = [len(s) for s in spans]
            driver.qsl, driver.has = _cu(lens), any(taken)
            y = engine(x[idx], sequence(lens))
            got[idx], got_scores[idx] = y, scores(y, targets[idx])
            taken = ends
        driver.decode = True
        static_x = x[:2].detach().clone()
        static_tg = targets[:2].clone()
        seq = sequence([1, 1])
        state0, conv0 = pool.ssm_state.clone(), pool.conv_state.clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            scores(engine(static_x, seq), static_tg)
        torch.cuda.current_stream().wait_stream(stream)
        pool.ssm_state.copy_(state0)
        pool.conv_state.copy_(conv0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = engine(static_x, seq)
            static_logits = head(static_out, vocab_weight).float()
        pool.ssm_state.copy_(state0)
        pool.conv_state.copy_(conv0)
        for step in range(5):
            order = [step % 2, 1 - step % 2]
            idx = torch.tensor([cu[i] + lengths[i] - 5 + step for i in order], device="cuda")
            driver.slots.copy_(torch.tensor([driver.slots_cpu[i] for i in order], device="cuda", dtype=torch.int32))
            static_x.copy_(x[idx])
            static_tg.copy_(targets[idx])
            graph.replay()
            # Scoring is a separate runtime stage. Nonunit temperature creates
            # its device scalar outside capture; the mixer and head are captured.
            got[idx] = static_out
            got_scores[idx] = logprobs(static_logits, static_tg, 0.7)
    assert bitwise_equal(got, want)
    assert bitwise_equal(got_scores, want_scores)


@pytest.mark.parametrize("chunk_size", [0, 32, 128, 64.5])
def test_engine_refuses_a_different_chunk_recipe(chunk_size):
    with pytest.raises(ValueError, match="requires chunk_size=64"):
        engine_state.CprSm90GDN(capacity=1, chunk_size=chunk_size)
