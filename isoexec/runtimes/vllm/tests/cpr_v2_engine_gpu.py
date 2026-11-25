"""CPR GDN under the vLLM engine loop: every Model Runner V2 schedule serves the eager bits, and the private pool stays exact.

Every case builds the real engine on one GPU (in-process engine core) for a GDN checkpoint with the CPR
core of the live arch (``gdn_cpr_sm90`` or ``gdn_cpr_sm100``) and greedy-decodes the same synthetic
prompts with ``logprobs=1``. A case's token ids and sampled-logprob bits must equal the eager
(``NONE``, no chunked prefill) Model Runner V2 reference:

* ``v2``: Model Runner V2, ``FULL_AND_PIECEWISE`` graphs;
* ``v2_chunked``: the same with chunked prefill, 512 batched tokens (prompts split across steps and
  prefilled beside decodes), and with 1024 batched tokens at 16 sequences;
* ``v2_sleep``: V2 + chunked, two level-1 sleep/wake cycles between repeats; every repeat equals the
  reference (level 2 discards the weights, which only a framework's weight sync puts back);
* ``v2_lifecycle``: 256-way concurrency with mid-flight aborts, then a second wave; after every engine
  step the pool's mapped rows are exactly the driver's live requests, and none are left at the end;
* ``v2_preempt``: a KV cache small enough that the scheduler preempts; preempted requests resume from
  token zero into a fresh row and still give the reference bits.

Point ``ISOEXEC_TEST_GDN_MODEL_PATH`` at a Qwen3.5/3.6 checkpoint (Qwen3.5-2B takes a few minutes per
case). Run under pytest, or directly: ``python -m isoexec.runtimes.vllm.tests.cpr_v2_engine_gpu``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch

MODEL = os.environ.get("ISOEXEC_TEST_GDN_MODEL_PATH", "")

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU"),
    pytest.mark.skipif(not (Path(MODEL) / "config.json").is_file(),
                       reason="set ISOEXEC_TEST_GDN_MODEL_PATH to a Qwen3.5/3.6 GDN checkpoint"),
]

EAGER = {"cudagraph_mode": "NONE"}
V2 = {"cudagraph_mode": "FULL_AND_PIECEWISE"}
PROMPTS, MAX_TOKENS = 64, 512


def _prompts(vocab: int, n: int, seed: int = 0) -> list[list[int]]:
    """``n`` prompts of uneven length (17..1500 tokens), deterministic in ``seed``."""
    g = torch.Generator().manual_seed(seed)
    lens = torch.linspace(17, 1500, n).long().tolist()
    order = torch.randperm(n, generator=g).tolist()
    return [torch.randint(10, vocab - 10, (lens[i],), generator=g).tolist() for i in order]


def _bits(x: float) -> int:
    return int(torch.tensor(x, dtype=torch.float32).view(torch.int32))


def _row(output) -> dict:
    c = output.outputs[0]
    return {"ids": list(c.token_ids), "lp": [_bits(d[t].logprob) for t, d in zip(c.token_ids, c.logprobs)]}


def _build(case: dict):
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["ISOEXEC_GDN_KERNEL"] = "cpr"
    from isoexec.runtimes.environment import resolved_environment

    for k, v in resolved_environment(MODEL).items():
        os.environ.setdefault(k, v)
    from vllm import LLM

    from isoexec.compiler.source import ModelSource
    from isoexec.runtimes.vllm.registration import register_native

    raw = json.loads(Path(MODEL, "config.json").read_text())
    kw = dict(max_model_len=4096, max_num_seqs=case.get("max_num_seqs", 64),
              kv_cache_memory_bytes=int(case.get("kv_gib", 4) * 2**30),
              max_num_batched_tokens=case.get("max_num_batched_tokens", 4096), enable_prefix_caching=False,
              seed=0, gpu_memory_utilization=0.3, enable_sleep_mode=bool(case.get("sleep")),
              async_scheduling=bool(case.get("async")),
              language_model_only=True, compilation_config={"mode": 0, "cudagraph_mode": case["cudagraph_mode"]})
    kw = register_native(ModelSource("isoexec.models.qwen3_5"), raw, checkpoint=MODEL, engine_kwargs=kw,
                         cudagraph_mode=case["cudagraph_mode"], enable_chunked_prefill=bool(case.get("chunked")))
    return LLM(**kw)


def _pool_audit(where: str, live: int) -> int:
    """The pool's owners are exactly the driver's requests, at most ``live`` of them; returns the count."""
    from isoexec.ops.gdn import CPR_LAYERS
    from isoexec.runtimes.vllm import gdn_engine_patch as patch

    owners = set(patch._DRIVER_SLOT_BY_REQ.values())
    for j, layer in enumerate(CPR_LAYERS):
        mapped = set(layer._slot2row)
        assert len(mapped) + len(layer._free) == layer.capacity, f"{where}: layer {j} leaks rows"
        assert mapped == owners, f"{where}: layer {j} maps {len(mapped)} slots, the driver owns {len(owners)}"
    assert len(owners) <= live, f"{where}: {len(owners)} rows mapped for {live} live requests"
    return len(owners)


_PREEMPTED = [0]


def _count_preemptions() -> None:
    """Count the requests Model Runner V2 retires as preempted (its ``finish_requests`` sees every one)."""
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    finish = GPUModelRunner.finish_requests

    def finish_requests(self, scheduler_output, *args, **kw):
        _PREEMPTED[0] += len(scheduler_output.preempted_req_ids or ())
        return finish(self, scheduler_output, *args, **kw)

    GPUModelRunner.finish_requests = finish_requests


def _generate(case: dict) -> dict:
    from vllm import SamplingParams

    llm = _build(case)
    _count_preemptions()  # wraps the driver's hook, installed with the engine
    prompts = _prompts(llm.get_tokenizer().vocab_size, PROMPTS)
    params = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, ignore_eos=True, logprobs=1)
    repeats = []
    for level in [None, *case.get("sleep", ())]:
        if level is not None:
            llm.sleep(level=level)
            llm.wake_up()
        outs = llm.generate([{"prompt_token_ids": p} for p in prompts], params, use_tqdm=False)
        repeats.append([_row(o) for o in outs])
    return {"repeats": repeats, "preempted": _PREEMPTED[0], "async": _async_on(llm), **_graphs(llm)}


def _async_on(llm) -> bool:
    """Whether the built engine really schedules asynchronously (vLLM may decline the request)."""
    return bool(llm.llm_engine.vllm_config.scheduler_config.async_scheduling)


def _graphs(llm) -> dict:
    """The graph mode vLLM built the engine with, and how many mixed batches ran breakable PIECEWISE graphs."""
    from isoexec.runtimes.vllm._binding import cudagraph_mode
    from isoexec.runtimes.vllm.vllm_patches import _V2_GRAPH_STATE

    return {"built_mode": cudagraph_mode(llm.llm_engine.vllm_config),
            "piecewise_runs": _V2_GRAPH_STATE["piecewise_runs"], "full_replays": _V2_GRAPH_STATE["replays"]}


def _lifecycle(case: dict) -> dict:
    """Drive the engine step by step: two waves, the first with aborts; audit the pool after every step."""
    from vllm import SamplingParams

    llm = _build(case)
    engine = llm.llm_engine
    base = _prompts(llm.get_tokenizer().vocab_size, PROMPTS)
    n = case["requests"]
    aborts = case.get("aborts", True)
    audit = _pool_audit
    done: dict[str, dict] = {}
    aborted: set[str] = set()
    for wave in range(2):
        ids = []
        for i in range(n):
            rid = f"w{wave}_r{i}"
            prompt = base[(i * (wave + 1)) % len(base)][: 64 + (i * 37) % 700]
            params = SamplingParams(temperature=0.0, max_tokens=32 + (i * 13) % 224, ignore_eos=True, logprobs=1)
            engine.add_request(rid, {"prompt_token_ids": prompt}, params)
            ids.append(rid)
        step, recent = 0, [0, 0, 0] if case.get("async") else [0, 0]
        while engine.has_unfinished_requests():
            outs = engine.step()
            step += 1
            # A finish or abort reaches the worker a step later (two under async scheduling, whose step N+1 is
            # scheduled before step N's outputs are known): count those steps' as still live.
            recent = [*recent[1:], sum(1 for o in outs if o.finished)]
            for o in outs:
                if o.finished and o.request_id not in aborted:
                    done[o.request_id] = _row(o)
            if aborts and wave == 0 and step == 8:
                victims = [r for k, r in enumerate(ids) if k % 3 == 1 and r not in done]
                engine.abort_request(victims)
                aborted.update(victims)
                recent[1] += len(victims)
            audit(f"wave {wave} step {step}", engine.get_num_unfinished_requests() + sum(recent))
        for _ in range(3):  # token-free steps deliver the last finishes
            engine.step()
        audit(f"wave {wave} end", 0)
    return {"done": done, "aborted": sorted(aborted), "async": _async_on(llm)}


def _child(case: dict, out: Path) -> None:
    result = _lifecycle(case) if case.get("requests") else _generate(case)
    out.write_text(json.dumps(result))


def _run(case: dict) -> object:
    """One engine per process: a case's engine never shares a process with another's."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "result.json"
        proc = subprocess.run([sys.executable, "-m", "isoexec.runtimes.vllm.tests.cpr_v2_engine_gpu", "--child",
                               json.dumps(case), str(out)], capture_output=True, text=True)
        if proc.returncode != 0:
            raise AssertionError(f"case {case} failed:\n{proc.stdout[-4000:]}\n{proc.stderr[-8000:]}")
        return json.loads(out.read_text())


def _same(want: list, got: list, what: str) -> None:
    assert len(want) == len(got), what
    for i, (a, b) in enumerate(zip(want, got)):
        assert a["ids"] == b["ids"], f"{what}: request {i} tokens differ"
        assert a["lp"] == b["lp"], f"{what}: request {i} sampled-logprob bits differ"


_REFERENCE: list = []


def _reference() -> list:
    if not _REFERENCE:
        ref = _run(EAGER)
        assert ref["piecewise_runs"] == 0 and ref["full_replays"] == 0, "the eager reference replayed graphs"
        _REFERENCE.append(ref["repeats"][0])
    return _REFERENCE[0]


def _ran_piecewise(got: dict) -> None:
    # vLLM rewrites FULL_AND_PIECEWISE to FULL_DECODE_ONLY, with one log line, when the process lacks
    # breakable graphs; equal bits alone would not notice that the mixed batches ran eagerly.
    assert got["built_mode"] == "FULL_AND_PIECEWISE", f"engine built with {got['built_mode']}"
    assert got["piecewise_runs"] > 0, "no mixed batch ran a breakable PIECEWISE graph"
    assert got["full_replays"] > 0, "no uniform decode replayed a FULL graph"


def test_v2_piecewise_equals_eager():
    got = _run(V2)
    _ran_piecewise(got)
    _same(_reference(), got["repeats"][0], "V2 FULL_AND_PIECEWISE vs eager")


@pytest.mark.parametrize(("batched", "seqs"), [(512, 64), (1024, 16)])
def test_v2_chunked_prefill_equals_eager(batched, seqs):
    case = {**V2, "chunked": True, "max_num_batched_tokens": batched, "max_num_seqs": seqs}
    got = _run(case)
    _ran_piecewise(got)
    _same(_reference(), got["repeats"][0], f"V2 chunked ({batched} tokens, {seqs} seqs) vs eager")


def test_v2_sleep_wake_keeps_the_bits():
    repeats = _run({**V2, "chunked": True, "max_num_batched_tokens": 512, "sleep": [1, 1]})["repeats"]
    for r, got in enumerate(repeats):
        _same(_reference(), got, f"V2 chunked, repeat {r} after sleep/wake")


def test_v2_lifecycle_aborts_and_high_concurrency_keep_the_pool_exact():
    case = {**V2, "chunked": True, "max_num_batched_tokens": 2048, "max_num_seqs": 256, "requests": 384}
    got = _run(case)
    ref = _run({**EAGER, "max_num_seqs": 64, "requests": 384, "aborts": False})
    assert got["aborted"] and len(got["done"]) + len(got["aborted"]) == 2 * 384
    for rid, row in got["done"].items():
        assert row == ref["done"][rid], f"{rid}: differs from the eager run"


def test_v2_preemption_resumes_with_the_reference_bits():
    got = _run({**V2, "chunked": True, "max_num_batched_tokens": 1024, "kv_gib": 0.25})
    assert got["preempted"] > 0, "the KV cache was large enough that nothing was preempted"
    _same(_reference(), got["repeats"][0], "V2 under preemption vs eager")


# Async scheduling (step N+1 prepared on the host while step N runs): the private pool's driver reads only the
# scheduler's host output, and every pool mutation is stream-ordered after step N, so the bits are V1's.
ASYNC = {**V2, "async": True}


def test_v2_async_scheduling_equals_v1():
    got = _run({**ASYNC, "chunked": True, "max_num_batched_tokens": 512})
    assert got["async"], "the engine declined async scheduling"
    _same(_reference(), got["repeats"][0], "V2 chunked + async scheduling vs V1")


def test_v2_async_preemption_resumes_with_the_reference_bits():
    got = _run({**ASYNC, "chunked": True, "max_num_batched_tokens": 1024, "kv_gib": 0.25})
    assert got["async"] and got["preempted"] > 0
    _same(_reference(), got["repeats"][0], "V2 + async under preemption vs V1")


def test_v2_async_lifecycle_aborts_keep_the_pool_exact():
    case = {**ASYNC, "chunked": True, "max_num_batched_tokens": 2048, "max_num_seqs": 256, "requests": 384}
    got = _run(case)
    assert got["async"]
    ref = _run({**V1, "max_num_seqs": 64, "requests": 384, "aborts": False})
    assert got["aborted"] and len(got["done"]) + len(got["aborted"]) == 2 * 384
    for rid, row in got["done"].items():
        assert row == ref["done"][rid], f"{rid}: differs from the V1 run"


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--child":
        _child(json.loads(sys.argv[2]), Path(sys.argv[3]))
    else:
        sys.exit(pytest.main([__file__, "-v", *sys.argv[1:]]))
