"""CPR prefix caching under the vLLM engine loop: a prefix-cache hit equals a cold prefill, bitwise.

Every case builds the real engine (in-process engine core; TP>1 workers in their own processes) for a GDN
checkpoint with the CPR core, Model Runner V2, ``FULL_AND_PIECEWISE`` graphs and async scheduling, and
greedy-decodes two waves of requests with ``logprobs=1``: wave 1 the base prompts, wave 2 the same prompts
again plus variants sharing a long prefix with one of them (a different suffix). With prefix caching on,
wave 2 resumes from the checkpoints wave 1 left (``runtimes/vllm/cpr_apc.py``). Every request's token ids
and sampled-logprob bits must equal the cold reference -- the same requests with prefix caching off:

* ``apc``: prefix caching, 4096 batched tokens (prompts prefill whole except for vLLM's align-mode stop at
  the last block boundary), TP1 and TP2;
* ``apc_chunked``: prefix caching with 512 batched tokens: prompts split into chunks, hits resume
  mid-prompt and continue with chunked prefill beside decodes, then decode -- against the reference that
  prefilled each prompt in one step (V1, no chunking);
* ``apc_evicted``: between the waves every worker drops its checkpoints while vLLM keeps its prefix cache,
  so vLLM finds hits the workers cannot serve: the scheduler must truncate them to none and recompute;
* ``apc_budget``: a store with room for a few checkpoints, so LRU eviction runs during the waves.

Each case also checks that hits really happened (or, for ``apc_evicted``, were really refused).

Point ``ISOEXEC_TEST_GDN_MODEL_PATH`` at a Qwen3.5/3.6 checkpoint. Run under pytest, or directly:
``python -m isoexec.runtimes.vllm.tests.cpr_apc_engine_gpu``. TP2 cases need two visible GPUs.
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

V1_ONE_PREFILL = {"runner": "v1", "cudagraph_mode": "FULL_DECODE_ONLY", "prefix": False, "chunked": False}
V2 = {"runner": "v2", "cudagraph_mode": "FULL_AND_PIECEWISE", "async": True, "chunked": True}
BASES, MAX_TOKENS = 12, 96


def _requests(vocab: int) -> tuple[list[list[int]], list[list[int]]]:
    """(wave 1, wave 2): ``BASES`` prompts of 300..1500 tokens; then each again, and a variant of each that
    keeps its first 60..90% and continues differently."""
    g = torch.Generator().manual_seed(7)
    lens = torch.linspace(300, 1500, BASES).long().tolist()
    base = [torch.randint(10, vocab - 10, (n,), generator=g).tolist() for n in lens]
    variants = []
    for i, p in enumerate(base):
        cut = int(len(p) * (0.6 + 0.3 * i / BASES))
        variants.append(p[:cut] + torch.randint(10, vocab - 10, (64 + 17 * i,), generator=g).tolist())
    order = torch.randperm(2 * BASES, generator=g).tolist()
    wave2 = [(base + variants)[k] for k in order]
    return base, wave2


def _bits(x: float) -> int:
    return int(torch.tensor(x, dtype=torch.float32).view(torch.int32))


def _row(output) -> dict:
    c = output.outputs[0]
    return {"ids": list(c.token_ids), "lp": [_bits(d[t].logprob) for t, d in zip(c.token_ids, c.logprobs)]}


def _build(case: dict):
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["ISOEXEC_GDN_KERNEL"] = "cpr"
    os.environ["ISOEXEC_GDN_NATIVE_KERNELS"] = "1"
    os.environ["ISOEXEC_GDN_NATIVE_STATE"] = "0"
    if case.get("budget_mb"):
        os.environ["ISOEXEC_GDN_CPR_APC_MB"] = str(case["budget_mb"])
    from isoexec.runtimes.environment import resolved_environment

    for k, v in resolved_environment(MODEL, model_runner=case["runner"]).items():
        os.environ.setdefault(k, v)
    from vllm import LLM

    from isoexec.compiler.source import ModelSource
    from isoexec.runtimes.vllm.registration import register_native

    raw = json.loads(Path(MODEL, "config.json").read_text())
    kw = dict(max_model_len=4096, max_num_seqs=case.get("max_num_seqs", 64),
              kv_cache_memory_bytes=int(case.get("kv_gib", 4) * 2**30),
              max_num_batched_tokens=case.get("max_num_batched_tokens", 4096),
              enable_prefix_caching=bool(case["prefix"]), enable_chunked_prefill=bool(case["chunked"]),
              tensor_parallel_size=case.get("tp", 1), seed=0, gpu_memory_utilization=0.3,
              async_scheduling=bool(case.get("async")), language_model_only=True,
              compilation_config={"mode": 0, "cudagraph_mode": case["cudagraph_mode"]})
    kw = register_native(ModelSource("isoexec.models.qwen3_5"), raw, checkpoint=MODEL, engine_kwargs=kw,
                         cudagraph_mode=case["cudagraph_mode"])
    llm = LLM(**kw)
    resolved = llm.llm_engine.vllm_config
    assert bool(resolved.cache_config.enable_prefix_caching) == bool(case["prefix"]), "prefix caching not as asked"
    return llm


def _generate(case: dict) -> dict:
    from vllm import SamplingParams

    from isoexec.runtimes.vllm import cpr_apc

    llm = _build(case)
    wave1, wave2 = _requests(llm.get_tokenizer().vocab_size)
    params = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, ignore_eos=True, logprobs=1)
    rows = [_row(o) for o in llm.generate([{"prompt_token_ids": p} for p in wave1], params, use_tqdm=False)]
    after_wave1 = dict(cpr_apc.SCHED_STATS)
    if case.get("evict"):
        # Every worker drops its checkpoints (and retracts them from its index); vLLM's prefix cache keeps
        # its blocks, so the scheduler sees hits it must refuse.
        def drop(*_args):
            from isoexec.ops.gdn.shared.gdn_cpr_state import cpr_apc_reset

            cpr_apc_reset()

        if case.get("tp", 1) == 1:
            drop()
        else:
            llm.collective_rpc(drop)
    rows += [_row(o) for o in llm.generate([{"prompt_token_ids": p} for p in wave2], params, use_tqdm=False)]
    cfg = llm.llm_engine.vllm_config
    return {"rows": rows, "sched": dict(cpr_apc.SCHED_STATS), "sched_wave1": after_wave1,
            "worker": dict(cpr_apc.WORKER_STATS), "async": bool(cfg.scheduler_config.async_scheduling),
            "block_size": int(cfg.cache_config.block_size)}


def _child(case: dict, out: Path) -> None:
    out.write_text(json.dumps(_generate(case)))


def _run(case: dict) -> dict:
    """One engine per process: a case's engine never shares a process with another's."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "result.json"
        proc = subprocess.run([sys.executable, "-m", "isoexec.runtimes.vllm.tests.cpr_apc_engine_gpu", "--child",
                               json.dumps(case), str(out)], capture_output=True, text=True)
        if proc.returncode != 0:
            raise AssertionError(f"case {case} failed:\n{proc.stdout[-6000:]}\n{proc.stderr[-8000:]}")
        result = json.loads(out.read_text())
        print(f"case {case}: sched {result['sched']} worker {result['worker']} block {result['block_size']}")
        return result


def _same(want: list, got: list, what: str) -> None:
    assert len(want) == len(got), what
    for i, (a, b) in enumerate(zip(want, got)):
        assert a["ids"] == b["ids"], f"{what}: request {i} tokens differ"
        assert a["lp"] == b["lp"], f"{what}: request {i} sampled-logprob bits differ"


_REFERENCE: dict = {}


def _reference(tp: int = 1) -> list:
    if tp not in _REFERENCE:
        case = {**V1_ONE_PREFILL} if tp == 1 else {**V2, "prefix": False, "tp": tp}
        _REFERENCE[tp] = _run(case)["rows"]
    return _REFERENCE[tp]


def _gpus() -> int:
    return torch.cuda.device_count()


@pytest.mark.parametrize("tp", [1, 2])
def test_prefix_hit_equals_a_cold_prefill(tp):
    if _gpus() < tp:
        pytest.skip(f"needs {tp} GPUs")
    got = _run({**V2, "prefix": True, "tp": tp})
    assert got["async"], "the engine declined async scheduling"
    assert got["block_size"] % 64 == 0, f"vLLM's block size {got['block_size']} is not on the CPR chunk grid"
    assert got["sched"]["admitted"] > 0, f"no prefix-cache hit was admitted: {got['sched']}"
    if tp == 1:
        # (the scheduler may admit a request's hit on several steps before it can be scheduled)
        assert 0 < got["worker"]["adopted"] <= got["sched"]["admitted"] and got["worker"]["captured"] > 0
    _same(_reference(tp), got["rows"], f"TP{tp} prefix caching vs cold")


def test_chunked_prefill_prefix_hit_and_decode_equal_one_prefill():
    got = _run({**V2, "prefix": True, "max_num_batched_tokens": 512})
    assert got["sched"]["admitted"] > 0, got["sched"]
    # some hits resume in the middle of a prompt that was chunked: not only at a prompt's last block
    _same(_reference(1), got["rows"], "chunked prefill + prefix hit + decode vs one prefill")


def test_evicted_checkpoints_are_recomputed_not_resumed():
    got = _run({**V2, "prefix": True, "evict": True})
    refused = got["sched"]["truncated"] - got["sched_wave1"]["truncated"]
    assert refused > 0, f"vLLM offered no hit the workers could not serve: {got['sched']}"
    _same(_reference(1), got["rows"], "evicted checkpoints vs cold")


def test_store_eviction_under_a_small_budget_keeps_the_bits():
    got = _run({**V2, "prefix": True, "budget_mb": 64})
    assert got["sched"]["admitted"] > 0, got["sched"]
    _same(_reference(1), got["rows"], "LRU-evicting store vs cold")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--child":
        _child(json.loads(sys.argv[2]), Path(sys.argv[3]))
    else:
        sys.exit(pytest.main([__file__, "-v", "-s", *sys.argv[1:]]))
