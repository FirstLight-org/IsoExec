"""Eligibility of the Model Runner V2 vocab-parallel sampling head (vllm_patches._v2_shard_sample).

Every step it cannot serve bitwise must decline to vLLM's gathered path; an eligible step must call the
shard head with the request mapping, the sampler's own temperature/seed tensors and the row positions."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from isoexec.runtimes.vllm import vllm_patches as vp


class _Sampler:
    def __init__(self, nreq=4, vocab=128):
        self.compute_nans = False
        self.trace_replay_state = None
        self.return_sampling_mask = False
        self.use_fp64_gumbel = False
        self.logprobs_mode = "raw_logprobs"
        self.needs_logits_processing = np.zeros(nreq, dtype=bool)
        self.dims = (0, 0)
        self.sampling_states = SimpleNamespace(
            vocab_size=vocab,
            top_k=SimpleNamespace(np=np.full(nreq, vocab)),
            top_p=SimpleNamespace(np=np.ones(nreq, dtype=np.float32)),
            temperature=SimpleNamespace(gpu=torch.ones(nreq)),
            seeds=SimpleNamespace(gpu=torch.arange(nreq, dtype=torch.int64)),
        )
        self.req_states = SimpleNamespace(prefill_len=SimpleNamespace(gpu=torch.zeros(nreq, dtype=torch.int64)))

    def get_logprobs_dims(self, idx_np):
        return self.dims


def _setup(monkeypatch):
    calls = []

    def fake_shard_sample(local, **kw):
        calls.append(kw)
        n = local.shape[0]
        return torch.arange(n), torch.zeros(n), torch.ones(n, dtype=torch.int64)

    import isoexec.ops.logprobs as logprobs

    monkeypatch.setattr(logprobs, "shard_sample", fake_shard_sample, raising=False)
    sampler = _Sampler()
    model = SimpleNamespace(_tp_group=None, isoexec_shard_logits=lambda h: h, isoexec_shard_nan_col=5)
    runner = SimpleNamespace(sampler=sampler, model=model, batch_sharder=None)
    batch = SimpleNamespace(
        num_reqs=3, num_draft_tokens=0, idx_mapping_np=np.array([0, 2, 3]), idx_mapping=torch.tensor([0, 2, 3]),
        logits_indices=torch.tensor([1, 4, 7]), expanded_idx_mapping=torch.tensor([0, 2, 3], dtype=torch.int64),
        positions=torch.arange(8, dtype=torch.int64) * 10, seq_lens=torch.ones(3, dtype=torch.int32),
        cu_num_logits=torch.tensor([0, 1, 2, 3]))
    return runner, batch, calls


def _call(runner, batch, grammar=None):
    return vp._v2_shard_sample(
        runner, torch.randn(8, 16), batch, grammar, sampler_cls=_Sampler,
        sampler_output_cls=lambda **kw: SimpleNamespace(**kw),
        logprobs_tensors_cls=lambda **kw: SimpleNamespace(**kw),
        get_num_sampled_and_rejected=lambda *a: (torch.ones(3, dtype=torch.int64), torch.zeros(3, dtype=torch.int64)),
        temperature_tensor=lambda: None)


def test_eligible_step_is_served(monkeypatch):
    runner, batch, calls = _setup(monkeypatch)
    out = _call(runner, batch)
    assert out is not None and len(calls) == 1
    kw = calls[0]
    assert kw["req_of_row"].dtype == torch.int32 and kw["req_of_row"].tolist() == [0, 2, 3]
    assert kw["row_pos"].tolist() == [10, 40, 70]
    assert kw["req_seed"] is runner.sampler.sampling_states.seeds.gpu and kw["ranks"] is True
    assert kw["nan_col"] == 5
    sampler_output = out[0]
    assert sampler_output.sampled_token_ids.shape == (3, 1)
    assert sampler_output.logprobs_tensors.logprobs.shape == (3, 1)


def test_no_logprobs_skips_the_rank_reduction(monkeypatch):
    runner, batch, calls = _setup(monkeypatch)
    runner.sampler.dims = None
    out = _call(runner, batch)
    assert out[0].logprobs_tensors is None and calls[0]["ranks"] is False


@pytest.mark.parametrize("breaker", [
    "grammar", "batch_sharder", "draft", "processing", "top_k", "top_p", "topk_logprobs", "logits_mode",
    "nans", "fp64", "mask", "no_model_head", "head_declines", "not_prepared", "subclass",
])
def test_ineligible_steps_decline(monkeypatch, breaker):
    runner, batch, calls = _setup(monkeypatch)
    s = runner.sampler
    grammar = None
    if breaker == "grammar":
        grammar = object()
    elif breaker == "batch_sharder":
        runner.batch_sharder = object()
    elif breaker == "draft":
        batch.num_draft_tokens = 1
    elif breaker == "processing":
        s.needs_logits_processing[2] = True
    elif breaker == "top_k":
        s.sampling_states.top_k.np[3] = 5
    elif breaker == "top_p":
        s.sampling_states.top_p.np[0] = 0.9
    elif breaker == "topk_logprobs":
        s.dims = (5, 0)
    elif breaker == "logits_mode":
        s.logprobs_mode = "raw_logits"
    elif breaker == "nans":
        s.compute_nans = True
    elif breaker == "fp64":
        s.use_fp64_gumbel = True
    elif breaker == "mask":
        s.return_sampling_mask = True
    elif breaker == "no_model_head":
        runner.model = SimpleNamespace(_tp_group=None)
    elif breaker == "head_declines":
        runner.model = SimpleNamespace(_tp_group=None, isoexec_shard_logits=lambda h: None, isoexec_shard_nan_col=5)
    elif breaker == "not_prepared":
        runner.model.isoexec_shard_nan_col = None
    elif breaker == "subclass":
        runner.sampler = type("Watermark", (_Sampler,), {})()
    assert _call(runner, batch, grammar) is None
    assert not calls


# ------------------------------------------------------------------------- install-time and init-time gates
def test_version_pin_refuses_other_triton(monkeypatch):
    triton = pytest.importorskip("triton")
    pytest.importorskip("vllm.v1.worker.gpu.sample.gumbel")
    monkeypatch.setattr(triton, "__version__", "9.9.0")
    assert "Triton 9.9.0" in vp.shard_sampler_version_refusal()


def test_version_pin_matches_the_installed_vllm():
    """The pinned digest is the installed vLLM's gumbel source (a vLLM bump must re-verify and re-pin)."""
    pytest.importorskip("vllm.v1.worker.gpu.sample.gumbel")
    import triton

    if ".".join(triton.__version__.split(".")[:2]) != vp.SHARD_SAMPLER_TRITON:
        pytest.skip("another Triton")
    assert vp.shard_sampler_version_refusal() is None


def test_version_pin_refuses_a_changed_gumbel_source(monkeypatch):
    gumbel = pytest.importorskip("vllm.v1.worker.gpu.sample.gumbel")
    monkeypatch.setattr(gumbel, "gumbel_sample", lambda *a, **k: None)  # source differs
    refusal = vp.shard_sampler_version_refusal()
    assert refusal is not None


@pytest.mark.parametrize("got,expected", [([1023, 1023], 1023), ([0, 0], 0), ([3, 4], None), ([4096, 4096], None),
                                          ([-1, -1], None)])
def test_all_nan_column_probe_requires_one_in_range_column(monkeypatch, got, expected):
    gumbel = pytest.importorskip("vllm.v1.worker.gpu.sample.gumbel")
    seen = {}

    def fake(logits, req, temp, seed, pos, **kw):
        seen["nan"] = bool(torch.isnan(logits).all())
        seen["temps"] = temp.tolist()
        return torch.tensor(got)

    monkeypatch.setattr(gumbel, "gumbel_sample", fake)
    assert vp.vllm_all_nan_column(4096, "cpu") == expected
    assert seen == {"nan": True, "temps": [0.0, 1.0]}  # one greedy and one noisy all-NaN row


def _agree_rank(rank, world, verdicts):
    import torch.distributed as dist

    vp._shard_head_local_check = lambda head, w: verdicts[rank]
    model = SimpleNamespace(head=SimpleNamespace(vocab=64, padded_vocab=64), _tp_group=dist.group.WORLD)
    return vp.prepare_v2_shard_head(model)


@pytest.mark.parametrize("verdicts,expected", [
    ([(True, 7, ""), (True, 7, "")], 7),
    ([(True, 7, ""), (False, 0, "build/probe failed: boom")], None),  # one rank's build failed: none serve
    ([(False, 0, "build/probe failed: nvcc"), (True, 7, "")], None),
    ([(True, 7, ""), (True, 9, "")], None),  # different probes: decline
])
def test_init_agreement_is_group_wide(verdicts, expected):
    from isoexec.ops._testing import run_gloo

    out = run_gloo(_agree_rank, 2, verdicts)
    assert out == {0: expected, 1: expected}


def test_no_init_collective_when_the_head_is_not_installed(monkeypatch):
    """Flag off (or refused install): nothing to agree on, and no collective at model construction."""
    monkeypatch.setattr(vp, "_shard_head_local_check", lambda head, w: (False, 0, "head not installed"))
    monkeypatch.setattr(vp, "agree_shard_head", lambda *a: pytest.fail("collective issued"))
    model = SimpleNamespace(head=SimpleNamespace(vocab=64, padded_vocab=64), _tp_group=SimpleNamespace(size=lambda: 2))
    assert vp.prepare_v2_shard_head(model) is None and model.isoexec_shard_nan_col is None
