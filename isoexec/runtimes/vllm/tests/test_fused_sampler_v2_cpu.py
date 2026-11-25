"""Which vLLM Model Runner V2 sampling steps the decode sampler serves, and which keep V2's sampler."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("vllm")

from vllm.v1.worker.gpu.sample.sampler import Sampler  # noqa: E402

from isoexec.runtimes.vllm import vllm_patches as patches  # noqa: E402

VOCAB = 64
SLOTS = 4


def _sampler(*, temperature=(1.0, 1.0), num_logprobs=1, token_ids=0, **over):
    temp = np.zeros(SLOTS, dtype=np.float32)
    temp[: len(temperature)] = temperature
    states = SimpleNamespace(
        vocab_size=VOCAB,
        temperature=SimpleNamespace(np=temp, gpu=torch.from_numpy(temp)),
        top_k=SimpleNamespace(np=np.full(SLOTS, VOCAB, dtype=np.int32)),
        top_p=SimpleNamespace(np=np.ones(SLOTS, dtype=np.float32)),
        min_p=SimpleNamespace(np=np.zeros(SLOTS, dtype=np.float32)),
        seeds_set=np.zeros(SLOTS, dtype=bool),
    )
    dims = None if num_logprobs is None and not token_ids else (num_logprobs or 0, token_ids)
    base = dict(
        sampling_states=states, logprobs_mode="raw_logprobs", num_speculative_tokens=1, trace_replay_state=None,
        return_sampling_mask=False, needs_logits_processing=np.zeros(SLOTS, dtype=bool),
        _isoexec_processed=np.zeros(SLOTS, dtype=bool), get_logprobs_dims=lambda idx: dims,
    )
    base.update(over)
    return SimpleNamespace(**base)


IDX = np.arange(2)


def plan(sampler, logits=None, *, supports=lambda logits: True, override=None):
    logits = torch.zeros(2, VOCAB) if logits is None else logits
    return patches._v2_fused_sampler_plan(sampler, logits, IDX, supports, override)


def test_plain_decode_is_served():
    assert plan(_sampler()) == ("plain", 1)
    assert plan(_sampler(num_logprobs=0)) == ("plain", 0)
    assert plan(_sampler(num_logprobs=None)) == ("plain", None)
    assert plan(_sampler(temperature=(0.0, 0.0))) == ("plain", 1)
    assert plan(_sampler(temperature=(0.0, 0.7))) == ("plain", 1)


def _set(sampler, path, value):
    obj = sampler.sampling_states
    for name in path[:-1]:
        obj = getattr(obj, name)
    getattr(obj, path[-1])[0] = value
    return sampler


@pytest.mark.parametrize(
    "sampler, why",
    [
        (_set(_sampler(), ("top_k", "np"), 5), "top_k/top_p"),
        (_set(_sampler(), ("top_p", "np"), 0.9), "top_k/top_p"),
        (_set(_sampler(), ("seeds_set",), True), "per_request_seed"),
        (_set(_sampler(), ("min_p", "np"), 0.1), "min_p"),
        (_sampler(token_ids=2), "logprob_token_ids"),
        (_sampler(num_logprobs=5), "max_num_logprobs=5"),
        (_sampler(num_logprobs=VOCAB), f"max_num_logprobs={VOCAB}"),  # logprobs=-1: the full row
        (_sampler(logprobs_mode="processed_logprobs"), "logprobs_mode"),
        (_sampler(logprobs_mode="raw_logits"), "logprobs_mode"),
        (_sampler(num_speculative_tokens=3), "spec_decode"),
        (_sampler(trace_replay_state=object()), "trace_replay"),
        (_sampler(return_sampling_mask=True), "sampling_mask"),
    ],
)
def test_unsupported_steps_fall_back(sampler, why):
    assert plan(sampler) == (None, why)


def test_processors_layout_expanded_rows_and_overrides_fall_back():
    sampler = _sampler()
    sampler._isoexec_processed[1] = True
    assert plan(sampler) == (None, "logits_processors")
    sampler._isoexec_processed[1] = False
    sampler._isoexec_processed[3] = True  # a slot outside this batch does not count
    assert plan(sampler) == ("plain", 1)
    assert plan(_sampler(), supports=lambda logits: False) == (None, "layout")
    assert plan(_sampler(), logits=torch.zeros(3, VOCAB)) == (None, "spec_decode")
    assert plan(_sampler(), override="GPUWatermarkSampler") == (None, "sampler_override:GPUWatermarkSampler")


def test_watermark_sampler_is_an_override():
    from vllm.v1.watermarking.gpu_sampler import GPUWatermarkSampler

    assert patches._v2_sampler_override(Sampler, Sampler) is None
    assert patches._v2_sampler_override(type("Plain", (Sampler,), {}), Sampler) is None
    assert patches._v2_sampler_override(GPUWatermarkSampler, Sampler) == "GPUWatermarkSampler"


class _Proc:
    def __init__(self, key):
        self.key = key

    def add_request(self, req_idx, params):
        return bool(getattr(params, self.key))


def _add_request_host():
    """A V2 sampler whose states answer like V2's: temperature 1.0 and 0.0 need no processing."""
    states = SimpleNamespace(add_request=lambda i, p: p.temperature not in (0.0, 1.0))
    return SimpleNamespace(
        sampling_states=states,
        thinking_budget_state=_Proc("thinking"),
        logits_processors=[_Proc("bias"), _Proc("penalty")],
        needs_logits_processing=np.zeros(SLOTS, dtype=bool),
        logprob_token_ids_state=SimpleNamespace(add_request=lambda i, p: None),
        trace_replay_state=None,
    )


class LogitBiasState:
    """Answers like vLLM's: True for allowed tokens, logit bias, or min_tokens with stop tokens."""

    def __init__(self):
        self.min_lens = SimpleNamespace(np=np.zeros(SLOTS, dtype=np.int32))

    def add_request(self, req_idx, params):
        self.min_lens.np[req_idx] = params.prompt_len + params.min_tokens
        return bool(params.allowed_token_ids or params.logit_bias or params.min_tokens)


def test_a_min_tokens_mask_counts_only_until_its_length():
    add_request = patches._make_vllm_v2_add_request(Sampler.add_request)
    host = _add_request_host()
    host.logits_processors = [LogitBiasState(), _Proc("penalty")]

    def params(**over):
        base = dict(temperature=1.0, thinking=False, penalty=False, prompt_len=10, min_tokens=0,
                    allowed_token_ids=None, logit_bias=None)
        return SimpleNamespace(**dict(base, **over))

    add_request(host, 0, params(min_tokens=1))  # SkyRL's default: the mask lasts through length 11
    add_request(host, 1, params(min_tokens=1, penalty=True))
    add_request(host, 2, params(min_tokens=1, logit_bias={3: 1.0}))
    assert host._isoexec_processed[:3].tolist() == [True, True, True]
    assert host._isoexec_min_only_len[:3].tolist() == [11, -1, -1]
    sampler = _sampler(_isoexec_processed=host._isoexec_processed, _isoexec_min_only_len=host._isoexec_min_only_len,
                       needs_logits_processing=host.needs_logits_processing)
    idx = np.array([0])

    def at(next_len):
        return patches._v2_fused_sampler_plan(sampler, torch.zeros(1, VOCAB), idx, lambda x: True,
                                              next_lens=np.array([next_len]))

    assert at(10) == (None, "logits_processors"), "the first sampled token is still masked"
    assert at(11) == ("plain", 1) and at(500) == ("plain", 1)
    for slot in (1, 2):
        idx[0] = slot
        assert at(500) == (None, "logits_processors")


def test_add_request_keeps_the_processor_part_apart_and_vllms_flag_whole():
    add_request = patches._make_vllm_v2_add_request(Sampler.add_request)
    host = _add_request_host()
    states_add = host.sampling_states.add_request

    def params(**over):
        return SimpleNamespace(**dict(dict(temperature=1.0, thinking=False, bias=False, penalty=False), **over))

    add_request(host, 0, params(temperature=0.7))
    add_request(host, 1, params(penalty=True))
    add_request(host, 2, params(temperature=0.7, bias=True))
    add_request(host, 3, params(thinking=True))
    assert host.needs_logits_processing.tolist() == [True, True, True, True]  # vLLM's own OR, unchanged
    assert host._isoexec_processed.tolist() == [False, True, True, True]
    add_request(host, 1, params())  # a reused slot is rewritten
    assert host.needs_logits_processing[1] == False and host._isoexec_processed[1] == False  # noqa: E712
    assert host.sampling_states.add_request is states_add  # the states' own method is restored


class _Provider:
    def __init__(self):
        self.calls = []

    def supports(self, logits):
        return True

    def next_philox(self, device):
        return 7, 8

    def decode_step(self, x, temp, seed, offset, k, st, xs=None):
        self.calls.append((None if temp is None else temp.tolist(), seed, offset, k, st))
        rows = x.shape[0]
        lp = None if k is None else (torch.zeros(rows, 1 + k, dtype=torch.int32), torch.zeros(rows, 1 + k),
                                     torch.zeros(rows, dtype=torch.int64))
        return torch.arange(rows), lp


def _call(provider, monkeypatch):
    original_calls, outputs = [], []

    def original(self, logits, input_batch):
        original_calls.append(input_batch)
        return "vllm"

    monkeypatch.setattr(patches, "_vllm_v2_sampler_output",
                        lambda sampler, logits, batch, token, lp: outputs.append((token, lp)) or "fused")
    # The stub sampler class stands in for V2's Sampler, so it overrides nothing.
    call = patches._make_vllm_v2_fused_sampler_call(original, SimpleNamespace, lambda: provider, lambda: None)
    return call, original_calls, outputs


BATCH = SimpleNamespace(idx_mapping_np=np.array([1, 0]), expanded_idx_mapping=torch.tensor([1, 0]),
                        num_computed_tokens_np=np.array([40, 7]), num_scheduled_tokens=np.array([1, 1]))


def test_call_serves_plain_steps_with_v1s_draw(monkeypatch):
    provider = _Provider()
    call, original_calls, outputs = _call(provider, monkeypatch)
    assert call(_sampler(temperature=(0.5, 0.0)), torch.zeros(2, VOCAB), BATCH) == "fused"
    # Per-row temperature in batch order (greedy rows 0), the generator's (seed, offset), k, score temperature.
    assert provider.calls == [([0.0, 0.5], 7, 8, 1, None)]
    assert not original_calls and len(outputs) == 1
    # An all-greedy step draws no noise: no temperature, no generator advance, as V1's all_greedy.
    call(_sampler(temperature=(0.0, 1e-6)), torch.zeros(2, VOCAB), BATCH)
    assert provider.calls[-1] == (None, 0, 0, 1, None)
    sampler = _set(_sampler(), ("top_p", "np"), 0.9)
    assert call(sampler, torch.zeros(2, VOCAB), BATCH) == "vllm" and original_calls == [BATCH]


def test_call_without_a_provider_keeps_v2(monkeypatch):
    call, original_calls, _ = _call(None, monkeypatch)
    assert call(_sampler(), torch.zeros(2, VOCAB), BATCH) == "vllm" and len(original_calls) == 1


def test_disabled_by_flag_installs_nothing(monkeypatch):
    before_call, before_add = Sampler.__call__, Sampler.add_request
    monkeypatch.setenv("ISOEXEC_FUSED_SAMPLER", "0")
    assert patches.patch_vllm_v2_fused_sampler() is False
    assert Sampler.__call__ is before_call and Sampler.add_request is before_add
