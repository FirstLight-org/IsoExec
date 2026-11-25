"""Adapter dispatch preserves the kernel result without periodic reference work."""

from types import SimpleNamespace

import pytest
import torch

from isoexec.ops.logprobs.logprobs_sm100 import integration


def _unexpected(*args, **kwargs):
    raise AssertionError("adapter evaluated a reference or added a collective")


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(integration, "_S", {
        "full_calls": 0, "full_rows": 0, "sampled_calls": 0,
        "tokens": 0, "declined": 0, "decline_reason": "",
    })
    monkeypatch.setattr(integration.dist, "all_gather", _unexpected)
    monkeypatch.setattr(integration.dist, "all_reduce", _unexpected)


def test_full_adapter_never_evaluates_reference(monkeypatch):
    logits = SimpleNamespace(ndim=2, is_cuda=True, dtype=torch.bfloat16, shape=(2, 248320))
    temperature = torch.tensor(0.7)
    result = torch.tensor([[float("nan")]])
    calls = []

    def kernel(value, *, temperature):
        assert value is logits and not torch.is_grad_enabled()
        calls.append(temperature)
        return result

    monkeypatch.setattr(integration.ops, "full_logprobs", kernel)
    for _ in range(513):
        assert integration.full_logprobs(logits, reference=_unexpected, temperature=temperature) is result
    assert len(calls) == 513 and all(t is temperature for t in calls)
    assert integration.stats() == {
        "full_calls": 513, "full_rows": 1026, "sampled_calls": 0,
        "tokens": 0, "declined": 0, "decline_reason": "",
    }


@pytest.mark.parametrize("world", [1, 4])
def test_sampled_adapter_never_evaluates_reference_or_changes_autograd(monkeypatch, world):
    width = 248320 // world
    logits = SimpleNamespace(is_cuda=True, dtype=torch.float32, shape=(1, 2, width))
    target = torch.tensor([[7, 17]])
    group = object() if world > 1 else None
    start = width if world > 1 else 0
    leaf = torch.tensor(2.0, requires_grad=True)
    result = leaf.square()
    calls = []
    monkeypatch.setattr(integration.dist, "get_world_size", lambda *, group: world)

    def kernel(value, targets, *, vocab_start_index, group):
        assert value is logits and targets is target and torch.is_grad_enabled()
        calls.append((vocab_start_index, group))
        return result

    monkeypatch.setattr(integration.ops, "sampled_logprobs", kernel)
    for _ in range(513):
        assert integration.sampled_logprobs(
            logits, target, vocab_start_index=start, vocab_end_index=start + width,
            group=group, reference=_unexpected,
        ) is result
    assert calls == [(start, group)] * 513
    result.backward()
    assert leaf.grad.item() == 4
    assert integration.stats() == {
        "full_calls": 0, "full_rows": 0, "sampled_calls": 513,
        "tokens": 1026, "declined": 0, "decline_reason": "",
    }


def test_decline_does_not_evaluate_reference_or_count_a_serve(monkeypatch):
    monkeypatch.setattr(integration.ops, "full_logprobs", _unexpected)
    monkeypatch.setattr(integration.ops, "sampled_logprobs", _unexpected)
    assert integration.full_logprobs(torch.zeros(2, 64), reference=_unexpected) is None
    unsupported = SimpleNamespace(ndim=2, is_cuda=True, dtype=torch.float32, shape=(2, 248077))
    assert integration.full_logprobs(unsupported, reference=_unexpected) is None
    shard = SimpleNamespace(is_cuda=True, dtype=torch.float32, shape=(1, 2, 248320))
    assert integration.sampled_logprobs(
        shard, torch.tensor([[0, 1]]), vocab_start_index=0, vocab_end_index=1,
        group=None, reference=_unexpected,
    ) is None
    assert integration.stats() == {
        "full_calls": 0, "full_rows": 0, "sampled_calls": 0,
        "tokens": 0, "declined": 3, "decline_reason": "vocab 248320x1 / partition",
    }
