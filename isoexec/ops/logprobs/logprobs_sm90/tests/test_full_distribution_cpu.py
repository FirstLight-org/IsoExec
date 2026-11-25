"""Full-distribution observers validate ownership and never count as production engagement."""

import pytest
import torch

from isoexec.ops.logprobs.logprobs_sm90 import integration


@pytest.mark.parametrize("shape", [(), (64,), (0, 64), (1, 0), (1, 1, 64)])
def test_observer_refuses_invalid_rows(shape):
    with pytest.raises(RuntimeError, match="non-empty"):
        integration.full_logprobs_for_check(torch.empty(shape))


@pytest.mark.parametrize("logits", [torch.zeros(2, 64, dtype=torch.bfloat16), torch.zeros(2, 128)[:, ::2]])
def test_observer_requires_contiguous_fp32(logits):
    with pytest.raises(RuntimeError, match="contiguous fp32"):
        integration.full_logprobs_for_check(logits)


def test_observer_uses_current_provider_without_counting_a_serve(monkeypatch):
    x = torch.zeros(2, 64)
    result = torch.ones_like(x)
    before = integration.stats()

    def full(logits):
        assert logits is x and not torch.is_grad_enabled()
        return result

    monkeypatch.setattr(integration.ops, "full_logprobs", full)
    assert integration.full_logprobs_for_check(x) is result
    assert integration.stats() == before


def test_sharded_observer_requires_rank_owned_vocabulary(monkeypatch):
    group = object()
    monkeypatch.setattr(integration.dist, "get_world_size", lambda **kw: 2)
    monkeypatch.setattr(integration.dist, "get_rank", lambda **kw: 1)
    x = torch.zeros(2, 32)
    with pytest.raises(RuntimeError, match="bounds"):
        integration.full_logprobs_for_check_sharded(
            x, vocab_start_index=0, vocab_end_index=32, group=group
        )
    before = integration.stats()

    def sharded(logits, *, group):
        assert logits is x and not torch.is_grad_enabled()
        return x

    monkeypatch.setattr(integration.ops, "full_logprobs_sharded", sharded)
    assert integration.full_logprobs_for_check_sharded(
        x, vocab_start_index=32, vocab_end_index=64, group=group
    ) is x
    assert integration.stats() == before
