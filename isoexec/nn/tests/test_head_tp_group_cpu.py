import pytest
import torch
from torch import distributed as dist

from isoexec.nn import LMHead


@pytest.mark.parametrize("rank", [0, 1])
def test_native_tp_group_masks_local_vocabulary_padding(rank):
    head = LMHead(vocab=6, hidden=4, padded_vocab=8)
    head.tp_group = dist.ProcessGroup(rank, 2)
    logits = torch.zeros(3, 4)

    trimmed = head.trim_to_vocab(logits)

    masked = [False] * 4 if rank == 0 else [False, False, True, True]
    assert trimmed.shape == (3, 4)
    assert torch.isneginf(trimmed).tolist() == [masked] * 3
    assert head.trim_to_vocab(torch.zeros(3, 8)).shape == (3, 6)


@pytest.mark.parametrize("vocab,rank", [(8, 0), (8, 1), (6, 0)])
def test_tp_shard_without_padding_preserves_storage_and_gradients(vocab, rank):
    head = LMHead(vocab=vocab, hidden=4, padded_vocab=8)
    head.tp_group = dist.ProcessGroup(rank, 2)
    logits = torch.tensor([[0.0, -0.0, float("nan"), float("inf")]], requires_grad=True)

    trimmed = head.trim_to_vocab(logits)

    assert trimmed is logits
    assert torch.equal(trimmed.detach().view(torch.int32), logits.detach().view(torch.int32))
    trimmed.backward(torch.ones_like(trimmed))
    assert torch.equal(logits.grad, torch.ones_like(logits))


def test_tp_padding_is_masked_without_mutating_input_and_blocks_gradients():
    head = LMHead(vocab=6, hidden=4, padded_vocab=8)
    head.tp_group = dist.ProcessGroup(1, 2)
    logits = torch.arange(4.0).reshape(1, 4).requires_grad_()

    trimmed = head.trim_to_vocab(logits)

    assert torch.equal(logits, torch.arange(4.0).reshape(1, 4))
    assert torch.equal(trimmed[:, :2], logits[:, :2])
    assert torch.isneginf(trimmed[:, 2:]).all()
    trimmed.backward(torch.ones_like(trimmed))
    assert torch.equal(logits.grad, torch.tensor([[1.0, 1.0, 0.0, 0.0]]))


def test_tp_unpadded_head_still_rejects_invalid_shard_width():
    head = LMHead(vocab=8, hidden=4)
    head.tp_group = dist.ProcessGroup(0, 2)
    with pytest.raises(RuntimeError, match="neither the global vocabulary nor its local shard"):
        head.trim_to_vocab(torch.zeros(3, 3))
