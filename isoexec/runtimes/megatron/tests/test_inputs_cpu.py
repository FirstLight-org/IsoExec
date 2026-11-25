"""Packed batch fields preserve sequence boundaries and supplied positions."""

import pytest
import torch

from isoexec.runtimes.megatron._inputs import packed_sequence_context


@pytest.mark.parametrize("mrope", [False, True])
def test_packed_context_resets_each_sequence_and_keeps_host_boundaries(mrope):
    ids = torch.tensor([[5, 6, 7, 8, 9]], dtype=torch.int32)
    cu = torch.tensor([0, 2, 5], dtype=torch.int32)

    seq = packed_sequence_context(ids, None, cu, max_q=3, max_kv=3, mrope=mrope)

    expected = torch.tensor([0, 1, 0, 1, 2])
    assert torch.equal(seq.positions, expected.expand(3, -1) if mrope else expected)
    assert seq.positions.is_contiguous()
    assert torch.equal(seq.input_ids, ids[0].long())
    assert torch.equal(seq.cu_seqlens, cu)
    assert seq.cu_seqlens_cpu.device.type == "cpu" and torch.equal(seq.cu_seqlens_cpu, cu)
    assert seq.max_q == seq.max_kv == 3


def test_packed_context_preserves_explicit_positions():
    ids = torch.tensor([[5, 6, 7, 8, 9]])
    positions = torch.tensor([[10, 11, 20, 21, 22]])
    cu = torch.tensor([0, 2, 5], dtype=torch.int32)

    seq = packed_sequence_context(ids, positions, cu, max_q=3, max_kv=None, mrope=False)

    assert torch.equal(seq.positions, positions[0])
    assert torch.equal(positions, torch.tensor([[10, 11, 20, 21, 22]]))
    assert seq.max_kv == 3


def test_later_stage_context_uses_packed_boundaries_without_token_ids():
    cu = torch.tensor([0, 2, 7], dtype=torch.int32)
    seq = packed_sequence_context(None, None, cu, max_q=5, max_kv=5, mrope=True)
    assert seq.input_ids is None
    assert seq.positions.shape == (3, 7)
    assert torch.equal(seq.positions[0], torch.tensor([0, 1, 0, 1, 2, 3, 4]))
    assert torch.equal(seq.cu_seqlens, cu) and seq.max_q == seq.max_kv == 5


@pytest.mark.parametrize("boundaries", [[1, 2, 5], [0, 2, 4], [0, 0, 5], [0, 5]])
def test_packed_context_rejects_inconsistent_boundaries(boundaries):
    with pytest.raises(ValueError, match="boundaries must cover"):
        packed_sequence_context(torch.tensor([[5, 6, 7, 8, 9]]), None,
                                torch.tensor(boundaries, dtype=torch.int32), max_q=3, max_kv=3, mrope=False)
