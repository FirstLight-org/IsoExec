"""General HF conversion operations and Qwen3.5 declarations."""

import json
from pathlib import Path
from typing import cast

import pytest
import torch
from transformers import Qwen3_5Config
from transformers.core_model_loading import Split, WeightConverter, WeightRenaming

from isoexec.models.qwen3_5.weight import checkpoint_conversion_mapping
from isoexec.weights import AddScalar, Deinterleave


def _config() -> Qwen3_5Config:
    path = Path(__file__).parents[2] / "models/qwen3_5/tests/fixtures/qwen3_5_9b.json"
    raw = json.loads(path.read_text())
    raw.pop("model_type")
    return Qwen3_5Config(**raw)


@pytest.mark.parametrize("dim", [0, 1, -1])
def test_deinterleave_preserves_head_width_and_roundtrips(dim: int) -> None:
    tensor = torch.arange(48).reshape(12, 4) if dim == 0 else torch.arange(48).reshape(4, 12)
    operation = Deinterleave(groups=3, parts=2, dim=dim)
    result = operation.convert({"packed": tensor}, ["packed"], ["shuffled"])
    order = [0, 1, 4, 5, 8, 9, 2, 3, 6, 7, 10, 11]
    expected = tensor[order] if dim == 0 else tensor[:, order]
    assert torch.equal(result["shuffled"], expected)
    restored = operation.reverse_op.convert(result, ["shuffled"], ["packed"])
    assert torch.equal(cast(torch.Tensor, restored["packed"]), tensor)


@pytest.mark.parametrize("groups,parts", [(0, 2), (2, 0), (-1, 2)])
def test_deinterleave_refuses_invalid_groups(groups: int, parts: int) -> None:
    with pytest.raises(ValueError, match="positive integers"):
        Deinterleave(groups=groups, parts=parts)


def test_deinterleave_refuses_incomplete_groups() -> None:
    with pytest.raises(ValueError, match="complete groups"):
        Deinterleave(groups=3, parts=2).convert({"source": torch.zeros(13, 4)}, ["source"], ["target"])


def test_add_scalar_preserves_hf_operation_and_reverse_semantics() -> None:
    source = torch.tensor([1.0, 1.5], dtype=torch.bfloat16)
    operation = AddScalar(-1.0, dtype=torch.float32)
    converted = operation.convert({"gamma": source}, ["gamma"], ["out_norm"])
    assert converted["out_norm"].dtype == torch.float32
    assert torch.equal(converted["out_norm"], torch.tensor([0.0, 0.5]))
    restored = operation.reverse_op.convert(converted, ["out_norm"], ["gamma"])
    assert torch.equal(cast(torch.Tensor, restored["gamma"]), source.float())


def test_qwen35_uses_official_split_for_qkv_and_q_gate() -> None:
    mapping = checkpoint_conversion_mapping(_config())
    assert mapping and all(isinstance(item, (WeightRenaming, WeightConverter)) for item in mapping)
    qkv = next(
        item
        for item in mapping
        if isinstance(item, WeightConverter) and item.source_patterns == [r"\.linear_attn\.in_proj_qkv\.weight$"]
    )
    assert len(qkv.operations) == 1 and isinstance(qkv.operations[0], Split)
    assert qkv.target_patterns == [".mixer.q", ".mixer.k", ".mixer.v"]
    q_gate = next(
        item
        for item in mapping
        if isinstance(item, WeightConverter) and item.source_patterns == [r"\.self_attn\.q_proj\.weight$"]
    )
    assert isinstance(q_gate.operations[0], Deinterleave) and isinstance(q_gate.operations[1], Split)
