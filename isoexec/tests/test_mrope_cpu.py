"""MRoPE uses checkpoint axes while identical text axes preserve the old table exactly."""

import dataclasses

import pytest
import torch

from isoexec.core.rope import inv_freq, resolve_rope
from isoexec.schema import Attn

SPEC = Attn(num_q_heads=4, num_kv_heads=2, head_dim=16, rope_theta=10000.0, mrope_section=(3, 3, 2))


@pytest.mark.parametrize("interleaved", [False, True])
def test_distinct_axes_follow_the_declared_sections(interleaved):
    spec = dataclasses.replace(SPEC, mrope_interleaved=interleaved)
    positions = torch.tensor([[0, 1, 2], [4, 8, 12], [3, 6, 9]], dtype=torch.int64, device="cpu")
    axes = (0, 1, 2, 0, 1, 2, 0, 1) if interleaved else (0, 0, 0, 1, 1, 1, 2, 2)
    f = inv_freq(spec.rope_theta, spec.rotary_dim, device=positions.device)
    expected = torch.stack([positions[axis].float() * f[column] for column, axis in enumerate(axes)], dim=-1)
    table = resolve_rope(spec)(positions)
    assert torch.equal(table, expected)
    order = torch.tensor([2, 0, 1], dtype=torch.int64, device=positions.device)
    assert torch.equal(resolve_rope(spec)(positions[:, order]), table[order])


@pytest.mark.parametrize("interleaved", [False, True])
def test_text_axes_are_bitwise_the_original_one_dimensional_table(interleaved):
    spec = dataclasses.replace(SPEC, mrope_interleaved=interleaved)
    positions = torch.arange(67, dtype=torch.int64, device="cpu")
    tables = resolve_rope(spec)
    assert torch.equal(tables(positions), tables(positions.expand(3, -1)))


def test_multimodal_positions_without_geometry_refuse():
    spec = dataclasses.replace(SPEC, mrope_section=None)
    with pytest.raises(ValueError, match="mrope_section"):
        resolve_rope(spec)(torch.zeros((3, 2), dtype=torch.int64, device="cpu"))

