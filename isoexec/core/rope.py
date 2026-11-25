"""Build shared rotary angle tables from positional configuration. Each forward carries one table
on SequenceContext; text and MRoPE positions share the same lifetime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from isoexec.schema import Attn, SequenceContext


@dataclass(frozen=True)
class RotaryTables:
    rotary_dim: int  # dims rotated (first rotary_dim of head_dim)
    style: str  # "neox" | "interleaved"
    theta: float
    fn: Callable[[torch.Tensor], torch.Tensor]  # positions[T] or [3,T] -> angles [T, rotary_dim//2] fp32

    def __call__(self, positions: torch.Tensor) -> torch.Tensor:
        """The fp32 angle table of ``positions``: a pure function of (positions, theta, rotary_dim)."""
        return self.fn(positions)


def inv_freq(theta: float, rotary_dim: int, device=None) -> torch.Tensor:
    """``1 / theta^(2i/rotary_dim)`` for i in [0, rotary_dim/2) in fp32 -- Megatron/HF's formula
    (fp32 ``arange`` divided by the dim, then ``theta ** (.)``)."""
    if rotary_dim % 2:
        raise ValueError(f"rotary_dim must be even, got {rotary_dim}")
    ar = torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device)
    return 1.0 / (theta ** (ar / rotary_dim))


def rotary_freqs(positions: torch.Tensor, theta: float, rotary_dim: int) -> torch.Tensor:
    """``positions[:, None] * inv_freq[None, :]`` in fp32 -> [T, rotary_dim/2]."""
    f = inv_freq(theta, rotary_dim, device=positions.device)
    return torch.outer(positions.to(torch.float32), f)


def resolve_rope(spec: Attn) -> RotaryTables:
    """The angle table of one softmax row. Every ``Attn`` ropes -- a row that did not would be a
    different kind, and no shipped model has one."""
    rotary_dim = spec.rotary_dim
    if rotary_dim <= 0 or rotary_dim > spec.head_dim or rotary_dim % 2:
        raise ValueError(
            f"partial_rotary {spec.partial_rotary} gives rotary_dim {rotary_dim} for head_dim {spec.head_dim}"
        )
    theta = float(spec.rope_theta)

    def fn(positions: torch.Tensor) -> torch.Tensor:
        if positions.ndim == 1:
            return rotary_freqs(positions, theta, rotary_dim)
        if positions.ndim != 2 or positions.shape[0] != 3:
            raise ValueError(f"positions: expected [T] or [3,T], got {tuple(positions.shape)}")
        sections = spec.mrope_section
        if sections is None or len(sections) != 3 or sum(sections) != rotary_dim // 2:
            raise ValueError("mrope_section: [3,T] positions require three sections covering rotary_dim/2")
        f = inv_freq(theta, rotary_dim, device=positions.device)
        freqs = positions.to(torch.float32)[:, :, None] * f[None, None, :]
        if spec.mrope_interleaved:
            # Pinned Qwen3.5 layout: T is the base; H/W replace their interleaved columns.
            result = freqs[0]
            for axis in (1, 2):
                if 3 * (sections[axis] - 1) + axis >= rotary_dim // 2:
                    raise ValueError("mrope_section: interleaved H/W columns exceed rotary_dim/2")
                columns = slice(axis, sections[axis] * 3, 3)
                result[:, columns] = freqs[axis, :, columns]
            return result
        chunks = freqs.split(sections, dim=-1)
        return torch.cat([chunk[axis] for axis, chunk in enumerate(chunks)], dim=-1)

    return RotaryTables(rotary_dim=rotary_dim, style=spec.rope_style, theta=theta, fn=fn)


def ensure_rope_freqs(seq: SequenceContext, tables: RotaryTables) -> torch.Tensor:
    """Populate seq.rope_freqs once per forward and return it. Callers must create a fresh context
    when positions change.
    """
    freqs = seq.rope_freqs
    if freqs is None:
        freqs = tables(seq.positions)
        seq.rope_freqs = freqs
    return freqs


__all__ = [
    "RotaryTables",
    "ensure_rope_freqs",
    "inv_freq",
    "resolve_rope",
    "rotary_freqs",
]
