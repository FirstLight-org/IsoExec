"""Torch transcriptions of the rope_sm90 v1 rounding recipe, for explicitly requested reference execution.
Separate float64 definitions provide autograd references for backward checks."""

from __future__ import annotations

import math
import struct

import torch


def inv_freq_double(theta: float, rotary_dim: int) -> list[float]:
    """``1 / theta ** (2j / rotary_dim)`` in host IEEE double (C ``pow``, then a correctly rounded division)."""
    if rotary_dim <= 0 or rotary_dim % 2:
        raise ValueError(f"rotary_dim must be positive and even, got {rotary_dim}")
    return [1.0 / (float(theta) ** ((2 * j) / rotary_dim)) for j in range(rotary_dim // 2)]


def _f32(x: float) -> float:
    return struct.unpack("f", struct.pack("f", x))[0]


def table_rows_host(theta: float, rotary_dim: int, positions) -> torch.Tensor:
    """Host (glibc) evaluation of table rows: fp32_rn(cos(double(p) * inv_freq_j)) | fp32_rn(sin(.)).

    Independent provider check for the device table (the contract names the device libm; the two agree
    wherever neither double result lies within 2^-29 relative of an fp32 rounding boundary)."""
    inv = inv_freq_double(theta, rotary_dim)
    rows = []
    for p in positions:
        a = [float(p) * f for f in inv]
        rows.append([_f32(math.cos(x)) for x in a] + [_f32(math.sin(x)) for x in a])
    return torch.tensor(rows, dtype=torch.float32)


def mrope_masks(rotary_dim: int, sections, interleaved: bool) -> tuple[int, int]:
    """Column -> axis bitmasks (axis 1, axis 2) for MRoPE, matching ``common/rope.py::resolve_rope``."""
    if sections is None:
        return 0, 0
    half = rotary_dim // 2
    if len(sections) != 3 or sum(sections) != half:
        raise ValueError("mrope_section: three sections covering rotary_dim/2 required")
    axis = [0] * half
    if interleaved:
        for a in (1, 2):
            for col in range(a, sections[a] * 3, 3):
                if col >= half:
                    raise ValueError("mrope_section: interleaved H/W columns exceed rotary_dim/2")
                axis[col] = a
    else:
        col = 0
        for a, n in enumerate(sections):
            for _ in range(n):
                axis[col] = a
                col += 1
    m1 = sum(1 << j for j, a in enumerate(axis) if a == 1)
    m2 = sum(1 << j for j, a in enumerate(axis) if a == 2)
    if half > 64 and (m1 or m2):
        raise ValueError("SM90 MRoPE uses 64-bit column masks; rotary_dim must be <=128")
    return m1, m2


def gather_cos_sin(table: torch.Tensor, positions: torch.Tensor, m1: int, m2: int):
    """Per-token, per-column cos/sin [T, h] from the table (the kernel's in-kernel lookup)."""
    half = table.shape[1] // 2
    if positions.ndim == 1:
        rows = table[positions]
        return rows[:, :half], rows[:, half:]
    axis = torch.tensor(
        [((m1 >> j) & 1) + 2 * ((m2 >> j) & 1) for j in range(half)], device=positions.device, dtype=torch.long
    )
    pos = positions[axis]  # [h, T]
    cols = torch.arange(half, device=positions.device)
    cos = table[pos, cols[:, None]].t()
    sin = table[pos, (cols + half)[:, None]].t()
    return cos, sin


def _to_neox(x: torch.Tensor, rot: int, interleaved: bool):
    r = x[..., :rot]
    if interleaved:
        return r[..., 0::2], r[..., 1::2]
    return r[..., : rot // 2], r[..., rot // 2 :]


def _from_pairs(y1: torch.Tensor, y2: torch.Tensor, interleaved: bool) -> torch.Tensor:
    if interleaved:
        return torch.stack((y1, y2), dim=-1).flatten(-2)
    return torch.cat((y1, y2), dim=-1)


def rope_recipe(x, positions, table, m1=0, m2=0, *, rot=None, in_interleaved=False, out_interleaved=False,
                transpose=False):
    """Forward recipe on one operand ``x [T, H, D]``; returns ``[T, H, D]`` in x.dtype."""
    rot = table.shape[1] if rot is None else rot
    cos, sin = gather_cos_sin(table, positions, m1, m2)
    c, s = cos[:, None, :], sin[:, None, :]
    if transpose:
        s = -s
    x1, x2 = _to_neox(x.float(), rot, in_interleaved)
    y1 = x1 * c - x2 * s
    y2 = x2 * c + x1 * s
    y = _from_pairs(y1, y2, out_interleaved).to(x.dtype)
    return torch.cat((y, x[..., rot:]), dim=-1)


def tree_sum(v: torch.Tensor) -> torch.Tensor:
    """Balanced binary tree over the last dim (power of two), lower index on the left."""
    n = v.shape[-1]
    if n & (n - 1):
        pad = (1 << n.bit_length()) - n
        v = torch.cat((v, v.new_zeros(*v.shape[:-1], pad)), dim=-1)
    while v.shape[-1] > 1:
        v = v[..., 0::2] + v[..., 1::2]
    return v[..., 0]


def qk_norm_rope_recipe(x, w, positions, table, eps, m1=0, m2=0, *, rot=None, zero_centered=True):
    """Forward recipe for one operand ``x [T, H, d]`` -> (y [T, H, d] in x.dtype, inv [T, H] fp32)."""
    rot = (table.shape[1] if table is not None else 0) if rot is None else rot
    d = x.shape[-1]
    xf = x.float()
    ss = tree_sum(xf * xf)
    inv = 1.0 / torch.sqrt(ss / float(d) + torch.tensor(eps, dtype=torch.float32, device=x.device))
    g = w.float()
    if zero_centered:
        g = 1.0 + g
    n = (xf * inv[..., None]) * g
    if rot:
        if table is None:
            raise ValueError("A positive rotary_dim requires a RoPE table")
        cos, sin = gather_cos_sin(table, positions, m1, m2)
        c, s = cos[:, None, :], sin[:, None, :]
        n1, n2 = n[..., : rot // 2], n[..., rot // 2 : rot]
        n = torch.cat((n1 * c - n2 * s, n2 * c + n1 * s, n[..., rot:]), dim=-1)
    return n.to(x.dtype), inv


# float64 calculus references (backward checks)
def rope_float64(x, cos, sin, rot, interleaved_in=False, interleaved_out=False):
    """x float64 [T,H,D]; cos/sin float64 [T, rot/2]."""
    c, s = cos[:, None, :], sin[:, None, :]
    x1, x2 = _to_neox(x, rot, interleaved_in)
    y = _from_pairs(x1 * c - x2 * s, x2 * c + x1 * s, interleaved_out)
    return torch.cat((y, x[..., rot:]), dim=-1)


def qk_norm_rope_float64(x, w, cos, sin, eps, rot, zero_centered=True):
    d = x.shape[-1]
    inv = torch.rsqrt((x * x).sum(-1, keepdim=True) / d + eps)
    g = 1.0 + w if zero_centered else w
    n = x * inv * g
    if rot:
        c, s = cos[:, None, :], sin[:, None, :]
        n1, n2 = n[..., : rot // 2], n[..., rot // 2 : rot]
        n = torch.cat((n1 * c - n2 * s, n2 * c + n1 * s, n[..., rot:]), dim=-1)
    return n
