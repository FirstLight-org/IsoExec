"""SM100 RoPE and fused q/k norm use FP32 rounded tables."""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Read the recipe identity from import-light metadata.
from isoexec.ops.attention.rope_sm100 import reference as _ref
from isoexec.ops.attention.rope_sm100._extension import extension
from isoexec.ops.attention.rope_sm100.spec import RECIPE_VERSION

__all__ = ["RECIPE_VERSION", "RopeTable", "build_table", "qk_norm_rope", "rope"]

# Backward weight gradients: one plain-stored fp32 partial per (2**_GRAD_SHIFT-token group, head,
# column), tokens folded in ascending order, then a fixed two-level tree over the partial rows
# (csrc/qknorm.cu). No float atomics; the bits are a function of the layout alone.
_GRAD_SHIFT = 4


@dataclass(frozen=True, eq=False)
class RopeTable:
    """Per-model cos/sin table. ``table[p] = [cos_0..cos_{h-1} | sin_0..sin_{h-1}]`` fp32, ``h = rotary_dim/2``."""

    table: torch.Tensor
    rotary_dim: int
    theta: float
    m1: int = 0  # MRoPE: bit j set -> column j reads position axis 1
    m2: int = 0  # MRoPE: bit j set -> column j reads position axis 2

    @property
    def max_positions(self) -> int:
        return self.table.shape[0]


def build_table(theta: float, rotary_dim: int, max_positions: int, device, *, mrope_section=None,
                mrope_interleaved: bool = False) -> RopeTable:
    """Build the table for positions ``[0, max_positions)``. Positions outside it are a device assert."""
    inv = torch.tensor(_ref.inv_freq_double(theta, rotary_dim), dtype=torch.float64, device=device)
    table = torch.empty((int(max_positions), rotary_dim), dtype=torch.float32, device=device)
    extension().table(inv, table, 0)
    m1, m2 = _ref.mrope_masks(rotary_dim, mrope_section, bool(mrope_interleaved))
    return RopeTable(table, rotary_dim, float(theta), m1, m2)


# rope
def _rope_raw(q, k, positions, tab: RopeTable, in_i: bool, out_i: bool, transpose: bool):
    out = extension().rope(q, k, positions, tab.table, tab.m1, tab.m2, in_i, out_i, transpose, False)
    if out:
        return out[0], out[1]
    raise ValueError("SM100 rope declined the operands: requires CUDA [T,H,D] bf16/fp16/fp32 tensors, "
                     "int64 positions [T] or [3,T], and 0 < rotary_dim <= D divisible by 4; "
                     "use the explicit reference recipe for unsupported inputs")


class _RopeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, positions, tab, in_i, out_i):
        ctx.meta = (positions, tab, in_i, out_i, q.shape, None if k is None else k.shape, q.dtype, q.device)
        return _rope_raw(q, k, positions, tab, in_i, out_i, False)

    @staticmethod
    def backward(ctx, dq, dk):
        positions, tab, in_i, out_i, qs, ks, dtype, device = ctx.meta
        if dq is None:
            dq = torch.zeros(qs, dtype=dtype, device=device)
        if ks is not None and dk is None:
            dk = torch.zeros(ks, dtype=dtype, device=device)
        gq, gk = _rope_raw(dq, dk if ks is not None else None, positions, tab, out_i, in_i, True)
        return gq, gk, None, None, None, None


_STYLES = {"neox": False, "interleaved": True}


def rope(q: torch.Tensor, k: torch.Tensor | None, positions: torch.Tensor, table: RopeTable, *,
         style: str = "neox", out_style: str | None = None, inplace: bool = False):
    """Rotate ``q [T, Hq, D]`` and ``k [T, Hk, D]`` over their first ``table.rotary_dim`` lanes."""
    in_i = _STYLES[style]
    out_i = in_i if out_style is None else _STYLES[out_style]
    if inplace:
        if torch.is_grad_enabled() and (q.requires_grad or (k is not None and k.requires_grad)):
            raise ValueError("rope(inplace=True) records no autograd node")
        out = extension().rope(q, k, positions, table.table, table.m1, table.m2, in_i, out_i, False, True)
        if not out:
            raise ValueError("rope(inplace=True) needs equal in/out layouts, rotary_dim % 4 == 0 and operands "
                             "on even element offsets of 4-byte aligned storage")
        return q, k
    if torch.is_grad_enabled() and (q.requires_grad or (k is not None and k.requires_grad)):
        return _RopeFn.apply(q, k, positions, table, in_i, out_i)
    return _rope_raw(q, k, positions, table, in_i, out_i, False)


# fused q/k norm + rope
def _qk_fwd_raw(q, k, wq, wk, positions, tab, eps, zc, want_inv, inplace=False):
    table = None if tab is None else tab.table
    m1, m2 = (0, 0) if tab is None else (tab.m1, tab.m2)
    out = extension().qknorm_rope_forward(q, k, wq, wk, positions if tab is not None else None, table, m1, m2,
                                          float(eps), zc, want_inv, inplace)
    if out:
        return out[0], out[1], out[2]
    raise ValueError("SM100 qk_norm_rope forward declined the operands: requires CUDA [T,H,D] "
                     "bf16/fp16/fp32 tensors with D in {64,128,256}, matching per-head weights "
                     "and supported RoPE metadata (and, with inplace=True, operands on even element "
                     "offsets of 4-byte aligned storage); use the explicit reference recipe otherwise")


class _QKNormRopeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, wq, wk, positions, tab, eps, zc):
        yq, yk, inv = _qk_fwd_raw(q, k, wq, wk, positions, tab, eps, zc, True)
        ctx.save_for_backward(q, k, wq, wk, inv)
        ctx.meta = (positions, tab, eps, zc)
        return yq, yk

    @staticmethod
    def backward(ctx, dyq, dyk):
        q, k, wq, wk, inv = ctx.saved_tensors
        positions, tab, eps, zc = ctx.meta
        if dyq is None:
            dyq = torch.zeros_like(q)
        if dyk is None:
            dyk = torch.zeros_like(k)
        needs = ctx.needs_input_grad
        want_w = needs[2] or needs[3]
        table = None if tab is None else tab.table
        m1, m2 = (0, 0) if tab is None else (tab.m1, tab.m2)
        out = extension().qknorm_rope_backward(dyq, dyk, q, k, wq, wk, inv, positions if tab is not None else None,
                                               table, m1, m2, zc, want_w, _GRAD_SHIFT)
        if not out:
            raise ValueError("SM100 qk_norm_rope backward declined the operands or saved inverse norms")
        dxq, dxk, gwq, gwk = out
        return (dxq if needs[0] else None, dxk if needs[1] else None, gwq if needs[2] else None,
                gwk if needs[3] else None, None, None, None, None)


def qk_norm_rope(q: torch.Tensor, k: torch.Tensor, wq: torch.Tensor, wk: torch.Tensor, positions: torch.Tensor | None,
                 table: RopeTable | None, eps: float, *, zero_centered: bool = True, inplace: bool = False):
    """Per-head RMSNorm (``(x*inv)*(1+w)`` when ``zero_centered`` else ``(x*inv)*w``) followed by partial neox RoPE over the first ``table.rotary_dim`` lanes.

    Differentiable in q, k, wq and wk; the forward bits do not depend on whether a graph is recorded."""
    if inplace:
        if torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or wq.requires_grad
                                        or wk.requires_grad):
            raise ValueError("qk_norm_rope(inplace=True) records no autograd node")
        yq, yk, _ = _qk_fwd_raw(q, k, wq, wk, positions, table, eps, bool(zero_centered), False, True)
        return yq, yk
    if torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or wq.requires_grad or wk.requires_grad):
        return _QKNormRopeFn.apply(q, k, wq, wk, positions, table, float(eps), bool(zero_centered))
    yq, yk, _ = _qk_fwd_raw(q, k, wq, wk, positions, table, eps, bool(zero_centered), False)
    return yq, yk
