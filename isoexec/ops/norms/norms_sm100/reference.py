"""Executable statement of the norms_sm100 numerical schedule, independent of the CUDA kernels.

The kernels must reproduce THIS, bit for bit, for every batch size, composition, padding, phase and execution
plan. It is written with torch tensor ops in fp32, one rounded operation per line, and is vectorised over
rows and segments only, never over a segment's chain, which is the ordered part.

Every sum is a plain fp32 sum scheduled by :mod:`isoexec.ops.schedule_policy` (rule 3, scalar step; see
``schedule.py``): segments of ``S`` consecutive positions, each a chain from ``+0`` in ascending order, the last
segment short, combined by the adjacent tree (pair (0, 1), (2, 3), ...; an unpaired last partial moves up).

    ssq     = tree over segments g of  chain_{c in segment g} fl32(acc + fl32(v_c * v_c))     (hidden axis)
    rstd    = rn32(1 / sqrt(rn32(rn32(ssq / N) + eps)))                                       (correctly rounded)
    y       = rn_bf16( fl32( fl32(v * rstd) * g ) ),  g = w | fl32(1 + w)
    gated   y = rn_bf16( fl32( fl32(fl32(v * rstd) * g) * silu(z) ) ),  silu(z) = z / (1 + exp(-z))
    backward  xhat = fl32(x * rstd), gy = fl32(dy * g) (dy when there is no weight)
              dn   = rn32( tree over hidden segments of chain fl32(acc + fl32(xhat * gy)) / N )
              dx   = rn_bf16( fl32(rstd * fl32(gy - fl32(xhat * dn))) [+ dres] )
              dW[h] = tree over ROW segments of chain_{rows ascending} fl32(acc + fl32(dy * xhat))

``v`` is the exact fp32 value of the bf16 input; for the fused add form ``v`` is the bf16-rounded residual sum
``rn_bf16(x + r)``, which is also what the kernel stores. This schedule is not norms_sm90's (strided lanes and a
descending-XOR fold), so the two architectures no longer agree bit for bit on these operators.
"""

from __future__ import annotations

import torch

from isoexec.ops.norms.norms_sm100 import schedule
from isoexec.ops.norms.norms_sm90.reference import _gamma, inv_rms, silu


def adjacent_tree(parts: torch.Tensor) -> torch.Tensor:
    """Adjacent tree over the LAST axis: pair (0, 1), (2, 3), ...; an unpaired last partial moves up."""
    while parts.shape[-1] > 1:
        g = parts.shape[-1]
        merged = parts[..., 0 : g - 1 : 2] + parts[..., 1:g:2]
        parts = torch.cat([merged, parts[..., g - 1 :]], dim=-1) if g % 2 else merged
    return parts[..., 0]


def segment_sum(terms: torch.Tensor, seg: "schedule.policy.Segments") -> torch.Tensor:
    """The scheduled fp32 sum of ``terms`` over its LAST axis (terms are summed as given: exact inputs)."""
    if seg.merge != "tree" or seg.G is not None:
        raise ValueError(f"norms_sm100 states segments of fixed length under a tree, not {seg}")
    n, s = terms.shape[-1], seg.S
    g = -(-n // s)
    pad = g * s - n
    t = terms.float()
    if pad:
        # A short last segment adds only the terms that exist. Adding +0 for an absent one gives the same bits:
        # a chain that starts at +0 is never -0, and x + (+0) is x for every other x, NaN payloads included.
        t = torch.cat([t, torch.zeros(*t.shape[:-1], pad, device=t.device, dtype=torch.float32)], dim=-1)
    t = t.view(*t.shape[:-1], g, s)
    acc = torch.zeros(t.shape[:-1], device=t.device, dtype=torch.float32)
    for j in range(s):  # the ordered chain
        acc = acc + t[..., j]
    return adjacent_tree(acc)


def row_sum_of_squares(x: torch.Tensor) -> torch.Tensor:
    """The per-row sum of ``x*x`` in fp32. ``x`` is [M, N] (any float dtype, read exactly)."""
    v = x.float()
    return segment_sum(v * v, schedule.hidden_sum(x.shape[-1]))


def rms_norm(x: torch.Tensor, w: torch.Tensor | None, eps: float, zero_centered: bool) -> torch.Tensor:
    """RMSNorm. Returns bf16 in ``x``'s shape; also the reference for the fused add form's second half."""
    x2 = x.reshape(-1, x.shape[-1])
    r = inv_rms(row_sum_of_squares(x2), x2.shape[-1], eps)
    y = x2.float() * r[:, None]
    g = _gamma(w, zero_centered)
    if g is not None:
        y = y * g
    return y.bfloat16().view(x.shape)


def rms_norm_round_scale(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """``rn_bf16( fl32( rn_bf16(fl32(v * rstd)) * w ) )``: the round-then-scale form."""
    x2 = x.reshape(-1, x.shape[-1])
    r = inv_rms(row_sum_of_squares(x2), x2.shape[-1], eps)
    return ((x2.float() * r[:, None]).bfloat16().float() * w.float()).bfloat16().view(x.shape)


def add_rms_norm(x: torch.Tensor, res: torch.Tensor, w: torch.Tensor | None, eps: float,
                 zero_centered: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """``h = rn_bf16(x + res)`` (the stored residual stream), then RMSNorm on ``h``."""
    h = (x.float() + res.float()).bfloat16()
    return rms_norm(h, w, eps, zero_centered), h


def gated_rms_norm(x: torch.Tensor, z: torch.Tensor, w: torch.Tensor | None, eps: float,
                   zero_centered: bool = True) -> torch.Tensor:
    """Gated out-norm: ``[T, hv, N]`` -> ``[T, hv*N]`` bf16."""
    t, hv, n = x.shape
    x2 = x.reshape(-1, n)
    r = inv_rms(row_sum_of_squares(x2), n, eps)
    y = x2.float() * r[:, None]
    g = _gamma(w, zero_centered)
    if g is not None:
        y = y * g
    return (y * silu(z.reshape(-1, n))).bfloat16().view(t, hv * n)


def rms_backward(dy: torch.Tensor, x: torch.Tensor, w: torch.Tensor | None, eps: float, zero_centered: bool,
                 dres: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
    """``(dx bf16 [M, N], dW fp32 [N] or None)`` of RMSNorm for rows ``x`` [M, N] and upstream ``dy``."""
    m, n = x.shape
    rs = inv_rms(row_sum_of_squares(x), n, eps)
    h = x.float() * rs[:, None]
    g = _gamma(w, zero_centered)
    gy = dy.float() if g is None else dy.float() * g
    dn = segment_sum(h * gy, schedule.hidden_sum(n)) / torch.full((m,), float(n), device=x.device, dtype=torch.float32)
    o = rs[:, None] * (gy - h * dn[:, None])
    if dres is not None:
        o = o + dres.float()
    dx = o.bfloat16()
    if w is None:
        return dx, None
    contrib = dy.float() * h  # fl32(dy * xhat), one rounding
    return dx, segment_sum(contrib.t(), schedule.row_sum(m))


__all__ = ["add_rms_norm", "adjacent_tree", "gated_rms_norm", "inv_rms", "rms_backward", "rms_norm",
           "rms_norm_round_scale", "row_sum_of_squares", "segment_sum", "silu"]
