"""Executable statement of norms_sm90 recipe v1, independent of the CUDA kernels.

The kernels must reproduce THIS, bit for bit, for every batch size, composition, padding, phase and
execution plan (``tests/test_sm90.py::test_matches_the_declared_recipe``). It is written with torch
tensor ops in fp32 -- one rounded operation per line, in the recipe's order -- and is vectorized over
rows and lanes only, never over a lane's chain, which is the ordered part.

Recipe v1, for a row of width ``N = L * 8 * K`` (``L``, ``K`` from :func:`lanes`):

    lane l owns columns  c(k, j) = k*8L + 8l + j   for k in [0, K), j in [0, 8)
    chain   acc_l = +0; for k ascending, j ascending: acc_l = fl32(acc_l + fl32(v*v))
    fold    for s = L/2, L/4, ..., 1:  acc_l = fl32(acc_l + acc_{l^s})     (all lanes hold the root)
    rstd    = rn32(1 / sqrt(rn32(rn32(root / N) + eps)))                   (correctly rounded)
    y       = rn_bf16( fl32( fl32(v * rstd) * g ) ),  g = w | fl32(1 + w)
    gated   y = rn_bf16( fl32( fl32(fl32(v * rstd) * g) * silu(z) ) ),  silu(z) = z / (1 + exp(-z))

``v`` is the exact fp32 value of the bf16 input (bf16 -> fp32 is exact); for the fused add form
``v`` is the bf16-rounded residual sum ``rn_bf16(x + r)``, which is also what the kernel stores.
"""

from __future__ import annotations

import torch


def lanes(n: int) -> tuple[int, int]:
    """``(L, K)`` for width ``n`` -- the same table as ``csrc/plan.h::fwd_lanes``."""
    if n % 256 == 0:
        return 32, n // 256
    if n in (128, 384):
        return 16, n // 128
    raise ValueError(f"norms_sm90 recipe v1 has no fold for width {n}")


def _swizzle(n: int) -> torch.Tensor:
    """Column order as (lane, chain position): ``idx[l, t]`` is the column of lane ``l``'s t-th leaf."""
    L, K = lanes(n)
    k = torch.arange(K).view(K, 1, 1)
    l = torch.arange(L).view(1, L, 1)
    j = torch.arange(8).view(1, 1, 8)
    cols = (k * 8 * L + 8 * l + j)  # (K, L, 8)
    return cols.permute(1, 0, 2).reshape(L, K * 8)  # (L, C)


def row_sum_of_squares(x: torch.Tensor) -> torch.Tensor:
    """The recipe's per-row fold of ``x*x`` in fp32. ``x`` is [M, N] (any float dtype, read exactly)."""
    n = x.shape[-1]
    L, K = lanes(n)
    idx = _swizzle(n).to(x.device)
    v = x.float()[:, idx]  # (M, L, C)
    acc = torch.zeros(v.shape[0], L, device=x.device, dtype=torch.float32)
    for t in range(v.shape[2]):  # the ordered chain: one rounded fma-free step per leaf
        acc = acc + v[:, :, t] * v[:, :, t]
    s = L >> 1
    while s:
        acc = acc + acc[:, torch.arange(L, device=x.device) ^ s]
        s >>= 1
    return acc[:, 0]


def inv_rms(ssq: torch.Tensor, n: int, eps: float) -> torch.Tensor:
    """``frsqrt_rn(add_rn(div_rn(ssq, N), eps))``: all three correctly rounded fp32 operations.

    Each step runs in float64 and is rounded to fp32 once. Rounding a float64 divide / square root to
    fp32 gives the correctly rounded fp32 result (53 >= 2*24 + 2, so the double rounding is benign),
    while torch's own fp32 divide by a scalar lowers to a reciprocal multiply on CUDA and is 1 ulp
    off on about a quarter of the arguments -- which is what the kernel's ``div.rn`` is not.
    """
    ms = ((ssq.double() / float(n)).float().double() + float(eps)).float()
    return (1.0 / torch.sqrt(ms.double())).float()


def _gamma(w: torch.Tensor | None, zero_centered: bool) -> torch.Tensor | None:
    if w is None:
        return None
    g = w.float()
    return (1.0 + g) if zero_centered else g


def rms_norm(x: torch.Tensor, w: torch.Tensor | None, eps: float, zero_centered: bool) -> torch.Tensor:
    """Recipe v1 RMSNorm. Returns bf16 [M, N]; also the reference for the fused add form's second half."""
    x2 = x.reshape(-1, x.shape[-1])
    r = inv_rms(row_sum_of_squares(x2), x2.shape[-1], eps)
    y = x2.float() * r[:, None]
    g = _gamma(w, zero_centered)
    if g is not None:
        y = y * g
    return y.bfloat16().view(x.shape)


def add_rms_norm(x: torch.Tensor, res: torch.Tensor, w: torch.Tensor | None, eps: float,
                 zero_centered: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """``h = rn_bf16(x + res)`` (the stored residual stream), then recipe v1 on ``h``."""
    h = (x.float() + res.float()).bfloat16()
    return rms_norm(h, w, eps, zero_centered), h


def silu(z: torch.Tensor) -> torch.Tensor:
    """``z / (1 + exp(-z))`` in fp32 (CUDA's libdevice ``expf``; this reference uses torch's)."""
    return z.float() / (1.0 + torch.exp(-z.float()))


def gated_rms_norm(x: torch.Tensor, z: torch.Tensor, w: torch.Tensor | None, eps: float,
                   zero_centered: bool = True) -> torch.Tensor:
    """Recipe v1 gated out-norm: ``[T, hv, N]`` -> ``[T, hv*N]`` bf16."""
    t, hv, n = x.shape
    x2 = x.reshape(-1, n)
    r = inv_rms(row_sum_of_squares(x2), n, eps)
    y = x2.float() * r[:, None]
    g = _gamma(w, zero_centered)
    if g is not None:
        y = y * g
    return (y * silu(z.reshape(-1, n))).bfloat16().view(t, hv * n)


__all__ = ["add_rms_norm", "gated_rms_norm", "inv_rms", "lanes", "rms_norm", "row_sum_of_squares", "silu"]
