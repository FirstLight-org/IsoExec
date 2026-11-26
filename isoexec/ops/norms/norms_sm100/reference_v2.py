"""Executable statement of the norms_sm100 **forward recipe v2**, independent of the CUDA kernels."""

from __future__ import annotations

import torch

from isoexec.ops.norms.norms_sm90.reference import _gamma, inv_rms, silu

LMAX: dict[int, int] = dict.fromkeys((128, 384, 512, 768, 1536, 2048, 4096, 5120, 7168), 128)

DEFAULT_V2: frozenset[int] = frozenset({1536, 4096, 5120, 7168})


def lanes(n: int, lmax: int | None = None) -> tuple[int, int]:
    """``(L, K)`` for width ``n`` under recipe v2 -- ``csrc/plan.h::fwd_lanes_v2``."""
    if n <= 0 or n % 8 or not (n % 256 == 0 or n in (128, 384)):
        raise ValueError(f"norms_sm100 recipe v2 has no fold for width {n}")
    cap = LMAX.get(n, 32) if lmax is None else lmax
    leaves, ell = n // 8, 1
    while ell * 2 <= cap and leaves % (ell * 2) == 0:
        ell *= 2
    return ell, leaves // ell


def _swizzle(n: int, lmax: int | None = None) -> torch.Tensor:
    """Column order as (lane, chain position): ``idx[l, t]`` is the column of lane ``l``'s t-th leaf."""
    ell, k = lanes(n, lmax)
    kk = torch.arange(k).view(k, 1, 1)
    ll = torch.arange(ell).view(1, ell, 1)
    jj = torch.arange(8).view(1, 1, 8)
    cols = kk * 8 * ell + 8 * ll + jj  # (K, L, 8)
    return cols.permute(1, 0, 2).reshape(ell, k * 8)  # (L, C)


def row_sum_of_squares(x: torch.Tensor, lmax: int | None = None) -> torch.Tensor:
    """The recipe's per-row fold of ``x*x`` in fp32. ``x`` is [M, N] (any float dtype, read exactly)."""
    n = x.shape[-1]
    ell, _ = lanes(n, lmax)
    idx = _swizzle(n, lmax).to(x.device)
    v = x.float()[:, idx]  # (M, L, C)
    acc = torch.zeros(v.shape[0], ell, device=x.device, dtype=torch.float32)
    for t in range(v.shape[2]):  # the ordered chain: one rounded fma-free step per leaf
        acc = acc + v[:, :, t] * v[:, :, t]
    s = ell >> 1
    while s:
        acc = acc + acc[:, torch.arange(ell, device=x.device) ^ s]
        s >>= 1
    return acc[:, 0]


def rms_norm(x: torch.Tensor, w: torch.Tensor | None, eps: float, zero_centered: bool,
             lmax: int | None = None) -> torch.Tensor:
    """Recipe v2 RMSNorm. Returns bf16 [..., N]."""
    x2 = x.reshape(-1, x.shape[-1])
    r = inv_rms(row_sum_of_squares(x2, lmax), x2.shape[-1], eps)
    y = x2.float() * r[:, None]
    g = _gamma(w, zero_centered)
    if g is not None:
        y = y * g
    return y.bfloat16().view(x.shape)


def add_rms_norm(x: torch.Tensor, res: torch.Tensor, w: torch.Tensor | None, eps: float,
                 zero_centered: bool, lmax: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """``h = rn_bf16(x + res)`` (the stored residual stream), then recipe v2 on ``h``."""
    h = (x.float() + res.float()).bfloat16()
    return rms_norm(h, w, eps, zero_centered, lmax), h


__all__ = ["DEFAULT_V2", "LMAX", "add_rms_norm", "inv_rms", "lanes", "rms_norm",
           "row_sum_of_squares", "silu"]
