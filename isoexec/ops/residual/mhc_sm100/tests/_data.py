"""Deterministic test data shared by the mHC GPU tests."""

from __future__ import annotations

import torch

RMS_EPS, HC_EPS, ITERS = 1e-5, 1e-6, 20


def make(T, H=4096, seed=0, fn_dtype=torch.bfloat16, x_scale=2.0, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    r = lambda *s, sc=1.0: torch.randn(*s, device=device, generator=g) * sc
    x = r(T, 4, H, sc=x_scale).bfloat16()
    fn = r(24, 4 * H, sc=0.03).to(fn_dtype)
    base = r(24, sc=0.5)
    scale = torch.tensor([0.7, 0.4, 0.9], device=device)
    return x, fn, base, scale


def rand_like(T, H, seed, dtype=torch.float32, sc=1.0, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    return (torch.randn(T, H, device=device, generator=g) * sc).to(dtype)
