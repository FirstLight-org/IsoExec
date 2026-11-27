"""Shared inputs and comparisons of the glm_rec_sm100 battery."""

from __future__ import annotations

import pytest
import torch


def require_sm100():
    from isoexec.core.arch import live_device_arch

    if not torch.cuda.is_available() or live_device_arch() != "sm100":
        pytest.skip("requires an SM100 device")


def cu_of(lens, device="cuda"):
    out = [0]
    for n in lens:
        out.append(out[-1] + n)
    return torch.tensor(out, dtype=torch.int32, device=device)


def make_inputs(lens, H=4, *, seed=0, gate_lo=-5.0, beta_dtype=torch.bfloat16, state=True, state_scale=0.1, device="cuda"):
    """Packed q, k, v (bf16), fp32 log decay (GLM's bounded range), post-sigmoid beta, cu_seqlens, v-first state [N,H,V,K]."""
    gen = torch.Generator(device=device).manual_seed(seed)
    T = sum(lens)
    rnd = lambda *s: torch.randn(*s, device=device, generator=gen)
    q, k, v = (rnd(T, H, 128).bfloat16() for _ in range(3))
    g = gate_lo * torch.sigmoid(rnd(T, H, 128))
    b = torch.sigmoid(rnd(T, H)).to(beta_dtype)
    h0 = rnd(len(lens), H, 128, 128) * state_scale if state else None
    return q, k, v, g, b, cu_of(lens, device), h0


def rel_l2(a, b):
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def max_rel(a, b):
    a, b = a.double(), b.double()
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-30)).item()


def same_bits(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)) \
        if a.numel() else True


def bits_equal(a, b, what=""):
    assert a.shape == b.shape, f"{what}: shape {a.shape} vs {b.shape}"
    ua = a.contiguous().view(torch.uint8) if a.dtype != torch.uint8 else a
    ub = b.contiguous().view(torch.uint8) if b.dtype != torch.uint8 else b
    assert torch.equal(ua, ub), f"{what}: not bitwise equal ({(ua != ub).sum().item()} differing bytes)"
