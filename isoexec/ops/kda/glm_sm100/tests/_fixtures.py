"""Shared inputs and comparisons of the glm_sm100 battery."""

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


def make_inputs(lens, H=4, *, seed=0, gate_lo=-5.0, g_dtype=torch.float32, beta_dtype=torch.bfloat16,
                state=True, state_scale=0.1, device="cuda"):
    """Packed q, k, v, log-decay (GLM's bounded gate range by default), beta, cu_seqlens, initial state."""
    gen = torch.Generator(device=device).manual_seed(seed)
    T = sum(lens)
    rnd = lambda *s: torch.randn(*s, device=device, generator=gen)
    q, k, v = (rnd(T, H, 128).bfloat16() for _ in range(3))
    g = (gate_lo * torch.sigmoid(rnd(T, H, 128))).to(g_dtype)
    b = torch.sigmoid(rnd(T, H)).to(beta_dtype)
    h0 = rnd(len(lens), H, 128, 128) * state_scale if state else None
    return q, k, v, g, b, cu_of(lens, device), h0


def rel_l2(a, b, scale=None):
    a, b = a.double(), b.double()
    den = b.norm() if scale is None else torch.maximum(b.norm(), scale.double().norm())
    return ((a - b).norm() / den.clamp_min(1e-30)).item()


def max_rel(a, b, scale=None):
    a, b = a.double(), b.double()
    den = b.abs().max() if scale is None else torch.maximum(b.abs().max(), scale.double().abs().max())
    return ((a - b).abs().max() / den.clamp_min(1e-30)).item()


def assert_close(a, b, l2, mx, what="", scale=None):
    """rel-L2 and max-rel of a against b. `scale` (a tensor of the same physical scale) floors the
    denominators, for quantities whose exact value is zero or near zero by cancellation."""
    e2, em = rel_l2(a, b, scale), max_rel(a, b, scale)
    assert e2 <= l2 and em <= mx, f"{what}: rel-L2 {e2:.3e} (<= {l2}), max-rel {em:.3e} (<= {mx})"
