"""Shared inputs, comparisons and skips of the KDA batteries (glm_sm100, glm_rec_sm100, slot_reset_sm100)."""

from __future__ import annotations

import shutil

import pytest
import torch


def require_sm100():
    from isoexec.core.arch import live_device_arch

    if not torch.cuda.is_available() or live_device_arch() != "sm100":
        pytest.skip("requires an SM100 device")


def require_cuda_toolchain():
    """The SASS gates build the extension (no GPU needed) and read it with cuobjdump: skip without the toolchain."""
    from torch.utils.cpp_extension import CUDA_HOME

    if CUDA_HOME is None or shutil.which("cuobjdump") is None or shutil.which("nvcc") is None:
        pytest.skip("needs the CUDA toolchain (nvcc and cuobjdump on PATH, CUDA_HOME) to build and read the SASS")


def assert_imports_build_nothing(statements: str, extensions=("glm_sm100", "glm_rec_sm100")):
    """Run ``statements`` (imports) in a fresh interpreter and require that no KDA extension was built or loaded: each
    ``extension()`` is an ``lru_cache`` that is filled on the first call (the extension module never enters
    ``sys.modules``, so that is not a test of it)."""
    import subprocess
    import sys

    code = statements + "".join(f";from isoexec.ops.kda.{p} import _extension as _e{i}"
                                f";assert _e{i}.extension.cache_info().currsize == 0, '{p} was built'"
                                for i, p in enumerate(extensions)) + ";print('clean')"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0 and "clean" in out.stdout, (out.stdout[-2000:], out.stderr[-2000:])


def cu_of(lens, device="cuda"):
    out = [0]
    for n in lens:
        out.append(out[-1] + n)
    return torch.tensor(out, dtype=torch.int32, device=device)


def make_inputs(lens, H=4, *, seed=0, gate_lo=-5.0, g_dtype=torch.float32, beta_dtype=torch.bfloat16,
                state=True, state_scale=0.1, device="cuda"):
    """Packed q, k, v (bf16), log-decay (GLM's bounded gate range by default), post-sigmoid beta, cu_seqlens and an
    initial state ``[N, H, 128, 128]`` (the token reducer reads it v-first)."""
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


def same_bits(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)) \
        if a.numel() else True


def bits_equal(a, b, what=""):
    assert a.shape == b.shape, f"{what}: shape {a.shape} vs {b.shape}"
    ua = a.contiguous().view(torch.uint8) if a.dtype != torch.uint8 else a
    ub = b.contiguous().view(torch.uint8) if b.dtype != torch.uint8 else b
    assert torch.equal(ua, ub), f"{what}: not bitwise equal ({(ua != ub).sum().item()} differing bytes)"
