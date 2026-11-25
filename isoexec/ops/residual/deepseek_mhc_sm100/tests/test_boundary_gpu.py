"""GPU battery for the fused inference mHC boundary (``boundary_rows``) and the side-stream coefficients: bitwise
equal to the kernel sequences they replace, at every batch size."""

from __future__ import annotations

import pytest
import torch

from isoexec.core.arch import live_device_arch
from isoexec.ops.residual.deepseek_mhc_sm100 import ops
from isoexec.ops.residual.deepseek_mhc_sm100.tests._data import EPS, ITERS, RMS_EPS, H, make, probs, rnd

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or live_device_arch() != "sm100",
                                reason="the sm100 mHC kernels run on a Blackwell device only")
TS = [1, 3, 8, 16, 17, 64, 257, 1100]


def _bits(t):
    return t.contiguous().view(torch.uint8)


def _eq(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(_bits(a), _bits(b))


def _case(T, seed):
    x, w, scale, base = make(T, seed)
    a = rnd(T, H, seed=seed + 7, sc=3.0).bfloat16()
    post, comb = probs(T, seed + 11)
    pre = torch.sigmoid(rnd(T, 4, seed=seed + 13)) + EPS
    nw = (1 + rnd(H, seed=seed + 17, sc=0.2)).bfloat16()
    if T >= 3:
        x[1] = 0
        a[1] = 0
        x[2] *= 300
    return x, w, scale, base, a, post, comb, pre, nw


@pytest.mark.parametrize("T", TS)
@pytest.mark.parametrize("merge", [True, False])
@pytest.mark.parametrize("encode", [True, False])
def test_boundary_rows_bitwise_vs_expand_collapse_rms_norm(T, merge, encode):
    from isoexec.ops.attention.deepseek_v41_cache_sm100 import paged
    x, _, _, _, a, post, comb, pre, nw = _case(T, 0)
    with torch.inference_mode():
        rx = ops.expand(a, x, post, comb) if merge else x
        rh = paged.collapse_rms_norm(rx, pre, nw, RMS_EPS, encode=encode)
        renc = paged.encoding_of(rh) if encode else None
        for cpt in (0, 1, 2, 5):
            nx, nh = ops.boundary_rows(x, pre, nw, RMS_EPS, out=a if merge else None, post=post if merge else None,
                                       comb=comb if merge else None, encode=encode, cpt=cpt)
            assert _eq(nx, rx) and _eq(nh, rh), cpt
            if encode:
                enc = paged.encoding_of(nh)
                assert enc is not None and _eq(enc[0], renc[0]) and _eq(enc[1], renc[1]), cpt


def test_params_async_join_bitwise():
    x, w, scale, base = make(8, 2)
    with torch.inference_mode():
        ref = ops.mhc_params(x, w, scale, base, rms_eps=RMS_EPS, sinkhorn_eps=EPS, iterations=ITERS)
        *got, ev = ops.mhc_params_async(x, w, scale, base, rms_eps=RMS_EPS, sinkhorn_eps=EPS, iterations=ITERS,
                                        overlap=True)
        assert ev is not None
        ops.join(ev, x.device)
        assert all(_eq(u, v) for u, v in zip(got, ref))


@pytest.mark.parametrize("T", [1, 255, 256, 777, 2048])
def test_split_parts_and_tail_overlap_bitwise(T):
    """split_proj + split_tail (separately launchable) and the tail-only side-stream path equal split_forward (the
    projection kernel + tail kernel in one call) bit for bit."""
    from isoexec.ops.residual.mhc_sm100 import ops as G
    from isoexec.ops.residual.mhc_sm100._extension import extension
    x, w, scale, base = make(T, 3)
    with torch.inference_mode():
        ref = G.split_forward(x, w, scale, base, rms_eps=RMS_EPS, eps=EPS, iters=ITERS, collapse=None)[:3]
        for groups in (0, 1, 3):
            ws = extension().split_proj(x, w, groups)
            parts = extension().split_tail(ws, scale, base, RMS_EPS, EPS, ITERS)
            assert all(_eq(u.reshape(v.shape), v) for u, v in zip(parts, ref)), groups
        saved = ops.OVERLAP_MAX_ROWS, ops.OVERLAP_TAIL_MAX_ROWS
        try:
            ops.OVERLAP_MAX_ROWS, ops.OVERLAP_TAIL_MAX_ROWS = 0, 1 << 20
            *got, ev = ops.mhc_params_async(x, w, scale, base, rms_eps=RMS_EPS, sinkhorn_eps=EPS, iterations=ITERS)
            assert ev is not None
            ops.join(ev, x.device)
            assert all(_eq(u.reshape(v.shape), v) for u, v in zip(got, ref))
        finally:
            ops.OVERLAP_MAX_ROWS, ops.OVERLAP_TAIL_MAX_ROWS = saved
