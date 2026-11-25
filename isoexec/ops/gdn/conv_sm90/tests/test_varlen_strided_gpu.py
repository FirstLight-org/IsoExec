"""conv_sm90 varlen forward and VJP reading a row-strided x (the q|k|v window of the in_proj GEMM output).

The strided read replaces ``torch.cat([q, k, v])`` before the trainer's conv and the engine's prefill conv.
Claim: output, carried state, dx and dw are byte-identical to the packed form fed with the cat's copy (every
VJP plan), and the source GEMM buffer is never written.
"""

from __future__ import annotations

import pytest
import torch

from isoexec.ops._testing import bitwise_equal
from isoexec.ops.gdn.conv_sm90 import ops

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0), reason="requires H100/SM90"
)

BF = torch.bfloat16


def _case(lanes, D, extra, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    T = sum(lanes)
    gemm = (torch.randn(T, D + extra, generator=g, device="cuda") * 2).to(BF)
    w = (torch.randn(D, 4, generator=g, device="cuda") * 0.5).to(BF)
    cu = torch.tensor([0, *torch.tensor(lanes).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
    return gemm, w, cu


@pytest.mark.parametrize("lanes", [[1], [5, 1, 300], [2900, 2911], [4100, 3, 64, 1200]])
@pytest.mark.parametrize("geom", [(2560, 1560), (1024, 24), (512, 8)])
def test_varlen_forward_and_vjp_strided_match_packed(lanes, geom):
    D, extra = geom
    gemm, w, cu = _case(lanes, D, extra, seed=len(lanes))
    keep = gemm.clone()
    window = gemm[:, :D]
    packed = window.contiguous()
    T = packed.shape[0]
    dy = (torch.randn(T, D, device="cuda") * 0.1).to(BF)
    y0 = ops.varlen(packed, w, cu_seqlens=cu)
    y1 = ops.varlen(window, w, cu_seqlens=cu)
    assert bitwise_equal(y0, y1)
    for plan in ("bwd2", "bwd"):
        orig = ops.bwd_plan
        ops.bwd_plan = (lambda total, channels=2048, p=plan: (p, 16, 128) if p == "bwd2" else (p, 64, 64))
        try:
            grads = []
            for x in (packed, window):
                xl = gemm.detach().clone().requires_grad_(True) if x is window else x.detach().clone().requires_grad_(True)
                src = xl[:, :D] if x is window else xl
                wl = w.detach().clone().requires_grad_(True)
                y = ops.conv(src, wl, cu_seqlens=cu)
                dx, dw = torch.autograd.grad(y, (xl, wl), dy)
                grads.append((y, dx[:, :D] if x is window else dx, dw))
        finally:
            ops.bwd_plan = orig
        (ya, dxa, dwa), (yb, dxb, dwb) = grads
        assert bitwise_equal(ya, yb) and bitwise_equal(dxa, dxb) and bitwise_equal(dwa, dwb), plan
    assert bitwise_equal(gemm, keep)


def test_varlen_with_state_strided_matches_packed():
    lanes, D, extra = [70, 1, 9], 1024, 24
    gemm, w, cu = _case(lanes, D, extra, seed=9)
    g = torch.Generator(device="cuda").manual_seed(1)
    state0 = torch.randn(8, D, 3, generator=g, device="cuda").to(BF)
    idx = torch.tensor([3, 0, 5], dtype=torch.int32, device="cuda")
    has = torch.tensor([True, False, False], device="cuda")
    outs = []
    for x in (gemm[:, :D].contiguous(), gemm[:, :D]):
        st = state0.clone()
        outs.append((ops.varlen(x, w, cu_seqlens=cu, conv_state=st, cache_indices=idx, has_initial_state=has), st))
    assert bitwise_equal(outs[0][0], outs[1][0]) and bitwise_equal(outs[0][1], outs[1][1])
