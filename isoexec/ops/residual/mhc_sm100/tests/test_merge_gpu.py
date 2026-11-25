"""GPU battery for the mHC merge (forward and backward)."""

from __future__ import annotations

import pytest
import torch

from isoexec.core.arch import live_device_arch
from isoexec.ops.residual.mhc_sm100 import ops
from isoexec.ops.residual.mhc_sm100 import reference as R
from isoexec.ops.residual.mhc_sm100.tests._data import rand_like

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or live_device_arch() != "sm100",
                                reason="the sm100 mHC kernels run on a Blackwell device only")


def inputs(T, H=4096, seed=0):
    x = rand_like(T * 4, H, seed, torch.bfloat16, 2.0).view(T, 4, H)
    out = rand_like(T, H, seed + 1, torch.bfloat16, 1.0)
    post = torch.sigmoid(rand_like(T, 4, seed + 2)) * 2
    comb = torch.softmax(rand_like(T, 16, seed + 3), -1).view(T, 4, 4)
    return x, out, post, comb


@pytest.mark.parametrize("H", [4096, 5120, 8])
@pytest.mark.parametrize("T", [0, 1, 7, 256, 1000])
def test_forward_bitwise_declared_program(T, H):
    x, out, post, comb = inputs(T, H)
    assert torch.equal(ops.merge_forward(x, out, post, comb), R.merge_declared(x, out, post, comb))


@pytest.mark.parametrize("T", [1, 64, 1000])
def test_forward_accuracy_vs_float64_and_eager(T):
    x, out, post, comb = inputs(T)
    y = ops.merge_forward(x, out, post, comb)
    exact = R.merge_reference(x, out, post, comb)
    mag = (post.double().unsqueeze(-1) * out.double().unsqueeze(-2)).abs() + (
        comb.double().transpose(-1, -2).abs() @ x.double().abs())
    # casts of post/comb to bf16 (2^-9 each) + three bf16 roundings
    assert ((y.double() - exact).abs() <= mag * 2 ** -6).all()
    eager = R.eager_merge(x, out, post, comb)
    # same cast boundaries; the only freedom is the 4-term accumulation order -> within 1 bf16 ulp of the magnitude
    assert ((y.double() - eager.double()).abs() <= mag * 2 ** -7).all()


def test_forward_repeat_and_batch_invariance():
    x, out, post, comb = inputs(1024)
    a = ops.merge_forward(x, out, post, comb)
    assert torch.equal(a, ops.merge_forward(x, out, post, comb))
    for i in (0, 333, 1023):
        assert torch.equal(ops.merge_forward(x[i:i + 1], out[i:i + 1], post[i:i + 1], comb[i:i + 1]), a[i:i + 1])
    nc = torch.zeros(1024, 4, 4096 + 16, dtype=torch.bfloat16, device="cuda")
    nc[..., :4096] = x
    assert torch.equal(ops.merge_forward(nc[..., :4096], out, post, comb), a)


@pytest.mark.parametrize("H", [4096, 5120])
@pytest.mark.parametrize("T", [1, 7, 130, 1000])
def test_backward_bitwise_declared_program(T, H):
    x, out, post, comb = inputs(T, H)
    g = rand_like(T * 4, H, 11, torch.bfloat16, 0.5).view(T, 4, H)
    got = ops.merge_backward(g, x, out, post, comb)
    want = R.merge_backward_declared(g, x, out, post, comb)
    for a, b in zip(got, want):
        assert a.dtype == b.dtype and torch.equal(a, b)


@pytest.mark.parametrize("T", [1, 64, 600])
def test_backward_accuracy_vs_float64_autograd(T):
    x, out, post, comb = inputs(T)
    g = rand_like(T * 4, 4096, 11, torch.bfloat16, 0.5).view(T, 4, 4096)
    xs, os_, ps, cs = (t.double().requires_grad_(True) for t in (x, out, post, comb))
    (R.merge_reference(xs, os_, ps, cs) * g.double()).sum().backward()
    dx, do, dp, dc = ops.merge_backward(g, x, out, post, comb)
    rel = lambda a, b: ((a.double() - b).abs().max() / b.abs().max()).item()
    assert rel(dx, xs.grad) < 2 ** -6 and rel(do, os_.grad) < 2 ** -6
    assert rel(dp, ps.grad) < 2 ** -6 and rel(dc, cs.grad) < 2 ** -6


def test_backward_repeat_and_row_invariance():
    x, out, post, comb = inputs(800)
    g = rand_like(800 * 4, 4096, 11, torch.bfloat16, 0.5).view(800, 4, 4096)
    a = ops.merge_backward(g, x, out, post, comb)
    b = ops.merge_backward(g, x, out, post, comb)
    assert all(torch.equal(u, v) for u, v in zip(a, b))
    for i in (0, 411, 799):
        one = ops.merge_backward(g[i:i + 1].contiguous(), x[i:i + 1].contiguous(), out[i:i + 1].contiguous(),
                                 post[i:i + 1].contiguous(), comb[i:i + 1].contiguous())
        assert all(torch.equal(u, v[i:i + 1]) for u, v in zip(one, a))


def test_cast_boundaries_are_witnessed():
    # GLM: post/comb are cast to bf16 and the two terms are rounded before the add. An all-fp32 merge (one rounding) differs
    # on some elements, and the native result is bitwise the declared (cast) program, not the all-fp32 one.
    x, out, post, comb = inputs(256)
    cast = ops.merge_forward(x, out, post, comb, cast=True)
    nocast = ops.merge_forward(x, out, post, comb, cast=False)
    assert torch.equal(cast, R.merge_declared(x, out, post, comb))
    assert not torch.equal(cast, nocast)
    assert (cast != nocast).float().mean().item() > 0.01
    # the fp32 variant is bitwise the all-fp32 program (post/comb fp32, one rounding)
    xf, of = x.float(), out.float()
    acc = comb[:, 0, :].unsqueeze(-1) * xf[:, 0].unsqueeze(-2)
    for s in range(1, 4):
        acc = acc + comb[:, s, :].unsqueeze(-1) * xf[:, s].unsqueeze(-2)
    assert torch.equal(nocast, ((post.unsqueeze(-1) * of.unsqueeze(-2)) + acc).to(torch.bfloat16))
