"""GPU battery for the mHC split (forward and backward)."""

from __future__ import annotations

import pytest
import torch

from isoexec.core.arch import live_device_arch
from isoexec.ops.residual.mhc_sm100 import ops
from isoexec.ops.residual.mhc_sm100 import reference as R
from isoexec.ops.residual.mhc_sm100.tests._data import HC_EPS, ITERS, RMS_EPS, make, rand_like

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or live_device_arch() != "sm100",
                                reason="the sm100 mHC kernels run on a Blackwell device only")
KW = dict(rms_eps=RMS_EPS, eps=HC_EPS, iters=ITERS)
TS = [1, 5, 16, 33, 100, 256, 513, 1024]


def fwd(x, fn, base, scale, **kw):
    return ops.split_forward(x, fn, scale, base, **{**KW, **kw})


def oracle(x, fn, base, scale):
    return R.split_reference(x, fn, scale, base, rms_eps=RMS_EPS, hc_eps=HC_EPS, iters=ITERS)


@pytest.mark.parametrize("T", TS)
@pytest.mark.parametrize("fn_dtype", [torch.bfloat16, torch.float32])
def test_forward_accuracy_vs_float64_oracle(T, fn_dtype):
    x, fn, base, scale = make(T, fn_dtype=fn_dtype)
    pre, post, comb, col, (proj, rstd) = fwd(x, fn, base, scale, stats=True)
    rp, rpost, rcomb, rcol, rproj, rrstd = oracle(x, fn, base, scale)
    d = lambda a, b: (a.double() - b).abs().max().item()
    assert d(pre, rp) < 3e-6 and d(post, rpost) < 3e-6 and d(comb, rcomb) < 3e-6   # fp32 evaluation of a 16384-term projection (<= ~25 ulp at 2.0)
    assert d(rstd, rrstd) < 1e-6 * rrstd.max().item()
    # raw projection: fp32 accumulation of exact products; bound by 2^-23 * sum |x||fn| scale
    bound = 2e-6 * (x.double().abs().flatten(-2) @ fn.double().abs().t()).max().item()
    assert d(proj, rproj) < max(bound, 1e-5)
    # collapse: fp32 pre (abs error ~1 ulp), fp32 chain over four products, then one bf16 rounding (half ulp <= 2^-8 relative)
    atol = 4e-7 * x.double().abs().sum(-2)
    assert ((col.double() - rcol).abs() <= rcol.abs() * 2 ** -8 + atol).all()


@pytest.mark.parametrize("fn_dtype", [torch.bfloat16, torch.float32])
def test_forward_repeat_bitwise(fn_dtype):
    x, fn, base, scale = make(700, fn_dtype=fn_dtype)
    a = fwd(x, fn, base, scale, stats=True)
    for _ in range(4):
        b = fwd(x, fn, base, scale, stats=True)
        for u, v in zip(a[:4] + a[4], b[:4] + b[4]):
            assert torch.equal(u, v)


@pytest.mark.parametrize("fn_dtype", [torch.bfloat16, torch.float32])
def test_forward_batch_invariance_and_tile_independence(fn_dtype):
    x, fn, base, scale = make(1024, fn_dtype=fn_dtype)
    full = fwd(x, fn, base, scale)
    # rows alone, sub-batches, shuffled order, different tile shapes (mt) -> identical bits
    for i in (0, 1, 15, 16, 511, 1023):
        one = fwd(x[i:i + 1].contiguous(), fn, base, scale)
        for u, v in zip(full, one):
            assert torch.equal(u[i:i + 1], v)
    for lo, hi in ((0, 37), (300, 700), (990, 1024)):
        sub = fwd(x[lo:hi].contiguous(), fn, base, scale)
        for u, v in zip(full, sub):
            assert torch.equal(u[lo:hi], v)
    perm = torch.randperm(1024, device="cuda", generator=torch.Generator(device="cuda").manual_seed(1))
    sh = fwd(x[perm].contiguous(), fn, base, scale)
    for u, v in zip(full, sh):
        assert torch.equal(u[perm], v)
    for mt in (1, 2):
        alt = fwd(x, fn, base, scale, mt=mt)
        for u, v in zip(full, alt):
            assert torch.equal(u, v)


def test_planted_duplicate_rows_get_identical_outputs():
    x, fn, base, scale = make(2048)
    x[1000] = x[3]; x[2047] = x[3]; x[33] = x[3]
    pre, post, comb, col = fwd(x, fn, base, scale)
    for i in (1000, 2047, 33):
        for u in (pre, post, comb, col):
            assert torch.equal(u[i], u[3])


def test_leading_shape_and_noncontiguous():
    x, fn, base, scale = make(384)
    full = fwd(x, fn, base, scale)
    shaped = fwd(x.view(3, 128, 4, 4096), fn, base, scale)
    for u, v in zip(full, shaped):
        assert torch.equal(u.view(v.shape), v)
    big = torch.zeros(384, 4, 4096 + 64, dtype=torch.bfloat16, device="cuda")
    big[..., :4096] = x
    nc = fwd(big[..., :4096], fn, base, scale)
    for u, v in zip(full, nc):
        assert torch.equal(u, v)


def test_cuda_graph_replay():
    x, fn, base, scale = make(512)
    ref = fwd(x, fn, base, scale)
    fwd(x, fn, base, scale)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fwd(x, fn, base, scale)
    g.replay(); torch.cuda.synchronize()
    for u, v in zip(ref, out):
        assert torch.equal(u, v)


def test_zero_rows_and_refusals():
    x, fn, base, scale = make(1)
    out = fwd(x[:0], fn, base, scale)
    assert out[0].shape == (0, 4) and out[3].shape == (0, 4096)
    with pytest.raises(ValueError):
        fwd(x[..., :2048].contiguous(), fn, base, scale)
    with pytest.raises(ValueError):
        fwd(x.float(), fn, base, scale)
    with pytest.raises(ValueError):
        fwd(x, fn[:, :100], base, scale)


def test_extremes_finite_and_nonfinite_propagation():
    x, fn, base, scale = make(64, x_scale=300.0)       # large residual magnitude (rms-normalised)
    pre, post, comb, col, _ = fwd(x, fn, base, scale, stats=True)
    assert torch.isfinite(comb).all() and torch.isfinite(post).all()
    rp, rpost, rcomb, rcol, *_ = oracle(x, fn, base, scale)
    assert (comb.double() - rcomb).abs().max() < 3e-6
    x[3, 1, 7] = float("inf"); x[4, 0, 0] = float("nan")
    _, _, c2, _ = fwd(x, fn, base, scale)
    assert torch.isnan(c2[3]).any() or torch.isnan(c2[4]).any()
    assert torch.equal(c2[5:], fwd(make(64, x_scale=300.0)[0], fn, base, scale)[2][5:])  # other rows unaffected


def test_sinkhorn_iteration_counts_and_eps():
    x, fn, base, scale = make(48)
    for iters in (1, 2, 7, 20, 32):
        _, _, comb, _ = ops.split_forward(x, fn, scale, base, rms_eps=1e-5, eps=3e-6, iters=iters)
        rc = R.split_reference(x, fn, scale, base, rms_eps=1e-5, hc_eps=3e-6, iters=iters)[2]
        assert (comb.double() - rc).abs().max() < 3e-6


# ------------------------------------------------------------------ backward
def grads_oracle(x, fn, base, scale, dpost, dcomb, dcol, dpre=None):
    xs = x.double().requires_grad_(True)
    fns = fn.double().requires_grad_(True)
    bs = base.double().requires_grad_(True)
    ss = scale.double().requires_grad_(True)
    pre, post, comb, col, *_ = R.split_reference(xs, fns, ss, bs, rms_eps=RMS_EPS, hc_eps=HC_EPS, iters=ITERS)
    loss = (post * dpost.double()).sum() + (comb * dcomb.double()).sum() + (col * dcol.double()).sum()
    if dpre is not None:
        loss = loss + (pre * dpre.double()).sum()
    loss.backward()
    return xs.grad, fns.grad, bs.grad, ss.grad


def bwd_inputs(T, fn_dtype=torch.bfloat16, seed=0, H=4096):
    x, fn, base, scale = make(T, fn_dtype=fn_dtype, seed=seed, H=H)
    pre, post, comb, col, (proj, rstd) = fwd(x, fn, base, scale, stats=True)
    dpost = rand_like(T, 4, seed + 1, sc=0.5)
    dcomb = rand_like(T, 16, seed + 2, sc=0.5).view(T, 4, 4)
    dcol = rand_like(T, H, seed + 3, dtype=torch.bfloat16, sc=0.5)
    return x, fn, base, scale, proj, rstd, dpost, dcomb, dcol


@pytest.mark.parametrize("T", [1, 16, 33, 100, 512, 1000, 1537])
def test_backward_accuracy_vs_float64_autograd(T):
    x, fn, base, scale, proj, rstd, dpost, dcomb, dcol = bwd_inputs(T)
    dx, dfn, dscale, dbase, _ = ops.split_backward(x, fn, scale, base, proj, rstd, dpost, dcomb, eps=HC_EPS,
                                                   iters=ITERS, dcol=dcol)
    gx, gfn, gb, gs = grads_oracle(x, fn, base, scale, dpost, dcomb, dcol)
    rel = lambda a, b: ((a.double() - b).abs().max() / b.abs().max().clamp_min(1e-30)).item()
    assert dx.dtype == torch.bfloat16 and dfn.dtype == torch.bfloat16
    assert rel(dx, gx) < 1.0 * 2 ** -8       # bf16 output rounding
    assert rel(dfn, gfn) < 2.0 * 2 ** -8
    assert rel(dbase, gb) < 1e-5 and rel(dscale, gs) < 1e-5


def test_backward_fp32_fn_dfn_dtype_and_accuracy():
    x, fn, base, scale, proj, rstd, dpost, dcomb, dcol = bwd_inputs(300, fn_dtype=torch.float32)
    dx, dfn, dscale, dbase, _ = ops.split_backward(x, fn, scale, base, proj, rstd, dpost, dcomb, eps=HC_EPS,
                                                   iters=ITERS, dcol=dcol)
    gx, gfn, gb, gs = grads_oracle(x, fn, base, scale, dpost, dcomb, dcol)
    rel = lambda a, b: ((a.double() - b).abs().max() / b.abs().max().clamp_min(1e-30)).item()
    assert dfn.dtype == torch.float32 and rel(dfn, gfn) < 1e-5 and rel(dx, gx) < 2 ** -8


def test_backward_repeat_bitwise_and_row_invariance():
    T = 1100
    x, fn, base, scale, proj, rstd, dpost, dcomb, dcol = bwd_inputs(T)
    args = dict(eps=HC_EPS, iters=ITERS, dcol=dcol)
    a = ops.split_backward(x, fn, scale, base, proj, rstd, dpost, dcomb, **args)
    for _ in range(3):
        b = ops.split_backward(x, fn, scale, base, proj, rstd, dpost, dcomb, **args)
        assert all(torch.equal(u, v) for u, v in zip(a[:4], b[:4]))
    # dx of a row is independent of the other rows / batch size
    for lo, hi in ((0, 1), (500, 531), (1090, 1100)):
        sub = ops.split_backward(x[lo:hi].contiguous(), fn, scale, base, proj[lo:hi].contiguous(), rstd[lo:hi].contiguous(),
                                 dpost[lo:hi].contiguous(), dcomb[lo:hi].contiguous(), eps=HC_EPS, iters=ITERS,
                                 dcol=dcol[lo:hi].contiguous())
        assert torch.equal(sub[0], a[0][lo:hi])


def test_backward_cuda_graph_and_zero_rows():
    x, fn, base, scale, proj, rstd, dpost, dcomb, dcol = bwd_inputs(640)
    f = lambda: ops.split_backward(x, fn, scale, base, proj, rstd, dpost, dcomb, eps=HC_EPS, iters=ITERS, dcol=dcol)
    ref = f(); f()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = f()
    g.replay(); torch.cuda.synchronize()
    assert all(torch.equal(u, v) for u, v in zip(ref[:4], out[:4]))
    z = ops.split_backward(x[:0], fn, scale, base, proj[:0], rstd[:0], dpost[:0], dcomb[:0], eps=HC_EPS, iters=ITERS,
                           dcol=dcol[:0])
    assert z[0].shape == (0, 4, 4096) and not z[1].any() and not z[2].any() and not z[3].any()


def test_backward_padding_rows_do_not_change_parameter_gradients():
    # appending zero-gradient rows (dpost = dcomb = dcol = 0) must leave dfn/dscale/dbase unchanged up to the
    # segment structure; with the same 512-row segmentation the bits are equal when the extra rows land in new tiles.
    x, fn, base, scale, proj, rstd, dpost, dcomb, dcol = bwd_inputs(512)
    a = ops.split_backward(x, fn, scale, base, proj, rstd, dpost, dcomb, eps=HC_EPS, iters=ITERS, dcol=dcol)
    x2, _, _, _, proj2, rstd2, *_ = bwd_inputs(512 + 64, seed=0)
    pad = lambda t, n: torch.cat([t, torch.zeros(64, *t.shape[1:], dtype=t.dtype, device=t.device)])
    b = ops.split_backward(pad(x, 64) if False else x2.clone(), fn, scale, base,
                           torch.cat([proj, proj2[512:]]), torch.cat([rstd, rstd2[512:]]),
                           pad(dpost, 64), pad(dcomb, 64), eps=HC_EPS, iters=ITERS, dcol=pad(dcol, 64))
    # segment 0 is rows 0..511 in both calls; the appended zero-gradient rows form segment 1 and contribute exact zeros
    assert torch.equal(a[2], b[2]) and torch.equal(a[3], b[3])
    assert torch.equal(a[1], b[1])


def test_forward_independent_of_cta_schedule_groups_and_tile_shape():
    x, fn, base, scale = make(1500)
    ref = fwd(x, fn, base, scale, stats=True)
    for groups in (1, 2, 3, 7, 11):
        for mt in (1, 2):
            got = fwd(x, fn, base, scale, stats=True, groups=groups, mt=mt)
            assert all(torch.equal(u, v) for u, v in zip(ref[:4] + ref[4], got[:4] + got[4])), (groups, mt)


def test_backward_independent_of_cta_schedule_groups():
    x, fn, base, scale, proj, rstd, dpost, dcomb, dcol = bwd_inputs(1700)    # 4 segments of 512 rows
    ref = ops.split_backward(x, fn, scale, base, proj, rstd, dpost, dcomb, eps=HC_EPS, iters=ITERS, dcol=dcol)
    for groups in (1, 2, 3, 4):
        got = ops.split_backward(x, fn, scale, base, proj, rstd, dpost, dcomb, eps=HC_EPS, iters=ITERS, dcol=dcol,
                                 groups=groups)
        assert all(torch.equal(u, v) for u, v in zip(ref[:4], got[:4])), groups


@pytest.mark.parametrize("iters", [1, 2, 7, 32])
@pytest.mark.parametrize("rms_eps,hc_eps", [(1e-5, 1e-6), (1e-3, 4e-6)])
def test_backward_iteration_counts_and_separate_epsilons_vs_float64(iters, rms_eps, hc_eps):
    T = 70
    x, fn, base, scale = make(T)
    _, _, _, _, (proj, rstd) = ops.split_forward(x, fn, scale, base, rms_eps=rms_eps, eps=hc_eps, iters=iters, stats=True)
    dpost, dcomb = rand_like(T, 4, 1, sc=0.5), rand_like(T, 16, 2, sc=0.5).view(T, 4, 4)
    dcol = rand_like(T, 4096, 3, dtype=torch.bfloat16, sc=0.5)
    dx, dfn, dscale, dbase, _ = ops.split_backward(x, fn, scale, base, proj, rstd, dpost, dcomb, eps=hc_eps, iters=iters, dcol=dcol)
    xs, fns, bs, ss = (t.double().requires_grad_(True) for t in (x, fn, base, scale))
    pre, post, comb, col, *_ = R.split_reference(xs, fns, ss, bs, rms_eps=rms_eps, hc_eps=hc_eps, iters=iters)
    ((post * dpost.double()).sum() + (comb * dcomb.double()).sum() + (col * dcol.double()).sum()).backward()
    rel = lambda a, b: ((a.double() - b).abs().max() / b.abs().max().clamp_min(1e-30)).item()
    assert rel(dx, xs.grad) < 2 ** -8 and rel(dfn, fns.grad) < 2 ** -7
    assert rel(dbase, bs.grad) < 2e-5 and rel(dscale, ss.grad) < 2e-5
