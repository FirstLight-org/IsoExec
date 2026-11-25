"""norms_sm100 at N = 1536 and N = 7168 (measured / measured class q_a latent and hidden widths)."""

from __future__ import annotations

import pytest
import torch

from isoexec.ops._testing import assert_hazard_exercised, bitwise_equal
from isoexec.ops.norms.norms_sm100 import ops as O
from isoexec.ops.norms.norms_sm100 import reference as R1
from isoexec.ops.norms.norms_sm100 import reference_v2 as R2
from isoexec.ops.norms.norms_sm100._extension import extension
from isoexec.ops.norms.norms_sm100.tests.test_sm100_gpu import EPS, hazards, rows, weight

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
    reason="requires B200/SM100",
)

NEW_WIDTHS = (1536, 7168)
BATCHES = (1, 7, 256, 2048, 16384)
REF = {1: R1, 2: R2}


@pytest.fixture
def recipe(request):
    """Both widths ship recipe v2; no mutable recipe selection is involved."""
    return request.param


recipes = pytest.mark.parametrize("recipe", (2,), indirect=True, ids=("shipped_v2",))
widths = pytest.mark.parametrize("n", NEW_WIDTHS)


def _ref_rms(ver, x, w, zc):
    """The reference over any number of rows, in chunks so T = 16384 x 7168 stays small."""
    return torch.cat([REF[ver].rms_norm(x[i : i + 2048], w, EPS, zc) for i in range(0, x.shape[0], 2048)])


@widths
def test_lane_programs_are_as_documented(n):
    want = {1536: ((32, 6), (64, 3), (32, 6)), 7168: ((32, 28), (128, 7), (128, 7))}[n]
    _, _, bl, bk = O.recipe(n)
    assert (R1.lanes(n), O.active_recipe(n)[1:3], (bl, bk)) == want
    assert O.active_recipe(n)[0] == 2
    assert O.supported_width(n)


@widths
@recipes
@pytest.mark.parametrize("zero_centered", [False, True])
def test_forward_matches_the_declared_recipe(n, recipe, zero_centered):
    x, w = rows(2048 + 3, n, seed=70), weight(n, zero_centered=zero_centered)
    h = hazards(x)
    assert_hazard_exercised("signed_zero", h["signed_zero"] > 0)
    assert_hazard_exercised("subnormal", h["subnormal"] > 0)
    x[5] = 0.0  # an all-zero row (rstd = rsqrt(eps)) and a cancelling one
    x[6] = -0.0
    x[7, : n // 2] = x[7, n // 2 :].flip(0) * -1  # sums cancel pairwise; the ssq does not
    assert bitwise_equal(O.rms_norm(x, w, EPS, zero_centered), REF[recipe].rms_norm(x, w, EPS, zero_centered))
    assert bitwise_equal(O.rms_norm(x, w.float(), EPS, zero_centered),
                         O.rms_norm(x, w, EPS, zero_centered))


@widths
@recipes
@pytest.mark.parametrize("zero_centered", [False, True])
def test_fused_add_matches_the_declared_recipe(n, recipe, zero_centered):
    x, r = rows(1030, n, seed=71), rows(1030, n, seed=72, scale=4.0)
    w = weight(n, zero_centered=zero_centered)
    y, h = O.add_rms_norm(x, r, w, EPS, zero_centered)
    ry, rh = REF[recipe].add_rms_norm(x, r, w, EPS, zero_centered)
    assert bitwise_equal(h, rh) and bitwise_equal(y, ry)
    assert bitwise_equal(h, x + r)
    assert bitwise_equal(y, O.rms_norm(h, w, EPS, zero_centered))


@widths
@recipes
def test_overflow_nan_and_inf_rows_match_the_recipe(n, recipe):
    x, w = rows(64, n, seed=73), weight(n, zero_centered=True)
    big = torch.tensor(3.0e38, dtype=torch.bfloat16, device="cuda")
    x[1] = big
    x[2, 17] = float("inf")
    x[3, 5] = float("nan")
    x[4, :] = -big
    y = O.rms_norm(x, w, EPS, True)
    ref = REF[recipe].rms_norm(x, w, EPS, True)
    ok = torch.ones(64, dtype=torch.bool, device="cuda")
    ok[1:5] = False
    assert bitwise_equal(y[ok], ref[ok])
    assert torch.equal(torch.isnan(y[1:5]), torch.isnan(ref[1:5]))
    assert torch.equal(torch.nan_to_num(y[1:5], nan=7.0), torch.nan_to_num(ref[1:5], nan=7.0))
    assert_hazard_exercised("overflow_nan_inf", True)


@widths
@recipes
@pytest.mark.parametrize("zero_centered", [False, True])
def test_forward_is_invariant_to_batch_composition(n, recipe, zero_centered):
    m = 16384
    x, w = rows(m, n, seed=74), weight(n, zero_centered=zero_centered)
    full = O.rms_norm(x, w, EPS, zero_centered)
    assert bitwise_equal(full, _ref_rms(recipe, x, w, zero_centered))  # T = 16384 IS the recipe
    for t in BATCHES:
        for start in (0, 5, m - t):
            sl = x[start : start + t].clone()  # fresh storage, different alignment of the batch
            assert bitwise_equal(O.rms_norm(sl, w, EPS, zero_centered), full[start : start + t]), (t, start)
    g = torch.Generator(device="cuda").manual_seed(3)
    for t in (7, 256, 2048):
        idx = torch.randperm(m, generator=g, device="cuda")[:t]
        assert bitwise_equal(O.rms_norm(x[idx], w, EPS, zero_centered), full[idx]), t


@widths
@recipes
def test_fused_add_is_invariant_to_batch_composition(n, recipe):
    m = 16384
    x, r, w = rows(m, n, seed=75), rows(m, n, seed=76, scale=2.0), weight(n, zero_centered=True)
    y, h = O.add_rms_norm(x, r, w, EPS, True)
    assert bitwise_equal(h, x + r)
    for t in BATCHES:
        for start in (0, m - t):
            yy, hh = O.add_rms_norm(x[start : start + t].clone(), r[start : start + t].clone(), w, EPS, True)
            assert bitwise_equal(yy, y[start : start + t]) and bitwise_equal(hh, h[start : start + t]), (t, start)
    assert bitwise_equal(y, O.rms_norm(h, w, EPS, True))


@widths
@recipes
def test_padding_rows_are_inert(n, recipe):
    x, w = rows(300, n, seed=77), weight(n, zero_centered=True)
    live = O.rms_norm(x, w, EPS, True)
    for fill in (float("nan"), float("inf"), 0.0, -0.0):
        pad = torch.full((212, n), fill, device="cuda", dtype=torch.bfloat16)
        assert bitwise_equal(O.rms_norm(torch.cat([x, pad]), w, EPS, True)[:300], live), fill
    y, h = O.add_rms_norm(x, x.flip(0).contiguous(), w, EPS, True)
    for fill in (float("nan"), 0.0):
        pad = torch.full((212, n), fill, device="cuda", dtype=torch.bfloat16)
        yy, hh = O.add_rms_norm(torch.cat([x, pad]), torch.cat([x.flip(0), pad]), w, EPS, True)
        assert bitwise_equal(yy[:300], y) and bitwise_equal(hh[:300], h), fill


@widths
@recipes
def test_fused_add_execution_plans_are_bitwise_identical(n, recipe):
    """Fused outputs and rstd follow the declared recipe across shipped batch plans."""
    ext = extension()
    x, r, w = rows(2048, n, seed=106), rows(2048, n, seed=107), weight(n, zero_centered=True)
    ry, rh = REF[recipe].add_rms_norm(x, r, w, EPS, True)
    rr = REF[recipe].inv_rms(REF[recipe].row_sum_of_squares(rh), n, EPS)
    for m in (1, 7, 300, 1024, 2048):
        y, h, rstd = ext.raw_rms_fwd(x[:m].clone(), r[:m].clone(), w, EPS, True)
        assert bitwise_equal(y, ry[:m]) and bitwise_equal(h, rh[:m]), (n, recipe, m)
        assert bitwise_equal(rstd, rr[:m]), (n, recipe, m)


def _graph(fn, static_inputs, warm=3):
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        for _ in range(warm):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    return g, out


@widths
@recipes
@pytest.mark.parametrize("m", (7, 512))
def test_graph_replay_forward_and_fused_add(n, recipe, m):
    x, r, w = rows(m, n, seed=78), rows(m, n, seed=79), weight(n, zero_centered=True)
    ey, (ay, ah) = O.rms_norm(x, w, EPS, True), O.add_rms_norm(x, r, w, EPS, True)
    sx, sr = x.clone(), r.clone()
    g, out = _graph(lambda: (O.rms_norm(sx, w, EPS, True), *O.add_rms_norm(sx, sr, w, EPS, True)), None)
    for _ in range(3):
        sx.copy_(x), sr.copy_(r)
        g.replay()
        torch.cuda.synchronize()
        assert bitwise_equal(out[0], ey) and bitwise_equal(out[1], ay) and bitwise_equal(out[2], ah)
    x2, r2 = rows(m, n, seed=80), rows(m, n, seed=81)
    sx.copy_(x2), sr.copy_(r2)
    g.replay()
    torch.cuda.synchronize()
    y2, h2 = O.add_rms_norm(x2, r2, w, EPS, True)
    assert bitwise_equal(out[0], O.rms_norm(x2, w, EPS, True))
    assert bitwise_equal(out[1], y2) and bitwise_equal(out[2], h2)
    assert_hazard_exercised("graph_replay", True)


@widths
@pytest.mark.parametrize("m", (7, 512, 2048))
def test_graph_replay_backward(n, m):
    x = rows(m, n, seed=82).requires_grad_(True)
    w = weight(n, zero_centered=True, dtype=torch.float32).requires_grad_(True)
    gout = rows(m, n, seed=83)

    def run():
        return torch.autograd.grad(O.rms_norm(x, w, EPS, True), (x, w), gout)

    eager = tuple(t.clone() for t in run())
    g, grads = _graph(run, None)
    for _ in range(3):
        g.replay()
        torch.cuda.synchronize()
        assert all(bitwise_equal(a, b) for a, b in zip(grads, eager))


VIEWS = [(1536, 512 + 64, 0), (1536, 8, 0), (7168, 64, 0), (7168, 7168, 0)]


@pytest.mark.parametrize("n,extra,off", VIEWS)
@recipes
def test_strided_row_views_match_contiguous(n, extra, off, recipe):
    t = 777
    base = rows(t, n + extra, seed=84)
    view = base[:, off : off + n]
    w = weight(n, zero_centered=False)
    res = rows(t, n + extra, seed=85)[:, off : off + n]
    assert_hazard_exercised("non_contiguous_views", not view.is_contiguous())
    assert bitwise_equal(O.rms_norm(view, w, EPS, False), O.rms_norm(view.contiguous(), w, EPS, False))
    assert bitwise_equal(O.rms_norm(view, w, EPS, False), REF[recipe].rms_norm(view, w, EPS, False))
    y, h = O.add_rms_norm(view, res, w, EPS, False)
    yc, hc = O.add_rms_norm(view.contiguous(), res.contiguous(), w, EPS, False)
    assert bitwise_equal(y, yc) and bitwise_equal(h, hc)
    xv = view.detach().clone().requires_grad_(True)
    gout_base = rows(t, n + extra, seed=86)
    gview = gout_base[:, off : off + n]
    wf = weight(n, zero_centered=False, dtype=torch.float32).requires_grad_(True)
    xs = view.detach().requires_grad_(True)
    dxs, dws = torch.autograd.grad(O.rms_norm(xs, wf, EPS, False), (xs, wf), gview)
    dxc, dwc = torch.autograd.grad(O.rms_norm(xv, wf, EPS, False), (xv, wf), gview.contiguous())
    assert bitwise_equal(dxs, dxc) and bitwise_equal(dws, dwc)


@widths
@recipes
def test_only_4_byte_aligned_row_strides_match(n, recipe):
    t = 65
    buf = rows(t, n + 9, seed=87)
    view = buf[:, 1 : n + 1]
    assert view.data_ptr() % 16 != 0 and view.stride(0) % 8 != 0
    w = weight(n, zero_centered=True)
    assert bitwise_equal(O.rms_norm(view, w, EPS, True), O.rms_norm(view.contiguous(), w, EPS, True))
    assert bitwise_equal(O.rms_norm(view, w, EPS, True), REF[recipe].rms_norm(view, w, EPS, True))


@widths
def test_leading_dims_do_not_matter(n):
    """[T, N], [B, S, N] and a transposed-batch view are the same rows."""
    x, w = rows(512, n, seed=88), weight(n, zero_centered=True)
    flat = O.rms_norm(x, w, EPS, True)
    assert bitwise_equal(O.rms_norm(x.view(8, 64, n), w, EPS, True).reshape(512, n), flat)
    assert bitwise_equal(O.rms_norm(x.view(64, 8, n).transpose(0, 1), w, EPS, True).transpose(0, 1).reshape(512, n), flat)


def _fwd_rstd(ver, x, n):
    """The recipe's own rstd for the rows of `x` (not the kernel's)."""
    return REF[ver].inv_rms(REF[ver].row_sum_of_squares(x), n, EPS)


def bwd_reference(dy, x, w, zero_centered, dres=None, version=1, bands_override=0):
    m, n = x.shape
    _, _, L, K = O.recipe(n)
    ext = extension()
    bands, iters = ext.plan(n, m)[3:5]
    if bands_override:
        raise NotImplementedError
    rs = _fwd_rstd(version, x, n)
    h = x.float() * rs[:, None]
    g = None if w is None else (1.0 + w.float() if zero_centered else w.float())
    gj = dy.float() if g is None else dy.float() * g
    kk = torch.arange(K, device=x.device).view(K, 1, 1)
    ll = torch.arange(L, device=x.device).view(1, L, 1)
    jj = torch.arange(8, device=x.device).view(1, 1, 8)
    idx = (kk * 8 * L + 8 * ll + jj).permute(1, 0, 2).reshape(L, K * 8)
    prod = (h * gj)[:, idx]  # (M, L, C)
    acc = torch.zeros(m, L, device=x.device, dtype=torch.float32)
    for t in range(prod.shape[2]):
        acc = acc + prod[:, :, t]
    s, lane = L >> 1, torch.arange(L, device=x.device)
    while s:
        acc = acc + acc[:, lane ^ s]
        s >>= 1
    dn = acc[:, 0] / torch.full((m,), float(n), device=x.device)
    o = rs[:, None] * (gj - h * dn[:, None])
    if dres is not None:
        o = o + dres.float()
    dx = o.bfloat16()
    if w is None:
        return dx, None
    rpi = 128 // L if L <= 64 else 1
    total = bands * iters * rpi
    pad = total - m
    contrib = dy.float() * h  # fl32(d * h), one rounding
    contrib = torch.cat([contrib, torch.zeros(pad, n, device=x.device)]) if pad else contrib
    contrib = contrib.view(bands, iters, rpi, n)  # row = (b*iters + it)*rpi + g
    part = torch.zeros(bands, rpi, n, device=x.device)
    for it in range(iters):
        part = part + contrib[:, it]
    band_part = part[:, 0]
    for q in range(1, rpi):
        band_part = band_part + part[:, q]
    dw = torch.zeros(n, device=x.device)
    for b in range(bands):
        dw = dw + band_part[b]
    return dx, dw


def _check_bwd(n, m, zero_centered, version, with_res):
    x, w = rows(m, n, seed=90 + m % 7), weight(n, zero_centered=zero_centered, dtype=torch.float32)
    dy = rows(m, n, seed=91)
    dres = rows(m, n, seed=92, scale=0.5) if with_res else None
    ext = extension()
    _, _, rstd = ext.raw_rms_fwd(x, None, w, EPS, zero_centered)
    assert bitwise_equal(rstd, _fwd_rstd(version, x, n))  # the forward's rstd IS the recipe's
    dx, dw = ext.raw_rms_bwd(dy, x, rstd, w, dres, zero_centered, 0)
    rdx, rdw = bwd_reference(dy, x, w, zero_centered, dres, version)
    assert bitwise_equal(dx, rdx), f"{n} m={m}: dx differs from the backward's statement"
    assert bitwise_equal(dw, rdw), f"{n} m={m}: dW differs from the backward's statement"
    wb = w.bfloat16()
    dxb, dwb = ext.raw_rms_bwd(dy, x, rstd, wb, dres, zero_centered, 0)
    rdxb, rdwb = bwd_reference(dy, x, wb, zero_centered, dres, version)
    assert dwb.dtype == torch.bfloat16  # the kernel's dW is fp32 until the autograd cast
    assert bitwise_equal(dxb, rdxb) and bitwise_equal(dwb, rdwb.bfloat16())


@pytest.mark.parametrize("m", (1, 7, 300, 1025, 3000))
@pytest.mark.parametrize("zero_centered", [False, True])
@pytest.mark.parametrize("with_res", [False, True])
def test_backward_reference_is_valid_at_a_shipped_width(m, zero_centered, with_res):
    for n in (2048, 768):
        _check_bwd(n, m, zero_centered, O.recipe_default(n), with_res)


@widths
@recipes
@pytest.mark.parametrize("m", (1, 7, 300, 1025, 3000))
@pytest.mark.parametrize("zero_centered", [False, True])
@pytest.mark.parametrize("with_res", [False, True])
def test_backward_matches_the_declared_program_bit_for_bit(n, recipe, m, zero_centered, with_res):
    _check_bwd(n, m, zero_centered, recipe, with_res)


@widths
@pytest.mark.parametrize("m", (256, 2048, 16384))
def test_backward_dx_is_invariant_to_batch_composition(n, m):
    x, w = rows(m, n, seed=93), weight(n, zero_centered=True, dtype=torch.float32)
    dy = rows(m, n, seed=94)
    ext = extension()
    _, _, rstd = ext.raw_rms_fwd(x, None, w, EPS, True)
    dx, _ = ext.raw_rms_bwd(dy, x, rstd, w, None, True, 0)
    for t in (1, 7, 100):
        for start in (0, m - t):
            sl = slice(start, start + t)
            d1, _ = ext.raw_rms_bwd(dy[sl].clone(), x[sl].clone(), rstd[sl].clone(), w, None, True, 0)
            assert bitwise_equal(d1, dx[sl]), (m, t, start)


@widths
def test_backward_against_float64_and_residual_gradient(n):
    """The package's own tolerance check (5e-3 relative, bf16 data) for x, W and the residual branch."""
    m = 1025
    x = rows(m, n, seed=95).requires_grad_(True)
    r = rows(m, n, seed=96, scale=4.0).requires_grad_(True)
    w = weight(n, zero_centered=True).requires_grad_(True)
    gy, gh = rows(m, n, seed=97), rows(m, n, seed=98, scale=0.5)
    y, h = O.add_rms_norm(x, r, w, EPS, True)
    dx, dr, dw = torch.autograd.grad((y, h), (x, r, w), (gy, gh))
    xd, rd, wd = (t.detach().double().requires_grad_(True) for t in (x, r, w))
    hd = xd + rd
    yd = hd * torch.rsqrt(hd.pow(2).mean(-1, keepdim=True) + EPS) * (1.0 + wd)
    dxd, drd, dwd = torch.autograd.grad((yd, hd), (xd, rd, wd), (gy.double(), gh.double()))

    def rel(a, b):
        return ((a.double() - b).norm() / b.norm().clamp_min(1e-30)).item()

    assert rel(dx, dxd) < 5e-3 and rel(dr, drd) < 5e-3 and rel(dw, dwd) < 5e-3
    assert bitwise_equal(dx, dr)


@widths
@pytest.mark.parametrize("m", (7, 300, 1024, 3000))
def test_backward_execution_plans_with_residual_gradient(n, m):
    """Residual-gradient backward matches the numerical oracle under each shipped plan."""
    ext = extension()
    x, w = rows(m, n, seed=99), weight(n, zero_centered=True, dtype=torch.float32)
    dy, dres = rows(m, n, seed=100), rows(m, n, seed=101)
    _, _, rstd = ext.raw_rms_fwd(x, None, w, EPS, True)
    rdx, rdw = bwd_reference(dy, x, w, True, dres, O.recipe_default(n))
    for _ in range(3):
        dx, dw = ext.raw_rms_bwd(dy, x, rstd, w, dres, True, 0)
        assert bitwise_equal(dx, rdx) and bitwise_equal(dw, rdw), (n, m)


@widths
def test_weight_gradient_is_bitwise_repeatable(n):
    t = 20480
    x = rows(t, n, seed=102).requires_grad_(True)
    w = weight(n, zero_centered=True, dtype=torch.float32).requires_grad_(True)
    gout = rows(t, n, seed=103)
    first = None
    for _ in range(20):
        (dw,) = torch.autograd.grad(O.rms_norm(x, w, EPS, True), (w,), gout)
        assert torch.isfinite(dw).all()
        if first is None:
            first = dw.clone()
        else:
            assert bitwise_equal(dw, first)


@widths
def test_backward_dx_is_repeatable_across_many_launches(n):
    m = 16384
    x, w = rows(m, n, seed=104), weight(n, zero_centered=True, dtype=torch.float32)
    dy = rows(m, n, seed=105)
    ext = extension()
    _, _, rstd = ext.raw_rms_fwd(x, None, w, EPS, True)
    first = None
    for _ in range(200):
        dx, dw = ext.raw_rms_bwd(dy, x, rstd, w, None, True, 0)
        if first is None:
            first = (dx.clone(), dw.clone())
        else:
            assert bitwise_equal(dx, first[0]) and bitwise_equal(dw, first[1])


def test_row_fold_is_double_buffered():
    import pathlib

    src = (pathlib.Path(__file__).resolve().parents[1] / "csrc" / "bwd.cuh").read_text()
    assert "__shared__ float sh[2][128];" in src and "float *b = sh[buf & 1];" in src
    assert "row_fold<L>(acc, it)" in src
    launch = (pathlib.Path(__file__).resolve().parents[1] / "csrc" / "launch_bwd.cu").read_text()
    assert "X(7, 128)" in launch and "X(6, 32)" in launch
    fwd = (pathlib.Path(__file__).resolve().parents[1] / "csrc" / "common.cuh").read_text()
    assert "One `__syncthreads` per call and no reuse of the buffer" in fwd
