"""Shipped SM100 recipe v2: exact oracle, batch invariance, hazards and backward correctness."""

from __future__ import annotations

import pytest
import torch

from isoexec.ops._testing import assert_hazard_exercised, bitwise_equal
from isoexec.ops.norms.norms_sm100 import ops as O
from isoexec.ops.norms.norms_sm100 import reference as R1
from isoexec.ops.norms.norms_sm100 import reference_v2 as R2
from isoexec.ops.norms.norms_sm100.tests.test_sm100_gpu import EPS, rows, weight, shipped_reference

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
    reason="requires B200/SM100",
)

V2_WIDTHS = tuple(sorted(R2.DEFAULT_V2))
SAME_AS_V1_WIDTHS = (128, 384, 768)








@pytest.mark.parametrize("n", V2_WIDTHS + SAME_AS_V1_WIDTHS)
def test_the_lane_rule_matches_the_reference(n):
    """``csrc/plan.h::fwd_lanes_v2`` and ``reference_v2.lanes`` state one rule in two languages."""
    version, ell, k, chain = O.active_recipe(n)
    assert version == O.recipe_default(n)
    assert (ell, k) == shipped_reference(n).lanes(n), n
    assert ell * 8 * k == n and chain == 8 * k
    assert ell & (ell - 1) == 0, f"{n}: L={ell} is not a power of two, so the butterfly is not balanced"


def test_the_default_table_is_stated_in_both_places():
    widths = (128, 384, 512, 768, 1536, 2048, 4096, 5120, 6144, 7168, 8192)
    assert {n for n in widths if O.recipe_default(n) == 2} == set(R2.DEFAULT_V2)
    for n in widths:
        assert O.recipe_default(n) in (1, 2)


def test_the_default_is_per_width_and_never_per_batch():
    from isoexec.ops.norms.norms_sm100._extension import extension
    ext = extension()
    for n in V2_WIDTHS + SAME_AS_V1_WIDTHS + (512, 2048):
        version, ell, _, _ = O.active_recipe(n)
        assert version == O.recipe_default(n)
        for m in (1, 32, 512, 4096, 16384):
            threads = ext.plan(n, m)[0]
            assert threads % ell == 0, (n, m, threads, ell)
            if ell > 32:
                assert threads == ell, (n, m, threads, ell)


@pytest.mark.parametrize("n", V2_WIDTHS)
@pytest.mark.parametrize("zero_centered", [False, True])
def test_matches_the_declared_recipe_v2(n, zero_centered):
    x, w = rows(257, n), weight(n, zero_centered=zero_centered)
    assert bitwise_equal(O.rms_norm(x, w, EPS, zero_centered),
                         R2.rms_norm(x, w, EPS, zero_centered))


@pytest.mark.parametrize("n", V2_WIDTHS)
def test_matches_the_declared_recipe_v2_add(n):
    x, r = rows(129, n), rows(129, n, seed=3, scale=4.0)
    w = weight(n, zero_centered=True)
    y, h = O.add_rms_norm(x, r, w, EPS, True)
    ry, rh = R2.add_rms_norm(x, r, w, EPS, True)
    assert bitwise_equal(h, rh) and bitwise_equal(y, ry)
    assert bitwise_equal(h, x + r)
    assert bitwise_equal(y, O.rms_norm(h, w, EPS, True))


@pytest.mark.parametrize("n", V2_WIDTHS + (512, 2048))
def test_v2_really_is_a_different_rounding_schedule(n):
    from isoexec.ops.norms.norms_sm100._extension import extension

    ext = extension()
    x, w = rows(4096, n, seed=50), weight(n, zero_centered=True)
    rstd1 = R1.inv_rms(R1.row_sum_of_squares(x), n, EPS)
    y1 = R1.rms_norm(x, w, EPS, True)
    rstd2 = R2.inv_rms(R2.row_sum_of_squares(x), n, EPS)
    y2 = R2.rms_norm(x, w, EPS, True)
    actual_y, _, actual_rstd = ext.raw_rms_fwd(x, None, w, EPS, True)
    expected_y, expected_rstd = (y2, rstd2) if O.recipe_default(n) == 2 else (y1, rstd1)
    assert bitwise_equal(actual_y, expected_y)
    assert bitwise_equal(actual_rstd, expected_rstd)
    assert not bitwise_equal(rstd1, rstd2), n
    differing = int((rstd1.view(torch.int32) != rstd2.view(torch.int32)).sum())
    assert differing > 100, (n, differing)
    assert bitwise_equal(y1, R1.rms_norm(x, w, EPS, True))
    assert bitwise_equal(y2, R2.rms_norm(x, w, EPS, True))
    assert_hazard_exercised("recipe_version_changes_bits", True)


@pytest.mark.parametrize("n", SAME_AS_V1_WIDTHS)
def test_where_the_tables_agree_the_two_versions_are_one_program(n):
    assert O.recipe_default(n) == 1
    x, w = rows(300, n, seed=51), weight(n, zero_centered=True)
    y1 = O.rms_norm(x, w, EPS, True).clone()
    assert bitwise_equal(R2.rms_norm(x, w, EPS, True), y1)
    assert bitwise_equal(R1.rms_norm(x, w, EPS, True), y1)
    assert R2.lanes(n) == R1.lanes(n)


@pytest.mark.parametrize("n", V2_WIDTHS)
@pytest.mark.parametrize("zero_centered", [False, True])
def test_batch_invariance_v2(n, zero_centered):
    m = 4096
    x, w = rows(m, n), weight(n, zero_centered=zero_centered)
    full = O.rms_norm(x, w, EPS, zero_centered)
    for start, size in ((0, 1), (1, 1), (7, 3), (100, 64), (2048, 1000), (m - 1, 1), (0, m)):
        sl = x[start : start + size].clone()
        assert bitwise_equal(O.rms_norm(sl, w, EPS, zero_centered), full[start : start + size]), (
            start, size)


@pytest.mark.parametrize("n", V2_WIDTHS)
def test_decode_prefill_trainer_parity_v2(n):
    x, w = rows(20480, n), weight(n, zero_centered=True)
    ref = O.rms_norm(x, w, EPS, True)
    for m in (1, 8, 64, 256, 512, 2048, 4096, 20480):
        assert bitwise_equal(O.rms_norm(x[:m].clone(), w, EPS, True), ref[:m]), m
    xg = x[:2048].detach().clone().requires_grad_(True)
    wg = w.detach().clone().requires_grad_(True)
    assert bitwise_equal(O.rms_norm(xg, wg, EPS, True).detach(), ref[:2048])
    with torch.no_grad():
        assert bitwise_equal(O.rms_norm(x[:2048].clone(), w, EPS, True), ref[:2048])


@pytest.mark.parametrize("n", V2_WIDTHS)
def test_null_lanes_and_padding_are_inert_v2(n):
    x, w = rows(300, n), weight(n, zero_centered=True)
    live = O.rms_norm(x, w, EPS, True)
    for fill in (float("nan"), float("inf"), 0.0, -0.0):
        pad = torch.full((212, n), fill, device="cuda", dtype=torch.bfloat16)
        assert bitwise_equal(O.rms_norm(torch.cat([x, pad]), w, EPS, True)[:300], live), fill
    assert_hazard_exercised("null_lane", True)


@pytest.mark.parametrize("n", V2_WIDTHS)
def test_zero_row_is_defined_and_invariant_v2(n):
    x, w = rows(64, n, seed=52), weight(n, zero_centered=True)
    x[7] = 0.0
    x[8] = -0.0
    y = O.rms_norm(x, w, EPS, True)
    assert bitwise_equal(y, R2.rms_norm(x, w, EPS, True))
    assert bitwise_equal(O.rms_norm(x[7:9].clone(), w, EPS, True), y[7:9])


@pytest.mark.parametrize("n", V2_WIDTHS)
def test_repeat_launches_are_bitwise_v2(n):
    x, w = rows(777, n, seed=53), weight(n, zero_centered=True)
    first = O.rms_norm(x, w, EPS, True).clone()
    for _ in range(20):
        assert bitwise_equal(O.rms_norm(x, w, EPS, True), first)


@pytest.mark.parametrize("n", (4096, 5120, 1536, 7168))
def test_graph_replay_is_bitwise_v2(n):
    m = 512
    x, w = rows(m, n, seed=54), weight(n, zero_centered=True)
    eager = O.rms_norm(x, w, EPS, True)
    static = x.clone()
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        for _ in range(3):
            O.rms_norm(static, w, EPS, True)
    torch.cuda.current_stream().wait_stream(side)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = O.rms_norm(static, w, EPS, True)
    for _ in range(3):
        static.copy_(x)
        g.replay()
        torch.cuda.synchronize()
        assert bitwise_equal(out, eager)
    x2 = rows(m, n, seed=55)
    static.copy_(x2)
    g.replay()
    torch.cuda.synchronize()
    assert bitwise_equal(out, O.rms_norm(x2, w, EPS, True))
    assert_hazard_exercised("graph_replay", True)


def test_split_view_rows_match_contiguous_v2():
    """GLM's kv_a norm reads a [T, kv+rope] slice: a non-contiguous row stride, unit column stride."""
    t, n = 777, 512
    base = rows(t, n + 64, seed=56)
    view = base[:, :n]
    w = weight(n, zero_centered=False)
    assert bitwise_equal(O.rms_norm(view, w, EPS, False), O.rms_norm(view.contiguous(), w, EPS, False))
    assert bitwise_equal(O.rms_norm(view, w, EPS, False), shipped_reference(n).rms_norm(view, w, EPS, False))
    assert_hazard_exercised("non_contiguous_views", not view.is_contiguous())


@pytest.mark.parametrize("n", (2048, 4096, 1536, 7168))
def test_misaligned_storage_matches_v2(n):
    buf = rows(65, n + 8, seed=57)
    view = buf[:, 1 : n + 1]
    w = weight(n, zero_centered=True)
    assert bitwise_equal(O.rms_norm(view, w, EPS, True), O.rms_norm(view.contiguous(), w, EPS, True))
    assert_hazard_exercised("misaligned_storage", (view.data_ptr() % 16) != 0)


def test_fp32_weight_matches_bf16_value_v2():
    n = 4096
    x = rows(129, n, seed=58)
    wb = weight(n, zero_centered=True)
    assert bitwise_equal(O.rms_norm(x, wb.float(), EPS, True), O.rms_norm(x, wb, EPS, True))


def test_shape_of_the_input_view_does_not_matter_v2():
    n = 2048
    x, w = rows(512, n, seed=59), weight(n, zero_centered=False)
    flat = O.rms_norm(x, w, EPS, False)
    assert bitwise_equal(O.rms_norm(x.view(8, 64, n), w, EPS, False).reshape(512, n), flat)


def test_empty_batch_v2():
    n = 4096
    w = weight(n, zero_centered=True)
    assert O.rms_norm(torch.empty(0, n, device="cuda", dtype=torch.bfloat16), w, EPS, True).shape == (0, n)




@pytest.mark.parametrize("n", V2_WIDTHS)
def test_forward_execution_plans_are_bitwise_identical_v2(n):
    """Shipped forward plans preserve the v2 output and reduction bits across batches."""
    from isoexec.ops.norms.norms_sm100._extension import extension
    ext = extension()
    x, w = rows(2048, n, seed=60), weight(n, zero_centered=True)
    ref_y = R2.rms_norm(x, w, EPS, True)
    ref_rstd = R2.inv_rms(R2.row_sum_of_squares(x), n, EPS)
    for m in (1, 7, 300, 1024, 2048):
        y, _, rstd = ext.raw_rms_fwd(x[:m].clone(), None, w, EPS, True)
        assert bitwise_equal(y, ref_y[:m]), (n, m)
        assert bitwise_equal(rstd, ref_rstd[:m]), (n, m)


@pytest.mark.parametrize("n", V2_WIDTHS)
def test_backward_under_v2_matches_float64(n):
    m = 1025
    x = rows(m, n, seed=61).requires_grad_(True)
    w = weight(n, zero_centered=True).requires_grad_(True)
    gout = rows(m, n, seed=62)
    dx, dw = torch.autograd.grad(O.rms_norm(x, w, EPS, True), (x, w), gout)
    xd = x.detach().double().requires_grad_(True)
    wd = w.detach().double().requires_grad_(True)
    yd = xd * torch.rsqrt(xd.pow(2).mean(-1, keepdim=True) + EPS) * (1.0 + wd)
    dxd, dwd = torch.autograd.grad(yd, (xd, wd), gout.double())

    def rel(a, b):
        return ((a.double() - b).norm() / b.norm().clamp_min(1e-30)).item()

    assert rel(dx, dxd) < 5e-3 and rel(dw, dwd) < 5e-3


@pytest.mark.parametrize("n", (2048, 5120, 1536, 7168))
def test_weight_gradient_is_bitwise_repeatable_under_v2(n):
    t = 20480
    x = rows(t, n, seed=63).requires_grad_(True)
    w = weight(n, zero_centered=True, dtype=torch.float32).requires_grad_(True)
    gout = rows(t, n, seed=64)
    first = None
    for _ in range(10):
        (dw,) = torch.autograd.grad(O.rms_norm(x, w, EPS, True), (w,), gout)
        assert torch.isfinite(dw).all()
        if first is None:
            first = dw.clone()
        else:
            assert bitwise_equal(dw, first)
