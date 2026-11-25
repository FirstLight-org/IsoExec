"""Bitwise invariance of the ported norms_sm100 kernels (the policy's schedule) against themselves.

What is varied: repeated runs; batch size and row position; batch composition; storage (fresh clone, view);
engine (no_grad) against trainer (autograd) calls; fused add against the unfused pair; row counts around every
row-segment boundary for the weight gradient; padding rows. Equality is on raw bits (NaN payloads included).
"""

from __future__ import annotations

import pytest
import torch

from isoexec.ops._testing import bitwise_equal
from isoexec.ops.norms.norms_sm100 import ops as O
from isoexec.ops.norms.norms_sm100 import reference as R
from isoexec.ops.norms.norms_sm100 import schedule
from isoexec.ops.norms.norms_sm100._extension import extension
from isoexec.ops.norms.norms_sm100.tests.test_sm100_gpu import EPS, rows, weight

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0), reason="requires B200/SM100"
)

WIDTHS = (128, 384, 256, 512, 768, 1024, 2048, 6144, 8192, 1536, 4096, 5120, 7168)
BATCHES = (1, 2, 3, 7, 8, 9, 31, 32, 33, 64, 127, 128, 129, 255, 256, 257, 1023, 1024, 1025, 2048)
S = schedule.SUM_SHORTEST_SEGMENT
ROW_COUNTS = (1, S - 1, S, S + 1, 2 * S - 1, 2 * S, 2 * S + 1, 3 * S, 5 * S + 3, 127, 128, 129, 255, 256, 257,
              511, 512, 513, 1023, 1024, 1025, 2047, 2048, 2049, 4099)


def test_the_compiled_segment_lengths_are_the_policy_output():
    hidden, row = extension().schedule()
    for n in WIDTHS:
        assert hidden == schedule.hidden_sum(n).S, n
    for m in ROW_COUNTS:
        assert row == schedule.row_sum(m).S, m


@pytest.mark.parametrize("n", WIDTHS)
def test_forward_is_the_schedule_and_repeats(n):
    x, w = rows(513, n, seed=n), weight(n, zero_centered=True)
    ref = R.rms_norm(x, w, EPS, True)
    for _ in range(3):
        assert bitwise_equal(O.rms_norm(x, w, EPS, True), ref)
    rs = extension().raw_rms_fwd(x, None, w, EPS, True)[2]
    assert bitwise_equal(rs, R.inv_rms(R.row_sum_of_squares(x), n, EPS))


@pytest.mark.parametrize("n", WIDTHS)
def test_a_row_alone_is_the_row_in_every_batch_at_every_position(n):
    w = weight(n, zero_centered=False)
    probe = rows(3, n, seed=7 + n)
    alone = [O.rms_norm(probe[i : i + 1].clone(), w, EPS, False) for i in range(3)]
    for b in BATCHES:
        other = rows(b, n, seed=b, scale=50.0)
        for i, pos in enumerate(sorted({0, b // 2, b - 1})):
            batch = other.clone()
            batch[pos] = probe[i % 3]
            assert bitwise_equal(O.rms_norm(batch, w, EPS, False)[pos : pos + 1], alone[i % 3]), (n, b, pos)


@pytest.mark.parametrize("n", (128, 2048, 1536, 4096, 5120, 7168))
@pytest.mark.parametrize("fill", ("nan", "inf", "zero", "negzero", "huge", "tiny"))
def test_neighbours_do_not_change_a_row(n, fill):
    w = weight(n, zero_centered=True)
    x = rows(65, n, seed=3)
    want = O.rms_norm(x, w, EPS, True)
    value = {"nan": float("nan"), "inf": float("inf"), "zero": 0.0, "negzero": -0.0, "huge": 3e38, "tiny": 1e-38}[fill]
    mixed = torch.full((130, n), value, device="cuda", dtype=torch.bfloat16)
    mixed[::2] = x
    got = O.rms_norm(mixed, w, EPS, True)
    assert bitwise_equal(got[::2], want), (n, fill)
    assert bitwise_equal(got[1::2], R.rms_norm(mixed[1::2], w, EPS, True)), (n, fill)


@pytest.mark.parametrize("n", (128, 1024, 2048, 1536, 4096, 5120, 7168))
def test_engine_trainer_and_fused_calls_agree(n):
    x, r, w = rows(300, n, seed=11), rows(300, n, seed=12, scale=4.0), weight(n, zero_centered=True)
    with torch.no_grad():
        engine = O.rms_norm(r + x, w, EPS, True)
        fused_engine, h_engine = O.add_rms_norm_fused(x, r, w, EPS, True)
        decode = torch.cat([O.rms_norm((r + x)[i : i + 1].clone(), w, EPS, True) for i in (0, 1, 149, 299)])
        chunks = torch.cat([O.rms_norm((r + x)[a:b].clone(), w, EPS, True) for a, b in ((0, 1), (1, 130), (130, 300))])
    xg, rg, wg = (t.detach().clone().requires_grad_(True) for t in (x, r, w))
    trainer = O.rms_norm(rg + xg, wg, EPS, True)
    fused_trainer, h_trainer = O.add_rms_norm_fused(xg, rg, wg, EPS, True)
    assert bitwise_equal(trainer.detach(), engine) and bitwise_equal(fused_engine, engine)
    assert bitwise_equal(fused_trainer.detach(), engine) and bitwise_equal(h_engine, h_trainer.detach())
    assert bitwise_equal(decode, engine[[0, 1, 149, 299]]) and bitwise_equal(chunks, engine)


@pytest.mark.parametrize("hv", (1, 4, 6, 8))
@pytest.mark.parametrize("n", (128, 384, 512))
def test_gated_out_norm_rows_and_head_shards(n, hv):
    """A token's head rows are the same alone, inside any batch, and when the heads are split across ranks."""
    t = 67
    x, z = rows(t * hv, n, seed=5).view(t, hv, n), rows(t * hv, n, seed=6, scale=12.0).view(t, hv, n)
    w = weight(n, zero_centered=True)
    full = O.gated_rms_norm(x, z, w, EPS, True)
    assert bitwise_equal(full, R.gated_rms_norm(x, z, w, EPS, True))
    assert bitwise_equal(full, O.gated_rms_norm(x, z, w, EPS, True))
    for a, b in ((0, 1), (1, 9), (9, 67)):
        assert bitwise_equal(O.gated_rms_norm(x[a:b].clone(), z[a:b].clone(), w, EPS, True), full[a:b]), (a, b)
    for degree in (1, 2, 4):
        if hv % degree:
            continue
        per = hv // degree
        parts = [O.gated_rms_norm(x[:, k * per : (k + 1) * per].contiguous(), z[:, k * per : (k + 1) * per].contiguous(),
                                  w, EPS, True) for k in range(degree)]
        assert bitwise_equal(torch.cat(parts, dim=1), full), (n, hv, degree)


@pytest.mark.parametrize("n", (128, 384, 768, 2048, 1536, 4096, 5120, 7168))
@pytest.mark.parametrize("m", ROW_COUNTS)
def test_backward_is_the_schedule_at_every_row_count(n, m):
    """dx (row dot on hidden segments) and dW (row segments, short last segment) against the statement, twice."""
    ext = extension()
    x, w = rows(m, n, seed=m), weight(n, zero_centered=True, dtype=torch.float32)
    dy, dres = rows(m, n, seed=m + 1), rows(m, n, seed=m + 2, scale=0.5)
    rstd = ext.raw_rms_fwd(x, None, w, EPS, True)[2]
    for res in (None, dres):
        rdx, rdw = R.rms_backward(dy, x, w, EPS, True, res)
        for _ in range(2):
            dx, dw = ext.raw_rms_bwd(dy, x, rstd, w, res, True)
            assert bitwise_equal(dx, rdx), (n, m, "dx")
            assert bitwise_equal(dw, rdw), (n, m, "dW")


@pytest.mark.parametrize("n", (128, 2048, 1536, 4096, 5120))
def test_backward_dx_of_a_row_does_not_depend_on_its_batch(n):
    ext = extension()
    m = 1027
    x, w = rows(m, n, seed=21), weight(n, zero_centered=False, dtype=torch.float32)
    dy = rows(m, n, seed=22)
    rstd = ext.raw_rms_fwd(x, None, w, EPS, False)[2]
    dx = ext.raw_rms_bwd(dy, x, rstd, w, None, False)[0]
    for a, b in ((0, 1), (5, 6), (7, 9), (8, 16), (3, 260), (1026, 1027), (500, 1027)):
        part = ext.raw_rms_bwd(dy[a:b].clone(), x[a:b].clone(), rstd[a:b].clone(), w, None, False)[0]
        assert bitwise_equal(part, dx[a:b]), (n, a, b)


@pytest.mark.parametrize("n", (128, 384))
def test_gated_backward_repeats_and_rows_are_independent(n):
    ext = extension()
    t, hv = 37, 8
    x, z = rows(t * hv, n, seed=31).view(t, hv, n), rows(t * hv, n, seed=32, scale=6.0).view(t, hv, n)
    w = weight(n, zero_centered=True, dtype=torch.float32)
    dy = rows(t, hv * n, seed=33)
    rstd = ext.raw_gated_fwd(x, z, w, EPS, True)[1]
    dx, dz, dw = ext.raw_gated_bwd(dy, x, z, rstd, w, True)
    dx2, dz2, dw2 = ext.raw_gated_bwd(dy, x, z, rstd, w, True)
    assert bitwise_equal(dx, dx2) and bitwise_equal(dz, dz2) and bitwise_equal(dw, dw2)
    for a, b in ((0, 1), (1, 3), (3, 37)):
        px, pz, _ = ext.raw_gated_bwd(dy[a:b].clone(), x[a:b].clone(), z[a:b].clone(),
                                      rstd.view(t, hv)[a:b].reshape(-1).clone(), w, True)
        assert bitwise_equal(px, dx[a:b]) and bitwise_equal(pz, dz[a:b]), (a, b)
