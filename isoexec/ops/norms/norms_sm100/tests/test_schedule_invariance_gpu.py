"""norms_sm100 forward on the device: the compiled plan is ``schedule.py``, the kernel is the stated DAG, and
the kernel is bitwise invariant against itself.

What is varied: repeated runs; batch size and row position; batch composition (neighbour rows of NaN, inf,
zeros, huge and tiny values; zero rows stand for padding); storage (fresh clones); engine (no_grad) against
trainer (autograd) calls; one call against row-by-row and chunked calls; fused add against the unfused pair.
The hidden axis has no shard degree (``schedule.HIDDEN_SHARDS == 1``), so there is no TP case. Equality is on
raw bits (NaN payloads included).
"""

from __future__ import annotations

import pytest
import torch

from isoexec.ops._testing import bitwise_equal
from isoexec.ops.norms.norms_sm100 import ops as O
from isoexec.ops.norms.norms_sm100 import schedule
from isoexec.ops.norms.norms_sm100._extension import extension
from isoexec.ops.norms.norms_sm100.tests.test_sm100_gpu import EPS, rows, shipped_reference, weight

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0), reason="requires B200/SM100"
)

WIDTHS = (128, 384, 256, 512, 768, 1024, 2048, 6144, 8192, 1536, 4096, 5120, 7168)
BATCHES = (1, 2, 3, 7, 8, 9, 31, 32, 33, 64, 127, 128, 129, 255, 256, 257, 1023, 1024, 1025, 2048)
SUPPORTED = (128, 384) + tuple(range(256, 16384 + 1, 256))


def test_the_compiled_plan_is_schedule_py():
    """Read back from the built module: support, binding, and (version, L, K, chain) on every table entry."""
    ext = extension()
    for n in range(8, 16384 + 1, 8):
        assert ext.fwd_supported(n) == (n in SUPPORTED), n
    for n in SUPPORTED:
        red = schedule.reducer(n)
        ell, k = schedule.lanes(n)
        assert O.recipe_default(n) == red.version == schedule.BINDING.get(n, schedule.DEFAULT_VERSION), n
        assert tuple(O.active_recipe(n)) == (red.version, ell, k, schedule.LEAF * k), n
        assert schedule.segment(n).S == schedule.LEAF * ell * k


@pytest.mark.parametrize("n", WIDTHS)
def test_forward_is_the_stated_dag_and_repeats(n):
    """rstd from the kernel = inv_rms of the DAG evaluated from schedule.py's data, and of its reference."""
    R = shipped_reference(n)
    x, w = rows(513, n, seed=n), weight(n, zero_centered=True)
    red = schedule.reducer(n)
    ssq = schedule.evaluate(x)
    assert bitwise_equal(ssq, red.reference_fn()(x))
    out, _, rstd = extension().raw_rms_fwd(x, None, w, EPS, True)
    assert bitwise_equal(rstd, R.inv_rms(ssq, n, EPS)), n
    ref = R.rms_norm(x, w, EPS, True)
    assert bitwise_equal(out, ref)
    for _ in range(3):
        assert bitwise_equal(O.rms_norm(x, w, EPS, True), ref)


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
    R = shipped_reference(n)
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
