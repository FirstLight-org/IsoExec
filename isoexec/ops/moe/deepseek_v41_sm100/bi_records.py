"""Checked-in record table of the TP-sharded experts' two grouped GEMMs (``bi_tp``): the default placement, so the
choice is reproducible and identical on every rank and every run (no timing at warm-up). A record is placement only:
the bits do not depend on it (vendored BI class guarantee; ``tests/test_bi_tp_gpu.py`` checks record pairs).

Keys ``(tp, tokens, gemm)`` with gemm 1 = FC1 (N = 2 I / tp, K = H, the ``v41_actor`` chain) and 2 = FC2 (N = H / tp,
K = I); values the ``Knobs`` fields. Shapes and token counts not listed take the dispatcher's first record
(``schedule.rank_grouped``), also deterministic. Produced by timing the dispatcher's top-ranked records (plus the 2-CTA
256-row records at 8192 tokens; r4c-q1 added 512 / 2048 and re-measured 8192 with 2-CTA 256-row tiles of N 64 / 128 / 256) on one B200 at H 5120, I 2304, 384 + 1 experts, 7 slots, and keeping the fastest;
``BiTPExperts.tune(sizes)`` regenerates entries in this format.
"""
from __future__ import annotations

_SWAP = dict(use_2cta_instrs=False, use_tma_store=True, raster_along="m", swizzle_size=1, grid_clusters=0,
             cta_chunks=False, cluster_k=1, swap_ab=True)


def _swap(m, n):
    return dict(_SWAP, mma_tiler_mn=(m, n), cluster_shape_mn=(1, 1))


def _dense(m, n, cl, two, raster):
    return dict(mma_tiler_mn=(m, n), cluster_shape_mn=cl, use_2cta_instrs=two, use_tma_store=True,
                raster_along=raster, swizzle_size=1, grid_clusters=0, cta_chunks=False, cluster_k=1, swap_ab=False)


TABLE = {
    (8, 1, 1): _swap(64, 32),
    (8, 1, 2): _swap(64, 16),
    (8, 8, 1): _swap(64, 32),
    (8, 8, 2): _swap(128, 8),
    (8, 128, 1): _swap(64, 16),
    (8, 128, 2): _dense(64, 128, (1, 1), False, "m"),
    (8, 512, 1): _dense(64, 192, (1, 1), False, "m"),
    (8, 512, 2): _dense(64, 128, (1, 1), False, "n"),
    (8, 2048, 1): _dense(64, 192, (1, 1), False, "m"),
    (8, 2048, 2): _dense(64, 128, (1, 1), False, "m"),
    (8, 8192, 1): _dense(256, 256, (2, 1), True, "n"),
    (8, 8192, 2): _dense(256, 256, (2, 1), True, "n"),
}


def lookup(tp: int, tokens: int, gemm: int):
    """The table's ``Knobs`` for (tp, tokens, gemm), or None."""
    hit = TABLE.get((tp, tokens, gemm))
    if hit is None:
        return None
    from isoexec.ops._vendor.flashinfer_bi_moe.gemm.batch_invariant.generator import Knobs
    return Knobs(**hit)


__all__ = ["TABLE", "lookup"]
