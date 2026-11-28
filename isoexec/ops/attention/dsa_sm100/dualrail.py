"""The dual-rail sparse-attention leaf (recipes ``sm100_dsa_sparse_dualrail_v1`` / ``_sink_v1``; README "Dual-rail").

One CTA per query row x 64 heads (more heads: one launch per 64-head block; fewer: padded inside the CTA), the
whole list in list order, no split: a row's bits depend only on its own q, list and keys. Derived from FlashMLA
395cb3a's sm100 head-64 decode kernel (MIT; csrc/dualrail/PROVENANCE.md).

Execution plans (README "Dual-rail split plan"): the one-CTA plan above, and for small decode batches the SPLIT plan
(``csrc/dualrail/split.cuh``): the row's tiles' QK in parallel CTAs, then its output columns in parallel CTAs, each
running the leaf's own per-row program. Both plans give identical bits for every row (``tests/test_dualrail_split_gpu.py``);
the plan is chosen from the launch shape only and never enters the numerical program. The PERSISTENT plan
(``csrc/dualrail/persist.cuh``, FlashMLA's sm100 sparse prefill schedule: CTAs keep their state and take further rows
through cluster launch control, the next row's loads overlap the epilogue) serves many-row launches (prefill chunks,
trainer rows) with the same per-row program (``tests/test_dualrail_persist_gpu.py``).
"""

from __future__ import annotations

import os

import torch

from ._extension import extension

IMPL = "sm100_dsa_sparse_dualrail_v1"
IMPL_SINK = "sm100_dsa_sparse_dualrail_sink_v1"
QUANT_BYTES_FP8 = 528

#: The split plan runs for at most this many rows (B200, TP4 rank and TP1: K = 640 rows 1-8 20.4 -> 13.6 us,
#: 16 rows 21.1 -> 16.7, 32 rows 21.1 -> 19.9, 48 rows 21.4 -> 23.2; K = 128 rows 1-16 9.0-9.3 -> 8.2-8.9, 32 rows a loss).
#: Execution only: never a bit.
SPLIT_MAX_ROWS = 32
SPLIT_MAX_ROWS_SHORT = 16       # lists of <= 2 tiles (window-only layers)
#: 32 heads per CTA row (TP2 rank of 64 heads): in-situ decode layer graphs on B200 (K = 640): the split plan
#: wins up to 24 rows (M 1/8/16/24: -7.2/-5.7/-3.6/-2.4 us per layer), the one-CTA
#: plan from 32 rows (-2.0 / -3.3 us at C 4K / 45K). Execution only: never a bit (both plans compared in situ).
SPLIT_MAX_ROWS_H32 = 24
_SMS = 148
#: The widest per-row list this leaf serves: V4.1's 128 window + 512 compressed slots. The single chain over a row's
#: whole list is the declared program up to this width only; wider lists (GLM-class top-k) are the segmented leaf +
#: merge program, a different numerical program, and are refused by name here.
MAX_LIST_SLOTS = 640
#: The persistent plan runs from this many rows where it is admissible (``ISOEXEC_DUALRAIL_PLAN=persist`` / ``one`` force
#: a plan). B200, fp8 window + compressed lists (attn lane runs/r10-2): decode B 64 / 128 / 256 one-CTA 24.9 / 26.7 /
#: 52.1 us vs persistent 21.7 / 23.2 / 42.4 (H 64; H 8 23.9 / 24.5 / 47.3 vs 21.2 / 22.0 / 39.4); prefill T 1024-8192
#: 0.73-0.80 of the one-CTA plan. Below 64 rows the split / one-CTA plans (not measured against it). Execution only.
PERSIST_MIN_ROWS = 64


def persist_admissible(lists, topk_length, rope_q, rope_o, extra_lists=None):
    """The persistent plan's launch contract: every row has >= 2 tiles (K >= 128, no ``topk_length``), list rows 16-byte
    aligned (the fused rope is served on fp8 launches, as by the one-CTA plan)."""
    def aligned(t):
        return t.stride(0) % 4 == 0 and t.data_ptr() % 16 == 0
    return (lists.shape[1] >= 128 and topk_length is None and aligned(lists)
            and (extra_lists is None or aligned(extra_lists)))


def split_plan(rows: int, tiles: int, heads: int, fp8: bool):
    """``(tpc, nc)`` of the split plan for this launch shape (rows, list tiles), or None for the one-CTA plan.

    ``ISOEXEC_DUALRAIL_PLAN``: ``"one"`` forces the one-CTA plan, ``"tpc,nc"`` forces a split plan (tests/benchmarks)."""
    force = os.environ.get("ISOEXEC_DUALRAIL_PLAN")
    if not fp8 or heads > 64 or rows == 0:
        return None
    if force:
        if force == "one":
            return None
        tpc, nc = (int(v) for v in force.split(","))
        return min(max(tpc, 1), tiles), nc
    long_max = SPLIT_MAX_ROWS_H32 if heads == 32 else SPLIT_MAX_ROWS
    if rows > (SPLIT_MAX_ROWS_SHORT if tiles <= 2 else long_max):
        return None
    tpc = min(tiles, -(-rows * tiles // _SMS))
    return tpc, (8 if tiles <= 2 else 4)


def dualrail_attn(q, kv, lists, softmax_scale, *, sink=None, topk_length=None, extra_kv=None, extra_lists=None,
                  extra_topk_length=None, out=None, lse=None, rope=None, rope_q=False, rope_o=False):
    """``(out bf16 [R, H, 512], lse fp32 [R, H])`` (natural log, sink-inclusive; empty row: out 0, lse = sink or +inf).

    ``q`` bf16 [R, H, 512]; ``kv`` bf16 [S, 512] or the V4.1 fp8 cache (uint8, 528-byte tokens, any row count,
    any shape); ``lists`` int32 [R, K] token rows of ``kv`` (-1 pads, K % 64 == 0); ``extra_kv`` a second
    V4.1 fp8 cache (the compressed KV, 528-byte tokens, any row count) with ``extra_lists`` (fp8 ``kv`` only); ``sink``
    fp32 [H] (natural units).

    ``rope = (cos, sin, pos)`` (fp32 [P, 32] tables, int32 [R] positions) with ``rope_q`` / ``rope_o``: the query's
    RoPE tail rotated inside the leaf before the attention (bit for bit ``deepseek_v41_cache_sm100.ops.q_rope``) and the
    output's inverse RoPE tail after the leaf's bf16 rounding (bit for bit the row op ``ROW_OUT_INV_ROPE``), fp8 cache
    launches only. ``q`` itself is not modified."""
    r, h = q.shape[:2]
    width = lists.shape[1] + (0 if extra_lists is None else extra_lists.shape[1])
    if width > MAX_LIST_SLOTS:
        raise ValueError(f"dual-rail leaf ({IMPL}): a row list of {width} slots exceeds the declared single-chain width "
                         f"{MAX_LIST_SLOTS} (V4.1: 128 window + 512 compressed); wider lists need the segmented leaf + "
                         f"merge program")
    out = torch.empty((r, h, 512), dtype=torch.bfloat16, device=q.device) if out is None else out
    lse = torch.empty((r, h), dtype=torch.float32, device=q.device) if lse is None else lse
    if r == 0:
        return out, lse
    kvc = kv if kv.dtype == torch.bfloat16 else kv.reshape(-1)
    ex = None if extra_kv is None else extra_kv.reshape(-1)
    s = None if sink is None else sink.detach().float().contiguous()
    tiles = lists.shape[1] // 64 + (0 if extra_lists is None else extra_lists.shape[1] // 64)
    force = os.environ.get("ISOEXEC_DUALRAIL_PLAN")
    persist = persist_admissible(lists, topk_length, rope_q, rope_o, extra_lists) and (
        force == "persist" or (force is None and PERSIST_MIN_ROWS is not None and r >= PERSIST_MIN_ROWS))
    plan = None if (persist or force == "persist") else split_plan(r, tiles, h, kv.dtype != torch.bfloat16)
    rc = rs = rpos = None
    if rope_q or rope_o:
        rc, rs, rpos = rope
        rpos = rpos if (rpos.dtype == torch.int32 and rpos.is_contiguous()) else rpos.to(torch.int32).contiguous()
    with torch.cuda.device(q.device):
        if persist:
            extension("dualrail").dualrail_persist_fwd(q, kvc, lists, ex, extra_lists, extra_topk_length, s,
                                                       float(softmax_scale), out, lse, rc, rs, rpos, bool(rope_q),
                                                       bool(rope_o))
        elif plan is None:
            extension("dualrail").dualrail_fwd(q, kvc, lists, topk_length, ex, extra_lists, extra_topk_length, s,
                                               float(softmax_scale), out, lse, rc, rs, rpos, bool(rope_q), bool(rope_o))
        else:
            p_scr = torch.empty((r, tiles, 4096), dtype=torch.float32, device=q.device)
            v_scr = torch.empty((r, tiles, 65536), dtype=torch.uint8, device=q.device)
            extension("dualrail").dualrail_split_fwd(q, kvc, lists, topk_length, ex, extra_lists, extra_topk_length, s,
                                                     float(softmax_scale), out, lse, p_scr, v_scr, plan[0], plan[1],
                                                     rc, rs, rpos, bool(rope_q), bool(rope_o))
    return out, lse


__all__ = ["IMPL", "IMPL_SINK", "MAX_LIST_SLOTS", "PERSIST_MIN_ROWS", "dualrail_attn", "persist_admissible", "split_plan",
           "SPLIT_MAX_ROWS"]
