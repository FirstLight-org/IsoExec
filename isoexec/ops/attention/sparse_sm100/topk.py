"""The deterministic top-k of DSA (recipe ``sm100_dsa_topk_v1``, README.md section 3).

``dsa_topk(scores, ke, ks=None)`` returns int32 ``[rows, K]``: the K largest scores of each row's
window ``[ks, ke)`` under the total order (key desc, index asc), NaN lowest, as ASCENDING logical
positions, -1 padded.  The output is a pure function of (the scores inside the window, ks, ke, K):
the execution plan (``cluster`` CTAs per row, threads, stash size) never changes a bit, which the
battery checks (R-T8).  ``cluster`` defaults to the table rule: 1 CTA per row up to
``TOPK_ROW_SPLIT`` keys, 2 above it.
"""

from __future__ import annotations

import torch

from . import spec
from ._extension import extension

__all__ = ["dsa_topk", "default_plan"]

#: SMEM per SM available to CTAs on a B200 (228 KB) and the per-CTA reservation.
_SM_SMEM = 228 * 1024
_CTA_RESERVED = 1024
_OCC_MARGIN = 4096
#: SMs of a B200.
_SMS = 148
#: CTAs per SM the kernel's launch bounds target, by thread count.
_CTAS_PER_SM = {256: 4, 512: 2, 1024: 1}


def _fixed_smem(threads: int, cluster: int) -> int:
    nb = 4096
    nw = threads // 32
    return ((2 if cluster > 1 else 1) * nb + nw * 16 + 32 + 64 + nw * 64) * 4


def default_plan(max_len: int, dtype=torch.bfloat16) -> dict:
    """The execution plan the table picks for a launch whose widest row has ``max_len`` keys.

    Execution only (every plan returns the same list, R-T8): 256 threads (4 CTAs per SM) for rows up
    to 12K keys, where the per-row fixed cost dominates; 512 (2 per SM) above; 2 CTAs per row above
    TOPK_ROW_SPLIT.  The key stash takes whatever SMEM the occupancy target leaves after the
    histograms, the chunk queues and the per-chunk max summary."""
    threads = 256 if max_len <= 12288 else 512
    cluster = 1 if max_len <= spec.TOPK_ROW_SPLIT else 2
    per_cta = (max_len + cluster - 1) // cluster
    chunks = (max_len + 7) // 8 + 1
    cmax_bytes = (((chunks + cluster - 1) // cluster + 8 + 7) // 8 * 8) * 2
    # _OCC_MARGIN: the launch's own static shared memory and alignment, so the occupancy target really holds
    # (at a 64K-key width the stash otherwise filled the budget exactly and left 1 CTA per SM). Execution only.
    budget = (_SM_SMEM // _CTAS_PER_SM[threads] - _CTA_RESERVED - _fixed_smem(threads, cluster) - cmax_bytes
              - _OCC_MARGIN)
    key_bytes = 2 if dtype == torch.bfloat16 else 4
    stash = max(0, min(((per_cta + 15) // 8) * 8 + 8, budget // key_bytes // 8 * 8))
    if dtype != torch.bfloat16:
        threads = 512 if threads == 256 else threads
        stash = min(stash, 32768)
    return {"cluster": cluster, "threads": threads, "stash_keys": stash, "algo": 0}


#: Two-band execution (0: off): when the captured row width exceeds this many keys and its plan is a CTA cluster, rows whose live window is
#: shorter run in a launch planned for this length, the others in the launch planned for the width. Both
#: launches always run (graph-static); each writes only its band's rows. Execution only: the list of a row is
#: the same under every plan (``test_topk_two_band_equals_single_launch``).
SMALL_BAND = 32768
#: How the short rows of a cluster-planned launch run: "solo" (ONE launch of the cluster kernel; a short row runs
#: on the cluster's rank 0 alone as the 1-CTA plan, with no cluster barrier) or "two_launch" (a second launch
#: planned for SMALL_BAND). Execution only.
BAND_MODE = "solo"


def dsa_topk(scores: torch.Tensor, ke: torch.Tensor, ks: torch.Tensor | None = None,
             k: int = spec.TOPK, out: torch.Tensor | None = None, *, max_len: int | None = None,
             cluster: int | None = None, threads: int | None = None,
             stash_keys: int | None = None, algo: int | None = None, band: int | None = None,
             rows_out: torch.Tensor | None = None, rows_bt: torch.Tensor | None = None,
             rows_seq: torch.Tensor | None = None, rows_per: int = 64) -> torch.Tensor:
    """``max_len`` bounds the longest window (defaults to the row width, which needs no host sync).
    ``band``: the two-band split length (default ``SMALL_BAND``; 0 off).
    ``rows_out`` (int32 [rows, k], bf16 scores): also written by the emission, each selected id j as the cache row
    ``rows_bt[rows_seq[r], j // 64] * rows_per + j % 64`` (-1 pads) -- the paged attention extra list."""
    if scores.dim() != 2:
        raise ValueError("scores must be [rows, width]")
    rows = scores.shape[0]
    if out is None:
        out = torch.empty((rows, k), dtype=torch.int32, device=scores.device)
    plan = default_plan(scores.shape[1] if max_len is None else max_len, scores.dtype)
    if cluster is not None:
        plan["cluster"] = cluster
    if threads is not None:
        plan["threads"] = threads
    if stash_keys is not None:
        plan["stash_keys"] = stash_keys
    if algo is not None:
        plan["algo"] = algo
    ext = extension("topk")
    ke32 = ke.to(torch.int32).contiguous()
    band = SMALL_BAND if band is None else band
    width = scores.shape[1] if max_len is None else max_len
    if (plan["cluster"] > 1 and cluster is None and threads is None and stash_keys is None
            and rows * plan["cluster"] > _SMS and width <= 2 * spec.TOPK_ROW_SPLIT):
        # more rows than the clusters can spread over the SMs: the cluster buys no parallelism, so the 1-CTA plan
        # (its chunk summary still fits at this width). Execution only.
        plan = dict(default_plan(spec.TOPK_ROW_SPLIT, scores.dtype), algo=plan["algo"])
    rows_args = (rows_out, rows_bt, None if rows_seq is None else rows_seq.to(torch.int32).contiguous(), rows_per)
    if rows_out is not None and (scores.dtype != torch.bfloat16 or plan["algo"] != 0 or not hasattr(ext, "topk_rows_fwd")):
        raise ValueError("the rows output needs the bf16 kernel (algo 0) of an extension that has it")
    tuned = cluster is None and threads is None and stash_keys is None
    if (band and width > band and tuned and plan["cluster"] > 1 and default_plan(band, scores.dtype)["cluster"] == 1
            and BAND_MODE == "solo" and scores.dtype == torch.bfloat16 and plan["algo"] == 0
            and hasattr(ext, "topk_rows_fwd")):
        ext.topk_rows_fwd(scores, ks, ke32, out, k, plan["cluster"], plan["threads"], plan["stash_keys"], 0,
                          0, 2 ** 31 - 1, 0, band, *rows_args)
        return out
    if rows_out is not None:
        ext.topk_rows_fwd(scores, ks, ke32, out, k, plan["cluster"], plan["threads"], plan["stash_keys"], 0,
                          0, 2 ** 31 - 1, 0, 0, *rows_args)
        return out
    if (band and width > band and tuned and hasattr(ext, "topk_band_fwd")):
        small = default_plan(band, scores.dtype)
        if plan["cluster"] > 1 and small["cluster"] == 1:          # only the cluster plan is worth a second launch
            ext.topk_band_fwd(scores, ks, ke32, out, k, small["cluster"], small["threads"], small["stash_keys"],
                              small["algo"] if algo is None else algo, 0, band, band)
            ext.topk_band_fwd(scores, ks, ke32, out, k, plan["cluster"], plan["threads"], plan["stash_keys"],
                              plan["algo"], band, 2 ** 31 - 1, 0)
            return out
    ext.topk_fwd(scores, ks, ke32, out, k, plan["cluster"], plan["threads"], plan["stash_keys"], plan["algo"])
    return out
