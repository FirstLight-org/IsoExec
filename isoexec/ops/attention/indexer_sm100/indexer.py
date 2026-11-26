"""The deterministic lightning indexer of DSA (recipe ``sm100_dsa_indexer_v1``, README.md section 2).

``dsa_indexer_decode`` reads DeepGEMM's paged FP8 cache layout (``[pages, P, 1, 132]``: per page
P x 128 e4m3 then P packed ue8m0 words (the V4.1 cache writes four equal bytes: one exponent per key), P = 64 or 128 entries per page -- vLLM's ratio-1 index pages
hold 128 -- with any 16-byte multiple page stride; the block table holds one entry per P keys);
``dsa_indexer_prefill`` reads contiguous keys.  Both return
bf16 logits ``[rows, width]`` indexed by LOGICAL key position; only the window ``[ks, ke)`` of each
row is written (the entries outside it are not part of the output, and the top-k never reads them --
README "Deviations" 3).  The key span per work item, the query block and the grid are execution: the
bits of every in-window entry are a pure function of (q, k, w) at that (row, key).
"""

from __future__ import annotations

import torch

from ._extension import extension

__all__ = ["dsa_indexer_decode", "dsa_indexer_prefill", "NUM_SMS"]

NUM_SMS = 148


def _alloc_out(rows: int, width: int, device) -> torch.Tensor:
    stride = ((width + 255) // 256) * 256
    return torch.empty((rows, stride), dtype=torch.bfloat16, device=device)[:, :width]


def dsa_indexer_decode(q, cache, weights, ke, block_table, *, max_len: int | None = None, out=None,
                       span: int = 16, grid: int | None = None):
    q_fp, q_sf = q
    rows = q_fp.shape[0] * q_fp.shape[1]
    width = int(max_len) if max_len is not None else block_table.shape[1] * cache.shape[1]   # one entry per page
    if out is None:
        out = _alloc_out(rows, width, q_fp.device)
    tiles = (width + 127) // 128
    items = rows * ((tiles + span - 1) // span)
    g = min(items, NUM_SMS) if grid is None else grid
    extension("indexer").indexer_fwd(q_fp.view(torch.uint8).contiguous(), q_sf.contiguous(), cache, None,
                                     weights, None, ke, block_table, out, 1, span, max(g, 1))
    return out


def dsa_indexer_prefill(q, kv, weights, ks, ke, *, width: int | None = None, out=None, qb: int = 2,
                        span: int = 1 << 20, grid: int | None = None):
    # span: a whole row per item (measured at T = 16K: span 32 -> 2506 us, 128 -> 1399, whole -> 1398)
    q_fp, q_sf = q
    k_fp, k_sf = kv
    rows = q_fp.shape[0]
    width = k_fp.shape[0] if width is None else width
    if out is None:
        out = _alloc_out(rows, width, q_fp.device)
    tiles = (width + 127) // 128
    items = ((rows + qb - 1) // qb) * ((tiles + span - 1) // span)
    g = min(items, NUM_SMS) if grid is None else grid
    extension("indexer").indexer_fwd(q_fp.view(torch.uint8).contiguous(), q_sf.contiguous(),
                                     k_fp.view(torch.uint8).contiguous(), k_sf.contiguous(), weights, ks,
                                     ke, None, out, qb, span, max(g, 1))
    return out
