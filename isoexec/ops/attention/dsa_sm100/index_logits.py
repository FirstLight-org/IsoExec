"""Index scores on DeepGEMM's SM100 MQA logits kernels (recipe ``sm100_dsa_index_logits_dg_v1``).

The kernels are the vendored DeepGEMM ones (``isoexec/ops/_vendor/deep_gemm``, PROVENANCE.md there), compiled ahead of
time into the ``mqa`` sub-extension (``csrc/mqa``): no JIT and no DeepGEMM runtime. Per (query row t, key j) the
logit is upstream's program: the block-scaled FP8 MMA (FP32 accumulator over the 128 dims, UE8M0 scales of Q and K
in the MMA), per head ``bf16(relu(d))``, four BF16 ``fma.rn`` chains over the heads ``4i + c`` in ascending ``i``,
``(c0 + c2, c1 + c3)``, then low + high: one BF16 logit, computed and stored by one thread. The schedule (which CTA
computes which keys) assigns whole keys and changes no bit.

Queries are FP8 codes ``[rows, H, 128]`` with one power-of-two scale per (row, head) carried as four equal UE8M0
bytes in ``q_sf`` (H = 32 or 64); weights BF16 ``[rows, H]``. Keys: the paged index-key pool
``[pages, P, 1, 132]`` (P = 64 or 128) or contiguous codes ``[keys, 128]`` + scale words ``[keys]``.
Row ``r`` gets the logits of keys ``[0, ke[r])`` (``[ks[r], ke[r])`` contiguous, at column ``j - ks[r]``); the rest of
the row is not written.

``index_logits_gathered`` scores paged keys with the contiguous kernel, as native prefill does: the keys of the rows'
sequences are packed contiguous (``gather_keys``, byte copies) and each row gets its sequence's range
(``key_ranges``), so the logits come out at the same columns with the same bits as ``index_logits_paged``; which
form runs is an execution choice.
"""

from __future__ import annotations

import torch

from ._extension import extension

__all__ = ["paged_schedule", "index_logits_paged", "index_logits", "alloc_logits", "key_ranges", "gather_keys",
           "index_logits_gathered", "gather_capacity", "NUM_SMS"]

#: CTAs of every launch (csrc/mqa/mqa.h): the schedule's division of work, the same on every device.
NUM_SMS = 148
#: sequences (block-table rows) of one ``key_ranges`` launch (csrc/mqa/bindings.cpp MAX_GATHER_SEQS)
MAX_GATHER_SEQS = 8192


def _ext():
    return extension("mqa")


def alloc_logits(rows: int, width: int, device, *, heads: int = 32) -> torch.Tensor:
    """BF16 ``[rows, width]`` with a 256-element row stride; storage rows rounded up to a whole Q block (the
    contiguous kernel writes its last block's spare rows)."""
    stride = ((width + 255) // 256) * 256
    bq = 128 // heads
    return torch.empty(((rows + bq - 1) // bq * bq, stride), dtype=torch.bfloat16, device=device)[:rows, :width]


def paged_schedule(ke: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """The paged work division for these rows (``[NUM_SMS + 1, 2]`` int32): a pure function of ``ke`` and
    ``indices``, shared by every layer that scores the same rows (one launch per step and compress ratio)."""
    meta = torch.empty((NUM_SMS + 1, 2), dtype=torch.int32, device=ke.device)
    _ext().paged_schedule(ke, indices, meta)
    return meta


def index_logits_paged(q, q_sf, w, cache, ke, block_table, indices, *, width: int, schedule=None, out=None):
    """Index scores of ``rows`` query rows over paged index keys. ``indices[r]``: the block-table row of query row r
    (rows of one sequence adjacent and with equal index share their key loads). ``schedule``: ``paged_schedule(ke,
    indices)`` (computed here when None)."""
    if schedule is None:
        schedule = paged_schedule(ke, indices)
    if out is None:
        out = alloc_logits(q.shape[0], width, q.device, heads=q.shape[1])
    _ext().paged_fwd(q.view(torch.uint8), q_sf, cache, w, ke, block_table, indices, schedule, out)
    return out


def index_logits(q, q_sf, w, k, k_sf, ke, *, width: int, ks=None, out=None):
    """Index scores of ``rows`` query rows over contiguous index keys ``k [keys, 128]`` / ``k_sf [keys]``."""
    if out is None:
        out = alloc_logits(q.shape[0], width, q.device, heads=q.shape[1])
    _ext().fwd(q.view(torch.uint8), q_sf, k.view(torch.uint8), k_sf, w, ks, ke, out)
    return out


def gather_capacity(rows: int, seqs: int, width: int) -> int:
    """Keys the packed buffer of ``index_logits_gathered`` holds: at most ``min(rows, seqs)`` sequences have a
    row, each with at most ``width`` keys (a launch-shape bound, so the buffer has a static shape)."""
    return min(rows, seqs) * width


def key_ranges(ke, rows_seq, seqs: int, width: int):
    """Per row r the range ``[ks[r], ke_out[r])`` of its sequence ``rows_seq[r]`` (in ``[0, seqs)``) in the packed
    buffer, and the sequences' start offsets ``[seqs + 1]``: sequence s holds its first ``max ke`` keys (of its
    rows) at ``offsets[s]``. One launch per step and compress ratio, shared by the index layers of that ratio."""
    rows = ke.numel()
    ks = torch.empty(rows, dtype=torch.int32, device=ke.device)
    ke_out = torch.empty_like(ks)
    offsets = torch.empty(seqs + 1, dtype=torch.int32, device=ke.device)
    _ext().key_ranges(ke, rows_seq, width, ks, ke_out, offsets)
    return ks, ke_out, offsets


def gather_keys(cache, block_table, offsets, capacity: int):
    """The packed keys of ``key_ranges``: codes ``[capacity, 128]`` uint8 and scale words ``[capacity]`` read from
    the paged pool through block-table row s for sequence s (keys past the packed total are not written)."""
    codes = torch.empty((capacity, 128), dtype=torch.uint8, device=cache.device)
    sf = torch.empty(capacity, dtype=torch.int32, device=cache.device)
    _ext().gather_keys(cache, block_table, offsets, codes, sf)
    return codes, sf


def index_logits_gathered(q, q_sf, w, cache, ke, block_table, rows_seq, *, width: int, ranges=None, out=None):
    """``index_logits_paged`` (``indices = rows_seq``, one block-table row per sequence) on the contiguous kernel:
    the rows' sequences' keys packed by ``gather_keys`` and each row scored over its own range, at column ``j`` for
    key ``j`` of its sequence. ``ranges``: ``key_ranges(ke, rows_seq, block_table.shape[0], width)`` of the step
    (computed here when None)."""
    seqs = block_table.shape[0]
    if ranges is None:
        ranges = key_ranges(ke, rows_seq, seqs, width)
    ks, ke_out, offsets = ranges
    k, k_sf = gather_keys(cache, block_table, offsets, gather_capacity(q.shape[0], seqs, width))
    return index_logits(q, q_sf, w, k, k_sf, ke_out, width=width, ks=ks, out=out)
