"""Float64 oracle for causal GQA over a paged or packed KV cache, plus the segment-tree structure.

``gqa_reference_fwd`` is the accuracy oracle: q [T, h_q, d] packed varlen attends keys
[0, p] where ``p = seqused_k[b] - q_len_b + i`` is the row's logical position, and returns
``(out, lse)`` in float64 with natural-log lse. It does not reproduce kernel rounding.

``segment_bounds`` and ``tree_merges`` state the *structure* of the deterministic program in
host Python: the fixed ``SEG``-token segments anchored at key 0 and the canonical adjacent-pair
tree over them. The CPU contract test and the plan builder both read these, so there is one
definition of the tree in the package.
"""

from __future__ import annotations

import torch


def _host_ints(t) -> list:
    return [int(x) for x in (t.tolist() if isinstance(t, torch.Tensor) else t)]


# --- the numerical program's structure (host side, integer only) ---------------------------------


def segment_count(border: int, seg: int) -> int:
    """Number of fixed segments a row whose causal border is ``border`` keys is made of."""
    if border <= 0:
        return 0
    return (border + seg - 1) // seg


def segment_bounds(border: int, seg: int) -> list[tuple[int, int]]:
    """The row's segments ``[k*seg, min((k+1)*seg, border))``, anchored at key 0."""
    return [(k * seg, min((k + 1) * seg, border)) for k in range(segment_count(border, seg))]


def tree_merges(n: int) -> list[tuple[int, int, int]]:
    """The canonical adjacent-pair tree over ``n`` leaves as ``(left_first, right_first, size)``.

    Level 0 pairs leaves (0,1), (2,3), ...; level l pairs the level-(l-1) nodes. A node without a
    partner is carried up verbatim and produces no merge. ``right_first - 1`` is the unique merge id
    the combine kernel keys its published scales by (mla_sm90/csrc/fwd/combine.cu uses the same id).
    """
    merges: list[tuple[int, int, int]] = []
    size = 1
    while size < n:
        first = 0
        while first + size < n:
            merges.append((first, first + size, size))
            first += 2 * size
        size *= 2
    return merges


def node_size_at(k: int, n: int, run: int) -> int:
    """Size of the node a worker publishes at leaf ``k`` when it folds aligned runs of ``run``.

    Full runs while they fit, then the remainder as descending powers of two. Every such node is a
    node of ``tree_merges(n)``; mla_sm90/csrc/fwd/params.h::node_size_at is the same function.
    """
    if k + run <= n:
        return run
    s = 1
    while 2 * s <= n - k:
        s *= 2
    return s


# --- float64 oracle -----------------------------------------------------------------------------


def _flat_kv(k_cache, v_cache, block_table, b, used, page):
    """The request's logical keys/values ``[used, h_kv, d]``, gathered in ascending key order."""
    if block_table is None:
        return k_cache[:used], v_cache[:used]
    n_blocks = (used + page - 1) // page
    pages = block_table[b, :n_blocks].long()
    k = k_cache[pages].reshape(-1, k_cache.shape[2], k_cache.shape[3])[:used]
    v = v_cache[pages].reshape(-1, v_cache.shape[2], v_cache.shape[3])[:used]
    return k, v


def gqa_reference_fwd(
    q,
    k_cache,
    v_cache,
    cu_seqlens_q,
    seqused_k,
    softmax_scale: float,
    *,
    block_table=None,
    cu_seqlens_k=None,
    chunk_rows: int = 128,
):
    """float64 ``(out [T, h_q, d], lse [T, h_q])`` of causal GQA.

    Paged: ``k_cache/v_cache`` are ``[num_blocks, page, h_kv, d]`` and ``block_table`` is
    ``[B, max_blocks]``. Packed: they are ``[T_kv, h_kv, d]`` with ``cu_seqlens_k``.
    """
    cu = _host_ints(cu_seqlens_q)
    used = _host_ints(seqused_k)
    page = k_cache.shape[1] if block_table is not None else 0
    cuk = _host_ints(cu_seqlens_k) if cu_seqlens_k is not None else None
    h_q, d = q.shape[1], q.shape[2]
    outs, lses = [], []
    for b in range(len(used)):
        start, end = cu[b], cu[b + 1]
        n_q, n_k = end - start, used[b]
        out = q.new_zeros((n_q, h_q, d), dtype=torch.float64)
        lse = q.new_full((n_q, h_q), float("-inf"), dtype=torch.float64)
        if n_q and n_k > 0:
            if block_table is not None:
                keys, values = _flat_kv(k_cache, v_cache, block_table, b, n_k, page)
            else:
                lo = cuk[b] if cuk is not None else 0
                keys, values = k_cache[lo : lo + n_k], v_cache[lo : lo + n_k]
            keys = keys.to(torch.float64)
            values = values.to(torch.float64)
            h_kv = keys.shape[1]
            rep = h_q // h_kv
            key_pos = torch.arange(n_k, device=q.device)
            positions = torch.arange(n_k - n_q, n_k, device=q.device)
            for r0 in range(0, n_q, chunk_rows):
                r1 = min(n_q, r0 + chunk_rows)
                qr = q[start + r0 : start + r1].to(torch.float64)  # [r, h_q, d]
                kk = keys.repeat_interleave(rep, dim=1)  # [n_k, h_q, d]
                vv = values.repeat_interleave(rep, dim=1)
                score = torch.einsum("rhd,khd->rhk", qr, kk) * softmax_scale
                mask = key_pos.view(1, 1, -1) > positions[r0:r1].view(-1, 1, 1)
                score = score.masked_fill(mask, float("-inf"))
                lse[r0:r1] = torch.logsumexp(score, dim=-1)
                prob = torch.softmax(score, dim=-1)
                prob = torch.where((positions[r0:r1] < 0).view(-1, 1, 1), torch.zeros_like(prob), prob)
                out[r0:r1] = torch.einsum("rhk,khv->rhv", prob, vv)
        outs.append(out)
        lses.append(lse)
    if not outs:
        return (q.new_zeros((0, h_q, d), dtype=torch.float64), q.new_zeros((0, h_q), dtype=torch.float64))
    return torch.cat(outs), torch.cat(lses)


__all__ = [
    "gqa_reference_fwd",
    "node_size_at",
    "segment_bounds",
    "segment_count",
    "tree_merges",
]
