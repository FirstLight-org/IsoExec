"""The deterministic DSA sparse attention (recipe ``sm100_dsa_sparse_segtree_v1``, README.md section 4).

``dsa_sparse_attn_decode(q, cache, block_table, lists, softmax_scale)``:
  q bf16 ``[B, 1, h_q, 512]`` (or ``[T, h_q, 512]``), the V4.1 fp8 cache ``[pages, 64, 1, 528]``,
  ``block_table`` int32 ``[B, pages_per_req]``, ``lists`` int32 ``[B, 1, K]`` of ASCENDING logical
  positions, -1 padded.  Returns (out bf16 ``[B, 1, h_q, 512]``, lse fp32 ``[B, 1, h_q]``).

A list that is not ascending over its valid prefix is REFUSED (SPEC 4.1): segment membership follows
logical position, never arrival.  ``G = plan.groups_for(h_q, K)`` fixed groups (SPEC 4.4).
"""

from __future__ import annotations

import torch

from . import plan as _plan
from ._extension import extension

__all__ = ["dsa_sparse_attn_decode", "dsa_sparse_attn", "dsa_sparse_attn_prefill", "check_ascending"]


def check_ascending(lists: torch.Tensor) -> None:
    """Refuse a list whose valid entries are not strictly ascending (pads only trailing)."""
    l2 = lists.reshape(-1, lists.shape[-1])
    valid = l2 >= 0
    bad_order = ((l2[:, 1:] <= l2[:, :-1]) & valid[:, 1:]).any()
    pad_then_valid = (valid[:, 1:] & ~valid[:, :-1]).any()
    if bool(bad_order) or bool(pad_then_valid):
        raise ValueError("the sparse attention takes ascending lists with trailing -1 pads (SPEC 4.1); "
                         "canonicalise with torch.sort first")


def dsa_sparse_attn(q, cache, block_table, lists, softmax_scale, *, bt_row=None, topk_length=None,
                    groups=None, check=False, out=None, lse=None, recipe=1):
    """Core entry over token rows: q ``[T, h_q, 512]``, lists ``[T, K]``.

    The leaf checks every list on the device (SPEC 4.1: valid entries strictly ascending, pads only
    trailing, every selected block-table entry >= 0) and poisons a refused row's out and lse with NaN;
    no host kernel or synchronisation is launched.  ``check=True`` adds the host-side refusal
    (``ValueError``, a device synchronisation) for callers that want an exception.
    ``recipe``: 1 = ``sm100_dsa_sparse_segtree_v1`` (exact running max), 2 = ``sm100_dsa_sparse_v2``
    (v1 with the lazy rescale at 8 log2 units; a different program, its own impl id).
    """
    T, h_q, _ = q.shape
    K = lists.shape[-1]
    lists = lists.reshape(T, K)
    if check:
        check_ascending(lists)
    G = _plan.groups_for(h_q if h_q in (64, 128) else 64, K) if groups is None else groups
    seg = K // G
    if out is None:
        out = torch.empty((T, h_q, 512), dtype=torch.bfloat16, device=q.device)
    if lse is None:
        lse = torch.empty((T, h_q), dtype=torch.float32, device=q.device)
    partials = None
    if G > 1:
        partials = torch.empty((T, G, h_q, 520), dtype=torch.float32, device=q.device)
    extension("attn").sparse_fwd(q, cache, block_table, bt_row, lists, topk_length, float(softmax_scale), seg,
                                 out, lse, partials, int(recipe))
    return out, lse


def dsa_sparse_attn_decode(q, cache, block_table, lists, softmax_scale, **kw):
    B = q.shape[0]
    q3 = q.reshape(B * (q.shape[1] if q.dim() == 4 else 1), q.shape[-2], q.shape[-1])
    out, lse = dsa_sparse_attn(q3, cache, block_table, lists.reshape(q3.shape[0], -1), softmax_scale, **kw)
    if q.dim() == 4:
        return out.view(q.shape), lse.view(q.shape[0], q.shape[1], q.shape[2])
    return out, lse


def dsa_sparse_attn_prefill(q, kv, lists, softmax_scale, *, block_table=None, bt_row=None, **kw):
    """Prefill rows: the SAME leaf, one CTA per (token, 64-head block), no query
    stacking.  ``q [T, h_q, 512]``; ``lists [T, K]`` or ``[T, 1, K]`` (FlashMLA's sparse layout).

    ``kv`` is either a contiguous bf16 ``[S_kv, 512]`` tensor (the list entries are its rows; the
    conversion point is the identity) or the fp8 V4.1 cache ``[pages, 64, 1, 528]`` with
    ``block_table`` (the chunked-prefill path: ``bt_row`` maps each token to its sequence's block-table
    row; omitted with a one-row table, every token reads row 0).
    """
    T = q.shape[0]
    lists = lists.reshape(T, -1)
    if kv.dtype == torch.bfloat16:
        dummy = torch.empty((1, 1), dtype=torch.int32, device=q.device)
        return dsa_sparse_attn(q, kv, dummy, lists, softmax_scale, **kw)
    if block_table is None:
        raise ValueError("an fp8 cache needs its block_table")
    if bt_row is None and block_table.shape[0] != T:
        if block_table.shape[0] != 1:
            raise ValueError("bt_row is needed when the block table has several rows")
        bt_row = torch.zeros(T, dtype=torch.int32, device=q.device)
    return dsa_sparse_attn(q, kv, block_table, lists, softmax_scale, bt_row=bt_row, **kw)
