"""Torch transcriptions of the parts of the sm100 MoE program that are exactly expressible."""

from __future__ import annotations

import torch


def canonical_plan(indices: torch.Tensor, num_experts: int):
    """The dispatch plan as a stable counting sort on the flat ``(token, slot)`` index."""
    T, k = indices.shape
    flat = indices.reshape(-1).to(torch.int64)
    counts = torch.bincount(flat, minlength=num_experts).to(torch.int32)
    offsets = torch.zeros(num_experts + 1, dtype=torch.int32, device=indices.device)
    offsets[1:] = torch.cumsum(counts.to(torch.int64), 0).to(torch.int32)
    # A stable sort by expert over ascending flat index IS the canonical order.
    order = torch.argsort(flat, stable=True)
    inverse = order.to(torch.int32)
    restore = torch.empty_like(inverse)
    restore[order] = torch.arange(flat.numel(), dtype=torch.int32, device=indices.device)
    # Canonical slot: rank of the route's expert inside the token's selected set.
    slot = torch.argsort(torch.argsort(indices.to(torch.int64), dim=1, stable=True), dim=1)
    combine_rows = torch.empty_like(restore.view(T, k))
    combine_rows.scatter_(1, slot, restore.view(T, k))
    return counts, offsets, inverse, restore.view(T, k), combine_rows


def tile_plan(offsets: torch.Tensor, block_m: int):
    """``(expert, first, end)`` rows, expert-major, ``block_m`` rows per tile."""
    rows = []
    off = offsets.tolist()
    for e in range(len(off) - 1):
        start, end = off[e], off[e + 1]
        for first in range(start, end, block_m):
            rows.append((e, first, end))
    return rows


def segment_tree(leaf_roots) -> torch.Tensor:
    """The balanced tree over already-rounded fp32 leaves, lower-index subtrees left."""
    node = list(leaf_roots)
    while len(node) > 1:
        node = [node[i] + node[i + 1] for i in range(0, len(node), 2)]
    return node[0]


def glu_epilogue_torch(gate_bf: torch.Tensor, up_bf: torch.Tensor, probs=None) -> torch.Tensor:
    """A readable transcription of the FC1 epilogue -- documentation, NOT an oracle."""
    dtype = gate_bf.dtype
    g = gate_bf.float()
    h = (g / (1.0 + torch.exp(-g))).to(dtype).float()
    h = (h * up_bf.float()).to(dtype).float()
    if probs is not None:
        h = (h * probs[:, None]).to(dtype).float()
    return h.to(dtype)


def combine_reference(rows_fp32: torch.Tensor, combine_rows: torch.Tensor, dtype) -> torch.Tensor:
    """One fp32 chain over the k slots in canonical slot order, seeded with +0 (schedule policy rule 2),
    then one conversion."""
    acc = torch.zeros_like(rows_fp32.index_select(0, combine_rows[:, 0].long()))
    for j in range(combine_rows.shape[1]):
        acc = acc + rows_fp32.index_select(0, combine_rows[:, j].long())
    return acc.to(dtype)


def selected_sum_reference(terms: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """The selected-set sum of ``route_mechanics`` (schedule policy rule 2): ``terms [T, k]`` in selection
    order, their experts ``indices [T, k]``; one fp32 chain in ASCENDING EXPERT ID seeded with +0."""
    order = torch.argsort(indices.to(torch.int64), dim=1, stable=True)
    by_id = torch.gather(terms, 1, order)
    acc = torch.zeros_like(by_id[:, :1])
    for j in range(by_id.shape[1]):
        acc = acc + by_id[:, j : j + 1]
    return acc


def biteq(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Integer-view equality: a tolerance would not be a proof of anything here."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    view = {
        torch.float32: torch.int32,
        torch.float64: torch.int64,
        torch.bfloat16: torch.int16,
        torch.float16: torch.int16,
    }.get(a.dtype)
    if view is None:
        return bool(torch.equal(a, b))
    return bool(torch.equal(a.contiguous().view(view), b.contiguous().view(view)))
