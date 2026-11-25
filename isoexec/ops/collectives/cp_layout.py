"""Map zigzag context-parallel token slices to their permitted attention prefixes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


class CpLayoutError(ValueError):
    """The packed boundaries cannot be cut into ``2 * cp`` equal chunks per sequence."""


@dataclass(frozen=True)
class PrefixEntry:
    """One varlen attention entry of a context-parallel rank."""

    sequence: int
    chunk: int
    q_start: int
    q_len: int
    kv_start: int
    kv_len: int


def _chunk_sizes(cu_seqlens: Sequence[int], cp: int) -> list[int]:
    if cp < 1:
        raise CpLayoutError(f"context-parallel degree must be >= 1, got {cp}")
    cu = [int(v) for v in cu_seqlens]
    if not cu or cu[0] != 0 or any(b < a for a, b in zip(cu, cu[1:])):
        raise CpLayoutError(f"cu_seqlens must start at 0 and be non-decreasing, got {cu[:8]}...")
    sizes = []
    for index, (a, b) in enumerate(zip(cu, cu[1:])):
        length = b - a
        if length == 0 or length % (2 * cp) != 0:
            raise CpLayoutError(
                f"sequence {index} has padded length {length}, which is not a positive multiple of 2*cp={2 * cp}; "
                "the worker pads every packed sequence to tp*cp*2 (packing_utils.get_packed_seq_align_size)"
            )
        sizes.append(length // (2 * cp))
    return sizes


def local_chunks(cp: int, rank: int) -> tuple[int, int]:
    """The two chunk ids rank ``rank`` holds, in the order they sit in its local stream."""
    if not 0 <= rank < cp:
        raise CpLayoutError(f"context-parallel rank {rank} outside [0, {cp})")
    return rank, 2 * cp - 1 - rank


def local_to_global(cu_seqlens: Sequence[int], cp: int, rank: int) -> list[int]:
    """Global packed index of every local token of ``rank``, in local order."""
    sizes = _chunk_sizes(cu_seqlens, cp)
    out: list[int] = []
    for (start, size) in zip(cu_seqlens, sizes):
        for chunk in local_chunks(cp, rank):
            first = int(start) + chunk * size
            out.extend(range(first, first + size))
    return out


def local_positions(cu_seqlens: Sequence[int], cp: int, rank: int) -> list[int]:
    """Position of every local token INSIDE its own sequence (what the rope angle table is read at)."""
    sizes = _chunk_sizes(cu_seqlens, cp)
    out: list[int] = []
    for size in sizes:
        for chunk in local_chunks(cp, rank):
            out.extend(range(chunk * size, (chunk + 1) * size))
    return out


def gathered_to_global(cu_seqlens: Sequence[int], cp: int) -> list[int]:
    """``perm`` with ``global_stream[g] = gathered[perm[g]]``, where ``gathered`` is the rank-ordered
    concatenation of the ``cp`` local streams (what ``all_gather_into_tensor`` produces)."""
    total = int(cu_seqlens[-1])
    local = total // cp
    perm = [-1] * total
    for rank in range(cp):
        for offset, g in enumerate(local_to_global(cu_seqlens, cp, rank)):
            perm[g] = rank * local + offset
    if any(p < 0 for p in perm):
        raise CpLayoutError("the zigzag chunks of the ranks do not tile the packed stream")
    return perm


def owner_and_local_index(cu_seqlens: Sequence[int], cp: int) -> tuple[list[int], list[int]]:
    """For every GLOBAL packed index: the rank that holds it and its index in that rank's stream.
    The same map SkyRL's ``model_utils._packed_cp_rank_and_local_indices`` computes on tensors."""
    total = int(cu_seqlens[-1])
    owner, index = [-1] * total, [-1] * total
    for rank in range(cp):
        for offset, g in enumerate(local_to_global(cu_seqlens, cp, rank)):
            owner[g], index[g] = rank, offset
    return owner, index


def prefix_entries(cu_seqlens: Sequence[int], cp: int, rank: int) -> list[PrefixEntry]:
    """The varlen entries of ``rank``: per sequence, its two chunks in local order, each attending
    the sequence prefix that ends where the chunk ends. With the kernel's bottom-right causal
    alignment, query ``i`` of a chunk sees keys ``<= chunk_start + i`` -- the global causal mask."""
    sizes = _chunk_sizes(cu_seqlens, cp)
    entries: list[PrefixEntry] = []
    q_start = 0
    for sequence, (start, size) in enumerate(zip(cu_seqlens, sizes)):
        for chunk in local_chunks(cp, rank):
            entries.append(
                PrefixEntry(
                    sequence=sequence,
                    chunk=chunk,
                    q_start=q_start,
                    q_len=size,
                    kv_start=int(start),
                    kv_len=(chunk + 1) * size,
                )
            )
            q_start += size
    return entries


__all__ = [
    "CpLayoutError",
    "PrefixEntry",
    "gathered_to_global",
    "local_chunks",
    "local_positions",
    "local_to_global",
    "owner_and_local_index",
    "prefix_entries",
]
