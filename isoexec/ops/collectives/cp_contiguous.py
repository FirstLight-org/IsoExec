"""Contiguous context-parallel layout of a packed token stream: who owns which rows, and what moves.

Rank ``r`` of ``cp`` owns rows ``[r*L, (r+1)*L)`` of the padded global packed stream (``L = T/cp``;
verl's ``cp_layout="contiguous"``). Sequence boundaries fall anywhere, so a sequence may span
several ranks. :class:`ContiguousPlan` derives everything a rank needs from the global boundaries
alone (host ints), so every rank derives the same plan and every point-to-point message is matched
by construction:

* the local pieces (one per sequence that intersects the shard) and their in-sequence offsets;
* attention: the key/value rows of the first local sequence that lower ranks hold (the *prefix*),
  which rank sends which rows to whom, and the varlen boundaries of the local queries against
  ``[prefix | local]`` keys. Queries are right-aligned to their key prefix, which is the chunked
  prefill case of the causal varlen kernel: each query row sees exactly the keys it sees at CP=1,
  with key segments anchored at its sequence's first key;
* recurrent state: whether the first piece continues a sequence from ``rank - 1`` and whether the
  last piece continues into ``rank + 1`` (the GDN state / conv-tail links);
* conv halo: how many pre-conv rows of the continuing sequence cross each link for a causal
  convolution of a given width.

Nothing here imports torch at module scope. Device tensors are built lazily, once per plan and
device, by :meth:`ContiguousPlan.device_tensors`.
"""

from __future__ import annotations

import bisect
import weakref
from dataclasses import dataclass, field
from typing import Any, Sequence


class CpPlanError(ValueError):
    """The packed boundaries cannot be cut into ``cp`` contiguous shards."""


@dataclass(frozen=True)
class Piece:
    """The part of one sequence a rank holds."""

    sequence: int  #: index of the sequence in the global boundaries
    start: int  #: first global packed row (inclusive)
    stop: int  #: last global packed row (exclusive)
    offset: int  #: position of ``start`` inside its sequence

    @property
    def length(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class KvTransfer:
    """Key/value rows ``src`` sends to ``dst`` so ``dst``'s first sequence sees its whole prefix."""

    src: int
    dst: int
    src_start: int  #: first local row on ``src`` (rows ``[src_start, L)`` are sent)
    rows: int
    dst_start: int  #: row of the receiver's prefix buffer where they land


def _check(cu: Sequence[int], cp: int, rank: int) -> tuple[list[int], int]:
    if type(cp) is not int or cp < 1:
        raise CpPlanError(f"context-parallel degree must be an int >= 1, got {cp!r}")
    if type(rank) is not int or not 0 <= rank < cp:
        raise CpPlanError(f"context-parallel rank {rank!r} outside [0, {cp})")
    bounds = [int(v) for v in cu]
    if len(bounds) < 2 or bounds[0] != 0 or any(b < a for a, b in zip(bounds, bounds[1:])):
        raise CpPlanError(f"cu_seqlens must start at 0 and be non-decreasing, got {bounds[:8]}...")
    total = bounds[-1]
    if total == 0 or total % cp:
        raise CpPlanError(
            f"the packed stream has {total} rows, which is not a positive multiple of cp={cp}; the host pads "
            "each sequence to a multiple of tp*cp (verl's preprocess_thd_engine) so every rank holds total/cp rows"
        )
    return bounds, total // cp


def _sequence_at(bounds: list[int], row: int) -> int:
    """Index of the non-empty sequence holding global ``row``."""
    return bisect.bisect_right(bounds, row) - 1


def _interior(bounds: list[int], row: int) -> int | None:
    """The sequence that ``row`` (a shard boundary) cuts, or ``None`` when a sequence starts there."""
    if row <= 0 or row >= bounds[-1]:
        return None
    n = _sequence_at(bounds, row)
    return n if bounds[n] < row else None


def kv_transfers(cu: Sequence[int], cp: int) -> tuple[KvTransfer, ...]:
    """Every attention prefix transfer of the layout, ordered by (dst, src)."""
    bounds, local = _check(cu, cp, 0)
    out: list[KvTransfer] = []
    for dst in range(1, cp):
        lo = dst * local
        n = _interior(bounds, lo)
        if n is None:
            continue
        first = bounds[n]
        filled = 0
        for src in range(first // local, dst):
            start = max(first, src * local)
            rows = (src + 1) * local - start
            out.append(KvTransfer(src=src, dst=dst, src_start=start - src * local, rows=rows, dst_start=filled))
            filled += rows
    return tuple(out)


def state_link(cu: Sequence[int], cp: int, boundary_rank: int) -> int | None:
    """The sequence cut at the shard boundary between ``boundary_rank - 1`` and ``boundary_rank``."""
    bounds, local = _check(cu, cp, 0)
    if not 0 < boundary_rank < cp:
        return None
    return _interior(bounds, boundary_rank * local)


@dataclass
class ContiguousPlan:
    """One microbatch's contiguous CP layout as seen by one rank. Build with :meth:`build`."""

    cp: int
    rank: int
    total: int  #: global padded packed rows
    local: int  #: rows this rank holds (``total // cp``)
    bounds: tuple[int, ...]  #: the global boundaries
    pieces: tuple[Piece, ...]
    local_cu: tuple[int, ...]  #: boundaries of the local pieces over the local rows
    recv_prev: bool  #: the first piece continues a sequence from ``rank - 1``
    send_next: bool  #: the last piece continues into ``rank + 1``
    kv_recv: tuple[KvTransfer, ...]  #: prefix rows this rank receives, ascending source rank
    kv_send: tuple[KvTransfer, ...]  #: rows this rank sends, ascending destination rank
    key_cu: tuple[int, ...]  #: key boundaries over ``[prefix | local]``
    max_q: int  #: longest local piece
    max_k: int  #: longest key range of a local piece
    #: Received tensors kept for a checkpoint region's recompute (see ``cp_transport``), by key.
    stash: dict = field(default_factory=dict, repr=False, compare=False)
    _device: dict = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def build(cls, cu: Sequence[int], cp: int, rank: int) -> "ContiguousPlan":
        bounds, local = _check(cu, cp, rank)
        lo, hi = rank * local, (rank + 1) * local
        pieces: list[Piece] = []
        n = _sequence_at(bounds, lo)
        while n < len(bounds) - 1 and bounds[n] < hi:
            start, stop = max(bounds[n], lo), min(bounds[n + 1], hi)
            if stop > start:
                pieces.append(Piece(sequence=n, start=start, stop=stop, offset=start - bounds[n]))
            n += 1
        local_cu = [0]
        for piece in pieces:
            local_cu.append(local_cu[-1] + piece.length)
        transfers = kv_transfers(bounds, cp)
        kv_recv = tuple(t for t in transfers if t.dst == rank)
        kv_send = tuple(sorted((t for t in transfers if t.src == rank), key=lambda t: t.dst))
        prefix = sum(t.rows for t in kv_recv)
        if pieces and prefix != pieces[0].offset:
            raise CpPlanError(f"rank {rank}: prefix transfers cover {prefix} rows, the first piece starts at "
                              f"in-sequence offset {pieces[0].offset}")
        key_cu = [0] + [c + prefix for c in local_cu[1:]]
        lengths = [b - a for a, b in zip(local_cu, local_cu[1:])]
        key_lengths = [b - a for a, b in zip(key_cu, key_cu[1:])]
        return cls(
            cp=cp, rank=rank, total=bounds[-1], local=local, bounds=tuple(bounds), pieces=tuple(pieces),
            local_cu=tuple(local_cu), recv_prev=state_link(bounds, cp, rank) is not None,
            send_next=state_link(bounds, cp, rank + 1) is not None, kv_recv=kv_recv, kv_send=kv_send,
            key_cu=tuple(key_cu), max_q=max(lengths), max_k=max(key_lengths),
        )

    @property
    def lo(self) -> int:
        return self.rank * self.local

    @property
    def prefix(self) -> int:
        """Key/value rows of the first local sequence that lower ranks hold."""
        return self.key_cu[1] - self.local_cu[1]

    def halo(self, width: int) -> tuple[int, int]:
        """``(rows in, rows out)`` of a causal convolution's tail across this rank's two links.

        A width-``w`` causal convolution reads ``w - 1`` earlier rows of the same sequence. The rows
        that cross a link are the last ``min(w - 1, rows of the sequence before the boundary)``, so a
        sequence that started fewer than ``w - 1`` rows before the boundary hands over all of them and
        the receiver sees the true sequence start.
        """
        if type(width) is not int or width < 1:
            raise CpPlanError(f"convolution width must be an int >= 1, got {width!r}")

        def rows(boundary_rank: int) -> int:
            n = state_link(self.bounds, self.cp, boundary_rank)
            return 0 if n is None else min(width - 1, boundary_rank * self.local - self.bounds[n])

        return rows(self.rank), rows(self.rank + 1)

    def halo_cu(self, width: int) -> tuple[int, ...]:
        """Local boundaries with the first piece extended by the incoming halo rows."""
        h_in, _ = self.halo(width)
        return (0, *(c + h_in for c in self.local_cu[1:]))

    def positions(self) -> list[int]:
        """In-sequence position of every local row."""
        out: list[int] = []
        for piece in self.pieces:
            out.extend(range(piece.offset, piece.offset + piece.length))
        return out

    def device_tensors(self, device) -> dict[str, Any]:
        """``local_cu``/``key_cu`` int32 on ``device`` (and ``local_cu_cpu``), built once per device."""
        import torch

        device = torch.device(device)
        key = str(device)
        hit = self._device.get(key)
        if hit is None:
            hit = {
                "local_cu": torch.tensor(self.local_cu, dtype=torch.int32, device=device),
                "key_cu": torch.tensor(self.key_cu, dtype=torch.int32, device=device),
                "local_cu_cpu": torch.tensor(self.local_cu, dtype=torch.int32),
            }
            self._device[key] = hit
        return hit

    def halo_cu_tensor(self, width: int, device):
        import torch

        device = torch.device(device)
        key = f"halo{width}@{device}"
        hit = self._device.get(key)
        if hit is None:
            hit = torch.tensor(self.halo_cu(width), dtype=torch.int32, device=device)
            self._device[key] = hit
        return hit


#: Plans by the identity of the local boundary tensor the forward hands every component.
_PLANS: "dict[int, tuple[weakref.ref, ContiguousPlan]]" = {}


def register_plan(cu_seqlens, plan: ContiguousPlan) -> None:
    """Attach ``plan`` to the local boundary tensor ``cu_seqlens`` for the life of that tensor."""
    key = id(cu_seqlens)

    def _drop(_ref, key=key):
        _PLANS.pop(key, None)

    _PLANS[key] = (weakref.ref(cu_seqlens, _drop), plan)


def plan_for(cu_seqlens) -> ContiguousPlan | None:
    """The plan registered for this boundary tensor, or ``None``."""
    entry = _PLANS.get(id(cu_seqlens))
    if entry is None or entry[0]() is not cu_seqlens:
        return None
    return entry[1]


def require_plan(cu_seqlens, *, cp: int, rank: int, site: str) -> ContiguousPlan:
    """The registered plan, checked against this component's CP axis; refuses when absent."""
    plan = plan_for(cu_seqlens)
    if plan is None:
        raise CpPlanError(
            f"{site}: context parallelism cp={cp} but no contiguous CP plan is registered for this forward's "
            "boundaries. The host (MegatronModel.forward) builds one from the global cu_seqlens and registers "
            "it on the local boundary tensor; without it this rank cannot know which keys and states other "
            "ranks hold"
        )
    if (plan.cp, plan.rank) != (cp, rank):
        raise CpPlanError(f"{site}: the registered plan is cp={plan.cp} rank={plan.rank}, this component is "
                          f"cp={cp} rank={rank}")
    return plan


__all__ = [
    "ContiguousPlan",
    "CpPlanError",
    "KvTransfer",
    "Piece",
    "kv_transfers",
    "plan_for",
    "register_plan",
    "require_plan",
    "state_link",
]
