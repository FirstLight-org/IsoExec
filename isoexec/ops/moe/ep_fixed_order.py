"""Deterministic expert parallelism with a FIXED combine order (GLM clamped FP8 MoE, DeepSeek-V4.1 MoE).

Every rank holds its own tokens and the experts a placement gives it (``ExpertMap``: any owner per
global expert id, the shared expert id ``E`` included -- owned by one rank, or ``"home"``: computed by
every rank on its own tokens). One rank's forward:

1. route at home (the recipe's router: ids ``[T, k+1]`` in ascending global expert id, shared last);
2. ship every route's bf16 token row, its owner-local expert id and its route weight to the owner
   (``all_to_all_single``; integer splits exchanged first);
3. the owner runs the recipe's single-rank expert program on what it received as a k = 1 batch over
   its local experts, rows of an expert in receive order = (source rank, source token) = the global
   token order of the concatenated batch, and returns fp32 (or exact bf16) roots;
4. the home rank PLACES (never adds) each returned root at (token, canonical slot) and folds the k+1
   slots with the recipe's own single-rank combine (one fp32 chain, ascending expert id, shared last,
   one rounding).

Every expert row reads only its own token row and its expert's weights, so a root is the same bits
on any owner, at any EP degree, under any expert-to-rank map; the fold is the single-rank fold of the
same values in the same order. Hence outputs are bitwise those of EP = 1 on the concatenated batch.
Backward mirrors it: the home rank forms each slot's dY (and the weight gradient where the weight
sits in the combine), ships it along the forward's routes, the owner runs the recipe's expert
backward (dgrad, activation backward, wgrad over its experts' rows in global token order -- the
same rows, order and SEG_T segments as EP = 1) and returns fp32 dX rows (+ the weight gradient where
the weight sits before the down projection); the home rank places and folds them with the router's
dx exactly as the single-rank backward does. No atomics, nothing reduced in transit.

Replicated router and shared-at-home expert gradients are summed over the EP group in fixed rank
order before returning to autograd. Do not reduce these gradients over that same EP group again.
These sums are deterministic for a fixed layout; they need not be bitwise EP-invariant.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

import torch


# --------------------------------------------------------------------------------------- comms ---


class DistComm:
    """``torch.distributed`` (NCCL) transport for one EP group."""

    def __init__(self, group=None, *, static_routes=False):
        import torch.distributed as dist

        self.group = group if group is not None else dist.group.WORLD
        self.rank = dist.get_rank(self.group)
        self.size = dist.get_world_size(self.group)
        self.static_routes = bool(static_routes)

    def counts(self, send: torch.Tensor) -> list[int]:
        import torch.distributed as dist

        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send, group=self.group)
        return recv.tolist()

    def counts_device(self, send: torch.Tensor) -> torch.Tensor:
        """The peers' counts for this rank, left on the device (no host sync; read once by ``_Routes``)."""
        import torch.distributed as dist

        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send, group=self.group)
        return recv

    def a2a(self, x: torch.Tensor, send_splits, recv_splits) -> torch.Tensor:
        import torch.distributed as dist

        out = x.new_empty((int(sum(recv_splits)),) + tuple(x.shape[1:]))
        dist.all_to_all_single(out, x.contiguous(), list(recv_splits), list(send_splits), group=self.group)
        return out

    def sum(self, x: torch.Tensor) -> torch.Tensor:
        from isoexec.ops.collectives.ordered_sum import ordered_sum
        return ordered_sum(x, self.group)


class SimWorld:
    """A one-process EP world for tests: ``size`` ranks as threads on one GPU, exchanging tensors
    through shared slots between barriers (the all_to_all_single contract, no NCCL)."""

    def __init__(self, size: int):
        self.size = size
        self.barrier = threading.Barrier(size)
        self.slots: list = [None] * size

    def comm(self, rank: int) -> "SimComm":
        return SimComm(self, rank)


class SimComm:
    def __init__(self, world: SimWorld, rank: int):
        self.world, self.rank, self.size = world, rank, world.size

    def _exchange(self, parts: list) -> list:
        w = self.world
        torch.cuda.synchronize()
        w.slots[self.rank] = parts
        w.barrier.wait()
        got = [w.slots[src][self.rank] for src in range(self.size)]
        w.barrier.wait()
        return got

    def counts(self, send: torch.Tensor) -> list[int]:
        return [int(v) for v in self._exchange([int(v) for v in send.tolist()])]

    def counts_device(self, send: torch.Tensor) -> torch.Tensor:
        return torch.tensor(self.counts(send), dtype=send.dtype, device=send.device)

    def a2a(self, x: torch.Tensor, send_splits, recv_splits) -> torch.Tensor:
        parts = list(x.contiguous().split([int(s) for s in send_splits], 0))
        got = self._exchange([p.clone() for p in parts])
        out = torch.cat(got, 0) if got else x.new_empty((0,) + tuple(x.shape[1:]))
        assert out.size(0) == int(sum(recv_splits))
        return out

    def sum(self, x: torch.Tensor) -> torch.Tensor:
        from isoexec.ops.collectives.ordered_sum import sum_partials
        return sum_partials(self._exchange([x.clone() for _ in range(self.size)]))


def reduce_replicated_grads(comm, emap, router, gate_up, down):
    """Complete replicated gradients, including zero-token ranks; owned experts stay local."""
    if comm.size == 1:
        return router, gate_up, down
    router = comm.sum(router)
    if emap.shared_home:
        # local_experts puts the replicated shared expert last on every rank.
        gate_up[-1].copy_(comm.sum(gate_up[-1]))
        down[-1].copy_(comm.sum(down[-1]))
    return router, gate_up, down


def run_sim(size: int, fn):
    """Run ``fn(comm)`` on ``size`` simulated ranks (threads); returns the per-rank results."""
    world = SimWorld(size)
    out, err = [None] * size, [None] * size
    dev = torch.cuda.current_device()

    def body(r):
        try:
            torch.cuda.set_device(dev)
            torch.autograd.set_multithreading_enabled(False)
            out[r] = fn(world.comm(r))
        except BaseException as exc:  # noqa: BLE001 -- re-raised below
            err[r] = exc
            world.barrier.abort()

    threads = [threading.Thread(target=body, args=(r,)) for r in range(size)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for e in err:
        if e is not None and not isinstance(e, threading.BrokenBarrierError):
            raise e
    for e in err:
        if e is not None:
            raise e
    return out


# ---------------------------------------------------------------------------------- placement ---


@dataclass
class ExpertMap:
    """Owner of every global expert id ``0..E`` (``E`` = the shared expert). ``owner[E] = -1`` means
    ``"home"``: every rank computes the shared expert on its own tokens."""

    owner: list[int]
    size: int
    _tables: dict = field(default_factory=dict, repr=False)

    @staticmethod
    def contiguous(E: int, size: int, shared="home") -> "ExpertMap":
        if E % size:
            raise ValueError(f"{E} experts do not split over EP = {size}")
        per = E // size
        return ExpertMap([e // per for e in range(E)] + [-1 if shared == "home" else int(shared)], size)

    @staticmethod
    def permuted(E: int, size: int, seed: int, shared="home") -> "ExpertMap":
        """A balanced but scrambled placement (expert ids dealt to ranks in a seeded random order)."""
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(E, generator=g).tolist()
        owner = [0] * E
        for i, e in enumerate(perm):
            owner[e] = i % size
        return ExpertMap(owner + [-1 if shared == "home" else int(shared)], size)

    @property
    def E(self) -> int:
        return len(self.owner) - 1

    @property
    def shared_home(self) -> bool:
        return self.owner[-1] < 0

    def local_experts(self, rank: int) -> list[int]:
        """Global ids this rank computes, ascending (the shared expert last when it computes it)."""
        ids = [e for e in range(self.E) if self.owner[e] == rank]
        if self.shared_home or self.owner[-1] == rank:
            ids.append(self.E)
        return ids

    def tables(self, rank: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        """``(owner_rank[E+1], local_index[E+1])`` int64 for routes leaving ``rank`` (the shared
        expert resolves to ``rank`` itself when it is ``"home"``)."""
        key = (rank, str(torch.device(device)))
        hit = self._tables.get(key)
        if hit is not None:
            return hit
        owner = [rank if o < 0 else o for o in self.owner]
        local = [0] * (self.E + 1)
        for r in range(self.size):
            for i, e in enumerate(self.local_experts(r)):
                if owner[e] == r:
                    local[e] = i
        out = (torch.tensor(owner, dtype=torch.int64, device=device), torch.tensor(local, dtype=torch.int64, device=device))
        self._tables[key] = out          # built once per (rank, device): no host copy on later steps
        return out


# ------------------------------------------------------------------------------------- engine ---


@dataclass
class _Routes:
    T: int
    S: int                     # slots a token: k routed + shared
    order: torch.Tensor        # routes grouped by owner rank (stable): index into flat (t * S + s)
    _send: list[int] | None
    _recv: list[int] | None
    pos: torch.Tensor | None = None   # inverse of order: flat (t * S + s) -> its row in the send buffer
    valid: torch.Tensor | None = None  # fixed-capacity send slots belonging to their destination
    counts_dev: torch.Tensor | None = None  # [2, world] (send, recv) on the device, read on first use

    def _resolve(self):
        # The one host sync of a dynamic route plan (NCCL all_to_all takes host splits): deferred to the first
        # exchange, so the payload gathers queued before it overlap the count exchange.
        if self._send is None:
            send, recv = self.counts_dev.tolist()
            self._send, self._recv = [int(v) for v in send], [int(v) for v in recv]

    @property
    def send(self) -> list[int]:
        self._resolve()
        return self._send

    @property
    def recv(self) -> list[int]:
        self._resolve()
        return self._recv

    def home_rows(self) -> torch.Tensor:
        """``[T, S]`` int32: the send/return-buffer row of each (token, canonical slot). Folds and the
        combine backward read rows through it, so returned rows are never placed (copied) at home."""
        return self.pos.to(torch.int32).view(self.T, self.S)


def _plan_routes(ids_ext: torch.Tensor, owner_tab: torch.Tensor, comm) -> _Routes:
    T, S = ids_ext.shape
    flat = ids_ext.reshape(-1).long()
    owner = owner_tab.index_select(0, flat)
    if getattr(comm, "static_routes", False):
        # Every peer has room for every route: no host counts, overflow or dropped
        # tokens. All ranks must enter with the same padded token bucket T.
        # Slots retain flat canonical order; expert kernels sort received rows
        # stably, so real rows retain source-rank/source-token order.
        n = T * S
        flat_rows = torch.arange(n, device=flat.device, dtype=torch.int64)
        order = flat_rows.repeat(comm.size)
        peers = torch.arange(comm.size, device=flat.device, dtype=torch.int64)
        valid = (peers[:, None] == owner[None, :]).reshape(-1)
        pos = owner * n + flat_rows
        return _Routes(T, S, order, [n] * comm.size, [n] * comm.size, pos, valid)
    order = torch.argsort(owner, stable=True)
    send_t = torch.bincount(owner, minlength=comm.size)
    if hasattr(comm, "counts_device"):
        counts_dev = torch.stack((send_t, comm.counts_device(send_t)))
        send = recv = None                       # resolved (one host sync) at the first exchange
    else:
        counts_dev, send = None, send_t.tolist()
        recv = comm.counts(send_t)
    pos = torch.empty_like(order)
    if order.numel():
        pos.index_copy_(0, order, torch.arange(order.numel(), dtype=order.dtype, device=order.device))
    return _Routes(T, S, order, send, recv, pos, counts_dev=counts_dev)


def _mask_send(rt: _Routes, payload: torch.Tensor) -> torch.Tensor:
    if rt.valid is None:
        return payload
    # Zero codes, scales, route weights and local ids for unused peer slots.
    # Local id zero is a valid dummy expert; returned dummy roots are never read.
    mask = rt.valid.reshape((-1,) + (1,) * (payload.ndim - 1))
    return torch.where(mask, payload, torch.zeros_like(payload))


def dispatch(rt: _Routes, comm, *payloads):
    """Ship per-route payloads (``[T*S, ...]`` in flat (token, slot) order) to the owners."""
    return [comm.a2a(_mask_send(rt, p.index_select(0, rt.order)), rt.send, rt.recv) for p in payloads]


def dispatch_all(rt: _Routes, comm, x: torch.Tensor, *payloads):
    """``dispatch_rows(x)`` and ``dispatch(*payloads)`` with every send-buffer gather queued BEFORE the route plan's
    host count sync, so the gathers overlap the count exchange; same bytes as the separate calls."""
    tok = torch.div(rt.order, rt.S, rounding_mode="floor")
    bufs = [_mask_send(rt, x.index_select(0, tok))] + [_mask_send(rt, p.index_select(0, rt.order)) for p in payloads]
    return [comm.a2a(b, rt.send, rt.recv) for b in bufs]


def dispatch_rows(rt: _Routes, comm, x: torch.Tensor) -> torch.Tensor:
    """Ship each route's token row (``x [T, H]``) to its owner."""
    tok = torch.div(rt.order, rt.S, rounding_mode="floor")
    return comm.a2a(_mask_send(rt, x.index_select(0, tok)), rt.send, rt.recv)


def send_ordered(rt: _Routes, comm, t: torch.Tensor) -> torch.Tensor:
    """Ship a payload that is already in send-buffer order (e.g. computed through ``home_rows``)."""
    return comm.a2a(_mask_send(rt, t), rt.send, rt.recv)


def ret(rt: _Routes, comm, local: torch.Tensor) -> torch.Tensor:
    """Return owner rows (receive order) home, left in send-buffer order (read via ``home_rows``)."""
    return comm.a2a(local, rt.recv, rt.send)


def collect(rt: _Routes, comm, local: torch.Tensor) -> torch.Tensor:
    """Return owner rows (receive order) home and PLACE them at flat (token, slot): ``[T*S, ...]``."""
    back = comm.a2a(local, rt.recv, rt.send)
    if rt.valid is not None:
        return back.index_select(0, rt.pos)
    out = back.new_empty((rt.T * rt.S,) + tuple(back.shape[1:]))
    if rt.T:
        out.index_copy_(0, rt.order, back)
    return out


def owner_plan(lids: torch.Tensor, n_local: int):
    """moe_sm100's exact-rank plan of a received k = 1 batch over the owner's local experts: an
    expert's rows keep receive order (stable counting sort)."""
    from isoexec.ops.moe.moe_sm100 import ops as base

    return base.route(lids.to(torch.int32).view(-1, 1).contiguous(), n_local)


def to_receive_order(grouped: torch.Tensor, plan) -> torch.Tensor:
    """Grouped rows (expert-major) -> receive order (route i's row)."""
    return grouped.index_select(0, plan.restore[:, 0].long())


def to_grouped(recv_rows: torch.Tensor, plan) -> torch.Tensor:
    return recv_rows.index_select(0, plan.inverse.long())


__all__ = ["DistComm", "ExpertMap", "SimComm", "SimWorld", "collect", "dispatch", "dispatch_all", "dispatch_rows", "owner_plan", "ret",
           "run_sim", "send_ordered", "to_grouped", "to_receive_order"]


def static_route_budget(tokens, slots, world, hidden, storage):
    """Deterministic transport capacity, excluding expert-kernel workspace.

    `tokens` is each rank's padded home-token capacity. The byte sum is the
    aggregate payload allocation if no graph-pool reuse occurs, not a measured
    CUDA allocator peak. Record actual capture memory separately.
    """
    if min(tokens, slots, world, hidden) <= 0:
        raise ValueError('static EP dimensions must be positive')
    if storage == 'glm_fp8':
        if hidden % 128:
            raise ValueError('GLM FP8 row scales require 128-aligned hidden width')
        per_route = hidden + hidden // 128 * 4 + 8 + 4 + hidden * 4
    elif storage == 'deepseek_fp4':
        per_route = hidden * 2 + 8 + 4 + hidden * 2
    else:
        raise ValueError('unknown static EP checkpoint storage')
    capacity = tokens * slots * world
    return dict(home_tokens=tokens, slots=slots, world=world, peer_capacity=tokens * slots,
                received_route_capacity=capacity, compact_balanced_route_capacity=tokens * slots,
                payload_bytes_each_direction=capacity * per_route,
                send_receive_payload_allocation_bound=2 * capacity * per_route,
                routing_tensor_bytes=capacity * 9 + tokens * slots * 8,
                excludes='expert activations, kernel workspace, NCCL internal buffers, graph allocator alignment')
