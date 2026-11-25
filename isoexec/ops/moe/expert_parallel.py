"""Expert-parallel movement and expert-tensor-parallel fixed-tree reduction for MoE bindings.
EP places terms by ascending global expert ID; ETP folds fixed K leaves, and placements forbid combining EP>1 with ETP>1.
Deferred owner combination must also receive an owner-aware shared expert to avoid reducing its fc2 result twice."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, cast

import torch

from isoexec.ops.moe.carriers import ExpertOutputs
from isoexec.ops.moe.carriers import RoutingResult
from isoexec.schema import Placement, ShardError

# Cross-rank expert partials travel fp32 (03 section 2.3 / V5): a bf16 wire needs a discharge.
# The bf16 wire is legal only under the losslessness discharge, which is not issued.
EP_WIRE_DTYPE = "fp32"

#: The keyword-only seam ``moe.expert_mlp`` exposes for the leaves this rank folds (G // ETP).
SEAM_N_LEAVES = "n_leaves"


class _EPRowExchange(torch.autograd.Function):
    """Exchange FP32 expert rows by placement, without addition; backward selects owned rows."""

    @staticmethod
    def forward(ctx, rows, dest, counts, nmax, n_slots, group):  # type: ignore[override]
        import torch.distributed as dist

        ep, n, H = len(counts), int(rows.shape[0]), int(rows.shape[1])
        # Every retained row is overwritten exactly once, so zero-filling these buffers is unnecessary.
        pad_rows = rows.new_empty(nmax, H)
        pad_dest = dest.new_empty(nmax)
        if n:
            pad_rows[:n].copy_(rows)
            pad_dest[:n].copy_(dest)
        all_rows = rows.new_empty(ep * nmax, H)
        all_dest = dest.new_empty(ep * nmax)
        dist.all_gather_into_tensor(all_rows, pad_rows, group=group)
        dist.all_gather_into_tensor(all_dest, pad_dest, group=group)
        out = rows.new_empty(n_slots, H)
        for r, c in enumerate(counts):
            if c:
                out.index_copy_(0, all_dest[r * nmax : r * nmax + c], all_rows[r * nmax : r * nmax + c])
        ctx.save_for_backward(dest)
        return out

    @staticmethod
    def backward(ctx, g):  # type: ignore[override]
        (dest,) = ctx.saved_tensors
        # Index selection already returns contiguous storage; avoid copying the full gradient first.
        return g.index_select(0, dest), None, None, None, None, None


class _EPAllToAll(torch.autograd.Function):
    """Dispatch and return expert rows using variable-split all-to-all."""

    @staticmethod
    def forward(ctx, x, out_splits, in_splits, group):  # type: ignore[override]
        import torch.distributed as dist

        out = x.new_empty((int(sum(out_splits)),) + tuple(x.shape[1:]))
        dist.all_to_all_single(out, x.contiguous(), list(out_splits), list(in_splits), group=group)
        ctx.splits, ctx.group = (list(in_splits), list(out_splits)), group
        return out

    @staticmethod
    def backward(ctx, g):  # type: ignore[override]
        import torch.distributed as dist

        in_splits, out_splits = ctx.splits
        back = g.new_empty((int(sum(in_splits)),) + tuple(g.shape[1:]))
        dist.all_to_all_single(back, g.contiguous(), in_splits, out_splits, group=ctx.group)
        return back, None, None, None


class _RouteRows(torch.autograd.Function):
    """Each dispatched route's token row; the adjoint folds a token's k route gradients in fp32
    over its router slots in ascending order and rounds once.

    ``index_select``'s own adjoint scatter-adds the k copies of a token with float atomics in the
    operand dtype, so a token's dx would depend on atomic arrival order whenever torch's
    deterministic mode is off (the trainer's default). This is the EP sibling of
    ``moe_sm100.integration._Gather``: the same slot order, the same single rounding.
    """

    @staticmethod
    def forward(ctx, xs, tok, restore):  # type: ignore[override]
        ctx.save_for_backward(restore)
        ctx.dtype = xs.dtype
        return xs.index_select(0, tok)

    @staticmethod
    def backward(ctx, g):  # type: ignore[override]
        (restore,) = ctx.saved_tensors  # [tl, k]: route (t, j)'s row in the dispatch order
        dx = g.index_select(0, restore[:, 0]).float()
        for j in range(1, restore.size(1)):
            dx = dx + g.index_select(0, restore[:, j]).float()
        return dx.to(ctx.dtype), None, None


def _a2a(x, out_splits, in_splits, group):
    """Non-differentiable sibling of :class:`_EPAllToAll` for the integer/index payloads."""
    import torch.distributed as dist

    out = x.new_empty((int(sum(out_splits)),) + tuple(x.shape[1:]))
    dist.all_to_all_single(out, x.contiguous(), list(out_splits), list(in_splits), group=group)
    return out


class _TPShardTokens(torch.autograd.Function):
    """Select this TP rank's tokens and restore replicated gradients in backward."""

    @staticmethod
    def forward(ctx, x, t0, t1, n_full, group):  # type: ignore[override]
        ctx.slice, ctx.group, ctx.n_full = (t0, t1), group, n_full
        return x.narrow(0, t0, t1 - t0).contiguous()

    @staticmethod
    def backward(ctx, g):  # type: ignore[override]
        import torch.distributed as dist

        t0, t1 = ctx.slice
        full = g.new_zeros((ctx.n_full,) + tuple(g.shape[1:]))
        full.narrow(0, t0, t1 - t0).copy_(g)
        dist.all_reduce(full, op=dist.ReduceOp.SUM, group=ctx.group)
        return full, None, None, None, None


def token_shard(n_tokens: int, ds: int, ds_rank: int) -> tuple[int, int]:
    """This dispatch-shard rank's half-open token range. Contiguous and remainder-tolerant, so the
    slices concatenate back in token order and every rank can compute every other rank's extent
    analytically (no collective to size the restore all-gather)."""
    return n_tokens * ds_rank // ds, n_tokens * (ds_rank + 1) // ds


_EP_CENSUS: dict[str, Any] = {"calls": 0, "routes_sent": None, "routes_received": None}
_EP_CENSUS_ENV = "ISOEXEC_FULL_DISTRIBUTION_EVIDENCE_DIR"


def ep_routing_census_state() -> dict[str, Any]:
    """Copy of this process's cumulative expert-parallel routing tally (see ``ep_routing_census``)."""
    return {
        "calls": _EP_CENSUS["calls"],
        "routes_sent": list(_EP_CENSUS["routes_sent"] or []),
        "routes_received": list(_EP_CENSUS["routes_received"] or []),
    }


def ep_routing_census(
    send_counts, recv_counts, *, ep_rank: int, ep: int, emit: bool | None = None, group: str | None = None
) -> str | None:
    """Record expert dispatch send/receive counts when evidence collection is enabled."""
    send = [int(v) for v in send_counts]
    recv = [int(v) for v in recv_counts]
    if len(send) != ep or len(recv) != ep:
        raise ShardError(f"EP census: split vectors of length {len(send)}/{len(recv)} for ep={ep}")
    if _EP_CENSUS["routes_sent"] is None or len(_EP_CENSUS["routes_sent"]) != ep:
        _EP_CENSUS["routes_sent"] = [0] * ep
        _EP_CENSUS["routes_received"] = [0] * ep
        _EP_CENSUS["calls"] = 0
    _EP_CENSUS["calls"] += 1
    _EP_CENSUS["routes_sent"] = [a + b for a, b in zip(_EP_CENSUS["routes_sent"], send)]
    _EP_CENSUS["routes_received"] = [a + b for a, b in zip(_EP_CENSUS["routes_received"], recv)]
    line = (
        f"[ISOEXEC-MOE-EP] CENSUS pid={os.getpid()} ep_rank={ep_rank} ep={ep} calls={_EP_CENSUS['calls']} "
        f"routes_sent={_EP_CENSUS['routes_sent']} routes_received={_EP_CENSUS['routes_received']}"
        + (f" group={group}" if group is not None else "")
    )
    if emit is None:
        emit = bool(os.environ.get(_EP_CENSUS_ENV))
    if emit:
        print(line, flush=True)
    return line


def expert_reduce(
    inner: Callable[..., ExpertOutputs],
    *,
    etp: int,
    group,
    deferred: bool = False,
    tp_copy: Callable[[torch.Tensor], torch.Tensor] | None = None,
    compact_root: bool = False,
) -> Callable[..., ExpertOutputs]:
    """Combine FP32 expert partials with the fixed ETP tree before the ordered expert fold.

    ``inner`` is the single-rank ``moe.expert_mlp`` callable with its ``n_leaves`` seam already
    bound: the leaf count is the caller's, because it is the placement's leaf plan divided by the
    degree, and a value this function re-derived would be a second copy of it.
    """

    def reduce(rows: torch.Tensor) -> torch.Tensor:
        # This PIK transport is shared by both expert providers. The facade
        # selects a GEMM package; SM100 does not own a second copy of this tree.
        from isoexec.ops.moe.moe_sm90 import moe_batch_invariant

        rows = rows if rows.dtype == torch.float32 else rows.to(torch.float32)
        return moe_batch_invariant._PikTreeAllReduce.apply(rows, group)

    def f(x, routing, gate_up, down, **kw) -> ExpertOutputs:
        weight_after_fc2 = bool(kw.get("weight_after_fc2", False))
        if tp_copy is not None:
            # Sum the expert FC1's partial input gradients across its ETP group.
            x = tp_copy(x)
            if not weight_after_fc2:
                routing = RoutingResult(routing.indices, tp_copy(routing.probs), routing.scores_full)
        if weight_after_fc2:
            # BF16 rounding and routing weights belong AFTER the complete FC2
            # tree. Applying them independently to rank partials changes bytes.
            kw["defer_weight_after_fc2"] = True
        if compact_root:
            kw["compact_root"] = True
        out: ExpertOutputs = inner(x, routing, gate_up, down, **kw)

        def finish(rows: torch.Tensor) -> torch.Tensor:
            result = reduce(rows).clone()
            if weight_after_fc2:
                if out.layout != "permuted" or out.combine_rows is None:
                    raise ShardError("After-FC2 ETP weighting requires permuted rows and combine_rows")
                # combine_rows walks ascending expert IDs, whereas the router's
                # slots follow descending scores. Preserve each expert/prob pair.
                probabilities_by_expert = routing.probs.gather(1, routing.indices.argsort(dim=1))
                probabilities = torch.empty_like(routing.probs).flatten().scatter(
                    0, out.combine_rows.flatten().long(), probabilities_by_expert.flatten())
                result = (result.to(x.dtype) * probabilities[:, None].to(x.dtype)).float()
            return result

        if deferred:
            return ExpertOutputs(
                rows=out.rows,
                layout="permuted_partial",
                routing=out.routing,
                extra=out.extra,
                combine_rows=out.combine_rows,
                reduce=finish if weight_after_fc2 else reduce,
                weight_after_fc2=weight_after_fc2,
            )
        # Clone borrowed symmetric reduction output before another collective can overwrite it.
        return ExpertOutputs(
            rows=finish(out.rows),
            layout=out.layout,
            routing=out.routing,
            extra=out.extra,
            combine_rows=out.combine_rows,
        )

    f.__name__ = f.__qualname__ = f"expert_mlp_etp{etp}" + ("|deferred" if deferred else "")
    return f


def expert_dispatch(
    inner: Callable[..., ExpertOutputs],
    *,
    ep: int,
    ep_rank: int,
    ep_group,
    combine: Callable[..., torch.Tensor],
    placement: Placement,
) -> Callable[..., ExpertOutputs]:
    """Dispatch token slices to expert owners and fold returned terms in fixed expert order.
    Under SP, keep folded rows local instead of gathering and slicing them again.

    The token slices are cut over the TP group only, whose ranks hold the same stream. Peers in the
    EP group may hold different tokens (DP replicas, or the zigzag rows of a CP cut): each sends its own
    routes and folds its own tokens, and nothing here reads a token's sequence position."""
    ds, ds_rank = placement.tp.size, placement.tp.rank
    ds_group = cast("torch.distributed.ProcessGroup | None", placement.tp.group)

    def f(x: torch.Tensor, routing: RoutingResult, gate_up: torch.Tensor, down: torch.Tensor, **kw) -> ExpertOutputs:
        import torch.distributed as dist

        # Require process-group rank order to match expert and token ownership.
        if dist.get_rank(ep_group) != ep_rank or (ds > 1 and dist.get_rank(ds_group) != ds_rank):
            raise ShardError(
                f"EP: Placement ranks (ep {ep_rank}, tp {ds_rank}) disagree with the group handles "
                f"(ep {dist.get_rank(ep_group)}, tp {dist.get_rank(ds_group) if ds > 1 else ds_rank}); "
                "the expert ranges and the dispatch destinations are numbered by group rank"
            )
        dev = x.device
        T, k = int(routing.indices.shape[0]), int(routing.indices.shape[1])
        if placement.sp and T % ds:
            raise ShardError(f"SP expert output needs equal TP token slices; got {T} tokens over TP={ds}")
        e_local, H = int(gate_up.shape[0]), int(x.shape[-1])
        t0, t1 = token_shard(T, ds, ds_rank)
        tl = t1 - t0
        # (1) this rank's token slice; the backward puts the TP replication back (SP gradient rule)
        xs = x if ds == 1 else _TPShardTokens.apply(x, t0, t1, T, ds_group)
        ps = routing.probs if ds == 1 else _TPShardTokens.apply(routing.probs, t0, t1, T, ds_group)
        idx = routing.indices[t0:t1].long()  # [tl, k] GLOBAL expert ids
        # the k-slot of each route in ascending GLOBAL expert id order (top-k picks distinct experts,
        # so rank-of-element is a bijection onto 0..k-1 -- argsort of argsort, no ties to break)
        slot = idx.argsort(dim=1).argsort(dim=1).reshape(-1)
        flat = idx.reshape(-1)
        owner = torch.div(flat, e_local, rounding_mode="floor")
        order = torch.argsort(owner, stable=True)  # routes grouped by destination rank
        send = torch.bincount(owner, minlength=ep)
        recv = torch.empty(ep, dtype=torch.int64, device=dev)
        dist.all_to_all_single(recv, send, group=ep_group)
        both = torch.stack([send, recv]).tolist()  # ONE device->host sync for both split vectors
        send_c, recv_c = both[0], both[1]
        n_recv = int(sum(recv_c))
        # Evidence only (printed when the campaign evidence sink is on): which owners this rank's
        # routes went to and whose routes it serves. Both vectors are already on the host.
        ep_routing_census(send_c, recv_c, ep_rank=ep_rank, ep=ep, group=f"g{min(dist.get_process_group_ranks(ep_group))}")
        # (arange(tl*k) // k)[order] == order // k, an exact integer identity: arange(N)[i] == i.
        # Three tl*k int64 buffers and three launches collapse to one.
        tok = torch.div(order, k, rounding_mode="floor")
        # restore[t, j]: where route (t, j) sits in the dispatch order (``order`` is a permutation).
        restore = torch.empty_like(order).index_copy_(
            0, order, torch.arange(tl * k, device=dev, dtype=order.dtype)).view(tl, k)
        # (2) dispatch: rows, local expert ids and probs to the expert owners
        r_rows = _EPAllToAll.apply(_RouteRows.apply(xs, tok, restore), recv_c, send_c, ep_group)
        r_exp = _a2a(torch.remainder(flat, e_local).index_select(0, order), recv_c, send_c, ep_group)
        r_prb = _EPAllToAll.apply(ps.reshape(-1, 1).index_select(0, order), recv_c, send_c, ep_group)
        # (3) compute -- the unchanged single-rank binding, as a k == 1 pseudo-batch
        if n_recv:
            got: ExpertOutputs = inner(
                r_rows,
                RoutingResult(r_exp.to(torch.int32).view(n_recv, 1), r_prb.view(n_recv, 1), None),
                gate_up,
                down,
                **kw,
            )
            if got.layout != "permuted" or not isinstance(got.extra, torch.Tensor):
                raise ShardError(
                    "moe.expert_mlp at EP>1 needs the permuted-rows layout with the row permutation in "
                    f"``extra``; got layout {got.layout!r} extra {type(got.extra).__name__}"
                )
            rows = got.rows if got.rows.dtype == torch.float32 else got.rows.to(torch.float32)
            # Restore receive order into a fully covered buffer without zero-fill or cloning.
            local = rows.new_empty(n_recv, H).index_copy_(0, got.extra.long(), rows)
        else:
            # Empty owners must still enter the return and both dispatch backward collectives.
            local = r_rows.to(torch.float32) + r_prb.to(torch.float32)
        # (4) return, (5) place-don't-add into this rank's own [tl*k, H] term buffer
        back = _EPAllToAll.apply(local, send_c, recv_c, ep_group)
        # Place returned terms by a bijective destination mapping; every output slot is written once.
        buf = back.new_empty(tl * k, H).index_copy_(0, (tok * k + slot.index_select(0, order)), back)
        # (6) the CONTRACTED fold, over this rank's own tokens, by the unchanged combine kernel
        if tl:
            si = torch.arange(tl, device=dev, dtype=torch.int64).repeat_interleave(k)
            combine_rows = torch.arange(tl * k, dtype=torch.int32, device=dev).view(tl, k)
            folded = combine(
                ExpertOutputs(
                    rows=buf,
                    layout="permuted",
                    routing=RoutingResult(idx.to(torch.int32), ps, None),
                    extra=si,
                    combine_rows=combine_rows,
                ),
                None,
            )
        else:
            # An empty token shard must keep the return exchange reachable through TP restoration.
            folded = buf
        # SP owns exactly these token_shard slices. Restoring the full routed stream only to
        # scatter it after the pointwise shared join adds communication without changing bytes.
        if placement.sp:
            return ExpertOutputs(rows=folded, layout="combined_local", routing=routing, extra=None)
        # (7) restore the TP replication the non-SP residual stream expects
        if ds == 1:
            full = folded
        else:
            ext = [token_shard(T, ds, r) for r in range(ds)]
            counts = [b - a for a, b in ext]
            full = _EPRowExchange.apply(folded, torch.arange(t0, t1, device=dev), counts, max(counts), T, ds_group)
        return ExpertOutputs(rows=full, layout="combined", routing=routing, extra=None)

    f.__name__ = f.__qualname__ = f"expert_mlp_ep{ep}_a2a"
    return f


#: Host-side bound on the tokens any EP peer feeds one MoE call, injected by the engine runtime
#: (``ops/`` may not import an engine). ``None`` means every peer feeds this rank's own count.
_ENGINE_TOKEN_BOUND: Callable[[int], int] | None = None
_ENGINE_EP_CENSUS: dict[str, Any] = {"calls": 0, "captured_calls": 0, "routes_sent": None}


def set_engine_token_bound(bound: Callable[[int], int] | None) -> None:
    """Install the engine's cross-peer token bound used to size static expert dispatch buffers."""
    global _ENGINE_TOKEN_BOUND
    _ENGINE_TOKEN_BOUND = bound


def _engine_census(send: torch.Tensor, *, ep: int, ep_rank: int, tokens: int = 0, shard: tuple[int, int] = (0, 0),
                   tp_rank: int = 0) -> None:
    """Count routes by destination outside graph capture; a captured replay is counted only as a call."""
    _ENGINE_EP_CENSUS["calls"] += 1
    if send.is_cuda and torch.cuda.is_current_stream_capturing():
        _ENGINE_EP_CENSUS["captured_calls"] += 1
        return
    if not os.environ.get(_EP_CENSUS_ENV):
        return
    counts = send.tolist()
    totals = _ENGINE_EP_CENSUS["routes_sent"] or [0] * ep
    _ENGINE_EP_CENSUS["routes_sent"] = [a + int(b) for a, b in zip(totals, counts)]
    print(
        f"[ISOEXEC-MOE-EP-ENGINE] pid={os.getpid()} ep_rank={ep_rank} ep={ep} calls={_ENGINE_EP_CENSUS['calls']} "
        f"captured={_ENGINE_EP_CENSUS['captured_calls']} routes_sent={_ENGINE_EP_CENSUS['routes_sent']} "
        f"remote_routes={sum(v for r, v in enumerate(_ENGINE_EP_CENSUS['routes_sent']) if r != ep_rank)} "
        f"tp_rank={tp_rank} token_shard={shard[0]}:{shard[1]} of {tokens}",
        flush=True,
    )


def engine_expert_dispatch(
    inner: Callable[..., ExpertOutputs],
    *,
    ep: int,
    ep_rank: int,
    ep_group,
    combine: Callable[..., torch.Tensor],
    placement: Placement,
) -> Callable[..., ExpertOutputs]:
    """Forward-only EP dispatch with static shapes and no host synchronization, for graph capture.

    Each TP rank takes the same contiguous token slice as :func:`expert_dispatch`, sends every
    route to its expert's owner and folds the returned FP32 terms with the same combine in the
    same ascending global-expert slot order. Instead of exchanging split sizes, every rank sends a
    fixed ``capacity`` block to every peer (``ceil(bound / TP) * top_k`` routes, the most one token
    slice can route to one owner); unused slots carry a zero row for local expert 0 with probability
    zero, are computed by the owner like any other row, and are never placed in a token's fold.
    The owner's per-row expert computation is the same callable on the same rows, so only the
    batch it rides in differs, as it already does between trainer and engine layouts.
    """
    ds, ds_rank = placement.tp.size, placement.tp.rank
    ds_group = cast("torch.distributed.ProcessGroup | None", placement.tp.group)
    if placement.sp:
        raise ShardError("engine EP dispatch keeps the replicated residual stream (Placement.sp=False)")

    def f(x: torch.Tensor, routing: RoutingResult, gate_up: torch.Tensor, down: torch.Tensor, **kw) -> ExpertOutputs:
        import torch.distributed as dist

        if dist.get_rank(ep_group) != ep_rank or (ds > 1 and dist.get_rank(ds_group) != ds_rank):
            raise ShardError("engine EP: Placement ranks disagree with the group handles")
        dev = x.device
        T, k = int(routing.indices.shape[0]), int(routing.indices.shape[1])
        bound = T if _ENGINE_TOKEN_BOUND is None else int(_ENGINE_TOKEN_BOUND(T))
        if bound < T:
            raise ShardError(f"engine EP: token bound {bound} is below this call's {T} tokens")
        e_local, H = int(gate_up.shape[0]), int(x.shape[-1])
        t0, t1 = token_shard(T, ds, ds_rank)
        tl = t1 - t0
        capacity = -(-bound // ds) * k
        xs, ps = x[t0:t1], routing.probs[t0:t1]
        idx = routing.indices[t0:t1].long()
        flat = idx.reshape(-1)
        owner = torch.div(flat, e_local, rounding_mode="floor")
        slot_asc = idx.argsort(dim=1).argsort(dim=1).reshape(-1)
        tok = torch.arange(tl, device=dev, dtype=torch.int64).repeat_interleave(k)
        onehot = torch.nn.functional.one_hot(owner, ep)
        within = (onehot.cumsum(dim=0) - 1).gather(1, owner.view(-1, 1)).view(-1)
        send_slot = owner * capacity + within
        send_rows = x.new_zeros(ep * capacity, H)
        send_rows.index_copy_(0, send_slot, xs.index_select(0, tok))
        send_exp = torch.zeros(ep * capacity, dtype=torch.int32, device=dev)
        send_exp.index_copy_(0, send_slot, torch.remainder(flat, e_local).to(torch.int32))
        send_prb = ps.new_zeros(ep * capacity, 1)
        send_prb.index_copy_(0, send_slot, ps.reshape(-1, 1))
        _engine_census(onehot.sum(dim=0), ep=ep, ep_rank=ep_rank, tokens=T, shard=(t0, t1), tp_rank=ds_rank)
        recv_rows = torch.empty_like(send_rows)
        recv_exp = torch.empty_like(send_exp)
        recv_prb = torch.empty_like(send_prb)
        dist.all_to_all_single(recv_rows, send_rows, group=ep_group)
        dist.all_to_all_single(recv_exp, send_exp, group=ep_group)
        dist.all_to_all_single(recv_prb, send_prb, group=ep_group)
        got: ExpertOutputs = inner(
            recv_rows, RoutingResult(recv_exp.view(-1, 1), recv_prb, None), gate_up, down, **kw,
        )
        if got.layout != "permuted" or not isinstance(got.extra, torch.Tensor):
            raise ShardError("moe.expert_mlp at engine EP>1 needs the permuted-rows layout with its permutation")
        rows = got.rows if got.rows.dtype == torch.float32 else got.rows.to(torch.float32)
        local = rows.new_empty(ep * capacity, H).index_copy_(0, got.extra.long(), rows)
        back = torch.empty_like(local)
        dist.all_to_all_single(back, local, group=ep_group)
        # Place each of this rank's routes at its token's ascending-expert slot; padding is never read.
        mine = back.index_select(0, send_slot)
        buf = mine.new_empty(tl * k, H).index_copy_(0, tok * k + slot_asc, mine)
        if tl:
            si = torch.arange(tl, device=dev, dtype=torch.int64).repeat_interleave(k)
            combine_rows = torch.arange(tl * k, dtype=torch.int32, device=dev).view(tl, k)
            folded = combine(
                ExpertOutputs(rows=buf, layout="permuted", routing=RoutingResult(idx.to(torch.int32), ps, None),
                              extra=si, combine_rows=combine_rows),
                None,
            )
        else:
            folded = buf
        if ds == 1:
            full = folded
        else:
            ext = [token_shard(T, ds, r) for r in range(ds)]
            counts = [b - a for a, b in ext]
            full = _EPRowExchange.apply(folded, torch.arange(t0, t1, device=dev), counts, max(counts), T, ds_group)
        return ExpertOutputs(rows=full, layout="combined", routing=routing, extra=None)

    f.__name__ = f.__qualname__ = f"expert_mlp_ep{ep}_engine_static"
    return f


def ep_combine(inner: Callable[..., torch.Tensor], *, ep: int) -> Callable[..., torch.Tensor]:
    """Add shared output to already-folded expert rows; local layouts require matching shared slices."""

    def f(routed: ExpertOutputs, shared):
        if routed.layout not in ("combined", "combined_local"):
            return inner(routed, shared)
        if (
            routed.layout == "combined_local"
            and shared is not None
            and (not isinstance(shared, torch.Tensor) or shared.shape != routed.rows.shape)
        ):
            raise ShardError("combined_local needs a shared Tensor with the same local row shape")
        return routed.rows if shared is None else routed.rows + shared

    f.__name__ = f.__qualname__ = f"combine_ep{ep}_owner_fold"
    return f


__all__ = [
    "EP_WIRE_DTYPE",
    "SEAM_N_LEAVES",
    "engine_expert_dispatch",
    "ep_combine",
    "ep_routing_census",
    "ep_routing_census_state",
    "expert_dispatch",
    "expert_reduce",
    "set_engine_token_bound",
    "token_shard",
]
