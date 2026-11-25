"""Tensor- and sequence-parallel byte movement with consumer-specific adjoints.
Partial input gradients use fixed-tree reduction; replicated complete gradients select the local chunk,
and discarded rows receive zero gradients."""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable

import torch

from isoexec.core.bindings import VOCAB_PARALLEL, BoundOp, no_movement_binding
from isoexec.schema import Placement


def _pik():
    from isoexec.ops.collectives.pik_bootstrap import ensure_pik

    return ensure_pik()


class TpBytesError(ValueError):
    """A geometry or placement these collectives refuse."""


_SEQ_GATHER_COUNTS = {"p2p_calls": 0, "nccl_calls": 0, "p2p_fallbacks": 0}
_SEQ_GATHER_FIRST_FALLBACK: dict = {}


def seq_gather_transport_counts() -> dict:
    """SP byte-gather telemetry, separate from the fixed-tree reduction engagement hooks."""
    return {**_SEQ_GATHER_COUNTS, "first_fallback": dict(_SEQ_GATHER_FIRST_FALLBACK) or None}


def _tp(placement: Placement) -> tuple[int, int, object]:
    tp, rank, group = placement.tp.size, placement.tp.rank, placement.tp.group
    if tp > 1 and group is None:
        raise TpBytesError(
            "Placement.tp.group is None at tp>1: these collectives never create groups (host-owned handles)"
        )
    return tp, rank, group


def tree_grad_sum(g: torch.Tensor, group) -> torch.Tensor:
    """The fixed-tree TP sum of partial input gradients (``pik.allreduce.tree_all_reduce``: the owner route
    when ``ISOEXEC_PIK_OWNER_ROUTE`` admits it, else the legacy transports)."""
    from isoexec.ops.collectives.pik.allreduce import tree_all_reduce

    return tree_all_reduce(g.contiguous(), group=group)


def lane_grad_sum(g: torch.Tensor, group):
    """``fn(g) -> sum`` that computes ``tree_grad_sum(g)``'s bits with ONLY a symmetric-memory kernel, safe to
    run on a side stream; None when that cannot be guaranteed (then the sum runs serially on the caller's
    stream). Requires the owner route (``ISOEXEC_PIK_OWNER_ROUTE``) on a P2P group, the size below the
    group's P2P cap and the class already admitted: no NCCL call, no admission collective on the lane."""
    if not (g.is_cuda and g.numel() and g.dtype in (torch.bfloat16, torch.float32) and g.is_contiguous()):
        return None
    _pik()
    import pik.owner as _owner  # type: ignore  # canonical name: one module identity (see pik_bootstrap)

    if not _owner._env_on("ISOEXEC_PIK_OWNER_ROUTE"):
        return None
    st = _owner.state(group, g.device)
    if st is None or st.transport != "p2p":
        return None
    import pik.pipeline as _pipe  # type: ignore

    if _pipe.pick_state(st, g.numel() * g.element_size()) is not st:
        return None
    flat = g.view(1, -1)
    mode = _owner.admitted_mode(flat, group, g.dtype, st)
    if mode is None:
        return None
    shape = g.shape
    return lambda t: _owner._execute(st, t.view(1, -1), group, t.dtype, mode).view(shape)


class _TpCopyLink:
    """Ties one tp_copy output to the GEMM that consumes it, so that GEMM's backward can start the tree
    sum of its input gradient on the comm lane BEFORE computing its weight gradient (Megatron/TE's async
    TP all-reduce). Only a sole consumer may do so: with several, autograd sums their gradients first.
    Exists only when the group agreed on overlap (opt-in, more than one hardware queue). The lane runs only a
    symmetric-memory kernel (``early``); anything else (NCCL, an unadmitted class) runs serially."""

    __slots__ = ("group", "reduce", "early", "uses", "pending", "key")

    def __init__(self, group, reduce, early):
        self.group = group
        self.reduce = reduce  # the seam's adjoint collective: g -> its fixed-tree sum (or reduce-scatter)
        self.early = early  # g -> lane-safe fn computing reduce(g)'s bits, or None
        self.uses = 0
        self.pending = None
        self.key = None

    def launch(self, dx: torch.Tensor) -> None:
        fn = self.early(dx)
        if fn is None:  # not lane-safe: the adjoint runs serially when autograd reaches it
            _LINK_COUNTS["not_lane_safe"] += 1
            return
        _pik()
        import pik.lane as _lane  # type: ignore

        self.key = (dx.data_ptr(), dx._version, tuple(dx.shape), dx.dtype)
        self.pending = _lane.submit(self.group, fn, dx)
        _LINK_COUNTS["early"] += 1

    def resolve(self, g: torch.Tensor):
        """The adjoint of ``g``: the early result if ``g`` is exactly the gradient launched, else run it now."""
        if self.pending is not None:
            pending, self.pending = self.pending, None
            if self.matches(g):  # g IS the sole consumer's dgrad, already reduced on the lane
                _LINK_COUNTS["consumed"] += 1
                return pending.wait()
            # Something else added into g after the launch: the early result is not this gradient. Every
            # rank took the same branch (same graph), so the collective sequence stays aligned; redo it.
            pending.wait()
            _LINK_COUNTS["discarded"] += 1
        _LINK_COUNTS["serial"] += 1
        return self.reduce(g)

    def matches(self, g: torch.Tensor) -> bool:
        return self.key == (g.data_ptr(), g._version, tuple(g.shape), g.dtype)


_LINK_COUNTS = {"early": 0, "consumed": 0, "discarded": 0, "serial": 0, "not_lane_safe": 0}


def tp_copy_link_counts() -> dict:
    return dict(_LINK_COUNTS)


class _TpCopy(torch.autograd.Function):
    """Keep inputs unchanged; sum partial gradients with the fixed TP tree."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, group, link):  # type: ignore[override]
        ctx.group = group
        ctx.link = link
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g: torch.Tensor):  # type: ignore[override]
        if ctx.link is not None:
            return ctx.link.resolve(g), None, None
        _LINK_COUNTS["serial"] += 1
        return tree_grad_sum(g, ctx.group), None, None


def _new_link(x: torch.Tensor, group, reduce, early):
    """A link when the group agreed on overlap (opt-in, eager, CUDA, >1 hardware queue), else None."""
    if not x.is_cuda:
        return None
    _pik()
    import pik.lane as _lane  # type: ignore

    return _TpCopyLink(group, reduce, early) if _lane.overlap_agreed(group, x.device) else None


def make_tp_copy(placement: Placement) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build an identity-forward, fixed-tree-gradient TP operation."""
    tp, _rank, group = _tp(placement)
    return _make_tp_copy(tp, group)


def _make_tp_copy(tp, group):
    def tp_copy(x: torch.Tensor) -> torch.Tensor:
        if not x.requires_grad or not torch.is_grad_enabled():
            return x
        link = _new_link(x, group, lambda t: tree_grad_sum(t, group), lambda t: lane_grad_sum(t, group))
        y = _TpCopy.apply(x, group, link)
        if link is not None:
            y._isoexec_tp_copy = link
        return y

    tp_copy.__name__ = tp_copy.__qualname__ = f"tp_copy_tp{tp}"
    return tp_copy


def bind_tp_copy(*, tp_group: torch.distributed.ProcessGroup, site: str, sp: bool = False) -> BoundOp:
    """Bind identity forward and a fixed-tree sum of partial gradients.

    Under sequence parallelism, gather tokens in forward and reduce-scatter their partial
    gradients in backward, as ``Seams.tp_copy`` does.
    """
    from isoexec.ops._resolve import resolution
    from isoexec.ops.collectives.pik_tp_invariant import _leaf_dtype, _num_leaves

    tp = tp_group.size()
    if tp == 1:
        return no_movement_binding("collectives.tp_copy", tp=1, side="trainer", sp=sp)
    impl = "seq_all_gather_fwd_tree_rs_bwd" if sp else "identity_fwd_tree_sum_bwd"
    fn = _make_seq_all_gather(tp, tp_group) if sp else _make_tp_copy(tp, tp_group)
    return resolution("collectives.tp_copy", impl, fn, provider="ops.collectives", site=site,
                      inputs={"tp": tp, "side": "trainer", "sp": sp, "leaves": _num_leaves(),
                              "leaf_dtype": "bf16" if _leaf_dtype() == torch.bfloat16 else "fp32"})


class _AllGather(torch.autograd.Function):
    """Concatenate peer columns in rank order; reduce-scatter their gradients."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, tp: int, group, lo: int, hi: int):  # type: ignore[override]
        import torch.distributed as dist

        T, n = int(x.shape[0]), int(x.shape[1])
        ctx.shape, ctx.group, ctx.slots, ctx.tp = (T, n), group, (lo, hi), tp
        if T == 0:
            return x.new_empty((0, (hi - lo) * n))
        if x.device.type == "meta":  # shape propagation (meta build, fake-tensor tracing): no wire
            return x.new_empty((T, (hi - lo) * n))
        # Allocate a flat receive buffer for Gloo, then reshape the same bytes.
        flat = x.new_empty(tp * T, n)
        dist.all_gather_into_tensor(flat, x.contiguous(), group=group)
        return flat.view(tp, T, n)[lo:hi].permute(1, 0, 2).reshape(T, (hi - lo) * n)

    @staticmethod
    def backward(ctx, g: torch.Tensor):  # type: ignore[override]
        from isoexec.ops.collectives.pik.allreduce import tree_reduce_scatter

        T, n = ctx.shape
        lo, hi = ctx.slots
        if T == 0:
            return g.new_empty((0, n)), None, None, None, None
        gbuf = g.new_zeros(ctx.tp, T, n)
        gbuf[lo:hi] = g.reshape(T, hi - lo, n).permute(1, 0, 2)
        dx = tree_reduce_scatter(gbuf.reshape(ctx.tp * T, n), group=ctx.group, root_dtype=g.dtype)
        return dx, None, None, None, None


def make_all_gather(placement: Placement) -> Callable[..., torch.Tensor]:
    """Build a column gather with a fixed-tree reduce-scatter adjoint."""
    tp, _rank, group = _tp(placement)
    return _make_all_gather(tp, group)


def _make_all_gather(tp, group):
    def all_gather(x: torch.Tensor, *, select: tuple[int, int] | None = None) -> torch.Tensor:
        lo, hi = (0, tp) if select is None else select
        if not (0 <= lo < hi <= tp):
            raise TpBytesError(f"select={select!r} is not a range of rank slots inside [0, {tp})")
        if x.dim() != 2:
            raise TpBytesError(f"all_gather takes a [T, n] activation, got shape {tuple(x.shape)}")
        return _AllGather.apply(x, tp, group, lo, hi)

    all_gather.__name__ = all_gather.__qualname__ = f"all_gather_tp{tp}"
    return all_gather


def make_kv_gather(placement: Placement, hk_total: int) -> Callable[[torch.Tensor], torch.Tensor]:
    """Gather split KV heads when the TP degree exceeds the head count."""
    tp, rank, group = _tp(placement)
    return _make_kv_gather(tp, rank, group, hk_total)


def _make_kv_gather(tp, rank, group, hk_total):
    if hk_total < 1 or tp % hk_total != 0:
        raise TpBytesError(f"tp={tp} is not a multiple of the {hk_total} declared kv heads (cannot gather a head)")
    if hk_total >= tp:
        raise TpBytesError(f"{hk_total} kv heads at tp={tp}: nothing to gather, the head is whole on every rank")
    step = tp // hk_total
    u = rank // step
    gather = _make_all_gather(tp, group)
    fn = functools.partial(gather, select=(u * step, (u + 1) * step))
    fn.__name__ = fn.__qualname__ = f"kv_gather_tp{tp}_heads{hk_total}"  # type: ignore[attr-defined]
    return fn


def bind_kv_gather(*, tp_group: torch.distributed.ProcessGroup, kv_heads: int, site: str) -> BoundOp:
    """Bind the trainer gather of whole KV heads from ranks that hold head fractions."""
    from isoexec.ops._resolve import resolution

    tp = tp_group.size()
    if tp == 1 or kv_heads >= tp:
        return no_movement_binding("collectives.all_gather", tp=tp, side="trainer", kv_heads=kv_heads)
    return resolution("collectives.all_gather", "nccl_bytes_tree_rs_bwd",
                      _make_kv_gather(tp, tp_group.rank(), tp_group, kv_heads), provider="ops.collectives",
                      site=site, inputs={"tp": tp, "side": "trainer", "kv_heads": kv_heads})


# Sequence parallelism uses rank-ordered token slices and fixed-tree gradient reductions.


def _seq_chunk(total: int, tp: int, what: str) -> int:
    if total % tp != 0:
        raise TpBytesError(
            f"{what}: {total} packed tokens do not divide tp={tp}. Sequence parallelism slices the token "
            "stream into equal rank-ordered chunks; the host must pad the packed stream to a multiple of TP"
        )
    return total // tp


def _gather_dim0_p2p(x: torch.Tensor, tp: int, group) -> torch.Tensor:
    """Rank-ordered byte concatenation from the existing double-buffered P2P input pool."""
    from isoexec.ops.collectives.pik import allreduce as ar
    from pik.lane import fence  # type: ignore

    fence(x.device)
    n = x.numel()
    pool = ar._sym_in_pool(n, x.device, group, x.dtype)
    inp, handle = pool.stage()
    if x.data_ptr() != inp.data_ptr():
        inp[:n].copy_(x.view(-1))
    handle.barrier()
    peers = [handle.get_buffer(rank, tuple(x.shape), x.dtype) for rank in range(tp)]
    # Same-dtype cat copies bytes, including signed zero and NaN payloads. It also avoids a
    # generated kernel specialization for every packed token count seen during training.
    out = torch.cat(peers, dim=0)
    pool.flip()
    return out


def _gather_dim0(x: torch.Tensor, tp: int, group) -> torch.Tensor:
    import torch.distributed as dist

    x = x.contiguous()
    shape = (tp * int(x.shape[0]), *x.shape[1:])
    if x.device.type == "meta" or x.numel() == 0:  # shape propagation / empty stream: no wire
        return x.new_empty(shape)
    if x.is_cuda:
        from isoexec.ops.collectives.pik import allreduce as ar

        if ar.p2p_available(group) and ar._p2p_size_ok(x, group):
            try:
                gathered = _gather_dim0_p2p(x, tp, group)
            except Exception as exc:  # noqa: BLE001 -- retain the NCCL byte-copy fallback
                _SEQ_GATHER_COUNTS["p2p_fallbacks"] += 1
                if not _SEQ_GATHER_FIRST_FALLBACK:
                    _SEQ_GATHER_FIRST_FALLBACK.update(
                        exception=repr(exc), shape=tuple(x.shape), dtype=str(x.dtype), world=tp
                    )
                    logging.getLogger(__name__).warning(
                        "SP P2P byte-gather failed; using NCCL: %r. Read seq_gather_transport_counts().", exc
                    )
            else:
                _SEQ_GATHER_COUNTS["p2p_calls"] += 1
                return gathered
    out = x.new_empty(shape)
    dist.all_gather_into_tensor(out, x, group=group)
    _SEQ_GATHER_COUNTS["nccl_calls"] += 1
    return out


def tree_grad_reduce_scatter(g: torch.Tensor, group) -> torch.Tensor:
    """The fixed-tree reduce-scatter of partial input gradients over the sequence dim (SP adjoint)."""
    from isoexec.ops.collectives.pik.allreduce import tree_reduce_scatter

    return tree_reduce_scatter(g.contiguous(), group=group, root_dtype=g.dtype)


def lane_grad_reduce_scatter(g: torch.Tensor, group):
    """``fn(g)`` computing ``tree_grad_reduce_scatter(g)``'s bits with only the symmetric-memory kernel (safe on
    a side stream), or None (then it runs serially): a P2P group, the size below the group's P2P cap."""
    from isoexec.ops.collectives.pik import allreduce as ar

    world = ar.dist.get_world_size(group)
    if not (g.is_cuda and g.numel() and g.is_contiguous() and g.dtype in (torch.bfloat16, torch.float32)
            and g.shape[0] % world == 0 and ar.p2p_available(group) and ar._p2p_size_ok(g, group)):
        return None
    out_shape = (g.shape[0] // world, *g.shape[1:])

    def fn(t):
        mine = ar._tree_reduce_scatter_p2p(t, group).view(out_shape)
        return mine.to(torch.bfloat16) if t.dtype == torch.bfloat16 else mine

    return fn


class _SeqAllGather(torch.autograd.Function):
    """Gather sequence chunks for column-parallel consumers; reduce-scatter their partial gradients."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, tp: int, group, link):  # type: ignore[override]
        ctx.group = group
        ctx.link = link
        return _gather_dim0(x, tp, group)

    @staticmethod
    def backward(ctx, g: torch.Tensor):  # type: ignore[override]
        if ctx.link is not None:  # early reduce-scatter of the sole consumer's dgrad (under its wgrad)
            return ctx.link.resolve(g), None, None, None
        return tree_grad_reduce_scatter(g, ctx.group), None, None, None


class _SeqAllGatherReplicated(torch.autograd.Function):
    """The same gather for consumers whose ``dx`` is already COMPLETE and identical on every rank
    (a replicated-weight GEMM such as the MoE router, or a block that sums internally): the backward
    keeps this rank's chunk and sums nothing -- a SUM here would count the gradient ``tp`` times."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, tp: int, rank: int, group):  # type: ignore[override]
        ctx.tp, ctx.rank = tp, rank
        return _gather_dim0(x, tp, group)

    @staticmethod
    def backward(ctx, g: torch.Tensor):  # type: ignore[override]
        chunk = int(g.shape[0]) // ctx.tp
        return g.narrow(0, ctx.rank * chunk, chunk).contiguous(), None, None, None


class _SeqScatter(torch.autograd.Function):
    """Copy this rank's sequence slice; gather complete gradients without floating-point addition."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, tp: int, rank: int, group):  # type: ignore[override]
        ctx.tp, ctx.group = tp, group
        chunk = _seq_chunk(int(x.shape[0]), tp, "seq_scatter")
        # A copy, not a view: the residual stream is updated in place downstream, and an autograd
        # Function output that aliases its input may not be written to.
        return x.narrow(0, rank * chunk, chunk).clone()

    @staticmethod
    def backward(ctx, g: torch.Tensor):  # type: ignore[override]
        return _gather_dim0(g, ctx.tp, ctx.group), None, None, None


def make_seq_all_gather(placement: Placement) -> Callable[[torch.Tensor], torch.Tensor]:
    """``collectives.tp_copy`` under ``Placement.sp``: the module-input seam becomes the sequence
    all-gather (forward bytes, fixed-tree reduce-scatter backward)."""
    tp, _rank, group = _tp(placement)
    return _make_seq_all_gather(tp, group)


def _make_seq_all_gather(tp, group):
    def seq_all_gather(x: torch.Tensor) -> torch.Tensor:
        link = None
        if x.requires_grad and torch.is_grad_enabled():
            link = _new_link(x, group, lambda t: tree_grad_reduce_scatter(t, group),
                             lambda t: lane_grad_reduce_scatter(t, group))
        y = _SeqAllGather.apply(x, tp, group, link)
        if link is not None:
            y._isoexec_tp_copy = link
        return y

    seq_all_gather.__name__ = seq_all_gather.__qualname__ = f"seq_all_gather_tp{tp}"
    return seq_all_gather


def make_seq_all_gather_replicated(placement: Placement) -> Callable[[torch.Tensor], torch.Tensor]:
    """The gather whose backward is a slice (see :class:`_SeqAllGatherReplicated`)."""
    tp, rank, group = _tp(placement)

    def seq_all_gather_replicated(x: torch.Tensor) -> torch.Tensor:
        return _SeqAllGatherReplicated.apply(x, tp, rank, group)

    seq_all_gather_replicated.__name__ = seq_all_gather_replicated.__qualname__ = f"seq_all_gather_replicated_tp{tp}"
    return seq_all_gather_replicated


def make_seq_scatter(placement: Placement) -> Callable[[torch.Tensor], torch.Tensor]:
    """Whole replicated stream -> this rank's chunk (forward slice, all-gather backward)."""
    tp, rank, group = _tp(placement)
    return _make_seq_scatter(tp, rank, group)


def _make_seq_scatter(tp, rank, group):
    def seq_scatter(x: torch.Tensor) -> torch.Tensor:
        return _SeqScatter.apply(x, tp, rank, group)

    seq_scatter.__name__ = seq_scatter.__qualname__ = f"seq_scatter_tp{tp}"
    return seq_scatter


def bind_seq_scatter(*, tp_group: torch.distributed.ProcessGroup, site: str) -> BoundOp:
    """Bind the trainer scatter of the whole residual stream to this rank's sequence chunk."""
    from isoexec.ops._resolve import resolution

    tp, rank = tp_group.size(), tp_group.rank()
    return resolution("collectives.sp_scatter", "seq_scatter_fwd_all_gather_bwd",
                      _make_seq_scatter(tp, rank, tp_group), provider="ops.collectives", site=site,
                      inputs={"tp": tp, "sp": True, "rank": rank})


# Vocabulary parallelism. The forward gathers bytes and selects; the adjoint masks and sums
# nothing, because a token's embedding row has exactly one owning shard. That is why it sits
# beside the movements above rather than beside the fixed reduction tree.


class _OwnerSelect(torch.autograd.Function):
    """Select each token's owning vocabulary shard by copying bytes; mask local gradients in backward."""

    @staticmethod
    def forward(ctx, local: torch.Tensor, owner: torch.Tensor, rank: int, tp: int, group):  # type: ignore[override]
        import torch.distributed as dist

        T, H = int(local.shape[0]), int(local.shape[1])
        ctx.save_for_backward(owner)
        ctx.rank = rank
        # Use a flat receive buffer for Gloo compatibility without changing the gathered bytes.
        flat = local.new_empty(tp * T, H)
        dist.all_gather_into_tensor(flat, local.contiguous(), group=group)
        return flat.view(tp, T, H)[owner, torch.arange(T, device=local.device)]

    @staticmethod
    def backward(ctx, g: torch.Tensor):  # type: ignore[override]
        (owner,) = ctx.saved_tensors
        return g * (owner == ctx.rank).unsqueeze(1).to(g.dtype), None, None, None, None


def make_vocab_parallel_embed(base_fn, placement: Placement):
    """Gather local embedding lookups and select the owning shard without altering signed zeros."""
    tp, rank, group = placement.tp.size, placement.tp.rank, placement.tp.group
    return _make_vocab_parallel_embed(base_fn, tp, rank, group)


def _make_vocab_parallel_embed(base_fn, tp, rank, group):

    def f(ids: torch.Tensor, w_local: torch.Tensor) -> torch.Tensor:
        rows = int(w_local.shape[0])
        start = rank * rows
        owner = torch.div(ids, rows, rounding_mode="floor")
        local = base_fn((ids - start).clamp(0, rows - 1), w_local)
        return _OwnerSelect.apply(local, owner, rank, tp, group)

    f.__name__ = f.__qualname__ = f"vocab_parallel_embed_tp{tp}"
    return f


def bind_vocab_parallel_embed(base: BoundOp, *, tp_group: torch.distributed.ProcessGroup) -> BoundOp:
    """Bind local lookup followed by byte selection of each token's owner."""
    from dataclasses import replace

    tp = tp_group.size()
    if tp == 1:
        return base
    return replace(base, fn=_make_vocab_parallel_embed(base.fn, tp, tp_group.rank(), tp_group),
                   inputs={**base.inputs, "vocab_parallel": VOCAB_PARALLEL})


__all__ = [
    "TpBytesError",
    "bind_seq_scatter",
    "bind_tp_copy",
    "bind_vocab_parallel_embed",
    "make_all_gather",
    "make_kv_gather",
    "make_seq_all_gather",
    "make_seq_all_gather_replicated",
    "make_seq_scatter",
    "make_tp_copy",
    "make_vocab_parallel_embed",
]
