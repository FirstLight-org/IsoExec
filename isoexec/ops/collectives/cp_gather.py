"""Gather context-parallel tokens into packed order while preserving CP=1 attention results."""

from __future__ import annotations

import weakref
from dataclasses import dataclass
from typing import Callable

import torch

from isoexec.schema import Placement
from isoexec.ops.collectives.cp_layout import gathered_to_global, local_positions, local_to_global
from isoexec.ops.collectives.spec import CP_LAYOUT


class CpGatherError(ValueError):
    """The stream, the layout and the placement do not describe the same context-parallel cut."""


@dataclass(frozen=True)
class CpLayout:
    """One microbatch's zigzag layout as device tensors (built once, used by every attention row)."""

    cp: int
    rank: int
    total: int  #: global packed tokens (padded)
    local: int  #: this rank's tokens
    perm: torch.Tensor  #: ``global = gathered.index_select(0, perm)``
    inv_perm: torch.Tensor  #: ``gathered = global.index_select(0, inv_perm)``
    mine: torch.Tensor  #: global index of every local row, in local order
    positions: torch.Tensor  #: position of every local token inside its own sequence


def build_layout(cu_seqlens: "list[int]", cp: int, rank: int, device) -> CpLayout:
    perm = gathered_to_global(cu_seqlens, cp)
    inv = [0] * len(perm)
    for g, src in enumerate(perm):
        inv[src] = g
    mine = local_to_global(cu_seqlens, cp, rank)

    def t(values):
        return torch.tensor(values, dtype=torch.long, device=device)

    return CpLayout(
        cp=cp, rank=rank, total=len(perm), local=len(mine), perm=t(perm), inv_perm=t(inv), mine=t(mine),
        positions=t(local_positions(cu_seqlens, cp, rank)),
    )


#: Cache layouts by tensor identity using weak references.
_LAYOUTS: "dict[int, tuple[weakref.ref, CpLayout]]" = {}


def register_layout(cu_seqlens: torch.Tensor, layout: CpLayout) -> None:
    key = id(cu_seqlens)

    def _drop(_ref, key=key):
        _LAYOUTS.pop(key, None)

    _LAYOUTS[key] = (weakref.ref(cu_seqlens, _drop), layout)


def layout_for(cu_seqlens: torch.Tensor, cp: int, rank: int) -> CpLayout:
    """The registered layout of this boundary tensor, or a freshly built one (one host sync) when a
    caller did not register it (tests, a host that drives the seam itself)."""
    entry = _LAYOUTS.get(id(cu_seqlens))
    if entry is not None and entry[0]() is cu_seqlens and (entry[1].cp, entry[1].rank) == (cp, rank):
        return entry[1]
    layout = build_layout([int(v) for v in cu_seqlens.tolist()], cp, rank, cu_seqlens.device)
    register_layout(cu_seqlens, layout)
    return layout


def _cp_axis(placement: Placement):
    axis = placement.cp
    if axis.size != 2:
        raise CpGatherError(
            f"context-parallel gather at cp={axis.size}: the adjoint's SUM is order-free only for two "
            "contributors (one commutative add per element); other degrees need a fixed-order reduction"
        )
    if axis.group is None:
        raise CpGatherError("Placement.cp carries no process group; the host must hand in the live CP group")
    return axis


class _GatherGlobal(torch.autograd.Function):
    """``[T_local, ...]`` on every CP rank -> the GLOBAL packed ``[T, ...]`` (bytes), adjoint: un-permute
    and reduce-scatter SUM (one add per element at cp=2)."""

    @staticmethod
    def forward(ctx, x, group, layout: CpLayout):  # noqa: D102
        import torch.distributed as dist

        if x.shape[0] != layout.local:
            raise CpGatherError(f"stream has {x.shape[0]} rows but this rank's layout holds {layout.local}")
        x = x.contiguous()
        gathered = torch.empty((layout.cp * layout.local, *x.shape[1:]), dtype=x.dtype, device=x.device)
        dist.all_gather_into_tensor(gathered, x, group=group)
        ctx.group, ctx.layout = group, layout
        return gathered.index_select(0, layout.perm)

    @staticmethod
    def backward(ctx, grad):  # noqa: D102
        import torch.distributed as dist

        layout = ctx.layout
        ordered = grad.index_select(0, layout.inv_perm).contiguous()  # back to rank order, rank r's chunk at r
        out = torch.empty((layout.local, *grad.shape[1:]), dtype=grad.dtype, device=grad.device)
        dist.reduce_scatter_tensor(out, ordered, op=dist.ReduceOp.SUM, group=ctx.group)
        return out, None, None


class _KeepOwned(torch.autograd.Function):
    """The rows of this rank's own tokens out of a global stream; the adjoint writes them back into
    zeros with ``index_copy_`` (unique indices: deterministic, unlike ``index_add_``)."""

    @staticmethod
    def forward(ctx, x, mine):  # noqa: D102
        ctx.save_for_backward(mine)
        ctx.shape = tuple(x.shape)
        return x.index_select(0, mine)

    @staticmethod
    def backward(ctx, grad):  # noqa: D102
        (mine,) = ctx.saved_tensors
        out = torch.zeros(ctx.shape, dtype=grad.dtype, device=grad.device)
        out.index_copy_(0, mine, grad.contiguous())
        return out, None


def make_cp_gather(placement: Placement) -> Callable[[torch.Tensor, CpLayout], torch.Tensor]:
    """``gather(x_local, layout) -> x_global`` over ``placement.cp`` (forward runs with autograd off too)."""
    axis = _cp_axis(placement)

    def gather(x: torch.Tensor, layout: CpLayout) -> torch.Tensor:
        return _GatherGlobal.apply(x, axis.group, layout)

    return gather


def make_cp_attention_core(inner: Callable[..., torch.Tensor], placement: Placement) -> Callable[..., torch.Tensor]:
    """Wrap attention with context-parallel gathering and local query selection."""
    axis = _cp_axis(placement)
    gather = make_cp_gather(placement)

    def core(q, k, v, cu_seqlens, max_q, scale):
        layout = layout_for(cu_seqlens, axis.size, axis.rank)
        out = inner(gather(q, layout), gather(k, layout), gather(v, layout), cu_seqlens, max_q, scale)
        if out.shape[0] != layout.total:
            raise CpGatherError(f"the inner attention core returned {out.shape[0]} rows for {layout.total} tokens")
        return _KeepOwned.apply(out, layout.mine)

    core.cp_layout = CP_LAYOUT
    return core


__all__ = [
    "CpGatherError",
    "CpLayout",
    "build_layout",
    "layout_for",
    "make_cp_attention_core",
    "make_cp_gather",
    "register_layout",
]
