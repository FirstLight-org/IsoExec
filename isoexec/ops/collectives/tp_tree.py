"""Row-parallel K reduction using fixed leaves and a balanced fp32 tree.

Placement callers and kernel factories own reduction settings. Native model
interfaces receive parallelism semantics through ``pg_collection`` without also
selecting kernel implementation settings such as leaf count and leaf dtype.
The native factory consumes the TP group and resolves the existing kernel
settings internally.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from isoexec.core.bindings import BoundOp
from isoexec.schema import Placement, ShardError

if TYPE_CHECKING:
    from isoexec.ops.collectives.pik.plan import ReductionPlan


def _tree_domain(placement: Placement, what: str) -> None:
    """Require a power-of-two leaf count with whole leaves owned by each TP rank.
    Placement validates dtype; table validation separately checks agreement with the recorded plan."""
    g, tp = int(placement.leaves), placement.tp.size
    if g < 1 or (g & (g - 1)) != 0:
        raise ShardError(f"{what}: leaves must be a power of two, got {g}")
    if tp > g or g % tp != 0:
        raise ShardError(f"{what}: tp={tp} must divide leaves={g} (every rank owns whole leaves of the fixed tree)")


def _reduction_plan(placement: Placement):
    from isoexec.ops.collectives.pik_bootstrap import ensure_pik

    ensure_pik()
    from pik import ReductionPlan  # type: ignore

    return ReductionPlan(
        num_leaves=int(placement.leaves),
        leaf_dtype=torch.bfloat16 if placement.leaf_dtype == "bf16" else torch.float32,
    )


def make_row_parallel(placement: Placement) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Compute local projection partials and combine them with the fixed TP tree."""
    _tree_domain(placement, "make_row_parallel")
    tp, rank, group = placement.tp.size, placement.tp.rank, placement.tp.group
    if tp > 1 and group is None:
        raise ShardError("Placement.tp.group is None at tp>1: shard() never creates groups (host-owned handles)")
    plan_box: list = []

    def _plan():
        if not plan_box:
            plan_box.append(_reduction_plan(placement))
        return plan_box[0]

    return _make_row_parallel(tp=tp, rank=rank, group=group, get_plan=_plan)


def _make_row_parallel(
    *, tp: int, rank: int, group: object | None, get_plan: Callable[[], ReductionPlan],
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Build the existing projection and autograd callable for a resolved TP group."""
    def _forward(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        plan = get_plan()  # bootstraps the vendored package before its children are imported
        from pik.linear import row_parallel_linear  # type: ignore

        k_full = int(w.shape[1]) * tp
        with torch.no_grad():
            return row_parallel_linear(
                x, w, None, plan=plan, tp_size=tp, tp_rank=rank, k_full=k_full, group=group, out_dtype=x.dtype
            )

    class _Fn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, w):
            from isoexec.ops.mm.wgrad import fusable

            ctx.save_for_backward(x, w)
            ctx.wsink = fusable(w) if w.requires_grad else None
            return _forward(x, w)

        @staticmethod
        def backward(ctx, dy):
            from isoexec.ops.mm.wgrad import weight_grad

            x, w = ctx.saved_tensors
            dx = torch.ops.aten.mm.out(
                dy, w, out=torch.empty(dy.shape[0], w.shape[1], dtype=dy.dtype, device=dy.device)
            )
            dw = weight_grad(dy, x, w, ctx.wsink) if ctx.needs_input_grad[1] else None
            return dx, dw

    def row_parallel(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        if x.requires_grad or w.requires_grad:
            return _Fn.apply(x, w)
        return _forward(x, w)

    def with_handoff(handoff) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """The same site whose no-grad reduction is deferred to ``handoff``'s fused add+norm (pik.owner
        ``RootHandoff``). Same bits: the deferred path is admitted by a bit compare; outside its domain (grad,
        tp=1, owner path off, too large) this is ``row_parallel`` itself."""

        def row_parallel_handoff(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            x = x.contiguous()
            if tp > 1 and x.dim() == 2 and not (torch.is_grad_enabled() and (x.requires_grad or w.requires_grad)):
                plan = get_plan()
                from pik import owner as _owner  # type: ignore

                with torch.no_grad():
                    y = _owner.row_parallel_deferred(
                        x, w, None, plan=plan, tp_size=tp, tp_rank=rank, k_full=int(w.shape[1]) * tp, group=group,
                        out_dtype=x.dtype, handoff=handoff,
                    )
                if y is not None:
                    return y
            # Not deferred: a root left by an aborted earlier forward must not meet this site's norm.
            handoff.pending = None
            return row_parallel(x, w)

        row_parallel_handoff.__name__ = row_parallel_handoff.__qualname__ = f"pik_row_parallel_handoff_tp{tp}"
        return row_parallel_handoff

    row_parallel.__name__ = row_parallel.__qualname__ = f"pik_row_parallel_tp{tp}"
    row_parallel.with_handoff = with_handoff  # type: ignore[attr-defined]
    return row_parallel


def bind_row_parallel(*, tp_group: torch.distributed.ProcessGroup, site: str, sp: bool = False) -> BoundOp:
    """Bind a row-parallel projection using the existing kernel settings.

    ``tp_group`` supplies parallelism semantics: TP size, rank, and group handle.
    Placement-based callers and kernel factories own reduction settings. Keeping
    leaf count and leaf dtype here avoids making native model interfaces select
    kernel implementation settings in addition to the parallelism semantics
    supplied by ``pg_collection``. Under sequence parallelism, the same tree
    returns this rank's token slice, as ``Seams.row_parallel`` does.
    """
    from isoexec.ops._resolve import resolution
    from isoexec.ops.collectives.pik_bootstrap import ensure_pik
    from isoexec.ops.collectives.pik_tp_invariant import _leaf_dtype, _num_leaves

    tp = tp_group.size()
    leaves, dtype = _num_leaves(), _leaf_dtype()
    if tp > leaves or leaves % tp:
        raise ShardError(f"{site}: TP={tp} must divide leaves={leaves}")
    plan = ensure_pik().ReductionPlan(num_leaves=leaves, leaf_dtype=dtype)
    scattered = sp and tp > 1
    make = _make_row_parallel_rs if scattered else _make_row_parallel
    fn = make(tp=tp, rank=tp_group.rank(), group=tp_group, get_plan=lambda: plan)
    return resolution(
        "collectives.row_parallel", "pik_tree_rs" if scattered else "pik_tree", fn,
        provider="ops.collectives", site=site,
        inputs={"tp": tp, "leaves": plan.num_leaves,
                "leaf_dtype": "bf16" if plan.leaf_dtype == torch.bfloat16 else "fp32"},
    )


def make_row_parallel_rs(placement: Placement) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Reduce projection partials with the fixed tree and retain this SP rank's token slice."""
    _tree_domain(placement, "make_row_parallel_rs")
    tp, rank, group = placement.tp.size, placement.tp.rank, placement.tp.group
    if tp < 2 or group is None:
        raise ShardError("make_row_parallel_rs needs tp >= 2 and a live TP group (host-owned handle)")
    plan_box: list = []

    def _plan():
        if not plan_box:
            plan_box.append(_reduction_plan(placement))
        return plan_box[0]

    return _make_row_parallel_rs(tp=tp, rank=rank, group=group, get_plan=_plan)


def _make_row_parallel_rs(
    *, tp: int, rank: int, group: object, get_plan: Callable[[], ReductionPlan],
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Build the existing scattered projection and autograd callable for a resolved TP group."""
    def _forward(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        plan = get_plan()
        from pik.linear import row_parallel_linear_rs  # type: ignore

        k_full = int(w.shape[1]) * tp
        with torch.no_grad():
            return row_parallel_linear_rs(
                x, w, plan=plan, tp_size=tp, tp_rank=rank, k_full=k_full, group=group, out_dtype=x.dtype
            )

    class _Fn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, w):
            from isoexec.ops.mm.wgrad import fusable

            ctx.save_for_backward(x, w)
            ctx.wsink = fusable(w) if w.requires_grad else None
            return _forward(x, w)

        @staticmethod
        def backward(ctx, dy):
            from isoexec.ops.collectives.tp_bytes import _gather_dim0
            from isoexec.ops.mm.wgrad import weight_grad

            x, w = ctx.saved_tensors
            dy = dy.contiguous()
            full = _gather_dim0(dy, tp, group)  # bytes: rank-ordered chunks of dY
            dx = torch.ops.aten.mm.out(
                full, w, out=torch.empty(full.shape[0], w.shape[1], dtype=full.dtype, device=full.device)
            )
            dw = weight_grad(full, x, w, ctx.wsink) if ctx.needs_input_grad[1] else None
            return dx, dw

    def row_parallel_rs(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        if x.requires_grad or w.requires_grad:
            return _Fn.apply(x, w)
        return _forward(x, w)

    row_parallel_rs.__name__ = row_parallel_rs.__qualname__ = f"pik_row_parallel_rs_tp{tp}"
    return row_parallel_rs


def make_row_parallel_partial(placement: Placement) -> Callable[..., torch.Tensor]:
    """Compute the local projection partial while deferring the cross-rank tree reduction."""
    _tree_domain(placement, "make_row_parallel_partial")
    tp, rank, group = placement.tp.size, placement.tp.rank, placement.tp.group
    if tp > 1 and group is None:
        raise ShardError("Placement.tp.group is None at tp>1: shard() never creates groups (host-owned handles)")
    plan_box: list = []

    def _plan():
        if not plan_box:
            plan_box.append(_reduction_plan(placement))
        return plan_box[0]

    def wire_dtype() -> torch.dtype:
        plan = _plan()
        # row_parallel_linear / ti_gemm: bf16 on the wire only when this rank's partial IS a leaf.
        return torch.bfloat16 if (plan.bf16_leaves and plan.leaves_per_rank(tp) == 1) else torch.float32

    def finish(part: torch.Tensor, out_dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        plan = _plan()
        from pik.allreduce import tree_all_reduce  # type: ignore

        full = tree_all_reduce(part, group=group)
        if plan.bf16_leaves:
            full = full.to(torch.bfloat16)
        return full.to(out_dtype)

    def partial(x: torch.Tensor, w: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor:
        plan = _plan()
        from pik.gemm import ti_gemm  # type: ignore

        x = x.contiguous()
        lead, k_local = x.shape[:-1], x.shape[-1]
        k_full = int(w.shape[1]) * tp
        with torch.no_grad():
            part = ti_gemm(x.reshape(-1, k_local), w, plan=plan, tp_size=tp, tp_rank=rank, k_full=k_full, out=out)
        return part.reshape(*lead, -1)

    partial.__name__ = partial.__qualname__ = f"pik_row_parallel_partial_tp{tp}"
    partial.partial_only = True  # type: ignore[attr-defined]  -- what a shared-expert binding checks for
    partial.wire_dtype = wire_dtype  # type: ignore[attr-defined]
    partial.finish = finish  # type: ignore[attr-defined]
    partial.group = group  # type: ignore[attr-defined]
    return partial


def make_tree_all_reduce(placement: Placement) -> Callable[..., torch.Tensor]:
    """Build a fixed-tree all-reduce over the placement's TP group."""
    _tree_domain(placement, "make_tree_all_reduce")
    group = placement.tp.group
    plan_box: list = []

    def _plan():
        if not plan_box:
            plan_box.append(_reduction_plan(placement))
        return plan_box[0]

    def tree_all_reduce(partial: torch.Tensor, out=None, root_dtype=None) -> torch.Tensor:
        _plan()  # the same ReductionPlan row_parallel builds; resolving it bootstraps the vendored package
        from pik.allreduce import tree_all_reduce as _ar  # type: ignore

        return _ar(partial, group=group, out=out, root_dtype=root_dtype)

    tree_all_reduce.__name__ = tree_all_reduce.__qualname__ = f"pik_tree_all_reduce_tp{placement.tp.size}"
    return tree_all_reduce


__all__ = [
    "bind_row_parallel",
    "make_row_parallel",
    "make_row_parallel_partial",
    "make_row_parallel_rs",
    "make_tree_all_reduce",
]
