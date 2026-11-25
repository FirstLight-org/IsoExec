"""Trainer TP=1 row-parallel and all-reduce bindings using the shared pik leaf plan.
Placement-aware component bindings use tp_tree factories; tp_copy defaults to identity."""

from __future__ import annotations

import torch


def _identity(x):
    """The default of the ``tp_copy`` seam: what a TP=1 placement and every engine leave in place."""
    return x


def _pik_plan():
    from isoexec.ops.collectives.pik_tp_invariant import get_plan

    return get_plan()


class _RowParallelFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w):
        from isoexec.ops.mm.wgrad import fusable

        ctx.save_for_backward(x, w)
        ctx.wsink = fusable(w) if w.requires_grad else None
        return _row_parallel_forward(x, w)

    @staticmethod
    def backward(ctx, dy):
        from isoexec.ops.mm.wgrad import weight_grad

        x, w = ctx.saved_tensors
        dx = torch.ops.aten.mm.out(dy, w, out=torch.empty(dy.shape[0], w.shape[1], dtype=dy.dtype, device=dy.device))
        dw = weight_grad(dy, x, w, ctx.wsink) if ctx.needs_input_grad[1] else None
        return dx, dw


def _row_parallel_forward(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    from pik.linear import (
        row_parallel_linear,  # type: ignore  # vendored pik, registered by ensure_pik()
    )

    with torch.no_grad():
        return row_parallel_linear(
            x, w, None, plan=_pik_plan(), tp_size=1, tp_rank=0, k_full=int(w.shape[1]), group=None, out_dtype=x.dtype
        )


def row_parallel_trainer(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """``collectives.row_parallel`` at TP=1: ``pik.linear.row_parallel_linear`` with the trainer's
    plan (G=8 bf16 leaves, fp32 tree, one bf16 root round), the reduction every non-expert row-parallel
    site uses (attn o_proj, GDN out_proj, dense fc2, shared-expert fc2)."""
    x = x.contiguous()
    if x.requires_grad or w.requires_grad:
        return _RowParallelFn.apply(x, w)
    return _row_parallel_forward(x, w)


__all__ = ["row_parallel_trainer"]
