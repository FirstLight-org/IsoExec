"""Linear projections with identical fixed-tree arithmetic across tensor-parallel sizes."""

from __future__ import annotations

import torch

from .allreduce import (
    p2p_available,
    sym_partial,
    tree_all_reduce,
    tree_all_reduce_rounded,
    tree_reduce_scatter,
)
from .gemm import ti_gemm
from .plan import DEFAULT_PLAN, ReductionPlan


def row_parallel_linear(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    plan: ReductionPlan = DEFAULT_PLAN,
    tp_size: int = 1,
    tp_rank: int = 0,
    k_full: int | None = None,
    group=None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Compute a row-parallel projection with a fixed reduction tree."""
    lead, k_local = x.shape[:-1], x.shape[-1]
    x2 = x.reshape(-1, k_local)
    m_leaves = plan.leaves_per_rank(tp_size)

    if tp_size > 1:
        # Owner all-reduce (pik/owner.py): wire planes straight into the transport, ONE reduce launch, the
        # same tree and root round. None = off for this group (agreed collectively) -> the legacy site below.
        from . import owner as _owner

        y = _owner.row_parallel(
            x2, w, bias, plan=plan, tp_size=tp_size, tp_rank=tp_rank, k_full=k_full, group=group,
            out_dtype=out_dtype,
        )
        if y is not None:
            return y.reshape(*lead, -1)

    # fp32 on the wire unless this rank's partial is itself a leaf.
    wire_dtype = torch.bfloat16 if (plan.bf16_leaves and m_leaves == 1) else torch.float32

    # Stage the GEMM output in peer-visible symmetric memory so the all-reduce reads it over NVLink, no copy.
    dst = None
    if tp_size > 1 and p2p_available(group):
        dst = sym_partial((x2.shape[0], w.shape[0]), x2.device, group, dtype=wire_dtype)

    if tp_size == 1 and bias is None and out_dtype == torch.bfloat16:
        # The whole tree is local: the root is rounded where it is stored (the tree-reduce store, the
        # fused tree's store, or the split-1 copy), one fp32 -> bf16 RNE, the same rounding the
        # ``.to(bfloat16)`` below applies to an fp32 root -- one launch fewer, the same bits.
        root = torch.empty((x2.shape[0], w.shape[0]), device=x2.device, dtype=torch.bfloat16)
        return ti_gemm(x2, w, plan=plan, tp_size=1, tp_rank=tp_rank, k_full=k_full, out=root).reshape(*lead, -1)

    part = ti_gemm(x2, w, plan=plan, tp_size=tp_size, tp_rank=tp_rank, k_full=k_full, out=dst)

    # Fuse the BF16 root cast only when bias and leaf-dtype conditions permit it.
    if tp_size > 1 and bias is None and not plan.bf16_leaves and out_dtype == torch.bfloat16:
        return tree_all_reduce_rounded(part, group=group, out_dtype=torch.bfloat16).reshape(*lead, -1)

    full = tree_all_reduce(part, group=group) if tp_size > 1 else part

    # Round the root before bias at every TP size to preserve identical arithmetic.
    if plan.bf16_leaves:
        full = full.to(torch.bfloat16)

    if bias is not None:
        full = full + bias.to(full.dtype)  # after the reduction, and exactly once
    return full.to(out_dtype).reshape(*lead, -1)


# SP keeps each rank's sequence slice of the same leaf-tree projection.
def row_parallel_linear_rs(
    x: torch.Tensor,
    w: torch.Tensor,
    *,
    plan: ReductionPlan = DEFAULT_PLAN,
    tp_size: int = 1,
    tp_rank: int = 0,
    k_full: int | None = None,
    group=None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Compute a row-parallel projection and return this sequence-parallel rank's rows."""
    lead, k_local = x.shape[:-1], x.shape[-1]
    assert len(lead) >= 1 and lead[0] % tp_size == 0, (
        f"row_parallel_linear_rs: sequence dim {lead[0] if lead else None} must divide tp_size "
        f"{tp_size} (x shape {tuple(x.shape)})"
    )
    x2 = x.reshape(-1, k_local)

    # Match the all-reduce path's direct staging: the P2P reduce-scatter can consume the GEMM
    # destination without copying the full stream into its peer-visible input buffer.
    dst = None
    if tp_size > 1 and p2p_available(group):
        wire_dtype = torch.bfloat16 if plan.bf16_leaves and plan.leaves_per_rank(tp_size) == 1 else torch.float32
        dst = sym_partial((x2.shape[0], w.shape[0]), x2.device, group, dtype=wire_dtype)
    part = ti_gemm(x2, w, plan=plan, tp_size=tp_size, tp_rank=tp_rank, k_full=k_full, out=dst)

    if tp_size > 1:
        sliced = tree_reduce_scatter(part.view(lead[0], -1), group=group)
    else:
        sliced = part.view(lead[0], -1)

    # Round the ROOT before the out_dtype cast, same rule as row_parallel_linear; elementwise, so
    # slice-invariant.
    if plan.bf16_leaves:
        sliced = sliced.to(torch.bfloat16)

    out_lead = (lead[0] // tp_size, *lead[1:])
    return sliced.to(out_dtype).reshape(*out_lead, -1)


