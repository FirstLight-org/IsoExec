"""Plain-torch reference for fixed-leaf row-parallel reduction.
It preserves leaf boundaries, leaf rounding and balanced fp32 folding, but torch GEMMs do not reproduce pik leaf bits."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isoexec.core.bindings import BoundOp

if TYPE_CHECKING:  # the plan is DUCK-TYPED at runtime -- ops/reference.py imports this module, never the reverse
    from isoexec.ops.reference import ReferencePlan


#: The leaf plan :func:`bindings` states when its caller names no other -- the production
#: contract (``runtimes/trainer_installs.py::PIK_LEAVES``, i.e. ``ISOEXEC_PIK_LEAVES``'s default) with the
#: bf16 wire every shipped ``Placement`` carries.
REF_LEAVES = 8


REF_LEAF_DTYPE = "bf16"


def ref_row_parallel(*, leaves: int, leaf_dtype: str = REF_LEAF_DTYPE, tp: int = 1):
    """Fold fixed contiguous K leaves in a balanced fp32 tree with one root round.
    Each leaf uses torch GEMM and rounds to leaf_dtype, preserving the plan's expression but not pik leaf bits.
    TP>1 refuses because a local-only reference cannot fold the missing cross-rank subtrees."""
    if leaves < 1 or (leaves & (leaves - 1)) != 0:
        raise ValueError(f"ref_row_parallel: leaves must be a power of two, got {leaves}")
    if leaf_dtype not in ("bf16", "fp32"):
        raise ValueError(f"ref_row_parallel: leaf_dtype must be 'bf16' or 'fp32', got {leaf_dtype!r}")
    round_leaf = leaf_dtype == "bf16"

    def ref_row_parallel_leaf_tree(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        if tp > 1:
            raise NotImplementedError(
                f"ref_row_parallel at tp={tp}: the cross-rank fold of the fixed tree is a collective "
                f"over leaf ranges this rank does not hold, and this module binds no collective. A "
                f"placement at tp>1 takes ops/collectives/tp_tree.py::make_row_parallel, which shard() "
                f"installs; the reference states the whole tree on one rank."
            )
        k = int(x.shape[-1])
        if k % leaves:
            raise ValueError(f"ref_row_parallel: K={k} is not divisible by leaves={leaves} (equal leaves)")
        lk = k // leaves
        parts = [
            x[..., j * lk : (j + 1) * lk].to(torch.float32) @ w[:, j * lk : (j + 1) * lk].to(torch.float32).t()
            for j in range(leaves)
        ]
        if round_leaf:  # the leaf is the rounding point the plan pins; the tree above it is fp32
            parts = [p.to(torch.bfloat16).to(torch.float32) for p in parts]
        while len(parts) > 1:  # combine_order: pairwise levels, left operand the lower leaf index
            parts = [parts[i] + parts[i + 1] for i in range(0, len(parts), 2)]
        return parts[0].to(x.dtype)  # the single root round, at every TP the tree admits

    return ref_row_parallel_leaf_tree


def bindings(plan: "ReferencePlan") -> dict[str, BoundOp]:
    """Build reference bindings from the caller's leaf plan or this family's defaults.
    Reference IDs remain distinct from pik_tree because torch leaf GEMMs have different rounding."""
    leaves = REF_LEAVES if plan.leaves is None else plan.leaves
    leaf_dtype = REF_LEAF_DTYPE if plan.leaf_dtype is None else plan.leaf_dtype
    return {
        "collectives.row_parallel": BoundOp(
            "collectives.row_parallel",
            "ref.leaf_tree",
            None,
            "ops.reference",
            fn=ref_row_parallel(leaves=leaves, leaf_dtype=leaf_dtype),
            inputs={"leaves": leaves, "leaf_dtype": leaf_dtype},
        )
    }


__all__ = [
    "REF_LEAF_DTYPE",
    "REF_LEAVES",
    "bindings",
    "ref_row_parallel",
]
