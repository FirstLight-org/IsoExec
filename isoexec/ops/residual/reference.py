"""Plain-torch reference for both residual joins.
Cast sublayer output to h.dtype, then perform one add with fp32 opmath and one bf16 round."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isoexec.core.bindings import BoundOp

if TYPE_CHECKING:  # the plan is DUCK-TYPED at runtime -- ops/reference.py imports this module, never the reverse
    from isoexec.ops.reference import ReferencePlan


def ref_residual_add(h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """``h + y`` in the residual dtype: the sublayer output is cast to ``h.dtype`` at its boundary
    (``boundary_out_dtype``), then ONE bf16-rounded add. (A no-op cast when y is already bf16.)"""
    return h + y.to(h.dtype)


def bindings(plan: "ReferencePlan") -> dict[str, BoundOp]:
    """This family's contribution to ``ops/reference.py::reference_bindings``: ONE function, two sites."""
    del plan
    return {
        op: BoundOp(op, "ref.add", None, "ops.reference", fn=ref_residual_add)
        for op in ("residual.add.attn", "residual.add.mlp")
    }


__all__ = [
    "bindings",
    "ref_residual_add",
]
