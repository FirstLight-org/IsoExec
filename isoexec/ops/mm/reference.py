"""Plain-torch GEMM reference with fp32 accumulation over the whole K dimension and one output round.
Fixed-leaf row-parallel reduction has separate leaf rounding and uses collectives/reference.py."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isoexec.core.bindings import BoundOp

if TYPE_CHECKING:  # the plan is DUCK-TYPED at runtime -- ops/reference.py imports this module, never the reverse
    from isoexec.ops.reference import ReferencePlan


def ref_mm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """y = x @ w.T with fp32 accumulation, rounded to x's dtype (the mm family's boundary)."""
    return (x.to(torch.float32) @ w.to(torch.float32).t()).to(x.dtype)


def bindings(plan: "ReferencePlan") -> dict[str, BoundOp]:
    """This family's contribution to ``ops/reference.py::reference_bindings``."""
    del plan  # ``mm`` has one reference and no leaf plan: one fp32 accumulation over the whole K
    return {"mm": BoundOp("mm", "ref.mm", None, "ops.reference", fn=ref_mm)}


__all__ = [
    "bindings",
    "ref_mm",
]
