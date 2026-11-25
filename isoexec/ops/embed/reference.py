"""Plain-torch embedding reference over the logical vocabulary weight.
Tensor-parallel gathering has one nonzero partial per row and preserves the single-rank result."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from isoexec.core.bindings import BoundOp

if TYPE_CHECKING:  # the plan is DUCK-TYPED at runtime -- ops/reference.py imports this module, never the reverse
    from isoexec.ops.reference import ReferencePlan


def ref_embed_lookup(ids: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return F.embedding(ids, w)


def bindings(plan: "ReferencePlan") -> dict[str, BoundOp]:
    """This family's contribution to ``ops/reference.py::reference_bindings``."""
    del plan
    return {"embed.lookup": BoundOp("embed.lookup", "ref.embedding", None, "ops.reference", fn=ref_embed_lookup)}


__all__ = [
    "bindings",
    "ref_embed_lookup",
]
