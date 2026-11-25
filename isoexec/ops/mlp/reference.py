"""Plain-torch SwiGLU reference shared by dense and MoE stages.
SiLU and multiplication run in fp32, followed by one round to the gate dtype."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from isoexec.core.bindings import BoundOp

if TYPE_CHECKING:  # the plan is DUCK-TYPED at runtime -- ops/reference.py imports this module, never the reverse
    from isoexec.ops.reference import ReferencePlan


def ref_silu_mul(g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    return (F.silu(g.to(torch.float32)) * u.to(torch.float32)).to(g.dtype)


def bindings(plan: "ReferencePlan") -> dict[str, BoundOp]:
    """This family's contribution to ``ops/reference.py::reference_bindings``."""
    del plan
    return {"mlp.act": BoundOp("mlp.act", "ref.silu_mul", None, "ops.reference", fn=ref_silu_mul)}


__all__ = [
    "bindings",
    "ref_silu_mul",
]
