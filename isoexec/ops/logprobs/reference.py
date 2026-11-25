"""Plain-torch scoring references: identity row selection, ref_mm head projection and tempered log-softmax.
Engine row selection occurs before this stage; temperature division uses the shared fp32 tensor divisor."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isoexec.core.bindings import BoundOp
from isoexec.ops.mm.reference import ref_mm
from isoexec.ops.logprobs.temperature import temperature_divide

if TYPE_CHECKING:  # the plan is DUCK-TYPED at runtime -- ops/reference.py imports this module, never the reverse
    from isoexec.ops.reference import ReferencePlan


def ref_head(h: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return ref_mm(h, w)


def ref_log_softmax(logits_fp32: torch.Tensor, targets: torch.Tensor | None, temperature: float):
    lp = torch.log_softmax(temperature_divide(logits_fp32, temperature), dim=-1)
    if targets is None:
        return lp
    return lp.gather(-1, targets[:, None].long())[:, 0]


def ref_lm_head_slice(h):
    return h


def bindings(plan: "ReferencePlan") -> dict[str, BoundOp]:
    """This family's contribution to ``ops/reference.py::reference_bindings``."""
    del plan
    return {
        "logprobs.lm_head_slice": BoundOp(
            "logprobs.lm_head_slice", "ref.identity", None, "ops.reference", fn=ref_lm_head_slice
        ),
        "logprobs.head": BoundOp("logprobs.head", "ref.head_bf16", None, "ops.reference", fn=ref_head),
        "logprobs.log_softmax": BoundOp(
            "logprobs.log_softmax", "ref.log_softmax", None, "ops.reference", fn=ref_log_softmax
        ),
    }


__all__ = [
    "bindings",
    "ref_head",
    "ref_lm_head_slice",
    "ref_log_softmax",
]
