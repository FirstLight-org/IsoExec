"""Tensor carriers between MoE routing, expert execution and combination.
RoutingResult, ExpertOutputs and SharedExpertPartial hold per-forward values, including deferred reductions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch


@dataclass(frozen=True)
class RoutingResult:
    """What every routing function returns. A runtime value, not config: it is the output of the
    ``moe.route_mechanics`` region and it travels from there to ``moe.expert_mlp``."""

    indices: torch.Tensor  # [T, k] int32
    probs: torch.Tensor  # [T, k] fp32
    scores_full: torch.Tensor | None  # [T, E] fp32; collected only when the binding requests aux


@dataclass(eq=False)
class ExpertOutputs:
    """FP32 expert rows with an explicit routing, folded, or deferred-partial layout.
    The combined_local layout contains this SP rank's token rows only."""

    rows: torch.Tensor
    layout: str
    routing: RoutingResult
    extra: Any = None  # kernel-private (e.g. a permutation) -- opaque to the model
    reduce: Callable[[torch.Tensor], torch.Tensor] | None = None  # the deferred tree, iff layout == "permuted_partial"
    combine_rows: torch.Tensor | None = None  # explicit inverse permutation, [T,k] int32
    weight_after_fc2: bool = False  # deferred reduction also applies the final routing weight


@dataclass(eq=False)
class SharedExpertPartial:
    """Shared-expert partial whose fixed-tree reduction is deferred to the combine operation."""

    partial: torch.Tensor  # [T, H]
    gate: torch.Tensor | None  # [T, 1] bf16 = torch.sigmoid(F.linear(x, expert_gate)), already rounded; None if ungated
    finish: Callable[[torch.Tensor], torch.Tensor]

    def eager(self) -> torch.Tensor:
        y = self.finish(self.partial)
        return y * self.gate if self.gate is not None else y


__all__ = [
    "ExpertOutputs",
    "RoutingResult",
    "SharedExpertPartial",
]
