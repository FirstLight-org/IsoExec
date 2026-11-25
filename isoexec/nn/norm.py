"""RMS normalization with an explicit held-gamma convention. Placement selects an implementation
matching that convention and the site-specific norm operation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from isoexec.schema import Placement, ShardAxis, replicated, torch_dtype
from isoexec.schema.token_layout import TokenLayoutContract, TokenLayoutEffect
from isoexec.nn.component import Component

#: The three sites the contract types ``norms.rms`` by.
ROLES = ("pre_attn", "pre_mlp", "final")

#: The conventions a gamma can be held in. ``rms_zero_centered`` holds ``gamma - 1``.
CONVENTIONS = ("rms_zero_centered", "rms")

#: Family policy, not an argument: a gamma is held bf16 on both sides of the composition.
HELD_DTYPE = "bf16"


class RMSNorm(Component):
    """Normalize the last dimension using a learned ``[hidden]`` gamma. ``eps`` is added inside the
    square root.

    ``convention="rms"`` holds gamma; ``"rms_zero_centered"`` holds ``gamma - 1`` and applies
    ``1 + w``. Placement requires a matching implementation. ``role`` selects the site-specific
    ``norms.rms.*`` operation, and ``device`` defaults to meta.
    """

    SLOTS = ("norm", "add_norm")
    NATIVE_LAYOUT = TokenLayoutContract(TokenLayoutEffect.PRESERVE)
    SEQUENCE_GRADIENT_REDUCTION = True

    # Normalization reduces over one token's hidden width; CP needs no seam.
    CONTEXT_PARALLEL = "per token: the reduction is over one row's hidden width, never across tokens"

    def __init__(
        self,
        *,
        hidden: int,
        eps: float = 1e-6,
        convention: str = "rms_zero_centered",
        role: str = "pre_attn",
        device: torch.device | str = "meta",
    ) -> None:
        if hidden < 1:
            raise ValueError(f"RMSNorm: hidden must be >= 1, got {hidden}")
        if convention not in CONVENTIONS:
            raise ValueError(f"RMSNorm: convention must be one of {CONVENTIONS}, got {convention!r}")
        if role not in ROLES:
            raise ValueError(f"RMSNorm: role must be one of {ROLES}, got {role!r}")
        super().__init__(slot_ops={"norm": f"norms.rms.{role}", "add_norm": f"norms.rms.{role}"})
        # Filled only by the residual-norm fusion pass (compiler/realization/fusion.py), which also
        # rewrites the call site to pass the residual; unfilled, the norm runs exactly as authored.
        self.slots.declare_unbound("add_norm", "the residual + norm fusion pass fills it", required=False)
        self.hidden, self.eps, self.convention, self.role = hidden, float(eps), convention, role
        self.weight = nn.Parameter(torch.empty(hidden, dtype=torch_dtype(HELD_DTYPE), device=device))

    def tensor_axes(self) -> Mapping[str, ShardAxis]:
        """The gamma is whole on every rank: it multiplies a replicated channel axis."""
        return {"weight": replicated()}

    def facts(self) -> Mapping[str, Any]:
        return {
            "kind": "rms_norm",
            "hidden": self.hidden,
            "eps": self.eps,
            "convention": self.convention,
            "role": self.role,
            "held_dtype": HELD_DTYPE,
        }

    def _place(self, placement: Placement) -> None:
        del placement  # a norm is elementwise over the channel axis: no seam, no side dependence
        site = f"{self.site}.norm"
        op = self.slots.op_of("norm")
        # Width selects an admitted kernel; model configuration already records the shape.
        self.slots.norm = self.bind_op(
            op, site=site, inputs={"eps": self.eps}, convention=self.convention, facts={"width": self.hidden}
        )

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        """``norm(x)``; with ``residual`` (only a fused call site passes it) the residual join is
        absorbed and the result is ``(norm(residual + x), residual + x)``."""
        if residual is None:
            return self.slots.norm(x, self.weight, self.eps)
        return self.slots.add_norm(x, residual, self.weight, self.eps)

    #: The residual join a pre-norm of each role absorbs when fused (the join that feeds it).
    ABSORBED_JOIN = {"pre_mlp": "residual.add.attn", "pre_attn": "residual.add.mlp", "final": "residual.add.mlp"}

    def extra_repr(self) -> str:
        return f"hidden={self.hidden}, eps={self.eps}, convention={self.convention!r}, role={self.role!r}"


__all__ = ["CONVENTIONS", "HELD_DTYPE", "ROLES", "RMSNorm"]
