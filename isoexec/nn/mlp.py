"""Dense gated MLP with gate/up projections (one packed GEMM on the engine while the alias holds) and
a fixed-tree row-parallel output. Its intermediate width is elementwise shardable, and all stages
operate per token.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any

import torch
from torch import nn

from isoexec.schema import Placement, SequenceContext, ShardAxis, col, row, torch_dtype
from isoexec.schema.token_layout import TokenLayoutContract, TokenLayoutEffect
from isoexec.core.linear import split_columns
from isoexec.nn.component import Component

#: Family policy, not an argument: both fc1 projections and the fc2 are held bf16.
HELD_DTYPE = "bf16"

#: The activations this kind admits. ``silu_mul`` is ``silu(gate) * up``, folded in fp32 and rounded
#: once; it is the op ``mlp.act`` names, and a second entry here would need its own op id.
ACTIVATIONS = ("silu_mul",)


class MLP(Component):
    """Dense gated MLP over packed tokens, registering ``gate``, ``up`` and ``down`` at global
    shape. ``hidden`` is the residual width, ``intermediate`` is the TP-sharded fc1 width, and
    ``activation`` selects ``mlp.act``. Construction defaults to meta.
    """

    SLOT_OPS = MappingProxyType(
        {
            "mm": "mm",
            "act": "mlp.act",
            "row_parallel": "collectives.row_parallel",
            "tp_copy": "collectives.tp_copy",
        }
    )
    SLOTS = tuple(SLOT_OPS)
    NATIVE_LAYOUT = TokenLayoutContract(TokenLayoutEffect.DECODER)
    #: ``forward`` returns its ``row_parallel`` slot's result unmodified (the residual+norm fusion pass may defer
    #: that reduction into the consuming pre-norm; compiler/realization/fusion.py).
    ROW_PARALLEL_OUTPUT = True

    # Every stage operates per token, including the fixed-tree contraction, so CP needs no seam.
    CONTEXT_PARALLEL = "per token: every stage is a function of one row, so a CP cut moves no arithmetic"

    #: The name the packed ``gate|up`` operand is registered under (non-persistent buffer).
    packed_proj_name = "fc1_packed"

    def __init__(
        self,
        *,
        hidden: int,
        intermediate: int,
        activation: str = "silu_mul",
        device: torch.device | str = "meta",
    ) -> None:
        if min(hidden, intermediate) < 1:
            raise ValueError(f"MLP: hidden/intermediate must be >= 1, got {(hidden, intermediate)}")
        if activation not in ACTIVATIONS:
            raise ValueError(f"MLP: activation must be one of {ACTIVATIONS}, got {activation!r}")
        super().__init__()
        self.hidden, self.intermediate, self.activation = hidden, intermediate, activation
        dtype = torch_dtype(HELD_DTYPE)
        shapes = (("gate", (intermediate, hidden)), ("up", (intermediate, hidden)), ("down", (hidden, intermediate)))
        for role, shape in shapes:
            self.register_parameter(role, nn.Parameter(torch.empty(shape, dtype=dtype, device=device)))
        #: The free-alias hook for the fused gate|up operand. ``None`` until a build path calls
        #: ``ops/mm/packed_proj.attach``; ``None`` from the hook means "two separate GEMMs".
        self.packed_proj: Callable[[], torch.Tensor | None] | None = None

    def tensor_axes(self) -> Mapping[str, ShardAxis]:
        """Shard fc1 output rows and the fc2 contraction dimension. The intermediate width has no
        head granule.
        """
        return {"gate": col(), "up": col(), "down": row()}

    def facts(self) -> Mapping[str, Any]:
        return {
            "kind": "mlp",
            "hidden": self.hidden,
            "intermediate": self.intermediate,
            "activation": self.activation,
            "held_dtype": HELD_DTYPE,
        }


    def _place(self, placement: Placement) -> None:
        site = self.site
        # gate and up are column-parallel: one tp_copy for both, so autograd accumulates the two
        # partial input gradients before the single backward tree rather than folding two trees.
        seams = self.seams
        self.slots.tp_copy = seams.tp_copy()
        self.slots.row_parallel = seams.row_parallel()
        self.slots.mm = self.bind_op("mm", site=f"{site}.fc1")
        # The activation is identified by the resolved implementation and version.
        self.slots.act = self.bind_op("mlp.act", site=f"{site}.act", facts={"activation": self.activation})


    def forward(
        self, x: torch.Tensor, seq: SequenceContext | None = None, aux: dict[str, torch.Tensor] | None = None
    ) -> torch.Tensor:
        """``seq`` and ``aux`` are the uniform sublayer arguments a row's MLP half is called with: a
        dense MLP is position-independent and writes no aux entry, so it reads neither."""
        del seq, aux
        xc = self.slots.tp_copy(x)
        w = self.packed_proj() if self.packed_proj is not None else None
        if w is None:  # no hook installed, or the alias could not be verified
            g = self.slots.mm(xc, self.gate)
            u = self.slots.mm(xc, self.up)
        else:  # one GEMM (the census's mlp.fc1 shape); g and u are adjacent column views, never copies
            g, u = split_columns(self.slots.mm(xc, w), self.packed_proj_members())
        return self.slots.row_parallel(self.slots.act(g, u), self.down)

    def packed_proj_members(self) -> tuple[torch.Tensor, ...]:
        """The projection parameters the fused GEMM concatenates, in column order."""
        return (self.gate, self.up)

    def packed_proj_roles(self) -> tuple[str, ...]:
        """Projection roles packed into one fc1 operand, in column order."""
        return ("gate", "up")

    def extra_repr(self) -> str:
        return f"hidden={self.hidden}, intermediate={self.intermediate}, activation={self.activation!r}"


__all__ = ["ACTIVATIONS", "HELD_DTYPE", "MLP"]
