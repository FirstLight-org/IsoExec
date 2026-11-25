"""Implementation bindings shared by operation families, components and validation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

# Column-parallel inputs use identity forward and a fixed-tree TP gradient sum.
TP_COPY = "collectives.tp_copy"
# Reassemble split activation columns in rank order.
ALL_GATHER = "collectives.all_gather"
# Gather context-parallel tokens; backward restores local order and sums CP contributions.
CP_GATHER = "collectives.cp_gather"
# The dense GEMM every projection site runs.
MM = "mm"
# A ROW-parallel projection (o_proj, fc2): the GEMM and the K-reduction over the TP group as ONE
# callable, because the reduction's leaf tree is part of its arithmetic and not a step after it.
ROW_PARALLEL = "collectives.row_parallel"
# The K-reduction :data:`ROW_PARALLEL` folds its leaf partials with; a contracted region of its own.
TREE_ALL_REDUCE = "collectives.tree_all_reduce"


def identity(x: Any) -> Any:
    """The callable a per-module wire holds when the placement needs no movement."""
    return x


# Shared by the collectives (ops/collectives/tp_bytes.py) and the components' seams (nn/_seams.py,
# nn/embed.py). They live here, below both tiers, because ops/ may not import nn/.

#: The impl id a seam records where the placement asks for no movement at all.
NO_MOVEMENT = "none"

#: The provider every collective binding records.
COLLECTIVES_PROVIDER = "ops.collectives"

#: What the vocab-parallel embedding wrapper does, named on the resolution so a peer comparison can see it.
VOCAB_PARALLEL = "select_masked_then_tree_all_reduce"


@dataclass(frozen=True)
class BoundOp:
    """The callable a component executes and the numerical identity describing it.

    ``inputs`` records settings used to construct the callable. Validation compares
    the subset declared by the implementation. ``owns`` explicitly names the other
    operations computed by a fused or composed stage; including the primary op is
    permitted and must agree with ``impl``. Neither mapping is stored on the callable.
    Reference and test providers use ``version=None`` and distinct implementation IDs.
    """

    op: str
    impl: str
    version: int | None
    provider: str
    inputs: Mapping[str, Any] = field(default_factory=dict)
    fn: Callable[..., Any] = None  # type: ignore[assignment]
    owns: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not callable(self.fn):
            raise TypeError(f"BoundOp({self.op!r}, {self.impl!r}).fn is not callable")
        if not self.op or not self.impl:
            raise ValueError(f"BoundOp needs an op id and an impl id, got {self.op!r}/{self.impl!r}")
        if self.op in self.owns and self.owns[self.op] != self.impl:
            raise ValueError(f"BoundOp {self.op!r} ownership disagrees with implementation {self.impl!r}")
        object.__setattr__(self, "inputs", MappingProxyType(dict(self.inputs)))
        object.__setattr__(self, "owns", MappingProxyType(dict(self.owns)))

    @property
    def ops(self) -> tuple[str, ...]:
        """Every logical operation executed by this binding, primary operation first."""
        return tuple(dict.fromkeys((self.op, *self.owns)))


def no_movement(x: Any) -> Any:
    """The callable of a seam that needs no communication."""
    return x


def no_movement_binding(op: str, **inputs: Any) -> BoundOp:
    """The explicit binding a seam records where the placement asks for no movement (not nothing)."""
    return BoundOp(op, NO_MOVEMENT, None, COLLECTIVES_PROVIDER, inputs, no_movement)
