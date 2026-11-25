"""Describe runtime placements, parallel groups, and execution settings."""

from __future__ import annotations

from dataclasses import dataclass

# The two sides of the composition. A trainer holds gradients; an engine does not.
SIDES = ("trainer", "engine")

# The leaf dtypes the pik fixed tree admits on the wire.
LEAF_DTYPES = ("bf16", "fp32")


class ShardError(ValueError):
    """A placement, shard rule, or checkpoint combination cannot be represented."""


@dataclass(frozen=True)
class Axis:
    """One parallel axis on this rank: its size, this rank's position in it, and the host's
    process-group handle (opaque here -- ``schema`` imports no torch)."""

    size: int
    rank: int
    group: "object | None" = None

    def __post_init__(self) -> None:
        if not isinstance(self.size, int) or self.size < 1:
            raise ValueError(f"Axis.size must be an int >= 1, got {self.size!r}")
        if not isinstance(self.rank, int) or not (0 <= self.rank < self.size):
            raise ValueError(f"Axis.rank={self.rank!r} out of range for size={self.size}")


@dataclass(frozen=True)
class Placement:
    """The live placement of a model on this rank."""

    tp: Axis
    ep: Axis
    dp: Axis
    pp: Axis
    etp: Axis
    cp: Axis
    side: str
    leaves: int
    leaf_dtype: str = "bf16"
    host_groups: "object | None" = None
    # SP partitions the TP residual stream by tokens without changing weight-shard shapes.
    sp: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.sp, bool):
            raise ValueError(f"Placement.sp must be a bool, got {self.sp!r}")
        for name in ("tp", "ep", "dp", "pp", "etp", "cp"):
            if not isinstance(getattr(self, name), Axis):
                raise ValueError(f"Placement.{name} must be an Axis, got {type(getattr(self, name)).__name__}")
        if self.side not in SIDES:
            raise ValueError(f"Placement.side must be one of {list(SIDES)}, got {self.side!r}")
        if self.leaf_dtype not in LEAF_DTYPES:
            raise ValueError(f"Placement.leaf_dtype must be one of {list(LEAF_DTYPES)}, got {self.leaf_dtype!r}")
        # Validate leaf counts against operator pins during placement validation.
        if not isinstance(self.leaves, int) or self.leaves < 1:
            raise ValueError(f"Placement.leaves must be an int >= 1, got {self.leaves!r}")
        if self.cp.size > 1 and self.cp.group is None:
            raise ValueError(
                f"Placement.cp.size={self.cp.size} needs a live CP group (got group={self.cp.group!r}): context "
                "parallelism hands key/value prefixes and recurrent states between the CP ranks on that group"
            )
        if self.sp and self.tp.size < 2:
            raise ValueError(
                f"Placement.sp=True needs tp.size >= 2 (got side={self.side!r}, "
                f"tp.size={self.tp.size}): sequence parallelism shards the residual stream over the TP group, "
                "not an independent rank axis"
            )
        want_etp = self.tp.size if self.ep.size == 1 else 1
        if self.etp.size != want_etp:
            raise ValueError(
                f"Placement.etp.size={self.etp.size} but ep.size={self.ep.size} and tp.size={self.tp.size} "
                f"require etp.size={want_etp}: at EP=1 every rank tensor-shards every expert over the TP "
                "group; at EP>1 the expert-MODEL and expert-TENSOR grids together are a 2-D grid Placement "
                "carries no handle for"
            )
        if self.etp.size == self.tp.size and self.etp.size > 1:
            if self.etp.rank != self.tp.rank or self.etp.group is not self.tp.group:
                raise ValueError(
                    f"Placement.etp is the TP axis at etp.size == tp.size == {self.tp.size}, but carries "
                    f"rank={self.etp.rank}/group={self.etp.group!r} against tp rank={self.tp.rank}/"
                    f"group={self.tp.group!r}"
                )


__all__ = ["Axis", "LEAF_DTYPES", "Placement", "SIDES", "ShardError"]
