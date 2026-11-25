"""Declare tensor sharding, checkpoint mappings, held dtypes, and engine cache requirements. Shard
rules come from parameter owners; torch dtype resolution is lazy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SHARD_KINDS = ("replicated", "col", "row", "expert")

# The held-dtype names a declaration uses, and the ``torch`` attribute each one resolves to.
TORCH_DTYPES = {"bf16": "bfloat16", "fp32": "float32", "fp16": "float16"}


class HeldDtypeError(ValueError):
    """A held-dtype name no declaration in this schema can mean."""


def torch_dtype(name: str):
    """'bf16' | 'fp32' | 'fp16' -> torch.dtype (imported lazily; the table itself is torch-free)."""
    import torch

    try:
        return getattr(torch, TORCH_DTYPES[name])
    except KeyError:
        raise HeldDtypeError(f"unknown held dtype {name!r}") from None


@dataclass(frozen=True)
class ShardAxis:
    """One logical tensor's shard declaration. See the module docstring for each field."""

    kind: Literal["replicated", "col", "row", "expert"]
    dim: int | None = None
    stride: int = 1  # interleaved partition stride (Megatron's strided convention)
    units: int | None = None
    segments: tuple[tuple[int, int | None], ...] | None = None  # (size, units) per segment along dim
    reduce: Literal["none", "sum_over_tp"] = "none"


@dataclass(frozen=True)
class TensorPlanMetadata:
    """Owner-declared facts needed to plan a tensor without interpreting its path."""

    kind: str = "other"
    expert_id: int | None = None
    num_experts: int | None = None
    expert_axis: ShardAxis | None = None


def col(
    *,
    dim: int = 0,
    stride: int = 1,
    units: int | None = None,
    segments: tuple[tuple[int, int | None], ...] | None = None,
) -> ShardAxis:
    """Split along the output rows (dim 0 by default)."""
    return ShardAxis("col", dim=dim, stride=stride, units=units, segments=segments)


def row(*, dim: int = 1) -> ShardAxis:
    """Split along the contracted dim -- the fc2 / o_proj side of a row-parallel GEMM."""
    return ShardAxis("row", dim=dim)


def replicated(*, reduce: Literal["none", "sum_over_tp"] = "none") -> ShardAxis:
    """Whole on every rank. ``reduce="sum_over_tp"`` when its gradient is per-rank partial."""
    return ShardAxis("replicated", reduce=reduce)


def expert() -> ShardAxis:
    """Partitioned by expert id (and, at ETP > 1, tensor-split inside the expert by role)."""
    return ShardAxis("expert")


@dataclass(frozen=True)
class StateSpec:
    """An engine cache with concrete shape, dtype, indexing granularity, and prefix-reuse policy.
    Joint resume points must satisfy every cache's reuse boundaries.
    """

    name: str  # "kv", "conv_q", "conv_k", "conv_v", "ssm"
    shape: tuple[int, ...]
    dtype: str  # kv/conv: bf16; ssm: fp32 (a region pin of gdn.core, never a knob)
    granularity: Literal["per_token", "per_request"]
    reuse: Literal["none", "align", "all"]
    shard: ShardAxis = ShardAxis("replicated")


@dataclass(frozen=True)
class TieDecl:
    """Two logical names that are ONE tensor: exactly one owner emits it, the aliases are declared
    replicas (design A29). ``lm.embed.weight`` tied to ``lm.head.weight`` is the shipped case."""

    logical: str
    aliases: tuple[str, ...]
    owner: str = "first"


@dataclass(frozen=True)
class LogicalTensor:
    """A logical tensor's name, global shape, held dtype, and shard declaration."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    shard: ShardAxis


__all__ = [
    "SHARD_KINDS",
    "TORCH_DTYPES",
    "HeldDtypeError",
    "LogicalTensor",
    "ShardAxis",
    "TensorPlanMetadata",
    "col",
    "row",
    "replicated",
    "expert",
    "StateSpec",
    "TieDecl",
    "torch_dtype",
]
