"""Canonical authoring metadata for stateful model invocations."""

from dataclasses import dataclass

from .model import MLA, Attn


@dataclass(frozen=True, kw_only=True)
class GQAMetadata(Attn):
    scale: float


@dataclass(frozen=True, kw_only=True)
class MLAMetadata(MLA):
    scale: float


@dataclass(frozen=True, kw_only=True)
class GDNMetadata:
    num_key_heads: int
    num_value_heads: int
    key_head_dim: int
    value_head_dim: int
    conv_kernel: int


__all__ = ["GDNMetadata", "GQAMetadata", "MLAMetadata"]
