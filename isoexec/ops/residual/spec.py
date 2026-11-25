"""Declarations for attention and MLP residual joins, each one bf16 add after boundary casting."""

from __future__ import annotations

from isoexec.core.registry import ImplSpec

_SM90 = frozenset({"sm90", "sm100"})

IMPLS: tuple[ImplSpec, ...] = (
    ImplSpec(op="residual.add.attn", impl_id="eager", version=1, supported_archs=_SM90),
    ImplSpec(op="residual.add.mlp", impl_id="eager", version=1, supported_archs=_SM90),
)
