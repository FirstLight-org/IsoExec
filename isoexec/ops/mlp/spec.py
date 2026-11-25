"""Declarations for the SwiGLU stage between dense MLP GEMMs, also referenced by MoE recipes."""

from __future__ import annotations

from isoexec.core.registry import ImplSpec

_SM90 = frozenset({"sm90"})

IMPLS: tuple[ImplSpec, ...] = (ImplSpec(op="mlp.act", impl_id="eager", version=1, supported_archs=_SM90),)


from isoexec.ops.mlp.mlp_sm100.spec import IMPLS as _SM100_IMPLS
from isoexec.ops.mlp.mlp_sm90.spec import IMPLS as _SM90_FUSED_IMPLS

IMPLS += _SM100_IMPLS + _SM90_FUSED_IMPLS
