from __future__ import annotations

ARCH = "sm100"

RECIPE_VERSION = 1
ACT_IMPL = "sm100_fused_v1"

from isoexec.core.registry import ImplSpec

IMPLS = (ImplSpec(op="mlp.act", impl_id=ACT_IMPL, version=RECIPE_VERSION,
                  supported_archs=frozenset({ARCH})),)
