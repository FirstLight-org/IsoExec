"""SM100 CUDA norm family: RMSNorm (plain / zero-centred), fused add + RMSNorm, GDN gated out-norm."""

from __future__ import annotations

from isoexec.core.registry import ImplSpec, OneOf

ARCH = "sm100"

RECIPE_VERSION = 1
RMS_IMPL = "sm100_rms_v1"
GATED_IMPL = "sm100_gated_v1"

RECIPE_VERSION_V2 = 2
RMS_IMPL_V2 = "sm100_rms_v2"
RECIPE_VERSIONS = (RECIPE_VERSION, RECIPE_VERSION_V2)
#: The fused residual join + RMSNorm: ``h = rn_bf16(x + r)`` stored, then the width's RMS recipe on
#: ``h``. Bit-identical to the bf16 ``aten.add`` followed by that recipe's RMS impl, so it is declared
#: under the same RMS op ids, one identity per recipe, and records the join it absorbs in ``owns``.
ADD_RMS_IMPL = "sm100_add_rms_v1"
ADD_RMS_IMPL_V2 = "sm100_add_rms_v2"
_SM100 = frozenset({"sm100"})

RMS_STAGES = ("norms.rms.pre_attn", "norms.rms.pre_mlp", "norms.rms.final")
_GATED = {"convention": OneOf("gated_rms_zero_centered")}

IMPLS: tuple[ImplSpec, ...] = (
    *(
        ImplSpec(
            op=stage,
            impl_id=RMS_IMPL,
            version=RECIPE_VERSION,
            supported_archs=_SM100,
            inputs={"convention": OneOf("rms", "rms_zero_centered")},
        )
        for stage in RMS_STAGES
    ),
    ImplSpec(
        op="norms.gated_out",
        impl_id=GATED_IMPL,
        version=RECIPE_VERSION,
        supported_archs=_SM100,
        inputs=dict(_GATED),
    ),
    ImplSpec(
        op="norms.mla",
        impl_id=RMS_IMPL,
        version=RECIPE_VERSION,
        supported_archs=_SM100,
        inputs={"convention": OneOf("rms")},
    ),
    *(
        ImplSpec(
            op=stage,
            impl_id=RMS_IMPL_V2,
            version=RECIPE_VERSION_V2,
            supported_archs=_SM100,
            inputs={"convention": OneOf("rms", "rms_zero_centered")},
        )
        for stage in RMS_STAGES
    ),
    *(
        ImplSpec(
            op=stage,
            impl_id=impl,
            version=version,
            supported_archs=_SM100,
            inputs={"convention": OneOf("rms", "rms_zero_centered")},
        )
        for stage in RMS_STAGES
        for impl, version in ((ADD_RMS_IMPL, RECIPE_VERSION), (ADD_RMS_IMPL_V2, RECIPE_VERSION_V2))
    ),
    ImplSpec(
        op="norms.mla",
        impl_id=RMS_IMPL_V2,
        version=RECIPE_VERSION_V2,
        supported_archs=_SM100,
        inputs={"convention": OneOf("rms")},
    ),
)
