"""SM100 CUDA norm family: RMSNorm (plain / zero-centred), fused add + RMSNorm, GDN gated out-norm."""

from __future__ import annotations

from isoexec.core.registry import ImplSpec, OneOf

ARCH = "sm100"

#: One numerical schedule for every width (``schedule.py``; isoexec/ops/schedule_policy.py, design version 7.1).
#: Versions 1 and 2 were the strided-lane recipes keyed on the width; their identities are retired.
RECIPE_VERSION = 3
RMS_IMPL = "sm100_rms_v3"
GATED_IMPL = "sm100_gated_v3"
#: The fused residual join + RMSNorm: ``h = rn_bf16(x + r)`` stored, then the RMS schedule on ``h``.
#: Bit-identical to the bf16 ``aten.add`` followed by the RMS impl, so it is declared under the same RMS
#: op ids and records the join it absorbs in ``owns``.
ADD_RMS_IMPL = "sm100_add_rms_v3"
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
            impl_id=ADD_RMS_IMPL,
            version=RECIPE_VERSION,
            supported_archs=_SM100,
            inputs={"convention": OneOf("rms", "rms_zero_centered")},
        )
        for stage in RMS_STAGES
    ),
)
