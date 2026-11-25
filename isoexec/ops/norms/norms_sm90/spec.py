"""Import-light norm declarations; SM90 recipes have no eager bitwise twin claim and q/k fusion is declared by attention.
RMS v1 uses width-fixed fp32 lane chains/butterfly, rounded div/add/rsqrt and bf16((x*rstd)*g),
with g=w or 1+w; gated RMS multiplies fp32 z/(1+expf(-z)) before narrowing.
Gamma convention is caller-supplied; changing lane plans or rounding requires a version bump."""


from __future__ import annotations

from isoexec.core.registry import ImplSpec, OneOf

#: The recipe's identity. ``ops.py`` imports it rather than restating it: the version is the only
#: declared signal that a kernel's rounding schedule changed, and a second copy of that signal is
#: the one drift a reader cannot see.
RECIPE_VERSION = 1
RMS_IMPL = "sm90_rms_v1"
GATED_IMPL = "sm90_gated_v1"
#: The fused residual join + RMSNorm: ``h = rn_bf16(x + r)`` stored, then recipe v1 on ``h``. Bit-identical
#: to the bf16 ``aten.add`` followed by ``RMS_IMPL`` (same fold, same rounding), so it is declared under the
#: same RMS op ids and records the residual join it absorbs in the binding's ``owns``.
ADD_RMS_IMPL = "sm90_add_rms_v1"
_SM90 = frozenset({"sm90"})

# Hazards the battery EXERCISES (tests/README.md maps each to its test and witness).
HAZARDS = (
    "null_lane",
    "signed_zero",
    "subnormal",
    "non_contiguous_views",
    "misaligned_storage",
    "tp_head_slicing",
    "phase_parity",
    "graph_replay",
)

RMS_STAGES = ("norms.rms.pre_attn", "norms.rms.pre_mlp", "norms.rms.final")
_GATED = {"convention": OneOf("gated_rms_zero_centered")}

IMPLS: tuple[ImplSpec, ...] = (
    # The SM90 recipe serves either caller-supplied gamma convention at all RMS stages.
    *(
        ImplSpec(
            op=stage,
            impl_id=RMS_IMPL,
            version=RECIPE_VERSION,
            supported_archs=_SM90,
            inputs={"convention": OneOf("rms", "rms_zero_centered")},
        )
        for stage in RMS_STAGES
    ),
    # The fused residual join + RMSNorm serves the same stages and conventions.
    *(
        ImplSpec(
            op=stage,
            impl_id=ADD_RMS_IMPL,
            version=RECIPE_VERSION,
            supported_archs=_SM90,
            inputs={"convention": OneOf("rms", "rms_zero_centered")},
        )
        for stage in RMS_STAGES
    ),
    ImplSpec(
        op="norms.gated_out",
        impl_id=GATED_IMPL,
        version=RECIPE_VERSION,
        supported_archs=_SM90,
        inputs=dict(_GATED),
    ),
    # The MLA latent norm uses plain gamma.
    ImplSpec(
        op="norms.mla",
        impl_id=RMS_IMPL,
        version=RECIPE_VERSION,
        supported_archs=_SM90,
        inputs={"convention": OneOf("rms")},
    ),
)
