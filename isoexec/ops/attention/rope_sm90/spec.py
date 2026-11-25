"""Import-light declarations for SM90 table RoPE and fused q/k norm/RoPE.
SM90 uses device-double tables rounded to fp32,
fp32 rotation and one narrowing, with a balanced fp32 norm reduction in the fused variant.
These recipes have no twin claim; a rounding change requires a version bump."""


from __future__ import annotations

from isoexec.core.registry import ImplSpec, OneOf

#: Standalone RoPE and the fused forward retain their original arithmetic.
RECIPE_VERSION = 1
QK_RECIPE_VERSION = 2
QK_GRAD_SHIFT = 4
ROPE_IMPL = "sm90_table_v1"
QK_IMPL = "sm90_qk_norm_rope_v2"
_SM90 = frozenset({"sm90"})

HAZARDS = (
    "null_lane",
    "signed_zero",
    "subnormal",
    "overflow",
    "non_contiguous_views",
    "misaligned_storage",
    "position_table_bounds",
    "tp_head_slicing",
    "phase_parity",
)

IMPLS: tuple[ImplSpec, ...] = (
    # The table recipe takes nothing: its cos/sin source and rotation order are the recipe itself,
    # identified by the version, not a schedule the host may choose between.
    ImplSpec(op="rope.apply", impl_id=ROPE_IMPL, version=RECIPE_VERSION, supported_archs=_SM90),
    # The fused kernel spans the q/k norm AND the rotation, so it is declared on both ops under its
    # own impl id. A binding that runs it names the other op in `owns`.
    # Both gamma conventions are one recipe: g = 1 +rn w (rms_zero_centered) or g = w (rms), then
    # (x*inv)*g. The convention is a recorded input, so the two never share a digest.
    ImplSpec(
        op="norms.qk",
        impl_id=QK_IMPL,
        version=QK_RECIPE_VERSION,
        supported_archs=_SM90,
        inputs={"convention": OneOf("rms_zero_centered", "rms")},
    ),
    ImplSpec(
        op="rope.apply",
        impl_id=QK_IMPL,
        version=QK_RECIPE_VERSION,
        supported_archs=_SM90,
    ),
)
