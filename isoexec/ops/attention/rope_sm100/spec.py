"""SM100 table RoPE and fused q/k norm/RoPE declarations. The architecture selector validates
these records; the package batteries establish the stated numerical recipes."""

from __future__ import annotations

from isoexec.core.registry import ImplSpec, OneOf

#: This package's architecture. ``ops/_dispatch.FAMILIES`` names it on the sm100 row.
ARCH = "sm100"

#: The op ids this package owes a declaration for:
#:   - rope.apply
#:   - norms.qk

#: Bump when the numerical recipe changes; ops.py imports this identity. Still 1: the recipe is
#: bit-for-bit rope_sm90 v1 (see README.md), so this is the same version number under a different
#: arch, not a new recipe.
RECIPE_VERSION = 1
ROPE_IMPL = "sm100_table_v1"
QK_IMPL = "sm100_qk_norm_rope_v1"
_SM100 = frozenset({"sm100"})

#: Same hazard set as rope_sm90/spec.py -- the numerical recipe (and therefore what can go wrong in
#: it) is unchanged; every one is exercised by tests/test_forward_gpu.py and named in README.md.
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

#: The fused kernel spans the q/k norm AND the rotation, so it is declared on BOTH op ids under one
#: impl id, exactly as rope_sm90 does; a binding that runs it names the other op in ``owns``.

IMPLS: tuple[ImplSpec, ...] = (
    ImplSpec(op="rope.apply", impl_id=ROPE_IMPL, version=RECIPE_VERSION, supported_archs=_SM100),
    # Zero-centred only: the kernel has a plain-gamma (``rms``) path, but it has not been run through
    # this package's batteries on SM100 hardware for that convention, so a plain-gamma row (Qwen3)
    # refuses at resolution here instead of shipping unverified. rope_sm90 declares both.
    ImplSpec(op="norms.qk", impl_id=QK_IMPL, version=RECIPE_VERSION, supported_archs=_SM100,
             inputs={"convention": OneOf("rms_zero_centered")}),
    ImplSpec(op="rope.apply", impl_id=QK_IMPL, version=RECIPE_VERSION, supported_archs=_SM100),
)
