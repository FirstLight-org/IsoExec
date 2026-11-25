"""SM100 ``conv.short`` (width-4 causal depthwise convolution)."""

from __future__ import annotations

from isoexec.core.registry import ImplSpec

#: This package's architecture. ``ops/_dispatch.FAMILIES`` names it on the sm100 row.
ARCH = "sm100"

#: The op ids this package owes a declaration for:
#:   - conv.short

#: Impl id convention, mirroring the sm90 spelling: IMPL = "sm100_cuda_v1"

#: Taps per channel. conv_sm90 states 4 in both spec.py and csrc/common.cuh; the sm100 csrc must
#: state the same number in the same place, and the two must agree.
WIDTH = 4

#: Bump when the numerical recipe changes; ops.py imports this identity. Still 1: this package's
#: csrc/common.cuh is the same scalar graph as conv_sm90's (see its docstring for why an elementwise,
#: MMA-free kernel has nothing for Blackwell to change numerically). This is the DEFAULT and the only
#: version the engine and trainer seams reach.
RECIPE_VERSION = 1

#: The forward-only **option**: recipe v2, ``csrc/recipe_v2.cuh``. Recipe v1's tap order with the
#: three additions contracted into explicit ``fma.rn`` -- four roundings per token instead of seven,
#: a schedule this package *declares* rather than one the compiler chooses (which is what
#: docs/dtir.md forbids and what ``--fmad=false`` keeps out of v1). It is a different arithmetic
#: identity, so it gets its own impl id, its own version and its own battery
#: (``tests/test_recipe_v2_gpu.py``); it is NOT bitwise equal to v1 and must never be compared to
#: v1's bits. Reached only by an explicit ``recipe=2`` on ``ops.varlen``/``ops.update``: nothing in
#: ``integration.py`` selects it, and the trainer's autograd path stays on v1 (its VJP recomputes the
#: pre-activation, so a v2 trainer needs a v2 VJP, which is not written).
RECIPE_VERSION_FMA = 2

#: Declared once the GPU battery (tests/test_conv_sm100_gpu.py) passes on a leased B200. Until then
#: this stays empty and ``_assert_proven`` keeps refusing the sm100 row of ``conv.short``, which is
#: the correct interim answer.
#: The default implementation: recipe v1, the arithmetic conv_sm90 fixes.
IMPL_V1 = ImplSpec(op="conv.short", impl_id="sm100_cuda_v1", version=RECIPE_VERSION,
                   supported_archs=frozenset({"sm100"}))

#: The declared option: recipe v2's explicit-``fma.rn`` tap contraction. Declared beside v1 rather
#: than instead of it, and deliberately NOT in a twin class -- a twin claim is bitwise equality, and
#: these two are unequal by construction. Two impls of one op with different arithmetic is exactly
#: what an impl id is for: the id is the recipe, and a caller naming it gets that recipe's bits.
IMPL_FMA = ImplSpec(op="conv.short", impl_id="sm100_conv_fma_v1", version=RECIPE_VERSION_FMA,
                    supported_archs=frozenset({"sm100"}))

IMPLS: tuple[ImplSpec, ...] = (
    # Hosts bind the GDN row's SiLU; activation is not a declared caller input.
    IMPL_V1,
    IMPL_FMA,
)
