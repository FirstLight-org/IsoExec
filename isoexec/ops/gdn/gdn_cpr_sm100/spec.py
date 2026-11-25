"""Import-light declarations for the SM100 chunk-parallel GDN core and its owned stages."""

from isoexec.core.registry import ImplSpec

#: This package's architecture. ``ops/_dispatch.FAMILIES`` names it on the sm100 row.
ARCH = "sm100"

#: The chunk is the prefix recipe's, anchored at absolute position 0 and deployment-fixed (dtir.md 7).
CHUNK = 64

#: Bumped by any change to the arithmetic or the rounding schedule. Version 1 evaluates the program
#: csrc/recipe.cuh states, with mma.sync's ascending k16 steps -- see README.md, "Recipe version".
RECIPE_VERSION = 1

_SM100 = frozenset({"sm100"})
_FUSED_OPS = ("gdn.core", "gdn.l2norm", "gdn.gating")

IMPLS: tuple[ImplSpec, ...] = tuple(
    ImplSpec(op=op, impl_id="sm100_cpr_v1", version=RECIPE_VERSION, supported_archs=_SM100)
    for op in _FUSED_OPS
)
