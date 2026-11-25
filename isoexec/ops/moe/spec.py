"""Lazy MoE declarations and documented arithmetic constants.
Combination uses ascending global expert IDs and fp32 accumulation with one final narrowing; router GEMMs stay fp32.
These constants document the recipe; twin proofs enforce the placement-invariance claim."""

from typing import TYPE_CHECKING

from isoexec.ops._dispatch import family_impls

if TYPE_CHECKING:  # never executed: it names what the lazy ``__getattr__`` below otherwise hides.
    from isoexec.core.registry import ImplSpec

    IMPLS: tuple[ImplSpec, ...]

#: Ascending GLOBAL expert id, so an EP split cannot reorder a token's sum.
COMBINE_ORDER = "ascending_expert"

# Documented fp32 accumulator; kernels state it independently and the twin proof checks agreement.
COMBINE_ACCUM = "fp32"

#: The router GEMM's operand dtype and its output boundary: never quantized, never narrowed.
ROUTER_DTYPE = "fp32"
ROUTE_BOUNDARY_OUT_DTYPE = "fp32"

__all__ = ("COMBINE_ACCUM", "COMBINE_ORDER", "IMPLS", "ROUTER_DTYPE", "ROUTE_BOUNDARY_OUT_DTYPE")


from isoexec.ops.moe.deepseek_v41_sm100.spec import IMPLS as _DEEPSEEK_V41_IMPLS  # noqa: E402
from isoexec.ops.moe.router_sm100.spec import IMPLS as _ROUTER_IMPLS  # noqa: E402

__getattr__ = family_impls(__name__, "moe", own=_ROUTER_IMPLS + _DEEPSEEK_V41_IMPLS)
