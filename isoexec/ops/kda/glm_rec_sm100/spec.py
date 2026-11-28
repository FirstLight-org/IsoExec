"""Import-light declarations of the recurrent SM100 GLM KDA core (``kda.glm``, recipe ``sm100_kda_rec_v1``)."""

from isoexec.core.registry import ImplSpec

ARCH = "sm100"
HEAD_DIM = 128

#: Bumped by any change to the arithmetic of csrc/recipe.cuh / csrc/rec.cu.
RECIPE_VERSION = 1

IMPL_ID = "sm100_kda_rec_v1"
#: The CPR recipe this one coexists with (frozen, default).
CPR_IMPL_ID = "sm100_kda_v1"

IMPLS: tuple[ImplSpec, ...] = (
    ImplSpec(op="kda.glm", impl_id=IMPL_ID, version=RECIPE_VERSION, supported_archs=frozenset({ARCH})),
)
