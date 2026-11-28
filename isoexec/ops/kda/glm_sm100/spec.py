"""Import-light declarations for the SM100 GLM Kimi Delta Attention core (``kda.glm``)."""

from isoexec.core.registry import ImplSpec
from isoexec.ops.kda import schedule

#: This package's architecture.
ARCH = "sm100"

#: Storage height of a chunk tile in rows (``C`` of csrc/common.cuh): workspace and record sections are sized by it.
CHUNK = 64

#: The chunk length of the schedule: tokens folded by one step of the reducer, anchored at absolute sequence
#: position 0. Not a literal: rule 1 of the schedule policy on the reducer's exact block (isoexec/ops/kda/schedule.py).
CHUNK_LENGTH = schedule.segments(schedule.CHUNK).S

#: Key channels == value channels per head (GLM-5.3-Flash ``linear_head_dim``).
HEAD_DIM = 128

#: Bumped by any change to the arithmetic or rounding schedule of csrc/recipe.cuh.
RECIPE_VERSION = 1

IMPL_ID = "sm100_kda_v1"

IMPLS: tuple[ImplSpec, ...] = (
    ImplSpec(op="kda.glm", impl_id=IMPL_ID, version=RECIPE_VERSION, supported_archs=frozenset({ARCH})),
)
