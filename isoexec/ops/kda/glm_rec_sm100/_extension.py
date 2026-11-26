"""Build the recurrent KDA extension lazily with the shared SM100 profile (NUMERICS_FLAGS).

This is the per-token reducer (``schedule.TOKEN``, block 1): one step folds one token, so the kernel has no
block parameter and ``schedule.cuda_defines(TOKEN)`` is empty.
"""

from functools import lru_cache

from isoexec.ops.build_cache import DIAGNOSTIC_FLAGS, build_sm100_extension
from isoexec.ops.kda import schedule

SOURCES = ("bindings.cpp", "rec.cu", "bwd.cu")
FAMILY = schedule.TOKEN.family


@lru_cache(None)
def extension():
    return build_sm100_extension(FAMILY, __file__, SOURCES, include_csrc=True,
                                 extra_cuda_flags=(*DIAGNOSTIC_FLAGS, *schedule.cuda_defines(schedule.TOKEN)))
