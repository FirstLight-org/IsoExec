"""Build the SM100 KDA extension lazily with the shared SM100 profile (NUMERICS_FLAGS).

The chunk length of the schedule reaches the kernels here and nowhere else: ``schedule.cuda_defines(CHUNK)`` is
``-DKDA_CL=<S>`` with ``S`` from ``isoexec.ops.schedule_policy.no_merge``. The loaded module is asked for the value
it was compiled with and refused if it differs.
"""

from functools import lru_cache

from isoexec.ops.build_cache import DIAGNOSTIC_FLAGS, build_sm100_extension
from isoexec.ops.kda import schedule

SOURCES = ("bindings.cpp", "plan.cu", "intra.cu", "state.cu", "decode.cu", "bwd_state.cu", "bwd.cu", "layer.cu", "decode_rec.cu")
FAMILY = schedule.CHUNK.family


@lru_cache(None)
def extension():
    """The loaded ``isoexec_kda_glm_sm100_cuda`` module; compiled on the first call, never on import."""
    ext = build_sm100_extension(FAMILY, __file__, SOURCES, include_csrc=True,
                                extra_cuda_flags=(*DIAGNOSTIC_FLAGS, *schedule.cuda_defines(schedule.CHUNK)))
    want = schedule.segments(schedule.CHUNK).S
    if ext.chunk_len() != want:
        raise RuntimeError(f"isoexec_kda_glm_sm100_cuda was compiled with chunk length {ext.chunk_len()}, the "
                           f"schedule policy gives {want}; remove the stale build directory of {ext.__file__}")
    return ext
