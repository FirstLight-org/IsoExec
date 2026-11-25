"""Build the SM100 four-stream mHC extension lazily with the SM100 profile (``compute_100a``,
``--ftz=false``, ``--fmad=false``). Nothing here touches a device."""

from functools import lru_cache
from pathlib import Path

from isoexec.ops.build_cache import DIAGNOSTIC_FLAGS, build_sm100_extension, sm100_profile_digest
from isoexec.ops.norms.norms_sm100.schedule import cuda_defines as _norm_segment_defines

SOURCES = ("bindings.cpp", "split_fwd.cu", "split_bwd.cu", "merge.cu", "deepseek_rows.cu", "gather.cu")
FAMILY = "mhc4"
#: the fused norm kernels take their segment length from norms_sm100 (csrc/norm_segments.cuh)
CUDA_FLAGS = (*DIAGNOSTIC_FLAGS, *_norm_segment_defines())


@lru_cache(None)
def extension():
    """The loaded ``isoexec_mhc4_sm100_cuda`` module; compiled on the first call, never on import."""
    return build_sm100_extension(FAMILY, __file__, SOURCES, include_csrc=True, extra_cuda_flags=CUDA_FLAGS)


@lru_cache(None)
def profile_digest() -> str:
    return sm100_profile_digest(FAMILY, Path(__file__).parent / "csrc", extra_cuda_flags=CUDA_FLAGS)
