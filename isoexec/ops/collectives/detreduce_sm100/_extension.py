"""Build the detreduce extension lazily with the shared SM100 profile (NUMERICS_FLAGS).

Nothing builds on import. ``csrc/detreduce.cu`` includes the clamped-SwiGLU scalar header
(``moe/clamped_swiglu_sm100/csrc/clamped_numerics.cuh``), so the build key and the digest fold that
tree too.
"""
import hashlib
from functools import lru_cache
from pathlib import Path

from isoexec.ops.build_cache import DIAGNOSTIC_FLAGS, build_sm100_extension, sm100_profile_digest

FAMILY = "detreduce"
SOURCES = ("bindings.cpp", "detreduce.cu", "leaf_fold.cu")
_HERE = Path(__file__).parent
_CLAMPED = _HERE.parent.parent / "moe" / "clamped_swiglu_sm100" / "csrc"


@lru_cache(None)
def extension():
    return build_sm100_extension(FAMILY, __file__, SOURCES, include_csrc=True, extra_cuda_flags=DIAGNOSTIC_FLAGS,
                                 extra_sources=(_CLAMPED,))


@lru_cache(None)
def profile_digest() -> str:
    own = sm100_profile_digest(FAMILY, _HERE / "csrc", extra_cuda_flags=DIAGNOSTIC_FLAGS)
    dep = sm100_profile_digest(FAMILY, _CLAMPED, extra_cuda_flags=DIAGNOSTIC_FLAGS)
    return hashlib.sha256((own + dep).encode()).hexdigest()[:16]
