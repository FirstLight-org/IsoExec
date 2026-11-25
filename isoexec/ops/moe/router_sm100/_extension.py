"""Build the SM100 router extension lazily with PyTorch's CUDA C++ loader.

Nothing is built on import: ``extension()`` is the only entry point and it caches. The profile is
``build_cache.sm100_cuda_flags`` -- ``compute_100a``, ``--ftz=false``, ``--fmad=false`` -- and
CUTLASS is required (cute supplies the tcgen05, TMA and TMEM instruction wrappers).

The family key is ``moe_router`` and not ``moe``: ``build_sm100_extension`` names the module
``isoexec_<family>_sm100_cuda``, and ``moe_sm100`` already owns ``isoexec_moe_sm100_cuda``. The
relation that feeds this router's table into ``moe_sm100``'s dispatch loads both in one process.
"""

from functools import lru_cache
from pathlib import Path

from isoexec.ops.build_cache import DIAGNOSTIC_FLAGS, build_sm100_extension, sm100_profile_digest

SOURCES = ("bindings.cpp", "router.cu", "backward.cu")
FAMILY = "moe_router"


@lru_cache(None)
def extension():
    """The loaded ``isoexec_moe_router_sm100_cuda`` module; compiled on the first call."""
    return build_sm100_extension(
        FAMILY,
        __file__,
        SOURCES,
        include_csrc=True,
        include_cccl=True,
        include_cutlass=True,
        extra_cuda_flags=DIAGNOSTIC_FLAGS,
    )


@lru_cache(None)
def profile_digest() -> str:
    """Flags + CUTLASS identity + source bytes: the identity of the compiled program."""
    return sm100_profile_digest(FAMILY, Path(__file__).parent / "csrc", extra_cuda_flags=DIAGNOSTIC_FLAGS)
