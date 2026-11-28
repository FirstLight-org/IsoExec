"""Build the SM100 fused-glue extension lazily (never on import) with the SM100 build profile
(``compute_100a``, ``--ftz=false``, ``--fmad=false``)."""

from functools import lru_cache
from pathlib import Path

from isoexec.ops.build_cache import DIAGNOSTIC_FLAGS, build_sm100_extension, sm100_profile_digest

SOURCES = ("bindings.cpp", "glue.cu")
FAMILY = "fused_glue"


@lru_cache(None)
def extension():
    """The loaded ``isoexec_fused_glue_sm100_cuda`` module; compiled on the first call."""
    return build_sm100_extension(FAMILY, __file__, SOURCES, include_csrc=True,
                                 extra_cuda_flags=DIAGNOSTIC_FLAGS)


@lru_cache(None)
def profile_digest() -> str:
    return sm100_profile_digest(FAMILY, Path(__file__).parent / "csrc", extra_cuda_flags=DIAGNOSTIC_FLAGS)
