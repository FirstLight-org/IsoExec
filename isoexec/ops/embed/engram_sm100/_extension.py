"""Build the SM100 Engram extension lazily (never on import; nothing here touches a device).
Family key ``engram`` -> ``isoexec_engram_sm100_cuda``."""

from functools import lru_cache
from pathlib import Path

from isoexec.ops.build_cache import DIAGNOSTIC_FLAGS, build_sm100_extension, sm100_profile_digest

SOURCES = ("bindings.cpp", "engram.cu")
FAMILY = "engram"


@lru_cache(None)
def extension():
    return build_sm100_extension(FAMILY, __file__, SOURCES, include_csrc=True, include_cccl=True,
                                 extra_cuda_flags=DIAGNOSTIC_FLAGS)


@lru_cache(None)
def profile_digest() -> str:
    return sm100_profile_digest(FAMILY, Path(__file__).parent / "csrc", extra_cuda_flags=DIAGNOSTIC_FLAGS)
