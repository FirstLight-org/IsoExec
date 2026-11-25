"""Build the SM100 CPR GDN extension lazily with PyTorch's CUDA C++ loader."""

import sys
from functools import lru_cache
from pathlib import Path

from isoexec.ops.build_cache import DIAGNOSTIC_FLAGS, arch_build_dir, build_sm100_extension

SOURCES = ("bindings.cu", "core.cu", "prepared.cu", "decode.cu", "rows.cu")
FAMILY = "gdn_cpr"


@lru_cache(None)
def extension():
    """The loaded ``isoexec_gdn_cpr_sm100_cuda`` module; compiled on the first call, never on import."""
    return build_sm100_extension(FAMILY, __file__, SOURCES, include_csrc=True, extra_cuda_flags=DIAGNOSTIC_FLAGS)


def build_dir() -> Path:
    """The directory the extension builds in (``ISOEXEC_GDN_CPR_SM100_BUILD_DIR`` or the torch extension
    cache), so a side build such as the hardware-facts battery can sit beside it."""
    import torch

    tag = f"torch-{torch.__version__}-py{sys.version_info.major}{sys.version_info.minor}"
    return arch_build_dir(f"ISOEXEC_{FAMILY.upper()}_SM100_BUILD_DIR", f"{FAMILY}-sm100", tag,
                          Path(__file__).resolve().parent / "csrc")
