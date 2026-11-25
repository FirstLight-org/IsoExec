"""Build the SM90 CPR GDN extension lazily with PyTorch's CUDA C++ loader."""

import sys
from functools import lru_cache
from pathlib import Path

from isoexec.ops.build_cache import DIAGNOSTIC_FLAGS, build_sm90_extension, sm90_build_dir

SOURCES = ("bindings.cpp", "prefill.cu", "wy.cu", "sweep.cu", "fused.cu", "decode.cu", "backward.cu", "rows.cu")
FAMILY = "gdn_cpr"


@lru_cache(None)
def extension():
    """The loaded ``isoexec_gdn_cpr_sm90_cuda`` module; compiled on the first call, never on import."""
    return build_sm90_extension(FAMILY, __file__, SOURCES, extra_cuda_flags=DIAGNOSTIC_FLAGS)


def build_dir() -> Path:
    """The directory the extension builds in (``ISOEXEC_GDN_CPR_SM90_BUILD_DIR`` or the torch extension cache),
    for side builds such as the hardware-facts tests to sit beside it."""
    import torch

    tag = f"torch-{torch.__version__}-py{sys.version_info.major}{sys.version_info.minor}"
    return sm90_build_dir(f"ISOEXEC_{FAMILY.upper()}_SM90_BUILD_DIR", f"{FAMILY}-sm90", tag,
                          Path(__file__).resolve().parent / "csrc")
