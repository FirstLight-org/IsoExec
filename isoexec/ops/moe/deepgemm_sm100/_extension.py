"""Lazy build of the MegaMoE EP data-movement kernels (``csrc/mega_ep_io.cu``) with the shared SM100 profile."""
from functools import lru_cache

from isoexec.ops.build_cache import build_sm100_extension

FAMILY = "deepgemm_mega_ep_io"
SOURCES = ("mega_ep_io.cu",)


@lru_cache(None)
def extension():
    return build_sm100_extension(FAMILY, __file__, SOURCES)
