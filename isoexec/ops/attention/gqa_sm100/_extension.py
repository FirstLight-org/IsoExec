"""Build and cache the sm100 GQA extension. Importing this module does not build anything."""

from __future__ import annotations

from functools import lru_cache

from isoexec.ops.build_cache import DIAGNOSTIC_FLAGS, build_sm100_extension

SOURCES = ("bindings.cpp", "host.cu", "plan.cu", "leaf_d128.cu", "leaf_d256.cu", "leaf_d256_fold.cu",
           "combine.cu", "decode_d256.cu")


@lru_cache(None)
def extension():
    """The compiled kernel, built with the sm100 profile (``--fmad=false``, ``--ftz=false``)."""
    return build_sm100_extension(
        "gqa", __file__, SOURCES, include_csrc=True, extra_cuda_flags=DIAGNOSTIC_FLAGS
    )


__all__ = ["SOURCES", "extension"]
