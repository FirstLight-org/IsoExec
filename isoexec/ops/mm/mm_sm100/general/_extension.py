"""Build the mm_sm100/general extension lazily (no build on import)."""

from functools import lru_cache

from isoexec.ops.build_cache import build_sm100_extension, sm100_profile_digest

SOURCES = ("gen.cu",)
EXTRA_CUDA_FLAGS = ("-Xptxas=-warn-spills",)


@lru_cache(None)
def extension():
    """The loaded ``isoexec_mmgen_sm100_cuda`` module; compiled on the first call."""
    return build_sm100_extension("mmgen", __file__, SOURCES, include_csrc=True, include_cutlass=True,
                                 extra_cuda_flags=EXTRA_CUDA_FLAGS)


@lru_cache(None)
def profile_digest() -> str:
    from pathlib import Path

    return sm100_profile_digest("mmgen", Path(__file__).resolve().parent / "csrc",
                                extra_cuda_flags=EXTRA_CUDA_FLAGS)
