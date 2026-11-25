"""Build the SM100 mm extension lazily with PyTorch's CUDA C++ loader (no build on import)."""

from functools import lru_cache

from isoexec.ops.build_cache import build_sm100_extension, sm100_profile_digest

SOURCES = ("mm_sm100_gemm.cu",)

#: Reported in the README beside every measurement: which CUTLASS checkout and which
#: nvcc flags produced the machine code the numbers were taken on.
EXTRA_CUDA_FLAGS = ("-Xptxas=-v", "-Xptxas=-warn-spills")


@lru_cache(None)
def extension():
    """The loaded ``isoexec_mm_sm100_cuda`` module; compiled on the first call, never on import."""
    return build_sm100_extension(
        "mm",
        __file__,
        SOURCES,
        include_csrc=True,
        include_cutlass=True,
        extra_cuda_flags=EXTRA_CUDA_FLAGS,
    )


@lru_cache(None)
def profile_digest() -> str:
    """Flags + CUTLASS identity + source bytes: the build identity of a measurement."""
    from pathlib import Path

    return sm100_profile_digest(
        "mm", Path(__file__).resolve().parent / "csrc", extra_cuda_flags=EXTRA_CUDA_FLAGS
    )
