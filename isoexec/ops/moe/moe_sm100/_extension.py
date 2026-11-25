"""Build the SM100 MoE extension lazily with PyTorch's CUDA C++ loader."""

from functools import lru_cache

from isoexec.ops.build_cache import DIAGNOSTIC_FLAGS, build_sm100_extension, sm100_profile_digest

SOURCES = ("bindings.cpp", "experts.cu", "route.cu")


@lru_cache(None)
def extension():
    """The loaded ``isoexec_moe_sm100_cuda`` module; compiled on the first call."""
    return build_sm100_extension(
        "moe",
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
    from pathlib import Path

    return sm100_profile_digest("moe", Path(__file__).parent / "csrc", extra_cuda_flags=DIAGNOSTIC_FLAGS)
