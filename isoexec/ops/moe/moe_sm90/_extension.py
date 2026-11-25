"""Build the SM90 MoE extension lazily with PyTorch's CUDA C++ loader."""

from functools import lru_cache

from isoexec.ops.build_cache import DIAGNOSTIC_FLAGS, build_sm90_extension

SOURCES = ("bindings.cpp", "gemm.cu", "routing.cu")


@lru_cache(None)
def extension():
    """The loaded ``isoexec_moe_sm90_cuda`` module; compiled on the first call, never on import."""
    return build_sm90_extension(
        "moe",
        __file__,
        SOURCES,
        include_cccl=True,
        extra_cuda_flags=DIAGNOSTIC_FLAGS,
    )
