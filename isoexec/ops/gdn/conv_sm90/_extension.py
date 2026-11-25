"""Build the SM90 conv extension lazily with PyTorch's CUDA C++ loader (no build on import)."""

from functools import lru_cache

from isoexec.ops.build_cache import DIAGNOSTIC_FLAGS, build_sm90_extension

SOURCES = ("bindings.cpp", "conv_fwd.cu", "conv_bwd.cu")


@lru_cache(None)
def extension():
    """The loaded ``isoexec_conv_sm90_cuda`` module; compiled on the first call, never on import."""
    return build_sm90_extension(
        "conv",
        __file__,
        SOURCES,
        include_cccl=True,
        extra_cuda_flags=DIAGNOSTIC_FLAGS,
    )
