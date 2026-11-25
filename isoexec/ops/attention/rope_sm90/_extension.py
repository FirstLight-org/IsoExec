"""Build the SM90 rope extension lazily with PyTorch's CUDA C++ loader."""

from functools import lru_cache

from isoexec.ops.build_cache import build_sm90_extension

SOURCES = ("bindings.cpp", "rope.cu", "qknorm.cu")


@lru_cache(None)
def extension():
    """The loaded ``isoexec_rope_sm90_cuda`` module; compiled on the first call, never on import."""
    return build_sm90_extension(
        "rope",
        __file__,
        SOURCES,
        # qknorm.cu's warp reduction is CCCL's <cuda/warp> (CCCL 3.x), which a toolkit whose own
        # bundled CCCL predates it carries beside itself.
        include_cccl=True,
    )
