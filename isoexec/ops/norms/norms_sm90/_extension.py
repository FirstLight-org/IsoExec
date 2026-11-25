"""Build the SM90 norm extension lazily with PyTorch's CUDA C++ loader (build dir outside the repo)."""

from functools import lru_cache

from isoexec.ops.build_cache import build_sm90_extension

SOURCES = ("host.cpp", "launch_fwd.cu", "launch_bwd.cu")


@lru_cache(None)
def extension():
    """The loaded ``isoexec_norms_sm90_cuda`` module; compiled on the first call, never on import."""
    return build_sm90_extension(
        "norms",
        __file__,
        SOURCES,
        include_csrc=True,
        extra_cuda_flags=("-Xptxas=-warn-spills",),
    )
