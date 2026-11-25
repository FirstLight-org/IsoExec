"""Build the SM100 mlp.act extension lazily with PyTorch's CUDA C++ loader (build dir outside the repo)."""

from functools import lru_cache

from isoexec.ops.build_cache import build_sm100_extension

SOURCES = ("host.cpp", "silu_mul.cu")


@lru_cache(None)
def extension():
    """The loaded ``isoexec_mlp_sm100_cuda`` module; compiled on the first call, never on import."""
    return build_sm100_extension("mlp", __file__, SOURCES, include_csrc=True)
