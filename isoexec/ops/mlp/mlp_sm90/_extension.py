"""Build the SM90 mlp.act extension lazily with PyTorch's CUDA C++ loader (build dir outside the repo;
``ISOEXEC_MLP_SM90_BUILD_DIR`` overrides it)."""

from functools import lru_cache

from isoexec.ops.build_cache import build_sm90_extension

SOURCES = ("host.cpp", "silu_mul.cu")


@lru_cache(None)
def extension():
    """The loaded ``isoexec_mlp_sm90_cuda`` module; compiled on the first call, never on import."""
    return build_sm90_extension("mlp", __file__, SOURCES, include_csrc=True)
