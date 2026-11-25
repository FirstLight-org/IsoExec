"""Build the SM100 logprobs extension lazily with PyTorch's CUDA C++ loader (no build on import)."""

from functools import lru_cache

from isoexec.ops.build_cache import DIAGNOSTIC_FLAGS, build_sm100_extension
from isoexec.ops.logprobs.logprobs_sm100.schedule import cuda_defines

SOURCES = ("bindings.cpp", "logprobs.cu")


@lru_cache(None)
def extension():
    """The loaded ``isoexec_logprobs_sm100_cuda`` module; compiled on the first call, never on import."""
    return build_sm100_extension(
        "logprobs",
        __file__,
        SOURCES,
        include_csrc=True,
        extra_cuda_flags=(*DIAGNOSTIC_FLAGS, *cuda_defines()),
    )
