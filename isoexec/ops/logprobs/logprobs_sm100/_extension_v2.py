"""Build the SM100 logprobs **recipe v2** extension lazily (no build on import)."""

from functools import lru_cache

from isoexec.ops.build_cache import DIAGNOSTIC_FLAGS, build_sm100_extension

SOURCES = ("bindings_v2.cpp", "logprobs_v2.cu")


@lru_cache(None)
def extension():
    """The loaded ``isoexec_logprobs_v2_sm100_cuda`` module; compiled on the first call."""
    return build_sm100_extension(
        "logprobs_v2",
        __file__,
        SOURCES,
        include_csrc=True,
        extra_cuda_flags=DIAGNOSTIC_FLAGS,
    )
