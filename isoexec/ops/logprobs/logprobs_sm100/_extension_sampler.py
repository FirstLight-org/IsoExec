"""Build the SM100 decode sampler extension lazily (no build on import)."""

from functools import lru_cache

from isoexec.ops.build_cache import DIAGNOSTIC_FLAGS, build_sm100_extension

SOURCES = ("sampler_bindings.cpp", "sampler.cu")


@lru_cache(None)
def extension():
    """The loaded ``isoexec_logprobs_sampler_sm100_cuda`` module; compiled on the first call."""
    return build_sm100_extension("logprobs_sampler", __file__, SOURCES, extra_cuda_flags=DIAGNOSTIC_FLAGS)
