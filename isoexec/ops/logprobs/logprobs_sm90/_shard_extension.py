"""Build the SM90 vocab-parallel sampling head extension lazily (no build on import).

A second extension from ``csrc/`` (the build directory hashes the whole directory, so a change to
``logprobs.cu``, which ``shard_sample.cu`` includes, rebuilds it too)."""

from functools import lru_cache

from isoexec.ops.build_cache import DIAGNOSTIC_FLAGS, build_sm90_extension

SOURCES = ("shard_bindings.cpp", "shard_sample.cu")


@lru_cache(None)
def extension():
    """The loaded ``isoexec_logprobs_shard_sm90_cuda`` module; compiled on the first call."""
    return build_sm90_extension("logprobs_shard", __file__, SOURCES, include_cccl=True,
                                extra_cuda_flags=DIAGNOSTIC_FLAGS)
