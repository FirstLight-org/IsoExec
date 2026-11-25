"""Build and load the mla_sm100 forward extension under one declared SM100 numerical profile."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from isoexec.ops.build_cache import (
    build_sm100_extension,
    cutlass_identity,
    sm100_cuda_flags,
    sm100_profile_digest,
)

__all__ = ["extension", "profile_digest", "profile_flags", "cutlass_id", "SOURCES"]

_SRC = Path(__file__).resolve().parent / "csrc"
SOURCES = ("fwd/bindings.cpp", "fwd/leaf_bf16.cu", "fwd/plan.cu", "fwd/combine.cu")

#: Flags appended to the shared SM100 profile.  Order is part of the digest.
EXTRA_CUDA_FLAGS = (
    "--ftz=true",
    # cute's TMEM copy helpers assert the thread-to-data-path mapping on every tcgen05 access, which
    # without NDEBUG puts an ISETP + BRA to __assertfail around each one and splits the basic block.
    # sm90's profile carries -DNDEBUG for the same reason; asserts are a development aid, not part of
    # the arithmetic.
    "-DNDEBUG",
    "--prec-div=false",
    "-lineinfo",
    "--ptxas-options=-v,--register-usage-level=10,--warn-on-spills",
    "--threads", "8",
)


def profile_flags() -> tuple[str, ...]:
    """The full nvcc arithmetic profile this op is compiled under."""
    return sm100_cuda_flags(extra=EXTRA_CUDA_FLAGS)


def profile_digest() -> str:
    """Identity of one build: the flag profile, the CUTLASS headers and the kernel sources."""
    return sm100_profile_digest("mla", _SRC, extra_cuda_flags=EXTRA_CUDA_FLAGS)


def cutlass_id() -> str:
    """Which CUTLASS checkout compiled the UMMA and TMA descriptors."""
    return cutlass_identity("mla")


@lru_cache(None)
def extension():
    """Build (once) and load the forward extension."""
    return build_sm100_extension(
        "mla",
        __file__,
        SOURCES,
        include_csrc=True,
        include_cutlass=True,
        extra_cuda_flags=EXTRA_CUDA_FLAGS,
    )
