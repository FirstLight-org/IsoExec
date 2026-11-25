"""Lazy declarations and documented dtype boundaries for scoring.
The head uses bf16 GEMM followed by fp32 widening; these constants document the contract and do not enforce it."""

from typing import TYPE_CHECKING

from isoexec.ops._dispatch import family_impls

if TYPE_CHECKING:  # never executed: it names what the lazy ``__getattr__`` below otherwise hides.
    from isoexec.core.registry import ImplSpec

    IMPLS: tuple[ImplSpec, ...]

#: The head GEMM's operand dtype and the region it runs in.
HEAD_GEMM_DTYPE = "bf16"
HEAD_GEMM_REGION = "mm"

#: The widening at the head's output, before the log-softmax row.
HEAD_BOUNDARY_OUT_DTYPE = "fp32"

__all__ = ("HEAD_BOUNDARY_OUT_DTYPE", "HEAD_GEMM_DTYPE", "HEAD_GEMM_REGION", "IMPLS")


__getattr__ = family_impls(__name__, "logprobs")
