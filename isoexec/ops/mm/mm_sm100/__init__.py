"""SM100 (Blackwell) deterministic dense GEMM: ``A[M,K] @ B[N,K].T``, bf16 in and out."""

from .spec import ARCH, IMPL_ID, IMPLS, RECIPE_VERSION

__all__ = ["ARCH", "IMPLS", "IMPL_ID", "RECIPE_VERSION"]
