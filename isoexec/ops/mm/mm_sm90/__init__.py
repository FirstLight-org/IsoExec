"""SM90 (Hopper) segment-tree dense GEMM: ``A[M,K] @ B[N,K].T``, bf16 in and out (recipe sm90_wgmma_segtree_v1)."""

from .spec import ARCH, IMPL_ID, IMPLS, RECIPE_VERSION

__all__ = ["ARCH", "IMPLS", "IMPL_ID", "RECIPE_VERSION"]
