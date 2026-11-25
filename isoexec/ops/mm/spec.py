"""Declarations for the installed batch-invariant GEMM providers, without a general bitwise twin claim.
The installed provider determines implementation identity."""

from __future__ import annotations

from isoexec.core.registry import ImplSpec

_SM90 = frozenset({"sm90"})

IMPLS: tuple[ImplSpec, ...] = (
    # The vendored batch-invariant Triton matmul. On sm90 it is the fall-through beneath the segment-tree
    # GEMM (shapes that recipe does not express: fp32 router, N = 1 gate, bias, K % 64), and the
    # recorded provider of a process that has not installed the segment-tree GEMM.
    ImplSpec(op="mm", impl_id="triton_batch_invariant", version=1, supported_archs=_SM90),
)


from isoexec.ops.mm.mm_sm100.spec import IMPLS as _SM100_IMPLS

IMPLS += _SM100_IMPLS
from isoexec.ops.mm.mm_sm90.spec import IMPLS as _SM90_SEGTREE_IMPLS

IMPLS += _SM90_SEGTREE_IMPLS
