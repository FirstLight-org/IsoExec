"""What this package owns on sm90."""

from __future__ import annotations

from isoexec.core.registry import ImplSpec

ARCH = "sm90"

#: The numerical program: K cut into segments that are a function of K alone (``segments.py``), each
#: segment accumulated from zero in ascending K by one chain of ``wgmma.mma_async`` k16 steps in fp32
#: (scale-d = 0 on the first step), segments merged by the balanced adjacent-pair tree over the segment
#: index in fp32 (``pik.plan.combine_order``), one BF16 (RN) rounding at the root.
#:
#: It can never be a bitwise twin of ``sm100_tcgen05_segtree_v2``: wgmma's K step is 16 against
#: tcgen05's 32, so the fp32 accumulation history inside a segment differs.
#:
#: At S = 1 (policy v1 publishes S = 1 for every K) the program is the single ascending wgmma chain
#: that ``triton_batch_invariant`` also computes -- measured bit-equal by ``tests/test_mm_sm90_gpu.py``
#: (and, before it was retired, to the pinned cuBLASLt provider ``cublaslt_pinned`` and pik's cuBLASLt
#: leaf loop) -- but that kernel defines no segment structure, so this is its own id; the equality is
#: evidence, not a twin claim. It replaces ``cublaslt_pinned`` and ``mm_tiles`` as the sm90 provider.
RECIPE_VERSION = "sm90_wgmma_segtree_v1"

IMPL_VERSION = 1

IMPL_ID = "sm90_wgmma_segtree"

OPS = ("mm",)

#: The twin class this recipe opens (recorded for the battery, not on the ImplSpec field: one member).
TWIN_CLASS = "mm.dense.segtree.sm90"

FLAG = "ISOEXEC_MM_SM90_SEGTREE"

IMPLS: tuple[ImplSpec, ...] = (
    ImplSpec(op="mm", impl_id=IMPL_ID, version=IMPL_VERSION, supported_archs=frozenset({ARCH})),
)
