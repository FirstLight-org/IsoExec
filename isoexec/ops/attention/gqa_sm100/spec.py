"""Import-light declarations for SM100 GQA attention: the ``sm100_gqa_segtree_v2`` recipe.

The forward is this package's own kernel (``csrc/``): fixed ``SEG``-token segments anchored at key 0,
merged by the canonical adjacent-pair tree, with one rounding at the root (README.md section 1). The
trainer core (packed THD rows) and the engine core (paged KV) launch the same kernel and differ only
in how the leaf addresses its keys, so they form one twin class: ``tests/test_seam_gpu.py`` proves
trainer rows == paged prefill rows == chunked prefill rows == decode rows bitwise, across batch
composition, page layout, CUDA-graph replay and the tensor-parallel head split, and
``tests/test_relations_gpu.py`` and ``tests/test_decode_gpu.py`` prove the program's execution freedoms
(among them the head-dim-256 decode leaf) change no bit.

The backward is FlashAttention-4's SM100 backward, vendored at ``FA4_COMMIT`` under
``isoexec/ops/_vendor/flash_attn_cute`` and run on this forward's output and LSE (ops.py). It is
deterministic per layout and is not part of the cross-host contract: gradients need not be equal
across tensor-parallel layouts.

The numerical identity of the forward is the recipe (README.md section 1) + ``SEG_CONTEXT`` (through
``ops.SEG_TOKENS``) + the SM100 build profile (``build_cache.sm100_cuda_flags``). A change to any of
them is a recipe version bump.
"""

from __future__ import annotations

from isoexec.core.registry import ImplSpec

ARCH = "sm100"

#: One KV-reduction program per row: the canonical tree over SEG segments, not a split heuristic.
NUM_SPLITS = 1

#: The recipe id and version. ``sm100_gqa_segtree_v2`` names the lazy-rescale leaf (README.md 1.0).
RECIPE = "sm100_gqa_segtree_v2"
RECIPE_VERSION = 2

#: The context the segment span is read at (``ops.seg_for(head_dim, SEG_CONTEXT)``). It is a
#: constant of the recipe, not a deployment input, so the trainer and every engine read the same
#: SEG whatever their own maximum length: 2048 at both head dims. Longer rows are served exactly,
#: with more segments.
SEG_CONTEXT = 32768

#: Head dims the forward compiles (``ops.HEAD_DIMS``); Q, K and V share the head dim.
HEAD_DIMS = (128, 256)

#: A paged cache's page must be a multiple of the 128-key KV tile, so a tile never crosses a page.
PAGE_MULTIPLE = 128

#: The FlashAttention commit the vendored backward was copied from (``_vendor/flash_attn_cute``).
FA4_COMMIT = "0bbb25a3a5ad3c58c029b3d287d6c9af56a5cad5"

TRAINER_IMPL = "sm100_gqa_segtree_v2_varlen"
PAGED_IMPL = "sm100_gqa_segtree_v2_paged"
TWIN_CLASS = "attention.gqa_sm100_segtree_v2"
_SM100 = frozenset({"sm100"})

IMPLS: tuple[ImplSpec, ...] = (
    ImplSpec(op="attention.core.gqa", impl_id=TRAINER_IMPL, version=RECIPE_VERSION, supported_archs=_SM100,
             twin_class=TWIN_CLASS),
    ImplSpec(op="attention.core.gqa", impl_id=PAGED_IMPL, version=RECIPE_VERSION, supported_archs=_SM100,
             twin_class=TWIN_CLASS),
)

HAZARDS = (
    "seg_mismatch",  # trainer and engine reading different SEG rows (SEG_CONTEXT is a recipe constant)
    "plan_capacity_overflow",  # the host's item/slot bound too small: part of the output unwritten
    "max_seqlen_k_understated",  # a row with more segments than max_seqlen_k / SEG sized the plan for
    "page_not_tile_multiple",  # a KV tile crossing a page boundary (refused)
    "paged_vs_packed_kv",  # block-table vs cu_seqlens_k addressing of the same logical keys
    "backward_layout_dependence",  # FA4's backward reduction order depends on the head layout
)
