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

The numerical identity of the forward is the recipe (README.md section 1) + the declarations below
(through ``ops.schedule_for``, i.e. ``isoexec.ops.schedule_policy``) + the SM100 build profile (``build_cache.sm100_cuda_flags``). A change to any of
them is a recipe version bump.
"""

from __future__ import annotations

from isoexec.core.registry import ImplSpec

ARCH = "sm100"

#: One KV-reduction program per row: the canonical tree over SEG segments, not a split heuristic.
NUM_SPLITS = 1

#: The recipe id and version. ``sm100_gqa_segtree_v2`` names the lazy-rescale leaf (README.md 1.0).
RECIPE = "sm100_gqa_segtree_v2"
#: 3: the segment span is the schedule policy's (one segment of ``LONGEST_CONTEXT`` = 10240 keys at the
#: declared batch) instead of the max-context table's 2048. The program and the recipe id are unchanged.
RECIPE_VERSION = 3

# --- declarations the schedule policy reads (design version 7.1; ops.schedule_for) --------------------
# Constants of the recipe, not per-call inputs, so the trainer and every engine read the same segment
# span whatever their own batch, length or tensor-parallel degree. This tree's deployment config
# (isoexec.config.IsoExecConfig) carries an engine-only ``max_model_len`` and no smallest batch, so the
# three declarations live here.

#: The declared longest context, in keys: the production ``max_model_len`` this tree's seam batteries
#: name (tests/test_seam_gpu.py, gqa_sm90/tests/test_seam_gpu.py). Longer rows are served exactly, with
#: more segments. The head-dim-256 decode leaf holds one segment of at most 128 tiles (16384 keys,
#: ``csrc/decode_d256.cu MAX_SLOT_CACHE``) and refuses a longer span, so a declaration whose span exceeds
#: that needs the leaf's tile cache raised first (tests/test_schedule_policy_cpu.py checks the bound).
LONGEST_CONTEXT = 10240

#: The declared smallest batch of a decode step (sequences per engine replica). 128 is the RL
#: deployment of policy_validation/gqa_sm100/ROUND_B.md (round E: 32 prompts x 8 samples over two
#: replicas); a deployment that decodes at batch 1-4 pays up to 2.6x for it (same file).
SMALLEST_BATCH = 128

#: KV heads of the MODEL (never a rank's): the smallest of the release models' attention layers
#: (Qwen3.5-0.8B and 35B-A3B: 2). At ``SMALLEST_BATCH`` every model of the tree gets the same span
#: (tests/test_schedule_policy_cpu.py checks 2, 4, 8 and 16).
MODEL_KV_HEADS = 2

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
    "seg_mismatch",  # trainer and engine reading different declarations (they are recipe constants)
    "plan_capacity_overflow",  # the host's item/slot bound too small: part of the output unwritten
    "max_seqlen_k_understated",  # a row with more segments than max_seqlen_k / SEG sized the plan for
    "page_not_tile_multiple",  # a KV tile crossing a page boundary (refused)
    "paged_vs_packed_kv",  # block-table vs cu_seqlens_k addressing of the same logical keys
    "backward_layout_dependence",  # FA4's backward reduction order depends on the head layout
)
