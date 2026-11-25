"""Import-light declarations for SM90 GQA attention: vendored FlashAttention-3 (hopper), one KV split or a fixed segment split.

The trainer core (packed THD varlen, forward + FA3's own backward) and the engine core (paged KV)
run the same FA3 kernels, built from ``fa3/`` with the profile below, under the segment split
(``SEG_*`` below, recipe ``sm90_fa3_segsplit_v1``, ``RECIPE_VERSION`` 2): the batch-invariant replacement
for FA3's split-KV. The KV axis is cut at fixed logical positions, so a row can run split (one work tile
per segment, for occupancy) or folded (one work tile walking its segments, when the GPU is saturated) and
give the same bits either way. The two cores form one twin class: ``tests/test_seam_gpu.py`` and
``tests/test_segsplit_gpu.py`` prove trainer rows == paged prefill == chunked prefill == decode rows
bitwise, on every execution path. Rows of at most SEG keys have exactly the bits of the previous ns1
recipe (``num_splits=1``, version 1), which absorbed MLA still pins.

The numerical identity is FA3 at the pinned commit + ``BUILD_DEFINES`` + ``NVCC_FLAGS`` + the pinned
CUTLASS headers (+ ``SEG_TOKENS`` and ``fa3/seg_merge.h`` under the segment split). A change to any of
them is a recipe version bump.
"""

from __future__ import annotations

from isoexec.core.registry import ImplSpec

#: One KV-reduction split (FA3 ``num_splits=1``): the ns1 recipe, and absorbed MLA's pin. The C++ refuses FA3's
#: proportional split (``num_splits > 1`` or its heuristic ``0``); a negative value selects the segment split.
NUM_SPLITS = 1

#: Segment split (recipe ``SEG_RECIPE``): every row's KV axis is cut into FIXED logical segments of ``SEG_TOKENS``
#: keys anchored at key 0, ``[s*SEG, (s+1)*SEG)``. Each segment is one FA3 online-softmax pass (the unsplit
#: mainloop restricted to the segment's 128/80-key tiles, newest tile first, state reset per segment) ending in
#: FA3's finalized fp32 ``(O_s, lse_s)``; a row's ``n = ceil(border / SEG)`` results are RIGHT-folded,
#: ``state = merge(seg[s], state)`` for s = n-2..0, with the one merge of ``fa3/seg_merge.h``. Two execution
#: paths run that program and are interchangeable per call: FOLD (the unsplit kernel walks a work tile's
#: segments and folds each into an L2-resident per-CTA slot -- the saturated path) and SPLIT (one work tile
#: per segment, partials in HBM, ``fa3/seg_combine.cu``). ``n == 1`` is the ns1 kernel's bits exactly (same
#: mainloop, same rounding of the same fp32 value). SEG is a constant of the recipe:
#: it never depends on batch, packing, chunk boundary, page layout, grid or SM count. It must be a multiple of
#: every KV tile it serves (80 keys at head dim 256, 128 at 64/128) -> a multiple of 640.
SEG_TOKENS = 2560
SEG_RECIPE = "sm90_fa3_segsplit_v1"
#: Head dims the segment split is built for (Q/K == V); the HDIMDIFF64 MLA kernels stay one-split.
SEG_HEAD_DIMS = (64, 128, 256)
#: The engine/trainer cores run the segment split (True, the release recipe) or the ns1 recipe (False; that
#: is a different twin class and recipe version, kept only for the absorbed-MLA pin and A/B comparisons).
CORE_USES_SEGMENT_SPLIT = True
#: fp32 elements of the engine workspace kept for split partials beyond the per-call regions: covers
#: n_seg * Hq * B * (Dv + 1) for decode up to e.g. B=256, Hq=6, Dv=256 at 40 segments (~100k keys). A call
#: needing more runs the fold path (same bits).
PAGED_PARTIAL_FLOATS = 16 << 20

#: The FA3 commit ``fa3/`` was copied from (fa3/PROVENANCE.json states the same).
FA3_COMMIT = "e81a70ce92d6dd96e213c08adbadd7c7e3d17f01"

#: Feature switches of the build. bf16 SM90 only; head dims 64/128/256 plus the HDIMDIFF64 (64, 256)
#: and (64, 512) forward kernels for absorbed MLA; varlen and paged KV kept; fp16/fp8/local/softcap/
#: sm80 out. Split-KV is built for the segment split only (block.h replaces FA3's proportional split
#: boundaries with the fixed segment; fa3/seg_combine.cu replaces FA3's combine).
BUILD_DEFINES = (
    "FLASHATTENTION_DISABLE_LOCAL",
    "FLASHATTENTION_DISABLE_SOFTCAP",
    "FLASHATTENTION_DISABLE_FP16",
    "FLASHATTENTION_DISABLE_FP8",
    "FLASHATTENTION_DISABLE_HDIM96",
    "FLASHATTENTION_DISABLE_HDIM192",
    "FLASHATTENTION_DISABLE_SM8x",
    "FLASHATTENTION_DISABLE_HDIMDIFF192",
)

#: FA3's own nvcc flags (hopper/setup.py at FA3_COMMIT), in its order. ``--use_fast_math`` is FA3's
#: recipe; IsoExec's ``NUMERICS_FLAGS`` do not apply to this vendored kernel.
NVCC_FLAGS = (
    "-O3",
    "-std=c++17",
    "--ftemplate-backtrace-limit=0",
    "--use_fast_math",
    "--resource-usage",
    "-lineinfo",
    "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED",
    "-DCUTLASS_ENABLE_GDC_FOR_SM90",
    "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
    "-DNDEBUG",
    "-gencode=arch=compute_90a,code=sm_90a",
)
#: Host flags for flash_api_stable.cpp (setup.py: ``cxx`` args, torch-stable-ABI target 2.9).
CXX_FLAGS = ("-O3", "-std=c++17", "-DPy_LIMITED_API=0x03090000", "-DTORCH_TARGET_VERSION=0x0209000000000000")

#: Q/K head-dim kernels built (``round_up_headdim``); V may be wider only through HDIMDIFF64.
HEAD_DIMS = (64, 128, 256)
#: (Q/K, V) forward kernels with a V wider than Q/K.
HDIMDIFF64 = ((64, 256), (64, 512))
#: FA3 compiles its deterministic backward only below head dim 256 (flash_bwd_launch_template.h:
#: ``Deterministic_ && kHeadDim < 256``; flash_api refuses ``deterministic`` at 256). Above it the
#: backward is FA3's atomic kernel, or torch's FA2 deterministic one when determinism is required (ops.py).
DETERMINISTIC_BWD_MAX_HEAD_DIM = 128

TRAINER_IMPL = "sm90_fa3_segsplit_v1_varlen"
PAGED_IMPL = "sm90_fa3_segsplit_v1_paged"
TWIN_CLASS = "attention.gqa_sm90_segsplit_v1"
#: 1 was the ns1 recipe (``sm90_fa3_varlen_ns1`` / ``sm90_fa3_paged_ns1``, twin class ``attention.gqa_sm90_ns1``).
RECIPE_VERSION = 2
_SM90 = frozenset({"sm90"})

IMPLS: tuple[ImplSpec, ...] = (
    ImplSpec(op="attention.core.gqa", impl_id=TRAINER_IMPL, version=RECIPE_VERSION, supported_archs=_SM90,
             twin_class=TWIN_CLASS),
    ImplSpec(op="attention.core.gqa", impl_id=PAGED_IMPL, version=RECIPE_VERSION, supported_archs=_SM90,
             twin_class=TWIN_CLASS),
)

HAZARDS = (
    "head_dim_fallthrough",  # a compiled-out head dim silently runs a wider kernel (forward_kernel refuses it)
    "split_kv_query_length_dependence",  # FA3's proportional split (refused by the C++; only 1 or -SEG pass)
    "segment_boundary_anchor",  # a segment anchored on anything but logical key 0 (block.h, seg_combine.cu)
    "pack_gqa_heuristic",  # FA3 chooses PackGQA per launch; per-row arithmetic must not depend on it
    "hdim256_backward_nondeterminism",  # FA3 has no deterministic backward at 256 (atomic dQ/dK/dV)
    "paged_vs_packed_kv",  # TMA vs non-TMA paged loads vs contiguous varlen
)
