# gqa_sm100: deterministic causal GQA on Blackwell (SM100)

`attention.core.gqa` for B200: bf16 in and out, fp32 accumulate, causal, packed varlen queries over a
paged or packed KV cache. **Head dim 128 and head dim 256**, query-to-KV head ratios 1 to 16 (dividing
128). One numerical program for training, prefill, chunked prefill and decode, and one numerical
program across the two head dims -- D256 runs a different kernel (a cluster of two CTAs,
`tcgen05.mma.cta_group::2`), not a different recipe.

* **Forward**: this package's kernel (`csrc/`), recipe `sm100_gqa_segtree_v2` (section 1). The trainer
  core (`sm100_gqa_segtree_v2_varlen`, packed THD rows) and the engine core
  (`sm100_gqa_segtree_v2_paged`, vLLM's paged cache) launch the same kernel; they differ only in how
  the leaf addresses its keys, and form the twin class `attention.gqa_sm100_segtree_v2`.
* **Backward**: FlashAttention-4's SM100 backward, vendored in `isoexec/ops/_vendor/flash_attn_cute`
  (section 4).

The forward's execution is derived from FlashAttention-4's SM100 forward (Dao-AILab/flash-attention
`flash_attn/cute`, BSD-3-Clause, Copyright (c) 2025 Tri Dao et al.) through TileFoundry's CUDA port of
it; the derived sources carry the notice. **Its numerical program is not FA4's** and is stated below.

## 1. The numerical contract

Recipe id **`sm100_gqa_segtree_v2`**. Every item in this section is the op's identity: a change to any
of it is a version bump, on this architecture and on any other that ever claims the same class.

### 1.0 Recipe versions

| id | what it fixed |
|---|---|
| `sm100_gqa_segtree_v1` | the running max inside a segment is EXACT: `RESCALE_THRESHOLD = 0`, so the output accumulator is rescaled at every KV tile whose max advances. Superseded. |
| **`sm100_gqa_segtree_v2`** | **the running max is LAZY at FA4's threshold of 8 log2 units** (section 1.4 item 2); everything else -- segments, tree, merge operator, exp2 selection, rounding points -- is unchanged |

v1's exact max put an O rescale on nearly every KV tile; v2's lazy one runs about 1.3x faster on
long single-segment rows.

**Why the lazy rescale is admissible.** The decision is `keep = (M_old - max(this tile)) * log2(e) *
softmax_scale >= -8`. It reads the row's own running max over its fixed 128-key tiles **in logical
order inside one segment**, and the running max is reset at every segment start. It is therefore a
pure function of `(logical position, KV bits, deployment constants)` and is batch-invariant: it cannot
see the batch, the packing order, the chunk boundary, the page layout, the split count, the grid, the
SM count or any other row. The published `(O, L, M)` stays exactly self-consistent (`O` and `L` are
both taken against the `M` that is published), so the merge, the root and the lse are untouched.

### 1.1 Logical coordinates

Query row `i` of request `b` (packed at absolute token `t`, `cu_seqlens_q[b] <= t < cu_seqlens_q[b+1]`)
sits at logical position

```
p = seqused_k[b] - (cu_seqlens_q[b+1] - t) ,     border = p + 1
```

and attends keys `[0, p]`. Batch slot, packing order, page ids, block-table layout, chunk offsets,
grid and worker count are **invisible to the arithmetic**: masks, the causal bound and the segment
anchor all read `p`, and the two KV addressing modes (paged, packed) hand the leaf the same logical
keys in the same order.

### 1.2 KV tile width

**128 keys.** Numerical, because every tile boundary inside a segment is a max/rescale point,
unlike a GEMM's K staging. A paged cache must therefore have
`page % 128 == 0`; the call refuses otherwise rather than gathering across a page boundary.

### 1.3 Segments, anchored at key 0

A row with border `B` is made of `n = ceil(B / SEG)` segments

```
[k*SEG, min((k+1)*SEG, B))      for k = 0 .. n-1
```

`SEG` is **fixed**: a table over `(head dim, max context)` in `ops.SEG_TOKENS`, read through
`ops.seg_for`. It never depends on batch, sequence length, KV length, chunk boundary, split
count, grid or SM count.

| head dim | max context | SEG | n at the max context |
|---|---|---|---|
| 128 | 2048 | 128 | 16 |
| 128 | 8192 | 512 | 16 |
| 128 | 32768 | 2048 | 16 |
| 128 | 131072 | 8192 | 16 |
| 256 | 2048 | 128 | 16 |
| 256 | 8192 | 512 | 16 |
| 256 | 32768 | 2048 | 16 |
| 256 | 131072 | 8192 | 16 |

The rule is `n_max = 16`, floored at one KV tile. A small SEG publishes a partial per segment and
loses on long-query rows; a large SEG leaves decode with too few workers per (request, KV head).

**The release reads one row of this table: `spec.SEG_CONTEXT = 32768`, so `SEG = 2048` at both head
dims** (`ops.segment_span`). It is a constant of the recipe, not a deployment input, so the trainer and
every engine read the same SEG whatever their own maximum length. A row longer than 32768 keys is
served exactly, with more than 16 segments (the combine admits up to 4096).

### 1.4 The leaf program: one segment, one packed query tile

State per row: `O[D] fp32`, `L fp32`, `M fp32`. **Reset at every segment start** (`M = -inf`, `L = 0`,
`O = 0` via `accumulate = false` on the segment's first PV). A worker that carries an accumulator
across a segment boundary has produced the flat chain; it is a bug and the relations refuse it.

Per KV tile of the segment, per row (one thread owns one TMEM lane, so one whole row of 128 scores,
and there is **no cross-lane reduction anywhere in the softmax**):

1. masked scores are replaced by `-inf` **before the max**, by `column > border - 1 - segment_start -
   128*tile` on logical positions;
2. the row max: four independent running maxima over 32 columns each, seeded with the running max,
   then `fmaxf(fmaxf(m0,m1), fmaxf(m2,m3))`. **Lazily committed** -- `RESCALE_THRESHOLD = 8.0`, FA4's
   own value: while this tile's max exceeds the running max by at
   most 8 log2 units the running max is KEPT, the bias does not move and no rescale runs. A tile's
   largest probability is then at most `2^8 = 256`, which bf16 holds at exactly the relative precision
   it holds 1.0 at. The first tile of a segment always commits (the running max is `-inf`, so the
   shift is `-inf`). Section 1.0 states why this is a pure function of logical position and why the
   published partial stays self-consistent;
3. `safe = (max == -inf) ? 0 : max`; `shift = __fmul_rn(M_old - safe, scale_log2)`;
   `keep = shift >= -RESCALE_THRESHOLD`; when `keep`, the running max and `safe` both stay at `M_old`;
   `alpha = keep ? 1 : ex2.approx.ftz.f32(shift)`;
4. `P = exp2(score * scale_log2 + (-safe * scale_log2))`, the affine part as one
   `fma.rn.ftz.f32x2` per lane pair, then **two declared exp2 implementations** selected by the
   column pair's index inside the fixed tile and by nothing else: `ex2.approx.ftz.f32` for 96 of the
   128 columns and FA4's degree-three polynomial emulation (`ex2_emulation_2`: `2^floor(x) *
   poly3(frac)`, floor by a round-toward-minus-infinity add of `2^23 + 2^22`, the integer part folded
   back with `shl.b32` + `add.s32`) for the other 32. The pattern is FA4's own rule at
   `freq = 8, res = 4, start_frg = 1` -- fragments 1 and 2 of four, `k % 8 >= 4`;
5. `P` to bf16 by `cvt.rn.bf16x2.f32`, written into the score slot it aliases;
6. the row sum: two packed accumulators, `sum01` seeded `__fmul_rn(L_old, alpha)`, then
   `add.rn.ftz.f32x2` over four groups of 32 in steps of four, closing
   `(sum01.lo + sum01.hi) + (sum23.lo + sum23.hi)`;
7. `O *= alpha` when `alpha != 1`, as four 32-column TMEM read-modify-writes with
   `mul.rn.ftz.f32x2` -- the same 128 elements in the same order as FA4's eight 16-column
   `correction_rescale`, at half the tcgen05 instructions and half the read-after-load fences;
8. `PV` accumulates into the rescaled accumulator.

`scale_log2 = softmax_scale * log2(e)`. `M` is published in the **raw (unscaled)** frame, exactly as
the running max holds it -- which under v2 is the last *committed* max, at most 8 log2 units below the
segment's true max -- so the merge applies the scale the same way the leaf's own rescale does, and `L`
and `O` are relative to that same `M`.
An empty segment publishes the finite sentinel `M = -3.0e38f`, `L = 0`; emptiness is an integer test
(`border <= k*SEG`) and such a segment is never published and never merged.

Value rows at or past `seqused_k` are unwritten cache memory. Their probabilities are exactly `+0.0`,
but `0 * NaN` is `NaN`, so the publisher warp **zeroes those rows of the V tile** before the PV MMA
can read them. Masked *keys* need no such care: their scores are replaced before the max.

### 1.5 The merge: the canonical adjacent-pair tree

Level 0 pairs leaves `(0,1), (2,3), ...`; level `l` pairs the level-`l-1` nodes. **A node without a
partner is carried up verbatim** and produces no merge; a tree over `n` leaves performs exactly
`n - 1` merges. The lower segment index is always the left operand.

```
merge(a, b):
  M  = fmaxf(a.M, b.M)
  sa = ex2.approx.ftz.f32(__fmul_rn(__fsub_rn(a.M, M), scale_log2))
  sb = ex2.approx.ftz.f32(__fmul_rn(__fsub_rn(b.M, M), scale_log2))
  O  = a.O * sa + b.O * sb          # per element, LEFT term first, products rounded separately
                                    # (mul.rn.ftz.f32x2 twice, then add.rn.ftz.f32x2)
  L  = __fadd_rn(__fmul_rn(a.L, sa), __fmul_rn(b.L, sb))

root(s):
  L' = (s.L == 0 || s.L != s.L) ? 1 : s.L
  inv = __fdiv_rn(1, L')                       # one IEEE round-to-nearest divide per row
  out = cvt.rn.bf16x2.f32(s.O * inv)           # THE one rounding
  lse = (s.L == 0 || s.L != s.L) ? -inf
        : (__fmul_rn(s.M, scale_log2) + log2f(L')) * ln2
```

This is the merge of the SM90 absorbed-MLA segment tree (`ops/attention/mla_sm90`), at `D = 128`
instead of 512 and with `M` kept in the raw frame. `n = 1` takes `root(leaf)` and is bit-identical whether the leaf
writes the output itself or publishes and the combine folds it -- that equality is a gated relation.

**A flat chain across segments is a different program and is refused.** So is a global-max merge (a
single max over all leaves, then a flat `fmaf` chain over the leaf numerators), which is not this
tree.

### 1.6 Scalar and tensor-core profile

One profile for every kernel of the op, `isoexec/ops/build_cache.sm100_cuda_flags`:
`-gencode=arch=compute_100a,code=sm_100a --ftz=false --fmad=false`, C++20. FMA contraction is off
because it can differ between template instantiations (the leaf at one wave and at two, the combine),
which breaks cross-kernel bit equality inside one architecture. The packed instructions state `.ftz`
explicitly; that is part of the recipe, not a compiler choice. Every kernel is BF16 x BF16 -> FP32
`tcgen05.mma.kind::f16` (`cta_group::1` at head dim 128, `cta_group::2` for the head-dim-256 CTA
pair, `cta_group::1` at M128 N16 for the head-dim-256 decode leaf); the QK descriptor is `0x08200490`
and the PV descriptor `0x08210490`, FA4's own.

## 2. Execution

Everything here is free: the relations battery forces the alternatives and requires identical bits.

| | |
|---|---|
| head dim | 128 in `leaf_d128.cu`, one CTA per work item, one or two 128-row query waves. 256 in `leaf_d256.cu`, one **CTA pair** per item: the pair splits the MMA's N axis and nothing else (64 of a tile's 128 keys per CTA for QK, 128 of the 256 output features per CTA for PV), so every output element's fp32 accumulation order is the D128 leaf's; the only arithmetic difference is the QK contraction's extent, which the head dim forces. `leaf_d256_fold.cu` is the same source built with `GQA_D256_FOLD`, for plans that fold a run of segments in one pair. |
| query tile | 128 packed rows = `128 / ratio` tokens x `ratio` query heads, head fastest. A row is one TMEM lane owned by one thread, so there is no cross-row reduction anywhere in the softmax, and a row's bits cannot see its tile neighbours -- which is why the head split across tensor-parallel ranks (which changes the ratio and the packing) changes no bit. |
| work item | `(request, query tile, KV head, [k0, k1))`, where `[k0, k1)` is a node of the canonical tree of every row it publishes. A whole-tree item writes the output and lse itself; any other publishes `(O, L, M)` and the combine folds the rest. The fold run (a power of two up to 16) and the wave count come from one host-side cost model (`host.cu::choose_plan`), which may read the SM count; SEG and the tree may not. |
| plan | a device-side kernel over `cu_seqlens_q` / `seqused_k`, so the launch is CUDA-graph capturable and nothing reads a device tensor on the host. Capacity is sized on the host from `T`, `B`, `max_seqlen_q`, `max_seqlen_k` and SEG: `max_seqlen_k` must bound every `seqused_k` (vLLM captures decode with `max_model_len`, which does), and overstating it changes no bit. `ISOEXEC_GQA_SM100_CHECK=1` makes the batteries read the plan's overflow flag on every call. |
| combine | one warp per output row, evaluating the tree with a streaming binary counter that merges whenever the top two stack entries have equal size -- exactly the canonical tree's merges. Not launched when no row can have more than one segment. |
| KV addressing | TMA over a `[tokens, heads, D]` view with explicit token and head strides: a packed `[T, h_kv, D]` cache (`cu_seqlens_k`) or a paged `[blocks, page, h_kv, D]` cache (`block_table`) whose blocks are `page` token strides apart, including vLLM's `LBNHC` layout where K and V interleave in one page (`[page, h_kv, 2D]`). `page % 128 == 0` so a 128-key tile never crosses a page. Page ids, page order and strides never reach the arithmetic. |
| decode leaf | `decode_d256.cu`, taken by `host.cu::forward` for a head-dim-256 call with `max_seqlen_q * ratio <= 16` (decode, and speculative decode up to `16 / ratio` tokens per request) and no granularity or run override; `ops.LEAF_GENERAL` / `ops.LEAF_DECODE` force either leaf. The rule reads the call's shape, never the batch or the lengths. It is the leaf program executed **swap-AB on one CTA**: the 128-key tile is the MMA's M operand and the item's at most 16 packed query rows its N (`tcgen05.mma.cta_group::1.kind::f16`, M128 N16 K16). Every score is still the contraction of one query row with one key in sixteen K16 steps in ascending feature order, and every output element still accumulates the tile's keys in eight K16 steps in ascending key order into the rescaled accumulator; which operand is "A" does not enter a product or a step's sum. One warp owns a row: the elementwise steps run per lane, the two row-sum chains run in the recipe's column order (shuffles), and the row max is a butterfly that falls back to the recipe's four seeded chains when the result is +-0 or NaN. K and V stream through a ring of 16 KiB TMA stages; one CTA walks the segments `y, y + split, ...` of one (request, KV head) and publishes each as its own leaf, and the decode combine folds rows with more than one segment by the canonical tree. There is no plan kernel. |
| profile | `build_cache.sm100_cuda_flags` (section 1.6), built by `_extension.py` into the torch extensions cache. |

## 3. Integration

| | |
|---|---|
| dispatch | `ops/_dispatch.py`: the `gqa` seam selects `attention.gqa_sm100` on `sm100`. `attention.core.gqa` resolves to `integration.trainer_core` (`sm100_gqa_segtree_v2_varlen`); the vLLM adapter binds `integration.paged_core` (`sm100_gqa_segtree_v2_paged`). |
| SEG | `spec.SEG_CONTEXT = 32768` -> `SEG = 2048` at both head dims (section 1.3), a constant of the recipe. |
| engine cache | the CUSTOM vLLM backend (`ops/attention/varlen_backend.py`) reads `integration.PAGE_MULTIPLE` (kernel block sizes `MultipleOf(128)`, preferred block size rounded up to 128, which vLLM's hybrid page alignment then keeps) and `integration.KV_CACHE_LAYOUTS` (`LBNHC`, vLLM's default and the block-compact layout hybrid models need), so the engine allocates pages this kernel reads in place. The kernel writes the engine's output buffer in place. |
| trainer install | `integration.install()` builds the extension, checks the CuTe DSL install (section 4) and returns the receipt (`recipe`, `seg_tokens`, the backward per head dim, `fa4_commit`). |

## 3a. Context parallelism (trainer, head dim 256)

`context_parallel.py`, bound by `Seams.cp_attention` as `collectives.cp_gather`
`kv_all_gather_fwd_owner_tiles_bwd`. A CP rank holds the zigzag chunks `r` and `2*cp-1-r` of every
sequence. **Forward**: K|V are all-gathered into the global packed order and each local chunk is one
request of the packed forward (key start at the sequence, `seqused_k` at the chunk's end), so every
local row is the CP=1 row by section 1.1. **Backward**: Q|dO and the preprocess statistics are
all-gathered as bytes, and FA4's head-dim-256 backward runs over the global packed problem with tile
masks: only the dQ tiles (256 rows) holding a local query and the dK/dV tiles (128 keys) holding a local
key are computed, each by the CTA pair that owns it at CP=1 (section 4: no cross-tile reduction). No
value is summed across ranks; each rank computes about half the tiles. Evidence:
`ops/collectives/tests/context_parallel_seams_gpu.py` (CP=2 rows and gradients == CP=1, bitwise). The other
components' CP forms, and CP with expert parallelism, are in
[Trainer context parallelism](../../../../docs/context-parallelism.md).

## 4. Backward

FlashAttention-4's SM100 backward, vendored at commit `0bbb25a3a5ad3c58c029b3d287d6c9af56a5cad5` in
`isoexec/ops/_vendor/flash_attn_cute` (PROVENANCE.md there: sources, licenses, the per-file manifest
and every modification). It is not imported from an installed `flash_attn`.

* **Inputs.** It runs on this forward's bf16 output and fp32 LSE. The forward's lse is
  `(M * scale_log2 + log2 L) * ln 2` (section 1.5): the natural log of `sum exp(scale * s)`, which is
  FA4's own convention; FA4 takes it head-major (`[Hq, T]`, so `ops.varlen_bwd` passes
  `lse.t().contiguous()`) and its preprocess forms `lse * log2(e)`. `dO * O` row sums are FA4's.
* **Head dim 128**: `FlashAttentionBackwardSm100` with `deterministic=True` and FA4's 2-CTA choice
  pinned on (`disable_2cta=False`, not the `FA_DISABLE_2CTA` environment variable). dQ, and dK/dV
  when query heads share a KV head, accumulate in fp32 through `cp.reduce.async.bulk add.f32` in a
  semaphore-ordered sequence, then FA4's postprocess converts to bf16.
* **Head dim 256**: FA4's dedicated two-CTA kernels (`sm100_hd256_2cta_fmha_backward*`). FA4 has no
  `deterministic` flag for them because they need none: each dQ and dK/dV tile is owned by one CTA of
  a two-CTA cluster, which loops over the KV tiles (dQ) or the query tiles and the KV head's query
  heads (dK/dV) and stores once; there is no global float reduction.
* **Determinism.** Both are bitwise repeatable per layout (`tests/test_ops_gpu.py`, ten runs with the
  allocator dirtied between them). Neither is invariant to the tensor-parallel head split, and the
  contract does not ask it to be.
* **Runtime.** `nvidia-cutlass-dsl` 4.8.0 compiles it at the first backward of each shape class, and
  `apache-tvm-ffi` carries the calls. The vendored copy carries upstream's replacements for the APIs
  4.8 removed (`_vendor/flash_attn_cute/PROVENANCE.md`); its bits at 4.8.0 equal those at 4.5.2,
  where it was first qualified. At 4.5.2 the two CuTe DSL compiler wheels,
  `nvidia-cutlass-dsl-libs-base` and `nvidia-cutlass-dsl-libs-cu13`, write 99 of the same files with
  different contents; an environment with both installed holds whichever was written last, and FA4's
  backward cannot compile against that mixture. `ops.require_consistent_cute_dsl()` (run by `install()` and before the first
  backward) refuses such an install by checking the installed files against the wheel's own record.

## 5. Evidence

| claim | test |
|---|---|
| Declarations, twin class, dispatch row, resolution on SM100, SEG table and recipe SEG, canonical tree and `node_size_at` against an independent construction, lazy import, the CuTe DSL refusal | `tests/test_contract_cpu.py` |
| **The seam**: trainer rows == paged prefill == chunked prefill == decode, on the release models' layers (Qwen3.5-35B-A3B, -9B, -0.8B at head dim 256; Qwen1.5-MoE-A2.7B at 128), page sizes 128 and 1152, dense pages and vLLM's interleaved `LBNHC` pages | `tests/test_seam_gpu.py` |
| TP 2/4/8 head splits give each rank exactly the TP=1 bits of its heads (trainer and decode) | `tests/test_seam_gpu.py` |
| A decode row is the same alone, in a batch of 64 and of 512; long-context decode (10240, 40000) equals the trainer row; a CUDA graph captured with `max_seqlen_k = max_model_len` replays the eager bits, and replays new lengths correctly | `tests/test_seam_gpu.py` |
| Float64 accuracy on four input populations, packed == paged, page layout, every execution plan, batch composition, chunked prefill, later-turn prompts, 50 repeats, graph replay, tails, refusals, head dim 256 folding | `tests/test_relations_gpu.py` (12 relations, 819 checks) |
| The decode leaf == the general leaf (out and lse bits) across head ratios 1-16, 1..16/ratio query tokens, pages 128 and 1152, permuted and NaN-poisoned pages, LBNHC, packed KV, input populations, segment splits, batch composition; decode rows == trainer rows; `paged_core` takes it and writes in place; graph replay | `tests/test_decode_gpu.py` |
| Backward accuracy against float32 autograd, bitwise repeatability of out/dQ/dK/dV, the LSE convention, refusals, the install receipt | `tests/test_ops_gpu.py` |
| The vendored FA4/QuACK files are upstream plus exactly the declared edits | `isoexec/ops/tests/test_fa4_vendor_provenance_cpu.py` |

## 6. Performance

One B200, 2026-11-07, CUDA kernel time per call (CUPTI, summed over every kernel the call launches,
mean of 20 calls after 5 warm-ups; ours is the better of two passes taken before and after FA4's).
The reference is FA4 at the vendored commit, in production configuration: split-KV heuristic
(`num_splits=0`) at head dim 128, `num_splits=1` at 256 where FA4 has no split-KV, paged decode with
128-key pages. Decode is one query per sequence at a uniform context `k`; prefill is `T` tokens as
four causal sequences. The GPU was shared with small processes of other users (about 5 GB, mostly
idle) during the run, so read the numbers as indicative rather than as a gate. The head-dim-256
decode rows were measured on the general leaf, before the decode leaf (section 2) took them.

| geometry (Hq/Hkv, D) | shape | gqa_sm100 (us) | FA4 (us) | ratio |
|---|---|---:|---:|---:|
| 35B-A3B TP4 (4/1, 256) | decode B256 k2048 | 148.2 | 397.7 | 0.37 |
| 35B-A3B TP4 (4/1, 256) | decode B256 k8192 | 517.3 | 1369.5 | 0.38 |
| 35B-A3B TP4 (4/1, 256) | decode B512 k2048 | 259.6 | 848.8 | 0.31 |
| 35B-A3B TP4 (4/1, 256) | decode B512 k8192 | 1021.4 | 2977.4 | 0.34 |
| 35B-A3B TP4 (4/1, 256) | decode B1024 k2048 | 508.9 | 1910.2 | 0.27 |
| 35B-A3B TP4 (4/1, 256) | decode B1024 k8192 | 2033.0 | 6561.3 | 0.31 |
| 35B-A3B TP4 (4/1, 256) | decode B2048 k2048 | 1015.6 | 5356.7 | 0.19 |
| 35B-A3B TP4 (4/1, 256) | decode B2048 k8192 | 4335.2 | 14189.1 | 0.31 |
| 35B-A3B TP4 (4/1, 256) | prefill T4096 x4 | 25.8 | 69.3 | 0.37 |
| 35B-A3B TP4 (4/1, 256) | prefill T8192 x4 | 55.6 | 138.7 | 0.40 |
| 35B-A3B TP4 (4/1, 256) | prefill T16384 x4 | 181.8 | 300.8 | 0.60 |
| 35B-A3B TP8 (2/1, 256) | decode B256 k2048 | 148.5 | 208.9 | 0.71 |
| 35B-A3B TP8 (2/1, 256) | decode B256 k8192 | 512.5 | 720.3 | 0.71 |
| 35B-A3B TP8 (2/1, 256) | decode B512 k2048 | 257.9 | 446.9 | 0.58 |
| 35B-A3B TP8 (2/1, 256) | decode B512 k8192 | 1016.9 | 1540.7 | 0.66 |
| 35B-A3B TP8 (2/1, 256) | decode B1024 k2048 | 508.6 | 994.2 | 0.51 |
| 35B-A3B TP8 (2/1, 256) | decode B1024 k8192 | 2039.4 | 3299.7 | 0.62 |
| 35B-A3B TP8 (2/1, 256) | decode B2048 k2048 | 1014.8 | 2511.4 | 0.40 |
| 35B-A3B TP8 (2/1, 256) | decode B2048 k8192 | 4330.2 | 7319.2 | 0.59 |
| 35B-A3B TP8 (2/1, 256) | prefill T4096 x4 | 25.2 | 56.1 | 0.45 |
| 35B-A3B TP8 (2/1, 256) | prefill T8192 x4 | 37.3 | 78.2 | 0.48 |
| 35B-A3B TP8 (2/1, 256) | prefill T16384 x4 | 108.5 | 168.8 | 0.64 |
| 9B TP2 (8/2, 256) | decode B256 k2048 | 262.3 | 786.7 | 0.33 |
| 9B TP2 (8/2, 256) | decode B256 k8192 | 1034.5 | 2850.2 | 0.36 |
| 9B TP2 (8/2, 256) | decode B512 k2048 | 515.8 | 1700.1 | 0.30 |
| 9B TP2 (8/2, 256) | decode B512 k8192 | 2061.0 | 7419.3 | 0.28 |
| 9B TP2 (8/2, 256) | decode B1024 k2048 | 1031.9 | 4003.5 | 0.26 |
| 9B TP2 (8/2, 256) | decode B1024 k8192 | 4339.3 | 13938.7 | 0.31 |
| 9B TP2 (8/2, 256) | decode B2048 k2048 | 2057.7 | 9970.8 | 0.21 |
| 9B TP2 (8/2, 256) | decode B2048 k8192 | 9161.7 | 29500.3 | 0.31 |
| 9B TP2 (8/2, 256) | prefill T4096 x4 | 40.4 | 124.9 | 0.32 |
| 9B TP2 (8/2, 256) | prefill T8192 x4 | 85.9 | 256.8 | 0.33 |
| 9B TP2 (8/2, 256) | prefill T16384 x4 | 352.3 | 551.5 | 0.64 |
| Qwen1.5-MoE TP4 (4/4, 128) | decode B256 k2048 | 186.3 | 194.2 | 0.96 |
| Qwen1.5-MoE TP4 (4/4, 128) | decode B256 k8192 | 674.2 | 660.7 | 1.02 |
| Qwen1.5-MoE TP4 (4/4, 128) | decode B512 k2048 | 363.8 | 404.3 | 0.90 |
| Qwen1.5-MoE TP4 (4/4, 128) | decode B512 k8192 | 1361.6 | 1370.8 | 0.99 |
| Qwen1.5-MoE TP4 (4/4, 128) | decode B1024 k2048 | 708.2 | 898.7 | 0.79 |
| Qwen1.5-MoE TP4 (4/4, 128) | decode B1024 k8192 | 2763.3 | 2898.2 | 0.95 |
| Qwen1.5-MoE TP4 (4/4, 128) | decode B2048 k2048 | 1412.1 | 2195.4 | 0.64 |
| Qwen1.5-MoE TP4 (4/4, 128) | decode B2048 k8192 | 6078.7 | 6532.2 | 0.93 |
| Qwen1.5-MoE TP4 (4/4, 128) | prefill T4096 x4 | 19.4 | 26.6 | 0.73 |
| Qwen1.5-MoE TP4 (4/4, 128) | prefill T8192 x4 | 37.9 | 31.0 | 1.22 |
| Qwen1.5-MoE TP4 (4/4, 128) | prefill T16384 x4 | 108.7 | 77.8 | 1.40 |

| geometry | shape | FA4 bwd deterministic (us) | FA4 bwd non-deterministic (us) | ratio |
|---|---|---:|---:|---:|
| 35B-A3B TP4 | prefill T4096 x4 | 117.3 | same kernel | 1.00 |
| 35B-A3B TP4 | prefill T8192 x4 | 239.4 | same kernel | 1.00 |
| 35B-A3B TP4 | prefill T16384 x4 | 629.5 | same kernel | 1.00 |
| 35B-A3B TP8 | prefill T4096 x4 | 74.4 | same kernel | 1.00 |
| 35B-A3B TP8 | prefill T8192 x4 | 142.2 | same kernel | 1.00 |
| 35B-A3B TP8 | prefill T16384 x4 | 360.3 | same kernel | 1.00 |
| 9B TP2 | prefill T4096 x4 | 160.6 | same kernel | 1.00 |
| 9B TP2 | prefill T8192 x4 | 399.6 | same kernel | 1.00 |
| 9B TP2 | prefill T16384 x4 | 1065.0 | same kernel | 1.00 |
| Qwen1.5-MoE TP4 | prefill T4096 x4 | 44.3 | 38.5 | 1.15 |
| Qwen1.5-MoE TP4 | prefill T8192 x4 | 98.6 | 88.3 | 1.12 |
| Qwen1.5-MoE TP4 | prefill T16384 x4 | 238.6 | 220.8 | 1.08 |

At head dim 256 the backward is FA4's one dedicated kernel (deterministic by construction). At head
dim 128 FA4's `deterministic=True` costs 8 to 15 % against its non-deterministic mode.

## 7. Files

| file | |
|---|---|
| `spec.py` | the declarations: recipe, version, SEG context, twin class, hazards |
| `ops.py` | the forward entry (`gqa_sm100_fwd`), the trainer/engine wrappers, FA4's backward, the autograd core, the CuTe DSL check |
| `integration.py` | the op-table face: `trainer_core`, `paged_core`, `install`, the engine cache contract |
| `reference.py` | the float64 oracle and the tree's structure in host Python |
| `_extension.py` | the build (`build_cache.build_sm100_extension`) |
| `csrc/host.cu` | argument checks, the plan cost model, capacities, TMA maps, launches |
| `csrc/plan.cu` | the device-side plan kernels |
| `csrc/leaf_d128.cu`, `csrc/leaf_d256.cu`, `csrc/leaf_d256_fold.cu` | the leaves |
| `csrc/combine.cu` | the canonical-tree combine |
| `csrc/decode_d256.cu` | the head-dim-256 decode leaf (swap-AB, one CTA per (request, KV head)) and its combine |
| `csrc/numerics.cuh`, `csrc/tmem*.cuh`, `csrc/plan_skeleton.cuh`, `csrc/params.h`, `csrc/launch.h`, `csrc/bindings.cpp` | the recipe's arithmetic, TMEM access, the fold's slot plan, shared types, the torch binding |

## Attribution

The forward's execution is derived from FlashAttention-4 (Dao-AILab/flash-attention, `flash_attn/cute`,
BSD-3-Clause) through TileFoundry's CUDA port of it, and the derived sources retain the notice. Two
parts are not derived from it: the segment tree with its backing-slot fold (a pending node of the
canonical tree held in shared or global memory while a later segment accumulates into the TMEM tile it
vacated), and the plan cost model. FA4 chooses its split count and its exp2 emulation rate from the
shape, which is exactly what this op may not do, so FA4 is the specification of the inner loop's
arithmetic and a performance reference, not a numerical target. The backward is FA4's own, vendored
with its provenance in `isoexec/ops/_vendor/flash_attn_cute/PROVENANCE.md`.
