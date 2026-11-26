# Vendored: FlashMLA (DeepSeek), MIT

Source: https://github.com/deepseek-ai/FlashMLA at commit `395cb3a` (2026-10-26, upstream HEAD on 2026-11-17; first
vendored at `2e5429f`), license `LICENSE-FlashMLA.txt`. Every vendored kernel-side file is identical at the two commits
except `head64/kernel.cuh`, whose only change is #241 (change 6 below); vllm-project/FlashMLA `b18ed74` (fork HEAD on
2026-11-17) carries the same head64 kernel. Re-vendor: `git diff <old>..<new> -- csrc/cuda_kernels/{defines.h,
kv_cache_format.h,sm100/{common_subroutine.h,kv_cache_utils.cuh,helpers.h,dequant_utils.cuh,decode/sparse/head64}}
csrc/params.h csrc/3rdparty/kerutils/include` and apply it under the changes listed here.

| here | upstream | change |
|---|---|---|
| `kerutils/include/**` | `csrc/3rdparty/kerutils/include/**` | verbatim, without `kerutils/supplemental/` (upstream's torch-tensor helpers for its API layer; IsoExec's host code is its own) |
| `flashmla/cuda_kernels/defines.h` | `csrc/cuda_kernels/` | verbatim |
| `flashmla/cuda_kernels/kv_cache_format.h` | `csrc/cuda_kernels/` | the FP4 format (`V41_FP4`, 288 B) removed; `IS_FP4` kept as constant `false` |
| `flashmla/cuda_kernels/sm100/{common_subroutine.h,kv_cache_utils.cuh}` | `csrc/cuda_kernels/sm100/` | verbatim |
| `flashmla/cuda_kernels/sm100/{helpers.h,dequant_utils.cuh}` | `csrc/cuda_kernels/sm100/` | FP4 branches and the e2m1 / fp4-scale conversions removed (the V4.1 FP8 record is the only cache format) |
| `flashmla/params.h` | `csrc/params.h` | stripped to `bf16`, `stream_t`, `ModelType` (no scheduler metadata, no param structs, no Ascend); `ModelType::V41_FP4` removed |
| `leaf.cuh` | `csrc/cuda_kernels/sm100/decode/sparse/head64/{config.h,kernel.cuh}` | rewritten, see below |
| `dualrail.cu` | `.../head64/kernel.cuh` `run()` | rewritten host side (+ the split plan's two launches, PDL) |
| `persist.cuh` | `csrc/cuda_kernels/sm100/prefill/sparse/fwd/head64/{config.h,phase1.cuh}`, `csrc/cuda_kernels/utils.h` (`RingBufferState`) | the persistent execution plan, see "Changes in persist.cuh" |
| `split.cuh` | `leaf.cuh` (above) | the split execution plan: the leaf's QK path per tile and its softmax / PV / epilogue per column block (sparse_sm100 README "Dual-rail split plan") |

Changes in `leaf.cuh` (each one a determinism / contract requirement of IsoExec):
1. Split-KV removed entirely: no `DecodingSchedMeta`, no `num_splits`, no partial `o_accum` / `lse_accum`, no combine;
   grid = one CTA per query row (`blockIdx.x`), the row's whole list. FlashMLA's split path depends on the batch
   (its scheduler balances requests over SMs); the no-split path does not, and is the only one kept.
2. Lazy-rescale decision per ROW (`cur_max - m > 6`), not `__any_sync` over a warp (32 heads): a head's bits do not
   depend on the CTA's other heads, so ranks with < 64 heads equal the 64-head run per head. The warp-collective
   TMEM correction runs when any row of the warp rescales; other rows multiply by exactly 1.0f.
3. The extra (compressed-KV) cache is a second V4.1 FP8 cache (`Leaf<false, EXTRA = true>`, its own tensor map and
   list); upstream's FP4 extra format is not served (port: the V4.1 cache program stores both pools as FP8 528 B).
3b. bf16-kv mode (new): the dequant warpgroup issues the tile's 128 TMA gather4s into the SW128 tile, three tile
   buffers, the UMMA warp waits the gather transactions.
4. Flat token coordinates (`idx` = cache row; paging resolved by the caller); IsoExec output / lse strides; lse
   sink-inclusive with an empty row's lse = sink (sparse_sm100 README.md convention); no PDL trigger.
5. Built under the sparse_sm100 numerical profile (`--fmad=false --ftz=true --prec-div=false`, no `--use_fast_math`);
   measured bit-identical to the upstream fast-math build of the unmodified no-split kernel.
6. Upstream fix ported (FlashMLA #241, commit `395cb3a`, 2026-10-26, "Fix a race on rowwise_max_buf before the li
   exchange"): one `NamedBarrier::arrive_and_wait(128, wg0_sync)` before the softmax warpgroup writes `li` into
   `rowwise_max_buf`, in `leaf.cuh` and in the split plan's PV kernel (`split.cuh`). Without it the partner thread
   (`idx ^ 64`, another warp) could read `li` as the last tile's row max (run-to-run nondeterminism). Synchronisation
   only: no arithmetic changes. Stress relation: `tests/test_dualrail_repeat_gpu.py`. This is the only head64 kernel
   change between `2e5429f` and upstream HEAD `395cb3a` (= vllm-project/FlashMLA `b18ed74` for this kernel).

Changes in `persist.cuh` (upstream's sparse prefill head64 kernel = what `flash_mla_sparse_fwd` runs at h_q <= 64;
kept: the CLC-persistent schedule, warp roles, 3-slot K/V ring shared with Q and the O staging, the two-half QK, the
per-tile subroutines, the epilogue's store path):
P1. Lazy-rescale decision per ROW instead of `__any_sync` over a warp (as change 2): a head's bits never depend on the
    CTA's other heads.
P2. K/V source: bf16 rows through upstream's TMA gather4 producer, or the V4.1 FP8 528 B records of one or two caches
    (window | compressed, two lists in list order): warp 11 (idle upstream) runs ahead over the ring and TMA-gathers a
    tile's raw records (the one-CTA leaf's raw-record maps, coordinates and 576 B rows) into the top 36 KB of the tile's
    own slot once the slot is free; the producer warpgroup reads them into registers, syncs, and writes the swizzled
    bf16 tile over the slot with `fp8x2_to_bf16x2_with_scale` (`KVBlockDequantizer`'s mapping). No separate raw buffer,
    so the 3-slot ring fits and the load latency is hidden by the ring depth. Upstream's prefill reads only bf16 (vLLM
    gathers and dequantizes into a workspace first).
P3. The leaf's per-row tile structure and validity rule (window tiles `max(ceil(len/64), 1)`, then the extra tiles; valid
    iff index >= 0 and position < length; an invalid slot's row is zero; no TMA skipping of all-invalid tiles), so a
    row's tile sequence is the one-CTA leaf's.
P4. The leaf's epilogue (sink-inclusive lse, empty row out 0 / lse = sink, O always read from TMEM, no max_logits),
    IsoExec strides, flat token rows, the last output store drained before exit.
P5. PDL wait before any global access.
P6. The leaf's fused RoPE (IsoExec `fused_rope.cuh`, fp8 launches): warp 10 (idle upstream) rotates the Q tile's tail in
    its ring slot between the Q TMA and the UTCCP (new barrier `bar_q_roped`, warp 10 joins the CLC worker count);
    the epilogue applies the inverse rotation to each bf16 output chunk before the store.
Per row the result is bit-identical to `leaf.cuh` (relation `tests/test_dualrail_persist_gpu.py`, also the prefill
row-invariance relation: chunking, order, one-row decode launches). Built under the same profile (change 5).

Audit: no global / floating-point atomics (SASS gate `tests/test_dualrail_sass_cpu.py`; integer shared
atomics only in the TMEM allocator); approximate intrinsics in the numerical path: `exp2f`, `__fdividef`, `logf`
(declared in the recipe; engine decode and trainer replay run this one kernel).

Host binding (`dualrail.cu`, 2026-11-20, defect 13b): the FP8 caches are accepted at any whole number of 528 B records,
not only whole 64-token pages. `quant_kv_map` builds the same 2-D gather map `{token words, rows}` (box height 1) with
`rows = tokens` instead of `pages * 64`; for whole pages the call and the map are unchanged. Every slot is a single-row
gather and an invalid slot's coordinate is -1 (out of bounds, zero fill) either way, so the bytes at every valid
coordinate and the zero fill are unchanged; no kernel source (`leaf.cuh`, `split.cuh`, `persist.cuh`) changes.
Needed by an engine pipeline stage whose vLLM allocation ends within a page after a layer's last record
(`models/deepseek_v4_1_flash/engine.record_rows`).
