// SPDX-License-Identifier: BSD-3-Clause
// Execution geometry derived from FlashAttention-4 (flash_attn/cute/flash_fwd_sm100.py) and from
// TileFoundry's CUDA port of it.
// Copyright (c) 2025, Tri Dao, Siyu Wang, Shengbin Di, Yuxi Chi, Johnsonms, Linfeng Zheng,
// Haoyan Huang, Lanbo Li, Yun Zhong, Man Yuan, Minmin Sun, Yong Li, Wei Lin, Jay Shah,
// Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani. Distributed under BSD-3-Clause.
//
// The NUMERICAL PROGRAM (SEG segments anchored at key 0, the canonical adjacent-pair tree, the merge
// operator, the one rounding at the root, the KV tile width) is IsoExec's and is stated in
// ../README.md. Everything else here -- tiles, warp roles, pipeline depth, TMEM placement, the work
// list, the grid -- is execution and may not move a bit.
#pragma once

#include <cstdint>

namespace gqa_sm100 {

// --- identity ------------------------------------------------------------------------------------

//: The KV tile width. Every tile boundary inside a segment is a max/rescale point, so this is
//: numerical, unlike a GEMM's K staging (../README.md section 1.2).
static constexpr int KEY_TILE = 128;

//: The leaf's published running-max sentinel for a segment that saw no unmasked key. Finite, so the
//: merge never evaluates exp2 on an infinity, and far below any real score.
static constexpr float M_EMPTY = -3.0e38f;

//: FA4's LAZY RESCALE threshold, in log2 units. Part of the recipe (`sm100_gqa_segtree_v2`): while a
//: KV tile's row max exceeds the row's running max by at most this many log2 units, the running max
//: is KEPT and the output accumulator is not rescaled, so a probability may exceed 1 by up to
//: 2^THRESHOLD. The decision is a pure function of the row's own running max over its fixed 128-key
//: tiles in logical order inside ONE segment, reset at every segment start -- therefore of (logical
//: position, KV bits, deployment constants) and of nothing else: not of batch, chunk, page, grid,
//: worker count or any other row. That is what makes it admissible under the numerical contract,
//: exactly as the exact-max program was; see ../README.md section 1.4.
//:
//: 8.0 is FA4's own value. At 8 the largest probability of a tile is
//: at most 2^8 = 256, which bf16 holds exactly at the same relative precision as 1.0, and the leaf's
//: published (O, L, M) is still exactly consistent (O and L are both taken relative to the M that is
//: published), so the merge and the root are unchanged.
static constexpr float RESCALE_THRESHOLD = 8.0f;

// --- execution -----------------------------------------------------------------------------------

//: KV tile ordinals whose cache slot the leaf precomputes in shared memory. An item runs at most
//: FOLD_MAX_RUN segments of SEG/128 tiles each; at the table's SEG = 512 a whole-tree run of 16 is 64
//: tiles, and 256 covers every (SEG, run) the fill heuristic reaches on the target shapes; a longer
//: item falls back to the address function per tile.
static constexpr int MAX_KEY_SLOTS = 256;

//: Deepest canonical tree the combine evaluates from published nodes (2^16 segments).
static constexpr int MAX_COMBINE_DEPTH = 17;

//: Worker in-place fold: the largest aligned run of segments one leaf worker may fold in place.
//:
//: The streaming binary counter over a run of `R` segments holds at most `floor(log2(R)) + 1` live
//: fp32 output tiles, of which one is the segment being accumulated. TMEM holds `tmem_slots<WAVES>()`
//: of them -- two at one wave (the other 256 of its 512 columns are the two score slots that let
//: `QK(g+1)` be issued before `PV(g)`) and one at two waves (both output tiles are the two waves'
//: accumulators). Every further pending node lives in a BACKING slot: a 64 KiB slot in shared memory
//: at one wave, and otherwise an L2-resident slot in global memory, one per (resident CTA, wave,
//: backing index). `fold_backing_slots` below is the count, and `build_segment_plan` spills the
//: deepest pending node that is still in TMEM into a free one.
//:
//: 16 is the cap, and it is the `n_max = 16` of the SEG table (README.md section 1.3): at a run of 16
//: every declared geometry folds its WHOLE tree in one worker and publishes nothing.
//:
//: Where a node lives is EXECUTION. `merge_from_node` performs the same `merge_o_pair` over the same
//: elements with the same left operand first as `tmem_merge`; only the address the left operand is
//: read from differs.
static constexpr int FOLD_MAX_RUN = 16;

//: Deepest the streaming binary counter's per-row stack goes at `FOLD_MAX_RUN`: `log2(16) + 1`.
static constexpr int FOLD_STACK = 5;

//: Most merges the counter performs after ONE segment, at `FOLD_MAX_RUN`: the last segment of a run
//: of sixteen collapses 1+1 -> 2, 2+2 -> 4, 4+4 -> 8, 8+8 -> 16.
static constexpr int FOLD_MERGES = 4;

//: Most backing slots any (tmem_slots, run) pair needs: `FOLD_STACK - 1` at one TMEM output tile.
static constexpr int FOLD_BACKING = FOLD_STACK - 1;

//: Backing slots a run of `run` needs beside `tmem` TMEM output tiles. The counter's stack is at most
//: `floor(log2(run)) + 1` deep, and `tmem` of those entries live in TMEM.
__host__ __device__ constexpr int fold_backing_slots(int tmem, int run) {
    int depth = 1;
    while ((1 << depth) <= run) ++depth;
    const int need = depth - tmem;
    return need < 0 ? 0 : need;
}

// One work item of the leaf program.
struct __align__(16) Item {
    int req;        // request index
    int tok0;       // first query token of the tile, absolute in the packed q tensor
    int kv_head;
    int k0;         // first segment of the run
    int k1;         // one past the last segment of the run
    //: The request's two numbers ride HERE rather than being looked up in `cu_seqlens_q` and
    //: `seqused_k`, exactly as `CombineTile` already carried them, and they cost nothing: they replace a
    //: field the leaf never read (`nseg`; only `sp.nsegs`, rebuilt from `k0`/`k1`, is ever used) and the
    //: explicit `pad`. That takes one DEPENDENT GLOBAL ROUND TRIP off the front of every CTA -- the
    //: entry chain was `counts` -> `items[blockIdx.x]` -> `cu_seqlens_q[req+1]`/`seqused_k[req]` ->
    //: `block_table[...]` -> the first TMA, and a short-item decode row pays that chain once per item.
    int q_end;      // cu_seqlens_q[req + 1]
    int used;       // seqused_k[req]
    int slot_base;  // partial slot of (packed row 0, segment k0), or -1: this item writes the output
    //: The item's FIRST KV tile's key slot in the flattened cache (logical key
    //: `k0 * seg`), resolved by the plan kernel, so the leaf's first K copy does not wait on a dependent
    //: `block_table` load. It replaces the unused `pad`, so the Item does not grow.
    int ks0;
};

// Combine entry: one packed query tile whose rows were published as single segments.
//: `q_end` and `used` are the request's own two numbers, carried here rather than looked up: the
//: combine is a chain of dependent global loads (`counts` -> this entry -> the first node), and
//: reading `cu_seqlens_q[req + 1]` and `seqused_k[req]` out of the entry instead of out of two more
//: device tensors takes one whole round trip off the front of every block. They are the same two
//: integers the plan already had in registers when it wrote the entry.
struct __align__(16) CombineTile {
    int req;
    int tok0;
    int kv_head;
    int slot_base;  // partial slot of (packed row 0, node 0)
    int run;        // the aligned run the items folded, so node_size_at names the published nodes
    int q_end;      // cu_seqlens_q[req + 1]
    int used;       // seqused_k[req]
    int pad;
};

// Size of the node a worker publishes at leaf k when items fold aligned runs of `run`: full runs
// while they fit, then the remainder as descending powers of two. Every such node is a node of the
// canonical adjacent-pair tree over n leaves (mla_sm90/csrc/fwd/params.h has the same function).
__host__ __device__ __forceinline__ int node_size_at(int k, int n, int run) {
    if (k + run <= n) return run;
    int s = 1;
    while (2 * s <= n - k) s *= 2;
    return s;
}

__host__ __device__ __forceinline__ int ceil_div_i(int a, int b) { return (a + b - 1) / b; }

//: Number of fixed segments of a row whose causal border is `border` keys.
__host__ __device__ __forceinline__ int segment_count(int border, int seg) {
    return border <= 0 ? 0 : ceil_div_i(border, seg);
}

enum PlanCount : int {
    NUM_ITEMS = 0,
    NUM_COMBINE_TILES = 1,
    NUM_SLOTS = 2,
    OVERFLOW_FLAG = 3,
    MAX_N = 4,
    NUM_COUNTS = 8,
};

// Granularity (execution only). The relations force each of these and require identical bits.
enum Granularity : int {
    GRAN_AUTO = 0,     // aligned runs of `max_run`, the whole tree when it fits
    GRAN_SINGLE = 1,   // one segment per item; every row goes through the combine
    GRAN_PAIRS = 2,    // aligned pairs
};

// Which leaf serves a head-dim-256 call (execution only; `host.cu::forward` states the rule). The
// relations force each and require identical bits.
enum LeafChoice : int {
    LEAF_AUTO = 0,     // the decode leaf when the call is decode-shaped, else the general leaf
    LEAF_GENERAL = 1,  // the plan, the general (or folding) leaf and the combine
    LEAF_DECODE = 2,   // the decode leaf (`decode_d256.cu`); refused for a call it cannot serve
};

//: Packed query rows the decode leaf holds per (request, KV head): its MMA's N. A head-dim-256 call
//: is decode-shaped when `max_seqlen_q * (h_q / h_kv) <= DECODE_ROWS`.
static constexpr int DECODE_ROWS = 16;

struct PlanParams {
    int b;
    int h_q;
    int h_kv;
    int head_ratio;
    int tile_tokens;      // query tokens one packed tile covers = 128 * waves / head_ratio
    int tile_rows;        // 128 * waves
    int seg;
    int granularity;
    int max_run;          // aligned-run cap the plan may emit, a power of two at most FOLD_MAX_RUN
    //: log2 of `seg`, `tile_tokens` and `h_kv` when they are powers of two, else -1: the plan is a
    //: latency-bound kernel and its per-unit integer divisions were most of its time.
    int seg_shift;
    int tt_shift;
    int hkv_shift;
    //: The host's `max_seqlen_q`: at most one query tile per request takes `gqa_plan_decode`.
    int max_q;
    int item_cap;
    int combine_cap;
    int slot_cap;
    const int* __restrict__ cu_seqlens_q;   // [b+1]
    const int* __restrict__ seqused_k;      // [b]
    //: The KV addressing, so the plan can resolve each item's first key slot (`Item::ks0`).
    int page;                               // paged: page size in key slots; 0 for a packed cache
    int64_t block_table_stride;
    const int* __restrict__ block_table;    // [b, max_blocks], or nullptr
    const int* __restrict__ cu_seqlens_k;   // [b+1] for a packed cache, else nullptr
    Item* __restrict__ items;
    CombineTile* __restrict__ combine_tiles;
    int* __restrict__ counts;
};

struct LeafParams {
    int h_q;
    int h_kv;
    int seg;
    int page;                 // paged: page size in key slots; 0 for a packed cache
    int64_t block_table_stride;
    float scale_log2;         // softmax_scale * log2(e)
    const int* __restrict__ cu_seqlens_q;
    const int* __restrict__ seqused_k;
    const int* __restrict__ block_table;    // [b, max_blocks], or nullptr
    const int* __restrict__ cu_seqlens_k;   // [b+1] for a packed cache, else nullptr
    const Item* __restrict__ items;
    const int* __restrict__ counts;
    void* __restrict__ out;                 // bf16 [T, h_q, D]
    float* __restrict__ lse;                // [T, h_q]
    float* __restrict__ partial_o;          // [slots, D]
    float* __restrict__ partial_l;          // [slots]
    float* __restrict__ partial_m;          // [slots]
    //: The fold's GLOBAL backing slots: `[ctas, waves, fold_globals, 128, D]` fp32, feature-major
    //: within a slot exactly as the shared node slot is, so a warp's 32 lanes touch 32 CONSECUTIVE
    //: floats of one feature and every access is one 128 B transaction. `nullptr` when the chosen
    //: (wave count, run) needs no global slot.
    //:
    //: Indexed by SM ID, not by `blockIdx.x`: the leaf's 192 to 224 KiB of shared memory admits ONE
    //: block per SM, so the resident CTAs are in bijection with the SMs and the workspace is bounded
    //: by the machine rather than by the grid -- 28 MB at one wave against 290 MB, which is what
    //: keeps it L2-resident. `launch_leaf_d128` verifies the occupancy at launch and refuses rather
    //: than letting a future change that admits two blocks per SM alias two CTAs onto one slot.
    float* __restrict__ fold_node;
    int fold_globals;                       // global backing slots per (CTA, wave); 0 if none
    //: SM ids the workspace was sized for. `%smid` is the CTA's slot index, and the PTX ISA does not
    //: guarantee that SM identifiers are contiguous or that `%nsmid` equals the multiprocessor count
    //: the driver reports -- they are equal and contiguous on the B200 this was measured on (max
    //: observed `%smid` 147, `%nsmid` 148, `multiProcessorCount` 148), but that is a property of the
    //: part, not of the code, and on a MIG instance the driver reports the INSTANCE's count while
    //: `%smid` stays physical. So the leaf compares against this and `__trap()`s rather than writing
    //: 64 KiB per spill past the end of the workspace.
    int fold_sms;
};

struct CombineParams {
    int h_q;
    int dim;
    int head_ratio;
    int tile_tokens;
    int tile_rows;
    //: Row groups of four a tile can have, `ceil(min(max_seqlen_q, tile_tokens) * ratio / 4)`. One
    //: block serves one `(entry, row group)` unit and the kernel grid-strides over them.
    int row_groups;
    int seg;
    float scale_log2;
    const CombineTile* __restrict__ combine_tiles;
    const int* __restrict__ counts;
    const float* __restrict__ partial_o;
    const float* __restrict__ partial_l;
    const float* __restrict__ partial_m;
    void* __restrict__ out;
    float* __restrict__ lse;
};

}  // namespace gqa_sm100
