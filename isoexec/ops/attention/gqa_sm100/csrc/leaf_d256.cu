// SPDX-License-Identifier: BSD-3-Clause
// The HD256 leaf: one fixed KV segment of one packed query tile, position-anchored, on a CTA PAIR.
//
// Execution derived from TileFoundry's CUDA port of FlashAttention-4: a cluster of two CTAs, 384
// threads, `cta_group::2` UMMA, 128 packed query rows per CTA, 128-key KV tiles through a two-slot
// TMA ring, one retire barrier per KV slot, scores in TMEM read one whole row per thread, a separate
// correction warpgroup for the O rescale.
// Copyright (c) 2025, Tri Dao et al., BSD-3-Clause.
//
// What is NOT theirs: the work item, the segment anchoring at key 0, varlen queries, the paged KV
// address function, the lazy-rescale running max in the recipe's own form, and the (O, L, M) partial
// the canonical adjacent-pair tree merges. The NUMERICAL program is the one in ../README.md section
// 1 and it is head-dim independent: every scalar operation below comes from numerics.cuh, shared
// verbatim with leaf_d128.cu, and the per-tile control flow is line-for-line that leaf's.
//
// Why a pair at all: at D = 256 the fp32 accumulator alone is 256 of TMEM's 512 columns, so one CTA
// cannot hold two score slots and an accumulator, and 320 KiB of bf16 operands do not fit 227 KiB of
// SMEM. The pair splits N -- 64 of the tile's 128 keys per CTA for QK, 128 of the 256 features per
// CTA for PV -- while each CTA keeps its OWN 128 query rows and ALL 256 output features in its own
// TMEM. TMEM is then 100 % full: `tmem_slots<D256>() == 1`, there is no in-TMEM fold, and a run of
// segments folds only through global backing slots (the folding build below).
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <cstdint>
#include <type_traits>

#include "params.h"
#include "numerics.cuh"
#include "tmem.cuh"
#include "tmem_d256.cuh"
#include "launch.h"
#ifdef GQA_D256_FOLD
#include "plan_skeleton.cuh"
#endif /* GQA_D256_FOLD */

//: This file is compiled TWICE. Here, the single-segment leaf every run-1 plan takes; in
//: `leaf_d256_fold.cu` (`GQA_D256_FOLD` defined), the leaf that folds a RUN of segments through
//: global backing slots. The fold machinery -- the warp-0 segment plan before the pair join, the
//: (L, M) ring handed to the correction after the last PV, the per-tile segment walk -- costs about
//: 1 to 2 us per item on rows that never fold, so a run-1 plan does not pay for it. Each build lives
//: in its own inner namespace so the two translation units share no definition.
#ifdef GQA_D256_FOLD
#define GQA_D256_NS d256_fold
#else
#define GQA_D256_NS d256_plain
#endif

namespace gqa_sm100 {
namespace GQA_D256_NS {

static constexpr int KV_STAGES_256 = 2;

//: ONE elected `arrive.release` per softmax warp on `statistics_ready` (count 4) instead of 128
//: means strictly fewer release fences, but it measured 1.6 to 2.6 % SLOWER on D256 decode rows:
//: the `__syncwarp` before lane 0's release puts the whole warp's alpha stores on the arrival's
//: path. Not used.

//: The fp32 partial epilogue is STAGED through shared memory so each row's 1 KiB leaves
//: as two 512 B warp transactions (full 32 B sectors) instead of 32 rows x 16 B per instruction
//: (16.1 B per sector, 2.28 GB of sectors for 1.14 GB at prefill 8192). Bit-neutral: bytes move,
//: no arithmetic.
static constexpr int DIM256 = 256;
static constexpr int CTA_ROWS = 128;          // packed query rows one CTA owns
static constexpr int PAIR_ROWS = 256;         // packed query rows the cluster owns
static constexpr int THREADS_256 = 384;

//: TMEM, all 512 columns: score slot `slot` at `slot * 128` (bf16 P aliases its first 64 columns),
//: the fp32 output accumulator at 256..511. FULL -- no merge stack in TMEM.
static constexpr int OUTPUT_COLUMN = 256;
static constexpr int TMEM_COLUMNS_256 = 512;
static_assert(OUTPUT_COLUMN + DIM256 == TMEM_COLUMNS_256,
              "at D256 the accumulator and two score slots fill TMEM exactly");

//: The one output slot TMEM has room for, hence the largest run a worker may fold IN TMEM: ONE. The
//: plain build takes single-segment items only; the folding build keeps every further pending node
//: in a global backing slot. host.cu refuses a wave override at D256.
__host__ __device__ __forceinline__ constexpr int tmem_slots_d256() {
    return (TMEM_COLUMNS_256 - OUTPUT_COLUMN) / DIM256;
}
static_assert(tmem_slots_d256() == 1, "TMEM at D256 is full: there is no in-place fold");

//: Per-warpgroup register budgets. ptxas allocates ONE static count for the whole kernel -- the
//: sub-partition file divided by the warps that land on it -- but it honours `setmaxnreg` and will
//: allocate a branch that raises its budget past that count. THE BUDGETS OF THE WARPS RESIDENT IN
//: ONE SUB-PARTITION MUST SUM TO AT MOST WHAT PTXAS ALLOCATED, which is its declared count times the
//: warps on that sub-partition -- NOT the 512-register file. A sum that exceeds it passes ptxas with
//: zero spills and then DEADLOCKS on the device: the increase waits for registers no warp will ever
//: release, and the kernel spins in its first `try_wait` (leaf_d128.cu carries the same note, where
//: 232/232/48 = 512 hung and 232/232/40 = 504 ran).
//:
//: At 384 threads this kernel has twelve warps and warp w lands on sub-partition w % 4, so every
//: sub-partition holds exactly one softmax warp (0-3), one correction warp (4-7) and one producer
//: warp (8-11). ptxas declares PTXAS_DECLARED_REGISTERS for this kernel, so the three budgets must
//: sum to at most three times that. Execution only: a register budget cannot move a bit.
//: Named here because the role branch and the launch-time feasibility check must agree; the role
//: branch carries the reasoning and `check_register_budget_d256` carries the constraint. ptxas
//: currently declares 168 and 3 x 168 = 504 >= 256 + 160 + 64 = 480, so a future change that makes it
//: declare less must fail the launch with a message rather than spin. 256 is the largest immediate
//: `setmaxnreg` takes, so the softmax warpgroup cannot be given more than this.
static constexpr int SOFTMAX_REGISTERS_256 = 256;       // holds 128 fp32 scores plus the packed P
static constexpr int CORRECTION_REGISTERS_256 = 160;    // 32-column TMEM read-modify-write
static constexpr int OTHER_REGISTERS_256 = 64;          // MMA issue, TMA submit, completion publish

//: Per-CTA bf16 operands, 192 KiB. Q is this CTA's 128 packed rows over all 256 features; K is 64 of
//: the tile's 128 keys over all 256 features (the pair supplies N = 128 keys); V is all 128 keys over
//: 128 of the 256 features (the pair supplies N = 256 features).
struct alignas(1024) LeafShared256 {
    uint16_t q[CTA_ROWS * DIM256];
    uint16_t k[KV_STAGES_256][64 * DIM256];
    uint16_t v[KV_STAGES_256][128 * 128];
};
static_assert(sizeof(LeafShared256) == 196608, "192 KiB of operands per CTA");

//: The key slot of logical key `key_base` of request `req` in the flattened cache. Page ids are
//: invisible to the arithmetic: both addressing modes hand the leaf the same 128 logical keys in the
//: same order, and the caller guarantees `page % KEY_TILE == 0` so a KV tile never crosses a page --
//: which is also what lets rank 1 read its 64 keys at `slot + 64` without consulting the table.
__device__ __forceinline__ int key_slot_256(const LeafParams& p, int req, int key_base) {
    if (p.page > 0) {
        const int block = p.block_table[int64_t(req) * p.block_table_stride + key_base / p.page];
        return block * p.page + (key_base % p.page);
    }
    return p.cu_seqlens_k[req] + key_base;
}

//: Every scalar the whole CTA derives from the work item, in SHARED memory, computed once.
//:
//: `q_end`, `seqused_k[req]` and the `Item` itself are GLOBAL loads, so ptxas cannot rematerialize
//: them: held as per-thread locals they are live from the prologue to the epilogue across all FOUR
//: role regions at once, and ptxas spills them -- 24 bytes of spill stores in the prologue and an
//: `LDL` in every region, measured. Shared memory is the same fix leaf_d128.cu applies to its
//: `SegmentPlan` for the same reason, and it costs one `LDS` where a spill costs an `LDL`.
struct ItemScalars {
    int req, tok0, kv_head, slot_base;
    int q_end;        // cu_seqlens_q[req + 1]
    int used;         // seqused_k[req]
#ifndef GQA_D256_FOLD
    int seg_begin;    // the item's single segment starts at this logical key
    int total;        // KV tiles of that segment, at least one
#else /* GQA_D256_FOLD */
    int seg_begin;    // the item's FIRST segment starts at this logical key (k0 * seg)
    int total;        // KV tiles over the whole item (all its segments), at least one per segment
    int k0;           // first segment of the run
#endif /* GQA_D256_FOLD */
};

#ifdef GQA_D256_FOLD
//: THE D256 FOLD. TMEM holds one fp32 output tile per CTA, so every pending node of
//: the streaming counter lives in a GLOBAL backing slot: `[SM][index][feature 256][row 128]` fp32,
//: 128 KiB, row-private (a thread owns one TMEM lane, one row of every slot): no barrier, no fence.
//: Each CTA of the pair spills and merges its OWN 128 rows x 256 features -- the pair splits N in
//: the operands, not the TMEM lanes. The plan skeleton is the D128 one at `tmem_slots == 1`.
static constexpr int NODE256 = 128 * DIM256;
//: The (L, M) handoff ring: segment j uses `seg_lm[j % 4]` with `segment_stats[j % 4]` at phase
//: `(j / 4) & 1`, and the correction reads it at the BOUNDARY, after the segment's last PV -- so the
//: last PV is not held behind the softmax's row sum (reading it before the last tile's
//: `output_ready` arrival costs ~1.8 us per item on `q1/k128`). The softmax can lead the
//: correction by at most three segment ends.
static constexpr int SEG_RING = 4;

//: A depth-indexed float stack kept in registers by compile-time selection (leaf_d128.cu's RegStack).
template <int N> struct RegStack256 {
    float v[N];
    __device__ __forceinline__ float get(int i) const {
        float r = v[0];
        #pragma unroll
        for (int k = 1; k < N; ++k)
            if (i == k) r = v[k];
        return r;
    }
    __device__ __forceinline__ void set(int i, float x) {
        #pragma unroll
        for (int k = 0; k < N; ++k)
            if (i == k) v[k] = x;
    }
};

//: Move this row of the output tile (256 features) into a global backing slot, feature-major
//: (`node[feature * 128 + row]`: a warp's 32 lanes write 32 consecutive floats). Row-private.
template <int C>
__device__ __forceinline__ void spill_node256(float* node, int row, uint32_t source) {
    #pragma unroll 1
    for (int chunk = 0; chunk < DIM256 / C; ++chunk) {
        float values[C];
        tload<C>(source + chunk * C, values);
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        #pragma unroll
        for (int j = 0; j < C; ++j) node[(chunk * C + j) * 128 + row] = values[j];
    }
}

//: `O_right <- O_left * sa + O_right * sb` with the LEFT operand from a backing slot and the right
//: (and the result) in the TMEM output tile: `merge_o_pair` over the same elements, left term first.
template <int C>
__device__ __forceinline__ void merge_from_node256(const float* node, int row, uint32_t right,
                                                   float sa, float sb) {
    #pragma unroll 1
    for (int chunk = 0; chunk < DIM256 / C; ++chunk) {
        float a[C], b[C];
        #pragma unroll
        for (int j = 0; j < C; ++j) a[j] = node[(chunk * C + j) * 128 + row];
        tload<C>(right + chunk * C, b);
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        #pragma unroll
        for (int j = 0; j < C; j += 2)
            merge_o_pair(a[j], a[j + 1], b[j], b[j + 1], sa, sb);
        tstore<C>(right + chunk * C, a);
    }
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}

#endif /* GQA_D256_FOLD */
template <int RATIO>
__global__ __launch_bounds__(THREADS_256, 1) void gqa_leaf_d256(
        const __grid_constant__ CUtensorMap mapq,
        const __grid_constant__ CUtensorMap mapk,
        const __grid_constant__ CUtensorMap mapv,
        const __grid_constant__ LeafParams p) {
    constexpr int KV = KV_STAGES_256;
    constexpr int BASE = 2 * CTA_ROWS;                       // 256: softmax 0-127, correction 128-255
    constexpr int MMA_THREAD = 256, SUBMIT_THREAD = 288, PUBLISH_THREAD = 320;
    constexpr int CTA_TOKENS = CTA_ROWS / RATIO;             // query tokens one CTA covers
    constexpr int TILE_TOKENS = PAIR_ROWS / RATIO;           // query tokens the cluster covers
    static_assert(CTA_ROWS % RATIO == 0, "a packed 128-row CTA tile must hold whole tokens");

    const int thread = threadIdx.x;
    //: THE EARLY EXIT, before any `mbarrier.init` and before any cluster barrier. Both CTAs of a
    //: cluster read the same item index and the same count, so they agree and neither is left
    //: waiting for a partner that exited. Getting this wrong deadlocks the partner, not this CTA.
    const int item_index = int(blockIdx.x) >> 1;
    if (item_index >= p.counts[NUM_ITEMS]) return;
    //: `rank` comes from `blockIdx`, which ptxas rematerializes from a special register, so unlike
    //: the item's scalars it costs nothing to keep live.
    const int rank = int(blockIdx.x) & 1;

    extern __shared__ __align__(1024) unsigned char storage[];
    auto& s = *reinterpret_cast<LeafShared256*>(storage);
    //: The KV tile's first key SLOT in the flattened cache, per tile ordinal, precomputed. The paged
    //: address function reads the block table from global memory, and the TMA submitter is one
    //: thread issuing serially: a dependent global load in front of every copy puts a full memory
    //: round trip on the issue path of every KV tile.
    __shared__ int key_slots[MAX_KEY_SLOTS];
    __shared__ __align__(8) uint64_t loaded_q, loaded_k[KV], loaded_v[KV];
    __shared__ __align__(8) uint64_t query_ready, keys_ready[KV], values_ready[KV];
    __shared__ __align__(8) uint64_t qk_done[KV], pv_done[KV];
    __shared__ __align__(8) uint64_t statistics_ready[KV];
    __shared__ __align__(8) uint64_t probabilities_low[KV], probabilities_ready[KV];
    __shared__ __align__(8) uint64_t output_ready;
    //: softmax -> correction: the row's rescale factor.
    __shared__ float row_alpha[KV][CTA_ROWS];
    //: softmax -> correction, read only after the final cluster join: the root's reciprocal.
    __shared__ float row_inverse[CTA_ROWS];
    __shared__ uint32_t tmem_slot;
    __shared__ ItemScalars sc;
#ifdef GQA_D256_FOLD
    //: The item's segment plan (the D128 leaf's skeleton at one TMEM tile) and the
    //: softmax -> correction (L, M) of each segment, alternating by SEGMENT parity with its own
    //: barrier (the D128 leaf's `segment_stats` argument: the correction consumes segment j's stats
    //: only at the next segment's first tile, after the softmax may have finished segment j + 1).
    __shared__ SegmentPlan sp;
    __shared__ float seg_lm[SEG_RING][2][CTA_ROWS];
    __shared__ __align__(8) uint64_t segment_stats[SEG_RING];
#endif /* GQA_D256_FOLD */

    if (thread == 0) {
        const Item it = p.items[item_index];
        sc.req = it.req;
        sc.tok0 = it.tok0;
        sc.kv_head = it.kv_head;
        sc.slot_base = it.slot_base;
        //: From the ITEM, not from two more device tensors: they ride in it (params.h), which takes
        //: one dependent global round trip off the front of every CTA pair.
        sc.q_end = it.q_end;
        sc.used = it.used;
        //: The causal border of absolute packed token t of this request is `used - q_end + t + 1`;
        //: its logical position is that minus one. Nothing else about the request reaches the
        //: arithmetic. `token_last` spans the WHOLE 256-row tile, so both CTAs of the pair derive the
        //: same KV tile count -- they must, because one MMA is issued for the pair.
        const int token_last = min(it.tok0 + TILE_TOKENS - 1, sc.q_end - 1);
        const int border_max = sc.used - sc.q_end + token_last + 1;
#ifndef GQA_D256_FOLD
        //: host.cu hands this build only run-1 plans, so every item is a single segment, `it.k0`.
        //: If that ever stops being true this kernel would silently compute
        //: the first segment of the run and publish it as the whole node, so it traps instead
        //: (leaf_d128.cu traps the analogous impossible slot exhaustion).
        if (it.k1 - it.k0 != 1) __trap();
#endif /* ! GQA_D256_FOLD */
        sc.seg_begin = it.k0 * p.seg;
#ifndef GQA_D256_FOLD
        const int seg_stop = min(sc.seg_begin + p.seg, border_max);
        int tiles = (seg_stop - sc.seg_begin + KEY_TILE - 1) / KEY_TILE;
        if (tiles < 1) tiles = 1;   // a segment the tile does not reach still runs one masked tile
        sc.total = tiles;
#else /* GQA_D256_FOLD */
        sc.k0 = it.k0;
#endif /* GQA_D256_FOLD */

        asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" :: "r"(shared_address(&loaded_q)));
        // One arrival from each CTA's completion publisher.
        asm volatile("mbarrier.init.shared::cta.b64 [%0],2;" :: "r"(shared_address(&query_ready)));
        // One elected lane per correction warp, both CTAs: 4 x 2.
        asm volatile("mbarrier.init.shared::cta.b64 [%0],8;" :: "r"(shared_address(&output_ready)));
        #pragma unroll
        for (int j = 0; j < KV; ++j) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" :: "r"(shared_address(&loaded_k[j])));
            asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" :: "r"(shared_address(&loaded_v[j])));
            asm volatile("mbarrier.init.shared::cta.b64 [%0],2;" :: "r"(shared_address(&keys_ready[j])));
            asm volatile("mbarrier.init.shared::cta.b64 [%0],2;" :: "r"(shared_address(&values_ready[j])));
            // One retire barrier per KV slot: each generation is armed exactly once per slot cycle
            // and waited exactly once, so `try_wait.parity` can never meet a producer that has
            // advanced an EVEN number of generations. The UMMA commit
            // multicasts one arrival into each CTA's copy, hence the count of one.
            asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" :: "r"(shared_address(&qk_done[j])));
            asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" :: "r"(shared_address(&pv_done[j])));
            // Every softmax thread (or, elected, every softmax warp) publishes its rows' alpha.
            asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;"
                         :: "r"(shared_address(&statistics_ready[j])),
                            "r"(128));
            // One elected lane per softmax warp, both CTAs: 4 x 2, twice per tile (the low half of
            // P and then the high half).
            asm volatile("mbarrier.init.shared::cta.b64 [%0],8;"
                         :: "r"(shared_address(&probabilities_low[j])));
            asm volatile("mbarrier.init.shared::cta.b64 [%0],8;"
                         :: "r"(shared_address(&probabilities_ready[j])));
#ifdef GQA_D256_FOLD

#endif /* GQA_D256_FOLD */
        }
    }
#ifdef GQA_D256_FOLD
    if (thread == 0) {
        // One elected release arrival per softmax warp of THIS CTA, per segment of the ring.
        #pragma unroll
        for (int r = 0; r < SEG_RING; ++r)
            asm volatile("mbarrier.init.shared::cta.b64 [%0],4;"
                         :: "r"(shared_address(&segment_stats[r])));
    }
    //: Warp 0 builds the plan. Both CTAs build the SAME plan from the same item: `border_max` spans
    //: the pair's whole 256-row tile, exactly as the tile count always has.
    if (thread < 32) {
        const Item it = p.items[item_index];
        const int token_last = min(it.tok0 + TILE_TOKENS - 1, it.q_end - 1);
        const int border_max = it.used - it.q_end + token_last + 1;
        build_segment_plan(sp, it.k0, it.k1, p.seg, border_max, 1, p.fold_globals);
    }
#endif /* GQA_D256_FOLD */
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    synchronize_pair();
#ifdef GQA_D256_FOLD
    //: The logical key a global tile ordinal starts at, over the item's segments.
    auto key_base_of = [&](int g) {
        int j = 0;
        while (j + 1 < sp.nsegs && g >= sp.pre[j + 1]) ++j;
        return (sc.k0 + j) * p.seg + (g - sp.pre[j]) * KEY_TILE;
    };
#endif /* GQA_D256_FOLD */

    //: Every read of `sc` below sits INSIDE the block that uses it, so no item scalar is live across
    //: a role boundary. Segments are consumed in ascending order and an item holds exactly one, so
    //: the tile ordinal is the only thing the TMA pipeline needs to know about the item's shape:
    //: `key_base(g) = seg_begin + g * KEY_TILE`.
    if (thread >= PUBLISH_THREAD && thread < PUBLISH_THREAD + 32) {
#ifndef GQA_D256_FOLD
        const int req = sc.req, seg_begin = sc.seg_begin, total = sc.total;
#else /* GQA_D256_FOLD */
        const int req = sc.req, total = sp.total;
#endif /* GQA_D256_FOLD */
        if (total <= MAX_KEY_SLOTS)
            for (int g = thread - PUBLISH_THREAD; g < total; g += 32)
#ifndef GQA_D256_FOLD
                key_slots[g] = key_slot_256(p, req, seg_begin + g * KEY_TILE);
#else /* GQA_D256_FOLD */
                key_slots[g] = key_slot_256(p, req, key_base_of(g));
#endif /* GQA_D256_FOLD */
    }

    if (thread == SUBMIT_THREAD) {
#ifndef GQA_D256_FOLD
        const int req = sc.req, seg_begin = sc.seg_begin, total = sc.total;
#else /* GQA_D256_FOLD */
        const int req = sc.req, total = sp.total;
#endif /* GQA_D256_FOLD */
        const int kv_head = sc.kv_head;
        //: This CTA's 128 packed rows are rows [rank*128, rank*128+128) of the pair's tile.
        const int cta_tok0 = sc.tok0 + rank * CTA_TOKENS;
        // Q first, then K0 and K1: the first QK reads Q and K0 only.
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"
                     :: "r"(shared_address(&loaded_q)), "r"(CTA_ROWS * DIM256 * 2) : "memory");
        #pragma unroll
        for (int chunk = 0; chunk < 4; ++chunk)
            load_tile(mapq, s.q + chunk * CTA_ROWS * 64, &loaded_q, chunk, kv_head * RATIO, cta_tok0);
        #pragma unroll 1
        for (int g = 0; g < 2; ++g) {
            if (g >= total) break;
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"
                         :: "r"(shared_address(&loaded_k[g])), "r"(64 * DIM256 * 2) : "memory");
            // The key-slot cache is published by the cluster join that follows tcgen05.alloc, which
            // every role passes before its steady state; these two copies precede it, so they read
            // the address function directly.
#ifndef GQA_D256_FOLD
            const int ks = key_slot_256(p, req, seg_begin + g * KEY_TILE);
#else /* GQA_D256_FOLD */
            const int ks = key_slot_256(p, req, key_base_of(g));
#endif /* GQA_D256_FOLD */
            #pragma unroll
            for (int chunk = 0; chunk < 4; ++chunk)
                load_tile(mapk, s.k[g] + chunk * 64 * 64, &loaded_k[g], chunk, kv_head,
                          ks + rank * 64);
        }
    }
    if (thread < 32)
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0],%1;"
                     :: "r"(shared_address(&tmem_slot)), "r"(TMEM_COLUMNS_256) : "memory");
    synchronize_pair();
    const uint32_t tmem = tmem_slot;

    const float scale_log2 = p.scale_log2;
    // M = 256 (the pair's packed query rows), N = 128 keys for QK and 256 features for PV,
    // bf16 x bf16 -> fp32. FA4's own D256 two-CTA descriptors.
    constexpr uint32_t base_instruction = (1u << 4) | (1u << 7) | (1u << 10) | ((256u / 16) << 24);
    constexpr uint32_t qk_instruction = base_instruction | ((128u / 8) << 17);
    constexpr uint32_t pv_instruction = base_instruction | ((256u / 8) << 17) | (1u << 16);
    static_assert(qk_instruction == 0x10200490u && pv_instruction == 0x10410490u);

    if (thread >= BASE) {
        //: The producer warpgroup's budget, as the FIRST statement of the role branch and after the
        //: last cluster join in front of it. All four of its warps -- MMA, submit, publish and the
        //: idle twelfth -- execute it, because `setmaxnreg.*.sync.aligned` is warpgroup-aligned and
        //: because a warp that kept the default count would blow its sub-partition's budget (see
        //: OTHER_REGISTERS_256 above: the idle warp sits on sub-partition 3 beside a 256-register
        //: softmax warp and a 160-register correction warp).
        registers_decrease<OTHER_REGISTERS_256>();

#ifndef GQA_D256_FOLD
        const int total = sc.total;
        const int seg_begin = sc.seg_begin;
#else /* GQA_D256_FOLD */
        const int total = sp.total;
#endif /* GQA_D256_FOLD */
        const bool slots_cached = total <= MAX_KEY_SLOTS;
        auto slot_of = [&](int g) {
            return slots_cached ? key_slots[g]
#ifndef GQA_D256_FOLD
                                : key_slot_256(p, sc.req, seg_begin + g * KEY_TILE);
#else /* GQA_D256_FOLD */
                                : key_slot_256(p, sc.req, key_base_of(g));
#endif /* GQA_D256_FOLD */
        };
        if (thread == SUBMIT_THREAD) {
            const int kv_head = sc.kv_head;
            // FA4 load order: K0, K1, V0, K2, V1, ..., Vlast, across the whole item.
            for (int ordinal = (total > 1 ? 2 : 1); ordinal < 2 * total; ++ordinal) {
                const bool is_key = (ordinal & 1) != 0 && ordinal < 2 * total - 1;
                const int g = is_key ? (ordinal + 1) / 2 : (ordinal - 1) / 2;
                const int slot = g % KV, cycle = g / KV;
                const int ks = slot_of(g);
                if (is_key) {
                    // Retire the QK that read this slot's K before overwriting it.
                    if (g >= KV) bar_wait(&qk_done[slot], (cycle - 1) & 1);
                    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"
                                 :: "r"(shared_address(&loaded_k[slot])), "r"(64 * DIM256 * 2)
                                 : "memory");
                    #pragma unroll
                    for (int chunk = 0; chunk < 4; ++chunk)
                        load_tile(mapk, s.k[slot] + chunk * 64 * 64, &loaded_k[slot], chunk,
                                  kv_head, ks + rank * 64);
                } else {
                    if (g >= KV) bar_wait(&pv_done[slot], (cycle - 1) & 1);
                    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"
                                 :: "r"(shared_address(&loaded_v[slot])), "r"(128 * 128 * 2)
                                 : "memory");
                    // This CTA owns features [rank*128, rank*128+128) of the 256: chunks rank*2 and
                    // rank*2 + 1 of the four 64-feature chunks the tensor map exposes.
                    #pragma unroll
                    for (int chunk = 0; chunk < 2; ++chunk)
                        load_tile(mapv, s.v[slot] + chunk * 128 * 64, &loaded_v[slot],
                                  rank * 2 + chunk, kv_head, ks);
                }
            }
        } else if (thread >= PUBLISH_THREAD && thread < PUBLISH_THREAD + 32) {
            const int lane = thread - PUBLISH_THREAD;
            // Fixed publication order Q, K0, K1, V0, K2, V1, ...: because ONE thread publishes in
            // this order, values_ready[slot] for tile g transitively implies K(g+1) has landed, so
            // the MMA issuer waits only on values_ready in the steady state.
            if (lane == 0) {
                bar_wait(&loaded_q, 0);
                arrive_cluster_release(&query_ready);
                bar_wait(&loaded_k[0], 0);
                arrive_cluster_release(&keys_ready[0]);
            }
            for (int ordinal = 1; ordinal < 2 * total; ++ordinal) {
                const bool is_key = (ordinal & 1) != 0 && ordinal < 2 * total - 1;
                const int g = is_key ? (ordinal + 1) / 2 : (ordinal - 1) / 2;
                const int slot = g % KV, phase = (g / KV) & 1;
                if (is_key) {
                    if (lane == 0) {
                        bar_wait(&loaded_k[slot], phase);
                        arrive_cluster_release(&keys_ready[slot]);
                    }
                    continue;
                }
                if (lane == 0) bar_wait(&loaded_v[slot], phase);
                __syncwarp();
                // Keys at or past seqused_k are unwritten cache slots. Their probabilities are
                // exactly +0.0, but 0 * NaN is NaN, so the value rows they would feed the PV MMA are
                // zeroed here instead of being trusted. Only a segment's last tile can be partial.
                // With the feature split each CTA zeroes the 128 features it owns.
                const int valid = min(KEY_TILE,
#ifndef GQA_D256_FOLD
                                      max(0, sc.used - (seg_begin + g * KEY_TILE)));  // logical
#else /* GQA_D256_FOLD */
                                      max(0, sc.used - key_base_of(g)));  // logical
#endif /* GQA_D256_FOLD */
                if (valid < KEY_TILE) {
                    const int words = (KEY_TILE - valid) * 64;   // 64 features per chunk row
                    #pragma unroll 1
                    for (int chunk = 0; chunk < 2; ++chunk) {
                        uint16_t* rows = s.v[slot] + chunk * 128 * 64 + valid * 64;
                        for (int i = lane * 8; i < words; i += 32 * 8)
                            *reinterpret_cast<uint4*>(rows + i) = make_uint4(0, 0, 0, 0);
                    }
                }
                // The consumer is the rank-0 MMA thread, which may be in the OTHER CTA, so the fence
                // and the arrival are both cluster-scoped.
                fence_async_cluster();
                __syncwarp();
                if (lane == 0) arrive_cluster_release(&values_ready[slot]);
            }
        } else if (thread == MMA_THREAD && rank == 0) {
            bar_wait_cluster(&query_ready, 0);
#ifdef GQA_D256_FOLD
            //: (segment, tile in segment) carried incrementally: the accumulator RESET is at every
            //: SEGMENT start (an item may hold a run of segments in this build).
            int seg_j = 0, seg_u = 0;
#endif /* GQA_D256_FOLD */
            for (int g = 0; g < total; ++g) {
                // QK(g) in the prologue, then one QK ahead per tile. The score slot QK(g+1) writes
                // is the one PV(g-1) read: the issuer waited `probabilities_ready[(g-1)&1]` one
                // iteration ago, after softmax(g-1) had loaded all 128 score columns and stored P,
                // and the P that PV(g-1) read is ordered by IN-ORDER UMMA issue from one CTA pair.
                for (int ahead = (g == 0 ? 0 : 1); ahead <= 1; ++ahead) {
                    const int u = g + ahead;
                    if (u >= total) continue;
                    const int slot = u % KV, phase = (u / KV) & 1;
                    bar_wait_cluster(&keys_ready[slot], phase);
                    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                    qk_sequence256(tmem + uint32_t(slot * 128), descriptor(s.q, false),
                                   descriptor(s.k[slot], false), qk_instruction);
                    commit_pair(&qk_done[slot]);
                }
                const int slot = g % KV, phase = (g / KV) & 1;
                bar_wait(&probabilities_low[slot], phase);
                // The consumer side of the TMEM handoff: P was written by tcgen05.st in the softmax
                // warps, which published it with fence::before_thread_sync.
                asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                bar_wait(&output_ready, g & 1);
                bar_wait_cluster(&values_ready[slot], phase);
                asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                const uint32_t probability = tmem + uint32_t(slot * 128);
                const uint64_t values = descriptor(s.v[slot], true);
#ifndef GQA_D256_FOLD
                // `g != 0` is the accumulator RESET at the segment start: the segment's first PV
                // overwrites, so no state crosses a segment boundary.
                pv256_keys_0_63(tmem + OUTPUT_COLUMN, probability, values, pv_instruction, g != 0);
#else /* GQA_D256_FOLD */
                // `seg_u != 0` is the accumulator RESET at every segment start: the segment's first
                // PV overwrites, so no state crosses a segment boundary. The output tile it
                // overwrites was spilled by the correction warpgroup BEFORE its `output_ready`
                // arrival for this tile, which this issuer waited above.
                pv256_keys_0_63(tmem + OUTPUT_COLUMN, probability, values, pv_instruction, seg_u != 0);
#endif /* GQA_D256_FOLD */
                bar_wait(&probabilities_ready[slot], phase);
                asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                pv256_keys_64_127(tmem + OUTPUT_COLUMN, probability, values, pv_instruction, true);
                commit_pair(&pv_done[slot]);
#ifdef GQA_D256_FOLD
                if (++seg_u == sp.upd[seg_j]) { ++seg_j; seg_u = 0; }
#endif /* GQA_D256_FOLD */
            }
        }
    } else if (thread < CTA_ROWS) {
        registers_increase<SOFTMAX_REGISTERS_256>();
#ifndef GQA_D256_FOLD
        const int total = sc.total, seg_begin = sc.seg_begin;
#endif /* ! GQA_D256_FOLD */
        const int q_end = sc.q_end, base = sc.used - q_end;
        const int cta_tok0 = sc.tok0 + rank * CTA_TOKENS;
        const int row = thread;
        const int token = cta_tok0 + (row / RATIO);
        const int border = base + token + 1;
        const bool live = token < q_end;
        const bool padded = cta_tok0 >= q_end;
        const bool all_live = cta_tok0 + CTA_TOKENS - 1 < q_end;

        float maximum = -CUDART_INF_F;
        float denominator = 0.0f;
#ifndef GQA_D256_FOLD
        //: Reset at every segment start; an item holds exactly ONE segment at D256, so this IS the
        //: reset and nothing can carry across a boundary.
        const bool seg_live = live && border > seg_begin;
        const int limit = border - 1 - seg_begin;

        //: One subtraction per tile: `key + c > limit` is `c > limit - key` for every compile-time
        //: column c, and a row the segment does not reach takes a limit below column zero.
        const int tile_limit = (base + cta_tok0 + 1) - 1 - seg_begin;
        int interior_end;
        if (padded) {
            // No row of this CTA owns any output, so the per-element mask is pure overhead: the
            // partner advances at the slower CTA's rate and nothing this CTA computes is written.
            interior_end = total;
        } else if (!all_live) {
            interior_end = 0;
        } else {
            const int last_interior = tile_limit - (KEY_TILE - 1);
            interior_end = last_interior < 0 ? 0 : last_interior / KEY_TILE + 1;
            if (interior_end > total) interior_end = total;
        }
#else /* GQA_D256_FOLD */
        //: Reset at every segment start (an item holds a RUN of segments in this build).
        bool seg_live = false;
        int limit = -1;
#endif /* GQA_D256_FOLD */

#ifndef GQA_D256_FOLD
        auto body = [&](auto interior_tag, int g) {
#else /* GQA_D256_FOLD */
        auto body = [&](auto interior_tag, int g, int u) {
#endif /* GQA_D256_FOLD */
            constexpr bool interior = decltype(interior_tag)::value;
            const int slot = g % KV, phase = (g / KV) & 1;
            bar_wait(&qk_done[slot], phase);
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
            float scores[4][32];
            #pragma unroll
            for (int group = 0; group < 4; ++group)
                load32(tmem + uint32_t(slot * 128) + group * 32, scores[group]);
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            if constexpr (!interior) {
                // Masked scores are replaced BEFORE the max.
#ifndef GQA_D256_FOLD
                const int local_limit = seg_live ? limit - g * KEY_TILE : -1;
#else /* GQA_D256_FOLD */
                const int local_limit = seg_live ? limit - u * KEY_TILE : -1;
#endif /* GQA_D256_FOLD */
                #pragma unroll
                for (int group = 0; group < 4; ++group) {
                    #pragma unroll
                    for (int jj = 0; jj < 32; ++jj)
                        if (group * 32 + jj > local_limit) scores[group][jj] = -CUDART_INF_F;
                }
            }
            float max0 = maximum, max1 = maximum, max2 = maximum, max3 = maximum;
            #pragma unroll
            for (int jj = 0; jj < 32; ++jj) {
                max0 = fmaxf(max0, scores[0][jj]);
                max1 = fmaxf(max1, scores[1][jj]);
                max2 = fmaxf(max2, scores[2][jj]);
                max3 = fmaxf(max3, scores[3][jj]);
            }
            float next_maximum = fmaxf(fmaxf(max0, max1), fmaxf(max2, max3));
            const bool empty_row = next_maximum == -CUDART_INF_F;
            float safe_maximum = empty_row ? 0.0f : next_maximum;
            const float shift = __fmul_rn(maximum - safe_maximum, scale_log2);
            // The recipe's LAZY RESCALE (params.h RESCALE_THRESHOLD, 8 log2 units, FA4's own value):
            // while this tile's row max exceeds the running max by at most the threshold, the running
            // max is kept, so the bias is unchanged, no O rescale runs, and this tile's probabilities
            // are simply taken against the older max. The decision reads only this row's own running
            // max over its fixed KV tiles in logical order inside ONE segment.
            const bool keep = shift >= -RESCALE_THRESHOLD;
            if (keep) {
                next_maximum = maximum;
                safe_maximum = maximum;
            }
            const float alpha = keep ? 1.0f : exp2_native(shift);
            //: Hand the correction warpgroup this row's factor. The BALLOT the HD128 leaf takes here
            //: is taken there instead: `__any_sync` plus its operands were live across the 128 fp32
            //: scores and were what made this warpgroup spill, while the correction warpgroup has
            //: 160 registers and almost nothing in them. The vote is the same vote and the
            //: `alpha != 1.0f` per-lane skip inside `correction256` is what makes it bit-neutral.
            //: An empty row -- one this segment has not reached, or whose every key so far was
            //: masked -- has `alpha == 0` and an accumulator that is exactly signed zero, so the
            //: rescale is the identity for it either way. Publishing the IDENTITY instead keeps it
            //: out of the correction warpgroup's ballot below, which is exactly the exclusion the
            //: HD128 leaf makes when it builds its own ballot (`!keep && !empty_row`), and it is what
            //: keeps chunked prefill and every ragged tail off the rescale path.
            row_alpha[slot][row] = empty_row ? 1.0f : alpha;
            asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _,[%0];"
                         :: "r"(shared_address(&statistics_ready[slot])) : "memory");

            const float bias = __fmul_rn(-safe_maximum, scale_log2);
            //: Line for line the HD128 leaf's step 4/5: four sixteen-pair quarters, each written into
            //: the score columns its probabilities alias. The probability tile is published in TWO
            //: halves -- after quarter 1 (P columns 0-31, keys 0-63, the first four PV K16 steps) and
            //: after quarter 3 -- so the tensor pipe starts PV after 64 of the 128 exponentials
            //: Sixteen packed words live at a time, not
            //: thirty-two: at 256 registers the softmax warpgroup already holds 128 fp32 scores.
            #pragma unroll
            for (int quarter = 0; quarter < 4; ++quarter) {
                uint32_t packed[16];
                #pragma unroll
                for (int pair = 0; pair < 16; ++pair) {
                    const int jj = pair * 2;
                    const FloatPair shifted = affine_pair(scores[quarter][jj],
                                                          scores[quarter][jj + 1],
                                                          scale_log2, bias);
                    //: The SAME two exp2 implementations and the SAME selection pattern as the
                    //: HD128 leaf: `emulated_exp2` is a function of the column pair inside the
                    //: fixed 128-key tile and of nothing else, at freq 8, res 4, start_frg 1.
                    //: FA4's own D256 tuning (freq 14, res 6, start 0) is NOT
                    //: adopted -- the pattern is part of the recipe, not of the head dim.
                    const FloatPair e = emulated_exp2(quarter * 16 + pair)
                        ? polynomial_exp2(shifted.lo, shifted.hi)
                        : FloatPair{exp2_native(shifted.lo), exp2_native(shifted.hi)};
                    scores[quarter][jj] = e.lo;
                    scores[quarter][jj + 1] = e.hi;
                    packed[pair] = pack_bf16_pair(e.lo, e.hi);
                }
                store16_bits(tmem + uint32_t(slot * 128) + quarter * 16, packed);
                if ((quarter & 1) == 0) continue;
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
                arrive_warp_cluster(quarter == 1 ? &probabilities_low[slot]
                                                 : &probabilities_ready[slot]);
            }
            FloatPair sum01 = {__fmul_rn(denominator, alpha), 0.0f}, sum23 = {0.0f, 0.0f};
            #pragma unroll
            for (int group = 0; group < 4; ++group) {
                #pragma unroll
                for (int jj = 0; jj < 32; jj += 4) {
                    sum01 = add_pair(sum01, {scores[group][jj], scores[group][jj + 1]});
                    sum23 = add_pair(sum23, {scores[group][jj + 2], scores[group][jj + 3]});
                }
            }
            denominator = (sum01.lo + sum01.hi) + (sum23.lo + sum23.hi);
            maximum = next_maximum;
        };

        int g = 0;
#ifndef GQA_D256_FOLD
        for (; g < interior_end; ++g) body(std::true_type{}, g);
        for (; g < total; ++g) body(std::false_type{}, g);

        const float leaf_l = denominator;
        const float leaf_m = maximum == -CUDART_INF_F ? M_EMPTY : maximum;
        //: The correction warpgroup applies the root's reciprocal; it never sees L or M otherwise.
        row_inverse[row] = root_inverse(leaf_l);
        //: The row's scalars are plain global stores that depend on nothing the epilogue reads, so
        //: they are published HERE rather than after the final cluster join -- which keeps L and M
        //: out of the live set that crosses it. `seg_live` is exactly the epilogue's `item_live`
        //: predicate: `token < q_end && border > first_begin`, and `first_begin == seg_begin`
        //: because an item holds one segment.
        if (seg_live) {
            const int pair_row = rank * CTA_ROWS + row;
            const int head = sc.kv_head * RATIO + (row % RATIO);
            const int slot_base = sc.slot_base;
            if (slot_base < 0) {
                p.lse[int64_t(token) * p.h_q + head] = root_lse(leaf_l, leaf_m, scale_log2);
#else /* GQA_D256_FOLD */
        for (int j = 0; j < sp.nsegs; ++j) {
            maximum = -CUDART_INF_F;
            denominator = 0.0f;
            const int seg_begin = (sc.k0 + j) * p.seg;
            seg_live = live && border > seg_begin;
            limit = border - 1 - seg_begin;
            //: One subtraction per tile: `key + c > limit` is `c > limit - key` for every
            //: compile-time column c, and a row the segment does not reach takes a limit below zero.
            const int tile_limit = (base + cta_tok0 + 1) - 1 - seg_begin;
            const int upd = sp.upd[j];
            int interior_end;
            if (padded) {
                interior_end = upd;
            } else if (!all_live) {
                interior_end = 0;
#endif /* GQA_D256_FOLD */
            } else {
#ifndef GQA_D256_FOLD
                const int64_t slot = int64_t(slot_base) + pair_row;
                p.partial_l[slot] = leaf_l;
                p.partial_m[slot] = leaf_m;
#else /* GQA_D256_FOLD */
                const int last_interior = tile_limit - (KEY_TILE - 1);
                interior_end = last_interior < 0 ? 0 : last_interior / KEY_TILE + 1;
                if (interior_end > upd) interior_end = upd;
#endif /* GQA_D256_FOLD */
            }
#ifdef GQA_D256_FOLD
            int u = 0;
            for (; u < interior_end; ++u, ++g) body(std::true_type{}, g, u);
            for (; u < upd; ++u, ++g) body(std::false_type{}, g, u);
            //: The segment's (L, M), raw frame (-inf kept; the correction applies M_EMPTY), to the
            //: correction warpgroup: elected RELEASE arrive after `__syncwarp`.
            seg_lm[j % SEG_RING][0][row] = denominator;
            seg_lm[j % SEG_RING][1][row] = maximum;
            __syncwarp();
            if ((thread & 31) == 0)
                asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _,[%0];"
                             :: "r"(shared_address(&segment_stats[j % SEG_RING])) : "memory");
#endif /* GQA_D256_FOLD */
        }
    } else {
        registers_decrease<CORRECTION_REGISTERS_256>();
#ifndef GQA_D256_FOLD
        const int total = sc.total;
#else /* GQA_D256_FOLD */
        const int total = sp.total;
#endif /* GQA_D256_FOLD */
        const int row = thread - CTA_ROWS;
#ifdef GQA_D256_FOLD
        //: This CTA's global backing slots, indexed by SM id (one CTA per SM: 192 KiB of shared
        //: memory, verified at launch).
        int64_t fold_offset = 0;
        if (p.fold_globals > 0) {
            uint32_t sm;
            asm volatile("mov.u32 %0,%%smid;" : "=r"(sm));
            if (int(sm) >= p.fold_sms) __trap();
            fold_offset = int64_t(sm) * p.fold_globals * NODE256;
        }
        auto backing_of = [&](int b) -> float* { return p.fold_node + fold_offset + int64_t(b) * NODE256; };
        RegStack256<FOLD_STACK> stL{}, stM{};
        int depth = 0;
        float Lseg = 0.0f, Mseg = 0.0f;
        //: The segment BOUNDARY, after its last PV: push (L, M) and run its merges in plan order.
        //: At one TMEM tile every left operand is a backing slot and every right operand (and the
        //: result) the output tile -- the same `merge_o_pair`, left term first, as the combine and
        //: the D128 leaf.
        auto boundary = [&](int jb) {
            //: Segment jb's (L, M) from the ring, now that its last PV has retired.
            bar_wait(&segment_stats[jb % SEG_RING], (jb / SEG_RING) & 1);
            Lseg = seg_lm[jb % SEG_RING][0][row];
            Mseg = seg_lm[jb % SEG_RING][1][row];
            stL.set(depth, Lseg);
            stM.set(depth, Mseg == -CUDART_INF_F ? M_EMPTY : Mseg);
            ++depth;
            #pragma unroll 1
            for (int m = 0; m < sp.mrg_n[jb]; ++m) {
                const float la = stL.get(depth - 2), lb = stL.get(depth - 1);
                const MergeScales ms = merge_scales(stM.get(depth - 2), stM.get(depth - 1), scale_log2);
                const int left = sp.mrg_l[jb][m];
                if (!is_backing(left) || sp.mrg_r[jb][m] != 0) __trap();
                merge_from_node256<32>(backing_of(backing_index(left)), row, tmem + OUTPUT_COLUMN,
                                       ms.sa, ms.sb);
                stL.set(depth - 2, merge_l(la, lb, ms.sa, ms.sb));
                stM.set(depth - 2, ms.m);
                --depth;
            }
            asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
        };
        int g = 0;
#endif /* GQA_D256_FOLD */
        #pragma unroll 1
#ifndef GQA_D256_FOLD
        for (int g = 0; g < total; ++g) {
            const int slot = g % KV, phase = (g / KV) & 1;
            bar_wait(&statistics_ready[slot], phase);
            const float alpha = row_alpha[slot][row];
            //: The HD128 leaf's warp ballot, taken here instead of in the softmax warpgroup. Two
            //: skips, both bit-neutral: the segment's first tile has nothing to rescale because its
            //: PV overwrites, and a warp whose every lane kept its running max (`alpha == 1` for all
            //: 32, the common case at RESCALE_THRESHOLD = 8) would write back exactly the bits it
            //: loaded. When the warp does run, `correction256` still skips the multiply per lane, so
            //: a kept lane's denormal accumulator elements are not flushed by a multiply by 1.0.
            //: This whole decision is EXECUTION: it changes only whether 16 TMEM instructions issue.
            const bool rescale = g != 0 && __any_sync(0xffffffffu, alpha != 1.0f);
            if (g != 0) bar_wait(&pv_done[(g - 1) % KV], ((g - 1) / KV) & 1);
#else /* GQA_D256_FOLD */
        for (int j = 0; j < sp.nsegs; ++j) {
            const int upd = sp.upd[j];
            #pragma unroll 1
            for (int u = 0; u < upd; ++u, ++g) {
                const int slot = g % KV, phase = (g / KV) & 1;
                bar_wait(&statistics_ready[slot], phase);
                const float alpha = row_alpha[slot][row];
                //: The HD128 leaf's warp ballot. Two bit-neutral skips: a segment's first tile has
                //: nothing to rescale (its PV overwrites), and a warp whose every lane kept its max
                //: would write back exactly the bits it loaded; `correction256` skips per lane.
                const bool rescale = u != 0 && __any_sync(0xffffffffu, alpha != 1.0f);
                if (g != 0) bar_wait(&pv_done[(g - 1) % KV], ((g - 1) / KV) & 1);
                asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                if (u == 0 && j > 0) {
                    //: PV(g - 1) was segment j - 1's last: its boundary, then the spill of the node it
                    //: left in the output tile, BEFORE this tile's `output_ready` arrival -- which is
                    //: what releases PV(first of j), the overwrite, in BOTH CTAs (count 8).
                    boundary(j - 1);
                    if (sp.spill_src[j] != NO_SPILL)
                        spill_node256<32>(backing_of(sp.spill_dst[j]), row, tmem + OUTPUT_COLUMN);
                }
                if (rescale) correction256(tmem + OUTPUT_COLUMN, alpha);
                asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
                arrive_warp_cluster(&output_ready);
            }
        }
        //: The last segment's boundary after the item's LAST PV, which every correction thread waits
        //: itself (each generation once, in order); the root is then final in the output tile.
        {
            const int last = total - 1;
            bar_wait(&pv_done[last % KV], (last / KV) & 1);
#endif /* GQA_D256_FOLD */
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
#ifndef GQA_D256_FOLD
            if (rescale) correction256(tmem + OUTPUT_COLUMN, alpha);
            asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
            arrive_warp_cluster(&output_ready);
#else /* GQA_D256_FOLD */
            boundary(sp.nsegs - 1);
        }
        const float root_l = stL.v[0];
        const float root_m = stM.v[0];
        row_inverse[row] = root_inverse(root_l);
        {
            const int pair_row = rank * CTA_ROWS + row;
            const int token = sc.tok0 + pair_row / RATIO;
            const int border = sc.used - sc.q_end + token + 1;
            if (token < sc.q_end && border > sc.seg_begin) {
                const int head = sc.kv_head * RATIO + (row % RATIO);
                const int slot_base = sc.slot_base;
                if (slot_base < 0) {
                    p.lse[int64_t(token) * p.h_q + head] = root_lse(root_l, root_m, scale_log2);
                } else {
                    const int64_t slot = int64_t(slot_base) + pair_row;
                    p.partial_l[slot] = root_l;
                    p.partial_m[slot] = root_m;
                }
            }
#endif /* GQA_D256_FOLD */
        }
    }

#ifndef GQA_D256_FOLD
    //: The item's last PV has not been waited by anyone yet: the correction loop waits PV(g-1), so
    //: PV(total-1) is outstanding. An idle role may reach this point early and must NOT guess a
    //: future parity, so exactly one thread that has observed every preceding generation waits.
    if (thread == CTA_ROWS) {
        const int last = sc.total - 1;
        bar_wait(&pv_done[last % KV], (last / KV) & 1);
    }
#endif /* ! GQA_D256_FOLD */
    synchronize_pair();

    //: The epilogue. One correction thread owns one query row and ALL 256 of its features (the pair
    //: splits N across the CTAs' operands, not across their TMEM lanes), so the fp32 accumulator
    //: goes straight from TMEM registers to global with no shared staging and no cross-CTA exchange.
    if (thread >= CTA_ROWS && thread < BASE) {
        const int pair_row = rank * CTA_ROWS + (thread - CTA_ROWS);
        const int q_end = sc.q_end, first_begin = sc.seg_begin;
        const int token = sc.tok0 + pair_row / RATIO;
        const int head = sc.kv_head * RATIO + (pair_row % RATIO);
        const int border = sc.used - q_end + token + 1;
        //: The SAME predicate the lse and the (L, M) publication use: a token with no keys at all in
        //: this item owns nothing here, and storing it would publish an uninitialised accumulator.
        const bool item_live = token < q_end && border > first_begin;
        const int slot_base = sc.slot_base;

        if (slot_base < 0) {
            // A whole-tree item (n == 1): the root of this row's canonical tree is in TMEM, so the
            // single rounding happens here and nothing is published.
            const float inv = row_inverse[thread - CTA_ROWS];
            auto* out = static_cast<__nv_bfloat16*>(p.out);
            __nv_bfloat16* const destination = out + (int64_t(token) * p.h_q + head) * DIM256;
            #pragma unroll 1
            for (int col = 0; col < DIM256; col += 16) {
                float values[16];
                load16(tmem + OUTPUT_COLUMN + col, values);
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                #pragma unroll
                for (int block = 0; block < 2; ++block) {
                    uint4 line;
                    uint32_t* words = reinterpret_cast<uint32_t*>(&line);
                    #pragma unroll
                    for (int jj = 0; jj < 4; ++jj) {
                        float lo = values[block * 8 + 2 * jj], hi = values[block * 8 + 2 * jj + 1];
                        multiply_pair(lo, hi, inv);
                        words[jj] = pack_bf16_pair(lo, hi);
                    }
                    if (item_live)
                        *reinterpret_cast<uint4*>(destination + col + block * 8) = line;
                }
            }
        } else {
            float* const destination = p.partial_o + (int64_t(slot_base) + pair_row) * DIM256;
            #pragma unroll 1
            for (int col = 0; col < DIM256; col += 16) {
                float values[16];
                load16(tmem + OUTPUT_COLUMN + col, values);
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                if (item_live) {
                    #pragma unroll
                    for (int block = 0; block < 4; ++block)
                        *reinterpret_cast<float4*>(destination + col + block * 4) =
                            *reinterpret_cast<const float4*>(values + block * 4);
                }
            }
        }
        asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    }

    synchronize_pair();
    if (thread < 32) {
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;" ::: "memory");
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0,%1;"
                     :: "r"(tmem), "r"(TMEM_COLUMNS_256) : "memory");
    }
}

// --- launcher ------------------------------------------------------------------------------------

namespace {

//: `setmaxnreg` draws from what ptxas ALLOCATED for the CTA, not from the 512-register file: the
//: budgets of the warps resident on one sub-partition must sum to at most `numRegs` times the warps
//: on it. A sum that exceeds it passes ptxas with zero spills and then hangs the device, because the
//: increase waits for registers no warp will release (observed at D128). At 384 threads
//: warp `w` lands on sub-partition `w % 4`, so every sub-partition holds exactly one softmax warp
//: (0-3), one correction warp (4-7) and one producer warp (8-11), and the sum to satisfy is
//: `3 * numRegs >= 256 + 160 + 64`. ptxas currently declares 168, so 504 >= 480; a future change
//: that makes it declare less must fail the launch with a message rather than spin.
template <int RATIO>
cudaError_t check_register_budget_d256() {
    cudaFuncAttributes attributes{};
    const cudaError_t status = cudaFuncGetAttributes(&attributes, gqa_leaf_d256<RATIO>);
    if (status != cudaSuccess) return status;
    if (3 * attributes.numRegs
            < SOFTMAX_REGISTERS_256 + CORRECTION_REGISTERS_256 + OTHER_REGISTERS_256)
        return cudaErrorLaunchOutOfResources;
    return cudaSuccess;
}

template <int RATIO>
cudaError_t launch_one_d256(const LeafParams& p, const CUtensorMap& mq, const CUtensorMap& mk,
                            const CUtensorMap& mv, int item_grid, cudaStream_t stream) {
    cudaError_t status = check_register_budget_d256<RATIO>();
    if (status != cudaSuccess) return status;
    status = cudaFuncSetAttribute(gqa_leaf_d256<RATIO>,
                                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                                              int(sizeof(LeafShared256)));
    if (status != cudaSuccess) return status;
    cudaLaunchConfig_t config{};
    //: One CTA PAIR per item, so the grid is twice the item capacity and is a multiple of two by
    //: construction. A CTA past the live item count exits before it initialises a barrier.
    config.gridDim = dim3(2 * item_grid, 1, 1);
    config.blockDim = dim3(THREADS_256, 1, 1);
    config.dynamicSmemBytes = sizeof(LeafShared256);
    config.stream = stream;
    cudaLaunchAttribute attribute{};
    attribute.id = cudaLaunchAttributeClusterDimension;
    attribute.val.clusterDim.x = 2;
    attribute.val.clusterDim.y = 1;
    attribute.val.clusterDim.z = 1;
    config.attrs = &attribute;
    config.numAttrs = 1;
    status = cudaLaunchKernelEx(&config, gqa_leaf_d256<RATIO>, mq, mk, mv, p);
    if (status != cudaSuccess) return status;
    return cudaGetLastError();
}

}  // namespace

cudaError_t launch_leaf_d256(const LeafParams& p, const CUtensorMap& mq, const CUtensorMap& mk,
                             const CUtensorMap& mv, int head_ratio, int item_grid,
                             cudaStream_t stream) {
    if (item_grid <= 0) return cudaSuccess;
    #define GQA_LEAF256(R) return launch_one_d256<R>(p, mq, mk, mv, item_grid, stream)
    switch (head_ratio) {
        case 1: GQA_LEAF256(1);
        case 2: GQA_LEAF256(2);
        case 4: GQA_LEAF256(4);
        case 8: GQA_LEAF256(8);
        case 16: GQA_LEAF256(16);
        default: return cudaErrorInvalidValue;
    }
    #undef GQA_LEAF256
}

}  // namespace GQA_D256_NS

#ifdef GQA_D256_FOLD
cudaError_t launch_leaf_d256_fold(const LeafParams& p, const CUtensorMap& mq, const CUtensorMap& mk,
                                  const CUtensorMap& mv, int head_ratio, int item_grid,
                                  cudaStream_t stream) {
    return d256_fold::launch_leaf_d256(p, mq, mk, mv, head_ratio, item_grid, stream);
}
#else
cudaError_t launch_leaf_d256(const LeafParams& p, const CUtensorMap& mq, const CUtensorMap& mk,
                             const CUtensorMap& mv, int head_ratio, int item_grid,
                             cudaStream_t stream) {
    return d256_plain::launch_leaf_d256(p, mq, mk, mv, head_ratio, item_grid, stream);
}
#endif

}  // namespace gqa_sm100
