// SPDX-License-Identifier: BSD-3-Clause
// The HD128 leaf: one fixed KV segment run of one packed query tile, position-anchored.
//
// Execution derived from FlashAttention-4's SM100 forward (flash_attn/cute/flash_fwd_sm100.py: the
// softmax / correction / MMA / load warp roles, `softmax_step`, `correction_loop`, the per-role
// `setmaxnreg` budgets) and from TileFoundry's CUDA port of it (the two-wave issue order, the
// prologue, the staged epilogue transpose). Copyright (c) 2025, Tri Dao et al., BSD-3-Clause.
//
// What is NOT theirs: the work item, the segment anchoring at key 0, varlen queries, the paged KV
// address function, the lazy-rescale running max in the recipe's form, the in-CTA fold of a run of
// segments over the canonical tree, and the (O, L, M) partial the combine merges.
// See ../README.md for the numerical contract.
//
// STRUCTURE (README.md section 2). `128 * WAVES + 256` threads in FOUR roles:
//
//   softmax     threads [0, 128 * WAVES): thread owns TMEM lane `row` of its wave. Per tile it waits
//               `qk_done` and NOTHING ELSE: S, mask, max, the lazy-rescale decision, P stores and
//               the `probabilities` arrivals, `alpha` into the stats lane, the row sum, and at a
//               segment's last tile (L, M) into the stats lane.
//   correction  threads [128 * WAVES, 128 * WAVES + 128): thread owns row `row` of EVERY wave.
//               The O rescale, the "O tile ready" arrival the MMA waits on, the per-row (L, M)
//               merge stack, every spill and merge of the fold, and the epilogue.
//   producer    four warps: the MMA issuer thread, an idle warp, the TMA submitter thread and the
//               completion-publisher warp.
//
// The handoff is FA4's: the softmax publishes through the score slot's free columns 64-66 (the
// "stats lane") and two mbarriers, `stats_ready[wave][g % KV]` and `segment_stats[wave][j & 1]`;
// `probabilities[s][0]` takes four softmax and four correction arrivals. The protocol was modelled
// on the CPU before a line of this kernel was written, with the two TMEM stats hazards asserted as
// checks in that model.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

#include "params.h"
#include "numerics.cuh"
#include "tmem.cuh"
#include "launch.h"
#include "plan_skeleton.cuh"

namespace gqa_sm100 {

static constexpr int KV_STAGES = 2;
static constexpr int DIM = 128;

//: Whether the correction warpgroup waits PV(g-1) before it rescales O for tile g: 1 = only when it
//: rescales (the shipping form; at one wave QK(g) is issued BEFORE PV(g-1), so `qk_done(g)` does not
//: imply PV(g-1) retired and the wait is required); 0 = at TWO waves never (`qk_done(w, g)` is a
//: cumulative commit issued after PV(w, g-1) by the same thread, which is FA4's argument), at one
//: wave still when rescaling.
//: The two-wave correction warpgroup's register budget; the producer gets `512 - 2 * 192 - this`.
//: FA4's hd128 value is 72 (producer 56).
//: With `GQA_W2_SOFTMAX_EPILOGUE` the two-wave correction does not write the epilogue,
//: so it gives 8 registers back and the softmax, which does, takes them: 208 / 208 / 56 / 40
//: (4 x 128 = 512), 0 spill on all five ratios. At 200 / 64 / 48 the softmax epilogue spilled 8-12 B.
#ifndef GQA_CORR_REGS_W2
#define GQA_CORR_REGS_W2 56
#endif
#ifndef GQA_SOFT_REGS_W2
#define GQA_SOFT_REGS_W2 208
#endif

//: Where the softmax publishes alpha: 1 = right after the row max (before the exponentials), 0 =
//: after the quarter-1 `wait::st` together with "P quarter 1 stored"; 3 = pinned (below).
#ifndef GQA_EARLY_STATS
#define GQA_EARLY_STATS 3
#endif
//: `GQA_EARLY_STATS 1` asks for the stats arrive right after the
//: max, but ptxas sank it to instruction 461 of the 593-instruction interior body (74 of 112 FFMA2
//: hoisted above it). `3` PINS it between quarter 0 and quarter 1: quarters 1-3's bias is routed
//: through the arrive's own asm, so no exp2 of those quarters can be scheduled above it, and
//: quarter 0's work hides the alpha store's `wait::st`. Bit-neutral (the same value, moved).
//: `GQA_STATS_RELEASE` gives the two stats arrives RELEASE semantics (the d256 leaf's form).
#ifndef GQA_STATS_RELEASE
#define GQA_STATS_RELEASE 1
#endif
//: At TWO waves stream every merge's backing (left) operand through
//: a 16 KiB shared double buffer with `cp.async`, one 16-column chunk ahead, the first two chunks
//: issued before the segment-end `pv_done` wait. Zero registers; bit-neutral (bytes copied).
#ifndef GQA_MERGE_CPASYNC
#define GQA_MERGE_CPASYNC 1
#endif
//: At TWO waves each wave's SOFTMAX warps run their own segment boundary -- the segment-end
//: `pv_done` wait, the merges and the spill -- in parallel across the waves. One correction thread
//: running both waves' boundaries serialises them onto both waves' critical paths: `clock64` stamps
//: put the exposed boundary at ~1800 cycles per tile against ~180 when each wave's softmax runs its
//: own. At two waves the segment-end wait is ~250 cycles (the other wave keeps the tensor pipe busy);
//: at ONE wave the correction keeps the boundary, where it IS off the critical path (the softmax
//: runs two tiles ahead of it). The correction keeps the O rescale at both wave counts.
#ifndef GQA_W2_SOFTMAX_FOLD
#define GQA_W2_SOFTMAX_FOLD 1
#endif
//: `GQA_W2_SOFTMAX_EPILOGUE`: at TWO waves each wave's SOFTMAX warps write their own wave's
//: epilogue, instead of the 128 correction threads writing both waves' tiles one after the other
//: (one serialised tile epilogue costs ~1.0 us).
//: `GQA_PARTIAL_WAVE_INTERIOR`: partly-live waves take the interior body on fully visible tiles
//: (otherwise every tile is masked, ~850 instructions per warp per tile instead of ~590 on every
//: decode tile). Kept OFF: it measured slower on decode rows with disjoint KV.
#ifndef GQA_PARTIAL_WAVE_INTERIOR
#define GQA_PARTIAL_WAVE_INTERIOR 0
#endif
#ifndef GQA_W2_SOFTMAX_EPILOGUE
#define GQA_W2_SOFTMAX_EPILOGUE 1
#endif

#ifndef GQA_CORR_PV_WAIT
#define GQA_CORR_PV_WAIT 1
#endif


//: Per-role register budgets, declared with `setmaxnreg` as the first statement of each role branch.
//: The rule that matters: the budgets of the warps resident on one sub-partition must sum
//: to at most `warps_per_subpartition * numRegs` -- what ptxas ALLOCATED -- never to 512 by
//: assumption, or the increase waits for registers no warp will release and the kernel hangs.
//:
//: TWO waves: 16 warps, 4 per sub-partition, `__launch_bounds__(512, 1)` allocates 128 registers:
//: 192 + 192 + CORR + OTHER = 4 x 128 = 512. FA4's own hd128 causal split
//: (`_TUNING_CONFIG[(False, True, 128, False)]`) is 192 / 72 / 56; this leaf's correction warpgroup
//: also carries two waves' merge stacks and the fold's 16-column merges, and ptxas reads its spill
//: (README.md section 2 records the measured spill-free split).
//: ONE wave: 12 warps, 3 per sub-partition, `__launch_bounds__(384, 1)` allocates 168:
//: 232 + 160 + 112 = 3 x 168 = 504 (the d256 leaf's 256/160/64 pattern at the same 168).
template <int WAVES> __host__ __device__ constexpr int softmax_registers() {
    return WAVES == 2 ? GQA_SOFT_REGS_W2 : 232;
}
template <int WAVES> __host__ __device__ constexpr int correction_registers() {
    return WAVES == 2 ? GQA_CORR_REGS_W2 : 160;
}
template <int WAVES> __host__ __device__ constexpr int other_registers() {
    return WAVES == 2 ? 512 - 2 * GQA_SOFT_REGS_W2 - GQA_CORR_REGS_W2 : 112;
}
//: TMEM columns the correction warpgroup moves per `tcgen05` round trip: 16 under the two-wave
//: budget, 32 under the one-wave one. Chunking moves no bit: every merge, rescale and spill is the
//: same per-element arithmetic on the same elements with the same left operand first.
template <int WAVES> __host__ __device__ constexpr int correction_chunk() {
    return WAVES == 2 ? 16 : 32;
}

//: One elected lane arrives (relaxed or release, `GQA_STATS_RELEASE`) and the asm returns `x`
//: unchanged: whatever consumes the result cannot be scheduled above the arrive.
__device__ __forceinline__ float arrive_warp_tied(uint64_t* barrier, float x) {
    float y;
#if GQA_STATS_RELEASE
    asm volatile("{ .reg .pred e; elect.sync _|e,0xffffffff; "
                 "@e mbarrier.arrive.release.cta.shared::cta.b64 _,[%1]; mov.b32 %0,%2; }"
                 : "=f"(y) : "r"(shared_address(barrier)), "f"(x) : "memory");
#else
    asm volatile("{ .reg .pred e; elect.sync _|e,0xffffffff; "
                 "@e mbarrier.arrive.relaxed.cta.shared::cta.b64 _,[%1]; mov.b32 %0,%2; }"
                 : "=f"(y) : "r"(shared_address(barrier)), "f"(x) : "memory");
#endif
    return y;
}
__device__ __forceinline__ void arrive_warp_stats(uint64_t* barrier) {
#if GQA_STATS_RELEASE
    asm volatile("{ .reg .pred e; elect.sync _|e,0xffffffff; "
                 "@e mbarrier.arrive.release.cta.shared::cta.b64 _,[%0]; }"
                 :: "r"(shared_address(barrier)) : "memory");
#else
    arrive_warp_local(barrier);
#endif
}
//: `merge_o_pair` with the result written into the RIGHT pair: the same two products, the same
//: `add.rn.ftz.f32x2` with the LEFT product first -- only the destination register differs.
__device__ __forceinline__ void merge_o_pair_into_right(float alo, float ahi, float& blo, float& bhi,
                                                        float sa, float sb) {
    FloatPair a = {alo, ahi};
    FloatPair b = {blo, bhi};
    multiply_pair(a.lo, a.hi, sa);
    multiply_pair(b.lo, b.hi, sb);
    const FloatPair r = add_pair(a, b);
    blo = r.lo;
    bhi = r.hi;
}
//: The two-wave SOFTMAX boundary's merge (`GQA_W2_SOFTMAX_FOLD`): the backing left operand 64
//: columns per global round trip (the 128 score registers are dead at a segment end), then four
//: 16-column TMEM read-modify-writes each. The same `merge_o_pair` arithmetic over the same
//: elements, left term first, in ascending column order, as every other merge in this file.
__device__ __forceinline__ void merge_from_node_half(const float* node, int row, uint32_t right,
                                                     float sa, float sb) {
    #pragma unroll 1
    for (int half = 0; half < 2; ++half) {
        float a[64];
        #pragma unroll
        for (int j = 0; j < 64; ++j) a[j] = node[(half * 64 + j) * 128 + row];
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            float b[16];
            load16(right + half * 64 + c * 16, b);
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            #pragma unroll
            for (int j = 0; j < 16; j += 2)
                merge_o_pair_into_right(a[c * 16 + j], a[c * 16 + j + 1], b[j], b[j + 1], sa, sb);
            store16(right + half * 64 + c * 16, b);
        }
    }
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}
//: The two-wave softmax's spill: two 64-column halves, two `tcgen05.ld` under each `wait::ld`.
__device__ __forceinline__ void spill_node_halves(float* node, int row, uint32_t source) {
    #pragma unroll 1
    for (int half = 0; half < 2; ++half) {
        float v0[32], v1[32];
        load32(source + half * 64, v0);
        load32(source + half * 64 + 32, v1);
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        #pragma unroll
        for (int j = 0; j < 32; ++j) node[(half * 64 + j) * 128 + row] = v0[j];
        #pragma unroll
        for (int j = 0; j < 32; ++j) node[(half * 64 + 32 + j) * 128 + row] = v1[j];
    }
}

__device__ __forceinline__ void cp_async4(uint32_t smem, const float* global) {
    asm volatile("cp.async.ca.shared.global [%0],[%1],4;" :: "r"(smem), "l"(global) : "memory");
}

template <int WAVES> __host__ __device__ constexpr int node_floats() {
    return WAVES == 1 ? 128 * 128 : 1;
}

//: Operands, plus at ONE wave the fold's 64 KiB shared node slot (backing slot 0). Feature-major
//: (`node[feature * 128 + row]`): one thread owns one row and touches only its own, so the slot
//: needs no barrier and no fence, and a warp's 32 lanes touch 32 consecutive floats of a feature.
template <int WAVES>
struct alignas(1024) LeafShared {
    uint16_t q[WAVES][128 * 128];
    uint16_t k[KV_STAGES][128 * 128];
    uint16_t v[KV_STAGES][128 * 128];
    float node[node_floats<WAVES>()];
};
static_assert(sizeof(LeafShared<1>) == 5 * 32768 + 65536,
              "one wave: 160 KiB of operands plus the 64 KiB shared node slot");

//: TMEM: two score slots at 0 and 128 (bf16 P aliases columns 0-63 of its own slot; columns 64-66
//: are the slot's STATS LANE), two fp32 output tiles at 256 and 384 -- the same map at both wave
//: counts. At two waves a score slot and an output tile belong to a WAVE; at one wave the score slot
//: belongs to the KV tile (`g & 1`) and both output tiles are the worker's merge stack.
__host__ __device__ constexpr int score_column(int slot) { return slot * 128; }
static constexpr int ALPHA_COLUMN = 64;          // this tile's alpha
static constexpr int STATS_COLUMN = 65;          // (L, M) at a segment's last tile
template <int WAVES> __host__ __device__ constexpr int tmem_columns() { return 512; }
static_assert(2 * 128 + 2 * DIM == 512, "two score slots and two fp32 output slots fill TMEM");

//: The key slot of logical key `key_base` of request `req` in the flattened cache.
__device__ __forceinline__ int key_slot(const LeafParams& p, int req, int key_base) {
    if (p.page > 0) {
        const int block = p.block_table[int64_t(req) * p.block_table_stride + key_base / p.page];
        return block * p.page + (key_base % p.page);
    }
    return p.cu_seqlens_k[req] + key_base;
}

template <int WAVES> __host__ __device__ __forceinline__ int slot_column(int slot, int wave) {
    return 256 + (WAVES == 1 ? slot : wave) * DIM;
}
template <int WAVES> __host__ __device__ __forceinline__ constexpr int tmem_slots() {
    return WAVES == 1 ? 2 : 1;
}
template <int WAVES> __host__ __device__ __forceinline__ constexpr int shared_backing() {
    return WAVES == 1 ? 1 : 0;
}
static_assert(fold_backing_slots(2, 4) == 1, "one wave at run 4: the shared node slot alone");
static_assert(fold_backing_slots(2, 16) == 3, "one wave at run 16: shared plus two global");
static_assert(fold_backing_slots(1, 16) == 4, "two waves at run 16: four global");
static_assert(fold_backing_slots(1, 1) == 0 && fold_backing_slots(2, 2) == 0,
              "a run the TMEM tiles cover needs no backing slot at all");
static_assert(FOLD_BACKING >= fold_backing_slots(1, FOLD_MAX_RUN), "FOLD_BACKING covers the cap");

//: A depth-indexed float stack that stays in REGISTERS: a runtime index into a plain array puts it
//: on the stack. Compile-time selection costs a few predicated moves and is bit-identical.
template <int N> struct RegStack {
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

//: Call `f(std::integral_constant<int, w>)` for every wave, so per-wave arrays are indexed by a
//: compile-time constant and stay in registers.
template <int WAVES, class F> __device__ __forceinline__ void for_waves(F&& f) {
    f(std::integral_constant<int, 0>{});
    if constexpr (WAVES > 1) f(std::integral_constant<int, 1>{});
}

//: `O_left <- O_left * sa + O_right * sb` over all DIM columns of two TMEM output tiles, in C-column
//: read-modify-writes. Exactly the merge operator's per-element arithmetic, left term first.
template <int C>
__device__ __forceinline__ void tmem_merge(uint32_t left, uint32_t right, float sa, float sb) {
    #pragma unroll 1
    for (int chunk = 0; chunk < DIM / C; ++chunk) {
        float a[C], b[C];
        tload<C>(left + chunk * C, a);
        tload<C>(right + chunk * C, b);
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        #pragma unroll
        for (int j = 0; j < C; j += 2)
            merge_o_pair(a[j], a[j + 1], b[j], b[j + 1], sa, sb);
        tstore<C>(left + chunk * C, a);
    }
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}

//: Move a pending node out of TMEM into a BACKING slot (shared at one wave, global otherwise),
//: freeing its TMEM tile for the next segment. Thread-local in both directions: no barrier, no
//: fence. Runs in the CORRECTION warpgroup, off the softmax critical path, in C-column
//: chunks (the two-wave correction budget does not hold 128 floats).
template <int C>
__device__ __forceinline__ void spill_node(float* node, int row, uint32_t source) {
    #pragma unroll 1
    for (int chunk = 0; chunk < DIM / C; ++chunk) {
        float values[C];
        tload<C>(source + chunk * C, values);
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        #pragma unroll
        for (int j = 0; j < C; ++j) node[(chunk * C + j) * 128 + row] = values[j];
    }
}

//: `O_right <- O_left * sa + O_right * sb` with the LEFT operand read out of a BACKING slot: the
//: same `merge_o_pair` over the same elements with the same left term first as `tmem_merge`; only
//: where the left operand is read from and where the result lands differ. Each chunk's backing loads
//: are issued before its `tcgen05.ld`, so the two latencies overlap. Four (or eight) round trips
//: per merge, but OFF the softmax critical path, overlapped with the next segment's tiles, which is
//: the point.
template <int C>
__device__ __forceinline__ void merge_from_node(const float* node, int row, uint32_t right,
                                                float sa, float sb) {
    #pragma unroll 1
    for (int chunk = 0; chunk < DIM / C; ++chunk) {
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

//: The item's segment plan, the skeleton table and `build_segment_plan` live in plan_skeleton.cuh,
//: shared with the D256 leaf.


template <int WAVES, int RATIO>
__global__ __launch_bounds__(128 * WAVES + 256, 1) void gqa_leaf_d128(
        const __grid_constant__ CUtensorMap mapq,
        const __grid_constant__ CUtensorMap mapk,
        const __grid_constant__ CUtensorMap mapv,
        const __grid_constant__ LeafParams p) {
    constexpr int KV = KV_STAGES;
    constexpr int BASE = 128 * WAVES;                 // softmax threads [0, BASE)
    constexpr int CORR = BASE;                        // correction threads [CORR, CORR + 128)
    constexpr int PROD = BASE + 128;                  // producer warpgroup
    constexpr int MMA_THREAD = PROD, SUBMIT_THREAD = PROD + 64, PUBLISH_THREAD = PROD + 96;
    constexpr int WAVE_TOKENS = 128 / RATIO;
    constexpr int TILE_TOKENS = (128 * WAVES) / RATIO;
    constexpr int CC = correction_chunk<WAVES>();
    static_assert(128 % RATIO == 0, "a packed 128-row tile must hold whole tokens");

    const int thread = threadIdx.x;
    //: The item and the live count are loaded TOGETHER (the grid is the capacity, so the index is
    //: always inside the allocation); behind the test they would serialise.
    const Item it = p.items[blockIdx.x];
    if (int(blockIdx.x) >= p.counts[NUM_ITEMS]) return;

    const int req = it.req, tok0 = it.tok0, kv_head = it.kv_head;
    const int q_end = it.q_end;
    const int used = it.used;
    const int token_last = min(tok0 + TILE_TOKENS - 1, q_end - 1);
    const int border_max = used - q_end + token_last + 1;

    extern __shared__ __align__(1024) unsigned char storage[];
    __shared__ SegmentPlan sp;
    __shared__ int key_slots[MAX_KEY_SLOTS];
    auto& s = *reinterpret_cast<LeafShared<WAVES>*>(storage);
    __shared__ __align__(8) uint64_t loaded_q[WAVES], loaded_k[KV], loaded_v[KV];
    __shared__ __align__(8) uint64_t keys_ready[KV], values_ready[KV];
    __shared__ __align__(8) uint64_t qk_done[WAVES][KV], pv_done[WAVES][KV];
    //: Indexed by SCORE SLOT (the wave at two waves, the KV tile's parity at one). [0] takes four
    //: softmax arrivals ("P quarter 1 stored") and four correction arrivals ("O tile ready"); [1]
    //: takes four softmax arrivals ("P quarter 3 stored").
    __shared__ __align__(8) uint64_t probabilities[2][2];
    //: softmax -> correction. `stats_ready[w][g % KV]`: this tile's alpha is in column 64
    //: of the score slot (phase `(g / KV) & 1`). `segment_stats[w][j & 1]`: segment j's (L, M) is in
    //: columns 65-66 (phase `(j >> 1) & 1`) -- alternating by SEGMENT parity, because at one wave
    //: the softmax can finish the next segment's first tile before the correction has consumed this
    //: segment's stats, and indexed by score slot a one-tile segment would put two generations on
    //: one barrier and deadlock.
    __shared__ __align__(8) uint64_t stats_ready[WAVES][KV], segment_stats[WAVES][2];
    __shared__ __align__(8) uint64_t query_ready[WAVES];
    __shared__ uint32_t tmem_slot;
    //: The softmax's per-wave geometry, in SHARED memory, read per segment. As registers these are
    //: live across the whole tile loop beside 128 fp32 scores, and at the two-wave 192-register
    //: budget ptxas spilled seven of them (the d256 leaf's `ItemScalars` is the same fix for the
    //: same reason: an `LDS` per segment where a spill costs an `LDL`).
    __shared__ int sc_w_border[WAVES], sc_wave_tok0[WAVES], sc_q_end, sc_k0;
    //: The correction warpgroup's per-row (L, M) merge stacks at TWO waves, in shared memory:
    //: `[L|M][wave][depth][row]`, 10 KiB of the 35 KiB two waves leave free, conflict-free (a
    //: warp's 32 lanes touch 32 consecutive floats). In registers they cost the correction 20 of its
    //: registers, which the softmax's 128 fp32 scores need more (README.md section 2). Row-private:
    //: no barrier, no fence. At one wave there is no room and the 160-register budget holds them.
    __shared__ float corr_stack[WAVES == 2 ? 2 * WAVES * FOLD_STACK * 128 : 1];
    //: The two-wave merge's left-operand double buffer (`GQA_MERGE_CPASYNC`): [buffer][column 16][row].
    __shared__ __align__(16) float merge_ring[(WAVES == 2 && GQA_MERGE_CPASYNC && !GQA_W2_SOFTMAX_FOLD) ? 2 * 16 * 128 : 1];

    //: THE PROLOGUE (a serial prologue put the first `qk_done` ~3 us after entry). Three things run
    //: in parallel in front of the first CTA barrier, none of them behind
    //: another: warp 0 builds the segment plan; thread 32 initialises every barrier but the three
    //: the submitter arms first; and the SUBMITTER initialises those three itself and issues Q0, K0
    //: (its slot rides in the Item, so no `block_table` round trip) and Q1 at once -- FA4's order,
    //: with the copies in flight while the rest of the CTA is still setting up. K1 is issued by the
    //: submitter's steady loop, whose slot cache the publisher fills after the first barrier.
    if (thread < 32)
        build_segment_plan(sp, it.k0, it.k1, p.seg, border_max, tmem_slots<WAVES>(),
                           shared_backing<WAVES>() + p.fold_globals);
    if (thread == SUBMIT_THREAD) {
        asm volatile("prefetch.tensormap [%0];" :: "l"(reinterpret_cast<uint64_t>(&mapq)) : "memory");
        asm volatile("prefetch.tensormap [%0];" :: "l"(reinterpret_cast<uint64_t>(&mapk)) : "memory");
        asm volatile("prefetch.tensormap [%0];" :: "l"(reinterpret_cast<uint64_t>(&mapv)) : "memory");
        #pragma unroll
        for (int i = 0; i < WAVES; ++i)
            asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" :: "r"(shared_address(&loaded_q[i])));
        asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" :: "r"(shared_address(&loaded_k[0])));
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"
                     :: "r"(shared_address(&loaded_q[0])), "r"(2 * 16384) : "memory");
        #pragma unroll
        for (int chunk = 0; chunk < 2; ++chunk)
            load_tile(mapq, s.q[0] + chunk * 128 * 64, &loaded_q[0], chunk, kv_head * RATIO, tok0);
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"
                     :: "r"(shared_address(&loaded_k[0])), "r"(32768) : "memory");
        //: The plan resolved this item's first key slot when it was free (a row's first node);
        //: otherwise (`ks0 < 0`) it is looked up here, behind Q0's copy.
        const int ks0 = it.ks0 >= 0 ? it.ks0 : key_slot(p, req, it.k0 * p.seg);
        #pragma unroll
        for (int chunk = 0; chunk < 2; ++chunk)
            load_tile(mapk, s.k[0] + chunk * 128 * 64, &loaded_k[0], chunk, kv_head, ks0);
        if constexpr (WAVES > 1) {
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"
                         :: "r"(shared_address(&loaded_q[1])), "r"(2 * 16384) : "memory");
            #pragma unroll
            for (int chunk = 0; chunk < 2; ++chunk)
                load_tile(mapq, s.q[1] + chunk * 128 * 64, &loaded_q[1], chunk,
                          kv_head * RATIO, tok0 + WAVE_TOKENS);
        }
    }
    if (thread == 32) {
        #pragma unroll
        for (int w = 0; w < WAVES; ++w) {
            sc_wave_tok0[w] = tok0 + w * WAVE_TOKENS;
            sc_w_border[w] = used - q_end + tok0 + w * WAVE_TOKENS + 1;
        }
        sc_q_end = q_end;
        sc_k0 = it.k0;
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0],8;" :: "r"(shared_address(&probabilities[i][0])));
            asm volatile("mbarrier.init.shared::cta.b64 [%0],4;" :: "r"(shared_address(&probabilities[i][1])));
        }
        #pragma unroll
        for (int i = 0; i < WAVES; ++i) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" :: "r"(shared_address(&query_ready[i])));
            #pragma unroll
            for (int j = 0; j < KV; ++j) {
                asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" :: "r"(shared_address(&qk_done[i][j])));
                asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" :: "r"(shared_address(&pv_done[i][j])));
                // One elected arrival per softmax warp of the wave.
                asm volatile("mbarrier.init.shared::cta.b64 [%0],4;" :: "r"(shared_address(&stats_ready[i][j])));
                asm volatile("mbarrier.init.shared::cta.b64 [%0],4;" :: "r"(shared_address(&segment_stats[i][j])));
            }
        }
        #pragma unroll
        for (int i = 0; i < KV; ++i) {
            if (i > 0)
                asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" :: "r"(shared_address(&loaded_k[i])));
            asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" :: "r"(shared_address(&loaded_v[i])));
            asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" :: "r"(shared_address(&keys_ready[i])));
            asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" :: "r"(shared_address(&values_ready[i])));
        }
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    synchronize_cta();

    auto key_base_of = [&](int g) {
        int j = 0;
        while (j + 1 < sp.nsegs && g >= sp.pre[j + 1]) ++j;
        return (it.k0 + j) * p.seg + (g - sp.pre[j]) * KEY_TILE;
    };

    const int total = sp.total;
    const bool slots_cached = total <= MAX_KEY_SLOTS;
    if (slots_cached && thread >= PUBLISH_THREAD && thread < PUBLISH_THREAD + 32) {
        for (int g = thread - PUBLISH_THREAD; g < total; g += 32)
            key_slots[g] = key_slot(p, req, key_base_of(g));
        __syncwarp();
    }
    auto slot_of = [&](int g) {
        return slots_cached ? key_slots[g] : key_slot(p, req, key_base_of(g));
    };

    if (thread < 32)
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0],%1;"
                     :: "r"(shared_address(&tmem_slot)), "r"(tmem_columns<WAVES>()) : "memory");
    synchronize_cta();
    const uint32_t tmem = tmem_slot;

    const float scale_log2 = p.scale_log2;
    constexpr uint32_t base_instruction = (1u << 4) | (1u << 7) | (1u << 10) | ((128u / 16) << 24);
    constexpr uint32_t qk_instruction = base_instruction | ((128u / 8) << 17);
    constexpr uint32_t pv_instruction = base_instruction | ((128u / 8) << 17) | (1u << 16);
    static_assert(qk_instruction == 0x08200490u && pv_instruction == 0x08210490u);

    //: THE EPILOGUE of one wave: the correction warpgroup's at one wave, each wave's
    //: SOFTMAX warps' at two (`GQA_W2_SOFTMAX_EPILOGUE`). Its scalars come in as arguments so the
    //: softmax can re-read them after its tile loop instead of holding them across it.
    auto wave_epilogue = [&](auto wc, auto narrow_tag, const int row, const uint32_t root, const float L, const float M,
                             const int slot_base, const int first_begin, const int tok0,
                             const int used, const int q_end, const int kv_head) {
            constexpr int w = decltype(wc)::value;
            //: 16 columns per `wait::ld` from the softmax (its loop is at the register edge), 32 from
            //: the correction.
            constexpr int CH = decltype(narrow_tag)::value ? 16 : 32;
            const int lane = row & 31;
            const int prow = w * 128 + row;                       // packed row of the CTA tile
            const int token = tok0 + prow / RATIO;
            const int border = used - q_end + token + 1;
            const bool item_live = token < q_end && border > first_begin;
            const int warp_base_row = prow & ~31;
            if (slot_base < 0) {
                const float inv = root_inverse(L);
                uint16_t* const stage = reinterpret_cast<uint16_t*>(storage);
                //: Two `tcgen05.ld` per `wait::ld` (32 columns): half the TMEM round trips of the
                //: epilogue.
                #pragma unroll 1
                for (int col = 0; col < DIM; col += CH) {
                    float pair_values[CH / 16][16];
                    #pragma unroll
                    for (int sub = 0; sub < CH / 16; ++sub) load16(root + col + sub * 16, pair_values[sub]);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    #pragma unroll
                    for (int sub = 0; sub < CH / 16; ++sub) {
                    const float (&values)[16] = pair_values[sub];
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
                            *reinterpret_cast<uint4*>(stage + prow * 136 + col + sub * 16 + block * 8) = line;
                    }
                    }
                }
                if (item_live) {
                    const int head = kv_head * RATIO + (prow % RATIO);
                    p.lse[int64_t(token) * p.h_q + head] = root_lse(L, M, scale_log2);
                }
                __syncwarp();
                auto* out = static_cast<__nv_bfloat16*>(p.out);
                #pragma unroll (CH == 16 ? 1 : 4)
                for (int i = 0; i < (32 * DIM) / 256; ++i) {
                    const int flat = warp_base_row * DIM + i * 256 + lane * 8;
                    const int r = flat >> 7, f = flat & 127;
                    const int head = kv_head * RATIO + (r % RATIO);
                    const int t = tok0 + r / RATIO;
                    if (t < q_end && used - q_end + t + 1 > first_begin)
                        *reinterpret_cast<uint4*>(out + (int64_t(t) * p.h_q + head) * DIM + f) =
                            *reinterpret_cast<const uint4*>(stage + r * 136 + f);
                }
            } else {
                if (item_live) {
                    const int64_t slot = int64_t(slot_base) + prow;
                    p.partial_l[slot] = L;
                    p.partial_m[slot] = M;
                }
                float* const stage = reinterpret_cast<float*>(storage);
                #pragma unroll 1
                for (int col = 0; col < DIM; col += CH) {
                    float values[CH / 16][16];
                    #pragma unroll
                    for (int sub = 0; sub < CH / 16; ++sub) load16(root + col + sub * 16, values[sub]);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    #pragma unroll
                    for (int block = 0; block < CH / 4; ++block)
                        if (item_live)
                            *reinterpret_cast<float4*>(stage + prow * 132 + col + block * 4) =
                                make_float4(values[block / 4][(block % 4) * 4],
                                            values[block / 4][(block % 4) * 4 + 1],
                                            values[block / 4][(block % 4) * 4 + 2],
                                            values[block / 4][(block % 4) * 4 + 3]);
                }
                __syncwarp();
                #pragma unroll (CH == 16 ? 1 : 4)
                for (int jj = 0; jj < 32; ++jj) {
                    const int r = warp_base_row + jj;
                    const int t = tok0 + r / RATIO;
                    const int b = used - q_end + t + 1;
                    if (t < q_end && b > first_begin)
                        *reinterpret_cast<float4*>(p.partial_o + (int64_t(slot_base) + r) * DIM + lane * 4) =
                            *reinterpret_cast<const float4*>(stage + r * 132 + lane * 4);
                }
            }
    };

    if (thread >= PROD) {
        // ====================================== PRODUCER ======================================
        //: The budget is the FIRST statement after the last CTA-wide barrier; all four warps of the
        //: warpgroup (MMA, idle, submit, publish) execute it (`setmaxnreg` is warpgroup-aligned).
        registers_decrease<other_registers<WAVES>()>();
        if (thread == SUBMIT_THREAD) {
            // FA4 load order: K0, K1, V0, K2, V1, ..., Vlast, across the whole item. (No L2
            // prefetch: it measured neutral.)
            //: From ordinal 1: K1 (if the item has a second tile) is the loop's first copy.
            for (int ordinal = 1; ordinal < 2 * total; ++ordinal) {
                const bool is_key = (ordinal & 1) != 0 && ordinal < 2 * total - 1;
                const int g = is_key ? (ordinal + 1) / 2 : (ordinal - 1) / 2;
                const int slot = g % KV, cycle = g / KV;
                const int ks = slot_of(g);
                if (is_key) {
                    if (g >= KV) {
                        #pragma unroll
                        for (int w = 0; w < WAVES; ++w) bar_wait(&qk_done[w][slot], (cycle - 1) & 1);
                    }
                    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"
                                 :: "r"(shared_address(&loaded_k[slot])), "r"(32768) : "memory");
                    #pragma unroll
                    for (int chunk = 0; chunk < 2; ++chunk)
                        load_tile(mapk, s.k[slot] + chunk * 128 * 64, &loaded_k[slot], chunk, kv_head, ks);
                } else {
                    if (g >= KV) {
                        #pragma unroll
                        for (int w = 0; w < WAVES; ++w) bar_wait(&pv_done[w][slot], (cycle - 1) & 1);
                    }
                    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"
                                 :: "r"(shared_address(&loaded_v[slot])), "r"(32768) : "memory");
                    #pragma unroll
                    for (int chunk = 0; chunk < 2; ++chunk)
                        load_tile(mapv, s.v[slot] + chunk * 128 * 64, &loaded_v[slot], chunk, kv_head, ks);
                }
            }
        } else if (thread >= PUBLISH_THREAD && thread < PUBLISH_THREAD + 32) {
            const int lane = thread - PUBLISH_THREAD;
            // Fixed publication order K0, K1, V0, K2, V1, ...: values_ready[slot] for tile g
            // transitively implies K(g+1) has landed.
            if (lane == 0) {
                bar_wait(&loaded_q[0], 0);
                arrive_local_release(&query_ready[0]);
                bar_wait(&loaded_k[0], 0);
                arrive_local_release(&keys_ready[0]);
                if constexpr (WAVES > 1) {
                    bar_wait(&loaded_q[1], 0);
                    arrive_local_release(&query_ready[1]);
                }
            }
            for (int ordinal = 1; ordinal < 2 * total; ++ordinal) {
                const bool is_key = (ordinal & 1) != 0 && ordinal < 2 * total - 1;
                const int g = is_key ? (ordinal + 1) / 2 : (ordinal - 1) / 2;
                const int slot = g % KV, phase = (g / KV) & 1;
                if (is_key) {
                    if (lane == 0) {
                        bar_wait(&loaded_k[slot], phase);
                        arrive_local_release(&keys_ready[slot]);
                    }
                    continue;
                }
                if (lane == 0) bar_wait(&loaded_v[slot], phase);
                __syncwarp();
                // Keys at or past seqused_k are unwritten cache slots: 0 * NaN is NaN, so their
                // value rows are zeroed before the PV MMA can read them.
                const int valid = min(KEY_TILE, max(0, used - key_base_of(g)));
                if (valid < KEY_TILE) {
                    const int words = (KEY_TILE - valid) * 64;
                    #pragma unroll 1
                    for (int chunk = 0; chunk < 2; ++chunk) {
                        uint16_t* rows = s.v[slot] + chunk * 128 * 64 + valid * 64;
                        for (int i = lane * 8; i < words; i += 32 * 8)
                            *reinterpret_cast<uint4*>(rows + i) = make_uint4(0, 0, 0, 0);
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                __syncwarp();
                if (lane == 0) arrive_local_release(&values_ready[slot]);
            }
        } else if (thread == MMA_THREAD) {
          if constexpr (WAVES == 1) {
            //: ONE WAVE, TWO SCORE SLOTS: `QK(g+1)` is issued BEFORE `PV(g)`, into the other slot
            //: (README.md section 2). Strip-mined by KV.
            bar_wait(&query_ready[0], 0);
            {
                bar_wait(&keys_ready[0], 0);
                asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                qk_sequence128(tmem + uint32_t(score_column(0)), descriptor(s.q[0], false),
                               descriptor(s.k[0], false), qk_instruction);
                commit_one(&qk_done[0][0]);
            }
            int j = 0, u = 0;
            uint32_t out = tmem + uint32_t(slot_column<WAVES>(sp.comp[0], 0));
            for (int base = 0; base < total; base += KV) {
                const int cycle = (base / KV) & 1;
                #pragma unroll
                for (int slot = 0; slot < KV; ++slot) {
                    const int g = base + slot;
                    if (g >= total) break;
                    if (g + 1 < total) {
                        const int s_phase = (cycle + slot) & 1;
                        bar_wait(&keys_ready[slot ^ 1], s_phase);
                        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                        qk_sequence128(tmem + uint32_t(score_column(slot ^ 1)),
                                       descriptor(s.q[0], false), descriptor(s.k[slot ^ 1], false),
                                       qk_instruction);
                        commit_one(&qk_done[0][slot ^ 1]);
                    }
                    //: `probabilities[slot][0]` also certifies the CORRECTION has finished with
                    //: tile g -- its O rescale and its read of the stats lane -- which is what makes
                    //: the QK one iteration later into this slot safe (hazard (i)).
                    bar_wait(&probabilities[slot][0], cycle);
                    bar_wait(&values_ready[slot], cycle);
                    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                    const uint32_t probability = tmem + uint32_t(score_column(slot));
                    const uint64_t values = descriptor(s.v[slot], true);
                    pv_keys_0_63(out, probability, values, pv_instruction, u != 0);
                    bar_wait(&probabilities[slot][1], cycle);
                    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                    pv_keys_64_127(out, probability, values, pv_instruction, true);
                    commit_one(&pv_done[0][slot]);
                    if (++u == sp.upd[j]) {
                        ++j; u = 0;
                        if (j < sp.nsegs) out = tmem + uint32_t(slot_column<WAVES>(sp.comp[j], 0));
                    }
                }
            }
          } else {
            //: TWO WAVES: `PV(w, g), QK(w, g+1)` into the slot the PV just read, wave 0 then wave 1
            //: -- the order of TileFoundry's port, verbatim.
            bar_wait(&query_ready[0], 0);
            bar_wait(&keys_ready[0], 0);
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
            qk_sequence128(tmem + uint32_t(score_column(0)), descriptor(s.q[0], false),
                           descriptor(s.k[0], false), qk_instruction);
            commit_one(&qk_done[0][0]);
            bar_wait(&query_ready[1], 0);
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
            qk_sequence128(tmem + uint32_t(score_column(1)), descriptor(s.q[1], false),
                           descriptor(s.k[0], false), qk_instruction);
            commit_one(&qk_done[1][0]);
            int g = 0;
            for (int j = 0; j < sp.nsegs; ++j) {
                for (int u = 0; u < sp.upd[j]; ++u, ++g) {
                    const int slot = g % KV, cycle = g / KV, parity = g & 1;
                    const int next = (slot + 1) % KV;
                    #pragma unroll 1
                    for (int wave = 0; wave < WAVES; ++wave) {
                        bar_wait(&probabilities[wave][0], parity);
                        bar_wait(&values_ready[slot], cycle & 1);
                        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                        const uint32_t out = tmem + uint32_t(slot_column<WAVES>(0, wave));
                        const uint32_t probability = tmem + uint32_t(score_column(wave));
                        const uint64_t values = descriptor(s.v[slot], true);
                        pv_keys_0_63(out, probability, values, pv_instruction, u != 0);
                        bar_wait(&probabilities[wave][1], parity);
                        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                        pv_keys_64_127(out, probability, values, pv_instruction, true);
                        commit_one(&pv_done[wave][slot]);
                        if (g + 1 < total) {
                            qk_sequence128(tmem + uint32_t(score_column(wave)),
                                           descriptor(s.q[wave], false),
                                           descriptor(s.k[next], false), qk_instruction);
                            commit_one(&qk_done[wave][next]);
                        }
                    }
                }
            }
          }
        }
    } else if (thread >= CORR) {
        // ===================================== CORRECTION =====================================
        registers_decrease<correction_registers<WAVES>()>();
        const int row = thread - CORR;
        //: This row's GLOBAL backing slots per wave, indexed by SM id (one block per SM; the launch
        //: verifies it). A thread owns its row of every slot: no barrier, no fence.
        int64_t fold_offset = 0;
        if (p.fold_globals > 0) {
            uint32_t sm;
            asm volatile("mov.u32 %0,%%smid;" : "=r"(sm));
            if (int(sm) >= p.fold_sms) __trap();
            fold_offset = int64_t(int(sm) * WAVES) * p.fold_globals * int64_t(128 * DIM);
        }
        auto backing_of = [&](int w, int b) -> float* {
            if (shared_backing<WAVES>() != 0 && b == 0) return s.node;
            return p.fold_node + fold_offset
                + (int64_t(w) * p.fold_globals + (b - shared_backing<WAVES>())) * int64_t(128 * DIM);
        };
        constexpr int STACK = FOLD_STACK;
        constexpr bool SHARED_STACK = WAVES == 2;
        RegStack<SHARED_STACK ? 1 : STACK> stL[WAVES], stM[WAVES];
        int depth[WAVES];
        float Lseg[WAVES], Mseg[WAVES];
        auto stack_get = [&](auto wc, int lm, int d) -> float {
            constexpr int w = decltype(wc)::value;
            if constexpr (SHARED_STACK)
                return corr_stack[((lm * WAVES + w) * STACK + d) * 128 + row];
            else
                return lm == 0 ? stL[w].get(d) : stM[w].get(d);
        };
        auto stack_set = [&](auto wc, int lm, int d, float x) {
            constexpr int w = decltype(wc)::value;
            if constexpr (SHARED_STACK)
                corr_stack[((lm * WAVES + w) * STACK + d) * 128 + row] = x;
            else if (lm == 0) stL[w].set(d, x);
            else stM[w].set(d, x);
        };
        for_waves<WAVES>([&](auto wc) {
            constexpr int w = decltype(wc)::value;
            depth[w] = 0; Lseg[w] = 0.0f; Mseg[w] = 0.0f;
            #pragma unroll
            for (int k = 0; k < (SHARED_STACK ? 1 : STACK); ++k) { stL[w].v[k] = 0.0f; stM[w].v[k] = 0.0f; }
        });

        //: THE SEGMENT BOUNDARY of segment `jb` for wave `w`, whose last tile was `gl`: wait that PV,
        //: push (L, M), run the merges in plan order. Hazard (b): this thread waits each `pv_done`
        //: generation once, in order; the submitter's waits consume nothing. Hazard (c): every merge
        //: result is read next by this same thread after `wait::st`, or by the epilogue.
        //: ORDER: it runs right before wave w's arrival for the NEXT segment's first tile, per
        //: wave, so at two waves wave 0 resumes after its own boundary and its tiles overlap wave
        //: 1's -- instead of both waves' PVs waiting for both waves' boundaries and spills.
        auto boundary = [&](auto wc, int jb, int gl) {
                    constexpr int w = decltype(wc)::value;
                    //: Two waves: every left operand is a backing slot, final before this wait
                    //: (a node this thread spilled earlier), so the first two chunks of the
                    //: first merge are requested NOW and land under the wait.
                    constexpr bool RING = WAVES == 2 && GQA_MERGE_CPASYNC && !GQA_W2_SOFTMAX_FOLD;
                    const int nmerge = sp.mrg_n[jb];
                    const int steps = nmerge * (DIM / 16);
                    auto issue = [&](int st) {
                        const int m = st / (DIM / 16), c = st % (DIM / 16);
                        const float* node = backing_of(w, backing_index(sp.mrg_l[jb][m]));
                        const uint32_t dst = shared_address(&merge_ring[(st & 1) * 16 * 128 + row]);
                        #pragma unroll
                        for (int jj = 0; jj < 16; ++jj)
                            cp_async4(dst + jj * 128 * 4, node + (c * 16 + jj) * 128 + row);
                        asm volatile("cp.async.commit_group;" ::: "memory");
                    };
                    if constexpr (RING) {
                        if (steps > 0) issue(0);
                        if (steps > 1) issue(1);
                    }
                    bar_wait(&pv_done[w][gl % KV], (gl / KV) & 1);
                    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                    stack_set(wc, 0, depth[w], Lseg[w]);
                    stack_set(wc, 1, depth[w], Mseg[w] == -CUDART_INF_F ? M_EMPTY : Mseg[w]);
                    ++depth[w];
                    #pragma unroll 1
                    for (int m = 0; m < sp.mrg_n[jb]; ++m) {
                        const int d = depth[w];
                        const float la = stack_get(wc, 0, d - 2), lb = stack_get(wc, 0, d - 1);
                        const MergeScales ms = merge_scales(stack_get(wc, 1, d - 2),
                                                            stack_get(wc, 1, d - 1), scale_log2);
                        const int left = sp.mrg_l[jb][m];
                        const uint32_t right = tmem + uint32_t(slot_column<WAVES>(sp.mrg_r[jb][m], w));
                        if (RING) {
                            //: The same `merge_o_pair` over the same elements, left term
                            //: first, in the same chunk order as `merge_from_node`; the left
                            //: operand's bytes arrive through the ring instead of a register
                            //: load. Thread-local: `cp.async.wait_group` is the only ordering.
                            #pragma unroll 1
                            for (int c = 0; c < DIM / 16; ++c) {
                                const int st = m * (DIM / 16) + c;
                                if (st + 1 < steps)
                                    asm volatile("cp.async.wait_group 1;" ::: "memory");
                                else
                                    asm volatile("cp.async.wait_group 0;" ::: "memory");
                                float a[16], b[16];
                                #pragma unroll
                                for (int jj = 0; jj < 16; ++jj)
                                    a[jj] = merge_ring[(st & 1) * 16 * 128 + jj * 128 + row];
                                tload<16>(right + c * 16, b);
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                #pragma unroll
                                for (int jj = 0; jj < 16; jj += 2)
                                    merge_o_pair(a[jj], a[jj + 1], b[jj], b[jj + 1], ms.sa, ms.sb);
                                tstore<16>(right + c * 16, a);
                                if (st + 2 < steps) issue(st + 2);
                            }
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        } else if (is_backing(left))
                            merge_from_node<CC>(backing_of(w, backing_index(left)), row, right,
                                                ms.sa, ms.sb);
                        else
                            tmem_merge<CC>(tmem + uint32_t(slot_column<WAVES>(left, w)), right,
                                           ms.sa, ms.sb);
                        stack_set(wc, 0, d - 2, merge_l(la, lb, ms.sa, ms.sb));
                        stack_set(wc, 1, d - 2, ms.m);
                        depth[w] = d - 1;
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
        };
        auto spill_for = [&](auto wc, int jn) {
            constexpr int w = decltype(wc)::value;
            spill_node<32>(backing_of(w, sp.spill_dst[jn]), row,
                           tmem + uint32_t(slot_column<WAVES>(sp.spill_src[jn], w)));
        };

        int g = 0;
        for (int j = 0; j < sp.nsegs; ++j) {
            const int nup = sp.upd[j];
            const int comp = sp.comp[j];
            for (int u = 0; u < nup; ++u, ++g) {
                const bool last = u == nup - 1;
                const int kslot = g % KV;
                const int phase = (g / KV) & 1;
                for_waves<WAVES>([&](auto wc) {
                    constexpr int w = decltype(wc)::value;
                    const int pslot = WAVES == 1 ? kslot : w;
                    const uint32_t s_base = tmem + uint32_t(score_column(pslot));
                    if (!(WAVES == 2 && GQA_W2_SOFTMAX_FOLD) && u == 0 && j > 0) {
                        //: Segment j-1's boundary, then the spill that frees this segment's
                        //: tile: both before this wave's first-tile arrival, which releases
                        //: PV(first of j) -- the overwrite (hazard (a): at two waves comp[j]
                        //: IS the tile being spilled; the spill's `wait::ld` precedes it).
                        boundary(wc, j - 1, g - 1);
                        if (sp.spill_src[j] != NO_SPILL) spill_for(wc, j);
                    }
                    bar_wait(&stats_ready[w][kslot], phase);
                    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                    if (u != 0) {
                        //: The warp ballot, here and not in the softmax: `alpha == 1` for a kept or
                        //: empty row, so `any(alpha != 1)` is exactly `any(!keep &&
                        //: !empty_row)`, and `correction128` still skips the multiply per lane.
                        const float alpha = load1(s_base + ALPHA_COLUMN);
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        if (__any_sync(0xffffffffu, alpha != 1.0f)) {
                            //: At one wave QK(g) is issued BEFORE PV(g-1), so qk_done(g) does not
                            //: imply PV(g-1) retired: the O tile may still be accumulating.
                            if (WAVES == 1 || GQA_CORR_PV_WAIT) {
                                bar_wait(&pv_done[w][(g - 1) % KV], ((g - 1) / KV) & 1);
                                asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                            }
                            correction128<CC>(tmem + uint32_t(slot_column<WAVES>(comp, w)), alpha);
                        }
                    }
                    if (last && !(WAVES == 2 && GQA_W2_SOFTMAX_FOLD)) {
                        //: (L, M) is read BEFORE this tile's arrival, and the order is load-bearing:
                        //: the arrival releases PV(g_last), after which the issuer may issue the QK
                        //: that overwrites columns 64-66 (hazard (i)).
                        bar_wait(&segment_stats[w][j & 1], (j >> 1) & 1);
                        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                        load2(s_base + STATS_COLUMN, Lseg[w], Mseg[w]);
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
                    arrive_warp_local(&probabilities[pslot][0]);
                });
            }
        }

        //: The LAST segment's boundary, after the item's last PV -- or, at two waves with the
        //: softmax folding, its "root ready" arrival (`segment_stats[w][0]`, once per item).
        if constexpr (WAVES == 2 && GQA_W2_SOFTMAX_FOLD && GQA_W2_SOFTMAX_EPILOGUE) {
            //: Nothing: each wave's softmax writes its own epilogue.
        } else if constexpr (WAVES == 2 && GQA_W2_SOFTMAX_FOLD) {
            for_waves<WAVES>([&](auto wc) {
                constexpr int w = decltype(wc)::value;
                bar_wait(&segment_stats[w][0], 0);
                asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
            });
        } else {
            for_waves<WAVES>([&](auto wc) { boundary(wc, sp.nsegs - 1, total - 1); });
        }

        //: THE EPILOGUE, per wave, in the correction warpgroup, with no CTA-wide rendezvous in
        //: front of it: this thread waited the final `pv_done` of every wave and wrote every merge,
        //: so each root is final in its own view. Byte ranges of the staging tile (`storage`):
        //: * wave 0 stages at `stage + row * 136` (bf16, ends at 34816 B) or `row * 132` (fp32,
        //:   ends at 67584 B): inside `s.q` plus at most 2 KiB of `s.k` at two waves, and plus at
        //:   most 35 KiB of `s.k` at one wave. Both are dead: every QK of the item precedes the last
        //:   PV in the issuer's in-order stream, and that PV has retired.
        //: * wave 1 stages at `stage + (128 + row) * ...`: the bf16 path ends at 69616 B (inside
        //:   `s.k[0]`), the fp32 partial path at 135152 B (into `s.v[0]`) -- whose last reader is
        //:   wave 1's own last PV, retired before this thread got here.
        //: * at one wave the 64 KiB node slot sits at 160 KiB and is never reached.
        //: The staging tile is read back only by the warp that wrote it (warp-local transpose).
        if constexpr (!(WAVES == 2 && GQA_W2_SOFTMAX_EPILOGUE)) {
            const int first_begin = it.k0 * p.seg;
            for_waves<WAVES>([&](auto wc) {
                constexpr int w = decltype(wc)::value;
                wave_epilogue(wc, std::false_type{}, row, tmem + uint32_t(slot_column<WAVES>(sp.root, w)),
                              stack_get(wc, 0, 0), stack_get(wc, 1, 0), it.slot_base, first_begin,
                              tok0, used, q_end, kv_head);
            });
        }
    } else {
        // ======================================= SOFTMAX =======================================
        registers_increase<softmax_registers<WAVES>()>();
        //: The WAVE is a compile-time constant in each softmax warpgroup's copy of this code (FA4's
        //: `stage`): as a runtime value ptxas hoists every per-wave barrier and TMEM address out of
        //: the tile loop and, at the 192-register two-wave budget, spills them.
        auto softmax_role = [&](auto wave_c) {
        constexpr int wave = decltype(wave_c)::value;
        const int row = thread & 127;

        float maximum = -CUDART_INF_F;
        float denominator = 0.0f;
        bool seg_live = false;
        int limit = -1;

        //: One KV tile. The ONLY wait is `qk_done`.
        auto body = [&](auto interior_tag, int g, int u) {
            constexpr bool interior = decltype(interior_tag)::value;
            const int pslot = WAVES == 1 ? (g % KV) : wave;
            const uint32_t s_base = tmem + uint32_t(score_column(pslot));
            bar_wait(&qk_done[wave][g % KV], (g / KV) & 1);
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
            float scores[4][32];
            #pragma unroll
            for (int group = 0; group < 4; ++group)
                load32(s_base + group * 32, scores[group]);
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            if constexpr (!interior) {
                const int local_limit = seg_live ? limit - u * KEY_TILE : -1;
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
            float safe_maximum = next_maximum == -CUDART_INF_F ? 0.0f : next_maximum;
            const float shift = __fmul_rn(maximum - safe_maximum, scale_log2);
            // The recipe's LAZY RESCALE (params.h RESCALE_THRESHOLD), unchanged.
            const bool keep = shift >= -RESCALE_THRESHOLD;
            const bool empty_row = next_maximum == -CUDART_INF_F;
            if (keep) {
                next_maximum = maximum;
                safe_maximum = maximum;
            }
            const float alpha = keep ? 1.0f : exp2_native(shift);
            //: Publish alpha into the stats lane. An empty row publishes the IDENTITY: its
            //: accumulator is exactly signed zero and its alpha exactly zero, so the rescale is the
            //: identity either way, and 1.0 keeps it out of the correction's ballot -- exactly a
            //: `!empty_row` exclusion. Covered by the quarter-1
            //: `wait::st` below.
            store1(s_base + ALPHA_COLUMN, empty_row ? 1.0f : alpha);
#if GQA_EARLY_STATS == 1
            //: Published HERE, right after the max and BEFORE the exponentials -- FA4's
            //: `softmax_step` signals its correction warpgroup right after `update_row_max` -- so the
            //: correction's read, ballot and "O tile ready" arrival overlap this tile's exp2 work
            //: instead of sitting between quarter 1 and the PV issue.
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
            arrive_warp_stats(&stats_ready[wave][g % KV]);
#endif
            const float bias = __fmul_rn(-safe_maximum, scale_log2);
            float bias_late = bias;
            #pragma unroll
            for (int quarter = 0; quarter < 4; ++quarter) {
                uint32_t packed[16];
                const float qbias = (GQA_EARLY_STATS == 3 && quarter > 0) ? bias_late : bias;
                #pragma unroll
                for (int pair = 0; pair < 16; ++pair) {
                    const int jj = pair * 2;
                    const FloatPair shifted = affine_pair(scores[quarter][jj], scores[quarter][jj + 1],
                                                          scale_log2, qbias);
                    const FloatPair e = emulated_exp2(quarter * 16 + pair)
                        ? polynomial_exp2(shifted.lo, shifted.hi)
                        : FloatPair{exp2_native(shifted.lo), exp2_native(shifted.hi)};
                    scores[quarter][jj] = e.lo;
                    scores[quarter][jj + 1] = e.hi;
                    packed[pair] = pack_bf16_pair(e.lo, e.hi);
                }
#if GQA_EARLY_STATS == 3
                if (quarter == 0) {
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
                    bias_late = arrive_warp_tied(&stats_ready[wave][g % KV], bias);
                }
#endif
                store16_bits(s_base + quarter * 16, packed);
                if ((quarter & 1) == 0) continue;
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
                if (quarter == 1) {
                    arrive_warp_local(&probabilities[pslot][0]);
                    if (GQA_EARLY_STATS == 0) arrive_warp_stats(&stats_ready[wave][g % KV]);
                } else {
                    arrive_warp_local(&probabilities[pslot][1]);
                }
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
        for (int j = 0; j < sp.nsegs; ++j) {
            // Reset at every segment start: carrying the accumulator across would be the flat chain.
            maximum = -CUDART_INF_F;
            denominator = 0.0f;
            const int w_border = sc_w_border[wave], wave_tok0 = sc_wave_tok0[wave];
            const int q_end_s = sc_q_end;
            const int border = w_border + row / RATIO;
            const int flags = (wave_tok0 + row / RATIO < q_end_s ? 1 : 0)
                            | (wave_tok0 >= q_end_s ? 2 : 0)
                            | (wave_tok0 + WAVE_TOKENS - 1 < q_end_s ? 4 : 0);
            const int seg_begin = (sc_k0 + j) * p.seg;
            seg_live = (flags & 1) && border > seg_begin;
            limit = border - 1 - seg_begin;
            const int tile_limit = w_border - 1 - seg_begin;
            int interior_end;
            if (flags & 2) {
                interior_end = sp.upd[j];
            } else if (!GQA_PARTIAL_WAVE_INTERIOR && !(flags & 4)) {
                interior_end = 0;
            } else {
                //: A PARTLY live wave (decode: one live token of 16) takes the interior body
                //: on the tiles every LIVE row sees in full. Row 0's token is live here (flags & 2 is
                //: clear) and has the smallest border of the wave, so `tile_limit` bounds every live
                //: row; the DEAD rows run unmasked, and they write nothing (every store is behind
                //: `item_live`) and share nothing (rows are private in M; the correction's warp
                //: ballot only decides whether the per-lane-guarded rescale issues).
                const int last_interior = tile_limit - (KEY_TILE - 1);
                interior_end = last_interior < 0 ? 0 : last_interior / KEY_TILE + 1;
                if (interior_end > sp.upd[j]) interior_end = sp.upd[j];
            }
            int u = 0;
            for (; u < interior_end; ++u, ++g) body(std::true_type{}, g, u);
            for (; u < sp.upd[j]; ++u, ++g) body(std::false_type{}, g, u);
            if constexpr (WAVES == 2 && GQA_W2_SOFTMAX_FOLD) {
                //: THIS WAVE's boundary (two waves): the segment's last PV, push (L, M) on the
                //: wave's stack in shared memory (the correction's epilogue reads the root there),
                //: the merges in plan order, then the spill that frees the next segment's tile --
                //: all before this wave's next `probabilities` arrival, which releases PV(first).
                bar_wait(&pv_done[wave][(g - 1) % KV], ((g - 1) / KV) & 1);
                asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                //: Nothing of the boundary is live across the tile loop (the two-wave softmax is at
                //: its register edge): the stack depth before segment j's push is the number of
                //: pending nodes, `j - sum(mrg_n[j' < j])`, and the slot base is recomputed.
                int sm_depth = j;
                for (int jj = 0; jj < j; ++jj) sm_depth -= sp.mrg_n[jj];
                int64_t sm_fold_offset = 0;
                if (p.fold_globals > 0) {
                    uint32_t smv;
                    asm volatile("mov.u32 %0,%%smid;" : "=r"(smv));
                    if (int(smv) >= p.fold_sms) __trap();
                    sm_fold_offset = (int64_t(int(smv) * WAVES + wave) * p.fold_globals)
                                   * int64_t(128 * DIM);
                }
                auto sm_backing = [&](int b) -> float* {
                    return p.fold_node + sm_fold_offset + int64_t(b) * int64_t(128 * DIM);
                };
                float* const stk = corr_stack;
                auto at = [&](int lm, int d) -> float& {
                    return stk[((lm * WAVES + wave) * FOLD_STACK + d) * 128 + row];
                };
                at(0, sm_depth) = denominator;
                at(1, sm_depth) = maximum == -CUDART_INF_F ? M_EMPTY : maximum;
                ++sm_depth;
                #pragma unroll 1
                for (int m = 0; m < sp.mrg_n[j]; ++m) {
                    const int d = sm_depth;
                    const float la = at(0, d - 2), lb = at(0, d - 1);
                    const MergeScales ms = merge_scales(at(1, d - 2), at(1, d - 1), scale_log2);
                    const int left = sp.mrg_l[j][m];
                    if (!is_backing(left)) __trap();
                    merge_from_node_half(sm_backing(backing_index(left)), row,
                                         tmem + uint32_t(slot_column<WAVES>(sp.mrg_r[j][m], wave)),
                                         ms.sa, ms.sb);
                    at(0, d - 2) = merge_l(la, lb, ms.sa, ms.sb);
                    at(1, d - 2) = ms.m;
                    sm_depth = d - 1;
                }
                asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
                if (j + 1 < sp.nsegs && sp.spill_src[j + 1] != NO_SPILL)
                    spill_node_halves(sm_backing(sp.spill_dst[j + 1]), row,
                                      tmem + uint32_t(slot_column<WAVES>(sp.spill_src[j + 1], wave)));
                if (j + 1 == sp.nsegs) {
                    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
                    arrive_warp_stats(&segment_stats[wave][0]);       // root ready
                    if constexpr (GQA_W2_SOFTMAX_EPILOGUE) {
                        //: This wave's EPILOGUE, here at its last boundary (code after the tile
                        //: loop made ptxas spill in it): once BOTH waves' roots are ready every QK
                        //: and PV of the item has retired and the staging tile is dead.
                        bar_wait(&segment_stats[1 - wave][0], 0);
                        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                        const int* const raw = reinterpret_cast<const int*>(p.items + blockIdx.x);
                        const int e_tok0 = __ldcg(raw + 1), e_kv_head = __ldcg(raw + 2);
                        const int e_k0 = __ldcg(raw + 3), e_q_end = __ldcg(raw + 5);
                        const int e_used = __ldcg(raw + 6), e_slot_base = __ldcg(raw + 7);
                        wave_epilogue(wave_c, std::true_type{}, row,
                                      tmem + uint32_t(slot_column<WAVES>(sp.root, wave)),
                                      at(0, 0), at(1, 0), e_slot_base, e_k0 * p.seg, e_tok0, e_used,
                                      e_q_end, e_kv_head);
                    }
                }
                continue;
            }
            //: The segment's (L, M), raw frame (-inf kept; the correction applies M_EMPTY), into
            //: columns 65-66 of the LAST tile's score slot -- then straight into the next segment.
            const int pslot = WAVES == 1 ? ((g - 1) % KV) : wave;
            store2(tmem + uint32_t(score_column(pslot)) + STATS_COLUMN, denominator, maximum);
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
            arrive_warp_stats(&segment_stats[wave][j & 1]);
        }
        };
        if (WAVES == 1 || thread < 128) softmax_role(std::integral_constant<int, 0>{});
        else softmax_role(std::integral_constant<int, WAVES - 1>{});
    }

    //: `synchronize_cta`, not a bare `__syncthreads`: the deallocating warp is a tcgen05 consumer of
    //: accesses other warps made.
    synchronize_cta();
    if (thread < 32) {
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0,%1;"
                     :: "r"(tmem), "r"(tmem_columns<WAVES>()) : "memory");
    }
}

// --- launcher ------------------------------------------------------------------------------------

namespace {

template <int WAVES, int RATIO>
cudaError_t configure_one() {
    return cudaFuncSetAttribute(gqa_leaf_d128<WAVES, RATIO>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                int(sizeof(LeafShared<WAVES>)));
}

//: `setmaxnreg` draws from what ptxas ALLOCATED: the budgets of the warps on one sub-partition must
//: sum to at most `(threads / 128) * numRegs`. Checked at BOTH wave counts (one wave declares
//: budgets too), so a future change that makes ptxas declare less fails the launch
//: with `cudaErrorLaunchOutOfResources` rather than hanging the device. Also checks the static plus
//: dynamic shared memory against the part's opt-in limit (one wave sits at the edge).
template <int WAVES, int RATIO>
cudaError_t check_register_budget() {
    static cudaError_t cached = cudaErrorUnknown;
    if (cached != cudaErrorUnknown) return cached;
    cudaFuncAttributes attributes{};
    const cudaError_t status = cudaFuncGetAttributes(&attributes, gqa_leaf_d128<WAVES, RATIO>);
    if (status != cudaSuccess) return status;
    constexpr int warps_per_subpartition = (128 * WAVES + 256) / 128;
    if (warps_per_subpartition * attributes.numRegs
            < WAVES * softmax_registers<WAVES>() + correction_registers<WAVES>()
              + other_registers<WAVES>())
        return cached = cudaErrorLaunchOutOfResources;
    if (attributes.sharedSizeBytes + sizeof(LeafShared<WAVES>) > 232448)
        return cached = cudaErrorLaunchOutOfResources;
    return cached = cudaSuccess;
}

template <int WAVES, int RATIO>
cudaError_t check_fold_occupancy() {
    static cudaError_t cached = cudaErrorUnknown;
    if (cached != cudaErrorUnknown) return cached;
    int blocks = 0;
    const cudaError_t status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks, gqa_leaf_d128<WAVES, RATIO>, 128 * WAVES + 256, int(sizeof(LeafShared<WAVES>)));
    if (status != cudaSuccess) return status;
    cached = blocks == 1 ? cudaSuccess : cudaErrorLaunchOutOfResources;
    return cached;
}

template <int WAVES, int RATIO>
cudaError_t launch_one(const LeafParams& p, const CUtensorMap& mq, const CUtensorMap& mk,
                       const CUtensorMap& mv, int item_grid, cudaStream_t stream) {
    cudaError_t status = configure_one<WAVES, RATIO>();
    if (status != cudaSuccess) return status;
    status = check_register_budget<WAVES, RATIO>();
    if (status != cudaSuccess) return status;
    if (p.fold_globals > 0) {
        status = check_fold_occupancy<WAVES, RATIO>();
        if (status != cudaSuccess) return status;
    }
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(item_grid, 1, 1);
    config.blockDim = dim3(128 * WAVES + 256, 1, 1);
    config.dynamicSmemBytes = sizeof(LeafShared<WAVES>);
    config.stream = stream;
    config.attrs = nullptr;
    config.numAttrs = 0;
    status = cudaLaunchKernelEx(&config, gqa_leaf_d128<WAVES, RATIO>, mq, mk, mv, p);
    if (status != cudaSuccess) return status;
    return cudaGetLastError();
}

}  // namespace

cudaError_t launch_leaf_d128(const LeafParams& p, const CUtensorMap& mq, const CUtensorMap& mk,
                            const CUtensorMap& mv, int waves, int head_ratio, int item_grid,
                            cudaStream_t stream) {
    if (item_grid <= 0) return cudaSuccess;
    #define GQA_LEAF(W, R) return launch_one<W, R>(p, mq, mk, mv, item_grid, stream)
    if (waves == 1) {
        switch (head_ratio) {
            case 1: GQA_LEAF(1, 1);
            case 2: GQA_LEAF(1, 2);
            case 4: GQA_LEAF(1, 4);
            case 8: GQA_LEAF(1, 8);
            case 16: GQA_LEAF(1, 16);
            default: return cudaErrorInvalidValue;
        }
    }
    if (waves == 2) {
        switch (head_ratio) {
            case 1: GQA_LEAF(2, 1);
            case 2: GQA_LEAF(2, 2);
            case 4: GQA_LEAF(2, 4);
            case 8: GQA_LEAF(2, 8);
            case 16: GQA_LEAF(2, 16);
            default: return cudaErrorInvalidValue;
        }
    }
    #undef GQA_LEAF
    return cudaErrorInvalidValue;
}

CUresult encode_map(void* storage, const void* input, int tokens, int heads, int dim,
                    int tile_rows, int head_box, int64_t token_stride, int64_t head_stride) {
    if (tokens < 1 || heads < 1 || head_box > heads || (dim % 64) != 0)
        return CUDA_ERROR_INVALID_VALUE;
    if (head_stride == 0) head_stride = dim;
    if (token_stride == 0) token_stride = int64_t(heads) * head_stride;
    if (head_stride < dim || token_stride < dim || (head_stride % 8) != 0 || (token_stride % 8) != 0)
        return CUDA_ERROR_INVALID_VALUE;
    const uint64_t chunks = uint64_t(dim / 64);
    const uint64_t head_bytes = uint64_t(head_stride) * 2;        // bytes between (token, head) rows
    const uint64_t token_bytes = uint64_t(token_stride) * 2;      // bytes between tokens
    uint64_t dims[5] = {64, chunks, uint64_t(heads), uint64_t(tokens), 1};
    uint64_t strides[4] = {128, head_bytes, token_bytes, uint64_t(tokens) * token_bytes};
    uint32_t box[5] = {64, 1, uint32_t(head_box), uint32_t(tile_rows), 1};
    uint32_t steps[5] = {1, 1, 1, 1, 1};
    return cuTensorMapEncodeTiled(static_cast<CUtensorMap*>(storage),
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 5, const_cast<void*>(input), dims, strides,
        box, steps, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}


}  // namespace gqa_sm100
