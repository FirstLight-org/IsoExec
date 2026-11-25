// SPDX-License-Identifier: BSD-3-Clause
// The fold's plan SKELETON: the streaming binary counter's slot allocation and merge lists for a run
// of `nsegs` segments over `tmem` TMEM output tiles, evaluated at compile time. Execution only.
// Included by leaf_d128.cu (device table).
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "params.h"

namespace gqa_sm100 {

__host__ __device__ __forceinline__ constexpr bool is_backing(int slot) { return slot < 0; }
__host__ __device__ __forceinline__ constexpr int backing_index(int slot) { return -1 - slot; }
__host__ __device__ __forceinline__ constexpr int backing_slot(int index) { return -1 - index; }
static constexpr int NO_SPILL = -1;

//: THE PLAN SKELETON. Every slot assignment and merge list of a run depends
//: only on `(tmem_slots, nsegs)` -- never on the tile counts -- so the streaming counter's allocator
//: is evaluated at COMPILE TIME into a `__constant__` table, and a CTA copies one row of it instead
//: of thread 0 running the allocator serially in front of the first barrier. The allocator below is
//: the streaming counter's (and `host.cu::fold_spill_count` counts with it): it spills the
//: DEEPEST pending node still in TMEM into the LOWEST free backing slot, so the merges it names are
//: exactly the canonical tree's and nothing can move a bit. The runtime-indexed arrays it needs
//: live in the compiler, not on the device stack, which keeps the leaf's stack frame at
//: to zero.
struct PlanSkeleton {
    int8_t comp[FOLD_MAX_RUN];
    int8_t spill_src[FOLD_MAX_RUN];
    int8_t spill_dst[FOLD_MAX_RUN];
    int8_t mrg_n[FOLD_MAX_RUN];
    int8_t mrg_l[FOLD_MAX_RUN][FOLD_MERGES];
    int8_t mrg_r[FOLD_MAX_RUN][FOLD_MERGES];
    int8_t root;
    int8_t backing_needed;      // highest backing index used + 1
    int8_t feasible;
};

__host__ __device__ constexpr PlanSkeleton make_skeleton(int tmem, int nsegs) {
    PlanSkeleton k{};
    k.feasible = 1;
    bool used[2] = {false, false};
    bool held[FOLD_BACKING] = {};
    int st_slot[FOLD_STACK + 1] = {}, st_z[FOLD_STACK + 1] = {};
    int depth = 0, backing_needed = 0;
    for (int j = 0; j < nsegs; ++j) {
        k.spill_src[j] = int8_t(NO_SPILL);
        k.spill_dst[j] = 0;
        int a = 0;
        while (a < tmem && used[a]) ++a;
        if (a >= tmem) {
            int d = 0;
            while (d < depth && is_backing(st_slot[d])) ++d;
            int b = 0;
            while (b < FOLD_BACKING && held[b]) ++b;
            if (d >= depth || b >= FOLD_BACKING) { k.feasible = 0; return k; }
            a = st_slot[d];
            k.spill_src[j] = int8_t(a);
            k.spill_dst[j] = int8_t(b);
            if (b + 1 > backing_needed) backing_needed = b + 1;
            st_slot[d] = backing_slot(b);
            held[b] = true;
            used[a] = false;
        }
        used[a] = true;
        k.comp[j] = int8_t(a);
        if (depth >= FOLD_STACK) { k.feasible = 0; return k; }
        st_slot[depth] = a;
        st_z[depth] = 1;
        ++depth;
        int count = 0;
        while (depth > 1 && (st_z[depth - 2] == st_z[depth - 1] || j == nsegs - 1)) {
            const int left = st_slot[depth - 2], right = st_slot[depth - 1];
            if (is_backing(right) || count >= FOLD_MERGES) { k.feasible = 0; return k; }
            k.mrg_l[j][count] = int8_t(left);
            k.mrg_r[j][count] = int8_t(right);
            ++count;
            if (is_backing(left)) {
                held[backing_index(left)] = false;
                st_slot[depth - 2] = right;
            } else {
                used[right] = false;
            }
            st_z[depth - 2] += st_z[depth - 1];
            --depth;
        }
        k.mrg_n[j] = int8_t(count);
    }
    if (nsegs < 1 || is_backing(st_slot[0])) { k.feasible = 0; return k; }
    k.root = int8_t(st_slot[0]);
    k.backing_needed = int8_t(backing_needed);
    return k;
}

struct SkeletonTable {
    PlanSkeleton s[2][FOLD_MAX_RUN + 1];      // [tmem - 1][nsegs]
};
__host__ __device__ constexpr SkeletonTable make_skeleton_table() {
    SkeletonTable t{};
    for (int tm = 1; tm <= 2; ++tm)
        for (int n = 0; n <= FOLD_MAX_RUN; ++n) t.s[tm - 1][n] = make_skeleton(tm, n);
    return t;
}
//: Every run this package's host can emit is feasible in the table at both wave counts.
static_assert(make_skeleton(1, FOLD_MAX_RUN).feasible && make_skeleton(2, FOLD_MAX_RUN).feasible,
              "the skeleton covers a run of FOLD_MAX_RUN at both TMEM slot counts");
static_assert(make_skeleton(1, FOLD_MAX_RUN).backing_needed == fold_backing_slots(1, FOLD_MAX_RUN) &&
              make_skeleton(2, FOLD_MAX_RUN).backing_needed == fold_backing_slots(2, FOLD_MAX_RUN),
              "the skeleton's backing count is fold_backing_slots");

//: The item's segment plan, identical on every thread: how many KV tiles each of its segments runs,
//: which fp32 output tile each computes into, and the merges the streaming binary counter performs
//: after each. Execution only -- the merges it names are exactly the canonical tree's.
//:
//: A slot id `>= 0` is a TMEM output tile. A slot id `< 0` is BACKING slot `-1 - id`: the 64 KiB
//: shared node slot at one wave (index 0) and a global slot of this (SM, wave) otherwise.

struct SegmentPlan {
    int nsegs;
    int total;                              // KV tiles over the whole item
    int root;                               // TMEM tile holding the item's root
    int upd[FOLD_MAX_RUN];                  // tiles of segment j
    int pre[FOLD_MAX_RUN + 1];              // prefix sums
    //: int8 and not int: the plan is in SHARED memory and one wave has about 1.7 KiB of it left, so
    //: a run of sixteen has to cost bytes rather than words. Every value is a slot id in
    //: [-FOLD_BACKING, tmem_slots) or a small count.
    int8_t comp[FOLD_MAX_RUN];              // output tile segment j computes into (always TMEM)
    int8_t spill_src[FOLD_MAX_RUN];         // TMEM tile vacated before segment j, or NO_SPILL
    int8_t spill_dst[FOLD_MAX_RUN];         // backing index it was moved to
    int8_t mrg_n[FOLD_MAX_RUN];             // merges after segment j
    int8_t mrg_l[FOLD_MAX_RUN][FOLD_MERGES];
    int8_t mrg_r[FOLD_MAX_RUN][FOLD_MERGES];
};

//: The one-wave D128 leaf sits at the 232448 B shared-memory limit (README section 2), so the plan's
//: size is pinned here.
static_assert(sizeof(SegmentPlan) == sizeof(int) * (3 + FOLD_MAX_RUN + FOLD_MAX_RUN + 1) +
                                     (((4 + 2 * FOLD_MERGES) * FOLD_MAX_RUN + 3) / 4) * 4,
              "SegmentPlan grew: the one-wave D128 shared budget has no room");

//: In GLOBAL memory, not `__constant__`: warp 0's lanes read DIFFERENT entries, which the constant
//: cache serialises; global loads of one row are a single coalesced transaction (L2-resident).
static __device__ SkeletonTable PLAN_SKELETON = make_skeleton_table();

//: Build the item's plan: the per-item tile counts (one warp lane per segment, a shuffle prefix) and
//: the skeleton row for `(tmem, nsegs)`. Called by ONE WARP; `backing` is what the host allocated,
//: and a skeleton that needs more traps (unreachable on a plan the host emitted).
__device__ __forceinline__ void build_segment_plan(SegmentPlan& sp, int k0, int k1, int seg,
                                                  int border_max, int tmem, int backing) {
    const int lane = threadIdx.x & 31;
    const int nsegs = k1 - k0;
    if (nsegs < 1 || nsegs > FOLD_MAX_RUN) __trap();
    //: ONE segment (every decode item and most prefill items) needs no table: compute into tile 0,
    //: no spill, no merge, root tile 0 -- the skeleton's own row for n = 1, without its load.
    const bool single = nsegs == 1;
    const PlanSkeleton& k = PLAN_SKELETON.s[tmem - 1][single ? 1 : nsegs];
    int u = 0;
    if (lane < nsegs) {
        const int begin = (k0 + lane) * seg;
        const int stop = min(begin + seg, border_max);
        u = (stop - begin + KEY_TILE - 1) / KEY_TILE;
        if (u < 1) u = 1;
        sp.upd[lane] = u;
        if (single) {
            sp.comp[0] = 0;
            sp.spill_src[0] = int8_t(NO_SPILL);
            sp.spill_dst[0] = 0;
            sp.mrg_n[0] = 0;
        } else {
            sp.comp[lane] = k.comp[lane];
            sp.spill_src[lane] = k.spill_src[lane];
            sp.spill_dst[lane] = k.spill_dst[lane];
            sp.mrg_n[lane] = k.mrg_n[lane];
            #pragma unroll
            for (int m = 0; m < FOLD_MERGES; ++m) {
                sp.mrg_l[lane][m] = k.mrg_l[lane][m];
                sp.mrg_r[lane][m] = k.mrg_r[lane][m];
            }
        }
    }
    int incl = u;
    #pragma unroll
    for (int step = 1; step < 32; step <<= 1) {
        const int other = __shfl_up_sync(0xffffffffu, incl, step);
        if (lane >= step) incl += other;
    }
    if (lane < nsegs) sp.pre[lane + 1] = incl;
    if (lane == 0) {
        if (!single && (!k.feasible || k.backing_needed > backing)) __trap();
        sp.pre[0] = 0;
        sp.nsegs = nsegs;
        sp.root = single ? 0 : k.root;
    }
    const int total = __shfl_sync(0xffffffffu, incl, nsegs - 1);
    if (lane == 0) sp.total = total;
}

}  // namespace gqa_sm100
