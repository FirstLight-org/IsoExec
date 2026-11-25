// SPDX-License-Identifier: BSD-3-Clause
// The plan: enumerate the work items and the combine entries from the device-side sequence lengths.
//
// Everything this kernel decides is EXECUTION -- which worker evaluates which node of the canonical
// tree, and where a published node lives. It may cut a row's tree only at a node, and the relations
// force several different plans (GRAN_SINGLE, GRAN_PAIRS, GRAN_WHOLE) over one input and require
// identical output bits.
#include <cuda_runtime.h>

#include "params.h"
#include "launch.h"

namespace gqa_sm100 {

//: One block, so it can zero `counts` itself: a separate at::zeros would be another kernel launch, and
//: on a decode row the launch floor is most of the call.
//:
//: The block walks the FLATTENED (request, query tile, KV head) work list in 128-wide windows, so a
//: window may span several requests. Restarting the window loop at every request would, on a
//: BATCHED DECODE, be one window per request with 127 of 128 threads idle, four `__syncthreads`
//: and two uncoalesced dependent loads each -- enough to make the plan most of a serving call.
//: Chunked, a decode batch of 1024 is EIGHT window iterations instead of 1024.
//:
//: It changes no emission order and therefore no bit. Within a chunk the flat index ascends through
//: requests in ascending order and, inside a request, through query tiles in DESCENDING order --
//: exactly the longest-processing-time sequence a per-request loop produces -- and the three
//: counters are exclusive prefix sums over that same sequence, so every item index and every
//: `slot_base` is the per-request walk's. Only the grouping of units into block scans differs.
//: Exclusive prefix sums of three per-thread counts over the 128-thread block, plus their totals.
//: Every thread of the block must reach this (it contains `__syncthreads`), which is why the work
//: loop below walks whole windows instead of striding until it runs out of work.
//: PLAN_WARPS is the block's warp count, a compile-time constant so the cross-warp combine unrolls.
//:
//: **128 threads, and 1024 was MEASURED WORSE** for this kernel, by a constant offset of about 7 us:
//: a chunk's scan is no longer four cross-warp partials but THIRTY-TWO -- 96 dependent adds per thread
//: in `block_scan3` alone -- and every `__syncthreads` is over 1024 threads on the one SM this
//: single-block kernel occupies. The per-chunk cost grows faster than the chunk count falls. (The
//: fast plan kernels below use 1024 threads with a two-level scan instead.)
static constexpr int PLAN_THREADS = 128;

//: The key slot of logical key `key_base` of request `req`: the leaf's `key_slot`, here so the plan
//: can hand every item its first tile's slot (`Item::ks0`).
__device__ __forceinline__ int plan_key_slot(const PlanParams& p, int req, int key_base) {
    if (p.page > 0)
        return p.block_table[int64_t(req) * p.block_table_stride + key_base / p.page] * p.page
               + (key_base % p.page);
    return p.cu_seqlens_k[req] + key_base;
}
static constexpr int PLAN_WARPS = PLAN_THREADS / 32;

__device__ __forceinline__ void block_scan3(int (&value)[3], int (&exclusive)[3], int (&total)[3],
                                           int (*sums)[PLAN_WARPS]) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int running[3] = {value[0], value[1], value[2]};
    #pragma unroll
    for (int step = 1; step < 32; step <<= 1) {
        #pragma unroll
        for (int i = 0; i < 3; ++i) {
            const int other = __shfl_up_sync(0xffffffffu, running[i], step);
            if (lane >= step) running[i] += other;
        }
    }
    if (lane == 31)
        #pragma unroll
        for (int i = 0; i < 3; ++i) sums[i][warp] = running[i];
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        int before = 0, all = 0;
        #pragma unroll
        for (int w = 0; w < PLAN_WARPS; ++w) {
            if (w < warp) before += sums[i][w];
            all += sums[i][w];
        }
        exclusive[i] = before + running[i] - value[i];
        total[i] = all;
    }
    __syncthreads();
}

//: The same scan over ONE per-thread count: how many work units each of a chunk's requests has.
__device__ __forceinline__ void block_scan1(int value, int& exclusive, int& total, int* sums) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int running = value;
    #pragma unroll
    for (int step = 1; step < 32; step <<= 1) {
        const int other = __shfl_up_sync(0xffffffffu, running, step);
        if (lane >= step) running += other;
    }
    if (lane == 31) sums[warp] = running;
    __syncthreads();
    int before = 0, all = 0;
    #pragma unroll
    for (int w = 0; w < PLAN_WARPS; ++w) {
        if (w < warp) before += sums[w];
        all += sums[w];
    }
    exclusive = before + running - value;
    total = all;
    __syncthreads();
}

__global__ __launch_bounds__(PLAN_THREADS) void gqa_plan_kernel(__grid_constant__ const PlanParams p) {
    constexpr int CHUNK = PLAN_THREADS;              // == blockDim.x, one request per thread
    __shared__ int sums[3][PLAN_WARPS], wsums[PLAN_WARPS];
    //: Exclusive work-unit offsets of this chunk's 128 requests plus the chunk total, so a thread
    //: holding a flat unit index finds its request with a seven-step binary search instead of the
    //: block restarting per request. Non-decreasing, and a request with no work leaves its neighbours
    //: equal -- for which "the largest i with off[i] <= f" is still the right answer, because the tie
    //: group ends at the first request that does have work.
    __shared__ int off[CHUNK + 1];
    //: The per-request scalars the window body needs, read ONCE and coalesced.
    __shared__ int sh_q_begin[CHUNK], sh_q_end[CHUNK], sh_used[CHUNK], sh_ks[CHUNK];
    __shared__ int base_items, base_slots, base_combines, max_n_seen, overflow, first_unwritten;
    if (threadIdx.x < NUM_COUNTS) p.counts[threadIdx.x] = 0;
    if (threadIdx.x == 0) {
        base_items = base_slots = base_combines = 0;
        max_n_seen = 0;
        overflow = 0;
        //: The first item index the plan did NOT write, which bounds what the leaf may read. Without
        //: it an overflow leaves a hole: the failing tile's own items are not written, but its
        //: `item_at` is below `item_cap`, so the leaf would launch CTAs for them and read `p.items[...]`
        //: out of uninitialised `at::empty` memory -- which under the caching allocator usually holds
        //: the PREVIOUS call's items, so the CTA would do valid-looking work for a different plan and
        //: write a partial slot that belongs to nobody, silently. Prefix sums are monotone, so once
        //: one tile overflows every later one does and a single minimum is exact.
        first_unwritten = 0x7fffffff;
    }
    __syncthreads();
    for (int chunk = 0; chunk < p.b; chunk += CHUNK) {
    {
        const int r = chunk + int(threadIdx.x);
        int units = 0;
        if (r < p.b) {
            const int qb = p.cu_seqlens_q[r], qe = p.cu_seqlens_q[r + 1], us = p.seqused_k[r];
            sh_q_begin[threadIdx.x] = qb;
            sh_q_end[threadIdx.x] = qe;
            sh_used[threadIdx.x] = us;
            sh_ks[threadIdx.x] = p.page > 0 ? p.block_table[int64_t(r) * p.block_table_stride] * p.page
                                            : p.cu_seqlens_k[r];
            if (qe - qb > 0 && us > 0) units = ceil_div_i(qe - qb, p.tile_tokens) * p.h_kv;
        }
        int excl = 0, total_units = 0;
        block_scan1(units, excl, total_units, wsums);
        off[threadIdx.x] = excl;
        if (threadIdx.x == 0) off[CHUNK] = total_units;
    }
    __syncthreads();
    const int chunk_units = off[CHUNK];

    // Longest-processing-time order: visit the LAST query tile of the request first, because under
    // causal masking a later tile attends more keys and so runs more KV tiles. EXACTLY that order:
    // the item index is a prefix sum over the work list, not an `atomicAdd` over a 128-thread window,
    // so the emission order is descending causal cost and is the same on every run. (It also takes
    // about 3000 same-address atomics off the plan at `q8192/k8192`, where they serialize in L2 and
    // cost 13.1 us against the 2.5-4.2 us launch floor everywhere else.) Execution only -- it changes
    // which worker runs when, never which node it evaluates. Across a mixed batch the order is still
    // per-request, because the requests are serialised here; a counting sort by segment count across
    // requests is the global LPT and is not done.
    const int windows = ceil_div_i(chunk_units, int(blockDim.x));
    for (int window = 0; window < windows; ++window) {
        const int flat = window * int(blockDim.x) + int(threadIdx.x);
        const bool live = flat < chunk_units;
        //: The largest `i` with `off[i] <= flat`: this thread's request inside the chunk.
        int local = 0;
        if (live) {
            int lo = 0, hi = CHUNK - 1;
            while (lo < hi) {
                const int mid = (lo + hi + 1) >> 1;
                if (off[mid] <= flat) lo = mid; else hi = mid - 1;
            }
            local = lo;
        }
        const int req = chunk + local;
        const int q_begin = live ? sh_q_begin[local] : 0;
        const int q_end = live ? sh_q_end[local] : 0;
        const int used = live ? sh_used[local] : 0;
        const int work_count = live ? off[local + 1] - off[local] : 0;
        const int slotless = live ? flat - off[local] : 0;
        const int work = live ? work_count - 1 - slotless : 0;
        const int tile = live ? work / p.h_kv : 0;
        const int kv_head = live ? work % p.h_kv : 0;
        const int tok0 = q_begin + tile * p.tile_tokens;
        const int token_last = min(tok0 + p.tile_tokens - 1, q_end - 1);
        // Causal border of absolute packed token t: used - q_end + t + 1.
        const int border_first = used - q_end + tok0 + 1;
        const int border_last = used - q_end + token_last + 1;
        const int n_first = segment_count(border_first, p.seg);
        const int n = live ? segment_count(border_last, p.seg) : 0;

        // The run the items fold in place. A run longer than one is legal only when every row of
        // the tile has the same segment count, because one item publishes one node for all of its
        // rows; a tile straddling a segment boundary runs as single segments.
        const bool uniform = (n_first == n);
        int run = 1;
        if (uniform && p.granularity != GRAN_SINGLE) {
            run = p.granularity == GRAN_PAIRS ? 2 : p.max_run;
            if (run > p.max_run) run = p.max_run;
            if (run < 1) run = 1;
        }
        const bool whole = n > 0 && (n <= run || (run == 1 && n == 1));
        int nodes = 0;
        if (n > 0 && !whole)
            for (int k = 0; k < n; k += node_size_at(k, n, run)) ++nodes;

        int mine[3] = {n <= 0 ? 0 : (whole ? 1 : nodes),
                       whole ? 0 : nodes * p.tile_rows,
                       (n > 0 && !whole) ? 1 : 0};
        int at[3], total[3];
        block_scan3(mine, at, total, sums);
        const int item_at = base_items + at[0];
        const int slot_at = base_slots + at[1];
        const int combine_at = base_combines + at[2];
        __syncthreads();
        if (threadIdx.x == 0) {
            base_items += total[0];
            base_slots += total[1];
            base_combines += total[2];
        }

        if (n > 0) {
            if (n > max_n_seen) atomicMax(&max_n_seen, n);
            if (whole) {
                if (item_at >= p.item_cap) {
                    overflow = 1;
                    atomicMin(&first_unwritten, item_at);
                } else {
                    Item it;
                    it.req = req; it.tok0 = tok0; it.kv_head = kv_head;
                    it.k0 = 0; it.k1 = n; it.q_end = q_end; it.used = used;
                    it.slot_base = -1;
                    it.ks0 = sh_ks[local];
                    p.items[item_at] = it;
                }
            } else if (slot_at + nodes * p.tile_rows > p.slot_cap
                       || item_at + nodes > p.item_cap || combine_at >= p.combine_cap) {
                //: The capacity bound in host.cu is chosen so this is unreachable. If a future bound
                //: is wrong anyway, the entry must still be WRITTEN and must REFUSE itself: the
                //: counters are prefix sums taken before this test, so `counts[NUM_COMBINE_TILES]`
                //: already includes this tile, and a combine block reading an unwritten entry reads
                //: uninitialised `at::empty` memory as a `slot_base` and dies on a wild global load.
                //: A negative `slot_base` makes it exit instead, so the symptom is part of the output
                //: never written -- which `OVERFLOW_FLAG` reports under ISOEXEC_GQA_SM100_CHECK --
                //: rather than an illegal access somewhere else.
                overflow = 1;
                atomicMin(&first_unwritten, item_at);
                if (combine_at < p.combine_cap) {
                    CombineTile ct;
                    ct.req = req; ct.tok0 = tok0; ct.kv_head = kv_head; ct.slot_base = -1;
                    ct.run = 1; ct.q_end = q_end; ct.used = used; ct.pad = 0;
                    p.combine_tiles[combine_at] = ct;
                }
            } else {
                int index = 0;
                for (int k = 0; k < n;) {
                    const int z = node_size_at(k, n, run);
                    Item it;
                    it.req = req; it.tok0 = tok0; it.kv_head = kv_head;
                    it.k0 = k; it.k1 = k + z; it.q_end = q_end; it.used = used;
                    it.slot_base = slot_at + index * p.tile_rows;
                    it.ks0 = k == 0 ? sh_ks[local] : -1;
                    p.items[item_at + index] = it;
                    ++index;
                    k += z;
                }
                CombineTile ct;
                ct.req = req; ct.tok0 = tok0; ct.kv_head = kv_head; ct.slot_base = slot_at;
                ct.run = run;
                ct.q_end = q_end; ct.used = used; ct.pad = 0;
                p.combine_tiles[combine_at] = ct;
            }
        }
        __syncthreads();
    }
    }
    if (threadIdx.x == 0) {
        //: Clamped: past an overflow the scan counted items the capacity could not hold, and the
        //: grid is the capacity. A count above it would only ever be read by a block that does not
        //: exist, but a count that matches what was WRITTEN is the invariant worth keeping.
        //: Clamped to what was actually WRITTEN, not to what the scan counted: the capacity, the
        //: total, and the first index an overflowing tile left unwritten. A CTA past this reads an
        //: uninitialised `Item`, which is worse than not running.
        p.counts[NUM_ITEMS] = min(min(base_items, p.item_cap), first_unwritten);
        p.counts[NUM_SLOTS] = base_slots;
        //: CLAMPED TO THE CAPACITY, for the same reason the item count is. On the
        //: `!whole` overflow path the refusing entry is written only `if (combine_at < combine_cap)`,
        //: and an unclamped count would tell the combine to stride over entries that were never written:
        //: `combine.cu` would read `p.combine_tiles[entry]` past the allocation, take an uninitialised
        //: `slot_base` and die on a wild global load -- exactly the class of hole the refusing entry
        //: exists to close, on the one path where the entry itself does not fit.
        p.counts[NUM_COMBINE_TILES] = min(base_combines, p.combine_cap);
        p.counts[MAX_N] = max_n_seen;
        p.counts[OVERFLOW_FLAG] = overflow;
    }
}

//: `a / d` for a >= 0, by a shift when `d` is a power of two (`sh >= 0`).
__device__ __forceinline__ int div_fast(int a, int d, int sh) { return sh >= 0 ? (a >> sh) : a / d; }

// --- the FAST plan kernels -------------------------------------------------------------------------
//
//: Measured with clock64 stamps per phase: the CUPTI floor of a one-block kernel that does nothing is
//: 1.7-1.9 us, one global round trip is 1000-1300 cycles, and every barrier-separated phase of the
//: walks above costs 1000-2000 cycles -- the single-block kernel spends 5.4 k cycles at B = 1, and a
//: multi-block form (every block re-walking every unit) 8.7 k at B = 256 and 13 k at B = 1024. A plan
//: at 2.5 us has about 1.5 k cycles. So the plan is ONE block of 1024 threads with as few dependent
//: phases as the work allows:
//:
//: * `gqa_plan_decode` -- every request has at most ONE query tile (`max_seqlen_q <= tile_tokens`, every
//:   decode row): a request's units are its `h_kv` heads with identical counts in closed form, so the
//:   kernel is one load round trip, the per-request counts, ONE two-level scan and the stores. No
//:   unit-to-request search, no offsets array, no second walk.
//: * `gqa_plan_wide` -- anything else: the single-block walk above with 1024-request chunks and
//:   1024-unit windows (one window for a small prefill batch), two-level scans, and a request search
//:   bounded by the chunk's live requests.
//:
//: Neither changes the emission order (ascending request; within a request descending query tile, and
//: within a tile descending KV head), so every `Item`, `CombineTile` and count is the single-block
//: kernel's, field for field.
//:
//: OVERFLOW WITHOUT A BARRIER. A unit is tested on exactly the counters it adds to (a whole-tree unit on
//: items, a folded one on items, slots and combines) against `prefix + own > cap`, and prefixes are
//: monotone, so the LAST unit adding to a counter ends at its total: some unit overflows iff some total
//: exceeds its cap. Every thread knows the totals after the scan; only an overflowing plan (which the
//: host's bounds make unreachable) takes the slow path for the exact `first_unwritten`.
#ifndef GQA_PLAN_FAST
#define GQA_PLAN_FAST 1
#endif
static constexpr int PDT = 1024;
static constexpr int PDW = PDT / 32;

//: Exclusive scan of three counts and a max over PDT threads: warp shuffles, then warp 0 scans the 32
//: warp partials with shuffles. Two barriers. `ws` is [4][PDW + 1]; row 3 carries the max.
__device__ __forceinline__ void pd_scan3max(const int (&value)[3], int vmax, int (&exclusive)[3],
                                            int (&total)[3], int& tmax, int (*ws)[PDW + 1]) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int running[3] = {value[0], value[1], value[2]};
    int m = vmax;
    #pragma unroll
    for (int step = 1; step < 32; step <<= 1) {
        #pragma unroll
        for (int i = 0; i < 3; ++i) {
            const int other = __shfl_up_sync(0xffffffffu, running[i], step);
            if (lane >= step) running[i] += other;
        }
        m = max(m, __shfl_xor_sync(0xffffffffu, m, step));
    }
    if (lane == 31) {
        #pragma unroll
        for (int i = 0; i < 3; ++i) ws[i][warp] = running[i];
        ws[3][warp] = m;
    }
    __syncthreads();
    if (warp == 0) {
        int x[3], s[3];
        #pragma unroll
        for (int i = 0; i < 3; ++i) s[i] = x[i] = ws[i][lane];
        int wm = ws[3][lane];
        #pragma unroll
        for (int step = 1; step < 32; step <<= 1) {
            #pragma unroll
            for (int i = 0; i < 3; ++i) {
                const int other = __shfl_up_sync(0xffffffffu, s[i], step);
                if (lane >= step) s[i] += other;
            }
            wm = max(wm, __shfl_xor_sync(0xffffffffu, wm, step));
        }
        #pragma unroll
        for (int i = 0; i < 3; ++i) {
            ws[i][lane] = s[i] - x[i];
            if (lane == 31) ws[i][PDW] = s[i];
        }
        if (lane == 0) ws[3][PDW] = wm;
    }
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        exclusive[i] = ws[i][warp] + running[i] - value[i];
        total[i] = ws[i][PDW];
    }
    tmax = ws[3][PDW];
}

//: Exclusive scan of one count over PDT threads (same two-level form). Two barriers.
__device__ __forceinline__ void pd_scan1(int value, int& exclusive, int& total, int* ws) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int running = value;
    #pragma unroll
    for (int step = 1; step < 32; step <<= 1) {
        const int other = __shfl_up_sync(0xffffffffu, running, step);
        if (lane >= step) running += other;
    }
    if (lane == 31) ws[warp] = running;
    __syncthreads();
    if (warp == 0) {
        const int x = ws[lane];
        int s = x;
        #pragma unroll
        for (int step = 1; step < 32; step <<= 1) {
            const int other = __shfl_up_sync(0xffffffffu, s, step);
            if (lane >= step) s += other;
        }
        ws[lane] = s - x;
        if (lane == 31) ws[PDW] = s;
    }
    __syncthreads();
    exclusive = ws[warp] + running - value;
    total = ws[PDW];
}

//: One unit's shape: segment count, run, nodes, and its (items, slots, combines).
struct UnitShape {
    int n, run, nodes;
    bool whole;
    int mine[3];
};
__device__ __forceinline__ UnitShape unit_shape(const PlanParams& p, int border_first, int border_last) {
    UnitShape u;
    const int n_first = border_first <= 0 ? 0
                      : div_fast(border_first + p.seg - 1, p.seg, p.seg_shift);
    u.n = border_last <= 0 ? 0 : div_fast(border_last + p.seg - 1, p.seg, p.seg_shift);
    u.run = 1;
    if (n_first == u.n && p.granularity != GRAN_SINGLE) {
        u.run = p.granularity == GRAN_PAIRS ? 2 : p.max_run;
        if (u.run > p.max_run) u.run = p.max_run;
        if (u.run < 1) u.run = 1;
    }
    u.whole = u.n > 0 && (u.n <= u.run || (u.run == 1 && u.n == 1));
    u.nodes = 0;
    if (u.n > 0 && !u.whole)
        for (int k = 0; k < u.n; k += node_size_at(k, u.n, u.run)) ++u.nodes;
    u.mine[0] = u.n <= 0 ? 0 : (u.whole ? 1 : u.nodes);
    u.mine[1] = u.whole ? 0 : u.nodes * p.tile_rows;
    u.mine[2] = (u.n > 0 && !u.whole) ? 1 : 0;
    return u;
}

__device__ __forceinline__ bool shape_overflows(const PlanParams& p, const UnitShape& u, const int (&at)[3]) {
    if (u.n <= 0) return false;
    if (u.whole) return at[0] >= p.item_cap;
    return at[1] + u.nodes * p.tile_rows > p.slot_cap || at[0] + u.nodes > p.item_cap
        || at[2] >= p.combine_cap;
}

//: Write one unit's items and combine entry at prefix `at` -- the single-block body, verbatim.
__device__ __forceinline__ void emit_unit(const PlanParams& p, const UnitShape& u, const int (&at)[3],
                                          int req, int tok0, int kv_head, int q_end, int used, int ks) {
    if (u.n <= 0) return;
    const bool over = shape_overflows(p, u, at);
    if (u.whole) {
        if (!over) {
            Item it;
            it.req = req; it.tok0 = tok0; it.kv_head = kv_head;
            it.k0 = 0; it.k1 = u.n; it.q_end = q_end; it.used = used;
            it.slot_base = -1; it.ks0 = ks;
            p.items[at[0]] = it;
        }
    } else if (over) {
        if (at[2] < p.combine_cap) {
            CombineTile ct;
            ct.req = req; ct.tok0 = tok0; ct.kv_head = kv_head; ct.slot_base = -1;
            ct.run = 1; ct.q_end = q_end; ct.used = used; ct.pad = 0;
            p.combine_tiles[at[2]] = ct;
        }
    } else {
        int index = 0;
        for (int k = 0; k < u.n;) {
            const int z = node_size_at(k, u.n, u.run);
            Item it;
            it.req = req; it.tok0 = tok0; it.kv_head = kv_head;
            it.k0 = k; it.k1 = k + z; it.q_end = q_end; it.used = used;
            it.slot_base = at[1] + index * p.tile_rows;
            it.ks0 = k == 0 ? ks : -1;
            p.items[at[0] + index] = it;
            ++index;
            k += z;
        }
        CombineTile ct;
        ct.req = req; ct.tok0 = tok0; ct.kv_head = kv_head; ct.slot_base = at[1];
        ct.run = u.run; ct.q_end = q_end; ct.used = used; ct.pad = 0;
        p.combine_tiles[at[2]] = ct;
    }
}

__device__ __forceinline__ int first_key_slot(const PlanParams& p, int r) {
    return p.page > 0 ? p.block_table[int64_t(r) * p.block_table_stride] * p.page : p.cu_seqlens_k[r];
}

//: `counts`, written by one thread once the totals and the exact first unwritten item are known.
__device__ __forceinline__ void write_counts(const PlanParams& p, const int (&tot)[3], int max_n,
                                             bool over, int first_unwritten) {
    p.counts[NUM_ITEMS] = min(min(tot[0], p.item_cap), first_unwritten);
    p.counts[NUM_SLOTS] = tot[1];
    p.counts[NUM_COMBINE_TILES] = min(tot[2], p.combine_cap);
    p.counts[MAX_N] = max_n;
    p.counts[OVERFLOW_FLAG] = over ? 1 : 0;
    #pragma unroll
    for (int i = 0; i < NUM_COUNTS; ++i)
        if (i != NUM_ITEMS && i != NUM_SLOTS && i != NUM_COMBINE_TILES && i != MAX_N && i != OVERFLOW_FLAG)
            p.counts[i] = 0;
}

//: Every request has at most one query tile, so its units are its `h_kv` heads. Thread t owns the
//: CONTIGUOUS units [t * per, t * per + per) of the B * h_kv, per = ceil(B * h_kv / 1024) -- unit
//: u is request u / h_kv, KV head h_kv - 1 - u % h_kv (the reference walk's order) -- so the heads
//: of one request are written by different threads.
//: SLICED (B * h_kv <= 1024): every block computes every unit's closed-form counts from the
//: coalesced lengths and scans them, but resolves the key slot of, and writes the items of, only
//: its own DSL-unit slice. One block's 1024 strided `block_table` reads and 1024 `Item` stores were
//: 1.3 us of B = 1024's 4.5; spread over B / DSL SMs they are not.
static constexpr int DSL = 128;
__global__ __launch_bounds__(PDT, 1) void gqa_plan_decode(__grid_constant__ const PlanParams p) {
    __shared__ int ws[4][PDW + 1];
    __shared__ int s_first, s_max;
    const int tid = threadIdx.x;
    const int U = p.b * p.h_kv;
    const int per = (U + PDT - 1) / PDT;
    const int u0 = min(U, tid * per), u1 = min(U, u0 + per);
    const int s0 = gridDim.x == 1 ? 0 : int(blockIdx.x) * DSL;
    const int s1 = gridDim.x == 1 ? U : min(U, s0 + DSL);
    if (tid == 0) { s_first = 0x7fffffff; s_max = 0; }
    //: A request's shape. Its tile is query tokens [q_begin, q_end), so the tile's last token is
    //: q_end - 1 and its causal borders are `used - len + 1` and `used`. A request longer than one
    //: tile breaks the host's `max_seqlen_q` contract: it emits nothing and flags the overflow.
    auto shape_of = [&](int qb, int qe, int us, bool& bad) {
        const int len = qe - qb;
        const bool live = len > 0 && us > 0;
        bad = live && len > p.tile_tokens;
        UnitShape u = unit_shape(p, us - len + 1, us);
        if (!live || bad) { u.n = 0; u.mine[0] = u.mine[1] = u.mine[2] = 0; }
        return u;
    };
    int sum[3] = {0, 0, 0}, my_max = 0;
    bool my_bad = false;
    int c_qb = 0, c_qe = 0, c_us = 0, c_ks = 0;     // unit u0's request, kept for the emission
    UnitShape c_u;
    c_u.n = 0; c_u.mine[0] = c_u.mine[1] = c_u.mine[2] = 0;
    for (int u = u0; u < u1; ++u) {
        const int r = div_fast(u, p.h_kv, p.hkv_shift);
        const int qb = p.cu_seqlens_q[r], qe = p.cu_seqlens_q[r + 1], us = p.seqused_k[r];
        const int ks = (u >= s0 && u < s1) ? first_key_slot(p, r) : 0;
        bool bad;
        const UnitShape sh = shape_of(qb, qe, us, bad);
        if (u == u0) { c_qb = qb; c_qe = qe; c_us = us; c_ks = ks; c_u = sh; }
        my_bad |= bad;
        sum[0] += sh.mine[0]; sum[1] += sh.mine[1]; sum[2] += sh.mine[2];
        my_max = max(my_max, sh.n);
    }
    int at[3], tot[3], max_n;
    pd_scan3max(sum, my_max | (my_bad ? int(0x40000000) : 0), at, tot, max_n, ws);
    const bool any_bad = (max_n & 0x40000000) != 0;
    //: The flag bit rides the max channel; on that (overflow) path the exact max is recomputed below.
    const bool over = any_bad || tot[0] > p.item_cap || tot[1] > p.slot_cap || tot[2] > p.combine_cap;
    int my_first = 0x7fffffff;
    for (int u = u0; u < u1; ++u) {
        const int r = div_fast(u, p.h_kv, p.hkv_shift);
        int qb = c_qb, qe = c_qe, us = c_us, ks = c_ks;
        UnitShape sh = c_u;
        bool bad = false;
        if (u != u0) {
            qb = p.cu_seqlens_q[r]; qe = p.cu_seqlens_q[r + 1]; us = p.seqused_k[r];
            ks = (u >= s0 && u < s1) ? first_key_slot(p, r) : 0;
            sh = shape_of(qb, qe, us, bad);
        } else {
            const int len = qe - qb;
            bad = len > 0 && us > 0 && len > p.tile_tokens;
        }
        if (bad) my_first = min(my_first, at[0]);
        if (over && shape_overflows(p, sh, at)) my_first = min(my_first, at[0]);
        if (u >= s0 && u < s1)
            emit_unit(p, sh, at, r, qb, p.h_kv - 1 - (u - r * p.h_kv), qe, us, ks);
        at[0] += sh.mine[0]; at[1] += sh.mine[1]; at[2] += sh.mine[2];
    }
    if (over) {                                       // uniform: every thread sees the same totals
        if (my_first != 0x7fffffff) atomicMin(&s_first, my_first);
        if (my_max > 0) atomicMax(&s_max, my_max);
        __syncthreads();
        if (tid == 0 && blockIdx.x == 0) write_counts(p, tot, s_max, true, s_first);
    } else if (tid == 0 && blockIdx.x == 0) {
        write_counts(p, tot, max_n, false, 0x7fffffff);
    }
}

//: Anything else (prefill, mixed batches): the single-block walk with 1024-wide chunks and windows.
__global__ __launch_bounds__(PDT, 1) void gqa_plan_wide(__grid_constant__ const PlanParams p) {
    constexpr int CHUNK = PDT;
    __shared__ int ws[4][PDW + 1], ws1[PDW + 1];
    __shared__ int off[CHUNK + 1];
    __shared__ int sh_q_begin[CHUNK], sh_q_end[CHUNK], sh_used[CHUNK], sh_ks[CHUNK];
    __shared__ int s_first;
    const int tid = threadIdx.x;
    if (tid == 0) s_first = 0x7fffffff;
    int base[3] = {0, 0, 0}, max_n = 0;
    bool over_any = false;
    int my_first = 0x7fffffff;
    for (int chunk = 0; chunk < p.b; chunk += CHUNK) {
        const int live_requests = min(CHUNK, p.b - chunk);
        if (live_requests <= 32) {
            //: A small batch: warp 0 loads and scans the requests alone with
            //: shuffles, and the one barrier below publishes them -- two barriers fewer.
            if (tid < 32) {
                const int r = chunk + tid;
                int units = 0;
                if (tid < live_requests) {
                    const int qb = p.cu_seqlens_q[r], qe = p.cu_seqlens_q[r + 1], us = p.seqused_k[r];
                    sh_q_begin[tid] = qb;
                    sh_q_end[tid] = qe;
                    sh_used[tid] = us;
                    sh_ks[tid] = first_key_slot(p, r);
                    if (qe - qb > 0 && us > 0)
                        units = div_fast(qe - qb + p.tile_tokens - 1, p.tile_tokens, p.tt_shift) * p.h_kv;
                }
                int run = units;
                #pragma unroll
                for (int step = 1; step < 32; step <<= 1) {
                    const int other = __shfl_up_sync(0xffffffffu, run, step);
                    if (tid >= step) run += other;
                }
                const int total_units = __shfl_sync(0xffffffffu, run, 31);
                off[tid] = run - units;
                if (tid == 0) { off[CHUNK] = total_units; off[live_requests] = total_units; }
            }
        } else {
            const int r = chunk + tid;
            int units = 0;
            if (tid < live_requests) {
                const int qb = p.cu_seqlens_q[r], qe = p.cu_seqlens_q[r + 1], us = p.seqused_k[r];
                sh_q_begin[tid] = qb;
                sh_q_end[tid] = qe;
                sh_used[tid] = us;
                sh_ks[tid] = first_key_slot(p, r);
                if (qe - qb > 0 && us > 0)
                    units = div_fast(qe - qb + p.tile_tokens - 1, p.tile_tokens, p.tt_shift) * p.h_kv;
            }
            int excl = 0, total_units = 0;
            pd_scan1(units, excl, total_units, ws1);   // its second barrier publishes the sh_* rows
            off[tid] = excl;
            if (tid == 0) off[CHUNK] = total_units;
            if (tid == 0 && live_requests < CHUNK) off[live_requests] = total_units;
        }
        __syncthreads();
        const int chunk_units = off[CHUNK];
        const int windows = (chunk_units + PDT - 1) / PDT;
        for (int window = 0; window < windows; ++window) {
            const int flat = window * PDT + tid;
            const bool live = flat < chunk_units;
            int local = 0;
            if (live) {
                int lo = 0, hi = live_requests - 1;
                while (lo < hi) {
                    const int mid = (lo + hi + 1) >> 1;
                    if (off[mid] <= flat) lo = mid; else hi = mid - 1;
                }
                local = lo;
            }
            UnitShape u;
            u.n = 0; u.mine[0] = u.mine[1] = u.mine[2] = 0;
            int tok0 = 0, kv_head = 0, q_end = 0, used = 0;
            if (live) {
                const int q_begin = sh_q_begin[local];
                q_end = sh_q_end[local];
                used = sh_used[local];
                const int work = (off[local + 1] - off[local]) - 1 - (flat - off[local]);
                const int tile = div_fast(work, p.h_kv, p.hkv_shift);
                kv_head = work - tile * p.h_kv;
                tok0 = q_begin + tile * p.tile_tokens;
                const int token_last = min(tok0 + p.tile_tokens - 1, q_end - 1);
                u = unit_shape(p, used - q_end + tok0 + 1, used - q_end + token_last + 1);
            }
            int at[3], tot[3], wmax;
            pd_scan3max(u.mine, u.n, at, tot, wmax, ws);
            #pragma unroll
            for (int i = 0; i < 3; ++i) at[i] += base[i];
            if (live) {
                if (shape_overflows(p, u, at)) my_first = min(my_first, at[0]);
                emit_unit(p, u, at, chunk + local, tok0, kv_head, q_end, used, sh_ks[local]);
            }
            #pragma unroll
            for (int i = 0; i < 3; ++i) base[i] += tot[i];
            max_n = max(max_n, wmax);
            //: The scan's barriers order this window's reads of `ws` before the next window's writes.
        }
        __syncthreads();                               // the chunk's sh_* / off reads before the next chunk
    }
    over_any = base[0] > p.item_cap || base[1] > p.slot_cap || base[2] > p.combine_cap;
    if (over_any) {
        if (my_first != 0x7fffffff) atomicMin(&s_first, my_first);
        __syncthreads();
        if (tid == 0) write_counts(p, base, max_n, true, s_first);
    } else if (tid == 0) {
        write_counts(p, base, max_n, false, 0x7fffffff);
    }
}

//: `grid_blocks <= 0` (the test-only binding's switch, and `GQA_PLAN_FAST 0`) runs the reference
//: single-block walk; otherwise one tile per request takes the decode kernel and the rest the wide one.
cudaError_t launch_plan(const PlanParams& p, int grid_blocks, cudaStream_t stream) {
    if (p.b <= 0) return cudaSuccess;
    if (GQA_PLAN_FAST && grid_blocks > 0) {
        if (p.max_q <= p.tile_tokens)
            gqa_plan_decode<<<p.b * p.h_kv <= PDT ? (p.b * p.h_kv + DSL - 1) / DSL : 1, PDT, 0,
                              stream>>>(p);
        else
            gqa_plan_wide<<<1, PDT, 0, stream>>>(p);
    } else {
        gqa_plan_kernel<<<1, PLAN_THREADS, 0, stream>>>(p);
    }
    return cudaGetLastError();
}

}  // namespace gqa_sm100
