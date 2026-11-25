// SPDX-License-Identifier: BSD-3-Clause
// The combine: evaluate the canonical adjacent-pair tree over the nodes a row's items published, and
// apply the one rounding at the root.
//
// A streaming binary counter merges two stack entries whenever the top two have equal size, which
// performs exactly the merges of the canonical tree (level 0 pairs (0,1), (2,3), ...; a node without
// a partner is carried up verbatim). This is the same evaluation mla_sm90/csrc/fwd/combine.cu makes,
// at D = 128 instead of 512.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <algorithm>
#include <vector>
#include <cstdint>

#include "params.h"
#include "numerics.cuh"
#include "launch.h"

namespace gqa_sm100 {

//: Carry the ROW GROUP in `blockIdx.y` instead of flattening it into `blockIdx.x`, so the grid-stride
//: loop needs no integer division. Execution only; default off.
#ifndef GQA_COMBINE_GRID2D
#define GQA_COMBINE_GRID2D 0
#endif


//: One warp per output row; a lane owns EPT contiguous features, so each leaf read is one coalesced
//: warp transaction. MAXDEPTH bounds the canonical tree the entry can carry.
//:
//: GROUP is how many published nodes the kernel loads before it merges any of them. At GROUP == 1
//: the loads are a serial chain -- the counter consumes node k before the loop comes round to issue
//: node k+1 -- and `clock64` stamps put 370 to 634 cycles per node in that wait against 151 to 461
//: in the merge. At GROUP == 4 four loads are in flight and the chain is ceil(n/4) round trips.
//: It moves no bit: the counter's merge sequence is a function of the node SIZES alone, which
//: `node_size_at` fixes, so the same canonical merges run on the same operands with the same left
//: term first. It costs 4 x EPT registers, which is why the host takes it only where occupancy
//: cannot be spent -- see `launch_combine`.
//: The streaming counter's merge stack. `depth` is a RUNTIME index, so a plain array lands in LOCAL
//: memory -- ptxas gives it a 144 to 752 byte frame, and `clock64` stamps priced the counter's own
//: arithmetic at 151 to 461 cycles per node, 23 to 38 % of a warp's cycles.
//:
//: `REG` selects the leaf's `RegStack` pattern instead: every index is a COMPILE-TIME constant behind
//: a predicated move, so the whole stack stays in registers. It is bit-identical -- the same
//: `merge_o_pair` over the same elements with the same left term first, only the storage differs --
//: and it costs about `DEPTH * (EPT + 3)` registers, which is why it is compiled only into the
//: instantiation the host takes where the combine cannot fill one block per SM and there is no
//: occupancy to spend. Above that, occupancy is the binding constraint and the same trade loses; that
//: is the same condition the four-node load group is taken on, and it is why the two travel together.
//:
//: The merge always combines entries `d-2` and `d-1` and writes `d-2`, so the two indices are TIED
//: and one unrolled selection over `a = d-2` covers both operands.
template <int DEPTH, int EPT, bool REG>
struct MergeStack {
    float o[DEPTH][EPT];
    float l[DEPTH], m[DEPTH];
    int z[DEPTH];

    __device__ __forceinline__ void push(int d, const float (&oo)[EPT], float ll, float mm, int zz) {
        if constexpr (!REG) {
            #pragma unroll
            for (int e = 0; e < EPT; ++e) o[d][e] = oo[e];
            l[d] = ll; m[d] = mm; z[d] = zz;
        } else {
            #pragma unroll
            for (int k = 0; k < DEPTH; ++k)
                if (k == d) {
                    #pragma unroll
                    for (int e = 0; e < EPT; ++e) o[k][e] = oo[e];
                    l[k] = ll; m[k] = mm; z[k] = zz;
                }
        }
    }

    //: `z[d-2] == z[d-1]`, the streaming counter's merge condition.
    __device__ __forceinline__ bool top_pair_equal(int d) const {
        if constexpr (!REG) return z[d - 2] == z[d - 1];
        bool r = false;
        #pragma unroll
        for (int a = 0; a < DEPTH - 1; ++a)
            if (a == d - 2) r = (z[a] == z[a + 1]);
        return r;
    }

    //: `entry[d-2] <- merge(entry[d-2], entry[d-1])`, the canonical merge with the LEFT term first.
    __device__ __forceinline__ void merge_top(int d, float scale_log2) {
        if constexpr (!REG) {
            const MergeScales ms = merge_scales(m[d - 2], m[d - 1], scale_log2);
            #pragma unroll
            for (int e = 0; e < EPT; e += 2)
                merge_o_pair(o[d - 2][e], o[d - 2][e + 1], o[d - 1][e], o[d - 1][e + 1],
                             ms.sa, ms.sb);
            l[d - 2] = merge_l(l[d - 2], l[d - 1], ms.sa, ms.sb);
            m[d - 2] = ms.m;
            z[d - 2] += z[d - 1];
        } else {
            #pragma unroll
            for (int a = 0; a < DEPTH - 1; ++a)
                if (a == d - 2) {
                    //: Through TEMPORARIES, and that is load-bearing rather than tidy. `merge_o_pair`
                    //: takes `float&`, and handing it `o[a][e]` directly makes the stack ADDRESSABLE:
                    //: ptxas then keeps the whole array in local memory even though every index is a
                    //: compile-time constant, and `cuobjdump -res-usage` reported exactly the same
                    //: 144-byte frame as the runtime-indexed version. Copying into locals first, and
                    //: writing the result back, is what actually moves it into registers.
                    float x[EPT], y[EPT];
                    #pragma unroll
                    for (int e = 0; e < EPT; ++e) { x[e] = o[a][e]; y[e] = o[a + 1][e]; }
                    const MergeScales ms = merge_scales(m[a], m[a + 1], scale_log2);
                    #pragma unroll
                    for (int e = 0; e < EPT; e += 2)
                        merge_o_pair(x[e], x[e + 1], y[e], y[e + 1], ms.sa, ms.sb);
                    #pragma unroll
                    for (int e = 0; e < EPT; ++e) o[a][e] = x[e];
                    l[a] = merge_l(l[a], l[a + 1], ms.sa, ms.sb);
                    m[a] = ms.m;
                    z[a] += z[a + 1];
                }
        }
    }
};

template <int MAXDEPTH, int EPT, int GROUP = 1>
__global__ __launch_bounds__(128) void gqa_combine_kernel(__grid_constant__ const CombineParams p) {
    //: GRID-STRIDED over `(combine entry, row group)` units, with the grid sized by the MACHINE and
    //: not by the capacity. One block per unit meant the grid was `combine_cap x row_groups`, and the
    //: capacity is what the plan MIGHT emit, not what it did: at `q8192/k8192` the two-wave whole-tree
    //: fold publishes NOTHING (with `tile_tokens` dividing SEG in token space, every segment boundary
    //: falls on a tile boundary and no tile straddles), and the combine still launched 32768 blocks to
    //: read `counts` and exit -- 18.3 us, 5 % of the call, for no work at all. `q2048/k8192` paid 5.5
    //: the same way. Reading the live count ONCE per block and striding turns that into one small
    //: launch, and it costs the rows that do fill the machine nothing: the grid is already a full
    //: residency, and blocks past it were only ever queued.
    const int count = p.counts[NUM_COMBINE_TILES];
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
#if GQA_COMBINE_GRID2D
    //: The row group is `blockIdx.y` and only the ENTRY strides, so the unit index costs no integer
    //: division. The one-dimensional form below flattens `(entry, row group)` into `blockIdx.x` and
    //: pays `unit / groups` and `unit % groups` per iteration -- and `groups` is a runtime field, so
    //: those are real 20-to-40-cycle divisions on a kernel whose decode rows are 10 us of launch floor
    //: and a short dependent chain. The grid is sized the same way either form: `min(capacity,
    //: max(residency, expected))` units in total, here as `blocks.x * groups` with `blocks.x`
    //: carrying the stride.
    for (int entry = int(blockIdx.x); entry < count; entry += int(gridDim.x)) {
    const CombineTile ct = p.combine_tiles[entry];
    //: A REFUSING entry: the plan overflowed its slot capacity for this tile and published nothing,
    //: so there is no node to read. See plan.cu -- this is what keeps a wrong capacity bound from
    //: turning into a wild global load.
    if (ct.slot_base < 0) continue;
    const int r = int(blockIdx.y) * 4 + warp;
    if (r >= p.tile_rows) continue;
#else
    const int groups = p.row_groups;
    const int units = count * groups;
    for (int unit = int(blockIdx.x); unit < units; unit += int(gridDim.x)) {
    const int entry = unit / groups;
    const CombineTile ct = p.combine_tiles[entry];
    //: A REFUSING entry: the plan overflowed its slot capacity for this tile and published nothing,
    //: so there is no node to read. See plan.cu -- this is what keeps a wrong capacity bound from
    //: turning into a wild global load.
    if (ct.slot_base < 0) continue;
    const int r = (unit % groups) * 4 + warp;
    if (r >= p.tile_rows) continue;
#endif

    // The request's two numbers ride in the entry, so the block's dependent-load chain is
    // `counts` -> `combine_tiles[entry]` -> the first node, and not one round trip longer.
    const int q_end = ct.q_end;
    const int ratio = p.head_ratio;
    const int token = ct.tok0 + r / ratio;
    if (token >= q_end) continue;
    const int head = ct.kv_head * ratio + (r % ratio);
    const int border = ct.used - q_end + token + 1;
    const int n = segment_count(border, p.seg);
    if (n <= 0) continue;

    const int64_t slot0 = int64_t(ct.slot_base) + r;
    const int64_t stride = p.tile_rows;
    const int dim = p.dim;
    const int first = lane * EPT;

    //: The streaming tree, one node at a time, straight from global. TWO alternatives were built and
    //: measured and both are rejected: one node of register lookahead, which
    //: is neutral because the tree makes node k+1's USE depend on node k's merges so the loads stay a
    //: chain either way; and staging a window of nodes into shared memory with `cp.async`, which is
    //: neutral on the latency-bound decode rows and 38 % WORSE on `q8192/k8192`, where 16 KiB of
    //: static shared costs the occupancy the bandwidth-bound rows need. The merge stack is indexed
    //: by a runtime `depth`, so ptxas puts it in local memory (a 144 to 752 byte frame, the large end
    //: at D = 256 where every split row goes through this kernel). It stays IN LOCAL MEMORY, and that
    //: is the MEASURED answer rather than the expected one. Declared inside
    //: the grid-stride loop, so each unit starts from an empty counter.
    //:
    //: `REG = (GROUP > 1)` -- the compile-time-indexed stack in the instantiation the host takes where
    //: occupancy cannot be spent -- was built, gated bit-identical, and measured WORSE: `q1/k8192`
    //: 23.63 -> 26.11 us, `q1/k2048` 18.13 -> 18.66, `q16/k2048` 17.70 -> 18.34, and the D256 35B
    //: decode row 21.3 -> 23.0, with every other row flat. It costs ptxas 32 -> 63 registers at EPT = 4
    //: and 70 -> 79 at EPT = 8, and those are exactly the rows the grouped load is compiled for, so
    //: the two register costs compound on the same instantiation. `clock64` agrees: the merge went
    //: 483 cycles per node against 240 on the same row.
    //:
    //: So the stack stays in local memory by choice: it is L1-resident, and the combine's real cost
    //: on the rows the fold leaves it was the EMPTY LAUNCH, which the grid-stride loop above removes.
    MergeStack<MAXDEPTH, EPT, false> st;
    int depth = 0;
    int node = 0;

    //: Push one loaded node and run the streaming counter. Identical at every GROUP.
    auto absorb = [&](const float (&o)[EPT], float l, float m, int z, int& cursor) {
        st.push(depth, o, l, m, z);
        ++depth;
        ++node;
        cursor += z;
        #pragma unroll 1
        while (depth > 1 && (st.top_pair_equal(depth) || cursor == n)) {
            st.merge_top(depth, p.scale_log2);
            --depth;
        }
    };

    //: One coalesced float4 per lane per 4 features. The stack is a local array with no alignment
    //: guarantee, so it is filled scalar-wise from the vector load rather than cast to float4.
    auto fetch = [&](int index, float (&o)[EPT], float& l, float& m) {
        const int64_t slot = slot0 + int64_t(index) * stride;
        const float* src = p.partial_o + slot * dim + first;
        #pragma unroll
        for (int e = 0; e < EPT; e += 4) {
            const float4 piece = *reinterpret_cast<const float4*>(src + e);
            o[e] = piece.x;
            o[e + 1] = piece.y;
            o[e + 2] = piece.z;
            o[e + 3] = piece.w;
        }
        l = p.partial_l[slot];
        m = p.partial_m[slot];
    };

    #pragma unroll 1
    for (int k = 0; k < n;) {
        if constexpr (GROUP == 1) {
            float o[EPT], l, m;
            const int z = node_size_at(k, n, ct.run);
            fetch(node, o, l, m);
            absorb(o, l, m, z, k);
        } else {
            // GROUP independent loads in flight, then the same counter over them in order.
            float o[GROUP][EPT], l[GROUP], m[GROUP];
            int z[GROUP];
            int count = 0, scan = k;
            #pragma unroll
            for (int i = 0; i < GROUP; ++i) {
                if (scan >= n) continue;
                z[i] = node_size_at(scan, n, ct.run);
                fetch(node + i, o[i], l[i], m[i]);
                scan += z[i];
                ++count;
            }
            #pragma unroll
            for (int i = 0; i < GROUP; ++i)
                if (i < count) absorb(o[i], l[i], m[i], z[i], k);
        }
    }

    const float inv = root_inverse(st.l[0]);
    //: Copied out for the same reason the merge uses temporaries: `multiply_pair` takes `float&`.
    float root[EPT];
    #pragma unroll
    for (int e = 0; e < EPT; ++e) root[e] = st.o[0][e];
    auto* out = static_cast<__nv_bfloat16*>(p.out);
    const int64_t obase = (int64_t(token) * p.h_q + head) * dim + first;
    // A lane owns EPT features starting at lane*EPT, so the output offset is only EPT-aligned: at
    // EPT == 4 a 16-byte store would be misaligned and would overlap the next lane. Store 4 features
    // (8 bytes) at a time, which every EPT this kernel serves divides.
    #pragma unroll
    for (int e = 0; e < EPT; e += 4) {
        uint32_t words[2];
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            float lo = root[e + 2 * j], hi = root[e + 2 * j + 1];
            multiply_pair(lo, hi, inv);
            words[j] = pack_bf16_pair(lo, hi);
        }
        *reinterpret_cast<uint2*>(out + obase + e) = *reinterpret_cast<const uint2*>(words);
    }
    if (lane == 0)
        p.lse[int64_t(token) * p.h_q + head] = root_lse(st.l[0], st.m[0], p.scale_log2);
    }
}


//: Blocks for the combine's grid-stride loop. A pure performance choice -- the loop covers every unit
//: whatever the grid is -- and it needs BOTH bounds, because the empty case and the saturated case want
//: opposite grids and no property of the shape separates them:
//:
//: * capped at the CAPACITY, since blocks past the units that exist have nothing to do;
//: * floored at one RESIDENCY, so an under-estimate cannot serialise a kernel that is real work;
//: * otherwise sized by the units the cost model EXPECTS to be live, because where the fold publishes
//:   nothing a capacity-sized grid launched 32768 blocks to read `counts` and exit (18.3 us at
//:   `q8192/k8192`), and where every unit is real work a residency-sized grid cost 8 to 23 % of the
//:   D256 combine to tail imbalance.
int combine_grid_blocks(int64_t capacity, int64_t residency, int64_t expected) {
    int64_t want = std::max(residency, expected);
    want = std::min(capacity, want);
    return int(std::max<int64_t>(1, want));
}

//: The same total block count either way, shaped for whichever indexing the kernel is compiled with.
//: `blocks` is the number of UNITS the grid covers, so the two-dimensional form divides it by the row
//: groups and rounds up -- one extra column of blocks at worst, each of which reads `counts` and exits.
dim3 combine_grid_shape(int blocks, int groups) {
#if GQA_COMBINE_GRID2D
    return dim3(std::max(1, (blocks + groups - 1) / groups), groups, 1);
#else
    (void)groups;
    return dim3(blocks, 1, 1);
#endif
}

cudaError_t launch_combine(const CombineParams& p, int dim, int max_n, int combine_grid,
                          int live_rows, int sms, int pub_tiles, bool group_loads,
                          cudaStream_t stream) {
    const int ept = dim / 32;
    if (ept % 4) return cudaErrorInvalidValue;
    //: `row_groups` covers the rows a tile can actually HAVE, not the rows it is wide. One warp serves
    //: one (token, query head) and four warps a block, so a decode tile -- one token, `ratio` heads --
    //: has `ratio` live rows out of `tile_rows`: at `Hq16/Hkv2` that was 32 row groups to run one, and
    //: at the `Hq2/Hkv1` D256 decode shape 64 to run one. The bound is
    //: `min(max_seqlen_q, tile_tokens) * ratio`, which no request can exceed, and the rows it drops
    //: are exactly the ones the kernel's own `token >= q_end` test dropped.
    const int rows = std::max(1, std::min(p.tile_rows, live_rows));
    const int groups = (rows + 3) / 4;
    CombineParams params = p;
    params.row_groups = groups;
    //: The occupancy API is asked rather than a constant assumed, and the answer is cached per
    //: instantiation because it cannot change inside a process.
    #define GQA_COMBINE_GRID(DEPTH, EPTV, GROUPV)                                    \
        do {                                                                          \
            static int blocks_per_sm = 0;                                             \
            if (blocks_per_sm == 0) {                                                 \
                const cudaError_t s = cudaOccupancyMaxActiveBlocksPerMultiprocessor(  \
                    &blocks_per_sm, gqa_combine_kernel<DEPTH, EPTV, GROUPV>, 128, 0); \
                if (s != cudaSuccess) return s;                                       \
                if (blocks_per_sm < 1) blocks_per_sm = 1;                             \
            }                                                                         \
            const int blocks = combine_grid_blocks(int64_t(combine_grid) * groups,     \
                                                  int64_t(sms) * blocks_per_sm,        \
                                                  int64_t(pub_tiles) * groups);        \
            const dim3 grid = combine_grid_shape(blocks, groups);                       \
            gqa_combine_kernel<DEPTH, EPTV, GROUPV><<<grid, 128, 0, stream>>>(params);  \
        } while (0)
    #define GQA_COMBINE(DEPTH, EPT) GQA_COMBINE_GRID(DEPTH, EPT, 1)
    #define GQA_COMBINE_EPT(DEPTH)                                     \
        do {                                                            \
            if (ept == 4) { GQA_COMBINE(DEPTH, 4); }                    \
            else if (ept == 8) { GQA_COMBINE(DEPTH, 8); }               \
            else { return cudaErrorInvalidValue; }                      \
        } while (0)
    //: The grouped load is compiled only for the shallow tree, which is every geometry the SEG table
    //: declares (`n_max = 16`), and taken only where the host says occupancy cannot be spent.
    if (max_n <= 16 && group_loads) {
        if (ept == 4) { GQA_COMBINE_GRID(5, 4, 4); }
        else if (ept == 8) { GQA_COMBINE_GRID(5, 8, 4); }
        else { return cudaErrorInvalidValue; }
        return cudaGetLastError();
    }
    if (max_n <= 16) { GQA_COMBINE_EPT(5); }
    else if (max_n <= 64) { GQA_COMBINE_EPT(7); }
    else if (max_n <= 256) { GQA_COMBINE_EPT(9); }
    else if (max_n <= 4096) { GQA_COMBINE_EPT(13); }
    else { GQA_COMBINE_EPT(MAX_COMBINE_DEPTH); }
    #undef GQA_COMBINE_EPT
    #undef GQA_COMBINE
    #undef GQA_COMBINE_GRID
    return cudaGetLastError();
}

}  // namespace gqa_sm100
