// SPDX-License-Identifier: MIT
// Derived from DeepSeek FlashMLA (https://github.com/deepseek-ai/FlashMLA at ba89a34; csrc/kernels/params.h), Copyright (c) 2025 DeepSeek, MIT License.
// Modifications for IsoExec mla_sm90 (position-anchored runs of segments, full-precision nodes, tree combine).
#pragma once

#include <cstdint>

#include <cutlass/bfloat16.h>

namespace mla_sm90::fwd {

// Leaf partial record: O[512] | L | M | pad[2] (float32, log2 frame), never normalized before the merge.
static constexpr int HEAD_DIM_V = 512;
static constexpr int PARTIAL_WIDTH = HEAD_DIM_V + 4;   // 16-byte aligned slots: O[512], L, M, 2 pad

// A work item of the leaf program: request `req`, m_block `m_block` (64 query rows), segments [k0, k1). The run is aligned
// to a node of the canonical tree of every row it publishes: k1 - k0 is a power of two and k0 a multiple of it, or
// [0, n) (the whole tree). Execution only: the worker folds the run's segments in place with exactly the tree's merges.
struct __align__(16) Item {
    int req;
    int m_block;
    int k0;            // first segment
    int k1;            // one past the last segment
    int row_base;      // global row (token*h_q + head) of the m_block's row 0
    int pslot_base;    // partial slot of (row 0, segment 0); -1 if the row block publishes no partial state
    int cost;          // scheduling cost (KV blocks + overheads); the schedule's own bookkeeping
    int next;          // next item of this worker's list, -1 at its end
};
static constexpr int ITEM_FIELDS = 8;
// Combine entry: global row, n, slot of segment 0, slot stride between segments | (log2 node size << 8).
static constexpr int COMBINE_FIELDS = 4;

// Partial slot strides of an m_block (execution only): a full m_block publishes each node as one 64-row tile.
__forceinline__ __host__ __device__ int slot_row_stride(int nrows, int nmax) { return nrows == 64 ? 1 : nmax; }
__forceinline__ __host__ __device__ int slot_seg_stride(int nrows) { return nrows == 64 ? 64 : 1; }

// Node size at segment k when an m_block runs aligned runs of `size` segments (a power of two): full runs, then the
// remainder as descending powers of two. Every such node is a node of the canonical balanced tree over n leaves.
__forceinline__ __host__ __device__ int node_size_at(int k, int n, int size) {
    if (k + size <= n) return size;
    int s = 1;
    while (2*s <= n - k) s *= 2;
    return s;
}

// counts[] layout written by the plan kernel
enum PlanCount : int {
    NUM_ITEMS = 0,
    NUM_SLOTS = 1,
    NUM_COMBINE_ROWS = 2,
    OVERFLOW = 3,
    MAX_N = 4,
    TOTAL_COST = 5,
    TARGET = 6,        // cost target the granularity was chosen against (0: every tree whole)
    NUM_COUNTS = 8,
};

// Granularity (execution only): which aligned runs the plan emits.
enum Granularity : int {
    GRAN_AUTO = 0,     // whole trees when the workers are saturated, else split the longest trees to `target`
    GRAN_TARGET = 1,   // split every tree whose whole cost exceeds `target` (host-computed auto decision)
    GRAN_SINGLE = 2,   // one segment per item
    GRAN_WHOLE = 3,    // one item per tree
    GRAN_PAIRS = 4,    // aligned pairs
    GRAN_MIXED = 5,    // per m_block pseudo-random among single / pairs / whole
};

static constexpr int NUM_BUCKETS = 64;   // cost buckets the schedule orders row blocks by

struct PlanParams {
    int b;
    int h_q;
    int seg_tokens;
    int workers;           // leaf workers the items are balanced over
    int item_cap;
    int slot_cap;
    int combine_cap;
    int mb_cap;            // row-block records the scratch buffer holds
    int granularity;
    int target;            // GRAN_TARGET: cost target
    int item_overhead;     // cost of an item launch (Q load, publish), in KV blocks
    int fold_overhead;     // cost of an in-place segment boundary, in KV blocks
    int cost_cap;          // upper bound of a row block's whole cost (the cost histogram's span)
    int reverse;           // debug: build each worker's list backwards
    int smem_ints;         // int32 slots of dynamic shared memory the plan kernel may use
    int *__restrict__ cu_seqlens_q;   // [b+1]
    int *__restrict__ seqused_k;      // [b]
    Item *__restrict__ items;         // [item_cap]
    int *__restrict__ heads;          // [workers]: first item of each worker's list, -1 if none
    int *__restrict__ combine_rows;   // [combine_cap, 4]
    int *__restrict__ counts;         // [NUM_COUNTS]
    int *__restrict__ scratch;        // [9*mb_cap + 3*workers]: row-block records, their cost order, the worker heap
    cudaStream_t stream;
};

struct FwdParams {
    int h_q;
    int seg_tokens;
    int total_rows;        // T*h_q
    int workers;           // leaf workers launched; worker w walks the item list at heads[w]
    int pdl;               // launched with programmatic dependent launch (a combine kernel follows)
    int publish_rows;      // debug: publish every row by itself (execution only)
    int uniform_rows;      // rows per request when all requests have the same number of rows, else 0
    int stack_depth;       // levels of the per-worker merge stack
    float scale_softmax_log2;

    Item *__restrict__ items;
    int *__restrict__ heads;         // [workers]
    int *__restrict__ counts;
    int *__restrict__ cu_seqlens_q;
    int *__restrict__ seqused_k;
    int *__restrict__ block_table;
    int64_t block_table_stride;
    void *__restrict__ out;          // bf16 [total_rows, 512], row stride 512
    float *__restrict__ lse;         // [total_rows]
    float *__restrict__ partials;    // [slots, PARTIAL_WIDTH]
    float *__restrict__ stack;       // [workers, stack_depth, 64, PARTIAL_WIDTH]: pending tree nodes (O, L, M)

    cudaStream_t stream;
};

struct CombineParams {
    int pdl;               // the leaf kernel was launched with programmatic dependent launch
    int dbg;               // debug (execution only): bit 2 = evaluate the tree as a single group
    int *__restrict__ combine_rows;  // [cap, 4]
    int *__restrict__ counts;
    float *__restrict__ partials;
    void *__restrict__ out;
    float *__restrict__ lse;
    cudaStream_t stream;
};

}  // namespace mla_sm90::fwd
