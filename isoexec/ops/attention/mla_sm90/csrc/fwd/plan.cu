// SPDX-License-Identifier: MIT
// Derived from DeepSeek FlashMLA (https://github.com/deepseek-ai/FlashMLA at ba89a34; csrc/kernels/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu),
// Copyright (c) 2025 DeepSeek, MIT License.
//
// Plan position-anchored SEG runs aligned to canonical tree nodes; scheduling cannot change arithmetic.
// Whole trees use cost-ordered snake assignment when saturated; otherwise a min-heap balances split runs.
// Parallel passes stage lengths and records before the serial schedule to hide dependent-load latency.
#include <cuda_runtime_api.h>
#include <cutlass/cutlass.h>

#include "params.h"
#include "utils.h"
#include "plan.h"

namespace mla_sm90::fwd {

namespace {

constexpr int M_ROWS = 64;      // query rows per m_block
constexpr int N_TOK = 64;       // KV tokens per block
constexpr int LOG_N_TOK = 6;
constexpr int REC = 8;          // ints per row-block record in the scratch buffer
constexpr int PLAN_THREADS = 128;           // threads of the plan kernel's parallel passes
constexpr int SNAKE_ITEMS_PER_WORKER = 2;   // whole trees per worker above which the schedule deals them out snake-wise

}  // namespace

__global__ void __launch_bounds__(PLAN_THREADS, 1, 1)
mla_sm90_plan_kernel(__grid_constant__ const PlanParams params) {
    // Shared memory: the staged lengths and the per-request m_block offsets, then the row-block records, their cost
    // order and (for the serial schedule) the worker heap; whatever does not fit goes to the global scratch buffer.
    extern __shared__ int smem[];
    __shared__ int s_bucket[NUM_BUCKETS];
    __shared__ unsigned long long s_total_whole;
    __shared__ unsigned long long s_total_cost;
    __shared__ int s_max_n, s_live, s_mand_slots, s_mand_comb, s_target, s_items, s_slots, s_comb, s_overflow;
    const int h_q = params.h_q;
    const int SEG = params.seg_tokens;
    const int seg_blocks = SEG >> LOG_N_TOK;
    const int W = params.workers;
    int *const s_cu = smem;                          // [b+1]
    int *const s_used = smem + params.b + 1;         // [b]
    int *const s_mb_off = s_used + params.b;         // [b+1]: exclusive scan of the requests' m_block counts
    const int head_ints = 3*params.b + 2;
    const int need = (REC + 1)*params.mb_cap + 3*W;
    const bool in_smem = need <= params.smem_ints - head_ints;
    int *const rec = in_smem ? smem + head_ints : params.scratch;
    int *const order = rec + REC*params.mb_cap;
    int *const heap_load = order + params.mb_cap;    // [W] loads, [W] workers: the serial schedule's min-heap;
    int *const heap_who = heap_load + W;             //   then [W] list tails
    int *const tails = heap_who + W;

    for (int i = threadIdx.x; i < params.b + 1; i += PLAN_THREADS) {
        s_cu[i] = params.cu_seqlens_q[i];
        if (i < params.b) s_used[i] = params.seqused_k[i];
    }
    for (int j = threadIdx.x; j < NUM_BUCKETS; j += PLAN_THREADS) s_bucket[j] = 0;
    for (int w = threadIdx.x; w < W; w += PLAN_THREADS) params.heads[w] = -1;
    if (threadIdx.x == 0) {
        s_total_whole = 0; s_total_cost = 0; s_max_n = 1; s_live = 0; s_mand_slots = 0; s_mand_comb = 0;
        s_items = 0; s_slots = 0; s_comb = 0; s_overflow = 0;
    }
    __syncthreads();

    // Pass 1 (parallel over requests): m_blocks per request, then an exclusive scan over the requests
    for (int b = threadIdx.x; b < params.b; b += PLAN_THREADS) {
        const int rows = (s_cu[b+1] - s_cu[b])*h_q;
        s_mb_off[b] = (rows + M_ROWS - 1) >> 6;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int acc = 0;
        for (int b = 0; b < params.b; ++b) {
            const int n = s_mb_off[b];
            s_mb_off[b] = acc;
            acc += n;
        }
        s_mb_off[params.b] = acc;
    }
    __syncthreads();
    const int total_mb = s_mb_off[params.b];
    const int cost_scale = max(params.cost_cap, NUM_BUCKETS) / NUM_BUCKETS;   // buckets of the cost bound

    // The row block of a global index: (request, m_block) and everything the schedule needs of it. Integer division is
    // emulated on SM90, but here every row block is a thread's own work, so the latency is hidden.
    #define ROW_BLOCK(g, ...) {                                                                        \
        int lo_ = 0, hi_ = params.b - 1, b = 0;                                                         \
        while (lo_ <= hi_) {                                                                            \
            const int mid = (lo_ + hi_) >> 1;                                                           \
            if (s_mb_off[mid] <= (g)) { b = mid; lo_ = mid + 1; } else { hi_ = mid - 1; }                \
        }                                                                                               \
        const int mb = (g) - s_mb_off[b];                                                               \
        const int q_begin = s_cu[b];                                                                    \
        const int q_len = s_cu[b+1] - q_begin;                                                          \
        const int rows = q_len*h_q;                                                                     \
        const int base = s_used[b] - q_len;                                                             \
        const int fr = mb*M_ROWS;                                                                       \
        const int nrows = min(rows - fr, M_ROWS);                                                       \
        const int row_base = q_begin*h_q + fr;                                                          \
        const int t_lo = fr / h_q;                                                                      \
        const int r_lo = fr - t_lo*h_q;                                                                 \
        const int p0 = base + t_lo;                                                                     \
        const int p1 = base + (fr + nrows - 1) / h_q;                                                   \
        const int nmax = p1 >= 0 ? p1 / SEG + 1 : 0;                                                    \
        const bool live = p1 >= 0;                                                                      \
        const bool uniform = p0 >= 0 && p0 / SEG == p1 / SEG;                                           \
        const int whole_cost = ((p1 + 1 + N_TOK - 1) >> LOG_N_TOK) + params.item_overhead               \
                               + (nmax - 1)*params.fold_overhead;                                       \
        const int bucket_ = min(whole_cost / cost_scale, NUM_BUCKETS - 1);                               \
        const int row_span = max(fr + nrows - max((SEG - base)*h_q, fr), 0);   /* rows with p >= SEG */  \
        __VA_ARGS__                                                                                     \
    }

    // Pass 2 (parallel over row blocks): the records, the cost histogram, the totals, and the slots and combine
    // entries the plan MUST have (a row block whose rows have different trees -- it straddles a segment boundary --
    // can only run as single segments, because a run publishes one node for all of its rows).
    for (int g = threadIdx.x; g < total_mb && g < params.mb_cap; g += PLAN_THREADS) {
        ROW_BLOCK(g, {
            int *const e = rec + REC*g;
            e[0] = live ? nrows : 0; e[1] = mb; e[2] = p1; e[3] = nmax;
            e[4] = uniform; e[5] = t_lo; e[6] = r_lo; e[7] = b;
            if (live) {
                atomicAdd(&s_total_whole, (unsigned long long)whole_cost);
                atomicMax(&s_max_n, nmax);
                atomicAdd(&s_live, 1);
                atomicAdd(&s_bucket[bucket_], 1);
                if (!uniform && nmax > 1) {
                    atomicAdd(&s_mand_slots, nrows*nmax);
                    atomicAdd(&s_mand_comb, row_span);
                }
            }
        })
    }
    __syncthreads();

    // The cost target the schedule splits against (execution only): the mean worker load (with a little slack, since
    // splitting adds an item and a combine), but never below one segment's cost, and no target at all when whole trees
    // already saturate the workers (splitting would only add items, partials and combine work).
    __shared__ int s_pos[NUM_BUCKETS];
    if (threadIdx.x == 0) {
        int64_t target = params.target;
        if (params.granularity == GRAN_AUTO) {
            const int64_t mean = (int64_t)s_total_whole / max(W, 1);
            target = max(mean + mean/32, (int64_t)(seg_blocks + params.item_overhead));
            if (s_live >= SNAKE_ITEMS_PER_WORKER*W) target = 2147483647;
        }
        s_target = (int)min(target, (int64_t)2147483647);
        int acc = 0;
        for (int j = NUM_BUCKETS - 1; j >= 0; --j) {   // descending cost buckets
            s_pos[j] = acc;
            acc += s_bucket[j];
        }
    }
    __syncthreads();
    const int target = s_target;
    const int num_rec = min(s_live, params.mb_cap);
    const bool saturated = target == 2147483647;

    // Pass 3 (parallel over row blocks): order them by descending cost bucket, so that the longest runs are handed
    // out first and land on different workers
    for (int g = threadIdx.x; g < total_mb && g < params.mb_cap; g += PLAN_THREADS) {
        const int *const e = rec + REC*g;
        if (e[0] == 0) continue;     // not live
        const int whole_cost = ((e[2] + 1 + N_TOK - 1) >> LOG_N_TOK) + params.item_overhead
                               + (e[3] - 1)*params.fold_overhead;
        order[atomicAdd(&s_pos[min(whole_cost / cost_scale, NUM_BUCKETS - 1)], 1)] = g;
    }
    __syncthreads();

    if (saturated) {
        // Pass 4 (parallel): every tree is one whole run, except a straddling row block, which runs as single
        // segments. The worker is the snake worker of the row block's cost position; the lists are built by lock-free
        // prepends and the slots and combine entries by atomic cursors (execution only, all of it).
        for (int i = threadIdx.x; i < num_rec; i += PLAN_THREADS) {
            const int round = i / W, slot = i - round*W;
            const int w = (round & 1) ? W - 1 - slot : slot;
            ROW_BLOCK(order[i], {
                const bool split = !uniform && nmax > 1;
                const int count = split ? nmax : 1;
                int pslot = -1;
                if (split) {
                    pslot = atomicAdd(&s_slots, nrows*nmax);
                    const int crow = atomicAdd(&s_comb, row_span);
                    if (pslot + nrows*nmax > params.slot_cap || crow + row_span > params.combine_cap) {
                        atomicExch(&s_overflow, 1);
                    } else {
                        const int row_stride = slot_row_stride(nrows, nmax);
                        const int stride_node = slot_seg_stride(nrows);   // single segments: node size 1
                        int c = crow;
                        for (int r = 0; r < nrows; ++r) {
                            const int p = base + (fr + r) / h_q;
                            if (p >= SEG) {
                                params.combine_rows[c*4] = row_base + r;
                                params.combine_rows[c*4+1] = p / SEG + 1;
                                params.combine_rows[c*4+2] = pslot + r*row_stride;
                                params.combine_rows[c*4+3] = stride_node;
                                ++c;
                            }
                        }
                    }
                }
                const int idx = atomicAdd(&s_items, count);
                if (idx + count > params.item_cap) {
                    atomicExch(&s_overflow, 1);
                } else {
                    int total = 0;
                    for (int j = 0; j < count; ++j) {
                        const int k0 = split ? j : 0;
                        const int k1 = split ? j + 1 : nmax;
                        const int cost = ((min(k1*SEG, p1 + 1) + N_TOK - 1) >> LOG_N_TOK) - k0*seg_blocks
                                         + params.item_overhead + (k1 - k0 - 1)*params.fold_overhead;
                        const Item it = { b, mb, k0, k1, row_base, pslot, cost, j + 1 < count ? idx + j + 1 : -1 };
                        params.items[idx + j] = it;
                        total += cost;
                    }
                    atomicAdd(&s_total_cost, (unsigned long long)total);
                    // Prepend this row block's chain to the worker's list
                    params.items[idx + count - 1].next = atomicExch(&params.heads[w], idx);
                }
            })
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            params.counts[NUM_ITEMS] = min(s_items, params.item_cap);
            params.counts[NUM_SLOTS] = s_slots;
            params.counts[NUM_COMBINE_ROWS] = min(s_comb, params.combine_cap);
            params.counts[OVERFLOW] = s_overflow || total_mb > params.mb_cap;
            params.counts[MAX_N] = s_max_n;
            params.counts[TOTAL_COST] = (int)min((unsigned long long)2147483647, s_total_cost);
            params.counts[TARGET] = 0;
        }
        return;
    }

    // Serial schedule (one thread): the workers are not saturated by whole trees, so a tree whose cost would push the
    // least loaded worker past the target is split into aligned runs and every run goes to the least loaded worker.
    // There are few row blocks in this regime, which is why one thread can afford it.
    if (threadIdx.x != 0) return;
    for (int w = 0; w < W; ++w) {
        heap_load[w] = 0;
        heap_who[w] = w;
        tails[w] = -1;
    }
    int ni = 0, ns = 0, nc = 0, overflow = (total_mb > params.mb_cap) || s_overflow;
    int mand_slots = s_mand_slots, mand_comb = s_mand_comb;
    int64_t total_cost = 0;

    // Append an item to worker w's list (or prepend it, so the worker walks its items backwards)
    #define EMIT_ITEM(w, req_, mb_, k0_, k1_, row_base_, pslot_, cost_) do {                        \
        if (ni < params.item_cap) {                                                                 \
            Item it = { (req_), (mb_), (k0_), (k1_), (row_base_), (pslot_), (cost_), -1 };          \
            if (params.reverse) {                                                                   \
                it.next = params.heads[w];                                                          \
                params.items[ni] = it;                                                              \
                params.heads[w] = ni;                                                               \
            } else {                                                                                \
                params.items[ni] = it;                                                              \
                if (tails[w] < 0) params.heads[w] = ni;                                             \
                else params.items[tails[w]].next = ni;                                              \
                tails[w] = ni;                                                                      \
            }                                                                                       \
            total_cost += (cost_);                                                                  \
            ++ni;                                                                                   \
        } else {                                                                                    \
            overflow = 1;                                                                           \
        }                                                                                           \
    } while (0)

    // Sift the min-heap's root down after its load grew
    #define HEAP_UPDATE_MIN(new_load) do {                                                          \
        int i = 0;                                                                                  \
        heap_load[0] = (new_load);                                                                  \
        while (true) {                                                                              \
            const int l = 2*i + 1, r = 2*i + 2;                                                     \
            int m = i;                                                                              \
            if (l < W && heap_load[l] < heap_load[m]) m = l;                                        \
            if (r < W && heap_load[r] < heap_load[m]) m = r;                                        \
            if (m == i) break;                                                                      \
            const int tl = heap_load[i]; heap_load[i] = heap_load[m]; heap_load[m] = tl;            \
            const int tw = heap_who[i]; heap_who[i] = heap_who[m]; heap_who[m] = tw;                \
            i = m;                                                                                  \
        }                                                                                           \
    } while (0)

    for (int idx = 0; idx < num_rec; ++idx) {
        const int *const e = rec + REC*order[idx];
        const int nrows = e[0];
        const int mb = e[1];
        const int p1 = e[2];
        const int nmax = e[3];
        const bool uniform = e[4];
        const int t_lo = e[5];
        const int r_lo = e[6];
        const int b = e[7];
        const int fr = mb*M_ROWS;
        const int q_begin = s_cu[b];
        const int q_len = s_cu[b+1] - q_begin;
        const int base = s_used[b] - q_len;
        const int row_base = q_begin*h_q + fr;
        const int row_span = max(fr + nrows - max((SEG - base)*h_q, fr), 0);   // rows with p >= SEG
        const int whole_cost = ((p1 + 1 + N_TOK - 1) >> LOG_N_TOK) + params.item_overhead
                               + (nmax - 1)*params.fold_overhead;

        // Run size (execution only): 0 = the whole tree, else a power of two below nmax (see node_size_at)
        int size = 0;
        if (nmax > 1) {
            switch (params.granularity) {
            case GRAN_SINGLE: size = 1; break;
            case GRAN_WHOLE: size = 0; break;
            case GRAN_PAIRS: size = 2; break;
            case GRAN_MIXED: {
                const unsigned hsh = (unsigned)b*2654435761u ^ (unsigned)mb*40503u;
                const int pick = (int)((hsh >> 7) % 3);
                size = pick == 0 ? 1 : (pick == 1 ? 2 : 0);
                break;
            }
            default:
                // Split only when the whole tree would push the least loaded worker past the target
                if (heap_load[0] + whole_cost <= target) {
                    size = 0;
                } else {
                    size = 1;
                    while (2*size < nmax &&
                           heap_load[0] + 2*size*seg_blocks + params.item_overhead
                               + (2*size - 1)*params.fold_overhead <= target) {
                        size *= 2;
                    }
                }
                break;
            }
            if (!uniform) size = 1;    // rows with different trees: one segment per item
            if (size >= nmax) size = 0;
            if (size != 0) {
                // Splitting needs slots and combine entries; keep what the mandatory splits still need
                if (uniform) {
                    if (ns + nrows*nmax > params.slot_cap - mand_slots || nc + row_span > params.combine_cap - mand_comb) {
                        size = 0;
                    }
                } else {
                    mand_slots -= nrows*nmax;
                    mand_comb -= row_span;
                    if (ns + nrows*nmax > params.slot_cap || nc + row_span > params.combine_cap) overflow = 1;
                }
            }
        }

        int pslot = -1;
        if (size != 0) {
            // Partial slots and combine entries of this row block's published nodes
            pslot = ns;
            ns += nrows*nmax;
            const int row_stride = slot_row_stride(nrows, nmax);
            int stride_node = slot_seg_stride(nrows);
            for (int sz = size; sz > 1; sz >>= 1) stride_node += 1 << 8;
            int t = t_lo, r = r_lo, n = 1, n_end = SEG;
            while (base + t >= n_end) { ++n; n_end += SEG; }
            for (int i = 0; i < nrows; ++i) {
                if (base + t >= SEG) {
                    if (nc < params.combine_cap) {
                        params.combine_rows[nc*4] = row_base + i;
                        params.combine_rows[nc*4+1] = n;
                        params.combine_rows[nc*4+2] = pslot + i*row_stride;
                        params.combine_rows[nc*4+3] = stride_node;
                    } else {
                        overflow = 1;   // a published node with no combine entry: the capacities were too small
                    }
                    ++nc;
                }
                ++r;
                if (r >= h_q) { r -= h_q; ++t; while (base + t >= n_end) { ++n; n_end += SEG; } }
            }
        }

        int k = 0;
        while (k < nmax) {
            const int sz = size == 0 ? nmax : node_size_at(k, nmax, size);
            const int cost = ((min((k + sz)*SEG, p1 + 1) + N_TOK - 1) >> LOG_N_TOK) - k*seg_blocks
                             + params.item_overhead + (sz - 1)*params.fold_overhead;
            const int who = heap_who[0];
            const int load = heap_load[0];
            EMIT_ITEM(who, b, mb, k, k + sz, row_base, pslot, cost);
            HEAP_UPDATE_MIN(load + cost);
            k += sz;
        }
    }

    params.counts[NUM_ITEMS] = min(ni, params.item_cap);
    params.counts[NUM_SLOTS] = ns;
    params.counts[NUM_COMBINE_ROWS] = min(nc, params.combine_cap);
    params.counts[OVERFLOW] = overflow;
    params.counts[MAX_N] = s_max_n;
    params.counts[TOTAL_COST] = (int)min(total_cost, (int64_t)2147483647);
    params.counts[TARGET] = target >= 2147483647 ? 0 : target;
    #undef EMIT_ITEM
    #undef HEAP_UPDATE_MIN
    #undef ROW_BLOCK
}

void run_mla_sm90_plan_kernel(PlanParams &params) {
    const size_t smem = (size_t)params.smem_ints*sizeof(int);
    // Opt-in uses static + dynamic shared memory; testing dynamic size alone can exceed 48 KB.
    cudaFuncAttributes attr{};
    CHECK_CUDA(cudaFuncGetAttributes(&attr, mla_sm90_plan_kernel));
    if (smem + attr.sharedSizeBytes > 48*1024) {
        CHECK_CUDA(cudaFuncSetAttribute(mla_sm90_plan_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    }
    mla_sm90_plan_kernel<<<1, PLAN_THREADS, smem, params.stream>>>(params);
    CHECK_CUDA_KERNEL_LAUNCH();
}

}  // namespace mla_sm90::fwd
