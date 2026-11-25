// SPDX-License-Identifier: MIT
// IsoExec sparse_sm100: paged index keys gathered contiguous for the contiguous index-score kernel (integer addressing
// and byte copies only; no value is computed).
//
// key_ranges: per sequence s the key count K[s] = max ke over its rows (clamped to the row width), packed at
// offsets[s] = K[0] + ... + K[s - 1] (offsets[seqs] = total); per row r of sequence s = rows_seq[r]
// ks[r] = offsets[s], ke_out[r] = offsets[s] + ke[r]. Rows outside [0, seqs) get the empty range [total, total).
// gather_keys: key j of the packed buffer (offsets[s] <= j < offsets[s + 1]) is key k = j - offsets[s] of sequence
// s, read from page block_table[s, k / page_kv], entry k % page_kv: 128 code bytes and the scale word.
#include <cuda_runtime.h>
#include <cstdint>

#include "mqa/mqa.h"

namespace sparse_sm100::mqa {

namespace {

constexpr int kRangeThreads = 1024;

__global__ void __launch_bounds__(kRangeThreads, 1)
key_ranges_kernel(int rows, int seqs, int width, const int32_t *ke, const int32_t *rows_seq, int32_t *ks,
                  int32_t *ke_out, int32_t *offsets) {
    extern __shared__ int32_t smem[];
    int32_t *count = smem;                  // [seqs]: K[s], then offsets[s]
    int32_t *part = smem + seqs;            // [kRangeThreads]: per-thread chunk sums, scanned
    const int t = threadIdx.x;
    for (int s = t; s < seqs; s += kRangeThreads) count[s] = 0;
    __syncthreads();
    for (int r = t; r < rows; r += kRangeThreads) {
        const int s = rows_seq[r];
        const int e = min(max(ke[r], 0), width);
        if (s >= 0 && s < seqs && e > 0) atomicMax(&count[s], e);   // integer maximum: order-free
    }
    __syncthreads();
    const int chunk = (seqs + kRangeThreads - 1) / kRangeThreads;
    const int b = min(t * chunk, seqs), e = min(b + chunk, seqs);
    int32_t sum = 0;
    for (int s = b; s < e; ++s) sum += count[s];
    part[t] = sum;
    __syncthreads();
    for (int d = 1; d < kRangeThreads; d <<= 1) {        // inclusive scan of the chunk sums
        const int32_t v = t >= d ? part[t - d] : 0;
        __syncthreads();
        part[t] += v;
        __syncthreads();
    }
    int32_t base = t ? part[t - 1] : 0;
    for (int s = b; s < e; ++s) {
        const int32_t k = count[s];
        count[s] = base;
        offsets[s] = base;
        base += k;
    }
    const int32_t total = part[kRangeThreads - 1];
    if (t == 0) offsets[seqs] = total;
    __syncthreads();
    for (int r = t; r < rows; r += kRangeThreads) {
        const int s = rows_seq[r];
        if (s >= 0 && s < seqs) {
            const int32_t o = count[s];
            ks[r] = o;
            ke_out[r] = o + min(max(ke[r], 0), width);
        } else {
            ks[r] = total;
            ke_out[r] = total;
        }
    }
}

// eight threads per key: 16 code bytes each, the first also the scale word
__global__ void gather_keys_kernel(const uint8_t *cache, int64_t page_stride, int page_kv, const int32_t *block_table,
                                   int64_t bt_stride, int seqs, const int32_t *offsets, int64_t capacity,
                                   uint8_t *codes, int32_t *sf) {
    const int64_t total = min((int64_t)offsets[seqs], capacity);
    const int lane = threadIdx.x & 7;
    const int64_t step = (int64_t(gridDim.x) * blockDim.x) >> 3;
    for (int64_t j = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) >> 3; j < total; j += step) {
        int lo = 0, hi = seqs;             // offsets[lo] <= j < offsets[hi]: the last sequence starting at or before j
        while (hi - lo > 1) {
            const int mid = (lo + hi) >> 1;
            if (offsets[mid] <= j) lo = mid; else hi = mid;
        }
        const int64_t k = j - offsets[lo];
        const int64_t page = block_table[lo * bt_stride + k / page_kv];
        const int64_t entry = k % page_kv;
        const uint8_t *base = cache + page * page_stride;
        reinterpret_cast<uint4 *>(codes + j * HEAD_DIM)[lane] =
            reinterpret_cast<const uint4 *>(base + entry * HEAD_DIM)[lane];
        if (lane == 0) sf[j] = *reinterpret_cast<const int32_t *>(base + int64_t(page_kv) * HEAD_DIM + entry * 4);
    }
}

}  // namespace

size_t key_ranges_smem_bytes(int seqs) { return (size_t(seqs) + kRangeThreads) * sizeof(int32_t); }

void key_ranges(int rows, int seqs, int width, const int32_t *ke, const int32_t *rows_seq, int32_t *ks,
                int32_t *ke_out, int32_t *offsets, cudaStream_t stream) {
    key_ranges_kernel<<<1, kRangeThreads, key_ranges_smem_bytes(seqs), stream>>>(rows, seqs, width, ke, rows_seq, ks,
                                                                                 ke_out, offsets);
}

void gather_keys(const GatherArgs &a, cudaStream_t stream) {
    constexpr int kThreads = 256;
    const int64_t blocks = (a.capacity * 8 + kThreads - 1) / kThreads;
    const int grid = (int)(blocks < 8 * NUM_SMS ? blocks : 8 * NUM_SMS);
    if (grid == 0) return;
    gather_keys_kernel<<<grid, kThreads, 0, stream>>>(a.cache, a.page_stride, a.page_kv, a.block_table, a.bt_stride,
                                                      a.seqs, a.offsets, a.capacity, a.codes, a.sf);
}

}  // namespace sparse_sm100::mqa
