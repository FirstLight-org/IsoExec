// SPDX-License-Identifier: MIT
// IsoExec dsa_sm100: host interface of the deterministic lightning indexer (sm100_dsa_indexer_v1).
#pragma once
#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace dsa_sm100::indexer {

struct IndexerParams {
    int rows;                   // query rows (decode: requests, next_n = 1; prefill: tokens)
    int qb;                     // query rows per block (1 decode, 2 prefill)
    int num_qblocks;
    int splits;                 // key-span items per query block
    int span;                   // 128-key tiles per item (span_auto: 1 for the host's split bound)
    int span_auto;              // > 0: the kernel picks the span (a power of two <= span_auto) from the live tiles
    const int *ks;              // [rows] window start (nullptr: 0)
    const int *ke;              // [rows] window end
    const int *block_table;     // paged: [rows, bt_stride] (or [seqs, bt_stride] with bt_rows)
    int64_t bt_stride;
    const int *bt_rows;         // paged, nullable: query row r reads block-table row bt_rows[r] (else row r)
    bool paged;
    // paged: index-key entries per cache page (one block-table entry per page_rows keys). 128: vLLM's
    // ratio-1 page [128 x 128 codes][128 x 4 scales], key unit lp (64 keys) = page bt[lp >> 1], row
    // half lp & 1. Any other value: the 64-entry page [64 x 128][64 x 4] (the default; address math only).
    int page_rows;
    const __nv_bfloat16 *weights;   // [rows, w_stride]
    int64_t w_stride;
    __nv_bfloat16 *out;         // [rows, out_stride]
    int64_t out_stride;
    // sm100_dsa_indexer_tree_v1 (tree_leaves > 0): heads [0, 4 * tree_leaves) in leaves of 4, each an
    // ascending FP32 chain, combined by the balanced adjacent-pair tree; other heads ignored.
    // out_f32 != nullptr: the FP32 tree root is stored instead of its BF16 rounding.
    int tree_leaves;
    // the top tree_ieee_levels levels of that tree use IEEE FP32 adds (add.rn.f32, no flush to zero,
    // whatever this TU's --ftz) -- the arithmetic of the cross-rank leaf_tree / detreduce completion --
    // and the levels below them the kernel's own FTZ adds. 0 <= tree_ieee_levels <= log2(tree_leaves).
    int tree_ieee_levels;
    float *out_f32;
};

struct IndexerTensors {
    void *q;                    // e4m3 [rows * 64, 128]
    void *q_sf;                 // int32 [rows, 64]
    void *k;                    // paged: the cache base; contiguous: e4m3 [num_keys, 128]
    void *k_sf;                 // paged: cache base + page_rows * 128; contiguous: int32 [num_keys]
    int64_t num_pages, page_stride, num_keys;
};

size_t smem_bytes();
void launch(const IndexerParams &p, const IndexerTensors &t, int grid, cudaStream_t stream);

}  // namespace dsa_sm100::indexer
