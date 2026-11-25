// SPDX-License-Identifier: MIT
// IsoExec sparse_sm100: the index-score kernel on DeepGEMM's SM100 MQA logits (vendored, compiled ahead of time).
#pragma once

#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>

namespace sparse_sm100::mqa {

//: CTAs of every launch and entries of the paged schedule (``NUM_SMS + 1`` start pairs). A fixed count, not the
//: device's: the contiguous kernel strides its Q blocks by it and the paged schedule is cut into it, so any device
//: runs the same work division (B200: one CTA per SM).
constexpr int NUM_SMS = 148;
constexpr int SPLIT_KV = 384;
constexpr int HEAD_DIM = 128;

struct PagedArgs {
    int rows, heads, page_kv;
    const void *q;              // e4m3 [rows, heads, 128]
    const int32_t *q_sf;        // [rows, heads] packed UE8M0 x4
    const uint8_t *cache;       // [pages, page_kv, 1, 132]: page_kv x 128 codes, then page_kv packed UE8M0 x4 words
    int64_t num_pages, page_stride;    // bytes
    const void *weights;        // bf16 [rows, heads], row stride w_stride elements
    int64_t w_stride;
    const int32_t *ke;          // [rows] keys [0, ke) of row r
    const int32_t *block_table; // [sequences, bt_stride] page ids
    int64_t bt_stride;
    const int32_t *indices;     // [rows] row -> block-table row (consecutive equal values share key loads)
    const int32_t *meta;        // [NUM_SMS + 1, 2] (paged_schedule)
    void *out;                  // bf16 [rows, out_stride]
    int64_t out_stride;
};

struct ContigArgs {
    int rows, heads, num_keys;
    const void *q, *weights;
    const int32_t *q_sf;
    int64_t w_stride;
    const void *k;              // e4m3 [num_keys, 128]
    const int32_t *k_sf;        // [num_keys]
    const int32_t *ks, *ke;     // ks nullable (0)
    void *out;                  // bf16 [align(rows, 128 / heads), out_stride]: whole Q blocks are written
    int64_t out_stride;
};

//: gather.cu: a paged pool's keys of the rows' sequences packed contiguous for the contiguous kernel
struct GatherArgs {
    const uint8_t *cache;       // [pages, page_kv, 1, 132]
    int64_t page_stride;        // bytes
    int page_kv;
    const int32_t *block_table; // [seqs, bt_stride] page ids
    int64_t bt_stride;
    int seqs;
    const int32_t *offsets;     // [seqs + 1] (key_ranges)
    int64_t capacity;           // keys the destination holds
    uint8_t *codes;             // [capacity, 128]
    int32_t *sf;                // [capacity]
};

size_t key_ranges_smem_bytes(int seqs);
void key_ranges(int rows, int seqs, int width, const int32_t *ke, const int32_t *rows_seq, int32_t *ks,
                int32_t *ke_out, int32_t *offsets, cudaStream_t stream);
void gather_keys(const GatherArgs &a, cudaStream_t stream);

size_t paged_smem_bytes(int heads);
void paged_schedule(int rows, const int32_t *ke, const int32_t *indices, int32_t *meta, cudaStream_t stream);
void paged_logits(const PagedArgs &a, cudaStream_t stream);
void logits(const ContigArgs &a, cudaStream_t stream);

}  // namespace sparse_sm100::mqa
