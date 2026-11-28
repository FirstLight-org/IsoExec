// SPDX-License-Identifier: MIT
// IsoExec dsa_sm100: host interface of the sparse-attention leaf (recipe sm100_dsa_sparse_segtree_v1).
#pragma once
#include <cstdint>
#include <torch/types.h>
#include <cuda_runtime.h>

namespace dsa_sm100::attn {

struct LeafParams {
    // query rows: q[t, h, 0..512) bf16, row stride q_row_stride elements between tokens
    const void *q;
    int64_t q_tok_stride, q_head_stride;
    int num_tokens;
    int h_q;                       // heads of the tensor; the CTA owns 64 of them
    // the fp8 V4.1 cache [num_pages, 64, 1, 528] (contiguous rows of 528 B)
    const void *cache;
    int64_t num_pages;
    const int *block_table;        // [rows, bt_stride] page ids
    int64_t bt_stride;
    const int *bt_row;             // [num_tokens] block-table row of each token (nullptr: the token index)
    // the list: idx[t, 0..topk) ascending logical positions, -1 pads; topk_length[t] optional
    const int *idx;
    int64_t idx_stride;
    int topk;
    const int *topk_length;
    // groups (SPEC 4.4): positions [g * seg, (g+1) * seg) form group g; G = topk / seg
    int seg;
    float scale_log2;
    // outputs: root rows (G == 1) or fp32 partials (G > 1)
    void *out;                     // bf16 [num_tokens, h_q, 512]
    float *lse;                    // [num_tokens, h_q]
    float *partials;               // [num_tokens, G, h_q, 520] fp32: O[512], L, M, pad (G > 1)
    int fold;                      // 0: groups published as partials (G workers); 1: the in-place tree (one worker)
    int recipe;                    // 1: sm100_dsa_sparse_segtree_v1 (exact running max); 2: sm100_dsa_sparse_v2 (lazy rescale)
    int dqk;                       // 512 or 576 (576: bf16 kv only)
    int kv_bf16;                   // 1: `cache` is a contiguous bf16 kv [kv_rows, 512] and idx are its rows (prefill)
    int64_t kv_rows;
    int *err;                      // optional: bit 0 set by a CTA whose list is not ascending with trailing pads
    const float *sink;             // optional (G == 1, root published): learned-sink root fused into the epilogue
};

void launch_leaf_h64(const LeafParams &p, cudaStream_t stream);
void launch_combine(const float *partials, int G, int h_q, int rows, void *out, float *lse, cudaStream_t s);

void launch_sink_combine(const float *partials, int G, int h_q, int rows, void *out, float *lse,
                         const float *sink, cudaStream_t stream);
void sink_grad(const torch::Tensor &out, const torch::Tensor &dout, const torch::Tensor &lse,
               const torch::Tensor &sink, torch::Tensor &grad);

}  // namespace dsa_sm100::attn
