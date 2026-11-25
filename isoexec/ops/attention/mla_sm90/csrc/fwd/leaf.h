// SPDX-License-Identifier: MIT
// Derived from DeepSeek FlashMLA (https://github.com/deepseek-ai/FlashMLA at ba89a34; csrc/kernels/sm90/decode/dense/splitkv_mla.h), Copyright (c) 2025 DeepSeek, MIT License.
#pragma once

#include <cstdint>
#include "params.h"

namespace mla_sm90::fwd {

struct LeafTensors {
    void *q_ptr;           // bf16 [total_rows, 576] rows (token*h_q + head)
    int64_t total_rows;
    void *k_ptr;           // bf16 [num_blocks, 64, 1, 576]
    int64_t k_row_stride, k_head_stride, k_batch_stride;
    int64_t num_blocks;
    void *out_ptr;         // bf16 [total_rows, 512]
    void *partials_ptr;    // float32 [num_slots, PARTIAL_WIDTH]
    int64_t num_slots;
    int64_t uniform_rows;  // rows per request when every request has the same rows, else 0
};

template<typename InputT>
void run_mla_sm90_leaf_kernel(FwdParams &params, const LeafTensors &tensors);

}
