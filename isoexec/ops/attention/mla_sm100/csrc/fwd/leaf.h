// SPDX-License-Identifier: MIT
// IsoExec mla_sm100: host entry of the deterministic absorbed-MLA leaf on Blackwell.
#pragma once

#include <cstdint>
#include "params.h"

namespace mla_sm100::fwd {

struct LeafTensors {
    void *q_ptr;           // bf16 [total_rows, 576] rows (token*h_q + head)
    int64_t total_rows;
    void *k_ptr;           // bf16 [num_blocks, 64, 1, 576]
    int64_t k_row_stride, k_head_stride, k_batch_stride;
    int64_t num_blocks;
    void *out_ptr;         // bf16 [total_rows, 512]
    void *partials_ptr;    // float32 [num_slots, PARTIAL_WIDTH]
    int64_t num_slots;
};

template<typename InputT>
void run_mla_sm100_leaf_kernel(FwdParams &params, const LeafTensors &tensors);

}
