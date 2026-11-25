// SPDX-License-Identifier: MIT
// Derived from DeepSeek FlashMLA (https://github.com/deepseek-ai/FlashMLA at ba89a34; csrc/kernels/sm90/decode/dense/instantiations/bf16.cu), Copyright (c) 2025 DeepSeek, MIT License.
#include "leaf.cuh"
#include "leaf.h"

namespace mla_sm90::fwd {

template void run_mla_sm90_leaf_kernel<cutlass::bfloat16_t>(FwdParams &params, const LeafTensors &tensors);

}
