// SPDX-License-Identifier: MIT
// Derived from DeepSeek FlashMLA (https://github.com/deepseek-ai/FlashMLA at ba89a34, via mla_sm90; csrc/kernels/smxx/decode/combine/combine.h), Copyright (c) 2025 DeepSeek, MIT License.
#pragma once

#include "params.h"

namespace mla_sm100::fwd {

template<typename ElementT>
void run_mla_sm100_combine_kernel(CombineParams &params, int num_entries, int max_n, int config);

}
