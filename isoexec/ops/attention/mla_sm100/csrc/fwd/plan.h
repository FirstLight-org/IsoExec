// SPDX-License-Identifier: MIT
// Derived from DeepSeek FlashMLA (https://github.com/deepseek-ai/FlashMLA at ba89a34, via mla_sm90; csrc/kernels/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.h), Copyright (c) 2025 DeepSeek, MIT License.
#pragma once

#include "params.h"

namespace mla_sm100::fwd {

void run_mla_sm100_plan_kernel(PlanParams &params);

}
