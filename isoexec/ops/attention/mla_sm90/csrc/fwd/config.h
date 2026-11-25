// SPDX-License-Identifier: MIT
// Derived from DeepSeek FlashMLA (https://github.com/deepseek-ai/FlashMLA at ba89a34; csrc/kernels/sm90/decode/dense/config.h), Copyright (c) 2025 DeepSeek, MIT License.
// Modified for IsoExec mla_sm90 (deterministic absorbed MLA forward).
#pragma once

namespace mla_sm90::fwd {

namespace Config {

static constexpr int BLOCK_SIZE_M = 64;
static constexpr int PAGE_BLOCK_SIZE = 64;

static constexpr int HEAD_DIM_K = 576;
static constexpr int HEAD_DIM_V = 512;

}

}
