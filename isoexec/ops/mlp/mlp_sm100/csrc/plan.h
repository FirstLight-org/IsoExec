// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

namespace mlp_sm100 {

struct Plan {
  int threads, vpt, blocks, store;
};

constexpr int SM_COUNT = 148;
constexpr int BLOCKS_PER_SM_TARGET = 4;

inline Plan act_plan(int64_t n8, int64_t width, int64_t rows) {
  int vpt = 1;
  int threads = (n8 >= int64_t(BLOCKS_PER_SM_TARGET) * SM_COUNT * 256) ? 256 : 128;
  if (width % 256 == 0 && width / 8 >= 160 && width / 8 <= 1024 && rows <= 1024)
    threads = int(width / 8);
  int store = 0;
  int64_t want = (n8 + int64_t(threads) * vpt - 1) / (int64_t(threads) * vpt);
  int blocks = int(want < 65536 ? want : 65536);
  if (blocks < 1)
    blocks = 1;
  return {threads, vpt, blocks, store};
}

} // namespace mlp_sm100
