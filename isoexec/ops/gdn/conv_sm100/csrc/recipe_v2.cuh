// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "common.cuh"

namespace conv_sm100 {

/// Recipe v2's pre-activation: v1's tap order with the three additions contracted into `fma.rn`.
__device__ __forceinline__ float preact_fma(float h3, float h2, float h1, float x0, float w0,
                                            float w1, float w2, float w3) {
  float acc = __fmul_rn(w0, h3);
  acc = __fmaf_rn(w1, h2, acc);
  acc = __fmaf_rn(w2, h1, acc);
  acc = __fmaf_rn(w3, x0, acc);
  return acc;
}

/// Dispatch on the declared recipe version. `RECIPE == 1` is `common.cuh`'s graph, unchanged and
/// reached through the identical call, so the default path's generated code cannot drift.
template <int RECIPE>
__device__ __forceinline__ float preact_v(float h3, float h2, float h1, float x0, float w0,
                                          float w1, float w2, float w3) {
  if constexpr (RECIPE == 2)
    return preact_fma(h3, h2, h1, x0, w0, w1, w2, w3);
  else
    return preact(h3, h2, h1, x0, w0, w1, w2, w3);
}

}  // namespace conv_sm100
