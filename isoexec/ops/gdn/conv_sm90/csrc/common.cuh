// SPDX-License-Identifier: Apache-2.0
// conv_sm90 numerical recipe v1 (shared by every forward kernel, so decode, prefill and trainer
// evaluate the identical scalar graph for one (token, channel)):
//
//   h3, h2, h1 = the three previous inputs of the same sequence (oldest first), or the carried
//                conv state, or +0.0 where neither exists; x0 = the current input.
//   every operand is widened exactly to fp32;
//   acc = w0*h3;  acc = acc + w1*h2;  acc = acc + w2*h1;  acc = acc + w3*x0   (mul.rn / add.rn, no FMA)
//   acc = acc + bias                                                           (if bias)
//   y   = acc / (1 + expf(-acc))                                               (if SiLU; add.rn, CUDA expf,
//                                                                              __fdividef; no FTZ)
//   out = RNE cast of y to the output dtype.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <type_traits>

namespace conv_sm90 {

using bf16 = __nv_bfloat16;

template <class T> __device__ __forceinline__ float widen(T v) {
  if constexpr (std::is_same_v<T, bf16>)
    return __bfloat162float(v);
  else if constexpr (std::is_same_v<T, __half>)
    return __half2float(v);
  else
    return v;
}

template <class T> __device__ __forceinline__ T narrow(float v) {
  if constexpr (std::is_same_v<T, bf16>)
    return __float2bfloat16_rn(v);
  else if constexpr (std::is_same_v<T, __half>)
    return __float2half_rn(v);
  else
    return v;
}

// Eight contiguous elements, loaded/stored as one packed global memory operation for 16-bit types.
template <class T> struct __align__(16) pack8 {
  T v[8];
};

// Weights [D, 4] row-major: the 32 taps of channels c0..c0+7 are 4 packs of 8 elements for 16-bit
// types (2 channels per pack) and 8 packs of 4 for fp32. Pure data movement.
template <class TW, int CPT>
__device__ __forceinline__ void load_weights(const TW *w, int c0, float (&w0)[CPT], float (&w1)[CPT],
                                             float (&w2)[CPT], float (&w3)[CPT]) {
  if constexpr (sizeof(TW) == 2) {
    const auto *wp = reinterpret_cast<const pack8<TW> *>(w + int64_t(c0) * 4);
#pragma unroll
    for (int p = 0; p < CPT / 2; ++p) {
      pack8<TW> v = wp[p];
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        w0[2 * p + h] = widen(v.v[4 * h]);
        w1[2 * p + h] = widen(v.v[4 * h + 1]);
        w2[2 * p + h] = widen(v.v[4 * h + 2]);
        w3[2 * p + h] = widen(v.v[4 * h + 3]);
      }
    }
  } else {
#pragma unroll
    for (int j = 0; j < CPT; ++j) {
      const TW *wr = w + int64_t(c0 + j) * 4;
      w0[j] = widen(wr[0]);
      w1[j] = widen(wr[1]);
      w2[j] = widen(wr[2]);
      w3[j] = widen(wr[3]);
    }
  }
}

__device__ __forceinline__ float preact(float h3, float h2, float h1, float x0, float w0, float w1,
                                        float w2, float w3) {
  float acc = __fmul_rn(w0, h3);
  acc = __fadd_rn(acc, __fmul_rn(w1, h2));
  acc = __fadd_rn(acc, __fmul_rn(w2, h1));
  acc = __fadd_rn(acc, __fmul_rn(w3, x0));
  return acc;
}

__device__ __forceinline__ float silu(float a) {
  return __fdividef(a, __fadd_rn(1.f, expf(-a)));
}

}  // namespace conv_sm90
