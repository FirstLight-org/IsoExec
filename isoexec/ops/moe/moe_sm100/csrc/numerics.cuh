// SPDX-License-Identifier: Apache-2.0
// The MoE numerical program's scalar vocabulary.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <type_traits>

namespace moe100 {

template <class T> __device__ __forceinline__ T cast(float x) {
  if constexpr (std::is_same_v<T, __nv_bfloat16>)
    return __float2bfloat16_rn(x);
  else if constexpr (std::is_same_v<T, __half>)
    return __float2half_rn(x);
  else
    return x;
}

//: round an fp32 value to the operand dtype and back, RNE. The segment-result and
//: activation rounding sites both use it.
template <class T> __device__ __forceinline__ float rounded(float x) {
  if constexpr (std::is_same_v<T, __nv_bfloat16>)
    return __bfloat162float(__float2bfloat16_rn(x));
  else
    return __half2float(__float2half_rn(x));
}

//: exact upcast of an operand-dtype element to fp32.
template <class T> __device__ __forceinline__ float upcast(T x) {
  if constexpr (std::is_same_v<T, __nv_bfloat16>)
    return __bfloat162float(x);
  else if constexpr (std::is_same_v<T, __half>)
    return __half2float(x);
  else
    return x;
}

template <class T> __device__ __forceinline__ uint32_t pack_leaf(float a, float b) {
  if constexpr (std::is_same_v<T, __nv_bfloat16>)
    return uint32_t(__bfloat16_as_ushort(__float2bfloat16_rn(a))) |
           (uint32_t(__bfloat16_as_ushort(__float2bfloat16_rn(b))) << 16);
  else
    return uint32_t(__half_as_ushort(__float2half_rn(a))) |
           (uint32_t(__half_as_ushort(__float2half_rn(b))) << 16);
}

template <class T> __device__ __forceinline__ float unpack_leaf(uint32_t pair, int high) {
  uint16_t bits = pair >> (high * 16);
  if constexpr (std::is_same_v<T, __nv_bfloat16>)
    return __bfloat162float(__ushort_as_bfloat16(bits));
  else
    return __half2float(__ushort_as_half(bits));
}

// The FC1 fused epilogue, exactly as moe_sm90/csrc/chunk_fc1.cuh spells it: SiLU
// through CUDA expf and __fdividef, then the up product, then the fp32 probability,
// each rounded to the operand dtype on its own.
template <class T>
__device__ __forceinline__ float glu_epilogue(float gate_acc, float up_acc, float prob,
                                              bool has_prob) {
  float g = rounded<T>(gate_acc), up = rounded<T>(up_acc);
  float h = rounded<T>(__fdividef(g, __fadd_rn(1.f, expf(-g))));
  h = rounded<T>(__fmul_rn(h, up));
  if (has_prob)
    h = rounded<T>(__fmul_rn(h, prob));
  return h;
}

// The FC2 balanced tree over L rounded segment results, lower-index subtrees on the
// left. L in {1,2,4,8}; for eight leaves this is
// ((l0+l1)+(l2+l3)) + ((l4+l5)+(l6+l7)).
template <class T, int L> __device__ __forceinline__ float leaf_tree(const float *seg) {
  float node[L];
#pragma unroll
  for (int i = 0; i < L; ++i)
    node[i] = rounded<T>(seg[i]);
#pragma unroll
  for (int width = 1; width < L; width *= 2)
#pragma unroll
    for (int i = 0; i < L; i += 2 * width)
      node[i] = __fadd_rn(node[i], node[i + width]);
  return node[0];
}

} // namespace moe100
