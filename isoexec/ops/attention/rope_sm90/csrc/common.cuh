// SPDX-License-Identifier: Apache-2.0
// Scalar numerics shared by every rope_sm90 kernel.
//
// Every floating operation that can reach a forward output is an explicit
// round-to-nearest intrinsic (__fmul_rn, __fadd_rn, __fsub_rn, __fdiv_rn,
// __fsqrt_rn) or an explicit RN narrowing; the translation unit is also built
// with --fmad=false --ftz=false (see _extension.py). The emitted PTX uses
// mul.rn / add.rn / sub.rn / div.rn / sqrt.rn / cvt.rn only.
#pragma once
#include <assert.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <type_traits>

namespace rope_sm90 {

template <class T> __device__ __forceinline__ float widen(T x) {
  if constexpr (std::is_same_v<T, __nv_bfloat16>)
    return __bfloat162float(x);
  else if constexpr (std::is_same_v<T, __half>)
    return __half2float(x);
  else
    return x;
}

template <class T> __device__ __forceinline__ T narrow(float x) {
  if constexpr (std::is_same_v<T, __nv_bfloat16>)
    return __float2bfloat16_rn(x);
  else if constexpr (std::is_same_v<T, __half>)
    return __float2half_rn(x);
  else
    return x;
}

// Two adjacent elements as one 32-bit load (16-bit types) and exact widening.
template <class T>
__device__ __forceinline__ void rd2(const T *p, float &a, float &b) {
  if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162 *>(p);
    a = __low2float(v);
    b = __high2float(v);
  } else if constexpr (std::is_same_v<T, __half>) {
    __half2 v = *reinterpret_cast<const __half2 *>(p);
    a = __low2float(v);
    b = __high2float(v);
  } else {
    a = p[0];
    b = p[1];
  }
}

// Two adjacent elements, each rounded to nearest even, as one 32-bit store.
template <class T>
__device__ __forceinline__ void wr2(T *p, float a, float b) {
  if constexpr (std::is_same_v<T, __nv_bfloat16>)
    *reinterpret_cast<__nv_bfloat162 *>(p) = __floats2bfloat162_rn(a, b);
  else if constexpr (std::is_same_v<T, __half>)
    *reinterpret_cast<__half2 *>(p) = __floats2half2_rn(a, b);
  else {
    p[0] = narrow<T>(a);
    p[1] = narrow<T>(b);
  }
}

// Bit copy of two adjacent elements.
template <class T>
__device__ __forceinline__ void cp2(T *dst, const T *src) {
  if constexpr (sizeof(T) == 2)
    *reinterpret_cast<uint32_t *>(dst) = *reinterpret_cast<const uint32_t *>(src);
  else
    *reinterpret_cast<uint64_t *>(dst) = *reinterpret_cast<const uint64_t *>(src);
}

// Forward rotation R(theta): y1 = x1*c - x2*s ; y2 = x2*c + x1*s.
// Two RN products, then one RN sub/add.
__device__ __forceinline__ void rotate(float x1, float x2, float c, float s,
                                       float &y1, float &y2) {
  y1 = __fsub_rn(__fmul_rn(x1, c), __fmul_rn(x2, s));
  y2 = __fadd_rn(__fmul_rn(x2, c), __fmul_rn(x1, s));
}

// Transpose R(-theta), used by backward only: y1 = x1*c + x2*s ; y2 = x2*c - x1*s.
__device__ __forceinline__ void rotate_t(float x1, float x2, float c, float s,
                                         float &y1, float &y2) {
  y1 = __fadd_rn(__fmul_rn(x1, c), __fmul_rn(x2, s));
  y2 = __fsub_rn(__fmul_rn(x2, c), __fmul_rn(x1, s));
}

// Position lookup. Table row p is [cos_0..cos_{h-1} | sin_0..sin_{h-1}] (fp32),
// h = rot/2. Column j of token t reads the position of axis a(j): axis 0
// unless bit j of m1 (axis 1) or m2 (axis 2) is set (MRoPE). ps0 is the axis
// stride of positions (0 for 1-D positions), ps1 the token stride.
struct Positions {
  const int64_t *pos;
  const float *table;
  int64_t ps0, ps1, rows;
  uint64_t m1, m2;
  int gshift; // backward only: tokens per weight-gradient partial = 1 << gshift
};

inline bool mrope_mask_extent_ok(int64_t rot, int64_t m1, int64_t m2,
                                int64_t position_dims) {
  // Nonzero 2-D masks select columns by uint64 shifts; j >= 64 is undefined.
  return position_dims != 2 || (m1 | m2) == 0 || rot <= 128;
}

template <bool MROPE>
__device__ __forceinline__ const float *table_row(const Positions &P, int64_t t,
                                                  int j, int rot) {
  int64_t p;
  if constexpr (MROPE) {
    int64_t a = int64_t((P.m1 >> j) & 1) + 2 * int64_t((P.m2 >> j) & 1);
    p = P.pos[a * P.ps0 + t * P.ps1];
  } else {
    p = P.pos[t * P.ps1];
  }
  assert(uint64_t(p) < uint64_t(P.rows));
  return P.table + p * rot;
}

} // namespace rope_sm90
