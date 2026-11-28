// SPDX-License-Identifier: Apache-2.0
// Shared device helpers for the SM100 four-stream mHC kernels (GLM and DeepSeek recipes).
//
// Every floating-point operation on a numerical path is an explicit _rn intrinsic (or a PTX op with its
// rounding spelled); the build adds --fmad=false --ftz=false. The only fused multiply-adds are inside the
// tensor core (`mma_bf16`), whose products are exact (bf16 x bf16, or bf16 pieces of an fp32 operand).
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace mhc {

constexpr int kStreams = 4;   // n: published residual-stream count (both models)
constexpr int kOut = 24;      // n*n + 2n projection outputs: pre(4) | post(4) | comb(16)
constexpr int kCols = 25;     // + the sum of squares column

__device__ __forceinline__ uint32_t smem_u32(const void *p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async16(void *dst, const void *src, bool valid) {
  const int sz = valid ? 16 : 0;   // src-size 0 zero-fills the 16 bytes (rows beyond T)
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(smem_u32(dst)), "l"(src), "r"(sz) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_all;" ::: "memory"); }

// D = A * B + D, m16n8k16, bf16 inputs, fp32 accumulate.
__device__ __forceinline__ void mma_bf16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
               "{%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// D = A * B + D, m16n8k8 (A: rows g / g+8 x k 2t..2t+1; B: k 2t..2t+1 x n g), bf16 inputs, fp32 accumulate.
__device__ __forceinline__ void mma_bf16_k8(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t b0) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a0), "r"(a1), "r"(b0));
}

__device__ __forceinline__ float bf_lo(uint32_t v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float bf_hi(uint32_t v) { return __uint_as_float(v & 0xffff0000u); }
__device__ __forceinline__ uint32_t pack_bf16(__nv_bfloat16 lo, __nv_bfloat16 hi) {
  return (uint32_t)__bfloat16_as_ushort(lo) | ((uint32_t)__bfloat16_as_ushort(hi) << 16);
}
__device__ __forceinline__ uint32_t pack_f2_bf16(float lo, float hi) {
  return pack_bf16(__float2bfloat16_rn(lo), __float2bfloat16_rn(hi));
}

// Exact three-piece bf16 split of an fp32 value: f == p0 + p1 + p2 (each subtraction is exact).
__device__ __forceinline__ void split3(float f, __nv_bfloat16 &p0, __nv_bfloat16 &p1, __nv_bfloat16 &p2) {
  p0 = __float2bfloat16_rn(f);
  const float r = __fsub_rn(f, __bfloat162float(p0));
  p1 = __float2bfloat16_rn(r);
  p2 = __float2bfloat16_rn(__fsub_rn(r, __bfloat162float(p1)));
}

// Exact three-piece bf16 split by truncation: f == p0 + p1 + p2, where p0 keeps the top 8 significand bits, p1 the next 8 and
// p2 the last 8 (each subtraction is exact). The bf16 encoding of a piece is the top 16 bits of the float holding it.
// Pack the pieces of two consecutive elements (a, b) as bf16x2: piece k -> out[k] = (a_k.top16) | (b_k.top16 << 16).
__device__ __forceinline__ void split3_trunc_pair(float a, float b, uint32_t (&out)[3]) {
  const float a0 = __uint_as_float(__float_as_uint(a) & 0xffff0000u), b0 = __uint_as_float(__float_as_uint(b) & 0xffff0000u);
  const float ar = __fsub_rn(a, a0), br = __fsub_rn(b, b0);
  const float a1 = __uint_as_float(__float_as_uint(ar) & 0xffff0000u), b1 = __uint_as_float(__float_as_uint(br) & 0xffff0000u);
  const float a2 = __fsub_rn(ar, a1), b2 = __fsub_rn(br, b1);   // <= 8 significant bits: exactly a bf16
  asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(out[0]) : "r"(__float_as_uint(a0)), "r"(__float_as_uint(b0)));
  asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(out[1]) : "r"(__float_as_uint(a1)), "r"(__float_as_uint(b1)));
  asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(out[2]) : "r"(__float_as_uint(a2)), "r"(__float_as_uint(b2)));
}

__device__ __forceinline__ float sigmoid_rn(float a) {
  return __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-a)));
}

}  // namespace mhc

namespace mhc {

__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void *p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm_x2(uint32_t (&r)[2], const void *p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
               : "=r"(r[0]), "=r"(r[1]) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm_x4_trans(uint32_t (&r)[4], const void *p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm_x2_trans(uint32_t (&r)[2], const void *p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
               : "=r"(r[0]), "=r"(r[1]) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void cp_async_wait_group0() { asm volatile("cp.async.wait_group 0;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_async_wait_group_n() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }
__device__ __forceinline__ void cp_async_wait_group1_() { asm volatile("cp.async.wait_group 1;" ::: "memory"); }

// Balanced adjacent-pair tree over v[0..n): level by level, (v0+v1), (v2+v3), ...; an odd last element is carried
// to the next level unchanged. The shape depends only on n.
template <int MAXN>
__device__ __forceinline__ float tree_sum(float (&v)[MAXN], int n) {
  while (n > 1) {
    const int h = n >> 1;
#pragma unroll
    for (int i = 0; i < MAXN / 2; ++i)
      if (i < h) v[i] = __fadd_rn(v[2 * i], v[2 * i + 1]);
    if (n & 1) v[h] = v[n - 1];
    n = h + (n & 1);
  }
  return v[0];
}

}  // namespace mhc
