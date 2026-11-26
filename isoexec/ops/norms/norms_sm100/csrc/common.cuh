// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <type_traits>

namespace norms_sm100 {

__device__ __forceinline__ void split8(float *v, uint4 q) {
  asm volatile("{ .reg .b16 a; .reg .b16 b; .reg .b16 c; .reg .b16 d; "
               ".reg .b16 e; .reg .b16 f; .reg .b16 g; .reg .b16 h; "
               "mov.b32 {a, b}, %8; mov.b32 {c, d}, %9; mov.b32 {e, f}, %10; mov.b32 {g, h}, %11; "
               "cvt.f32.bf16 %0, a; cvt.f32.bf16 %1, b; cvt.f32.bf16 %2, c; cvt.f32.bf16 %3, d; "
               "cvt.f32.bf16 %4, e; cvt.f32.bf16 %5, f; cvt.f32.bf16 %6, g; cvt.f32.bf16 %7, h; }"
               : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]), "=f"(v[4]), "=f"(v[5]),
                 "=f"(v[6]), "=f"(v[7])
               : "r"(q.x), "r"(q.y), "r"(q.z), "r"(q.w));
}

__device__ __forceinline__ uint32_t pack2(float a, float b) {
  uint32_t r;
  asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(r) : "f"(a), "f"(b));
  return r;
}

__device__ __forceinline__ void load8(float *v, const __nv_bfloat16 *row, int k, int L, int lane) {
  split8(v, reinterpret_cast<const uint4 *>(row + k * 8 * L + 8 * lane)[0]);
}

__device__ __forceinline__ void load8u(float *v, const __nv_bfloat16 *row, int k, int lane, int L) {
  const uint32_t *p = reinterpret_cast<const uint32_t *>(row + k * 8 * L + 8 * lane);
  split8(v, make_uint4(p[0], p[1], p[2], p[3]));
}

__device__ __forceinline__ void load8f(float *v, const float *row, int k, int L, int lane) {
  const uint4 *p = reinterpret_cast<const uint4 *>(row + k * 8 * L + 8 * lane);
  uint4 a = p[0], b = p[1];
  asm volatile("{ mov.b32 %0, %8; mov.b32 %1, %9; mov.b32 %2, %10; mov.b32 %3, %11; "
               "mov.b32 %4, %12; mov.b32 %5, %13; mov.b32 %6, %14; mov.b32 %7, %15; }"
               : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]), "=f"(v[4]), "=f"(v[5]),
                 "=f"(v[6]), "=f"(v[7])
               : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b.x), "r"(b.y), "r"(b.z), "r"(b.w));
}

__device__ __forceinline__ void store8(__nv_bfloat16 *row, int k, int L, int lane, const float *y) {
  reinterpret_cast<uint4 *>(row + k * 8 * L + 8 * lane)[0] =
      make_uint4(pack2(y[0], y[1]), pack2(y[2], y[3]), pack2(y[4], y[5]), pack2(y[6], y[7]));
}

__device__ __forceinline__ void store8f(float *row, int k, int L, int lane, const float *y) {
  uint32_t u[8];
  asm volatile("{ mov.b32 %0, %8; mov.b32 %1, %9; mov.b32 %2, %10; mov.b32 %3, %11; "
               "mov.b32 %4, %12; mov.b32 %5, %13; mov.b32 %6, %14; mov.b32 %7, %15; }"
               : "=r"(u[0]), "=r"(u[1]), "=r"(u[2]), "=r"(u[3]), "=r"(u[4]), "=r"(u[5]),
                 "=r"(u[6]), "=r"(u[7])
               : "f"(y[0]), "f"(y[1]), "f"(y[2]), "f"(y[3]), "f"(y[4]), "f"(y[5]), "f"(y[6]),
                 "f"(y[7]));
  uint4 *p = reinterpret_cast<uint4 *>(row + k * 8 * L + 8 * lane);
  p[0] = make_uint4(u[0], u[1], u[2], u[3]);
  p[1] = make_uint4(u[4], u[5], u[6], u[7]);
}

__device__ __forceinline__ void prefetch_l2(const void *p) {
  asm volatile("prefetch.global.L2 [%0];" ::"l"(__cvta_generic_to_global(p)));
}

__device__ __forceinline__ void cp_async16(void *smem, const void *gmem) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(
                   (uint32_t)__cvta_generic_to_shared(smem)),
               "l"(__cvta_generic_to_global(gmem))
               : "memory");
}

__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }

template <int N> __device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
}

__device__ __forceinline__ void store_if(float *p, float v, int cond) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %2, 0; @p st.global.f32 [%0], %1; }"
               :: "l"((uint64_t)__cvta_generic_to_global(p)), "f"(v), "r"(cond) : "memory");
}

__device__ __forceinline__ float warp_fold(float acc, int L) {
  for (int s = L >> 1; s > 0; s >>= 1)
    acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, s));
  return acc;
}

// One `__syncthreads` per call and no reuse of the buffer: forward folds each row once.
template <int L> __device__ __forceinline__ float wide_fold(float acc) {
  constexpr int W = L / 32;
  __shared__ float sh[L];
  int t = threadIdx.x;
  sh[t] = acc;
  __syncthreads();
  int j = t & 31;
  float v[W];
#pragma unroll
  for (int w = 0; w < W; ++w)
    v[w] = sh[w * 32 + j];
#pragma unroll
  for (int s = W >> 1; s > 0; s >>= 1)
#pragma unroll
    for (int q = 0; q < s; ++q)
      v[q] = __fadd_rn(v[q], v[q + s]);
  return warp_fold(v[0], 32);
}

template <int L> __device__ __forceinline__ float lane_fold(float acc) {
  if constexpr (L <= 32)
    return warp_fold(acc, L);
  else
    return wide_fold<L>(acc);
}

__device__ __forceinline__ float inv_rms(float ssq, int n, float eps) {
  return __frsqrt_rn(__fadd_rn(__fdiv_rn(ssq, float(n)), eps));
}

} // namespace norms_sm100
