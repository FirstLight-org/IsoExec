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

// ---- The numerical schedule of the FORWARD sums of this package -------------------------------------------------------
// Segments of S consecutive positions, each a chain from +0 in ascending order, combined by the adjacent tree
// (pair (0,1), (2,3), ...; an unpaired last partial moves up). S comes from isoexec/ops/schedule_policy.py through
// norms_sm100/schedule.py, which passes it to the compiler; no segment length is written in this directory.
#if !defined(NORMS_SM100_HIDDEN_SEGMENT)
#error "norms_sm100: build through _extension.py; schedule.py supplies the segment lengths from schedule_policy"
#endif
constexpr int HIDDEN_SEG = NORMS_SM100_HIDDEN_SEGMENT; // the forward sum of squares over the hidden axis
// This realisation forms one hidden-axis segment per 128-bit load of eight BF16 values.
static_assert(HIDDEN_SEG == 8, "norms_sm100 kernels realise hidden-axis segments of one 16-byte BF16 load");

constexpr int pow2ceil(int k) {
  int m = 1;
  while (m < k)
    m *= 2;
  return m;
}

// One hidden-axis segment of the sum of squares: columns ascending, from +0.
__device__ __forceinline__ float segment_sq(const float *v) {
  float s = 0.f;
#pragma unroll
  for (int j = 0; j < HIDDEN_SEG; ++j)
    s = __fadd_rn(s, __fmul_rn(v[j], v[j]));
  return s;
}

// Adjacent tree over NV values one thread holds (NV is a compile-time count).
template <int NV> __device__ __forceinline__ float tree_local(float *t) {
  int n = NV;
  while (n > 1) {
    int h = n / 2;
    for (int i = 0; i < h; ++i)
      t[i] = __fadd_rn(t[2 * i], t[2 * i + 1]);
    if (n & 1)
      t[h] = t[n - 1];
    n = (n + 1) / 2;
  }
  return t[0];
}

// Adjacent tree over the K * L segment partials of one row. Thread `lane` of the row's L threads holds the partials
// of segments k * L + lane, k < K (a[k]). Realisation, all of it the same tree:
//   1. the partials go to shared memory; thread t reads the M consecutive segments [M t, M t + M), M = the power of
//      two >= K, and forms their (full) subtree;
//   2. shuffle levels pair neighbouring subtrees inside a warp, the lower one on the left; a subtree with no right
//      neighbour moves up unchanged;
//   3. with more than one warp per row, the warps' roots meet through shared memory.
// CAP: floats of shared memory for step 1, at least (row slots per CTA) * K * L. `buf` alternates the step-3 buffer
// between consecutive calls of one CTA so a fast warp's next row cannot overwrite a value a slow warp still reads.
// Every thread of the row returns the root.
template <int K, int L, int CAP> __device__ __forceinline__ float segment_tree(const float *a, int buf) {
  constexpr int M = pow2ceil(K);
  static_assert(M <= L && (K * L) % M == 0, "norms_sm100: segment_tree needs pow2ceil(K) <= L");
  constexpr int NODES = K * L / M;
  constexpr int WL = L < 32 ? L : 32;
  const int lane = threadIdx.x % L;
  float nv = a[0];
  if constexpr (K > 1) {
    __shared__ float leaf[CAP];
    float *row = leaf + (threadIdx.x / L) * (K * L);
#pragma unroll
    for (int k = 0; k < K; ++k)
      row[k * L + lane] = a[k];
    if constexpr (L > 32)
      __syncthreads();
    else
      __syncwarp();
    const int first = lane < NODES ? M * lane : 0; // threads past the last subtree read a valid slot; unused
    float u[M];
#pragma unroll
    for (int i = 0; i < M; ++i)
      u[i] = row[first + i];
#pragma unroll
    for (int s = 1; s < M; s <<= 1)
#pragma unroll
      for (int q = 0; q < M; q += 2 * s)
        u[q] = __fadd_rn(u[q], u[q + s]);
    nv = u[0];
  }
#pragma unroll
  for (int s = 1; s < WL; s <<= 1) {
    float o = __shfl_xor_sync(0xffffffffu, nv, s);
    if (lane + s < NODES) // read at lanes that are multiples of 2s: the right neighbour exists
      nv = __fadd_rn(nv, o);
  }
  if constexpr (L <= 32) {
    return __shfl_sync(0xffffffffu, nv, int(threadIdx.x & 31) - lane); // the row's first lane holds the root
  } else {
    constexpr int W = L / 32, NB = (NODES + 31) / 32;
    __shared__ float top[2][W];
    float *t = top[buf & 1];
    if ((lane & 31) == 0)
      t[lane >> 5] = nv;
    __syncthreads();
    float tv[NB];
#pragma unroll
    for (int b = 0; b < NB; ++b)
      tv[b] = t[b];
    return tree_local<NB>(tv);
  }
}

// The backward's fold (outside the policy; see bwd.cuh).
__device__ __forceinline__ float warp_fold(float acc, int L) {
  for (int s = L >> 1; s > 0; s >>= 1)
    acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, s));
  return acc;
}

__device__ __forceinline__ float inv_rms(float ssq, int n, float eps) {
  return __frsqrt_rn(__fadd_rn(__fdiv_rn(ssq, float(n)), eps));
}

} // namespace norms_sm100
