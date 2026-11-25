// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "plan.h"

namespace mlp_sm100 {

__device__ __forceinline__ __nv_bfloat16 silu_mul_lane(__nv_bfloat16 gv, __nv_bfloat16 uv) {
  float gf = __bfloat162float(gv);
  float sig = gf / (1.0f + expf(-gf));
  __nv_bfloat16 silu_bf16 = __float2bfloat16_rn(sig);
  float uf = __bfloat162float(uv) + 0.0f;
  float prod = __bfloat162float(silu_bf16) * uf;
  return __float2bfloat16_rn(prod);
}

__device__ __forceinline__ unsigned long long f2bits(float a, float b) {
  return (unsigned long long)__float_as_uint(a) | ((unsigned long long)__float_as_uint(b) << 32);
}
__device__ __forceinline__ float2 mul2(float2 a, float2 b) {
  unsigned long long r;
  asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(r) : "l"(f2bits(a.x, a.y)), "l"(f2bits(b.x, b.y)));
  return make_float2(__uint_as_float((unsigned)r), __uint_as_float((unsigned)(r >> 32)));
}
__device__ __forceinline__ float2 add2(float2 a, float2 b) {
  unsigned long long r;
  asm("add.rn.f32x2 %0, %1, %2;" : "=l"(r) : "l"(f2bits(a.x, a.y)), "l"(f2bits(b.x, b.y)));
  return make_float2(__uint_as_float((unsigned)r), __uint_as_float((unsigned)(r >> 32)));
}

__device__ __forceinline__ uint32_t silu_mul_pair(uint32_t gw, uint32_t uw) {
  const float g0 = __uint_as_float(gw << 16), g1 = __uint_as_float(gw & 0xffff0000u);
  const float2 den = add2(make_float2(1.0f, 1.0f), make_float2(expf(-g0), expf(-g1)));
  const float s0 = g0 / den.x;  // IEEE division, per lane, exactly as ATen's silu writes it
  const float s1 = g1 / den.y;
  const __nv_bfloat162 sb = __floats2bfloat162_rn(s0, s1);
  const uint32_t sw = *reinterpret_cast<const uint32_t *>(&sb);
  const float2 uf = add2(make_float2(__uint_as_float(uw << 16), __uint_as_float(uw & 0xffff0000u)),
                         make_float2(0.0f, 0.0f));
  const float2 p = mul2(make_float2(__uint_as_float(sw << 16), __uint_as_float(sw & 0xffff0000u)), uf);
  const __nv_bfloat162 ob = __floats2bfloat162_rn(p.x, p.y);
  return *reinterpret_cast<const uint32_t *>(&ob);
}

__device__ __forceinline__ uint4 silu_mul_vec8(const uint4 &g, const uint4 &u) {
  uint4 o;
  o.x = silu_mul_pair(g.x, u.x);
  o.y = silu_mul_pair(g.y, u.y);
  o.z = silu_mul_pair(g.z, u.z);
  o.w = silu_mul_pair(g.w, u.w);
  return o;
}

__global__ void silu_mul_kernel(const __nv_bfloat16 *__restrict__ g, const __nv_bfloat16 *__restrict__ u,
                                __nv_bfloat16 *__restrict__ out, int64_t n) {
  int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = int64_t(blockDim.x) * gridDim.x;
  for (; i < n; i += stride)
    out[i] = silu_mul_lane(g[i], u[i]);
}

template <int STORE>
__global__ void silu_mul_kernel_vec8(const uint4 *__restrict__ g8, const uint4 *__restrict__ u8,
                                     uint4 *__restrict__ out8, int64_t n8) {
  int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = int64_t(blockDim.x) * gridDim.x;
  for (; i < n8; i += stride) {
    uint4 gv = g8[i];
    uint4 uv = u8[i];
    const uint4 ov = silu_mul_vec8(gv, uv);
    if constexpr (STORE)
      __stcs(&out8[i], ov);
    else
      out8[i] = ov;
  }
}

template <int VPT>
__global__ void silu_mul_kernel_vecN(const uint4 *__restrict__ g8, const uint4 *__restrict__ u8,
                                     uint4 *__restrict__ out8, int64_t n8) {
  const int64_t stride = int64_t(blockDim.x) * gridDim.x;
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n8; i += stride * VPT) {
    uint4 gv[VPT], uv[VPT];
#pragma unroll
    for (int k = 0; k < VPT; k++) {
      const int64_t j = i + int64_t(k) * stride;
      if (VPT == 1 || j < n8) {
        gv[k] = g8[j];
        uv[k] = u8[j];
      }
    }
#pragma unroll
    for (int k = 0; k < VPT; k++) {
      const int64_t j = i + int64_t(k) * stride;
      if (VPT != 1 && j >= n8)
        continue;
      out8[j] = silu_mul_vec8(gv[k], uv[k]);
    }
  }
}

// Row-strided form: g and u are column windows of wider rows (the packed gate|up GEMM output), read in
// place; out is packed. Element (r, c) gets exactly silu_mul_vec8's arithmetic, so only the addressing
// differs from the packed kernels. blockIdx.y walks rows (grid-stride past 65535), x the row's packs.
__global__ void silu_mul_rows_vec8(const __nv_bfloat16 *__restrict__ g, int64_t sg,
                                   const __nv_bfloat16 *__restrict__ u, int64_t su,
                                   uint4 *__restrict__ out8, int64_t rows, int w8) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= w8)
    return;
  for (int64_t r = blockIdx.y; r < rows; r += gridDim.y) {
    const uint4 gv = reinterpret_cast<const uint4 *>(g + r * sg)[c];
    const uint4 uv = reinterpret_cast<const uint4 *>(u + r * su)[c];
    out8[r * w8 + c] = silu_mul_vec8(gv, uv);
  }
}

static bool aligned16(const void *p) { return (reinterpret_cast<uintptr_t>(p) & 15u) == 0; }

static int max_block_for(const void *kern) {
  cudaFuncAttributes a{};
  if (cudaFuncGetAttributes(&a, kern) != cudaSuccess)
    return 1024;
  return a.maxThreadsPerBlock > 0 ? a.maxThreadsPerBlock : 1024;
}

template <int VPT>
static void launch_vec(const uint4 *g8, const uint4 *u8, uint4 *o8, int64_t n8, Plan p,
                       cudaStream_t stream) {
  const int mx = max_block_for((const void *)silu_mul_kernel_vecN<VPT>);
  int threads = p.threads > mx ? mx : p.threads;
  if (threads < 32)
    threads = 32;
  int64_t want = (n8 + int64_t(threads) * VPT - 1) / (int64_t(threads) * VPT);
  int blocks = p.blocks;
  if (blocks > want)
    blocks = int(want);
  if (blocks < 1)
    blocks = 1;
  silu_mul_kernel_vecN<VPT><<<blocks, threads, 0, stream>>>(g8, u8, o8, n8);
}

void launch_silu_mul(const __nv_bfloat16 *g, const __nv_bfloat16 *u, __nv_bfloat16 *out, int64_t n,
                     int64_t width, cudaStream_t stream) {
  if (n <= 0)
    return;
  if (n % 8 == 0 && aligned16(g) && aligned16(u) && aligned16(out)) {
    const int64_t n8 = n / 8;
    const int64_t rows = width > 0 ? n / width : 1;
    const Plan p = act_plan(n8, width, rows);
    const uint4 *g8 = reinterpret_cast<const uint4 *>(g);
    const uint4 *u8 = reinterpret_cast<const uint4 *>(u);
    uint4 *o8 = reinterpret_cast<uint4 *>(out);
    switch (p.vpt) {
    case 2:
      launch_vec<2>(g8, u8, o8, n8, p, stream);
      return;
    case 4:
      launch_vec<4>(g8, u8, o8, n8, p, stream);
      return;
    case 8:
      launch_vec<8>(g8, u8, o8, n8, p, stream);
      return;
    default: {
      const int mx = max_block_for(p.store ? (const void *)silu_mul_kernel_vec8<1>
                                            : (const void *)silu_mul_kernel_vec8<0>);
      int threads = p.threads > mx ? mx : p.threads;
      if (threads < 32)
        threads = 32;
      int64_t want = (n8 + threads - 1) / threads;
      int blocks = p.blocks > want ? int(want) : p.blocks;
      if (p.store)
        silu_mul_kernel_vec8<1><<<blocks, threads, 0, stream>>>(g8, u8, o8, n8);
      else
        silu_mul_kernel_vec8<0><<<blocks, threads, 0, stream>>>(g8, u8, o8, n8);
      return;
    }
    }
  }
  const Plan p = act_plan(n, width, width > 0 ? n / width : 1);
  const int mx = max_block_for((const void *)silu_mul_kernel);
  int threads = p.threads > mx ? mx : p.threads;
  if (threads < 32)
    threads = 32;
  int64_t wanted = (n + threads - 1) / threads;
  int blocks = int(wanted < 65536 ? wanted : 65536); // grid-stride loop covers the rest
  silu_mul_kernel<<<blocks, threads, 0, stream>>>(g, u, out, n);
}

void launch_silu_mul_rows(const __nv_bfloat16 *g, int64_t sg, const __nv_bfloat16 *u, int64_t su,
                          __nv_bfloat16 *out, int64_t rows, int64_t width, cudaStream_t stream) {
  if (rows <= 0 || width <= 0)
    return;
  const int w8 = int(width / 8);
  // One 16-byte pack per thread in 128-thread blocks (faster than 256 threads or several packs per thread).
  const int threads = w8 >= 128 ? 128 : (w8 <= 32 ? 32 : (w8 + 31) / 32 * 32);
  const dim3 grid((w8 + threads - 1) / threads, unsigned(rows < 65535 ? rows : 65535));
  silu_mul_rows_vec8<<<grid, threads, 0, stream>>>(g, sg, u, su, reinterpret_cast<uint4 *>(out), rows, w8);
}

} // namespace mlp_sm100
