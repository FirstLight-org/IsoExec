// SPDX-License-Identifier: Apache-2.0
// SM90 fused mlp.act, recipe v1 = the eager trainer program F.silu(g) * (u + 0.0), rounded where it rounds:
//   s   = rn_bf16( div.rn(g, add.rn(1, expf(-g))) )     ATen's silu: x / (1 + exp(-x)) in fp32, one bf16 round
//   uf  = add.rn(fl32(u), +0)                            u + 0.0 (maps -0 to +0); exact otherwise
//   out = rn_bf16( mul.rn(fl32(s), uf) )                 bf16 * bf16 in fp32 opmath, one bf16 round
// Every op is an explicit round-to-nearest intrinsic (the build passes --fmad=false --ftz=false), and expf
// is libdevice's, as in ATen. Grid/block shape is an execution choice and never enters the arithmetic.
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace mlp_sm90 {
namespace {

__device__ __forceinline__ float lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }

__device__ __forceinline__ float silu_rn(float g) {
  return __fdiv_rn(g, __fadd_rn(1.0f, expf(-g)));
}

__device__ __forceinline__ uint32_t pack_rn(float a, float b) {
  __nv_bfloat162 r = __floats2bfloat162_rn(a, b);
  return *reinterpret_cast<uint32_t *>(&r);
}

__device__ __forceinline__ uint32_t silu_mul_pair(uint32_t gw, uint32_t uw) {
  const uint32_t sw = pack_rn(silu_rn(lo(gw)), silu_rn(hi(gw)));
  const float u0 = __fadd_rn(lo(uw), 0.0f), u1 = __fadd_rn(hi(uw), 0.0f);
  return pack_rn(__fmul_rn(lo(sw), u0), __fmul_rn(hi(sw), u1));
}

__global__ void silu_mul_vec8(const uint4 *__restrict__ g8, const uint4 *__restrict__ u8,
                              uint4 *__restrict__ o8, int64_t n8) {
  const int64_t stride = int64_t(blockDim.x) * gridDim.x;
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n8; i += stride) {
    const uint4 g = __ldg(g8 + i), u = __ldg(u8 + i);
    uint4 o;
    o.x = silu_mul_pair(g.x, u.x);
    o.y = silu_mul_pair(g.y, u.y);
    o.z = silu_mul_pair(g.z, u.z);
    o.w = silu_mul_pair(g.w, u.w);
    o8[i] = o;
  }
}

__global__ void silu_mul_scalar(const __nv_bfloat16 *__restrict__ g, const __nv_bfloat16 *__restrict__ u,
                                __nv_bfloat16 *__restrict__ out, int64_t n) {
  const int64_t stride = int64_t(blockDim.x) * gridDim.x;
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
    const __nv_bfloat16 s = __float2bfloat16_rn(silu_rn(__bfloat162float(g[i])));
    out[i] = __float2bfloat16_rn(__fmul_rn(__bfloat162float(s), __fadd_rn(__bfloat162float(u[i]), 0.0f)));
  }
}

// Row-strided operands (gate and up as adjacent column blocks of one packed fc1 GEMM output): the
// same per-element program as silu_mul_vec8, reading row r's 8-element group c at r*ld + c.
__global__ void silu_mul_rows_vec8(const uint4 *__restrict__ g8, const uint4 *__restrict__ u8,
                                   uint4 *__restrict__ o8, int64_t rows, int64_t cols8, int64_t ldg8,
                                   int64_t ldu8) {
  const int64_t n8 = rows * cols8;
  const int64_t stride = int64_t(blockDim.x) * gridDim.x;
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n8; i += stride) {
    const int64_t r = i / cols8, c = i - r * cols8;
    const uint4 g = __ldg(g8 + r * ldg8 + c), u = __ldg(u8 + r * ldu8 + c);
    uint4 o;
    o.x = silu_mul_pair(g.x, u.x);
    o.y = silu_mul_pair(g.y, u.y);
    o.z = silu_mul_pair(g.z, u.z);
    o.w = silu_mul_pair(g.w, u.w);
    o8[i] = o;
  }
}

// ---------------------------------------------------------------------------------------- backward
// The eager VJP of silu(g) * (u + 0.0), op for op (ATen's CUDA kernels, bf16 tensors, fp32 opmath):
//   t   = rn_bf16( mul.rn(dy, add.rn(u, +0)) )                       dy * (u + 0.0)
//   sg  = div.rn(1, add.rn(1, expf(-g)))                              silu_backward's sigmoid
//   dg  = rn_bf16( mul.rn(mul.rn(t, sg), fma.rn(g, sub.rn(1, sg), 1)) )
//                                                                    silu_backward(t, g): ATen's
//                                                                    dy*s*(1 + x*(1 - s)), compiled
//                                                                    with nvcc's default contraction
//   s   = rn_bf16( div.rn(g, add.rn(1, expf(-g))) )                   F.silu(g)
//   du  = rn_bf16( mul.rn(dy, s) )                                    dy * silu(g)
// One read of (dy, g, u) and one write of (dg, du) instead of four kernels and three temporaries.
__device__ __forceinline__ void silu_mul_bwd_one(float dy, float g, float u, float &dg, float &du) {
  const float e = expf(-g);
  const float den = __fadd_rn(1.0f, e);
  const float t = __bfloat162float(__float2bfloat16_rn(__fmul_rn(dy, __fadd_rn(u, 0.0f))));
  const float sg = __fdiv_rn(1.0f, den);
  dg = __fmul_rn(__fmul_rn(t, sg), __fmaf_rn(g, __fsub_rn(1.0f, sg), 1.0f));
  const float s = __bfloat162float(__float2bfloat16_rn(__fdiv_rn(g, den)));
  du = __fmul_rn(dy, s);
}

__device__ __forceinline__ void silu_mul_bwd_pair(uint32_t dyw, uint32_t gw, uint32_t uw, uint32_t &dgw,
                                                  uint32_t &duw) {
  float dg0, du0, dg1, du1;
  silu_mul_bwd_one(lo(dyw), lo(gw), lo(uw), dg0, du0);
  silu_mul_bwd_one(hi(dyw), hi(gw), hi(uw), dg1, du1);
  dgw = pack_rn(dg0, dg1);
  duw = pack_rn(du0, du1);
}

// Row-strided operands (rows x cols8 groups of 8 bf16); contiguous tensors are the ld == cols case.
// Outputs are contiguous [rows, cols].
__global__ void silu_mul_bwd_rows_vec8(const uint4 *__restrict__ dy8, const uint4 *__restrict__ g8,
                                       const uint4 *__restrict__ u8, uint4 *__restrict__ dg8,
                                       uint4 *__restrict__ du8, int64_t rows, int64_t cols8, int64_t ldy8,
                                       int64_t ldg8, int64_t ldu8) {
  const int64_t n8 = rows * cols8;
  const int64_t stride = int64_t(blockDim.x) * gridDim.x;
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n8; i += stride) {
    const int64_t r = i / cols8, c = i - r * cols8;
    const uint4 d = __ldg(dy8 + r * ldy8 + c), g = __ldg(g8 + r * ldg8 + c), u = __ldg(u8 + r * ldu8 + c);
    uint4 og, ou;
    silu_mul_bwd_pair(d.x, g.x, u.x, og.x, ou.x);
    silu_mul_bwd_pair(d.y, g.y, u.y, og.y, ou.y);
    silu_mul_bwd_pair(d.z, g.z, u.z, og.z, ou.z);
    silu_mul_bwd_pair(d.w, g.w, u.w, og.w, ou.w);
    dg8[i] = og;
    du8[i] = ou;
  }
}

__global__ void silu_mul_bwd_scalar(const __nv_bfloat16 *__restrict__ dy, const __nv_bfloat16 *__restrict__ g,
                                    const __nv_bfloat16 *__restrict__ u, __nv_bfloat16 *__restrict__ dg,
                                    __nv_bfloat16 *__restrict__ du, int64_t n) {
  const int64_t stride = int64_t(blockDim.x) * gridDim.x;
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
    float a, b;
    silu_mul_bwd_one(__bfloat162float(dy[i]), __bfloat162float(g[i]), __bfloat162float(u[i]), a, b);
    dg[i] = __float2bfloat16_rn(a);
    du[i] = __float2bfloat16_rn(b);
  }
}

constexpr int kThreads = 256;
constexpr int64_t kMaxBlocks = 132 * 16;  // H100 SXM: 132 SMs; the grid-stride loop covers the rest

bool aligned16(const void *p) { return (reinterpret_cast<uintptr_t>(p) & 15u) == 0; }

} // namespace

void launch_silu_mul(const __nv_bfloat16 *g, const __nv_bfloat16 *u, __nv_bfloat16 *out, int64_t n,
                     cudaStream_t stream) {
  if (n <= 0)
    return;
  if (n % 8 == 0 && aligned16(g) && aligned16(u) && aligned16(out)) {
    const int64_t n8 = n / 8;
    int64_t blocks = (n8 + kThreads - 1) / kThreads;
    blocks = blocks < kMaxBlocks ? blocks : kMaxBlocks;
    silu_mul_vec8<<<int(blocks), kThreads, 0, stream>>>(reinterpret_cast<const uint4 *>(g),
                                                       reinterpret_cast<const uint4 *>(u),
                                                       reinterpret_cast<uint4 *>(out), n8);
    return;
  }
  int64_t blocks = (n + kThreads - 1) / kThreads;
  blocks = blocks < kMaxBlocks ? blocks : kMaxBlocks;
  silu_mul_scalar<<<int(blocks), kThreads, 0, stream>>>(g, u, out, n);
}

bool launch_silu_mul_rows(const __nv_bfloat16 *g, const __nv_bfloat16 *u, __nv_bfloat16 *out, int64_t rows,
                          int64_t cols, int64_t ldg, int64_t ldu, cudaStream_t stream) {
  if (cols % 8 != 0 || ldg % 8 != 0 || ldu % 8 != 0 || !aligned16(g) || !aligned16(u) || !aligned16(out))
    return false;
  if (rows <= 0 || cols <= 0)
    return true;
  const int64_t cols8 = cols / 8, n8 = rows * cols8;
  int64_t blocks = (n8 + kThreads - 1) / kThreads;
  blocks = blocks < kMaxBlocks ? blocks : kMaxBlocks;
  silu_mul_rows_vec8<<<int(blocks), kThreads, 0, stream>>>(reinterpret_cast<const uint4 *>(g),
                                                          reinterpret_cast<const uint4 *>(u),
                                                          reinterpret_cast<uint4 *>(out), rows, cols8,
                                                          ldg / 8, ldu / 8);
  return true;
}

// Row-strided launch; false when the operands are not 16-byte vectorisable (the caller compacts).
bool launch_silu_mul_bwd_rows(const __nv_bfloat16 *dy, const __nv_bfloat16 *g, const __nv_bfloat16 *u,
                              __nv_bfloat16 *dg, __nv_bfloat16 *du, int64_t rows, int64_t cols, int64_t ldy,
                              int64_t ldg, int64_t ldu, cudaStream_t stream) {
  if (cols % 8 != 0 || ldy % 8 != 0 || ldg % 8 != 0 || ldu % 8 != 0 || !aligned16(dy) || !aligned16(g) ||
      !aligned16(u) || !aligned16(dg) || !aligned16(du))
    return false;
  if (rows <= 0 || cols <= 0)
    return true;
  const int64_t cols8 = cols / 8, n8 = rows * cols8;
  int64_t blocks = (n8 + kThreads - 1) / kThreads;
  blocks = blocks < kMaxBlocks ? blocks : kMaxBlocks;
  silu_mul_bwd_rows_vec8<<<int(blocks), kThreads, 0, stream>>>(
      reinterpret_cast<const uint4 *>(dy), reinterpret_cast<const uint4 *>(g), reinterpret_cast<const uint4 *>(u),
      reinterpret_cast<uint4 *>(dg), reinterpret_cast<uint4 *>(du), rows, cols8, ldy / 8, ldg / 8, ldu / 8);
  return true;
}

void launch_silu_mul_bwd(const __nv_bfloat16 *dy, const __nv_bfloat16 *g, const __nv_bfloat16 *u,
                         __nv_bfloat16 *dg, __nv_bfloat16 *du, int64_t n, cudaStream_t stream) {
  if (n <= 0)
    return;
  if (n % 8 == 0 && launch_silu_mul_bwd_rows(dy, g, u, dg, du, 1, n, n, n, n, stream))
    return;
  int64_t blocks = (n + kThreads - 1) / kThreads;
  blocks = blocks < kMaxBlocks ? blocks : kMaxBlocks;
  silu_mul_bwd_scalar<<<int(blocks), kThreads, 0, stream>>>(dy, g, u, dg, du, n);
}

} // namespace mlp_sm90
