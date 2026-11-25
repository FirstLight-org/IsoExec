// SPDX-License-Identifier: Apache-2.0
// Forward kernels. Numerical recipe version 1 (see README.md, "Numerical contract").
//
// A row of width N = L*8*K is folded as L lane chains plus a butterfly:
//   lane l chain: acc = +0; for k in [0,K), j in [0,8): acc = add.rn(acc, mul.rn(v, v)),
//                 v = fp32(x[k*8L + 8l + j])
//   butterfly:    for s = L/2 .. 1: acc_l = add.rn(acc_l, acc_{l^s})
//   rstd        = frsqrt_rn(add.rn(div.rn(root, N), eps))
//   y           = rnd_T(mul.rn(mul.rn(v, rstd), g)),  g = w | add.rn(1, w) | (none: y = rnd_T(mul.rn(v, rstd)))
// L and K depend on N only (the host picks them from a fixed table). Rows, blocks, threads per CTA
// and graph shape are execution choices and never enter the arithmetic above.
#pragma once
#include "common.cuh"

namespace norms_sm90 {

// Weight chunk in fp32 (g), from a bf16 or fp32 weight row.
template <int MODE, bool WF>
__device__ __forceinline__ void weight8(float *g, const void *w, int k, int L, int lane) {
  if constexpr (WF)
    load8f(g, static_cast<const float *>(w), k, L, lane);
  else
    load8(g, static_cast<const __nv_bfloat16 *>(w), k, L, lane);
  if constexpr (MODE == 2) {
#pragma unroll
    for (int j = 0; j < 8; ++j)
      g[j] = __fadd_rn(1.f, g[j]);
  }
}

// MODE: 0 no weight, 1 plain weight, 2 zero-centred (1 + w).
// RES: h = rnd_T(add.rn(x, r)) is stored to hout, and the norm reads h.
template <int K, int L, int MODE, bool WF, bool RES, bool SAVE>
__global__ void
rms_fwd(const __nv_bfloat16 *__restrict__ x, int64_t sx, const __nv_bfloat16 *__restrict__ r,
        int64_t sr, const void *__restrict__ w, __nv_bfloat16 *__restrict__ hout,
        __nv_bfloat16 *__restrict__ out, float *__restrict__ rstd, int64_t M, int rpb,
        float eps) {
  constexpr int C = 8 * K, N = 8 * L * K;
  int64_t row = int64_t(blockIdx.x) * rpb + threadIdx.x / L;
  row = row < M ? row : M - 1; // tail rows of the last block recompute row M-1 (same bits)
  int lane = threadIdx.x % L;
  const __nv_bfloat16 *xr = x + row * sx;
  float v[C];
  if constexpr (RES) {
    const __nv_bfloat16 *rr = r + row * sr;
    __nv_bfloat16 *hr = hout + row * N;
#pragma unroll
    for (int k = 0; k < K; ++k) {
      float a[8], b[8];
      load8(a, xr, k, L, lane);
      load8(b, rr, k, L, lane);
#pragma unroll
      for (int j = 0; j < 8; ++j)
        a[j] = __fadd_rn(a[j], b[j]);
      uint4 q = make_uint4(pack2(a[0], a[1]), pack2(a[2], a[3]), pack2(a[4], a[5]), pack2(a[6], a[7]));
      reinterpret_cast<uint4 *>(hr + k * 8 * L + 8 * lane)[0] = q;
      split8(v + 8 * k, q);
    }
  } else {
#pragma unroll
    for (int k = 0; k < K; ++k)
      load8(v + 8 * k, xr, k, L, lane);
  }
  float acc = 0.f;
#pragma unroll
  for (int c = 0; c < C; ++c)
    acc = __fadd_rn(acc, __fmul_rn(v[c], v[c]));
  acc = warp_fold(acc, L);
  float rs = inv_rms(acc, N, eps);
  if constexpr (SAVE)
    store_if(rstd + row, rs, lane == 0);
  __nv_bfloat16 *orow = out + row * N;
#pragma unroll
  for (int k = 0; k < K; ++k) {
    float y[8];
    if constexpr (MODE == 0) {
#pragma unroll
      for (int j = 0; j < 8; ++j)
        y[j] = __fmul_rn(v[8 * k + j], rs);
    } else {
      float g[8];
      weight8<MODE, WF>(g, w, k, L, lane);
#pragma unroll
      for (int j = 0; j < 8; ++j)
        y[j] = __fmul_rn(__fmul_rn(v[8 * k + j], rs), g[j]);
    }
    store8(orow, k, L, lane, y);
  }
}

// Fused residual join + RMSNorm, same arithmetic as rms_fwd<..., RES=true, ...> (and so as the unfused
// add then rms_fwd): h = rnd_bf16(add.rn(x, r)); the lane chain, fold, rstd and output are unchanged and
// read h's exact fp32 values in the same order. Only the schedule differs: every operand load is issued
// before any arithmetic (both rows in flight), and h is held packed (bf16 -> fp32 widening is exact),
// which the two-operand form needs to stay under the register file without serializing its loads.
__device__ __forceinline__ float bf_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf_hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }
__device__ __forceinline__ uint32_t rn_pair(float a, float b) {
  __nv_bfloat162 t = __floats2bfloat162_rn(a, b);  // cvt.rn.bf16x2.f32, low = a: pack2's instruction
  return *reinterpret_cast<uint32_t *>(&t);
}
__device__ __forceinline__ void widen8(float *v, uint4 q) {
  v[0] = bf_lo(q.x); v[1] = bf_hi(q.x); v[2] = bf_lo(q.y); v[3] = bf_hi(q.y);
  v[4] = bf_lo(q.z); v[5] = bf_hi(q.z); v[6] = bf_lo(q.w); v[7] = bf_hi(q.w);
}

template <int K, int L, int MODE, bool WF, bool SAVE>
__global__ void
add_rms_fwd(const __nv_bfloat16 *__restrict__ x, int64_t sx, const __nv_bfloat16 *__restrict__ r,
            int64_t sr, const void *__restrict__ w, __nv_bfloat16 *__restrict__ hout,
            __nv_bfloat16 *__restrict__ out, float *__restrict__ rstd, int64_t M, int rpb, float eps) {
  constexpr int N = 8 * L * K;
  int64_t row = int64_t(blockIdx.x) * rpb + threadIdx.x / L;
  row = row < M ? row : M - 1;
  int lane = threadIdx.x % L;
  const __nv_bfloat16 *xr = x + row * sx;
  const __nv_bfloat16 *rr = r + row * sr;
  uint4 hp[K];
  {
    uint4 xa[K], ra[K];
#pragma unroll
    for (int k = 0; k < K; ++k) {
      xa[k] = reinterpret_cast<const uint4 *>(xr + k * 8 * L + 8 * lane)[0];
      ra[k] = reinterpret_cast<const uint4 *>(rr + k * 8 * L + 8 * lane)[0];
    }
#pragma unroll
    for (int k = 0; k < K; ++k) {
      float a[8], b[8];
      widen8(a, xa[k]);
      widen8(b, ra[k]);
#pragma unroll
      for (int j = 0; j < 8; ++j)
        a[j] = __fadd_rn(a[j], b[j]);
      hp[k] = make_uint4(rn_pair(a[0], a[1]), rn_pair(a[2], a[3]), rn_pair(a[4], a[5]), rn_pair(a[6], a[7]));
    }
  }
  float acc = 0.f;
#pragma unroll
  for (int k = 0; k < K; ++k) {
    float v[8];
    widen8(v, hp[k]);
#pragma unroll
    for (int j = 0; j < 8; ++j)
      acc = __fadd_rn(acc, __fmul_rn(v[j], v[j]));
  }
  acc = warp_fold(acc, L);
  float rs = inv_rms(acc, N, eps);
  if constexpr (SAVE)
    store_if(rstd + row, rs, lane == 0);
  __nv_bfloat16 *orow = out + row * N, *hrow = hout + row * N;
#pragma unroll
  for (int k = 0; k < K; ++k) {
    reinterpret_cast<uint4 *>(hrow + k * 8 * L + 8 * lane)[0] = hp[k];
    float v[8], y[8];
    widen8(v, hp[k]);
    if constexpr (MODE == 0) {
#pragma unroll
      for (int j = 0; j < 8; ++j)
        y[j] = __fmul_rn(v[j], rs);
    } else {
      float g[8];
      weight8<MODE, WF>(g, w, k, L, lane);
#pragma unroll
      for (int j = 0; j < 8; ++j)
        y[j] = __fmul_rn(__fmul_rn(v[j], rs), g[j]);
    }
    store8(orow, k, L, lane, y);
  }
}

// GatedDeltaNet output norm: y = rnd_T(mul.rn(mul.rn(mul.rn(v, rstd), g), silu(z))),
// silu(z) = div.rn(z, add.rn(1, expf(-z))) with CUDA libdevice expf. Output row m reads the gate at
// token m / hv, head m % hv; the gate's token and head strides are free, the feature stride is 1.
template <int K, int L, int MODE, bool WF, bool SAVE, bool ZAL>
__global__ void
gated_fwd(const __nv_bfloat16 *__restrict__ x, int64_t sx, const __nv_bfloat16 *__restrict__ z,
          int64_t szt, int64_t szh, const void *__restrict__ w, __nv_bfloat16 *__restrict__ out,
          float *__restrict__ rstd, int64_t M, int hv, int rpb, float eps) {
  constexpr int C = 8 * K, N = 8 * L * K;
  int64_t row = int64_t(blockIdx.x) * rpb + threadIdx.x / L;
  row = row < M ? row : M - 1;
  int lane = threadIdx.x % L;
  const __nv_bfloat16 *xr = x + row * sx;
  const __nv_bfloat16 *zr = z + (row / hv) * szt + (row % hv) * szh;
  float v[C];
#pragma unroll
  for (int k = 0; k < K; ++k)
    load8(v + 8 * k, xr, k, L, lane);
  float acc = 0.f;
#pragma unroll
  for (int c = 0; c < C; ++c)
    acc = __fadd_rn(acc, __fmul_rn(v[c], v[c]));
  acc = warp_fold(acc, L);
  float rs = inv_rms(acc, N, eps);
  if constexpr (SAVE)
    store_if(rstd + row, rs, lane == 0);
  __nv_bfloat16 *orow = out + row * N;
#pragma unroll
  for (int k = 0; k < K; ++k) {
    float g[8], a[8], y[8];
    if constexpr (ZAL)
      load8(a, zr, k, L, lane);
    else
      load8u(a, zr, k, lane, L);
    if constexpr (MODE != 0)
      weight8<MODE, WF>(g, w, k, L, lane);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      float s = __fdiv_rn(a[j], __fadd_rn(1.f, expf(-a[j])));
      float u = __fmul_rn(v[8 * k + j], rs);
      if constexpr (MODE != 0)
        u = __fmul_rn(u, g[j]);
      y[j] = __fmul_rn(u, s);
    }
    store8(orow, k, L, lane, y);
  }
}

} // namespace norms_sm90
