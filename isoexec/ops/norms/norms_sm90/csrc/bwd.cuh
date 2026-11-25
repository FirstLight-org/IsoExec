// SPDX-License-Identifier: Apache-2.0
// Backward kernels. The execution plan fixes each band's row order; a separate deterministic
// reduction combines the weight-gradient partials without cross-CTA floating-point atomics.
//
// A grid block is a band of rows. Each 128-thread CTA iterates the band `iters` times, handling
// 128/L rows per iteration (L lanes per row). The weight gradient accumulates in lane registers,
// the row groups of a CTA are merged through CTA shared memory, and each band then folds its
// partial into its own fp32 [band, N] row. No initialization of that fully written buffer is needed.
#pragma once
#include "common.cuh"

namespace norms_sm90 {


// Row dot-product fold for L lanes. L <= 32: butterfly inside the warp; L == 64/128: one
// shared-memory stage for the cross-warp part, then the in-warp butterfly.
template <int L> __device__ __forceinline__ float row_fold(float acc) {
  if constexpr (L <= 32) {
    return warp_fold(acc, L);
  } else {
    __shared__ float sh[128];
    int t = threadIdx.x;
    sh[t] = acc;
    __syncthreads();
    if constexpr (L == 64)
      acc = __fadd_rn(sh[t], sh[t ^ 32]);
    else
      acc = __fadd_rn(__fadd_rn(sh[t], sh[t ^ 64]), __fadd_rn(sh[t ^ 32], sh[t ^ 96]));
    return warp_fold(acc, 32);
  }
}

template <class T> __device__ __forceinline__ T select0(int live, T v) { return live ? v : T(0); }

// dx = rstd * (gy - xh * dot/N) with gy = dy * g, xh = x * rstd; dw += dy * xh.
// RES = 1 adds the residual-stream gradient dres to dx before rounding (one fp32 add, one round).
// RES = 2 reproduces the UNFUSED program's bf16 accumulation: dx is rounded to bf16 first, then
//         rn_bf16(fl32(dx_bf16) + fl32(dres)) -- autograd's bf16 ``dx + dres``, bit for bit.
template <int K, int L, int MODE, bool WF, int RES>
__global__ void rms_bwd(const __nv_bfloat16 *__restrict__ dy, int64_t sdy,
                        const __nv_bfloat16 *__restrict__ x, int64_t sx,
                        const float *__restrict__ rstd, const void *__restrict__ w,
                        const __nv_bfloat16 *__restrict__ dres, int64_t sdres,
                        __nv_bfloat16 *__restrict__ dx, float *__restrict__ dwp, int64_t M,
                        int iters) {
  constexpr int C = 8 * K, N = 8 * L * K, RPI = L <= 64 ? 128 / L : 1;
  int lane = threadIdx.x % L;
  int64_t band = blockIdx.x;
  float dwa[C];
#pragma unroll
  for (int c = 0; c < C; ++c)
    dwa[c] = 0.f;
  float gw[C];
  if constexpr (MODE != 0) {
#pragma unroll
    for (int k = 0; k < K; ++k) {
      if constexpr (WF)
        load8f(gw + 8 * k, static_cast<const float *>(w), k, L, lane);
      else
        load8(gw + 8 * k, static_cast<const __nv_bfloat16 *>(w), k, L, lane);
#pragma unroll
      for (int j = 0; j < 8; ++j)
        if constexpr (MODE == 2)
          gw[8 * k + j] = __fadd_rn(1.f, gw[8 * k + j]);
    }
  }
  auto step = [&](int it) {
    int64_t raw = (band * iters + it) * RPI + threadIdx.x / L;
    int live = raw < M;
    int64_t row = live ? raw : M - 1;
    const __nv_bfloat16 *xr = x + row * sx, *dyr = dy + row * sdy;
    float rs = rstd[row];
    float xh[C], gy[C];
    float acc = 0.f;
#pragma unroll
    for (int k = 0; k < K; ++k) {
      float a[8], d[8];
      load8(a, xr, k, L, lane);
      load8(d, dyr, k, L, lane);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        float h = __fmul_rn(a[j], rs);
        float gj = d[j];
        if constexpr (MODE != 0)
          gj = __fmul_rn(d[j], gw[8 * k + j]);
        xh[8 * k + j] = h;
        gy[8 * k + j] = gj;
        acc = __fadd_rn(acc, __fmul_rn(h, gj));
        if constexpr (MODE != 0)
          dwa[8 * k + j] = __fadd_rn(dwa[8 * k + j], select0(live, __fmul_rn(d[j], h)));
      }
    }
    float dn = __fdiv_rn(row_fold<L>(acc), float(N));
    __nv_bfloat16 *dxr = dx + row * N;
#pragma unroll
    for (int k = 0; k < K; ++k) {
      float o[8];
#pragma unroll
      for (int j = 0; j < 8; ++j)
        o[j] = __fmul_rn(rs, __fsub_rn(gy[8 * k + j], __fmul_rn(xh[8 * k + j], dn)));
      if constexpr (RES != 0) {
        float e[8];
        load8(e, dres + row * sdres, k, L, lane);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          if constexpr (RES == 2)
            o[j] = __bfloat162float(__float2bfloat16_rn(o[j]));
          o[j] = __fadd_rn(o[j], e[j]);
        }
      }
      store8(dxr, k, L, lane, o);
    }
  };
  for (int it = 0; it < iters; ++it)
    step(it);
  if constexpr (MODE != 0 && RPI > 1) {
    // Merge the RPI row groups' lane partials inside the CTA: group 0 (threads [0, L)) emits.
    __shared__ float shdw[128 * C];
#pragma unroll
    for (int c = 0; c < C; ++c)
      shdw[threadIdx.x * C + c] = dwa[c];
    __syncthreads();
    if (threadIdx.x < L) {
#pragma unroll
      for (int q = 1; q < RPI; ++q) {
#pragma unroll
        for (int c = 0; c < C; ++c)
          dwa[c] = __fadd_rn(dwa[c], shdw[(q * L + lane) * C + c]);
      }
    }
  }
  if constexpr (MODE != 0) {
    // Each band owns a complete output row; the second kernel fixes the inter-band fold.
    if (RPI == 1 || threadIdx.x < L) {
#pragma unroll
      for (int k = 0; k < K; ++k) {
        float4 *p = reinterpret_cast<float4 *>(dwp + band * N + k * 8 * L + 8 * lane);
        *p = make_float4(dwa[8 * k], dwa[8 * k + 1], dwa[8 * k + 2], dwa[8 * k + 3]);
        *(p + 1) = make_float4(dwa[8 * k + 4], dwa[8 * k + 5], dwa[8 * k + 6], dwa[8 * k + 7]);
      }
    }
  }
}

// Gated out-norm VJP. y = xh * g * a(z), a = silu, xh = x * rstd.
//   s = 1 / (1 + exp(-z)), a = z * s, a' = s * (1 + z * (1 - s))
//   dz = dy * xh * g * a'
//   gy = dy * a * g;  dx = rstd * (gy - xh * dot(xh, gy) / N);  dw += dy * a * xh
template <int K, int L, int MODE, bool WF, bool ZAL>
__global__ void gated_bwd(const __nv_bfloat16 *__restrict__ dy, int64_t sdy,
                          const __nv_bfloat16 *__restrict__ x, int64_t sx,
                          const __nv_bfloat16 *__restrict__ z, int64_t szt, int64_t szh,
                          const float *__restrict__ rstd, const void *__restrict__ w,
                          __nv_bfloat16 *__restrict__ dx, __nv_bfloat16 *__restrict__ dz,
                          float *__restrict__ dwp, int64_t M, int hv, int iters) {
  constexpr int C = 8 * K, N = 8 * L * K, RPI = L <= 64 ? 128 / L : 1;
  int lane = threadIdx.x % L;
  int64_t band = blockIdx.x;
  float dwa[C];
#pragma unroll
  for (int c = 0; c < C; ++c)
    dwa[c] = 0.f;
  float gw[C];
  if constexpr (MODE != 0) {
#pragma unroll
    for (int k = 0; k < K; ++k) {
      if constexpr (WF)
        load8f(gw + 8 * k, static_cast<const float *>(w), k, L, lane);
      else
        load8(gw + 8 * k, static_cast<const __nv_bfloat16 *>(w), k, L, lane);
#pragma unroll
      for (int j = 0; j < 8; ++j)
        if constexpr (MODE == 2)
          gw[8 * k + j] = __fadd_rn(1.f, gw[8 * k + j]);
    }
  }
  auto step = [&](int it) {
    int64_t raw = (band * iters + it) * RPI + threadIdx.x / L;
    int live = raw < M;
    int64_t row = live ? raw : M - 1;
    const __nv_bfloat16 *xr = x + row * sx, *dyr = dy + row * sdy;
    const __nv_bfloat16 *zr = z + (row / hv) * szt + (row % hv) * szh;
    float rs = rstd[row];
    float xh[C], gy[C];
    float acc = 0.f;
    __nv_bfloat16 *dzr = dz + row * N;
#pragma unroll
    for (int k = 0; k < K; ++k) {
      float a[8], d[8], zz[8], o[8];
      load8(a, xr, k, L, lane);
      load8(d, dyr, k, L, lane);
      if constexpr (ZAL)
        load8(zz, zr, k, L, lane);
      else
        load8u(zz, zr, k, lane, L);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        float gj = 1.f;
        if constexpr (MODE != 0)
          gj = gw[8 * k + j];
        float h = __fmul_rn(a[j], rs);
        float s = __fdiv_rn(1.f, __fadd_rn(1.f, expf(-zz[j])));
        float act = __fmul_rn(zz[j], s);
        float dact = __fmul_rn(s, __fadd_rn(1.f, __fmul_rn(zz[j], __fsub_rn(1.f, s))));
        float dya = __fmul_rn(d[j], act);
        float gyj = __fmul_rn(dya, gj);
        xh[8 * k + j] = h;
        gy[8 * k + j] = gyj;
        acc = __fadd_rn(acc, __fmul_rn(h, gyj));
        o[j] = __fmul_rn(__fmul_rn(__fmul_rn(d[j], h), gj), dact);
        if constexpr (MODE != 0)
          dwa[8 * k + j] = __fadd_rn(dwa[8 * k + j], select0(live, __fmul_rn(dya, h)));
      }
      store8(dzr, k, L, lane, o);
    }
    float dn = __fdiv_rn(row_fold<L>(acc), float(N));
    __nv_bfloat16 *dxr = dx + row * N;
#pragma unroll
    for (int k = 0; k < K; ++k) {
      float o[8];
#pragma unroll
      for (int j = 0; j < 8; ++j)
        o[j] = __fmul_rn(rs, __fsub_rn(gy[8 * k + j], __fmul_rn(xh[8 * k + j], dn)));
      store8(dxr, k, L, lane, o);
    }
  };
  for (int it = 0; it < iters; ++it)
    step(it);
  if constexpr (MODE != 0 && RPI > 1) {
    // Merge the RPI row groups' lane partials inside the CTA: group 0 (threads [0, L)) emits.
    __shared__ float shdw[128 * C];
#pragma unroll
    for (int c = 0; c < C; ++c)
      shdw[threadIdx.x * C + c] = dwa[c];
    __syncthreads();
    if (threadIdx.x < L) {
#pragma unroll
      for (int q = 1; q < RPI; ++q) {
#pragma unroll
        for (int c = 0; c < C; ++c)
          dwa[c] = __fadd_rn(dwa[c], shdw[(q * L + lane) * C + c]);
      }
    }
  }
  if constexpr (MODE != 0) {
    // Each band owns a complete output row; the second kernel fixes the inter-band fold.
    if (RPI == 1 || threadIdx.x < L) {
#pragma unroll
      for (int k = 0; k < K; ++k) {
        float4 *p = reinterpret_cast<float4 *>(dwp + band * N + k * 8 * L + 8 * lane);
        *p = make_float4(dwa[8 * k], dwa[8 * k + 1], dwa[8 * k + 2], dwa[8 * k + 3]);
        *(p + 1) = make_float4(dwa[8 * k + 4], dwa[8 * k + 5], dwa[8 * k + 6], dwa[8 * k + 7]);
      }
    }
  }
}

} // namespace norms_sm90
