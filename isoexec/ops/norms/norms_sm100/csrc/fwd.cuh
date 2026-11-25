// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "common.cuh"
#include "plan.h" // FwdVar: the execution-plan enum, host and device share the spelling

namespace norms_sm100 {

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

template <int K, int L, int MODE, bool WF, bool RES, bool SAVE, int VAR = 0>
__device__ __forceinline__ void
rms_fwd_body(const __nv_bfloat16 *__restrict__ x, int64_t sx, const __nv_bfloat16 *__restrict__ r,
             int64_t sr, const void *__restrict__ w, __nv_bfloat16 *__restrict__ hout,
             __nv_bfloat16 *__restrict__ out, float *__restrict__ rstd, int64_t M, int rpb,
             float eps) {
  constexpr int C = 8 * K, N = 8 * L * K;
  constexpr bool STREAM = (VAR == VAR_STREAM || VAR == VAR_STREAM_PF || VAR == VAR_STREAM_L2) && !RES;
  constexpr bool PREF = (VAR == VAR_PREFETCH || VAR == VAR_STREAM_PF) && MODE != 0;
  constexpr bool PREL2 = (VAR == VAR_KEEP_L2 || VAR == VAR_STREAM_L2) && MODE != 0;
  int64_t row = int64_t(blockIdx.x) * rpb + threadIdx.x / L;
  row = row < M ? row : M - 1; // tail rows of the last block recompute row M-1 (same bits)
  int lane = threadIdx.x % L;
  const __nv_bfloat16 *xr = x + row * sx;
  float v[STREAM ? 8 : C];
  float gp[PREF ? C : 1];
  if constexpr (PREF) {
#pragma unroll
    for (int k = 0; k < K; ++k)
      weight8<MODE, WF>(gp + 8 * k, w, k, L, lane);
  }
  if constexpr (PREL2) {
    const char *wb = static_cast<const char *>(w);
    const int esz = WF ? 4 : 2;
#pragma unroll
    for (int k = 0; k < K; ++k)
      prefetch_l2(wb + int64_t(k * 8 * L + 8 * lane) * esz);
  }
  float acc = 0.f;
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
  } else if constexpr (STREAM) {
#pragma unroll
    for (int k = 0; k < K; ++k) {
      load8(v, xr, k, L, lane);
#pragma unroll
      for (int j = 0; j < 8; ++j)
        acc = __fadd_rn(acc, __fmul_rn(v[j], v[j]));
    }
  } else {
#pragma unroll
    for (int k = 0; k < K; ++k)
      load8(v + 8 * k, xr, k, L, lane);
  }
  if constexpr (!STREAM) {
#pragma unroll
    for (int c = 0; c < C; ++c)
      acc = __fadd_rn(acc, __fmul_rn(v[c], v[c]));
  }
  acc = lane_fold<L>(acc);
  float rs = inv_rms(acc, N, eps);
  if constexpr (SAVE)
    store_if(rstd + row, rs, lane == 0);
  __nv_bfloat16 *orow = out + row * N;
#pragma unroll
  for (int k = 0; k < K; ++k) {
    float y[8];
    const float *vk = v + (STREAM ? 0 : 8 * k);
    if constexpr (STREAM)
      load8(v, xr, k, L, lane);
    if constexpr (MODE == 0) {
#pragma unroll
      for (int j = 0; j < 8; ++j)
        y[j] = __fmul_rn(vk[j], rs);
    } else {
      float g[8];
      const float *gk;
      if constexpr (PREF) {
        gk = gp + 8 * k;
      } else {
        weight8<MODE, WF>(g, w, k, L, lane);
        gk = g;
      }
#pragma unroll
      for (int j = 0; j < 8; ++j)
        if constexpr (MODE == 3)
          y[j] = __fmul_rn(__bfloat162float(__float2bfloat16_rn(__fmul_rn(vk[j], rs))), gk[j]);
        else
          y[j] = __fmul_rn(__fmul_rn(vk[j], rs), gk[j]);
    }
    store8(orow, k, L, lane, y);
  }
}

template <int K, int L, int MODE, bool WF, bool RES, bool SAVE, int VAR = 0>
__global__ void
rms_fwd(const __nv_bfloat16 *__restrict__ x, int64_t sx, const __nv_bfloat16 *__restrict__ r,
        int64_t sr, const void *__restrict__ w, __nv_bfloat16 *__restrict__ hout,
        __nv_bfloat16 *__restrict__ out, float *__restrict__ rstd, int64_t M, int rpb, float eps) {
  rms_fwd_body<K, L, MODE, WF, RES, SAVE, VAR>(x, sx, r, sr, w, hout, out, rstd, M, rpb, eps);
}

template <int K, int L, int MODE, bool WF, bool RES, bool SAVE, int VAR = 0>
__global__ void
rms_fwd_v2(const __nv_bfloat16 *__restrict__ x, int64_t sx, const __nv_bfloat16 *__restrict__ r,
           int64_t sr, const void *__restrict__ w, __nv_bfloat16 *__restrict__ hout,
           __nv_bfloat16 *__restrict__ out, float *__restrict__ rstd, int64_t M, int rpb, float eps) {
  rms_fwd_body<K, L, MODE, WF, RES, SAVE, VAR>(x, sx, r, sr, w, hout, out, rstd, M, rpb, eps);
}

template <int K, int L, int MODE, bool WF, bool RES, bool SAVE, int VAR = 0>
__global__ void __launch_bounds__(L, 7)
rms_fwd_v2_occ(const __nv_bfloat16 *__restrict__ x, int64_t sx, const __nv_bfloat16 *__restrict__ r,
               int64_t sr, const void *__restrict__ w, __nv_bfloat16 *__restrict__ hout,
               __nv_bfloat16 *__restrict__ out, float *__restrict__ rstd, int64_t M, int rpb,
               float eps) {
  rms_fwd_body<K, L, MODE, WF, RES, SAVE, VAR>(x, sx, r, sr, w, hout, out, rstd, M, rpb, eps);
}

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
  acc = lane_fold<L>(acc);
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

} // namespace norms_sm100
