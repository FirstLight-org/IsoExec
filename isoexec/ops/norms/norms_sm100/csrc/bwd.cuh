// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "common.cuh"
#include "plan.h" // FwdVar/BwdVar: the execution-plan enums, host and device share the spelling

namespace norms_sm100 {


// The weight gradient of one CTA thread group is ONE segment of the sum over rows: rows ROW_SEG g .. ROW_SEG g +
// ROW_SEG - 1 of segment g, ascending, each column a chain from +0. A row past the end adds +0 (which leaves a chain
// that started at +0 unchanged), so the last segment is short. launch_bwd.cu::fold_dw_tree combines the segments.
template <class T> __device__ __forceinline__ T select0(int live, T v) { return live ? v : T(0); }

template <int K, int L, int MODE, bool WF, bool RES, int VAR = 0>
__global__ void rms_bwd(const __nv_bfloat16 *__restrict__ dy, int64_t sdy,
                        const __nv_bfloat16 *__restrict__ x, int64_t sx,
                        const float *__restrict__ rstd, const void *__restrict__ w,
                        const __nv_bfloat16 *__restrict__ dres, int64_t sdres,
                        __nv_bfloat16 *__restrict__ dx, float *__restrict__ dwp, int64_t M,
                        int64_t nseg) {
  constexpr int C = 8 * K, N = 8 * L * K, RPI = L <= 64 ? 128 / L : 1;
  constexpr bool STREAM = VAR == BWD_STREAM;
  constexpr bool RING = VAR == BWD_RING && L == 128 && K <= 5;
  constexpr int TPB = 128;
  __shared__ uint4 ring[RING ? 2 * 2 * K * TPB : 1];
  int lane = threadIdx.x % L;
  const int64_t seg = int64_t(blockIdx.x) * RPI + threadIdx.x / L; // this thread group's row segment
  float dwa[C];
#pragma unroll
  for (int c = 0; c < C; ++c)
    dwa[c] = 0.f;
  float gw[(MODE != 0 && !STREAM) ? C : 1];
  if constexpr (MODE != 0 && !STREAM) {
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
  auto weight_chunk = [&](float *g, int k) {
    if constexpr (WF)
      load8f(g, static_cast<const float *>(w), k, L, lane);
    else
      load8(g, static_cast<const __nv_bfloat16 *>(w), k, L, lane);
#pragma unroll
    for (int j = 0; j < 8; ++j)
      if constexpr (MODE == 2)
        g[j] = __fadd_rn(1.f, g[j]);
  };
  auto issue = [&](int it, int stage) {
    int64_t raw = seg * ROW_SEG + it;
    int64_t row = raw < M ? raw : M - 1;
    const __nv_bfloat16 *xr = x + row * sx, *dyr = dy + row * sdy;
#pragma unroll
    for (int k = 0; k < K; ++k) {
      cp_async16(&ring[((stage * 2 + 0) * K + k) * TPB + threadIdx.x], xr + k * 8 * L + 8 * lane);
      cp_async16(&ring[((stage * 2 + 1) * K + k) * TPB + threadIdx.x], dyr + k * 8 * L + 8 * lane);
    }
  };
  auto step = [&](int it, int stage) {
    int64_t raw = seg * ROW_SEG + it;
    int live = raw < M;
    int64_t row = live ? raw : M - 1;
    const __nv_bfloat16 *xr = x + row * sx, *dyr = dy + row * sdy;
    float rs = rstd[row];
    float xh[STREAM ? 1 : C], gy[STREAM ? 1 : C];
    float part[K]; // the row dot's partial of hidden segment k * L + lane
#pragma unroll
    for (int k = 0; k < K; ++k) {
      float a[8], d[8], gc[8];
      float acc = 0.f;
      if constexpr (RING) {
        split8(a, ring[((stage * 2 + 0) * K + k) * TPB + threadIdx.x]);
        split8(d, ring[((stage * 2 + 1) * K + k) * TPB + threadIdx.x]);
      } else {
        load8(a, xr, k, L, lane);
        load8(d, dyr, k, L, lane);
      }
      if constexpr (MODE != 0 && STREAM)
        weight_chunk(gc, k);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        float h = __fmul_rn(a[j], rs);
        float gj = d[j];
        if constexpr (MODE != 0 && STREAM)
          gj = __fmul_rn(d[j], gc[j]);
        else if constexpr (MODE != 0)
          gj = __fmul_rn(d[j], gw[8 * k + j]);
        if constexpr (!STREAM) {
          xh[8 * k + j] = h;
          gy[8 * k + j] = gj;
        }
        acc = __fadd_rn(acc, __fmul_rn(h, gj));
        if constexpr (MODE != 0)
          dwa[8 * k + j] = __fadd_rn(dwa[8 * k + j], select0(live, __fmul_rn(d[j], h)));
      }
      part[k] = acc;
    }
    float dn = __fdiv_rn(segment_tree<K, L, 128 * K>(part, it), float(N));
    __nv_bfloat16 *dxr = dx + row * N;
#pragma unroll
    for (int k = 0; k < K; ++k) {
      float o[8];
      if constexpr (STREAM) {
        float a[8], d[8], gc[8];
        load8(a, xr, k, L, lane);
        load8(d, dyr, k, L, lane);
        if constexpr (MODE != 0)
          weight_chunk(gc, k);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          float h = __fmul_rn(a[j], rs);
          float gj = d[j];
          if constexpr (MODE != 0)
            gj = __fmul_rn(d[j], gc[j]);
          o[j] = __fmul_rn(rs, __fsub_rn(gj, __fmul_rn(h, dn)));
        }
      } else {
#pragma unroll
        for (int j = 0; j < 8; ++j)
          o[j] = __fmul_rn(rs, __fsub_rn(gy[8 * k + j], __fmul_rn(xh[8 * k + j], dn)));
      }
      if constexpr (RES) {
        float e[8];
        load8(e, dres + row * sdres, k, L, lane);
#pragma unroll
        for (int j = 0; j < 8; ++j)
          o[j] = __fadd_rn(o[j], e[j]);
      }
      store8(dxr, k, L, lane, o);
    }
  };
  if constexpr (RING) {
    issue(0, 0);
    cp_async_commit();
    for (int it = 0; it < ROW_SEG; ++it) {
      if (it + 1 < ROW_SEG)
        issue(it + 1, (it + 1) & 1);
      cp_async_commit();
      cp_async_wait<1>();
      step(it, it & 1);
    }
  } else {
    for (int it = 0; it < ROW_SEG; ++it)
      step(it, 0);
  }
  if constexpr (MODE != 0) {
    if (seg < nseg) {
#pragma unroll
      for (int k = 0; k < K; ++k) {
        float4 *p = reinterpret_cast<float4 *>(dwp + seg * N + k * 8 * L + 8 * lane);
        *p = make_float4(dwa[8 * k], dwa[8 * k + 1], dwa[8 * k + 2], dwa[8 * k + 3]);
        *(p + 1) = make_float4(dwa[8 * k + 4], dwa[8 * k + 5], dwa[8 * k + 6], dwa[8 * k + 7]);
      }
    }
  }
}

template <int K, int L, int MODE, bool WF, bool ZAL>
__global__ void gated_bwd(const __nv_bfloat16 *__restrict__ dy, int64_t sdy,
                          const __nv_bfloat16 *__restrict__ x, int64_t sx,
                          const __nv_bfloat16 *__restrict__ z, int64_t szt, int64_t szh,
                          const float *__restrict__ rstd, const void *__restrict__ w,
                          __nv_bfloat16 *__restrict__ dx, __nv_bfloat16 *__restrict__ dz,
                          float *__restrict__ dwp, int64_t M, int hv, int64_t nseg) {
  constexpr int C = 8 * K, N = 8 * L * K, RPI = L <= 64 ? 128 / L : 1;
  int lane = threadIdx.x % L;
  const int64_t seg = int64_t(blockIdx.x) * RPI + threadIdx.x / L; // this thread group's row segment
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
    int64_t raw = seg * ROW_SEG + it;
    int live = raw < M;
    int64_t row = live ? raw : M - 1;
    const __nv_bfloat16 *xr = x + row * sx, *dyr = dy + row * sdy;
    const __nv_bfloat16 *zr = z + (row / hv) * szt + (row % hv) * szh;
    float rs = rstd[row];
    float xh[C], gy[C];
    float part[K];
    __nv_bfloat16 *dzr = dz + row * N;
#pragma unroll
    for (int k = 0; k < K; ++k) {
      float a[8], d[8], zz[8], o[8];
      float acc = 0.f;
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
      part[k] = acc;
    }
    float dn = __fdiv_rn(segment_tree<K, L, 128 * K>(part, it), float(N));
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
  for (int it = 0; it < ROW_SEG; ++it)
    step(it);
  if constexpr (MODE != 0) {
    if (seg < nseg) {
#pragma unroll
      for (int k = 0; k < K; ++k) {
        float4 *p = reinterpret_cast<float4 *>(dwp + seg * N + k * 8 * L + 8 * lane);
        *p = make_float4(dwa[8 * k], dwa[8 * k + 1], dwa[8 * k + 2], dwa[8 * k + 3]);
        *(p + 1) = make_float4(dwa[8 * k + 4], dwa[8 * k + 5], dwa[8 * k + 6], dwa[8 * k + 7]);
      }
    }
  }
}

} // namespace norms_sm100
