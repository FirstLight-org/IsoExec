// SPDX-License-Identifier: Apache-2.0
// Fused varlen VJP: recompute the pre-activation, SiLU derivative, dx, and per-tile dw/db partials in one
// reverse walk over each token tile. Backward owes no bitwise contract; this kernel is only correct and fast.
#include "common.cuh"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace conv_sm90 {
namespace {

__device__ __forceinline__ float dsilu(float a) {
  float s = __fdividef(1.f, 1.f + expf(-a));
  return s * (1.f + a * (1.f - s));
}

template <class T, class TW, int TT, int BT, int CPT, bool BIAS, bool SILU>
__global__ __launch_bounds__(BT) void varlen_bwd(
    T *__restrict__ dx, float *__restrict__ part, const T *__restrict__ x,
    const T *__restrict__ dy, const TW *__restrict__ w, const TW *__restrict__ bias,
    const int *__restrict__ cu, int nseq, int total, int D, int64_t xrow) {
  // xrow: x's row stride in 8-element packs (x may be a column window of a wider row); dy and dx are packed.
  int tile = blockIdx.x;
  int a = tile * TT;
  int e = min(a + TT, total);
  // hist[i]: valid history taps of token a+i (0..3); fut[i]: valid future taps (0..3). Index TT..TT+2
  // describe tokens e..e+2, the next tile's heads whose dz feed this tile's dx.
  __shared__ uint8_t hist[TT + 3];
  __shared__ uint8_t fut[TT + 3];
  if (threadIdx.x == 0) {
    int lo = 0, hi = nseq - 1;
    while (lo < hi) {
      int mid = (lo + hi + 1) >> 1;
      if (cu[mid] <= a)
        lo = mid;
      else
        hi = mid - 1;
    }
    int s = lo, start = cu[s], end = cu[s + 1];
    int stop = min(e + 3, total);
    for (int t = a; t < stop; ++t) {
      while (t >= end) {
        ++s;
        start = end;
        end = cu[s + 1];
      }
      int k = t - start, f = end - 1 - t;
      hist[t - a] = k < 3 ? k : 3;
      fut[t - a] = f < 3 ? f : 3;
    }
  }
  __syncthreads();
  int c0 = (blockIdx.y * BT + threadIdx.x) * CPT;
  if (c0 >= D)
    return;

  float w0[CPT], w1[CPT], w2[CPT], w3[CPT], b[CPT];
  load_weights<TW, CPT>(w, c0, w0, w1, w2, w3);
  if constexpr (BIAS) {
#pragma unroll
    for (int j = 0; j < CPT; ++j)
      b[j] = widen(bias[c0 + j]);
  }
  // Raw inputs x[t-m] (masked by hist at use) and the three following dz values (masked by fut).
  float r0[CPT], r1[CPT], r2[CPT], r3[CPT];
  float d1[CPT], d2[CPT], d3[CPT];
  float g0[CPT], g1[CPT], g2[CPT], g3[CPT], gb[CPT];
#pragma unroll
  for (int j = 0; j < CPT; ++j) {
    d1[j] = d2[j] = d3[j] = 0.f;
    g0[j] = g1[j] = g2[j] = g3[j] = gb[j] = 0.f;
  }

  auto dz_at = [&](int t, int j, float v0, float v1, float v2, float v3, int k) -> float {
    // v_m = x[t-m] raw; taps with m > k are +0.
    float h1 = k >= 1 ? v1 : 0.f, h2 = k >= 2 ? v2 : 0.f, h3 = k >= 3 ? v3 : 0.f;
    float acc = preact(h3, h2, h1, v0, w0[j], w1[j], w2[j], w3[j]);
    if constexpr (BIAS)
      acc = acc + b[j];
    float g = widen(dy[int64_t(t) * D + c0 + j]);
    if constexpr (SILU)
      return g * dsilu(acc);
    else
      return g;
  };

  // Heads of the next tile (same sequence only): their dz for this tile's last dx rows.
  int f_last = fut[e - 1 - a];
  if (f_last > 0) {
    float xs[6][CPT];  // x[e-3] .. x[e+2]
#pragma unroll
    for (int m = 0; m < 6; ++m) {
      int t = e - 3 + m;
      bool need = t >= 0 && t < total && t < e + f_last && (t >= e - 3 + 0);
#pragma unroll
      for (int j = 0; j < CPT; ++j)
        xs[m][j] = need ? widen(x[int64_t(t) * xrow * 8 + c0 + j]) : 0.f;
    }
#pragma unroll
    for (int j = 0; j < CPT; ++j) {
      float z[3];
      for (int m = 0; m < f_last; ++m) {
        int u = e + m, k = hist[u - a];
        z[m] = dz_at(u, j, xs[3 + m][j], xs[2 + m][j], xs[1 + m][j], xs[m][j], k);
      }
      d1[j] = f_last >= 1 ? z[0] : 0.f;
      d2[j] = f_last >= 2 ? z[1] : 0.f;
      d3[j] = f_last >= 3 ? z[2] : 0.f;
    }
  }

  const auto *xp = reinterpret_cast<const pack8<T> *>(x);
  int64_t prow = D / 8, pc = c0 / 8;
  auto ldx = [&](int t, float (&v)[CPT]) {
    pack8<T> p = xp[int64_t(t) * xrow + pc];
#pragma unroll
    for (int j = 0; j < CPT; ++j)
      v[j] = widen(p.v[j]);
  };
  // Window for t = e-1.
  {
    int t = e - 1, k = hist[t - a];
    ldx(t, r0);
    if (k >= 1) ldx(t - 1, r1);
    if (k >= 2) ldx(t - 2, r2);
    if (k >= 3) ldx(t - 3, r3);
  }
  for (int t = e - 1; t >= a; --t) {
    int k = hist[t - a], f = fut[t - a];
    const auto *dyp = reinterpret_cast<const pack8<T> *>(dy);
    pack8<T> gp = dyp[int64_t(t) * (D / 8) + c0 / 8];
    pack8<T> o;
#pragma unroll
    for (int j = 0; j < CPT; ++j) {
      float h1 = k >= 1 ? r1[j] : 0.f, h2 = k >= 2 ? r2[j] : 0.f, h3 = k >= 3 ? r3[j] : 0.f;
      float acc = preact(h3, h2, h1, r0[j], w0[j], w1[j], w2[j], w3[j]);
      if constexpr (BIAS)
        acc = acc + b[j];
      float g = widen(gp.v[j]);
      float dz;
      if constexpr (SILU)
        dz = g * dsilu(acc);
      else
        dz = g;
      float e1 = f >= 1 ? d1[j] : 0.f, e2 = f >= 2 ? d2[j] : 0.f, e3 = f >= 3 ? d3[j] : 0.f;
      float gx = w3[j] * dz + w2[j] * e1 + w1[j] * e2 + w0[j] * e3;
      o.v[j] = narrow<T>(gx);
      g3[j] += dz * r0[j];
      g2[j] += dz * h1;
      g1[j] += dz * h2;
      g0[j] += dz * h3;
      gb[j] += dz;
      d3[j] = d2[j];
      d2[j] = d1[j];
      d1[j] = dz;
    }
    reinterpret_cast<pack8<T> *>(dx)[int64_t(t) * (D / 8) + c0 / 8] = o;
    if (t > a) {
      int kn = hist[t - 1 - a];
      if (k == 0) {
        // t starts its sequence: t-1 belongs to the previous one, reload its window raw.
        ldx(t - 1, r0);
        if (kn >= 1) ldx(t - 2, r1);
        if (kn >= 2) ldx(t - 3, r2);
        if (kn >= 3) ldx(t - 4, r3);
      } else {
#pragma unroll
        for (int j = 0; j < CPT; ++j) {
          r0[j] = r1[j];
          r1[j] = r2[j];
          r2[j] = r3[j];
        }
        if (kn >= 3) ldx(t - 4, r3);
      }
    }
  }
  float *pp = part + int64_t(tile) * D * 5 + int64_t(c0) * 5;
#pragma unroll
  for (int j = 0; j < CPT; ++j) {
    pp[j * 5] = g0[j];
    pp[j * 5 + 1] = g1[j];
    pp[j * 5 + 2] = g2[j];
    pp[j * 5 + 3] = g3[j];
    pp[j * 5 + 4] = gb[j];
  }
}

template <class T, class TW, int TT, int BT, int CPT, bool BIAS, bool SILU>
void launch_bwd(at::Tensor &dx, at::Tensor &part, const at::Tensor &x, const at::Tensor &dy,
                const at::Tensor &w, const c10::optional<at::Tensor> &bias, const at::Tensor &cu,
                int nseq, int total, int D) {
  int groups = (D + CPT - 1) / CPT;
  dim3 grid((total + TT - 1) / TT, (groups + BT - 1) / BT);
  varlen_bwd<T, TW, TT, BT, CPT, BIAS, SILU><<<grid, BT, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<T *>(dx.data_ptr()), part.data_ptr<float>(),
      reinterpret_cast<const T *>(x.data_ptr()), reinterpret_cast<const T *>(dy.data_ptr()),
      reinterpret_cast<const TW *>(w.data_ptr()),
      BIAS ? reinterpret_cast<const TW *>(bias->data_ptr()) : nullptr, cu.data_ptr<int>(), nseq,
      total, D, x.stride(0) / 8);
}

// Per-token sequence metadata for the swapped-axis VJP: meta[t] = hist | (fut << 2), with hist and fut in
// 0..3 (valid history / future taps). Integer-only.
template <int TB>
__global__ __launch_bounds__(TB) void token_meta(uint8_t *__restrict__ meta, const int *__restrict__ cu,
                                                 int nseq, int total) {
  int a = blockIdx.x * TB;
  if (threadIdx.x != 0)
    return;
  int e = min(a + TB, total);
  int lo = 0, hi = nseq - 1;
  while (lo < hi) {
    int mid = (lo + hi + 1) >> 1;
    if (cu[mid] <= a)
      lo = mid;
    else
      hi = mid - 1;
  }
  int s = lo, start = cu[s], end = cu[s + 1];
  for (int t = a; t < e; ++t) {
    while (t >= end) {
      ++s;
      start = end;
      end = cu[s + 1];
    }
    int k = t - start, f = end - 1 - t;
    meta[t] = uint8_t((k < 3 ? k : 3) | ((f < 3 ? f : 3) << 2));
  }
}

// Swapped axes: block = (channel group, tile block), thread = token tile. dw/db partials of all tiles of a
// block fold in shared memory, so the global partial buffer is [tile blocks, D, 5].
template <class T, class TW, int TT, int TB, int CPT, bool BIAS, bool SILU>
__global__ __launch_bounds__(TB) void varlen_bwd2(
    T *__restrict__ dx, float *__restrict__ part, const T *__restrict__ x, const T *__restrict__ dy,
    const TW *__restrict__ w, const TW *__restrict__ bias, const uint8_t *__restrict__ meta, int total,
    int D, int64_t xrow) {
  int c0 = blockIdx.x * CPT, tb = blockIdx.y;
  int tile = tb * TB + threadIdx.x;
  int a = tile * TT;
  int e = min(a + TT, total);
  __shared__ float acc[TB * CPT * 5];
  float w0[CPT], w1[CPT], w2[CPT], w3[CPT], b[CPT];
  float g0[CPT], g1[CPT], g2[CPT], g3[CPT], gb[CPT];
#pragma unroll
  for (int j = 0; j < CPT; ++j)
    g0[j] = g1[j] = g2[j] = g3[j] = gb[j] = 0.f;
  if (a < total) {
    load_weights<TW, CPT>(w, c0, w0, w1, w2, w3);
    if constexpr (BIAS) {
#pragma unroll
      for (int j = 0; j < CPT; ++j)
        b[j] = widen(bias[c0 + j]);
    }
    const auto *xp = reinterpret_cast<const pack8<T> *>(x);
    const auto *dyp = reinterpret_cast<const pack8<T> *>(dy);
    int64_t prow = D / 8, pc = c0 / 8;
    auto ldx = [&](int t, float (&v)[CPT]) {
      pack8<T> p = xp[int64_t(t) * xrow + pc];
#pragma unroll
      for (int j = 0; j < CPT; ++j)
        v[j] = widen(p.v[j]);
    };
    auto hist = [&](int t) -> int { return meta[t] & 3; };
    auto fut = [&](int t) -> int { return (meta[t] >> 2) & 3; };
    float r0[CPT], r1[CPT], r2[CPT], r3[CPT], d1[CPT], d2[CPT], d3[CPT];
#pragma unroll
    for (int j = 0; j < CPT; ++j)
      d1[j] = d2[j] = d3[j] = 0.f;
    int f_last = fut(e - 1);
    if (f_last > 0) {
      float xs[6][CPT];  // x[e-3] .. x[e+2] (only rows the heads read)
#pragma unroll
      for (int m = 0; m < 6; ++m) {
        int t = e - 3 + m;
        if (t >= 0 && t < e + f_last)
          ldx(t, xs[m]);
        else {
#pragma unroll
          for (int j = 0; j < CPT; ++j)
            xs[m][j] = 0.f;
        }
      }
      float z[3][CPT];
#pragma unroll
      for (int m = 0; m < 3; ++m) {
        if (m < f_last) {
          int u = e + m, k = hist(u);
          pack8<T> gp = dyp[int64_t(u) * prow + pc];
#pragma unroll
          for (int j = 0; j < CPT; ++j) {
            float h1 = k >= 1 ? xs[2 + m][j] : 0.f, h2 = k >= 2 ? xs[1 + m][j] : 0.f,
                  h3 = k >= 3 ? xs[m][j] : 0.f;
            float pre = preact(h3, h2, h1, xs[3 + m][j], w0[j], w1[j], w2[j], w3[j]);
            if constexpr (BIAS)
              pre = pre + b[j];
            float g = widen(gp.v[j]);
            if constexpr (SILU)
              z[m][j] = g * dsilu(pre);
            else
              z[m][j] = g;
          }
        }
      }
#pragma unroll
      for (int j = 0; j < CPT; ++j) {
        d1[j] = f_last >= 1 ? z[0][j] : 0.f;
        d2[j] = f_last >= 2 ? z[1][j] : 0.f;
        d3[j] = f_last >= 3 ? z[2][j] : 0.f;
      }
    }
    {
      int t = e - 1, k = hist(t);
      ldx(t, r0);
      if (k >= 1) ldx(t - 1, r1);
      if (k >= 2) ldx(t - 2, r2);
      if (k >= 3) ldx(t - 3, r3);
    }
    for (int t = e - 1; t >= a; --t) {
      int k = hist(t), f = fut(t);
      pack8<T> gp = dyp[int64_t(t) * prow + pc];
      pack8<T> o;
#pragma unroll
      for (int j = 0; j < CPT; ++j) {
        float h1 = k >= 1 ? r1[j] : 0.f, h2 = k >= 2 ? r2[j] : 0.f, h3 = k >= 3 ? r3[j] : 0.f;
        float pre = preact(h3, h2, h1, r0[j], w0[j], w1[j], w2[j], w3[j]);
        if constexpr (BIAS)
          pre = pre + b[j];
        float g = widen(gp.v[j]);
        float dz;
        if constexpr (SILU)
          dz = g * dsilu(pre);
        else
          dz = g;
        float e1 = f >= 1 ? d1[j] : 0.f, e2 = f >= 2 ? d2[j] : 0.f, e3 = f >= 3 ? d3[j] : 0.f;
        o.v[j] = narrow<T>(w3[j] * dz + w2[j] * e1 + w1[j] * e2 + w0[j] * e3);
        g3[j] += dz * r0[j];
        g2[j] += dz * h1;
        g1[j] += dz * h2;
        g0[j] += dz * h3;
        gb[j] += dz;
        d3[j] = d2[j];
        d2[j] = d1[j];
        d1[j] = dz;
      }
      reinterpret_cast<pack8<T> *>(dx)[int64_t(t) * prow + pc] = o;
      if (t > a) {
        int kn = hist(t - 1);
        if (k == 0) {
          ldx(t - 1, r0);
          if (kn >= 1) ldx(t - 2, r1);
          if (kn >= 2) ldx(t - 3, r2);
          if (kn >= 3) ldx(t - 4, r3);
        } else {
#pragma unroll
          for (int j = 0; j < CPT; ++j) {
            r0[j] = r1[j];
            r1[j] = r2[j];
            r2[j] = r3[j];
          }
          if (kn >= 3) ldx(t - 4, r3);
        }
      }
    }
  }
  float *slot = acc + threadIdx.x * CPT * 5;
#pragma unroll
  for (int j = 0; j < CPT; ++j) {
    slot[j * 5] = g0[j];
    slot[j * 5 + 1] = g1[j];
    slot[j * 5 + 2] = g2[j];
    slot[j * 5 + 3] = g3[j];
    slot[j * 5 + 4] = gb[j];
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float sum[CPT * 5];
#pragma unroll
    for (int i = 0; i < CPT * 5; ++i)
      sum[i] = 0.f;
    for (int th = 0; th < TB; ++th)
#pragma unroll
      for (int i = 0; i < CPT * 5; ++i)
        sum[i] += acc[th * CPT * 5 + i];
    float *pp = part + int64_t(tb) * D * 5 + int64_t(c0) * 5;
#pragma unroll
    for (int i = 0; i < CPT * 5; ++i)
      pp[i] = sum[i];
  }
}

template <class T, class TW, int TT, int TB, int CPT, bool BIAS, bool SILU>
void launch_bwd2(at::Tensor &dx, at::Tensor &part, const at::Tensor &x, const at::Tensor &dy,
                 const at::Tensor &w, const c10::optional<at::Tensor> &bias, const at::Tensor &meta,
                 int total, int D) {
  int tiles = (total + TT - 1) / TT;
  dim3 grid(D / CPT, (tiles + TB - 1) / TB);
  varlen_bwd2<T, TW, TT, TB, CPT, BIAS, SILU><<<grid, TB, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<T *>(dx.data_ptr()), part.data_ptr<float>(),
      reinterpret_cast<const T *>(x.data_ptr()), reinterpret_cast<const T *>(dy.data_ptr()),
      reinterpret_cast<const TW *>(w.data_ptr()),
      BIAS ? reinterpret_cast<const TW *>(bias->data_ptr()) : nullptr, meta.data_ptr<uint8_t>(), total, D,
      x.stride(0) / 8);
}

}  // namespace
}  // namespace conv_sm90

using namespace conv_sm90;

// Returns part [tiles, D, 5] (dw taps 0..3, db); the caller folds it.
at::Tensor varlen_bwd_cuda(at::Tensor dx, const at::Tensor &x, const at::Tensor &dy,
                           const at::Tensor &w, const c10::optional<at::Tensor> &bias,
                           const at::Tensor &cu, bool silu_act, int64_t tile, int64_t threads) {
  const at::cuda::CUDAGuard guard(x.device());
  int total = x.size(0), D = x.size(1), nseq = cu.size(0) - 1;
  TORCH_CHECK(x.stride(1) == 1 && x.stride(0) % 8 == 0 && x.stride(0) >= D && (uintptr_t(x.data_ptr()) % 16) == 0 &&
                  dy.is_contiguous() && dx.is_contiguous(),
              "conv_sm90 VJP: x must have unit column stride and 16-byte aligned rows; dy and dx packed");
  int tiles = (total + tile - 1) / tile;
  auto part = at::empty({tiles, D, 5}, x.options().dtype(at::kFloat));
  if (total == 0 || nseq <= 0)
    return part.zero_();
  bool xbf = x.dtype() == at::kBFloat16, wbf = w.dtype() == at::kBFloat16, b = bias.has_value();
#define L(T, TW, TT, BT, CPT)                                                                        \
  if (b && silu_act) return launch_bwd<T, TW, TT, BT, CPT, true, true>(dx, part, x, dy, w, bias, cu, nseq, total, D), part; \
  if (b && !silu_act) return launch_bwd<T, TW, TT, BT, CPT, true, false>(dx, part, x, dy, w, bias, cu, nseq, total, D), part; \
  if (!b && silu_act) return launch_bwd<T, TW, TT, BT, CPT, false, true>(dx, part, x, dy, w, bias, cu, nseq, total, D), part; \
  return launch_bwd<T, TW, TT, BT, CPT, false, false>(dx, part, x, dy, w, bias, cu, nseq, total, D), part;
#define CFG(TT, BT)                                                                                  \
  if (tile == TT && threads == BT) {                                                                \
    if (xbf && wbf) { L(bf16, bf16, TT, BT, 8) }                                                    \
    if (xbf && !wbf) { L(bf16, float, TT, BT, 8) }                                                  \
    if (!xbf && wbf) { L(float, bf16, TT, BT, 8) }                                                  \
    L(float, float, TT, BT, 8)                                                                      \
  }
  CFG(64, 64)
  CFG(64, 128)
  CFG(128, 128)
  CFG(32, 64)
#undef CFG
#undef L
  TORCH_CHECK(false, "unsupported tile/threads");
}

// Swapped-axis VJP. Returns part [tile blocks, D, 5].
at::Tensor varlen_bwd2_cuda(at::Tensor dx, const at::Tensor &x, const at::Tensor &dy, const at::Tensor &w,
                            const c10::optional<at::Tensor> &bias, const at::Tensor &cu, bool silu_act,
                            int64_t tile, int64_t tb) {
  const at::cuda::CUDAGuard guard(x.device());
  int total = x.size(0), D = x.size(1), nseq = cu.size(0) - 1;
  TORCH_CHECK(x.stride(1) == 1 && x.stride(0) % 8 == 0 && x.stride(0) >= D && (uintptr_t(x.data_ptr()) % 16) == 0 &&
                  dy.is_contiguous() && dx.is_contiguous(),
              "conv_sm90 VJP: x must have unit column stride and 16-byte aligned rows; dy and dx packed");
  int tiles = (total + tile - 1) / tile;
  auto part = at::empty({(tiles + tb - 1) / tb, D, 5}, x.options().dtype(at::kFloat));
  if (total == 0 || nseq <= 0)
    return part.zero_();
  auto meta = at::empty({total + 3}, x.options().dtype(at::kByte));
  auto stream = at::cuda::getCurrentCUDAStream();
  token_meta<32><<<(total + 31) / 32, 32, 0, stream>>>(meta.data_ptr<uint8_t>(), cu.data_ptr<int>(), nseq, total);
  bool xbf = x.dtype() == at::kBFloat16, wbf = w.dtype() == at::kBFloat16, b = bias.has_value();
#define L(T, TW, TT, TB)                                                                             \
  if (b && silu_act) return launch_bwd2<T, TW, TT, TB, 8, true, true>(dx, part, x, dy, w, bias, meta, total, D), part; \
  if (b && !silu_act) return launch_bwd2<T, TW, TT, TB, 8, true, false>(dx, part, x, dy, w, bias, meta, total, D), part; \
  if (!b && silu_act) return launch_bwd2<T, TW, TT, TB, 8, false, true>(dx, part, x, dy, w, bias, meta, total, D), part; \
  return launch_bwd2<T, TW, TT, TB, 8, false, false>(dx, part, x, dy, w, bias, meta, total, D), part;
#define CFG(TT, TB)                                                                                  \
  if (tile == TT && tb == TB) {                                                                     \
    if (xbf && wbf) { L(bf16, bf16, TT, TB) }                                                       \
    if (xbf && !wbf) { L(bf16, float, TT, TB) }                                                     \
    if (!xbf && wbf) { L(float, bf16, TT, TB) }                                                     \
    L(float, float, TT, TB)                                                                         \
  }
  CFG(16, 128)
  CFG(32, 128)
#undef CFG
#undef L
  TORCH_CHECK(false, "unsupported tile/tile-block");
}
