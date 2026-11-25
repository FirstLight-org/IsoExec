// SPDX-License-Identifier: Apache-2.0
#include "common.cuh"
#include "vec.cuh"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace conv_sm100 {
namespace {

__device__ __forceinline__ float dsilu(float a) {
  float s = __fdividef(1.f, 1.f + expf(-a));
  return s * (1.f + a * (1.f - s));
}

/// Largest ``s`` in ``[0, nseq-1]`` with ``cu[s] <= t``.
__device__ __forceinline__ int seq_of(const int *__restrict__ cu, int nseq, int t) {
  int lo = 0, hi = nseq - 1;
  while (lo < hi) {
    int mid = (lo + hi + 1) >> 1;
    if (cu[mid] <= t)
      lo = mid;
    else
      hi = mid - 1;
  }
  return lo;
}

// `768 / BT` resident blocks per SM caps registers at 85 (80 after granularity) at BT=64: the packed-fp32
// interior loop (BODY2) otherwise allocates 92 and loses resident warps. Measured -7..-10% at the large
// RL shapes, bit-identical (execution only).
template <class T, class TW, int BT, int CPTb, bool BIAS, bool SILU>
__global__ __launch_bounds__(BT, 768 / BT) void varlen_bwd(
    T *__restrict__ dx, float *__restrict__ part, const T *__restrict__ x,
    const T *__restrict__ dy, const TW *__restrict__ w, const TW *__restrict__ bias,
    const int *__restrict__ cu, int nseq, int total, int D, int tile) {
  using vec = packv<T, CPTb>;
  const int ti = blockIdx.x;
  const int a = ti * tile;
  const int e = min(a + tile, total);
  const int c0 = (blockIdx.y * BT + threadIdx.x) * CPTb;
  if (a >= total || c0 >= D)
    return;

  float w0[CPTb], w1[CPTb], w2[CPTb], w3[CPTb], b[CPTb];
  load_weights<TW, CPTb>(w, c0, w0, w1, w2, w3);
#pragma unroll
  for (int j = 0; j < CPTb; ++j)
    b[j] = 0.f;
  if constexpr (BIAS) {
#pragma unroll
    for (int j = 0; j < CPTb; ++j)
      b[j] = widen(bias[c0 + j]);
  }

  const int64_t row = D / CPTb;
  const auto *xp = reinterpret_cast<const vec *>(x) + c0 / CPTb;
  const auto *dyp = reinterpret_cast<const vec *>(dy) + c0 / CPTb;
  auto *dxp = reinterpret_cast<vec *>(dx) + c0 / CPTb;
  auto ldx = [&](int t, float (&v)[CPTb]) { widen_vec<T, CPTb>(xp[int64_t(t) * row], v); };

  // Cursor at the tile's last token; `f_last` is how many of the next tile's head tokens still have
  // this tile's tail in their receptive field (same sequence only).
  int s = seq_of(cu, nseq, e - 1);
  int start = cu[s], end = cu[s + 1];
  const int f_last = min(end - 1 - (e - 1), 3);

  float g0[CPTb], g1[CPTb], g2[CPTb], g3[CPTb], gb[CPTb];
#pragma unroll
  for (int j = 0; j < CPTb; ++j)
    g0[j] = g1[j] = g2[j] = g3[j] = gb[j] = 0.f;

  if (a - start >= 3 && e + 3 <= end && a >= 8) {
    // --- interior fast path: pipelined reverse walk (see note 5) ---------------------------------
    // `X[p]` holds `x[u - p]` and `Y[p]` holds `dy[u - p]` for the eight positions `p` of the
    // current trip, `u` being the trip's newest token; `Z[p & 3]` holds that position's `dz`. Every
    // index below is a literal, so all three banks stay in registers.
    float X[8][CPTb];
    vec Y[8];
    float Z[4][CPTb];

    // `dz` of the three tokens above the tile -- positions -1, -2, -3, i.e. Z[3], Z[2], Z[1] -- as a
    // forward walk with four rolling registers. All three exist and have full history here.
    {
      float u0[CPTb], u1[CPTb], u2[CPTb], u3[CPTb];
      ldx(e - 3, u3);
      ldx(e - 2, u2);
      ldx(e - 1, u1);
      ldx(e, u0);
#pragma unroll
      for (int m = 0; m < 3; ++m) {
        vec gp = dyp[int64_t(e + m) * row];
#pragma unroll
        for (int j = 0; j < CPTb; ++j) {
          float acc = preact(u3[j], u2[j], u1[j], u0[j], w0[j], w1[j], w2[j], w3[j]);
          if constexpr (BIAS)
            acc = acc + b[j];
          float g = widen(gp.v[j]);
          if constexpr (SILU)
            Z[3 - m][j] = g * dsilu(acc);
          else
            Z[3 - m][j] = g;
        }
        if (m < 2) {
#pragma unroll
          for (int j = 0; j < CPTb; ++j) {
            u3[j] = u2[j];
            u2[j] = u1[j];
            u1[j] = u0[j];
          }
          ldx(e + m + 1, u0);
        }
      }
    }

    // Prime both banks with the first trip's eight tokens. The `a >= 8` in the guard above is what
    // lets every prefetch below run unclamped: the deepest index this path ever touches is `a - 8`.
    // (A tile that starts in the first eight tokens of the pack takes the general path; there is at
    // most one such tile per launch, and it is the one tile that cannot be interior anyway unless
    // the pack's first sequence starts at 0 and is long.)
#pragma unroll
    for (int p = 0; p < 8; ++p) {
      ldx(e - 1 - p, X[p]);
      Y[p] = dyp[int64_t(e - 1 - p) * row];
    }

    // `x`, `dy` and `dx` have one shape, so one row offset serves all three, and it walks the tile
    // by a constant stride: one 64-bit add per token instead of a 64-bit multiply per access (which
    // is what indexing by token id costs -- ptxas emitted three IMAD.WIDE per token for it).
    int u = e - 1;
    int64_t off = int64_t(u) * row;
    const int64_t row8 = 8 * row;
#define BODY(i)                                                                                    \
  {                                                                                                \
    vec o;                                                                                         \
    _Pragma("unroll")                                                                              \
    for (int j = 0; j < CPTb; ++j) {                                                                \
      const float h0 = X[(i) & 7][j], h1 = X[((i) + 1) & 7][j], h2 = X[((i) + 2) & 7][j],           \
                  h3 = X[((i) + 3) & 7][j];                                                        \
      float acc = preact(h3, h2, h1, h0, w0[j], w1[j], w2[j], w3[j]);                               \
      if constexpr (BIAS)                                                                           \
        acc = acc + b[j];                                                                           \
      float g = widen(Y[(i) & 7].v[j]);                                                             \
      float dz;                                                                                     \
      if constexpr (SILU)                                                                           \
        dz = g * dsilu(acc);                                                                        \
      else                                                                                          \
        dz = g;                                                                                     \
      float e1 = Z[((i) - 1) & 3][j], e2 = Z[((i) - 2) & 3][j], e3 = Z[((i) - 3) & 3][j];            \
      float gx = w3[j] * dz + w2[j] * e1 + w1[j] * e2 + w0[j] * e3;                                 \
      o.v[j] = narrow<T>(gx);                                                                        \
      g3[j] += dz * h0;                                                                              \
      g2[j] += dz * h1;                                                                              \
      g1[j] += dz * h2;                                                                              \
      g0[j] += dz * h3;                                                                              \
      gb[j] += dz;                                                                                   \
      Z[(i) & 3][j] = dz;                                                                            \
    }                                                                                                \
    dxp[off] = o;                                                                                    \
    const int64_t offn = off - row8;                                                                 \
    widen_vec<T, CPTb>(xp[offn], X[(i) & 7]);                                                        \
    Y[(i) & 7] = dyp[offn];                                                                          \
    off -= row;                                                                                      \
  }
#define BODY2(i)                                                                                   \
  {                                                                                                \
    const float2 h0 = make_float2(X[(i) & 7][0], X[(i) & 7][1]);                                   \
    const float2 h1 = make_float2(X[((i) + 1) & 7][0], X[((i) + 1) & 7][1]);                       \
    const float2 h2 = make_float2(X[((i) + 2) & 7][0], X[((i) + 2) & 7][1]);                       \
    const float2 h3 = make_float2(X[((i) + 3) & 7][0], X[((i) + 3) & 7][1]);                       \
    float2 acc = mul2(W0, h3);                                                                     \
    acc = add2(acc, mul2(W1, h2));                                                                 \
    acc = add2(acc, mul2(W2, h1));                                                                 \
    acc = add2(acc, mul2(W3, h0));                                                                 \
    if constexpr (BIAS)                                                                            \
      acc = add2(acc, B2);                                                                         \
    const float2 gg = make_float2(widen(Y[(i) & 7].v[0]), widen(Y[(i) & 7].v[1]));                 \
    float2 dz;                                                                                     \
    if constexpr (SILU) {                                                                          \
      const float e0 = expf(-acc.x), e1x = expf(-acc.y);                                           \
      const float2 den = add2(ONE2, make_float2(e0, e1x));                                         \
      const float2 sg = make_float2(__fdividef(1.f, den.x), __fdividef(1.f, den.y));               \
      const float2 r = mul2(sg, add2(ONE2, mul2(acc, sub2(ONE2, sg))));                            \
      dz = mul2(gg, r);                                                                            \
    } else {                                                                                       \
      dz = gg;                                                                                     \
    }                                                                                              \
    const float2 gx = add2(add2(add2(mul2(W3, dz), mul2(W2, Z2[((i) - 1) & 3])),                   \
                                mul2(W1, Z2[((i) - 2) & 3])),                                      \
                           mul2(W0, Z2[((i) - 3) & 3]));                                           \
    vec o;                                                                                         \
    o.v[0] = narrow<T>(gx.x);                                                                      \
    o.v[1] = narrow<T>(gx.y);                                                                      \
    G3 = add2(G3, mul2(dz, h0));                                                                   \
    G2 = add2(G2, mul2(dz, h1));                                                                   \
    G1 = add2(G1, mul2(dz, h2));                                                                   \
    G0 = add2(G0, mul2(dz, h3));                                                                   \
    GB = add2(GB, dz);                                                                             \
    Z2[(i) & 3] = dz;                                                                              \
    dxp[off] = o;                                                                                  \
    const int64_t offn = off - row8;                                                               \
    widen_vec<T, CPTb>(xp[offn], X[(i) & 7]);                                                      \
    Y[(i) & 7] = dyp[offn];                                                                        \
    off -= row;                                                                                    \
  }
    if constexpr (CPTb == 2) {
      // Channel pairs on packed fp32 (FMUL2/FADD2): the same scalar program, two lanes per issue.
      const float2 W0 = make_float2(w0[0], w0[1]), W1 = make_float2(w1[0], w1[1]),
                   W2 = make_float2(w2[0], w2[1]), W3 = make_float2(w3[0], w3[1]),
                   B2 = make_float2(b[0], b[1]), ONE2 = make_float2(1.f, 1.f);
      float2 Z2[4], G0 = make_float2(g0[0], g0[1]), G1 = make_float2(g1[0], g1[1]),
                    G2 = make_float2(g2[0], g2[1]), G3 = make_float2(g3[0], g3[1]),
                    GB = make_float2(gb[0], gb[1]);
#pragma unroll
      for (int k = 0; k < 4; ++k)
        Z2[k] = make_float2(Z[k][0], Z[k][1]);
      for (; u - 7 >= a; u -= 8) {
        BODY2(0) BODY2(1) BODY2(2) BODY2(3) BODY2(4) BODY2(5) BODY2(6) BODY2(7)
      }
      const int r2 = u - a + 1;
#pragma unroll
      for (int i = 0; i < 7; ++i) {
        if (i < r2) {
          BODY2(i)
        }
      }
      g0[0] = G0.x; g0[1] = G0.y; g1[0] = G1.x; g1[1] = G1.y; g2[0] = G2.x; g2[1] = G2.y;
      g3[0] = G3.x; g3[1] = G3.y; gb[0] = GB.x; gb[1] = GB.y;
    } else {
    for (; u - 7 >= a; u -= 8) {
      BODY(0) BODY(1) BODY(2) BODY(3) BODY(4) BODY(5) BODY(6) BODY(7)
    }
    const int r = u - a + 1;  // 0..7 tokens left, positions 0..r-1 of the next trip
#pragma unroll
    for (int i = 0; i < 7; ++i) {
      if (i < r) {
        BODY(i)
      }
    }
    }  // CPTb != 2
#undef BODY
#undef BODY2
  } else {
    // --- general path: sequence boundaries, missing history, short tiles -------------------------
    float d1[CPTb], d2[CPTb], d3[CPTb];
#pragma unroll
    for (int j = 0; j < CPTb; ++j)
      d1[j] = d2[j] = d3[j] = 0.f;

    float r0[CPTb], r1[CPTb], r2[CPTb], r3[CPTb];
    if (f_last > 0) {
      // dz of tokens e, e+1, e+2 -- a *forward* walk with the same four rolling registers the
      // reverse loop uses. Taps beyond the sequence start are zeroed once at load; the shift
      // propagates that, so the products are the same `w * 0.f` the masked form computed.
      const int k0 = min(e - start, 3);
      ldx(e, r0);
#pragma unroll
      for (int j = 0; j < CPTb; ++j)
        r1[j] = r2[j] = r3[j] = 0.f;
      if (k0 >= 1)
        ldx(e - 1, r1);
      if (k0 >= 2)
        ldx(e - 2, r2);
      if (k0 >= 3)
        ldx(e - 3, r3);
#pragma unroll
      for (int m = 0; m < 3; ++m) {
        if (m < f_last) {
          vec gp = dyp[int64_t(e + m) * row];
#pragma unroll
          for (int j = 0; j < CPTb; ++j) {
            float acc = preact(r3[j], r2[j], r1[j], r0[j], w0[j], w1[j], w2[j], w3[j]);
            if constexpr (BIAS)
              acc = acc + b[j];
            float g = widen(gp.v[j]);
            float z;
            if constexpr (SILU)
              z = g * dsilu(acc);
            else
              z = g;
            if (m == 0)
              d1[j] = z;
            else if (m == 1)
              d2[j] = z;
            else
              d3[j] = z;
          }
          if (m + 1 < f_last) {
#pragma unroll
            for (int j = 0; j < CPTb; ++j) {
              r3[j] = r2[j];
              r2[j] = r1[j];
              r1[j] = r0[j];
            }
            ldx(e + m + 1, r0);
          }
        }
      }
    }

    // Window for t = e-1. Taps past the sequence start are masked at use, never read.
    {
      const int k = min(e - 1 - start, 3);
      ldx(e - 1, r0);
      if (k >= 1) ldx(e - 2, r1);
      if (k >= 2) ldx(e - 3, r2);
      if (k >= 3) ldx(e - 4, r3);
    }
    for (int t = e - 1; t >= a; --t) {
      while (t < start && s > 0) {  // walked back into the previous sequence
        --s;
        end = start;
        start = cu[s];
      }
      const int k = min(t - start, 3), f = min(end - 1 - t, 3);
      vec gp = dyp[int64_t(t) * row];
      vec o;
#pragma unroll
      for (int j = 0; j < CPTb; ++j) {
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
      dxp[int64_t(t) * row] = o;
      if (t > a) {
        if (k == 0) {
          // t starts its sequence: t-1 belongs to the previous one, reload its window raw.
          const int kn = min(t - 1 - cu[seq_of(cu, nseq, t - 1)], 3);
          ldx(t - 1, r0);
          if (kn >= 1) ldx(t - 2, r1);
          if (kn >= 2) ldx(t - 3, r2);
          if (kn >= 3) ldx(t - 4, r3);
        } else {
          const int kn = min(t - 1 - start, 3);
#pragma unroll
          for (int j = 0; j < CPTb; ++j) {
            r0[j] = r1[j];
            r1[j] = r2[j];
            r2[j] = r3[j];
          }
          if (kn >= 3) ldx(t - 4, r3);
        }
      }
    }
  }

  float *pp = part + int64_t(ti) * D * 5 + int64_t(c0) * 5;
#pragma unroll
  for (int j = 0; j < CPTb; ++j) {
    pp[j * 5] = g0[j];
    pp[j * 5 + 1] = g1[j];
    pp[j * 5 + 2] = g2[j];
    pp[j * 5 + 3] = g3[j];
    pp[j * 5 + 4] = gb[j];
  }
}

template <int NA>
__device__ __forceinline__ float4 ld_row(const float4 *p) {
  if constexpr (NA) {
    float4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.f32 {%0,%1,%2,%3}, [%4];"
                 : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
                 : "l"(p));
    return v;
  } else {
    return __ldg(p);
  }
}

// CHB channels per block (= 5*CHB/4 float4 columns), U rows in flight per thread, NA: read with the
// L1 no-allocate hint (a hint, not a different load).
template <class TW, class TB, int CHB, int U, int NA>
__global__ void fold_dwdb(const float *__restrict__ part, TW *__restrict__ dw, TB *__restrict__ db,
                          int H, int D, int G) {
  constexpr int NC4 = CHB * 5 / 4;  // float4 columns per block
  constexpr int NF = CHB * 5;       // floats per block row
  __shared__ float sm[128 * NF];
  const int c4 = threadIdx.x % NC4;  // float4 column within the block's NF floats
  const int g = threadIdx.x / NC4;   // row group, 0..G-1
  const int64_t rowf4 = int64_t(D) * 5 / 4;
  const float4 *base = reinterpret_cast<const float4 *>(part) + int64_t(blockIdx.x) * NC4 + c4;
  float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
  int r = g;
  if (r < H)
    acc = ld_row<NA>(base + r * rowf4);
  r += G;
  for (; r + (U - 1) * G < H; r += U * G) {
    float4 v[U];
#pragma unroll
    for (int u = 0; u < U; ++u)
      v[u] = ld_row<NA>(base + int64_t(r + u * G) * rowf4);
#pragma unroll
    for (int u = 0; u < U; ++u) {
      acc.x = __fadd_rn(acc.x, v[u].x);
      acc.y = __fadd_rn(acc.y, v[u].y);
      acc.z = __fadd_rn(acc.z, v[u].z);
      acc.w = __fadd_rn(acc.w, v[u].w);
    }
  }
  for (; r < H; r += G) {
    float4 v = ld_row<NA>(base + int64_t(r) * rowf4);
    acc.x = __fadd_rn(acc.x, v.x);
    acc.y = __fadd_rn(acc.y, v.y);
    acc.z = __fadd_rn(acc.z, v.z);
    acc.w = __fadd_rn(acc.w, v.w);
  }
  reinterpret_cast<float4 *>(sm)[g * NC4 + c4] = acc;
  __syncthreads();
  for (int f = threadIdx.x; f < NF; f += blockDim.x) {  // the block's CHB channels x (taps 0..3, bias)
    float s = sm[f];
    const int ng = min(H, G);
    for (int t = 1; t < ng; ++t)
      s = __fadd_rn(s, sm[t * NF + f]);
    const int ch = blockIdx.x * CHB + f / 5, k = f % 5;
    if (k < 4) {
      if (dw != nullptr)
        dw[int64_t(ch) * 4 + k] = narrow<TW>(s);
    } else if (db != nullptr) {
      db[ch] = narrow<TB>(s);
    }
  }
}

template <class TW, class TB, int CHB, int U, int NA>
void launch_fold(const at::Tensor &part, at::Tensor *dw, at::Tensor *db, int H, int D, int G) {
  fold_dwdb<TW, TB, CHB, U, NA><<<D / CHB, (CHB * 5 / 4) * G, 0, at::cuda::getCurrentCUDAStream()>>>(
      part.data_ptr<float>(), dw ? reinterpret_cast<TW *>(dw->data_ptr()) : nullptr,
      db ? reinterpret_cast<TB *>(db->data_ptr()) : nullptr, H, D, G);
}

template <int CHB, int U, int NA>
void fold_types(at::ScalarType sw, at::ScalarType sb, const at::Tensor &part, at::Tensor *pw,
                at::Tensor *pb, int H, int D, int G) {
  if (sw == at::kBFloat16 && sb == at::kBFloat16) return launch_fold<bf16, bf16, CHB, U, NA>(part, pw, pb, H, D, G);
  if (sw == at::kBFloat16) return launch_fold<bf16, float, CHB, U, NA>(part, pw, pb, H, D, G);
  if (sb == at::kBFloat16) return launch_fold<float, bf16, CHB, U, NA>(part, pw, pb, H, D, G);
  return launch_fold<float, float, CHB, U, NA>(part, pw, pb, H, D, G);
}

template <class T, class TW, int BT, int CPTb, bool BIAS, bool SILU>
void launch_bwd(at::Tensor &dx, at::Tensor &part, const at::Tensor &x, const at::Tensor &dy,
                const at::Tensor &w, const c10::optional<at::Tensor> &bias, const at::Tensor &cu,
                int nseq, int total, int D, int tile) {
  int groups = D / CPTb;
  dim3 grid((total + tile - 1) / tile, (groups + BT - 1) / BT);
  varlen_bwd<T, TW, BT, CPTb, BIAS, SILU><<<grid, BT, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<T *>(dx.data_ptr()), part.data_ptr<float>(),
      reinterpret_cast<const T *>(x.data_ptr()), reinterpret_cast<const T *>(dy.data_ptr()),
      reinterpret_cast<const TW *>(w.data_ptr()),
      BIAS ? reinterpret_cast<const TW *>(bias->data_ptr()) : nullptr, cu.data_ptr<int>(), nseq,
      total, D, tile);
}

}  // namespace
}  // namespace conv_sm100

using namespace conv_sm100;

// Returns part [tiles, D, 5] (dw taps 0..3, db); the caller folds it over the tile axis with a tree
// that is fixed for a fixed plan (ops.py's `part.sum(0)`).
at::Tensor varlen_bwd_cuda(at::Tensor dx, const at::Tensor &x, const at::Tensor &dy,
                           const at::Tensor &w, const c10::optional<at::Tensor> &bias,
                           const at::Tensor &cu, bool silu_act, int64_t tile, int64_t threads,
                           int64_t cpt) {
  const at::cuda::CUDAGuard guard(x.device());
  int total = x.size(0), D = x.size(1), nseq = cu.size(0) - 1;
  TORCH_CHECK(tile >= 1 && tile <= (1 << 20), "conv_sm100: tile out of range");
  int tiles = int((total + tile - 1) / tile);
  auto part = at::empty({tiles, D, 5}, x.options().dtype(at::kFloat));
  if (total == 0 || nseq <= 0)
    return part.zero_();
  bool xbf = x.dtype() == at::kBFloat16, wbf = w.dtype() == at::kBFloat16, b = bias.has_value();
  int T = int(tile);
#define L(T_, TW_, BT, CP)                                                                           \
  if (b && silu_act) return launch_bwd<T_, TW_, BT, CP, true, true>(dx, part, x, dy, w, bias, cu, nseq, total, D, T), part; \
  if (b && !silu_act) return launch_bwd<T_, TW_, BT, CP, true, false>(dx, part, x, dy, w, bias, cu, nseq, total, D, T), part; \
  if (!b && silu_act) return launch_bwd<T_, TW_, BT, CP, false, true>(dx, part, x, dy, w, bias, cu, nseq, total, D, T), part; \
  return launch_bwd<T_, TW_, BT, CP, false, false>(dx, part, x, dy, w, bias, cu, nseq, total, D, T), part;
#define CFG(BT, CP)                                                                                  \
  if (threads == BT && cpt == CP) {                                                                  \
    if (xbf && wbf) { L(bf16, bf16, BT, CP) }                                                        \
    if (xbf && !wbf) { L(bf16, float, BT, CP) }                                                      \
    if (!xbf && wbf) { L(float, bf16, BT, CP) }                                                      \
    L(float, float, BT, CP)                                                                          \
  }
  // (threads, channels per thread). CPTb is the occupancy knob (see the header comment); the widths
  // give the channel axis enough blocks at D=1024 as well as D=2048. CPTb == 8 is deliberately not
  // instantiated: the full sweep found it never competitive at any shape (231 registers, 6 resident
  // warps), and the interior fast path's three register banks make it worse, not better.
  CFG(64, 4)
  CFG(128, 4)
  CFG(64, 2)
  CFG(128, 2)
#undef CFG
#undef L
  TORCH_CHECK(false, "unsupported threads/cpt (bwd: cpt is 2 or 4, threads 64 or 128)");
}

// dw [D, 4] (weight dtype) and db [D] (bias dtype) from the partial rows, by the declared fold tree
// above. Either output may be absent (not requested). `groups` is the tree's G (<= 128); `variant`
// picks an execution shape only (channels per block, rows in flight, load hint) and cannot move a bit.
void fold_dwdb_cuda(const at::Tensor &part, const c10::optional<at::Tensor> &dw,
                    const c10::optional<at::Tensor> &db, int64_t groups, int64_t variant) {
  const at::cuda::CUDAGuard guard(part.device());
  TORCH_CHECK(part.dim() == 3 && part.size(2) == 5 && part.scalar_type() == at::kFloat &&
                  part.is_contiguous(),
              "conv_sm100: fold wants a contiguous fp32 [tiles, D, 5] partial buffer");
  const int H = int(part.size(0)), D = int(part.size(1));
  TORCH_CHECK(D % 4 == 0, "conv_sm100: fold needs D % 4 == 0");
  TORCH_CHECK(groups >= 1 && groups <= 128, "conv_sm100: fold groups must be in [1, 128]");
  if (D == 0)
    return;
  const int G = int(groups);
  at::Tensor d0 = dw.value_or(at::Tensor()), d1 = db.value_or(at::Tensor());
  at::Tensor *pw = dw.has_value() ? &d0 : nullptr, *pb = db.has_value() ? &d1 : nullptr;
  auto kind = [](const at::Tensor *t) { return t ? t->scalar_type() : at::kFloat; };
  const auto sw = kind(pw), sb = kind(pb);
  TORCH_CHECK((sw == at::kBFloat16 || sw == at::kFloat) && (sb == at::kBFloat16 || sb == at::kFloat),
              "conv_sm100: fold outputs must be bf16 or fp32");
#define FV(ID, CHB, U, NA)                                                                          \
  if (variant == ID && D % CHB == 0 && (CHB * 5 / 4) * G <= 1024) return fold_types<CHB, U, NA>(sw, sb, part, pw, pb, H, D, G);
  FV(0, 8, 4, 0) FV(1, 4, 16, 0) FV(2, 4, 4, 0) FV(3, 8, 8, 0) FV(4, 8, 16, 0) FV(5, 4, 8, 1)
  FV(6, 8, 8, 1) FV(7, 16, 8, 0) FV(8, 4, 16, 1) FV(9, 8, 4, 0) FV(10, 8, 2, 0) FV(11, 4, 2, 0)
  FV(12, 8, 4, 1) FV(13, 16, 4, 0)
#undef FV
  return fold_types<4, 4, 0>(sw, sb, part, pw, pb, H, D, G);
}
