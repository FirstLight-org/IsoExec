// SPDX-License-Identifier: Apache-2.0
// Fused per-head RMSNorm (optionally zero-centred, g = 1 + w) + partial neox
// RoPE on q and k, forward and backward.
//
// Forward numerical contract (rope_sm90 v1, see README.md), per (token, head)
// row x[0..d), d = 2^k >= 64, all arithmetic fp32 RN, FMA off:
//   ss   = balanced binary tree over the d leaves x_i*x_i, lower index on the
//          left: level 0 adds (x_2u^2, x_2u+1^2), and so on.
//   inv  = 1 /rn sqrt_rn(ss /rn d +rn eps32)
//   g_i  = 1 +rn w_i (zero-centred) or w_i, w widened exactly
//   n_i  = (x_i *rn inv) *rn g_i
//   rotated lanes i < rot: rotate(n_j, n_j+h, cos_j, sin_j) as in rope.cu
//   out  = RN narrowing of the fp32 value; no intermediate narrowing.
// Execution plan (free): grid (token, head group of 128/U rows), U = d/2
// threads per row own the element pairs (2u, 2u+1). The tree is evaluated as
// a pair add per thread, an ascending-offset warp butterfly per 32 threads and
// a segment tree in the row leader.
//
// Backward v2 keeps the dInput calculus and uses a fixed FP32 dWeight fold.
#include "common.cuh"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda/warp>
#include <optional>
#include <vector>

using namespace rope_sm90;

namespace {

constexpr int DW_GROUP_SHIFT = 4;

struct Rows {
  int64_t s0, s1; // input strides
  int64_t o0, o1; // output strides
  int64_t d0, d1; // upstream-gradient strides (backward)
  int heads;
};

// Balanced tree over the 32 lanes of every arch warp, result in lane 0.
__device__ __forceinline__ float warp_tree(float acc, int tid) {
#pragma unroll
  for (int s = 0; s < 5; ++s) {
    auto sh = cuda::device::warp_shuffle_down(
        acc, 1 << s, 0xFFFFFFFFu, cuda::std::integral_constant<int, 32>{});
    if (sh.pred && (tid & ((2 << s) - 1)) == 0)
      acc = __fadd_rn(acc, sh.data);
  }
  return acc;
}

template <int SEGS>
__device__ __forceinline__ float segment_tree(const float *seg) {
  float a[SEGS];
#pragma unroll
  for (int g = 0; g < SEGS; ++g)
    a[g] = seg[g];
#pragma unroll
  for (int span = 1; span < SEGS; span *= 2)
#pragma unroll
    for (int g = 0; g + span < SEGS; g += 2 * span)
      a[g] = __fadd_rn(a[g], a[g + span]);
  return a[0];
}

template <class W, bool ZC>
__device__ __forceinline__ void gain2(const W *w, float &g0, float &g1) {
  rd2(w, g0, g1);
  if constexpr (ZC) {
    g0 = __fadd_rn(1.f, g0);
    g1 = __fadd_rn(1.f, g1);
  }
}

template <class T, class W, bool ZC, int LOGU, bool MROPE>
__global__ void qknorm_fwd(const T *q, const T *k, T *qo, T *ko, const W *wq,
                           const W *wk, float *inv_out,
                           const __grid_constant__ Rows rq,
                           const __grid_constant__ Rows rk,
                           const __grid_constant__ Positions P, int rot,
                           float eps) {
  constexpr int U = 1 << LOGU, D = 2 * U, SEGS = U / 32;
  constexpr int RPB = LOGU >= 7 ? 1 : (128 >> LOGU);
  const int heads = rq.heads + rk.heads;
  const int t = blockIdx.x, tid = threadIdx.x;
  const int w = tid >> LOGU, u = tid & (U - 1);
  const int h = blockIdx.y * RPB + w;
  const bool live = h < heads;
  const bool isq = h < rq.heads;
  const int hh = isq ? h : h - rq.heads;
  const T *src = isq ? q + t * rq.s0 + hh * rq.s1 : k + t * rk.s0 + hh * rk.s1;
  __shared__ float seg[RPB * SEGS];
  __shared__ float sinv[RPB];

  float x0 = 0.f, x1 = 0.f;
  if (live)
    rd2(src + 2 * u, x0, x1);
  float acc = warp_tree(__fadd_rn(__fmul_rn(x0, x0), __fmul_rn(x1, x1)), tid);
  if ((tid & 31) == 0)
    seg[tid >> 5] = acc;
  __syncthreads();
  if (u == 0 && live) {
    float ss = segment_tree<SEGS>(seg + w * SEGS);
    float inv = __fdiv_rn(1.f, __fsqrt_rn(__fadd_rn(__fdiv_rn(ss, float(D)), eps)));
    sinv[w] = inv;
    if (inv_out)
      inv_out[int64_t(t) * heads + h] = inv;
  }
  __syncthreads();

  const int half = rot / 2;
  const bool rotating = 2 * u < half, passing = 2 * u >= rot;
  if (live && (rotating || passing)) {
    const W *wt = isq ? wq : wk;
    T *dst = isq ? qo + t * rq.o0 + hh * rq.o1 : ko + t * rk.o0 + hh * rk.o1;
    const float inv = sinv[w];
    float g0, g1;
    gain2<W, ZC>(wt + 2 * u, g0, g1);
    float n0 = __fmul_rn(__fmul_rn(x0, inv), g0);
    float n1 = __fmul_rn(__fmul_rn(x1, inv), g1);
    if (rotating) {
      const int j = 2 * u;
      float p0, p1, e0, e1;
      rd2(src + j + half, p0, p1);
      gain2<W, ZC>(wt + j + half, e0, e1);
      float m0 = __fmul_rn(__fmul_rn(p0, inv), e0);
      float m1 = __fmul_rn(__fmul_rn(p1, inv), e1);
      const float *r0 = table_row<MROPE>(P, t, j, rot);
      const float *r1 = MROPE ? table_row<MROPE>(P, t, j + 1, rot) : r0;
      float a0, b0, a1, b1;
      rotate(n0, m0, r0[j], r0[half + j], a0, b0);
      rotate(n1, m1, r1[j + 1], r1[half + j + 1], a1, b1);
      wr2(dst + j, a0, a1);
      wr2(dst + j + half, b0, b1);
    } else {
      wr2(dst + 2 * u, n0, n1);
    }
  }
}

// dn = R(-theta) dy on rotated lanes, dy elsewhere; xh = x*inv; dxh = dn*g;
// dot = sum_i dxh_i*xh_i; dx = inv*(dxh - xh*dot/d).
// dInput is independent of the separately scheduled deterministic dWeight fold.
template <class T, class W, bool ZC, int LOGU, bool MROPE>
__global__ void qknorm_bwd(const T *dyq, const T *dyk, const T *q, const T *k,
                           T *dxq, T *dxk, const W *wq, const W *wk,
                           const float *inv_in,
                           const __grid_constant__ Rows rq,
                           const __grid_constant__ Rows rk,
                           const __grid_constant__ Positions P, int rot) {
  constexpr int U = 1 << LOGU, D = 2 * U, SEGS = U / 32;
  constexpr int RPB = LOGU >= 7 ? 1 : (128 >> LOGU);
  const int heads = rq.heads + rk.heads;
  const int t = blockIdx.x, tid = threadIdx.x;
  const int w = tid >> LOGU, u = tid & (U - 1);
  const int h = blockIdx.y * RPB + w;
  const bool live = h < heads;
  const bool isq = h < rq.heads;
  const int hh = isq ? h : h - rq.heads;
  const int half = rot / 2;
  const bool rotating = 2 * u < half, passing = 2 * u >= rot;
  const bool active = live && (rotating || passing);
  __shared__ float seg[RPB * SEGS];
  __shared__ float sfac[RPB];

  float inv = 0.f, xa0 = 0.f, xa1 = 0.f, xb0 = 0.f, xb1 = 0.f;
  float ga0 = 0.f, ga1 = 0.f, gb0 = 0.f, gb1 = 0.f;
  float da0 = 0.f, da1 = 0.f, db0 = 0.f, db1 = 0.f; // dn
  float local = 0.f;
  if (active) {
    const T *src = isq ? q + t * rq.s0 + hh * rq.s1 : k + t * rk.s0 + hh * rk.s1;
    const T *dsrc = isq ? dyq + t * rq.d0 + hh * rq.d1 : dyk + t * rk.d0 + hh * rk.d1;
    const W *wt = isq ? wq : wk;
    inv = inv_in[int64_t(t) * heads + h];
    const int j = 2 * u;
    float x0, x1, y0, y1;
    rd2(src + j, x0, x1);
    rd2(dsrc + j, y0, y1);
    gain2<W, ZC>(wt + j, ga0, ga1);
    xa0 = __fmul_rn(x0, inv);
    xa1 = __fmul_rn(x1, inv);
    if (rotating) {
      float p0, p1, e0, e1;
      rd2(src + j + half, p0, p1);
      rd2(dsrc + j + half, e0, e1);
      gain2<W, ZC>(wt + j + half, gb0, gb1);
      xb0 = __fmul_rn(p0, inv);
      xb1 = __fmul_rn(p1, inv);
      const float *r0 = table_row<MROPE>(P, t, j, rot);
      const float *r1 = MROPE ? table_row<MROPE>(P, t, j + 1, rot) : r0;
      rotate_t(y0, e0, r0[j], r0[half + j], da0, db0);
      rotate_t(y1, e1, r1[j + 1], r1[half + j + 1], da1, db1);
    } else {
      da0 = y0;
      da1 = y1;
    }
    local = __fadd_rn(__fadd_rn(__fmul_rn(__fmul_rn(da0, ga0), xa0),
                                __fmul_rn(__fmul_rn(da1, ga1), xa1)),
                      __fadd_rn(__fmul_rn(__fmul_rn(db0, gb0), xb0),
                                __fmul_rn(__fmul_rn(db1, gb1), xb1)));
  }
  float acc = warp_tree(local, tid);
  if ((tid & 31) == 0)
    seg[tid >> 5] = acc;
  __syncthreads();
  if (u == 0 && live)
    sfac[w] = __fdiv_rn(segment_tree<SEGS>(seg + w * SEGS), float(D));
  __syncthreads();
  if (active) {
    const float f = sfac[w];
    T *dst = isq ? dxq + t * rq.o0 + hh * rq.o1 : dxk + t * rk.o0 + hh * rk.o1;
    const int j = 2 * u;
    auto grad = [&](float dn, float g, float xh) {
      return __fmul_rn(inv, __fsub_rn(__fmul_rn(dn, g), __fmul_rn(xh, f)));
    };
    wr2(dst + j, grad(da0, ga0, xa0), grad(da1, ga1, xa1));
    if (rotating)
      wr2(dst + j + half, grad(db0, gb0, xb0), grad(db1, gb1, xb1));
  }
}

// One writer for each (16-token group, head, column). The contribution has
// exactly the old dn*xh FP32 rounding, with tokens folded in ascending order.
template <class T, int LOGU, bool MROPE>
__global__ void qknorm_dw_partials(const T *dyq, const T *dyk, const T *q,
                                  const T *k, const float *inv_in,
                                  float *pq, float *pk, int64_t tokens,
                                  const __grid_constant__ Rows rq,
                                  const __grid_constant__ Rows rk,
                                  const __grid_constant__ Positions P, int rot) {
  constexpr int U = 1 << LOGU, D = 2 * U;
  constexpr int RPB = LOGU >= 7 ? 1 : (128 >> LOGU);
  const int group = blockIdx.x, u = threadIdx.x & (U - 1);
  const int h = blockIdx.y * RPB + (threadIdx.x >> LOGU);
  const int heads = rq.heads + rk.heads;
  const int j = 2 * u, half = rot / 2;
  const bool rotating = j < half, passing = j >= rot;
  if (h >= heads || !(rotating || passing))
    return;
  const bool isq = h < rq.heads;
  const int hh = isq ? h : h - rq.heads;
  const Rows r = isq ? rq : rk;
  const T *x = isq ? q : k, *dy = isq ? dyq : dyk;
  float a0 = 0.f, a1 = 0.f, b0 = 0.f, b1 = 0.f;
  const int64_t begin = int64_t(group) << P.gshift;
  const int64_t stop = begin + (int64_t(1) << P.gshift);
  const int64_t end = stop < tokens ? stop : tokens;
  for (int64_t t = begin; t < end; ++t) {
    const T *src = x + t * r.s0 + hh * r.s1;
    const T *dsrc = dy + t * r.d0 + hh * r.d1;
    const float inv = inv_in[t * heads + h];
    float x0, x1, y0, y1;
    rd2(src + j, x0, x1);
    rd2(dsrc + j, y0, y1);
    float dn0 = y0, dn1 = y1;
    if (rotating) {
      float xb0, xb1, yb0, yb1, db0, db1;
      rd2(src + j + half, xb0, xb1);
      rd2(dsrc + j + half, yb0, yb1);
      const float *r0 = table_row<MROPE>(P, t, j, rot);
      const float *r1 = MROPE ? table_row<MROPE>(P, t, j + 1, rot) : r0;
      rotate_t(y0, yb0, r0[j], r0[half + j], dn0, db0);
      rotate_t(y1, yb1, r1[j + 1], r1[half + j + 1], dn1, db1);
      b0 = __fadd_rn(b0, __fmul_rn(db0, __fmul_rn(xb0, inv)));
      b1 = __fadd_rn(b1, __fmul_rn(db1, __fmul_rn(xb1, inv)));
    }
    a0 = __fadd_rn(a0, __fmul_rn(dn0, __fmul_rn(x0, inv)));
    a1 = __fadd_rn(a1, __fmul_rn(dn1, __fmul_rn(x1, inv)));
  }
  float *row = (isq ? pq : pk) + (int64_t(group) * r.heads + hh) * D;
  *reinterpret_cast<float2 *>(row + j) = make_float2(a0, a1);
  if (rotating)
    *reinterpret_cast<float2 *>(row + j + half) = make_float2(b0, b1);
}

// Finish in fixed group-major, then head-major order and narrow exactly once.
template <class W>
__global__ void qknorm_dw_fold(const float *pq, const float *pk, W *dwq, W *dwk,
                              int64_t groups, int hq, int hk, int width) {
  const int j = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (j >= width)
    return;
  const bool isq = blockIdx.y == 0;
  const float *partial = isq ? pq : pk;
  const int heads = isq ? hq : hk;
  float a0 = 0.f, a1 = 0.f;
  for (int64_t group = 0; group < groups; ++group)
    for (int head = 0; head < heads; ++head) {
      const float2 value = *reinterpret_cast<const float2 *>(partial + (group * heads + head) * width + j);
      a0 = __fadd_rn(a0, value.x);
      a1 = __fadd_rn(a1, value.y);
    }
  wr2((isq ? dwq : dwk) + j, a0, a1);
}

template <class T> T *ptr(const at::Tensor &t) {
  return t.defined() && t.numel() ? static_cast<T *>(t.data_ptr()) : nullptr;
}

Positions positions_of(const at::Tensor &positions, const at::Tensor &table,
                       int64_t m1, int64_t m2) {
  const bool one_d = positions.dim() == 1;
  return Positions{positions.data_ptr<int64_t>(), table.data_ptr<float>(),
                   one_d ? 0 : positions.stride(0), positions.stride(-1),
                   table.size(0), uint64_t(m1), uint64_t(m2), 0};
}

int log_units(int64_t d) {
  switch (d) {
  case 64:
    return 5;
  case 128:
    return 6;
  case 256:
    return 7;
  }
  TORCH_CHECK(false, "rope_sm90 qk norm: head_dim must be 64, 128 or 256");
}

} // namespace

bool pair_aligned(const at::Tensor &t);
at::Tensor pair_operand(const at::Tensor &t);
bool positions_ok(const at::Tensor &p, const at::Tensor &x);
bool table_ok(const at::Tensor &t, const at::Tensor &x);

namespace {
bool qk_ok(const at::Tensor &q, const at::Tensor &k, const at::Tensor &wq,
           const at::Tensor &wk,
           const std::optional<at::Tensor> &positions,
           const std::optional<at::Tensor> &table, int64_t m1, int64_t m2) {
  auto st = q.scalar_type();
  const bool rt = table.has_value();
  const int64_t rot = rt && table->dim() == 2 ? table->size(1) : 0;
  const int64_t d = q.dim() == 3 ? q.size(2) : 0;
  return q.is_cuda() && q.dim() == 3 && k.dim() == 3 &&
         k.device() == q.device() && wq.device() == q.device() &&
         wk.device() == q.device() && k.scalar_type() == st &&
         (st == at::kBFloat16 || st == at::kHalf || st == at::kFloat) &&
         k.size(0) == q.size(0) && k.size(2) == d &&
         (d == 64 || d == 128 || d == 256) && wq.dim() == 1 &&
         wk.dim() == 1 && wq.size(0) == d && wk.size(0) == d &&
         wq.scalar_type() == wk.scalar_type() &&
         (wq.scalar_type() == st || wq.scalar_type() == at::kFloat) &&
         q.size(0) < (int64_t(1) << 31) &&
         (!rt || (positions.has_value() && table_ok(*table, q) &&
                  positions_ok(*positions, q) && rot > 0 && rot % 4 == 0 &&
                  rot <= d && mrope_mask_extent_ok(rot, m1, m2, positions->dim())));
}
} // namespace

// {qo, ko, inv} or {} when not scheduled. inv [T, Hq+Hk] fp32 when requested.
std::vector<at::Tensor>
qknorm_rope_forward(const at::Tensor &q_in, const at::Tensor &k_in,
                    const at::Tensor &wq_in, const at::Tensor &wk_in,
                    const std::optional<at::Tensor> &positions,
                    const std::optional<at::Tensor> &table, int64_t m1,
                    int64_t m2, double eps, bool zero_centered, bool want_inv) {
  if (!qk_ok(q_in, k_in, wq_in, wk_in, positions, table, m1, m2))
    return {};
  const c10::cuda::CUDAGuard guard(q_in.device());
  at::Tensor q = pair_operand(q_in), k = pair_operand(k_in);
  at::Tensor wq = pair_operand(wq_in.reshape({1, 1, -1})).view(-1);
  at::Tensor wk = pair_operand(wk_in.reshape({1, 1, -1})).view(-1);
  const int64_t T = q.size(0);
  const int hq = q.size(1), hk = k.size(1);
  at::Tensor qo = at::empty(q.sizes(), q.options());
  at::Tensor ko = at::empty(k.sizes(), k.options());
  at::Tensor inv = want_inv ? at::empty({T, int64_t(hq + hk)}, q.options().dtype(at::kFloat))
                            : at::Tensor();
  if (!T)
    return {qo, ko, inv};
  const int logu = log_units(q.size(2));
  const int rpb = logu >= 7 ? 1 : (128 >> logu);
  const int64_t rot = table.has_value() ? table->size(1) : 0;
  Rows rq{q.stride(0), q.stride(1), qo.stride(0), qo.stride(1), 0, 0, hq};
  Rows rk{k.stride(0), k.stride(1), ko.stride(0), ko.stride(1), 0, 0, hk};
  Positions P = rot ? positions_of(*positions, *table, m1, m2)
                    : Positions{nullptr, nullptr, 0, 0, 0, 0, 0, 0};
  const bool mrope = rot && (m1 | m2) != 0 && positions->dim() == 2;
  dim3 grid(T, (hq + hk + rpb - 1) / rpb);
  auto stream = at::cuda::getCurrentCUDAStream();
  const bool w32 = wq.scalar_type() == at::kFloat;
  float eps32 = float(eps);
#define FWD(TT, WW, ZC, L)                                                     \
  qknorm_fwd<TT, WW, ZC, L, MR><<<grid, 128, 0, stream>>>(                     \
      ptr<const TT>(q), ptr<const TT>(k), ptr<TT>(qo), ptr<TT>(ko),            \
      ptr<const WW>(wq), ptr<const WW>(wk), ptr<float>(inv), rq, rk, P,        \
      int(rot), eps32)
#define BY_MR(TT, WW, ZC, L)                                                   \
  do {                                                                         \
    if (mrope) {                                                               \
      constexpr bool MR = true;                                                \
      FWD(TT, WW, ZC, L);                                                      \
    } else {                                                                   \
      constexpr bool MR = false;                                               \
      FWD(TT, WW, ZC, L);                                                      \
    }                                                                          \
  } while (0)
#define BY_LOGU(TT, WW, ZC)                                                    \
  do {                                                                         \
    if (logu == 7)                                                             \
      BY_MR(TT, WW, ZC, 7);                                                    \
    else if (logu == 6)                                                        \
      BY_MR(TT, WW, ZC, 6);                                                    \
    else                                                                       \
      BY_MR(TT, WW, ZC, 5);                                                    \
  } while (0)
#define BY_W(TT)                                                               \
  do {                                                                         \
    if (w32) {                                                                 \
      if (zero_centered)                                                       \
        BY_LOGU(TT, float, true);                                              \
      else                                                                     \
        BY_LOGU(TT, float, false);                                             \
    } else {                                                                   \
      if (zero_centered)                                                       \
        BY_LOGU(TT, TT, true);                                                 \
      else                                                                     \
        BY_LOGU(TT, TT, false);                                                \
    }                                                                          \
  } while (0)
#define BY_T()                                                                 \
  do {                                                                         \
    switch (q.scalar_type()) {                                                 \
    case at::kBFloat16:                                                        \
      BY_W(__nv_bfloat16);                                                     \
      break;                                                                   \
    case at::kHalf:                                                            \
      BY_W(__half);                                                            \
      break;                                                                   \
    default:                                                                   \
      if (zero_centered)                                                       \
        BY_LOGU(float, float, true);                                           \
      else                                                                     \
        BY_LOGU(float, float, false);                                          \
    }                                                                          \
  } while (0)
  BY_T();
#undef FWD
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {qo, ko, inv};
}

// {dxq, dxk, gwq, gwk} (weight grads undefined unless requested) or {}.
std::vector<at::Tensor>
qknorm_rope_backward(const at::Tensor &dyq_in, const at::Tensor &dyk_in,
                     const at::Tensor &q_in, const at::Tensor &k_in,
                     const at::Tensor &wq_in, const at::Tensor &wk_in,
                     const at::Tensor &inv,
                     const std::optional<at::Tensor> &positions,
                     const std::optional<at::Tensor> &table, int64_t m1,
                     int64_t m2, bool zero_centered, bool want_w,
                     int64_t gshift) {
  TORCH_CHECK(gshift == DW_GROUP_SHIFT,
              "rope_sm90 qk backward v2 requires gshift=4 (fixed 16-token reduction recipe)");
  if (!qk_ok(q_in, k_in, wq_in, wk_in, positions, table, m1, m2) ||
      dyq_in.device() != q_in.device() || dyk_in.device() != q_in.device() ||
      dyq_in.sizes() != q_in.sizes() || dyk_in.sizes() != k_in.sizes() ||
      inv.dim() != 2 || inv.device() != q_in.device() ||
      !inv.is_contiguous() || inv.scalar_type() != at::kFloat ||
      inv.size(0) != q_in.size(0) || inv.size(1) != q_in.size(1) + k_in.size(1))
    return {};
  const c10::cuda::CUDAGuard guard(q_in.device());
  at::Tensor q = pair_operand(q_in), k = pair_operand(k_in);
  at::Tensor dyq = pair_operand(dyq_in.to(q.scalar_type()));
  at::Tensor dyk = pair_operand(dyk_in.to(q.scalar_type()));
  at::Tensor wq = pair_operand(wq_in.reshape({1, 1, -1})).view(-1);
  at::Tensor wk = pair_operand(wk_in.reshape({1, 1, -1})).view(-1);
  const int64_t T = q.size(0), d = q.size(2);
  const int hq = q.size(1), hk = k.size(1);
  at::Tensor dxq = at::empty(q.sizes(), q.options());
  at::Tensor dxk = at::empty(k.sizes(), k.options());
  const int64_t groups = (T + (int64_t(1) << gshift) - 1) >> gshift;
  auto fopt = q.options().dtype(at::kFloat);
  at::Tensor pq = want_w ? at::empty({groups, hq, d}, fopt) : at::Tensor();
  at::Tensor pk = want_w ? at::empty({groups, hk, d}, fopt) : at::Tensor();
  if (T) {
    const int logu = log_units(d);
    const int rpb = logu >= 7 ? 1 : (128 >> logu);
    const int64_t rot = table.has_value() ? table->size(1) : 0;
    Rows rq{q.stride(0), q.stride(1), dxq.stride(0), dxq.stride(1),
            dyq.stride(0), dyq.stride(1), hq};
    Rows rk{k.stride(0), k.stride(1), dxk.stride(0), dxk.stride(1),
            dyk.stride(0), dyk.stride(1), hk};
    Positions P = rot ? positions_of(*positions, *table, m1, m2)
                      : Positions{nullptr, nullptr, 0, 0, 0, 0, 0, 0};
    P.gshift = int(gshift);
    const bool mrope = rot && (m1 | m2) != 0 && positions->dim() == 2;
    dim3 grid(T, (hq + hk + rpb - 1) / rpb);
    auto stream = at::cuda::getCurrentCUDAStream();
    const bool w32 = wq.scalar_type() == at::kFloat;
#define FWD(TT, WW, ZC, L)                                                     \
  qknorm_bwd<TT, WW, ZC, L, MR><<<grid, 128, 0, stream>>>(                     \
      ptr<const TT>(dyq), ptr<const TT>(dyk), ptr<const TT>(q),                \
      ptr<const TT>(k), ptr<TT>(dxq), ptr<TT>(dxk), ptr<const WW>(wq),         \
      ptr<const WW>(wk), ptr<const float>(inv), rq, rk, P, int(rot))
    BY_T();
#undef FWD
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (want_w) {
      dim3 partial_grid(groups, (hq + hk + rpb - 1) / rpb);
#define FWD(TT, WW, ZC, L)                                                     \
      qknorm_dw_partials<TT, L, MR><<<partial_grid, 128, 0, stream>>>(          \
          ptr<const TT>(dyq), ptr<const TT>(dyk), ptr<const TT>(q),            \
          ptr<const TT>(k), ptr<const float>(inv), ptr<float>(pq),            \
          ptr<float>(pk), T, rq, rk, P, int(rot))
      BY_T();
#undef FWD
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  }
  at::Tensor gwq, gwk;
  if (want_w) {
    gwq = at::empty({d}, wq_in.options());
    gwk = at::empty({d}, wk_in.options());
    dim3 fold_grid((d + 255) / 256, 2);
    auto stream = at::cuda::getCurrentCUDAStream();
#define FOLD(WW)                                                               \
    qknorm_dw_fold<WW><<<fold_grid, 128, 0, stream>>>(                          \
        ptr<float>(pq), ptr<float>(pk), ptr<WW>(gwq), ptr<WW>(gwk),            \
        groups, hq, hk, int(d))
    if (wq_in.scalar_type() == at::kBFloat16) {
      FOLD(__nv_bfloat16);
    } else if (wq_in.scalar_type() == at::kHalf) {
      FOLD(__half);
    } else {
      FOLD(float);
    }
#undef FOLD
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {dxq, dxk, gwq, gwk};
}
#undef BY_T
#undef BY_W
#undef BY_LOGU
#undef BY_MR
