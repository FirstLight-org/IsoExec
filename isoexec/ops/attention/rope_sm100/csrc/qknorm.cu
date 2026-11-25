// SPDX-License-Identifier: Apache-2.0
#include "common.cuh"
#include "vecio.cuh"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda/warp>
#include <optional>
#include <vector>

using namespace rope_sm100;

namespace {

// Grid-of-rows (plan 0) threads/CTA. rope_sm90's Hopper literal is 128; this is a separate literal
// swept on B200 (README.md "Execution plan"). Both are plan choices, never bits.
constexpr int64_t kThreadsPerBlock = 256;

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

struct Rows {
  int64_t s0, s1; // input strides
  int64_t o0, o1; // output strides
  int64_t d0, d1; // upstream-gradient strides (backward)
  int heads;
};

template <class T, class W, bool ZC, int LOGU, bool MROPE>
__global__ void qknorm_fwd(const T *q, const T *k, T *qo, T *ko, const W *wq,
                           const W *wk, float *inv_out,
                           const __grid_constant__ Rows rq,
                           const __grid_constant__ Rows rk,
                           const __grid_constant__ Positions P, int rot,
                           float eps) {
  constexpr int U = 1 << LOGU, D = 2 * U;
  constexpr int RPB = (kThreadsPerBlock >> LOGU) < 1 ? 1 : int(kThreadsPerBlock >> LOGU);
  constexpr int SEGS = U / 32;
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

template <class T, class W, bool ZC, int LOGD, int LOGP, bool XSHFL, bool SPLIT, bool MROPE>
__global__ void qknorm_fwd_wr(const T *q, const T *k, T *qo, T *ko, const W *wq,
                              const W *wk, float *inv_out,
                              const __grid_constant__ Rows rq,
                              const __grid_constant__ Rows rk,
                              const __grid_constant__ Positions P, int rot,
                              float eps, int ntok, bool tabvec) {
  constexpr int D = 1 << LOGD, PL = 1 << LOGP, V = D / PL;
  if constexpr (V < 2 || V > 16) {
    return; // not a selectable (d, LOGP) pair; the host never launches it
  } else {
    // Row indexing. `ntok > 0` selects the FLAT form: the grid is one dimension over all
    // `ntok * heads` rows, so a CTA is always full. The 2-D form (grid.x = token, grid.y = head group)
    // needs no division, but it quantizes the head axis to whole CTAs -- at a tensor-parallel rank's
    // head count that is the dominant cost, not the division: 5 heads in CTAs of 2 rows leaves 1 of 6
    // row slots dead, 17% of every lane in the launch. The branch is grid-uniform.
    const int heads = rq.heads + rk.heads;
    const int tid = threadIdx.x;
    const int lane = tid & 31, u = tid & (PL - 1);
    (void)lane;
    int t, h;
    bool live;
    if (ntok > 0) {
      const int row = blockIdx.x * int(blockDim.x >> LOGP) + (tid >> LOGP);
      t = row / heads;
      h = row - t * heads;
      live = t < ntok;
      if (!live)
        t = 0, h = 0; // keep the loads and the table lookup in bounds; nothing is stored
    } else {
      t = blockIdx.x;
      h = blockIdx.y * int(blockDim.x >> LOGP) + (tid >> LOGP);
      live = h < heads;
    }
    const bool isq = h < rq.heads;
    const int hh = isq ? h : h - rq.heads;
    const T *src = isq ? q + t * rq.s0 + hh * rq.s1 : k + t * rk.s0 + hh * rk.s1;
    const int i0 = u * V;

    float x[V];
#pragma unroll
    for (int i = 0; i < V; ++i)
      x[i] = 0.f;
    if (live)
      rdv<T, V>(src + i0, x);

    // Balanced tree: the per-lane register tree, then LOGP ascending-offset shuffle steps.
    float acc = sumsq_tree<V>(x);
#pragma unroll
    for (int st = 0; st < LOGP; ++st) {
      auto sh = cuda::device::warp_shuffle_down(
          acc, 1 << st, 0xFFFFFFFFu, cuda::std::integral_constant<int, 32>{});
      if (sh.pred && (tid & ((2 << st) - 1)) == 0)
        acc = __fadd_rn(acc, sh.data);
    }
    float inv = 0.f;
    if (u == 0) {
      inv = __fdiv_rn(1.f, __fsqrt_rn(__fadd_rn(__fdiv_rn(acc, float(D)), eps)));
      if (inv_out && live)
        inv_out[int64_t(t) * heads + h] = inv;
    }
    inv = __shfl_sync(0xFFFFFFFFu, inv, lane & ~(PL - 1));

    const W *wt = isq ? wq : wk;
    float g[V], n[V];
    rdv_narrow<W, V>(wt + i0, g); // d elements total: L1-resident, alignment not gated
#pragma unroll
    for (int i = 0; i < V; ++i) {
      if constexpr (ZC)
        g[i] = __fadd_rn(1.f, g[i]);
      n[i] = __fmul_rn(__fmul_rn(x[i], inv), g[i]);
    }

    const int half = rot / 2;
    const int hp = V ? (half / V) : 0;
    const bool lower = i0 < half, upper = i0 >= half && i0 < rot;
    // The lane holding this lane's rotary partner. !SPLIT only ever asks below `half`, so it wants
    // u + hp; SPLIT asks from both sides.
    const int pu = (SPLIT && upper) ? (u - hp) : (u + hp);
    float m[V];
    if constexpr (XSHFL) {
      const int psrc = ((lane & ~(PL - 1)) + pu) & 31;
#pragma unroll
      for (int i = 0; i < V; ++i)
        m[i] = __shfl_sync(0xFFFFFFFFu, n[i], psrc);
    }

    if (!live)
      return;
    T *dst = isq ? qo + t * rq.o0 + hh * rq.o1 : ko + t * rk.o0 + hh * rk.o1;
    const bool rotary = SPLIT ? (lower || upper) : (i0 + V <= half);
    if (rotary) {
      if constexpr (!XSHFL) {
        const int pbase = (SPLIT && upper) ? (i0 - half) : (i0 + half);
        float px[V], pg[V];
        rdv<T, V>(src + pbase, px);
        rdv_narrow<W, V>(wt + pbase, pg);
#pragma unroll
        for (int i = 0; i < V; ++i) {
          if constexpr (ZC)
            pg[i] = __fadd_rn(1.f, pg[i]);
          m[i] = __fmul_rn(__fmul_rn(px[i], inv), pg[i]);
        }
      }
      if constexpr (SPLIT) {
        // This lane's V table columns are CONTIGUOUS -- [jbase, jbase+V) of the cos half and the same
        // window of the sin half -- so without MRoPE they come in as two vector loads instead of 2V
        // scalar ones (`tabvec`, which the host clears if the table's base is not 16-byte aligned, and
        // which MRoPE clears because there each column may read a different table ROW). Same floats,
        // a quarter of the load instructions; the table is fp32 and read, never rounded.
        const int jbase = lower ? i0 : (i0 - half);
        float cs[V], sv[V];
        if (!MROPE && tabvec) {
          const float *r = table_row<false>(P, t, 0, rot);
          rdv<float, V>(r + jbase, cs);
          rdv<float, V>(r + half + jbase, sv);
        } else {
#pragma unroll
          for (int i = 0; i < V; ++i) {
            const float *r = table_row<MROPE>(P, t, jbase + i, rot);
            cs[i] = r[jbase + i];
            sv[i] = r[half + jbase + i];
          }
        }
        float y[V];
#pragma unroll
        for (int i = 0; i < V; ++i)
          // n[i] is x1 below `half` and x2 above it; m[i] is the other one.
          y[i] = lower ? __fsub_rn(__fmul_rn(n[i], cs[i]), __fmul_rn(m[i], sv[i]))
                       : __fadd_rn(__fmul_rn(n[i], cs[i]), __fmul_rn(m[i], sv[i]));
        wrv<T, V>(dst + i0, y);
      } else {
        float a[V], b[V];
#pragma unroll
        for (int i = 0; i < V; ++i) {
          const int j = i0 + i;
          const float *r = table_row<MROPE>(P, t, j, rot);
          rotate(n[i], m[i], r[j], r[half + j], a[i], b[i]);
        }
        wrv<T, V>(dst + i0, a);
        wrv<T, V>(dst + i0 + half, b);
      }
    } else if (i0 >= rot) {
      wrv<T, V>(dst + i0, n);
    }
  }
}

// dn = R(-theta) dy on rotated lanes, dy elsewhere; xh = x*inv; dxh = dn*g;
// dot = sum_i dxh_i*xh_i; dx = inv*(dxh - xh*dot/d).
//
// One CTA owns RPB (head) rows for the 2^gshift consecutive tokens of one token group and walks them
// in ascending order. dx of each (token, head) row is the rope_sm90 recipe unchanged. The weight
// gradient dw_c = sum over (t, h) of dn_c * xh_c is never accumulated in memory: each thread folds
// its columns' products over the group's tokens in ascending order in registers, starting from +0,
// and stores the group's partial p[group, head, c] with a plain store -- exactly one writer per
// element, no atomics. (Same partial bits as rope_sm90 v2's qknorm_dw_partials at gshift = 4.)
template <class T, class W, bool ZC, int LOGU, bool MROPE>
__global__ void qknorm_bwd(const T *dyq, const T *dyk, const T *q, const T *k,
                           T *dxq, T *dxk, const W *wq, const W *wk,
                           const float *inv_in, float *pq, float *pk,
                           int64_t tokens,
                           const __grid_constant__ Rows rq,
                           const __grid_constant__ Rows rk,
                           const __grid_constant__ Positions P, int rot) {
  constexpr int U = 1 << LOGU, D = 2 * U;
  constexpr int RPB = (kThreadsPerBlock >> LOGU) < 1 ? 1 : int(kThreadsPerBlock >> LOGU);
  constexpr int SEGS = U / 32;
  const int heads = rq.heads + rk.heads;
  const int tid = threadIdx.x;
  const int w = tid >> LOGU, u = tid & (U - 1);
  const int h = blockIdx.y * RPB + w;
  const bool live = h < heads;
  const bool isq = h < rq.heads;
  const int hh = isq ? h : h - rq.heads;
  const int half = rot / 2;
  const bool rotating = 2 * u < half, passing = 2 * u >= rot;
  const bool active = live && (rotating || passing);
  const int j = 2 * u;
  const Rows &r = isq ? rq : rk;
  const T *x = isq ? q : k, *dy = isq ? dyq : dyk;
  T *dx = isq ? dxq : dxk;
  const W *wt = isq ? wq : wk;
  float *pp = isq ? pq : pk;
  __shared__ float seg[RPB * SEGS];
  __shared__ float sfac[RPB];

  // The gains do not depend on the token.
  float ga0 = 0.f, ga1 = 0.f, gb0 = 0.f, gb1 = 0.f;
  if (active) {
    gain2<W, ZC>(wt + j, ga0, ga1);
    if (rotating)
      gain2<W, ZC>(wt + j + half, gb0, gb1);
  }
  float wa0 = 0.f, wa1 = 0.f, wb0 = 0.f, wb1 = 0.f; // this group's dw partial
  const int64_t begin = int64_t(blockIdx.x) << P.gshift;
  const int64_t stop = begin + (int64_t(1) << P.gshift);
  const int64_t end = stop < tokens ? stop : tokens;
  for (int64_t t = begin; t < end; ++t) {
    float inv = 0.f, xa0 = 0.f, xa1 = 0.f, xb0 = 0.f, xb1 = 0.f;
    float da0 = 0.f, da1 = 0.f, db0 = 0.f, db1 = 0.f; // dn
    float local = 0.f;
    if (active) {
      const T *src = x + t * r.s0 + hh * r.s1;
      const T *dsrc = dy + t * r.d0 + hh * r.d1;
      inv = inv_in[t * heads + h];
      float x0, x1, y0, y1;
      rd2(src + j, x0, x1);
      rd2(dsrc + j, y0, y1);
      xa0 = __fmul_rn(x0, inv);
      xa1 = __fmul_rn(x1, inv);
      if (rotating) {
        float p0, p1, e0, e1;
        rd2(src + j + half, p0, p1);
        rd2(dsrc + j + half, e0, e1);
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
      if (pp) { // ascending token order within the group
        wa0 = __fadd_rn(wa0, __fmul_rn(da0, xa0));
        wa1 = __fadd_rn(wa1, __fmul_rn(da1, xa1));
        if (rotating) {
          wb0 = __fadd_rn(wb0, __fmul_rn(db0, xb0));
          wb1 = __fadd_rn(wb1, __fmul_rn(db1, xb1));
        }
      }
    }
    float acc = warp_tree(local, tid);
    if ((tid & 31) == 0)
      seg[tid >> 5] = acc;
    __syncthreads();
    if (u == 0 && live)
      sfac[w] = __fdiv_rn(segment_tree<SEGS>(seg + w * SEGS), float(D));
    __syncthreads();
    // The next token's seg/sfac writes are ordered after its first barrier, which every thread
    // reaches only after its reads below: two barriers per token suffice.
    if (active) {
      const float f = sfac[w];
      T *dst = dx + t * r.o0 + hh * r.o1;
      auto grad = [&](float dn, float g, float xh) {
        return __fmul_rn(inv, __fsub_rn(__fmul_rn(dn, g), __fmul_rn(xh, f)));
      };
      wr2(dst + j, grad(da0, ga0, xa0), grad(da1, ga1, xa1));
      if (rotating)
        wr2(dst + j + half, grad(db0, gb0, xb0), grad(db1, gb1, xb1));
    }
  }
  if (pp && active) {
    float *row = pp + (int64_t(blockIdx.x) * r.heads + hh) * D;
    *reinterpret_cast<float2 *>(row + j) = make_float2(wa0, wa1);
    if (rotating)
      *reinterpret_cast<float2 *>(row + j + half) = make_float2(wb0, wb1);
  }
}

// The warp-row twin of qknorm_bwd: 2^LOGP lanes own one (token group, head) row, V = d / 2^LOGP
// contiguous columns a lane, and walk the group's tokens in ascending order warp-synchronously (no
// shared memory, no barrier). It computes qknorm_bwd's bits, operation for operation:
//   * dn, xh, every dx element and every dw partial element are elementwise, with qknorm_bwd's
//     operands in qknorm_bwd's order (dn of an upper rotary column reads its lower partner's dy,
//     re-read from global memory; the table column is the lower one's, as in qknorm_bwd).
//   * qknorm_bwd's per-thread leaf `local` for its column pair (2m, 2m+1) is
//       rotary  : ((P_2m + P_2m+1) + (P_2m+half + P_2m+1+half)),   P_c = (dn_c * g_c) * xh_c
//       upper   : +0 (that thread is idle there)
//       passing : ((P_2m + P_2m+1) + +0)
//     and its reduction (warp_tree, then segment_tree) is one balanced binary tree over the d/2
//     leaves in pair order, lower index on the left. Here a lane builds its V/2 consecutive leaves
//     (an upper lane's P values arrive by shuffle), reduces them by the same balanced tree, and the
//     LOGP ascending-offset shuffle steps finish the same tree.
// The host takes it only when every operand row is V-aligned and (rot/2) % V == 0; otherwise
// qknorm_bwd runs. Both produce the same bits (tests/test_backward_gpu.py compares them).
template <class T, class W, bool ZC, int LOGD, int LOGP, bool MROPE>
__global__ void qknorm_bwd_wr(const T *__restrict__ dyq, const T *__restrict__ dyk,
                              const T *__restrict__ q, const T *__restrict__ k,
                              T *__restrict__ dxq, T *__restrict__ dxk, const W *__restrict__ wq,
                              const W *__restrict__ wk, const float *__restrict__ inv_in,
                              float *__restrict__ pq, float *__restrict__ pk,
                              int64_t tokens, int64_t rows,
                              const __grid_constant__ Rows rq,
                              const __grid_constant__ Rows rk,
                              const __grid_constant__ Positions P, int rot, bool tabvec) {
  constexpr int D = 1 << LOGD, PL = 1 << LOGP, V = D / PL;
  if constexpr (V < 2 || V > 16) {
    return; // not a selectable (d, LOGP) pair; the host never launches it
  } else {
    const int heads = rq.heads + rk.heads;
    const int lane = threadIdx.x & 31, u = lane & (PL - 1);
    int64_t row = int64_t(blockIdx.x) * int(blockDim.x >> LOGP) + (threadIdx.x >> LOGP);
    const bool live = row < rows;
    if (!live)
      row = 0; // keep the loads in bounds; nothing is stored
    const int64_t grp = row / heads;
    const int h = int(row - grp * heads);
    const bool isq = h < rq.heads;
    const int hh = isq ? h : h - rq.heads;
    const Rows &r = isq ? rq : rk;
    const T *x = isq ? q : k, *dy = isq ? dyq : dyk;
    T *dx = isq ? dxq : dxk;
    const W *wt = isq ? wq : wk;
    float *pp = isq ? pq : pk;
    const int i0 = u * V, half = rot / 2;
    const bool lower = i0 < half, upper = i0 >= half && i0 < rot;
    const int jbase = lower ? i0 : i0 - half;             // table column of this lane's first element
    const int pbase = lower ? i0 + half : i0 - half;      // partner columns
    const int psrc = (lane & ~(PL - 1)) + ((u + (V ? half / V : 0)) & (PL - 1));

    int axes = 0; // MRoPE: bit a set when one of this lane's rotary columns reads position axis a
    if (MROPE && (lower || upper))
#pragma unroll
      for (int i = 0; i < V; ++i) {
        const int c = jbase + i;
        axes |= 1 << (int((P.m1 >> c) & 1) + 2 * int((P.m2 >> c) & 1));
      }
    (void)axes;
    float g[V], acc[V];
    rdv_narrow<W, V>(wt + i0, g);
#pragma unroll
    for (int i = 0; i < V; ++i) {
      if constexpr (ZC)
        g[i] = __fadd_rn(1.f, g[i]);
      acc[i] = 0.f;
    }
    // Every lane of the warp runs all 2^gshift steps (its rows may sit in different groups, and the
    // last group may be short): a step past the end recomputes the group's first token and stores
    // nothing, so the shuffles stay warp-uniform.
    const int64_t begin = grp << P.gshift;
    const int steps = 1 << P.gshift;
    for (int s = 0; s < steps; ++s) {
      const bool valid = begin + s < tokens;
      const int64_t t = valid ? begin + s : begin;
      const T *src = x + t * r.s0 + hh * r.s1;
      const T *dsrc = dy + t * r.d0 + hh * r.d1;
      const float inv = inv_in[t * heads + h];
      float xh[V], dn[V], pr[V];
      rdv<T, V>(src + i0, xh);
      rdv<T, V>(dsrc + i0, dn);
      if (lower || upper) {
        float py[V], cs[V], sv[V];
        rdv<T, V>(dsrc + pbase, py);
        if constexpr (MROPE) {
          // table_row<true>'s lookup with the token's three axis positions read once, not per column.
          // table_row's bounds assert, on exactly the axes this lane's columns read.
          int64_t pa[3];
#pragma unroll
          for (int a = 0; a < 3; ++a) {
            pa[a] = P.pos[a * P.ps0 + t * P.ps1];
            assert(!(axes >> a & 1) || uint64_t(pa[a]) < uint64_t(P.rows));
          }
#pragma unroll
          for (int i = 0; i < V; ++i) {
            const int c = jbase + i;
            const int a = int((P.m1 >> c) & 1) + 2 * int((P.m2 >> c) & 1);
            const float *tr = P.table + (a == 0 ? pa[0] : (a == 1 ? pa[1] : pa[2])) * rot;
            cs[i] = tr[c];
            sv[i] = tr[half + c];
          }
        } else if (tabvec) {
          const float *tr = table_row<false>(P, t, 0, rot);
          rdv<float, V>(tr + jbase, cs);
          rdv<float, V>(tr + half + jbase, sv);
        } else {
#pragma unroll
          for (int i = 0; i < V; ++i) {
            const float *tr = table_row<false>(P, t, jbase + i, rot);
            cs[i] = tr[jbase + i];
            sv[i] = tr[half + jbase + i];
          }
        }
#pragma unroll
        for (int i = 0; i < V; ++i) // rotate_t: lower = y1*c + y2*s, upper = y2*c - y1*s
          dn[i] = lower ? __fadd_rn(__fmul_rn(dn[i], cs[i]), __fmul_rn(py[i], sv[i]))
                        : __fsub_rn(__fmul_rn(dn[i], cs[i]), __fmul_rn(py[i], sv[i]));
      }
#pragma unroll
      for (int i = 0; i < V; ++i) {
        xh[i] = __fmul_rn(xh[i], inv);
        pr[i] = __fmul_rn(__fmul_rn(dn[i], g[i]), xh[i]);
      }
      // The upper partner's P values, for this lane's rotary leaves (all lanes take part).
      float pb[V];
#pragma unroll
      for (int i = 0; i < V; ++i)
        pb[i] = __shfl_sync(0xFFFFFFFFu, pr[i], psrc);
      float leaf[V / 2];
#pragma unroll
      for (int m = 0; m < V / 2; ++m) {
        const float a = __fadd_rn(pr[2 * m], pr[2 * m + 1]);
        leaf[m] = lower ? __fadd_rn(a, __fadd_rn(pb[2 * m], pb[2 * m + 1]))
                        : (upper ? 0.f : __fadd_rn(a, 0.f));
      }
#pragma unroll
      for (int span = 1; span < V / 2; span *= 2)
#pragma unroll
        for (int m = 0; m + span < V / 2; m += 2 * span)
          leaf[m] = __fadd_rn(leaf[m], leaf[m + span]);
      float sum = leaf[0];
#pragma unroll
      for (int st = 0; st < LOGP; ++st) {
        auto sh = cuda::device::warp_shuffle_down(
            sum, 1 << st, 0xFFFFFFFFu, cuda::std::integral_constant<int, 32>{});
        if (sh.pred && (lane & ((2 << st) - 1)) == 0)
          sum = __fadd_rn(sum, sh.data);
      }
      const float f = __shfl_sync(0xFFFFFFFFu, __fdiv_rn(sum, float(D)), lane & ~(PL - 1));
      float o[V];
#pragma unroll
      for (int i = 0; i < V; ++i) {
        o[i] = __fmul_rn(inv, __fsub_rn(__fmul_rn(dn[i], g[i]), __fmul_rn(xh[i], f)));
        if (valid)
          acc[i] = __fadd_rn(acc[i], __fmul_rn(dn[i], xh[i]));
      }
      if (live && valid)
        wrv<T, V>(dx + t * r.o0 + hh * r.o1 + i0, o);
    }
    if (pp && live) {
      float *prow = pp + (grp * r.heads + hh) * D + i0;
#pragma unroll
      for (int i = 0; i < V; i += 2)
        *reinterpret_cast<float2 *>(prow + i) = make_float2(acc[i], acc[i + 1]);
    }
  }
}

// Weight-gradient fold: a fixed two-level tree over the n = groups * heads partial rows of one
// operand (q at blockIdx.z == 0, k at 1), shaped by (n, kFoldChunk, kFoldLanes) alone -- never by the
// device, the occupancy or the launch order. A CTA owns 64 columns (32 lanes x float2) of rows
// [b * chunk, min((b + 1) * chunk, n)); row-lane l folds rows b*chunk + l, + kFoldLanes, ... in
// ascending order from +0, then a balanced tree over the kFoldLanes row-lanes (lower lane on the
// left) gives out[b, c]. Stage 1 runs it with chunk = kFoldChunk, stage 2 over the stage-1 rows with
// one chunk, and only stage 2 narrows (once, RN). Plain stores throughout.
constexpr int kFoldLanes = 8;
constexpr int64_t kDwGroupShift = 4;
constexpr int64_t kFoldChunk = 256;

__host__ __device__ inline int64_t fold_chunks(int64_t n, int64_t chunk) {
  return n <= 0 ? 1 : (n + chunk - 1) / chunk;
}

template <class O>
__global__ void qknorm_dw_fold(const float *pq, const float *pk, O *oq, O *ok,
                               int64_t nq, int64_t nk, int64_t chunk, int width) {
  const bool isq = blockIdx.z == 0;
  const float *src = isq ? pq : pk;
  O *dst = isq ? oq : ok;
  const int64_t n = isq ? nq : nk;
  const int64_t b = blockIdx.y;
  if (b >= fold_chunks(n, chunk))
    return;
  const int lane = threadIdx.x & 31, l = threadIdx.x >> 5;
  const int c = blockIdx.x * 64 + 2 * lane;
  const int64_t begin = b * chunk;
  const int64_t stop = begin + chunk < n ? begin + chunk : n;
  float a0 = 0.f, a1 = 0.f;
  if (c < width) {
#pragma unroll 8
    for (int64_t i = begin + l; i < stop; i += kFoldLanes) {
      const float2 v = *reinterpret_cast<const float2 *>(src + i * width + c);
      a0 = __fadd_rn(a0, v.x);
      a1 = __fadd_rn(a1, v.y);
    }
  }
  __shared__ float2 sh[kFoldLanes][32];
  sh[l][lane] = make_float2(a0, a1);
  __syncthreads();
#pragma unroll
  for (int span = 1; span < kFoldLanes; span *= 2) {
    if ((l & (2 * span - 1)) == 0) {
      const float2 o = sh[l + span][lane];
      sh[l][lane] = make_float2(__fadd_rn(sh[l][lane].x, o.x), __fadd_rn(sh[l][lane].y, o.y));
    }
    __syncthreads();
  }
  if (l == 0 && c < width)
    wr2(dst + b * width + c, sh[0][lane].x, sh[0][lane].y);
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
  TORCH_CHECK(false, "rope_sm100 qk norm: head_dim must be 64, 128 or 256");
}

int rpb_for_logu(int logu) {
  int r = int(kThreadsPerBlock >> logu);
  return r < 1 ? 1 : r;
}


struct QkChoice {
  int plan, threads;
};
QkChoice qk_default(int logd, int64_t rows) {
  if (logd < 7)
    return QkChoice{0, 0};
  if (rows <= (int64_t(1) << 10))
    return QkChoice{5, 128};
  return QkChoice{6, 64};
}

// A warp-row plan moves V elements per access; the operand rows must be aligned for that and their
// token/head strides must be whole multiples of V. Otherwise the shape falls back to plan 0, which
// computes the identical bits with 32-bit accesses.
bool vec_rows_ok(const at::Tensor &x, int velems) {
  if (!x.defined() || !x.numel())
    return true;
  const int64_t nb = int64_t(velems) * x.element_size();
  const int64_t chunk = (nb % 16 == 0) ? 16 : ((nb % 8 == 0) ? 8 : 4);
  return x.stride(2) == 1 && x.stride(0) % velems == 0 && x.stride(1) % velems == 0 &&
         (uintptr_t(x.data_ptr()) % uint64_t(chunk)) == 0;
}

} // namespace


namespace {
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
                    int64_t m2, double eps, bool zero_centered, bool want_inv,
                    bool inplace) {
  if (!qk_ok(q_in, k_in, wq_in, wk_in, positions, table, m1, m2))
    return {};
  // In place: the caller's own q and k are the output. It is declined rather than silently done on a
  // copy when an operand would have to be cloned to satisfy the 4-byte pair contract, and it is
  // incompatible with `want_inv` -- the backward needs the UN-normalized q, which in place destroys.
  if (inplace && (want_inv || !pair_aligned(q_in) || !pair_aligned(k_in)))
    return {};
  const c10::cuda::CUDAGuard guard(q_in.device());
  at::Tensor q = pair_operand(q_in), k = pair_operand(k_in);
  at::Tensor wq = pair_operand(wq_in.reshape({1, 1, -1})).view(-1);
  at::Tensor wk = pair_operand(wk_in.reshape({1, 1, -1})).view(-1);
  const int64_t T = q.size(0);
  const int hq = q.size(1), hk = k.size(1);
  at::Tensor qo = inplace ? q : at::empty(q.sizes(), q.options());
  at::Tensor ko = inplace ? k : at::empty(k.sizes(), k.options());
  at::Tensor inv = want_inv ? at::empty({T, int64_t(hq + hk)}, q.options().dtype(at::kFloat))
                            : at::Tensor();
  if (!T)
    return {qo, ko, inv};
  const int logu = log_units(q.size(2));
  const int rpb = rpb_for_logu(logu);
  const int64_t rot = table.has_value() ? table->size(1) : 0;
  Rows rq{q.stride(0), q.stride(1), qo.stride(0), qo.stride(1), 0, 0, hq};
  Rows rk{k.stride(0), k.stride(1), ko.stride(0), ko.stride(1), 0, 0, hk};
  Positions P = rot ? positions_of(*positions, *table, m1, m2)
                    : Positions{nullptr, nullptr, 0, 0, 0, 0, 0, 0};
  const bool mrope = rot && (m1 | m2) != 0 && positions->dim() == 2;
  auto stream = at::cuda::getCurrentCUDAStream();
  const bool w32 = wq.scalar_type() == at::kFloat;
  float eps32 = float(eps);

  // Execution plan. `wr_logp == 0` means plan 0 (grid-of-rows); otherwise the warp-row plan with
  // 2^wr_logp lanes per row. Falling back to plan 0 is always safe: the two produce the same bits.
  const int logd = logu + 1;
  const int64_t rows_total = T * int64_t(hq + hk);
  const QkChoice want = qk_default(logd, rows_total);
  const int plan = want.plan;
  int threads = int(rpb) << logu;
  dim3 grid(T, (hq + hk + rpb - 1) / rpb);
  int wr_logp = 0, wr_ntok = 0;
  // The cos/sin vector load needs the table's base 16-byte aligned; row offsets are p*rot floats with
  // rot % 4 == 0, so they are multiples of 16 bytes on their own.
  const bool tabvec = rot > 0 && (uintptr_t(table->data_ptr()) % 16) == 0;
  bool wr_shfl = true, wr_split = false;
  // In place, a lane must not re-read a block another lane may already have overwritten, so the
  // rotary partner has to come out of registers: the reload plans (3, 4, 7, 8) map to their shuffle
  // twins (1, 2, 5, 6). Same bits either way, so this is a plan substitution, not a refusal.
  const bool xshfl = inplace || plan == 1 || plan == 2 || plan == 5 || plan == 6;
  const bool split = plan >= 5;
  if (plan >= 1 && plan <= 8) {
    const int logp = (plan & 1) ? 5 : 4;
    const int vlen = (logd - logp) >= 1 && (logd - logp) <= 4 ? 1 << (logd - logp) : 0;
    if (vlen && (rot == 0 || (rot / 2) % vlen == 0) && vec_rows_ok(q, vlen) &&
        vec_rows_ok(k, vlen) && vec_rows_ok(qo, vlen) && vec_rows_ok(ko, vlen)) {
      int th = want.threads;
      th = ((th + 31) / 32) * 32;
      th = th < 32 ? 32 : (th > 1024 ? 1024 : th);
      const int rows_per_cta = th >> logp;
      wr_logp = logp;
      wr_shfl = xshfl;
      wr_split = split;
      threads = th;
      // Grid form, from the shape rather than from a literal: the 2-D grid (grid.x = token,
      // grid.y = head group) needs no division, but it rounds the head axis up to whole CTAs, so it is
      // only taken when the rows divide evenly. They do at TP=1 (20 heads, 4 rows a CTA -> grid.y = 5,
      // and the 2-D form measured 160.8 us vs 165.1 us for flat at T=16384) and they do not at the
      // deployed TP=4 (5 heads, 4 rows a CTA -> 1 of 8 row slots dead, and flat measured 45.5 us vs
      // 67.9 us). One condition, both measurements.
      const int flat = (((hq + hk) % rows_per_cta) ? 1 : 0);
      if (flat) {
        wr_ntok = int(T);
        const int64_t rows = T * int64_t(hq + hk);
        grid = dim3(uint32_t((rows + rows_per_cta - 1) / rows_per_cta));
      } else {
        grid = dim3(T, (hq + hk + rows_per_cta - 1) / rows_per_cta);
      }
    }
  }
#define FWD_WR2(TT, WW, ZC, L, LP, XS, SP)                                     \
  qknorm_fwd_wr<TT, WW, ZC, (L) + 1, LP, XS, SP, MR>                           \
      <<<grid, threads, 0, stream>>>(                                          \
          ptr<const TT>(q), ptr<const TT>(k), ptr<TT>(qo), ptr<TT>(ko),        \
          ptr<const WW>(wq), ptr<const WW>(wk), ptr<float>(inv), rq, rk, P,    \
          int(rot), eps32, wr_ntok, tabvec)
#define FWD_WR1(TT, WW, ZC, L, LP, XS)                                         \
  do {                                                                         \
    if (wr_split)                                                              \
      FWD_WR2(TT, WW, ZC, L, LP, XS, true);                                    \
    else                                                                       \
      FWD_WR2(TT, WW, ZC, L, LP, XS, false);                                   \
  } while (0)
#define FWD_WR(TT, WW, ZC, L, LP)                                              \
  do {                                                                         \
    if (wr_shfl)                                                               \
      FWD_WR1(TT, WW, ZC, L, LP, true);                                        \
    else                                                                       \
      FWD_WR1(TT, WW, ZC, L, LP, false);                                       \
  } while (0)
#define FWD(TT, WW, ZC, L)                                                     \
  do {                                                                         \
    if (wr_logp == 5)                                                          \
      FWD_WR(TT, WW, ZC, L, 5);                                                \
    else if (wr_logp == 4)                                                     \
      FWD_WR(TT, WW, ZC, L, 4);                                                \
    else                                                                       \
      qknorm_fwd<TT, WW, ZC, L, MR><<<grid, threads, 0, stream>>>(             \
          ptr<const TT>(q), ptr<const TT>(k), ptr<TT>(qo), ptr<TT>(ko),        \
          ptr<const WW>(wq), ptr<const WW>(wk), ptr<float>(inv), rq, rk, P,    \
          int(rot), eps32);                                                    \
  } while (0)
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
#undef FWD_WR
#undef FWD_WR1
#undef FWD_WR2
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
  // The token-group size is part of the weight-gradient recipe, not a tuning knob.
  TORCH_CHECK(gshift == kDwGroupShift,
              "rope_sm100 qk backward requires gshift=4 (fixed 16-token weight-gradient groups)");
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
  // Every element is stored exactly once by qknorm_bwd: no zero fill.
  at::Tensor pq = want_w ? at::empty({groups, hq, d}, fopt) : at::Tensor();
  at::Tensor pk = want_w ? at::empty({groups, hk, d}, fopt) : at::Tensor();
  auto stream = at::cuda::getCurrentCUDAStream();
  if (T) {
    const int logu = log_units(d);
    const int rpb = rpb_for_logu(logu);
    const int64_t rot = table.has_value() ? table->size(1) : 0;
    Rows rq{q.stride(0), q.stride(1), dxq.stride(0), dxq.stride(1),
            dyq.stride(0), dyq.stride(1), hq};
    Rows rk{k.stride(0), k.stride(1), dxk.stride(0), dxk.stride(1),
            dyk.stride(0), dyk.stride(1), hk};
    Positions P = rot ? positions_of(*positions, *table, m1, m2)
                      : Positions{nullptr, nullptr, 0, 0, 0, 0, 0, 0};
    P.gshift = int(gshift);
    const bool mrope = rot && (m1 | m2) != 0 && positions->dim() == 2;
    dim3 grid(uint32_t(groups), (hq + hk + rpb - 1) / rpb);
    const bool w32 = wq.scalar_type() == at::kFloat;
    // Execution plan: the warp-row twin (half a warp a row) when every operand row is V-aligned and
    // the rotary half is whole lanes, else grid-of-rows. Same bits either way. On B200, half a warp
    // a row ran 2-3x faster than grid-of-rows at d in {64, 128, 256} and beat a whole warp a row at
    // training token counts (it trails it by ~15% only at a few thousand rows).
    const int logd = logu + 1;
    constexpr int kBwdLogp = 4;
    const int v = 1 << (logd - kBwdLogp);
    const int bw_logp = ((rot == 0 || (rot / 2) % v == 0) && vec_rows_ok(q, v) && vec_rows_ok(k, v) &&
                         vec_rows_ok(dyq, v) && vec_rows_ok(dyk, v) && vec_rows_ok(dxq, v) &&
                         vec_rows_ok(dxk, v))
                            ? kBwdLogp
                            : 0;
    const int64_t wr_rows = groups * int64_t(hq + hk);
    const int wr_threads = 128;
    const bool tabvec = rot > 0 && (uintptr_t(table->data_ptr()) % 16) == 0;
#define BWD_WR(TT, WW, ZC, L, LP)                                              \
  qknorm_bwd_wr<TT, WW, ZC, (L) + 1, LP, MR>                                   \
      <<<uint32_t((wr_rows + (wr_threads >> LP) - 1) / (wr_threads >> LP)),     \
         wr_threads, 0, stream>>>(                                             \
          ptr<const TT>(dyq), ptr<const TT>(dyk), ptr<const TT>(q),            \
          ptr<const TT>(k), ptr<TT>(dxq), ptr<TT>(dxk), ptr<const WW>(wq),     \
          ptr<const WW>(wk), ptr<const float>(inv), ptr<float>(pq),            \
          ptr<float>(pk), T, wr_rows, rq, rk, P, int(rot), tabvec)
#define FWD(TT, WW, ZC, L)                                                     \
  do {                                                                         \
    if (bw_logp)                                                               \
      BWD_WR(TT, WW, ZC, L, kBwdLogp);                                         \
    else                                                                       \
      qknorm_bwd<TT, WW, ZC, L, MR><<<grid, rpb << L, 0, stream>>>(            \
          ptr<const TT>(dyq), ptr<const TT>(dyk), ptr<const TT>(q),            \
          ptr<const TT>(k), ptr<TT>(dxq), ptr<TT>(dxk), ptr<const WW>(wq),     \
          ptr<const WW>(wk), ptr<const float>(inv), ptr<float>(pq),            \
          ptr<float>(pk), T, rq, rk, P, int(rot));                             \
  } while (0)
    BY_T();
#undef BWD_WR
#undef FWD
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  at::Tensor gwq, gwk;
  if (want_w) {
    const int64_t nq = groups * hq, nk = groups * hk;
    const int64_t cq = fold_chunks(nq, kFoldChunk), ck = fold_chunks(nk, kFoldChunk);
    at::Tensor sq = at::empty({cq, d}, fopt), sk = at::empty({ck, d}, fopt);
    gwq = at::empty({d}, wq_in.options());
    gwk = at::empty({d}, wk_in.options());
    const int64_t chunk2 = std::max(cq, ck);
    TORCH_CHECK(chunk2 <= 65535, "rope_sm100 qk backward: too many weight-gradient partial rows");
    const uint32_t windows = uint32_t((d + 63) / 64);
    qknorm_dw_fold<float><<<dim3(windows, uint32_t(std::max(cq, ck)), 2), 32 * kFoldLanes, 0, stream>>>(
        ptr<float>(pq), ptr<float>(pk), ptr<float>(sq), ptr<float>(sk), nq, nk, kFoldChunk, int(d));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
#define FOLD(WW)                                                               \
  qknorm_dw_fold<WW><<<dim3(windows, 1, 2), 32 * kFoldLanes, 0, stream>>>(     \
      ptr<float>(sq), ptr<float>(sk), ptr<WW>(gwq), ptr<WW>(gwk), cq, ck,      \
      chunk2, int(d))
    if (wq_in.scalar_type() == at::kBFloat16)
      FOLD(__nv_bfloat16);
    else if (wq_in.scalar_type() == at::kHalf)
      FOLD(__half);
    else
      FOLD(float);
#undef FOLD
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {dxq, dxk, gwq, gwk};
}
#undef BY_T
#undef BY_W
#undef BY_LOGU
#undef BY_MR
