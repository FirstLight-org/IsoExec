// SPDX-License-Identifier: Apache-2.0
//
// THE SELECTION PROGRAM, shared verbatim by the select-only kernel (caller logits in global memory)
// and the fused kernel's epilogue (its own logits in shared memory). One definition is what makes
// `fused_vs_select_only` a structural identity and not a coincidence.
//
// Everything below is either an exact reducer (a total-order selection, an exact integer rank, a
// max) or a declared rounding site:
//
//   score      sigmoid(x) = 1 / (1 + expf(-x))                     (routing A; moe_sm100's spelling)
//              sqrtsoftplus(x) = sqrtf(x > 20 ? x : log1pf(expf(x)))  (routing B; DeepGEMM's)
//              CUDA expf / log1pf / IEEE sqrtf and IEEE divide, never __expf or a fast divide.
//   key        r = score + bias[e] (one __fadd_rn; routing B without a bias: r = score), NaN -> -inf.
//   order      (r desc, expert id asc); a masked expert is EMPTY (id sentinel) and loses to every
//              real candidate, including a -inf one -- the bypass is an integer test, never a
//              poisoned score.
//   group      (routing A, topk_group < n_group) G_g = top-2 SUM of r in group g: the two largest
//              keys of the group (a max and a second max, exact), then one __fadd_rn of two operands
//              (order-free). Groups ranked by (G desc, g asc); the first topk_group survive.
//   weights    s_j = score(logit[e_j]) for the selected experts in SELECTION-RANK order j;
//              denom = selected_sum(s) + 1e-20f; w_j = (s_j / denom) * scale, each op one IEEE
//              round-to-nearest. selected_sum: the schedule policy's rule 2 -- one segment per selected
//              expert, a chain in ASCENDING EXPERT ID from +0 (moe_sm100's ``selected_sum``).
//   slots      emitted in ASCENDING GLOBAL EXPERT ID (exact integer rank of <= 16 distinct ids);
//              routing B's shared slots E..E+S-1 follow with weight 1.0; a masked token writes
//              id -1 and weight 0 in every slot.
//
// Work split (free, not numerical): ONE WARP PER TOKEN. Lane l owns the EPL = ceil(E/32)
// consecutive experts [EPL*l, EPL*l + EPL), so a group of gs experts is gs/EPL whole lanes. The
// selection is K exact argmax rounds: each lane offers its best remaining (key, id) -- its own
// experts scanned in ascending id with a strict `>`, so a lane-local tie keeps the lower id --
// then `redux.sync.max` over the ORDERED-INTEGER image of the key and `redux.sync.min` over the ids
// that hold it: the winner is the (key desc, id asc) maximum, exactly. The winner's lane removes it
// by an integer test. Which lane owns which expert cannot move a result: every step is an exact
// reducer.
#pragma once
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace rt100 {

enum : int { kSigmoid = 0, kSqrtSoftplus = 1 };

constexpr unsigned kFull = 0xffffffffu;

#ifndef RT_REDUX
#define RT_REDUX 1   //: 1 = redux.sync rounds (shipped); 0 = shuffle butterfly (same result, measured slower)
#endif

template <int MODE> __device__ __forceinline__ float score(float x) {
  if constexpr (MODE == kSigmoid) {
    return 1.f / (1.f + expf(-x));
  } else {
    const float sp = x > 20.f ? x : log1pf(expf(x));
    return sqrtf(sp);
  }
}

__device__ __forceinline__ float canon(float r) { return r != r ? -INFINITY : r; }

//: the ORDERED-INTEGER image of a canonical (non-NaN) key: u(a) > u(b) iff a > b, with -0.0 folded
//: into +0.0 first (the float compare treats them as equal, so must the image). Every real key maps
//: to >= u(-inf) = 0x007fffff, so 0 is free to mean EMPTY -- a masked or already-selected expert.
__device__ __forceinline__ uint32_t ordered_key(float r) {
  const uint32_t b = __float_as_uint(__fadd_rn(r, 0.f));
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

//: THE renormalisation sum. Schedule policy rule 2 (``isoexec/ops/schedule_policy.py`` ``selected()``): a sum over
//: data-selected coordinates is ONE SEGMENT PER COORDINATE, chained in ASCENDING EXPERT ID from +0. ``v[j]`` is
//: the score of expert ``ids[j]`` (j = selection rank); the chain visits the scores by the rank of their id, so
//: the sum does not depend on the selection order or on which rank holds which expert.
template <int K>
__device__ __forceinline__ float selected_sum(const float (&v)[K], const uint32_t (&ids)[K]) {
  int rank[K];   // rank of each selected id among the K (ids are distinct): computed once, K * K compares
#pragma unroll
  for (int j = 0; j < K; ++j) {
    int slot = 0;
#pragma unroll
    for (int l = 0; l < K; ++l)
      slot += ids[l] < ids[j];
    rank[j] = slot;
  }
  float acc = 0.f;
#pragma unroll
  for (int s = 0; s < K; ++s) {
    float term = 0.f;
#pragma unroll
    for (int j = 0; j < K; ++j)
      term = rank[j] == s ? v[j] : term;
    acc = __fadd_rn(acc, term);
  }
  return acc;
}

struct SelectArgs {
  const float *bias; //: [E] fp32, or nullptr (routing B only)
  const uint8_t *mask; //: [T] bool, or nullptr (routing B only): false -> id -1, weight 0
  int E, n_group, topk_group, S;
  float scale;
  int *out_idx;   //: [T, K+S] int32
  float *out_w;   //: [T, K+S] fp32
};

//: load this lane's EPL consecutive entries of a row (global or shared), vectorised when aligned.
template <int EPL>
__device__ __forceinline__ void load_lane(const float *p, int n, float (&v)[EPL]) {
  if ((EPL % 4 == 0) && n == EPL && ((reinterpret_cast<uintptr_t>(p) & 15u) == 0)) {
#pragma unroll
    for (int u = 0; u < EPL; u += 4) {
      const float4 q = *reinterpret_cast<const float4 *>(p + u);
      v[u] = q.x;
      v[u + 1] = q.y;
      v[u + 2] = q.z;
      v[u + 3] = q.w;
    }
  } else {
#pragma unroll
    for (int u = 0; u < EPL; ++u)
      v[u] = u < n ? p[u] : 0.f;
  }
}

//: TPW tokens' selections on one warp, interleaved (TPW independent instruction streams for the
//: scheduler -- ILP only; each token's program is exactly the TPW = 1 program). `v[t]` is this
//: lane's EPL logits of token t (experts EPL*lane + u). EVERY lane of the warp must call this (the
//: reductions are full-warp); `valid[t] == false` marks a padding token, which writes nothing.
//: THE SELECTION CORE: winners in selection-rank order and their unbiased scores, on every lane.
//: The forward table and the backward both start here, so the backward's selection IS the forward's.
template <int MODE, int K, int EPL, int TPW>
__device__ __forceinline__ void select_core(const float (&v)[TPW][EPL], const bool (&valid)[TPW],
                                            const SelectArgs &a, uint32_t (&win)[TPW][K],
                                            float (&sw)[TPW][K]) {
  const int lane = int(threadIdx.x & 31u);
  const int E = a.E;
  const int e0 = lane * EPL;
  const int nl = max(0, min(EPL, E - e0));
  float b[EPL];
  if (a.bias) {
    if ((EPL % 4 == 0) && nl == EPL) {
#pragma unroll
      for (int u = 0; u < EPL; u += 4) {
        const float4 q = __ldg(reinterpret_cast<const float4 *>(a.bias + e0 + u));
        b[u] = q.x;
        b[u + 1] = q.y;
        b[u + 2] = q.z;
        b[u + 3] = q.w;
      }
    } else {
#pragma unroll
      for (int u = 0; u < EPL; ++u)
        b[u] = u < nl ? __ldg(a.bias + e0 + u) : 0.f;
    }
  }
  float sc[TPW][EPL], r[TPW][EPL];
  uint32_t key[TPW][EPL];
#pragma unroll
  for (int t = 0; t < TPW; ++t) {
    const int n = valid[t] ? nl : 0;
#pragma unroll
    for (int u = 0; u < EPL; ++u) {
      sc[t][u] = score<MODE>(v[t][u]);
      r[t][u] = canon(a.bias ? __fadd_rn(sc[t][u], b[u]) : sc[t][u]);
      key[t][u] = u < n ? ordered_key(r[t][u]) : 0u;
    }
  }
  // ---- group-limited selection (routing A, topk_group < n_group) ----
  if (a.topk_group < a.n_group) {
    const int lg = (E / a.n_group) / EPL;   // lanes per group, a power of two (host-checked)
    float f1[TPW], f2[TPW], G[TPW];
#pragma unroll
    for (int t = 0; t < TPW; ++t) {
      const int n = valid[t] ? nl : 0;
      f1[t] = -INFINITY;
      f2[t] = -INFINITY;
#pragma unroll
      for (int u = 0; u < EPL; ++u) {
        if (u < n) {
          const float n2 = fmaxf(f2[t], fminf(f1[t], r[t][u]));
          f1[t] = fmaxf(f1[t], r[t][u]);
          f2[t] = n2;
        }
      }
    }
    for (int m = 1; m < lg; m <<= 1) {
#pragma unroll
      for (int t = 0; t < TPW; ++t) {
        const float o1 = __shfl_xor_sync(kFull, f1[t], m);
        const float o2 = __shfl_xor_sync(kFull, f2[t], m);
        const float n2 = fmaxf(fminf(f1[t], o1), fmaxf(f2[t], o2));
        f1[t] = fmaxf(f1[t], o1);
        f2[t] = n2;
      }
    }
    const int g = lane / lg;
    int rank[TPW];
#pragma unroll
    for (int t = 0; t < TPW; ++t) {
      G[t] = canon(__fadd_rn(f1[t], f2[t]));
      rank[t] = 0;
    }
    for (int h = 0; h < a.n_group; ++h) {
#pragma unroll
      for (int t = 0; t < TPW; ++t) {
        const float gh = __shfl_sync(kFull, G[t], h * lg);
        rank[t] += (gh > G[t]) || (gh == G[t] && h < g);
      }
    }
#pragma unroll
    for (int t = 0; t < TPW; ++t)
      if (rank[t] >= a.topk_group) {
#pragma unroll
        for (int u = 0; u < EPL; ++u)
          key[t][u] = 0u;
      }
  }
  // ---- K exact argmax rounds; the winner's unbiased score travels with it ----
#pragma unroll
  for (int j = 0; j < K; ++j) {
    uint32_t bu[TPW], best[TPW], cand[TPW];
    int bi[TPW];
    float bs[TPW];
#pragma unroll
    for (int t = 0; t < TPW; ++t) {
      bu[t] = 0u;
      bi[t] = 0;
      bs[t] = 0.f;
#pragma unroll
      for (int u = 0; u < EPL; ++u) {
        if (key[t][u] > bu[t]) {     // ascending u, strict: a lane-local tie keeps the lower id
          bu[t] = key[t][u];
          bi[t] = u;
          bs[t] = sc[t][u];
        }
      }
    }
#if RT_REDUX
#pragma unroll
    for (int t = 0; t < TPW; ++t)
      best[t] = __reduce_max_sync(kFull, bu[t]);
#pragma unroll
    for (int t = 0; t < TPW; ++t)
      cand[t] = (bu[t] == best[t] && bu[t] != 0u) ? uint32_t(e0 + bi[t]) : 0xffffffffu;
#pragma unroll
    for (int t = 0; t < TPW; ++t)
      win[t][j] = __reduce_min_sync(kFull, cand[t]);
#else
    // Butterfly over the 64-bit total-order image (key << 32 | ~id): max is (key desc, id asc),
    // exactly the two-redux result, and every lane ends holding it.
#pragma unroll
    for (int t = 0; t < TPW; ++t) {
      (void)best;
      (void)cand;
      uint64_t c = bu[t] ? ((uint64_t(bu[t]) << 32) | uint64_t(~uint32_t(e0 + bi[t]))) : 0ull;
#pragma unroll
      for (int m = 16; m >= 1; m >>= 1) {
        const uint64_t o = __shfl_xor_sync(kFull, c, m);
        c = o > c ? o : c;
      }
      win[t][j] = c ? ~uint32_t(c) : 0xffffffffu;
    }
#endif
#pragma unroll
    for (int t = 0; t < TPW; ++t) {
      const uint32_t w = win[t][j];
      sw[t][j] = __shfl_sync(kFull, bs[t], int((w < uint32_t(32 * EPL) ? w : 0u) / EPL));
#pragma unroll
      for (int u = 0; u < EPL; ++u)
        if (uint32_t(e0 + u) == w)
          key[t][u] = 0u;
    }
  }
}

template <int MODE, int K, int EPL, int TPW>
__device__ __forceinline__ void select_warp_n(const float (&v)[TPW][EPL], const bool (&valid)[TPW],
                                              const int64_t (&tok)[TPW], const SelectArgs &a) {
  const int lane = int(threadIdx.x & 31u);
  const int E = a.E;
  uint32_t win[TPW][K];
  float sw[TPW][K];
  select_core<MODE, K, EPL, TPW>(v, valid, a, win, sw);
  // ---- weights in selection-rank order (lane j writes rank j), slots in ascending expert id ----
  const int KS = K + a.S;
#pragma unroll
  for (int t = 0; t < TPW; ++t) {
    if (!valid[t])
      continue;
    const float denom = __fadd_rn(selected_sum<K>(sw[t], win[t]), 1e-20f);
    const bool active = a.mask ? (a.mask[tok[t]] != 0) : true;
    if (lane < K) {
      uint32_t mine = 0u;
      float smine = 0.f;
#pragma unroll
      for (int j = 0; j < K; ++j) {
        mine = lane == j ? win[t][j] : mine;
        smine = lane == j ? sw[t][j] : smine;
      }
      int slot = 0;
#pragma unroll
      for (int l = 0; l < K; ++l)
        slot += win[t][l] < mine;
      const float w = __fmul_rn(smine / denom, a.scale);
      a.out_idx[tok[t] * KS + slot] = active ? int(mine) : -1;
      a.out_w[tok[t] * KS + slot] = active ? w : 0.f;
    } else if (lane < KS) {
      a.out_idx[tok[t] * KS + lane] = active ? E + (lane - K) : -1;
      a.out_w[tok[t] * KS + lane] = active ? 1.f : 0.f;
    }
  }
}

//: one token's selection on one warp (the TPW = 1 program).
template <int MODE, int K, int EPL>
__device__ __forceinline__ void select_warp(const float (&v)[EPL], bool valid, int64_t tok,
                                            const SelectArgs &a) {
  const float(&vv)[1][EPL] = *reinterpret_cast<const float(*)[1][EPL]>(&v);
  const bool vd[1] = {valid};
  const int64_t tk[1] = {tok};
  select_warp_n<MODE, K, EPL, 1>(vv, vd, tk, a);
}

} // namespace rt100
