#pragma once

#include <cmath>
#include <cstdint>

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#include <cute/atom/copy_traits_sm100.hpp>
#include <cutlass/arch/barrier.h>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/layout/mega_gate.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/utils.cuh>

// IsoExec: the gate's scoring and top-k are IsoExec's router selection program, so every output bit equals
// `router_sm100.route()`. The two blocks between the BEGIN/END markers are verbatim copies (a CPU test compares
// them byte for byte with their sources):
//   * `namespace rt100 { ... }` of `isoexec/ops/moe/router_sm100/csrc/select.cuh` (score, key, top-k with its tie
//     rule, unbiased weights, routed scale, shared slots, ascending-expert-id slots);
//   * `segment_tree` of `isoexec/ops/moe/router_sm100/csrc/router.cu` (the logits program's segment tree).
// Compiled without fast math, with --fmad=false --ftz=false --prec-div=true --prec-sqrt=true (the router extension's
// arithmetic profile), so the selection's library calls and divides round as in the router extension.

// ---- BEGIN verbatim isoexec/ops/moe/router_sm100/csrc/select.cuh ----
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

//: THE renormalisation fold: halffold over the K selected scores in selection-rank order.
template <int K, int KP> __device__ __forceinline__ float halffold(const float (&v)[KP]) {
  float a[KP];
#pragma unroll
  for (int j = 0; j < KP; ++j)
    a[j] = v[j];
  int n = K;
#pragma unroll
  for (int it = 0; it < 5; ++it) {
    if (n > 1) {
      const int m = (n + 1) / 2;
#pragma unroll
      for (int j = 0; j < KP; ++j) {
        // j + m < n with n, m compile-time after unrolling (K is a template constant).
        if (j + m < n && j + m < KP)
          a[j] = __fadd_rn(a[j], a[j + m < KP ? j + m : 0]);
      }
      n = m;
    }
  }
  return a[0];
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
    const float denom = __fadd_rn(halffold<K, K>(sw[t]), 1e-20f);
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
// ---- END verbatim isoexec/ops/moe/router_sm100/csrc/select.cuh ----

namespace rt100 {

// ---- BEGIN verbatim isoexec/ops/moe/router_sm100/csrc/router.cu segment_tree ----
//: THE segment tree: balanced, lower-index subtrees on the left, one __fadd_rn a node.
template <int S> __device__ __forceinline__ float segment_tree(const float (&p)[S]) {
  float a[S];
#pragma unroll
  for (int i = 0; i < S; ++i)
    a[i] = p[i];
#pragma unroll
  for (int width = 1; width < S; width *= 2)
#pragma unroll
    for (int i = 0; i + width < S; i += 2 * width)
      a[i] = __fadd_rn(a[i], a[i + width]);
  return a[0];
}
// ---- END verbatim isoexec/ops/moe/router_sm100/csrc/router.cu segment_tree ----

} // namespace rt100

namespace deep_gemm::epilogue::mega_gate {

using namespace layout::mega_gate;

// Drain a token block's segment accumulators: every (expert, token) element of this CTA's `kNumSegmentsPerCta`
// segments is loaded from its own accumulator (`kAccumCols` columns apart) and folded by the segment tree (the
// subtree this split owns), then stored as the split's FP32 partial logit. The lane/column mapping is upstream's.
template <uint32_t kNumSegmentsPerCta, uint32_t kAccumCols, uint32_t kScoreTokenStride,
          uint32_t UMMA_M, uint32_t kNumMmaCtas, uint32_t kNumGateWarps>
CUTLASS_DEVICE void sm100_store_tmem_partials_to_global(float* partials, const uint32_t& tmem_base_addr,
                                                        const uint32_t& effective_umma_n,
                                                        const uint32_t& expert_base_idx,
                                                        const uint32_t& mma_cta_rank,
                                                        const uint32_t& gate_warp_idx) {
    // NOTES: with `UMMA_M == 128` on a CTA pair, the 64 experts of a CTA are spread over the two lane halves,
    //        so warps 0/1 and 2/3 of a warpgroup see the same experts for the two token halves
    constexpr uint32_t kNumTmemFragmentRows = 8;
    constexpr bool kPartitionedTokenPaths = UMMA_M == 128 and kNumMmaCtas == 2;
    constexpr uint32_t kNumExpertsPerCta = UMMA_M / kNumMmaCtas;
    const auto warp_idx_in_wg = gate_warp_idx % 4;
    const auto wg_idx = gate_warp_idx / 4;
    constexpr uint32_t kTokenStep = kNumGateWarps / 4 * kNumTmemFragmentRows;
    const auto num_token_cols = kPartitionedTokenPaths ? effective_umma_n / 2 : effective_umma_n;
    const auto token_base_idx = kPartitionedTokenPaths ? warp_idx_in_wg / 2 * num_token_cols : 0u;
    const auto expert_atom_idx = kPartitionedTokenPaths ? warp_idx_in_wg % 2 : warp_idx_in_wg;
    const auto global_expert_idx = expert_base_idx + mma_cta_rank * kNumExpertsPerCta +
                                   expert_atom_idx * 32 + ptx::get_lane_idx();

    // NOTES: tensor memory addresses are simplified, as the hardware will ignore the warp index bits
    for (auto token_idx = wg_idx * kNumTmemFragmentRows; token_idx < num_token_cols; token_idx += kTokenStep) {
        uint32_t values[kNumSegmentsPerCta][kNumTmemFragmentRows];
        #pragma unroll
        for (uint32_t segment_idx = 0; segment_idx < kNumSegmentsPerCta; ++ segment_idx)
            ptx::tmem_load_32dp32b<kNumTmemFragmentRows>(tmem_base_addr + segment_idx * kAccumCols + token_idx, values[segment_idx]);
        cutlass::arch::fence_view_async_tmem_load();
        const auto row_base_ptr = partials + static_cast<uint64_t>(token_base_idx + token_idx) * kScoreTokenStride +
                                  global_expert_idx;
        #pragma unroll
        for (uint32_t row_idx = 0; row_idx < kNumTmemFragmentRows; ++ row_idx) {
            float roots[kNumSegmentsPerCta];
            #pragma unroll
            for (uint32_t segment_idx = 0; segment_idx < kNumSegmentsPerCta; ++ segment_idx)
                roots[segment_idx] = __uint_as_float(values[segment_idx][row_idx]);
            row_base_ptr[row_idx * kScoreTokenStride] = rt100::segment_tree<static_cast<int>(kNumSegmentsPerCta)>(roots);
        }
    }
}

// One token on one warp, after every split's partials are visible: the logit of each expert is the segment tree
// over the split partials (the tree's upper levels, lower split index on the left), then the router's selection.
// This is `router_sm100`'s `select_tree_kernel` body with the partials read from the gate scratch: lane l owns experts
// [EPL * l, EPL * l + EPL) and the same values reach the same `segment_tree` and `select_warp`. Only the loads are
// grouped differently (4 experts of every split at a time, `rt100::load_lane`'s vector rule) to bound the registers
// the split partials hold; a load moves no bit.
template <int kMode, int kNumTopk, int kNumExpertsPerLane, uint32_t kNumSplitK, uint32_t kSplitStride>
CUTLASS_DEVICE void select_token(const float* token_partials, const int64_t& token_idx, const rt100::SelectArgs& args) {
    constexpr int kChunk = kNumExpertsPerLane % 4 == 0 ? 4 : 1;
    const int lane = static_cast<int>(ptx::get_lane_idx());
    const int e0 = lane * kNumExpertsPerLane;
    const int n = max(0, min(kNumExpertsPerLane, args.E - e0));
    float v[kNumExpertsPerLane];
    #pragma unroll
    for (int u0 = 0; u0 < kNumExpertsPerLane; u0 += kChunk) {
        float q[kChunk][kNumSplitK];
        #pragma unroll
        for (uint32_t split_idx = 0; split_idx < kNumSplitK; ++ split_idx) {
            const float* src = token_partials + split_idx * kSplitStride + e0 + u0;
            bool vectorized = false;
            if constexpr (kChunk == 4) {
                if (n == kNumExpertsPerLane and (reinterpret_cast<uintptr_t>(src) & 15u) == 0) {
                    const float4 f = *reinterpret_cast<const float4*>(src);
                    q[0][split_idx] = f.x, q[1][split_idx] = f.y, q[2][split_idx] = f.z, q[3][split_idx] = f.w;
                    vectorized = true;
                }
            }
            if (not vectorized) {
                #pragma unroll
                for (int w = 0; w < kChunk; ++ w)
                    q[w][split_idx] = u0 + w < n ? src[w] : 0.f;
            }
        }
        #pragma unroll
        for (int w = 0; w < kChunk; ++ w)
            v[u0 + w] = rt100::segment_tree<static_cast<int>(kNumSplitK)>(q[w]);
    }
    rt100::select_warp<kMode, kNumTopk, kNumExpertsPerLane>(v, true, token_idx, args);
}

} // namespace deep_gemm::epilogue::mega_gate
