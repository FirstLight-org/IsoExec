// SPDX-License-Identifier: Apache-2.0
// Routing, canonical dispatch planning, and ordered combination.
// Score accumulation follows ascending hidden-index order; placement ranks
// routes by flat (token, slot) index rather than atomic arrival order.

#include "numerics.cuh"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>
#include <assert.h>
#include <cstdlib>
#include <cub/block/block_scan.cuh>
#include <string>
#include <type_traits>

namespace moe100 {
namespace {

constexpr int kRouteBlock = 1024;   // routes per plan block; also the rank granularity

// ------------------------------------------------------------------ route_score ---
//
// Each (token, expert) owns one ascending FP32 chain. BF16 products are exact
// in FP32, so explicit __fmaf_rn preserves the declared accumulation.
// Tiling and vector loads change operand scheduling, not chain order.
// Double buffering overlaps global loads with the previous chunk.
template <class T, int CH, int TR, int TC, int TM, int TN, int SETS = 1, int MB = 6, int UNR = 2>
__global__ __launch_bounds__(TR * TC, MB) void score_tile_kernel(const T *__restrict__ x,
                                                                const T *__restrict__ w,
                                                                float *__restrict__ logits,
                                                                int T_, int E, int H,
                                                                int64_t x_stride,
                                                                int64_t w_stride) {
  constexpr int NT = TR * TC;
  constexpr int BM = TR * TM, BN = TC * TN;
  constexpr int LD = CH + 4;              // Stagger float4 rows across 16-byte bank groups.
  constexpr int kVec = 16 / int(sizeof(T));  // 16-bit elements in one 16-byte global load
  constexpr int kRowLoads = CH / kVec;    // 16-byte loads that cover one staged row's chunk
  constexpr int kXItems = (BM * kRowLoads + NT - 1) / NT;
  constexpr int kWItems = (BN * kRowLoads + NT - 1) / NT;
  static_assert(CH % kVec == 0 && CH % 4 == 0, "a chunk is whole 16-byte loads and float4 reads");
  static_assert((LD * 4) % 16 == 0, "a staged row must start 16-byte aligned");
  __shared__ __align__(16) float xs[2][BM][LD];
  __shared__ __align__(16) float ws[2][BN][LD];

  const int t0 = blockIdx.y * BM, e0 = blockIdx.x * BN;
  const int tid = int(threadIdx.x);
  const int tr = tid / TC, ec = tid % TC;
  float acc[TM][TN];
#pragma unroll
  for (int a = 0; a < TM; ++a)
#pragma unroll
    for (int b = 0; b < TN; ++b)
      acc[a][b] = 0.f;

  // The vector staging path needs the row stride and both bases to be whole 16-byte loads and H
  // to be a whole number of chunks. Otherwise the scalar path stages straight to shared memory,
  // which is slower and correct; both zero-fill past the tensor, exactly as the first shape did.
  const bool vec = (H % CH == 0) && (x_stride % kVec == 0) && (w_stride % kVec == 0) &&
                   (reinterpret_cast<uintptr_t>(x) % 16 == 0) &&
                   (reinterpret_cast<uintptr_t>(w) % 16 == 0);
  // Two named register sets extend prefetch distance without dynamic register indexing.
  uint4 xreg[SETS][kXItems], wreg[SETS][kWItems];

  auto fetch = [&](int set, int h0) {
#pragma unroll
    for (int n = 0; n < kXItems; ++n) {
      const int i = tid + n * NT;
      xreg[set][n] = make_uint4(0u, 0u, 0u, 0u);
      if (i < BM * kRowLoads) {
        const int r = i / kRowLoads, q = i - r * kRowLoads;
        if (t0 + r < T_)
          xreg[set][n] = *reinterpret_cast<const uint4 *>(x + int64_t(t0 + r) * x_stride + h0 +
                                                          q * kVec);
      }
    }
#pragma unroll
    for (int n = 0; n < kWItems; ++n) {
      const int i = tid + n * NT;
      wreg[set][n] = make_uint4(0u, 0u, 0u, 0u);
      if (i < BN * kRowLoads) {
        const int r = i / kRowLoads, q = i - r * kRowLoads;
        if (e0 + r < E)
          wreg[set][n] = *reinterpret_cast<const uint4 *>(w + int64_t(e0 + r) * w_stride + h0 +
                                                          q * kVec);
      }
    }
  };

  //: registers -> fp32 shared. Two float4 stores per 16-byte load; the +4 pad staggers the rows
  //: so a warp's eight destinations are eight distinct 16-byte bank groups.
  auto place = [&](int set, int buf) {
#pragma unroll
    for (int n = 0; n < kXItems; ++n) {
      const int i = tid + n * NT;
      if (i < BM * kRowLoads) {
        const int r = i / kRowLoads, q = i - r * kRowLoads;
        const T *src = reinterpret_cast<const T *>(&xreg[set][n]);
        float *dst = &xs[buf][r][q * kVec];
#pragma unroll
        for (int v = 0; v < kVec; ++v)
          dst[v] = upcast<T>(src[v]);
      }
    }
#pragma unroll
    for (int n = 0; n < kWItems; ++n) {
      const int i = tid + n * NT;
      if (i < BN * kRowLoads) {
        const int r = i / kRowLoads, q = i - r * kRowLoads;
        const T *src = reinterpret_cast<const T *>(&wreg[set][n]);
        float *dst = &ws[buf][r][q * kVec];
#pragma unroll
        for (int v = 0; v < kVec; ++v)
          dst[v] = upcast<T>(src[v]);
      }
    }
  };

  //: the general path: one predicated element at a time, straight to shared memory.
  auto stage_scalar = [&](int buf, int h0) {
    for (int i = tid; i < CH * BM; i += NT) {
      const int r = i / CH, c = i - r * CH, h = h0 + c;
      xs[buf][r][c] =
          (t0 + r < T_ && h < H) ? upcast<T>(x[int64_t(t0 + r) * x_stride + h]) : 0.f;
    }
    for (int i = tid; i < CH * BN; i += NT) {
      const int r = i / CH, c = i - r * CH, h = h0 + c;
      ws[buf][r][c] =
          (e0 + r < E && h < H) ? upcast<T>(w[int64_t(e0 + r) * w_stride + h]) : 0.f;
    }
  };

  // THE CHAIN. Each output accumulates c ascending inside a chunk and chunks ascending in h,
  // so the order is the declared ascending chain over the hidden axis and is the same order
  // the one-thread-per-output shape below walks.
  auto group = [&](int buf, int c0) {
    float wv[TN][4];
#pragma unroll
    for (int b = 0; b < TN; ++b) {
      const float4 q = *reinterpret_cast<const float4 *>(&ws[buf][b * TC + ec][c0]);
      wv[b][0] = q.x;
      wv[b][1] = q.y;
      wv[b][2] = q.z;
      wv[b][3] = q.w;
    }
#pragma unroll
    for (int a = 0; a < TM; ++a) {
      const float4 q = *reinterpret_cast<const float4 *>(&xs[buf][a * TR + tr][c0]);
      const float xv[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
      for (int c = 0; c < 4; ++c)
#pragma unroll
        for (int b = 0; b < TN; ++b)
          acc[a][b] = __fmaf_rn(xv[c], wv[b][c], acc[a][b]);
    }
  };
  auto compute = [&](int buf) {
    if constexpr (UNR == 2) {
      // Two h-groups in flight: one more operand set, and it overlaps the shared-read latency.
#pragma unroll 2
      for (int c0 = 0; c0 < CH; c0 += 4)
        group(buf, c0);
    } else {
      // One group at a time: at TM*TN >= 32 the accumulator block already supplies the ILP, and a
      // second ``wv`` set would spill it.
#pragma unroll 1
      for (int c0 = 0; c0 < CH; c0 += 4)
        group(buf, c0);
    }
  };

  const int nch = (H + CH - 1) / CH;
  static_assert(SETS >= 1 && SETS <= 3, "one, two or three staging register sets");
  if (vec) {
    fetch(0, 0);
    place(0, 0);
    // Chunks 1 .. SETS-1 are already in flight when the loop opens. Spelled with literals, not
    // with an unrolled loop over the set index: a loop variable there, even one the unroller
    // turns into a constant, costs this kernel registers.
    if constexpr (SETS > 1)
      if (1 < nch)
        fetch(1, CH);
    if constexpr (SETS > 2)
      if (2 < nch)
        fetch(2, 2 * CH);
  } else {
    stage_scalar(0, 0);
  }
  __syncthreads();
  // S is literal at each call site so staging registers stay in registers.
  int k = 0;
#define SCORE_STEP(S, SNEXT)                                                            \
  do {                                                                                  \
    const int h0 = k * CH;                                                              \
    if (vec && k + SETS < nch)                                                          \
      fetch(S, h0 + SETS * CH);                                                         \
    compute(k & 1);                                                                     \
    if (k + 1 < nch) {                                                                  \
      /* Writing the other buffer needs no barrier of its own: every thread passed the  \
         barrier at the end of the previous step, after the compute that read it. */    \
      if (vec)                                                                          \
        place(SNEXT, (k + 1) & 1);                                                      \
      else                                                                              \
        stage_scalar((k + 1) & 1, h0 + CH);                                             \
      __syncthreads();                                                                  \
    }                                                                                   \
    ++k;                                                                                \
  } while (0)
  if constexpr (SETS == 1) {
    while (k < nch)
      SCORE_STEP(0, 0);
  } else if constexpr (SETS == 2) {
    // k is a multiple of SETS at the head of every round, so the literal matches k % SETS.
    while (k + 2 <= nch) {
      SCORE_STEP(0, 1);
      SCORE_STEP(1, 0);
    }
    if (k < nch)
      SCORE_STEP(0, 1);
  } else {
    while (k + 3 <= nch) {
      SCORE_STEP(0, 1);
      SCORE_STEP(1, 2);
      SCORE_STEP(2, 0);
    }
    if (k < nch)
      SCORE_STEP(0, 1);
    if (k < nch)
      SCORE_STEP(1, 2);
  }
#undef SCORE_STEP
#pragma unroll
  for (int a = 0; a < TM; ++a)
#pragma unroll
    for (int b = 0; b < TN; ++b) {
      const int t = t0 + a * TR + tr, e = e0 + b * TC + ec;
      if (t < T_ && e < E)
        logits[int64_t(t) * E + e] = acc[a][b];
    }
}

//: one thread per (token, expert), eight hidden elements per load.
template <class T>
__global__ __launch_bounds__(32, 8) void score_thread_kernel(const T *__restrict__ x,
                                                             const T *__restrict__ w,
                                                             float *__restrict__ logits, int T_,
                                                             int E, int H, int64_t x_stride,
                                                             int64_t w_stride) {
  // Four tokens by eight experts reduce cache-line traffic while preserving each FP32 chain.
  constexpr int kLaneE = 8, kLaneT = 32 / kLaneE;
  const int e_blocks = (E + kLaneE - 1) / kLaneE;
  const int wid = int(blockIdx.x) * (int(blockDim.x) >> 5) + (int(threadIdx.x) >> 5);
  const int lane = int(threadIdx.x) & 31;
  const int t = (wid / e_blocks) * kLaneT + lane / kLaneE;
  const int e = (wid - (wid / e_blocks) * e_blocks) * kLaneE + lane % kLaneE;
  if (t >= T_ || e >= E)
    return;
  const int64_t i = int64_t(t) * E + e;
  const T *xr = x + int64_t(t) * x_stride;
  const T *wr = w + int64_t(e) * w_stride;
  constexpr int kVec = 16 / int(sizeof(T));
  constexpr int kUnroll = 8;
  constexpr int kStep = kVec * kUnroll;
  float acc = 0.f;
  int h = 0;
  // A ``uint4`` load needs the ROW BASE 16-byte aligned, and a row base is ``row * stride``
  // elements in, so the stride must be a whole 16-byte load AND the tensor's own base must be
  // Vector loads require both row strides and data pointers to be 16-byte aligned.
  const bool vec = (H % kVec == 0) && (x_stride % kVec == 0) && (w_stride % kVec == 0) &&
                   (reinterpret_cast<uintptr_t>(x) % 16 == 0) &&
                   (reinterpret_cast<uintptr_t>(w) % 16 == 0);
  if (vec) {
    // Prefetch two named register blocks; dependent FFMAs remain in ascending order.
    uint4 xa[kUnroll], wa[kUnroll], xb[kUnroll], wb[kUnroll];
    auto issue = [&](uint4 (&xd)[kUnroll], uint4 (&wd)[kUnroll], int h0) {
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        xd[u] = *reinterpret_cast<const uint4 *>(xr + h0 + u * kVec);
        wd[u] = *reinterpret_cast<const uint4 *>(wr + h0 + u * kVec);
      }
    };
    auto consume = [&](const uint4 (&xd)[kUnroll], const uint4 (&wd)[kUnroll]) {
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        const T *xv = reinterpret_cast<const T *>(&xd[u]);
        const T *wv = reinterpret_cast<const T *>(&wd[u]);
#pragma unroll
        for (int j = 0; j < kVec; ++j)
          acc = __fmaf_rn(upcast<T>(xv[j]), upcast<T>(wv[j]), acc);
      }
    };
    const int blocks = H / kStep;
    if (blocks) {
      issue(xa, wa, 0);
      int b = 0;
      for (; b + 1 < blocks; b += 2) {
        issue(xb, wb, (b + 1) * kStep);
        consume(xa, wa);
        if (b + 2 < blocks)
          issue(xa, wa, (b + 2) * kStep);
        consume(xb, wb);
      }
      if (b < blocks)
        consume(xa, wa);
      h = blocks * kStep;
    }
    for (; h + kVec <= H; h += kVec) {
      const uint4 xv4 = *reinterpret_cast<const uint4 *>(xr + h);
      const uint4 wv4 = *reinterpret_cast<const uint4 *>(wr + h);
      const T *xv = reinterpret_cast<const T *>(&xv4);
      const T *wv = reinterpret_cast<const T *>(&wv4);
#pragma unroll
      for (int j = 0; j < kVec; ++j)
        acc = __fmaf_rn(upcast<T>(xv[j]), upcast<T>(wv[j]), acc);
    }
  } else {
    for (; h < H; ++h)
      acc = __fmaf_rn(upcast<T>(xr[h]), upcast<T>(wr[h]), acc);
  }
  logits[i] = acc;
}

// --------------------------------------------------------------- route_mechanics ---

//: THE selected-set sum. Schedule policy rule 2 (``isoexec/ops/schedule_policy.py`` ``selected()``): a sum over
//: data-selected coordinates is ONE SEGMENT PER COORDINATE, chained in ASCENDING EXPERT ID from +0. ``v[j]`` is
//: the term of expert ``pick[j]`` and ``pick`` is in selection order; the chain visits the terms by the rank of
//: their id, so the sum does not depend on the selection order or on which rank holds which expert. One line,
//: one place: ``moe_relations_gpu.py``'s router phase compares the weights against this chain.
template <int K, int BK>
__device__ __forceinline__ float selected_sum(const float (&v)[BK], const int (&pick)[K]) {
  int rank[K];
#pragma unroll
  for (int j = 0; j < K; ++j) {
    int lower = 0;
#pragma unroll
    for (int l = 0; l < K; ++l)
      lower += pick[l] < pick[j];
    rank[j] = lower;
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

//: THE SELECTED-SET WEIGHTS, on lane 0, for both scan shapes. One definition so that holding the
//: lane's keys in registers cannot move a rounding point: the expression below is the whole of
//: the declared program after selection -- amax, ``expf``, the ascending-id chain, the IEEE
//: non-flushing divide -- and it is reached identically from either scan.
template <int K, int BK, int MODE>
__device__ __forceinline__ void mechanics_weights(const float *__restrict__ logits,
                                                  const float (&best)[K], const int (&pick)[K],
                                                  int *__restrict__ indices,
                                                  float *__restrict__ probs, int t, int E,
                                                  float scale, int lane) {
  if (lane != 0)
    return;
  float sel[BK];
#pragma unroll
  for (int j = 0; j < BK; ++j)
    sel[j] = 0.f;
  float p[K];
  if constexpr (MODE == 0) {
    float m = best[0];
#pragma unroll
    for (int j = 1; j < K; ++j)
      m = fmaxf(m, best[j]);
#pragma unroll
    for (int j = 0; j < K; ++j)
      sel[j] = expf(best[j] - m);
    const float s = selected_sum<K, BK>(sel, pick);
#pragma unroll
    for (int j = 0; j < K; ++j)
      p[j] = sel[j] / s;   // IEEE divide, no subnormal flush
  } else {
    // The weights use the UNBIASED sigmoid score; the bias only selected.
#pragma unroll
    for (int j = 0; j < K; ++j)
      sel[j] = 1.f / (1.f + expf(-logits[int64_t(t) * E + pick[j]]));
    const float s = __fadd_rn(selected_sum<K, BK>(sel, pick), 1e-20f);
#pragma unroll
    for (int j = 0; j < K; ++j)
      p[j] = __fmul_rn(sel[j] / s, scale);
  }
#pragma unroll
  for (int j = 0; j < K; ++j) {
    indices[t * K + j] = pick[j];
    probs[t * K + j] = p[j];
  }
}

// MODE 0: softmax over the selected logits (softmax_renorm).
template <int K, int BK, int MODE, int EPL = 0>
__global__ __launch_bounds__(256, (K >= 16 ? 2 : 4)) void mechanics_kernel(
    const float *__restrict__ logits, const float *__restrict__ bias,
    int *__restrict__ indices, float *__restrict__ probs, int T_, int E, float scale) {
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  for (int t = blockIdx.x * 8 + warp; t < T_; t += gridDim.x * 8) {
    const float *row = logits + int64_t(t) * E;
    float best[K];
    int pick[K];
    if constexpr (EPL > 0) {
      float key[EPL];
#pragma unroll
      for (int i = 0; i < EPL; ++i) {
        const float s = row[lane + 32 * i];
        float k2;
        if constexpr (MODE == 0)
          k2 = s;
        else
          k2 = __fadd_rn(1.f / (1.f + expf(-s)), bias[lane + 32 * i]);
        key[i] = (k2 != k2) ? -INFINITY : k2;
      }
      uint32_t used_mask = 0u;
#pragma unroll
      for (int j = 0; j < K; ++j) {
        float bv = -INFINITY;
        int bi = E;
#pragma unroll
        for (int i = 0; i < EPL; ++i) {
          if (((used_mask >> i) & 1u) != 0u)
            continue;
          const int e = lane + 32 * i;
          if (key[i] > bv || (key[i] == bv && e < bi)) {
            bv = key[i];
            bi = e;
          }
        }
#pragma unroll
        for (int off = 16; off; off >>= 1) {
          float ov = __shfl_down_sync(0xffffffffu, bv, off);
          int oi = __shfl_down_sync(0xffffffffu, bi, off);
          if (ov > bv || (ov == bv && oi < bi)) {
            bv = ov;
            bi = oi;
          }
        }
        bv = __shfl_sync(0xffffffffu, bv, 0);
        bi = __shfl_sync(0xffffffffu, bi, 0);
        best[j] = bv;
        pick[j] = bi;
        if ((bi & 31) == lane && bi < E)
          used_mask |= (1u << (uint32_t(bi) >> 5));
      }
      mechanics_weights<K, BK, MODE>(logits, best, pick, indices, probs, t, E, scale, lane);
      continue;
    }
    // THE EXCLUSION MASK. A lane owns experts ``e = lane + 32*m``, so "expert e is already
    // picked" is bit ``m = e >> 5`` of one per-lane word -- an exact integer test, the same
    // one ``pick[p] == e`` spells, with no array and no O(K) rescan per element.
    uint32_t used_mask = 0u;
    // Exact argmax rounds. Each round is a total order on (key, expert id): the key is
    // the score (NaN mapped to -inf so it can never win) and ties go to the lowest id.
    // Selected experts are excluded by an integer test, never by poisoning the score.
#pragma unroll
    for (int j = 0; j < K; ++j) {
      float bv = -INFINITY;
      int bi = E;
      for (int e = lane; e < E; e += 32) {
        const float s = row[e];
        float key;
        if constexpr (MODE == 0) {
          key = s;
        } else {
          key = __fadd_rn(1.f / (1.f + expf(-s)), bias[e]);
        }
        if (key != key)
          key = -INFINITY;
        const uint32_t m = uint32_t(e) >> 5;
        bool used = m < 32u ? (((used_mask >> m) & 1u) != 0u) : false;
        if (m >= 32u) {
#pragma unroll
          for (int p = 0; p < j; ++p)
            used |= (pick[p] == e);
        }
        if (used)
          continue;
        if (key > bv || (key == bv && e < bi)) {
          bv = key;
          bi = e;
        }
      }
#pragma unroll
      for (int off = 16; off; off >>= 1) {
        float ov = __shfl_down_sync(0xffffffffu, bv, off);
        int oi = __shfl_down_sync(0xffffffffu, bi, off);
        if (ov > bv || (ov == bv && oi < bi)) {
          bv = ov;
          bi = oi;
        }
      }
      bv = __shfl_sync(0xffffffffu, bv, 0);
      bi = __shfl_sync(0xffffffffu, bi, 0);
      best[j] = bv;
      pick[j] = bi;
      // The winner is owned by exactly one lane, and that lane is the only one whose scan will
      // ever visit it again, so setting the bit there is the whole of the exclusion.
      if (uint32_t(bi) < 1024u && (bi & 31) == lane)
        used_mask |= (1u << (uint32_t(bi) >> 5));
    }
    mechanics_weights<K, BK, MODE>(logits, best, pick, indices, probs, t, E, scale, lane);
  }
}

// -------------------------------------------------------------------- dispatch ---


//: ``(expert, first, end)`` row blocks, expert-major, from the expert offsets. Shared by the
//: global plan and by the fused single-block plan so the tile geometry has one definition.
template <int NT> using PlanScanT = cub::BlockScan<int, NT>;
using PlanScan = PlanScanT<256>;

template <int NT = 256>
__device__ __forceinline__ void emit_tiles(const int *__restrict__ offsets,
                                           int *__restrict__ tiles, int *__restrict__ total,
                                           int E, int rows, int block_m,
                                           typename PlanScanT<NT>::TempStorage &tmp, int &carry) {
  if (!threadIdx.x)
    carry = 0;
  __syncthreads();
  for (int b0 = 0; b0 < E; b0 += NT) {
    const int e = b0 + threadIdx.x;
    const int start = e < E ? offsets[e] : 0, end = e < E ? offsets[e + 1] : 0;
    assert(start >= 0 && end >= start && end <= rows);
    const int count = (end - start + block_m - 1) / block_m;
    int prefix, aggregate;
    PlanScanT<NT>(tmp).ExclusiveSum(count, prefix, aggregate);
    const int tile = carry + prefix;
    for (int j = 0; j < count; ++j) {
      tiles[(tile + j) * 3] = e;
      tiles[(tile + j) * 3 + 1] = start + j * block_m;
      tiles[(tile + j) * 3 + 2] = end;
    }
    __syncthreads();
    if (!threadIdx.x)
      carry += aggregate;
    __syncthreads();
  }
  if (!threadIdx.x)
    *total = carry;
}

//: the canonical slot of route ``(t, s)`` with expert ``e``: how many of the token's selected
//: experts have a lower GLOBAL id, ties by route slot.
__device__ __forceinline__ int canonical_slot(const int *__restrict__ sids, int base, int t,
                                              int s, int e, int topk) {
  const int o = t * topk - base;
  int slot = 0;
  for (int j = 0; j < topk; ++j) {
    const int other = sids[o + j];
    slot += (other < e) || (other == e && j < s);
  }
  return slot;
}

__global__ void hist_kernel(const int *__restrict__ ids, int *__restrict__ block_hist,
                            int routes, int E) {
  const int block = blockIdx.x;
  const int base = block * kRouteBlock;
  int *mine = block_hist + int64_t(block) * E;
  for (int e = threadIdx.x; e < E; e += blockDim.x)
    mine[e] = 0;
  __syncthreads();
  for (int i = base + threadIdx.x; i < min(base + kRouteBlock, routes); i += blockDim.x) {
    const int e = ids[i];
    assert(e >= 0 && e < E);
    atomicAdd(mine + e, 1);
  }
}

//: exclusive prefix over experts, then the per-(block, expert) bases.
__global__ __launch_bounds__(256) void bases_kernel(int *__restrict__ offsets,
                                                    int *__restrict__ block_hist,
                                                    int *__restrict__ tiles,
                                                    int *__restrict__ total, int E, int blocks,
                                                    int rows, int block_m) {
  __shared__ PlanScan::TempStorage tmp;
  __shared__ int carry;
  if (!threadIdx.x)
    carry = 0;
  __syncthreads();
  for (int b0 = 0; b0 < E; b0 += 256) {
    const int e = b0 + threadIdx.x;
    int count = 0;
    if (e < E) {
      int b = 0;
      for (; b + 16 <= blocks; b += 16) {
        int c[16];
#pragma unroll
        for (int u = 0; u < 16; ++u)
          c[u] = block_hist[int64_t(b + u) * E + e];
#pragma unroll
        for (int u = 0; u < 16; ++u)
          count += c[u];
      }
      for (; b < blocks; ++b)
        count += block_hist[int64_t(b) * E + e];
    }
    int prefix, aggregate;
    PlanScan(tmp).ExclusiveSum(count, prefix, aggregate);
    if (e < E)
      offsets[e] = carry + prefix;
    __syncthreads();
    if (!threadIdx.x)
      carry += aggregate;
    __syncthreads();
  }
  if (!threadIdx.x)
    offsets[E] = carry;
  __syncthreads();
  // One thread per expert walks the blocks in order; E is at most a few hundred and the scan
  // is over `blocks` values, so this is a few thousand integer adds.
  for (int e = threadIdx.x; e < E; e += 256) {
    int running = offsets[e];
    int b = 0;
    for (; b + 16 <= blocks; b += 16) {
      int c[16];
#pragma unroll
      for (int u = 0; u < 16; ++u)
        c[u] = block_hist[int64_t(b + u) * E + e];
#pragma unroll
      for (int u = 0; u < 16; ++u) {
        block_hist[int64_t(b + u) * E + e] = running;
        running += c[u];
      }
    }
    for (; b < blocks; ++b) {
      const int c = block_hist[int64_t(b) * E + e];
      block_hist[int64_t(b) * E + e] = running;
      running += c;
    }
  }
  __syncthreads();
  // The tile plan reads the offsets this kernel just wrote, so it belongs in the same launch:
  // one fewer kernel on the decode path, and the values are the same integers either way.
  emit_tiles(offsets, tiles, total, E, rows, block_m, tmp, carry);
}

//: One ordered pass of 32 routes: the exact rank of each is its expert's running cursor plus
//: the number of LOWER LANES holding the same expert.
__device__ __forceinline__ void emit_ordered_chunk(const int *sids, int *cursor, int *inverse,
                                                   int *restore, int *combine_rows, int base,
                                                   int lo, int hi, int topk, int lane,
                                                   const int *slot_ids, int slot_base) {
  for (int c = lo; c < hi; c += 32) {
    const int i = c + lane;
    const int e = i < hi ? sids[i] : -1;
    const unsigned peers = __match_any_sync(0xffffffffu, e);
    const int leader = __ffs(peers) - 1;
    const unsigned lower = peers & ((1u << lane) - 1u);
    int row = 0;
    if (e >= 0)
      row = cursor[e] + __popc(lower);
    __syncwarp();
    if (e >= 0 && lane == leader)
      cursor[e] += __popc(peers);
    __syncwarp();
    if (e < 0)
      continue;
    const int flat = base + i;
    inverse[row] = flat;
    restore[flat] = row;
    if (combine_rows) {
      // Canonical slot: the number of this token's selected experts with a lower id, ties by
      // route slot. An EP or ETP split may not reorder a token's sum, so the key is the GLOBAL
      // expert id.
      const int t = flat / topk, s = flat - t * topk;
      combine_rows[t * topk + canonical_slot(slot_ids, slot_base, t, s, e, topk)] = row;
    }
  }
}

//: SUB-CHUNK CURSORS: the block's routes split across ``NW`` warps, each seeded by an
//: exclusive prefix over the earlier sub-chunks.
__device__ __forceinline__ void emit_block(const int *__restrict__ sids, int *__restrict__ cursor,
                                           const int *__restrict__ bases,
                                           int *__restrict__ inverse, int *__restrict__ restore,
                                           int *__restrict__ combine_rows, int base, int span,
                                           int E, int topk, int nw_in, const int *slot_ids,
                                           int slot_base) {
  int nw = nw_in;
  const int tid = int(threadIdx.x), nthreads = int(blockDim.x);
  // At least 128 routes a sub-chunk, so decode does not pay for the structure. At Qwen's
  // single-token row the block holds eight routes and eight sub-chunk histograms over 256
  // experts would be 2048 shared words zeroed and scanned for four ordered iterations of work.
  nw = max(1, min(nw, (span + 127) / 128));
  const int chunk = nw > 1 ? ((((span + nw - 1) / nw) + 31) & ~31) : span;
  const int warp = tid >> 5, lane = tid & 31;
  for (int i = tid; i < nw * E; i += nthreads)
    cursor[i] = 0;
  __syncthreads();
  if (nw > 1) {
    // Per-(sub-chunk, expert) counts. A route belongs to the sub-chunk that will emit it.
    for (int i = tid; i < span; i += nthreads) {
      const int w = min(i / chunk, nw - 1);
      atomicAdd(cursor + w * E + sids[i], 1);
    }
    __syncthreads();
    for (int e = tid; e < E; e += nthreads) {
      int run = bases[e];
      for (int w = 0; w < nw; ++w) {
        const int c = cursor[w * E + e];
        cursor[w * E + e] = run;
        run += c;
      }
    }
    __syncthreads();
  } else {
    for (int e = tid; e < E; e += nthreads)
      cursor[e] = bases[e];
    __syncthreads();
  }
  if (warp < nw) {
    const int lo = warp * chunk, hi = min(span, lo + chunk);
    if (lo < hi)
      emit_ordered_chunk(sids, cursor + warp * E, inverse, restore, combine_rows, base, lo, hi,
                         topk, lane, slot_ids, slot_base);
  }
}

//: emit the exact rank of every route, plus the two maps and the canonical combine slot.
__global__ __launch_bounds__(256, 4) void emit_kernel(
    const int *__restrict__ ids, const int *__restrict__ block_hist, int *__restrict__ inverse,
    int *__restrict__ restore, int *__restrict__ combine_rows, int routes, int E, int topk,
    int nw) {
  extern __shared__ int shared[];
  int *sids = shared;                   // this block's route ids
  int *cursor = shared + kRouteBlock;   // nw x E running row indices inside this block
  const int block = blockIdx.x;
  const int base = block * kRouteBlock;
  const int span = min(kRouteBlock, routes - base);
  const int *bases = block_hist + int64_t(block) * E;
  for (int i = int(threadIdx.x); i < span; i += int(blockDim.x))
    sids[i] = ids[base + i];
  __syncthreads();
  emit_block(sids, cursor, bases, inverse, restore, combine_rows, base, span, E, topk, nw, ids, 0);
}

__global__ __launch_bounds__(256) void tiles_kernel(const int *__restrict__ offsets,
                                                    int *__restrict__ tiles,
                                                    int *__restrict__ total, int E, int rows,
                                                    int block_m) {
  __shared__ PlanScan::TempStorage tmp;
  __shared__ int carry;
  emit_tiles(offsets, tiles, total, E, rows, block_m, tmp, carry);
}

//: The whole plan in one launch, for the route counts decode produces.
constexpr int kSmallRoutesOneWarpSet = 1024;   // the 256-thread CTA's band, unchanged
constexpr int kSmallRoutes = 2048;             // the 1024-thread CTA's band
constexpr int kSmallExperts = 1024;
//: room for the per-(sub-chunk, expert) cursors, in ints. Thirty-two sub-chunks fit up to E = 256;
//: above that the same kernel runs fewer of them, which is scheduling and not arithmetic.
constexpr int kSmallCursor = 8192;

//: the shared-memory bytes the fused plan needs: the block's route ids, the histogram, the expert
//: bases and the per-(sub-chunk, expert) cursors. Host-computable, which is why the launch can ask
//: for it without reading anything off the device.
inline int64_t small_plan_smem(int routes, int E, int nw) {
  return int64_t(routes + 2 * E + nw * E) * int64_t(sizeof(int));
}

inline int small_plan_warps(int E, int nt) {
  return max(1, min(nt / 32, E ? kSmallCursor / E : nt / 32));
}

template <int NT>
__global__ __launch_bounds__(NT) void small_plan_kernel(
    const int *__restrict__ ids, int *__restrict__ counts, int *__restrict__ offsets,
    int *__restrict__ inverse, int *__restrict__ restore, int *__restrict__ combine_rows,
    int *__restrict__ tiles, int *__restrict__ total, int routes, int E, int topk,
    int block_m, int nw) {
  // Dynamic, because at 8192 routes and 32 sub-chunks the four arrays are 66 KB and a static
  // ``__shared__`` declaration is capped at 48. The layout is the host's ``small_plan_smem``.
  extern __shared__ int plan_smem[];
  int *sids = plan_smem;                 // this launch's route ids, all of them
  int *hist = sids + routes;             // per-expert count
  int *base = hist + E;                  // per-expert exclusive prefix
  int *cursor = base + E;                // nw x E running row indices
  __shared__ typename PlanScanT<NT>::TempStorage tmp;
  __shared__ int carry;
  for (int e = int(threadIdx.x); e < E; e += NT)
    hist[e] = 0;
  for (int i = int(threadIdx.x); i < routes; i += NT)
    sids[i] = ids[i];
  __syncthreads();
  for (int i = int(threadIdx.x); i < routes; i += NT) {
    assert(sids[i] >= 0 && sids[i] < E);
    atomicAdd(hist + sids[i], 1);
  }
  __syncthreads();
  if (!threadIdx.x)
    carry = 0;
  __syncthreads();
  for (int b0 = 0; b0 < E; b0 += NT) {
    const int e = b0 + int(threadIdx.x);
    int prefix, aggregate;
    PlanScanT<NT>(tmp).ExclusiveSum(e < E ? hist[e] : 0, prefix, aggregate);
    if (e < E) {
      base[e] = carry + prefix;
      offsets[e] = base[e];
      counts[e] = hist[e];
    }
    __syncthreads();
    if (!threadIdx.x)
      carry += aggregate;
    __syncthreads();
  }
  if (!threadIdx.x)
    offsets[E] = carry;
  __syncthreads();
  emit_block(sids, cursor, base, inverse, restore, combine_rows, 0, routes, E, topk, nw, sids, 0);
  __syncthreads();
  emit_tiles<NT>(offsets, tiles, total, E, routes, block_m, tmp, carry);
}

// One warp per row; eight rows per block.
template <class T>
__global__ __launch_bounds__(256, 4) void gather_kernel(const T *__restrict__ x,
                                                       const float *__restrict__ p,
                                                       const int *__restrict__ inverse,
                                                       T *__restrict__ out,
                                                       float *__restrict__ op, int rows,
                                                       int hidden, int topk, bool per_warp) {
  const int warp = int(threadIdx.x) >> 5, lane = int(threadIdx.x) & 31;
  // A ``uint4`` load needs the ROW BASE 16-byte aligned, and a row base is ``row * hidden``
  // elements in, so ``hidden`` itself must be a whole number of 8 16-bit elements -- not merely
  // ``hidden / 8 > 0``, which is what the first version tested. At an odd hidden size (H = 2052,
  // say) the vector path faulted. Not a served shape, and now not a reachable one.
  const int nvec = (hidden % 8 == 0) ? hidden / 8 : 0;
  auto move = [&](int row, int first, int stride) {
    const int route = inverse[row];
    const T *src = x + int64_t(route / topk) * hidden;
    T *dst = out + int64_t(row) * hidden;
    for (int i = first; i < nvec; i += stride)
      reinterpret_cast<uint4 *>(dst)[i] = reinterpret_cast<const uint4 *>(src)[i];
    for (int i = nvec * 8 + first; i < hidden; i += stride)
      dst[i] = src[i];
    if (first == 0)
      op[row] = p[route];
  };
  if (per_warp) {
    // Enough rows to fill the machine: one warp a row, eight rows a block, grid-strided, so the
    // block-scheduling cost is paid once per SM instead of once per 4 KB row.
    for (int row = int(blockIdx.x) * 8 + warp; row < rows; row += int(gridDim.x) * 8)
      move(row, lane, 32);
  } else {
    // Too few rows for that: a warp moving 4 KB alone is eight dependent 16-byte loads per lane,
    // and with a handful of rows there is nothing else resident to hide them. Put the whole block
    // on one row instead -- one 16-byte load per lane, and the rows span as many SMs as there are.
    for (int row = int(blockIdx.x); row < rows; row += int(gridDim.x))
      move(row, int(threadIdx.x), int(blockDim.x));
  }
}

// ------------------------------------------------------------------ activation ---
//
// The FC1 fused epilogue on its own, applied to an already-rounded [R, 2I] projection. It calls
// exactly the same ``glu_epilogue`` the FC1 kernel calls, which is what makes
// ``fc1_glu(x, w) == activation(fc1(x, w))`` a bitwise structural proof: the epilogue's first
// act is to round the accumulator to the operand dtype, so it is a function of the stored
// projection alone. A torch transcription could not serve here -- ``__fdividef`` is the
// declared SiLU divide and torch has no such operator.
template <class T>
__global__ void activation_kernel(const T *__restrict__ gu, const float *__restrict__ probs,
                                  T *__restrict__ out, int64_t size, int n) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= size)
    return;
  const int64_t row = i / n;
  const int col = int(i - row * n);
  const float g = upcast<T>(gu[row * 2 * n + col]);
  const float up = upcast<T>(gu[row * 2 * n + n + col]);
  out[i] = cast<T>(glu_epilogue<T>(g, up, probs ? probs[row] : 0.f, probs != nullptr));
}

// --------------------------------------------------------------------- combine ---
//
// Prefetch KT slot loads, then add in canonical slot order (ascending global expert id: schedule policy
// rule 2, one segment per selected expert, a chain seeded with +0) with one BF16 conversion at the end.
template <class T, int KT>
__global__ void combine_kernel(const float *__restrict__ rows,
                               const int *__restrict__ order, T *__restrict__ out,
                               int hidden, int topk) {
  const int token = blockIdx.x;
  const int col = (blockIdx.y * 256 + threadIdx.x) * 4;
  if (col >= hidden)
    return;
  const int *slots = order + token * (KT ? KT : topk);
  // A float4 load needs the row base to be 16-byte aligned too, which it is exactly when the
  // hidden size is a multiple of four.
  const bool vec = (col + 4 <= hidden) && (hidden % 4 == 0);
  float acc[4];
  auto load = [&](int j, float *v) {
    const float *src = rows + int64_t(slots[j]) * hidden + col;
    if (vec) {
      const float4 q = *reinterpret_cast<const float4 *>(src);
      v[0] = q.x;
      v[1] = q.y;
      v[2] = q.z;
      v[3] = q.w;
    } else {
      for (int i = 0; i < 4; ++i)
        v[i] = col + i < hidden ? src[i] : 0.f;
    }
  };
  if constexpr (KT > 0) {
    // EVERY SLOT IN FLIGHT AT ONCE, then the same ascending chain.
    float v[KT][4];
#pragma unroll
    for (int j = 0; j < KT; ++j)
      load(j, v[j]);
#pragma unroll
    for (int i = 0; i < 4; ++i)
      acc[i] = 0.f;
#pragma unroll
    for (int j = 0; j < KT; ++j)
#pragma unroll
      for (int i = 0; i < 4; ++i)
        acc[i] = __fadd_rn(acc[i], v[j][i]);
    T *dst = out + int64_t(token) * hidden + col;
    for (int i = 0; i < 4 && col + i < hidden; ++i)
      dst[i] = cast<T>(acc[i]);
    return;
  }
#pragma unroll
  for (int i = 0; i < 4; ++i)
    acc[i] = 0.f;
  // Load all four slots before the ordered sum to hide memory latency.
  int j = 0;
  for (; j + 3 < topk; j += 4) {
    float v0[4], v1[4], v2[4], v3[4];
    load(j, v0);
    load(j + 1, v1);
    load(j + 2, v2);
    load(j + 3, v3);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      acc[i] = __fadd_rn(acc[i], v0[i]);
      acc[i] = __fadd_rn(acc[i], v1[i]);
      acc[i] = __fadd_rn(acc[i], v2[i]);
      acc[i] = __fadd_rn(acc[i], v3[i]);
    }
  }
  for (; j < topk; ++j) {
    float v[4];
    load(j, v);
#pragma unroll
    for (int i = 0; i < 4; ++i)
      acc[i] = __fadd_rn(acc[i], v[i]);
  }
  T *dst = out + int64_t(token) * hidden + col;
  for (int i = 0; i < 4 && col + i < hidden; ++i)
    dst[i] = cast<T>(acc[i]);
}

} // namespace

// ----------------------------------------------------------------------- host ---

//: the deployment target's SM count, read once. It selects a SHAPE, never an arithmetic.
int sm_count_for_score() {
  static int count = 0;
  if (count == 0) {
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device);
  }
  return count;
}

//: the tile shape wins once there are enough OUTPUTS to fill it, which is a claim about T*E
//: and not about T: at GLM's E = 64 a ``T >= 256`` rule ran the tile on rows where one thread
//: per output is still faster.
template <class T, int CH, int TR, int TC, int TM, int TN, int SETS, int MB, int UNR>
void launch_score_tile(const T *x, const T *w, float *logits, int T_, int E, int H,
                       int64_t x_stride, int64_t w_stride, cudaStream_t stream) {
  constexpr int BM = TR * TM, BN = TC * TN;
  const dim3 grid(uint32_t((E + BN - 1) / BN), uint32_t((T_ + BM - 1) / BM));
  score_tile_kernel<T, CH, TR, TC, TM, TN, SETS, MB, UNR>
      <<<grid, TR * TC, 0, stream>>>(x, w, logits, T_, E, H, x_stride, w_stride);
}

// THE FOUR MEASURED THRESHOLDS, from ``sweep_score_sm100.py`` on this device (bf16, H = 2048,
// E in {64, 256}, T in {128 .. 8192}, six samples a point, one child process per shape with a
// fixed reference GEMM as the witness -- worst-point spread 1.7 %).
constexpr int64_t kTileMinOutputs = 45000;
constexpr int64_t kTile32MinOutputs = 200000;
constexpr int64_t kWideRegTileMinOutputs = 700000;

//: the shape index the heuristic picks. Host-computable from (T, E) alone, like every other rule
//: in this package: no device read, and the battery pins each shape by name regardless.
int score_shape_for(int T_, int E) {
  const int64_t outputs = int64_t(T_) * E;
  if (outputs < kTileMinOutputs)
    return 0;                                     // thread: one chain per thread
  if (outputs < kTile32MinOutputs)
    return 5;                                     // t22:    2x2, 16x32 block, 128 threads
  if (outputs < kWideRegTileMinOutputs)
    return 1;                                     // tile32: 4x4, 32x32 block,  64 threads
  return 3;                                       // t84w:   8x4, 128x32 block, 128 threads
}

void score_cuda(const at::Tensor &x, const at::Tensor &w, at::Tensor logits) {
  const c10::cuda::CUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  const int T_ = int(x.size(0)), H = int(x.size(1)), E = int(w.size(0));
  if (!T_ || !E)
    return;
  // Which shape runs is a scheduling choice; both walk the same ascending chain per output and
  // the battery's ``router`` phase proves they agree bitwise at one geometry, in two processes.
  const int forced = -1;
  const int shape = forced >= 0 ? forced : score_shape_for(T_, E);
  // One 32-lane warp per (4 tokens x 8 experts) block of the output grid; see the kernel.
  const int threads_grid = ((T_ + 3) / 4) * ((E + 7) / 8);
#define SCORE_ARGS(T)                                                                   \
  reinterpret_cast<const T *>(x.data_ptr()), reinterpret_cast<const T *>(w.data_ptr()),  \
      logits.data_ptr<float>(), T_, E, H, x.stride(0), w.stride(0)
#define SCORE_TILE(T, CH, TR, TC, TM, TN, SETS, MB, UNR)                                \
  launch_score_tile<T, CH, TR, TC, TM, TN, SETS, MB, UNR>(                               \
      reinterpret_cast<const T *>(x.data_ptr()), reinterpret_cast<const T *>(w.data_ptr()), \
      logits.data_ptr<float>(), T_, E, H, x.stride(0), w.stride(0), stream)
  //          name    CH  TR  TC  TM  TN  SETS  MB  UNR   block tile   threads
#define SCORE(T)                                                                        \
  do {                                                                                  \
    switch (shape) {                                                                    \
    case 1: SCORE_TILE(T, 32,  8,  8, 4, 4, 2, 6, 2); break;  /* tile32   32x32    64 */ \
    case 2: SCORE_TILE(T, 32, 16,  8, 4, 4, 1, 6, 2); break;  /* tile     64x32   128 */ \
    case 3: SCORE_TILE(T, 32, 16,  8, 8, 4, 2, 4, 1); break;  /* t84w    128x32   128 */ \
    case 4: SCORE_TILE(T, 32,  8, 16, 4, 2, 2, 4, 2); break;  /* t42      32x32   128 */ \
    case 5: SCORE_TILE(T, 32,  8, 16, 2, 2, 2, 4, 2); break;  /* t22      16x32   128 */ \
    default:                                                                            \
      score_thread_kernel<T><<<threads_grid, 32, 0, stream>>>(SCORE_ARGS(T));            \
    }                                                                                   \
  } while (0)
  if (x.scalar_type() == at::kBFloat16) {
    SCORE(__nv_bfloat16);
  } else {
    SCORE(__half);
  }
#undef SCORE
#undef SCORE_ARGS
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void mechanics_cuda(const at::Tensor &logits, const c10::optional<at::Tensor> &bias,
                    at::Tensor indices, at::Tensor probs, double scale) {
  const c10::cuda::CUDAGuard guard(logits.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  const int T_ = int(logits.size(0)), E = int(logits.size(1)), K = int(indices.size(1));
  if (!T_)
    return;
  const float *bp = bias.has_value() ? bias->data_ptr<float>() : nullptr;
  const int blocks = std::min(2048, (T_ + 7) / 8);
  // The register-key scan needs E to be an exact multiple of 32 so a lane's keys are exactly
  // ``E/32`` slots; the two served geometries are E = 256 and E = 64. Anything else runs the
  // loop, which is the same program (see the kernel) and is what the battery's odd-E case gates.
#define MECH_EPL(K, BK, MODE, EPL)                                                      \
  mechanics_kernel<K, BK, MODE, EPL><<<blocks, 256, 0, stream>>>(                        \
      logits.data_ptr<float>(), bp, indices.data_ptr<int>(), probs.data_ptr<float>(),    \
      T_, E, float(scale))
#define MECH(K, BK, MODE)                                                               \
  do {                                                                                  \
    if (E == 256)                                                                       \
      MECH_EPL(K, BK, MODE, 8);                                                         \
    else if (E == 64)                                                                   \
      MECH_EPL(K, BK, MODE, 2);                                                         \
    else if (E == 32)                                                                   \
      MECH_EPL(K, BK, MODE, 1);                                                         \
    else                                                                                \
      MECH_EPL(K, BK, MODE, 0);                                                         \
  } while (0)
#define MECH_MODE(MODE)                                                                 \
  do {                                                                                  \
    switch (K) {                                                                        \
    case 1:                                                                             \
      MECH(1, 1, MODE);                                                                 \
      break;                                                                            \
    case 2:                                                                             \
      MECH(2, 2, MODE);                                                                 \
      break;                                                                            \
    case 4:                                                                             \
      MECH(4, 4, MODE);                                                                 \
      break;                                                                            \
    case 8:                                                                             \
      MECH(8, 8, MODE);                                                                 \
      break;                                                                            \
    case 16:                                                                            \
      MECH(16, 16, MODE);                                                               \
      break;                                                                            \
    default:                                                                            \
      TORCH_CHECK(false, "sm100 MoE router needs a power-of-two top-k, got ", K);                     \
    }                                                                                   \
  } while (0)
  if (bp) {
    MECH_MODE(1);
  } else {
    MECH_MODE(0);
  }
#undef MECH_MODE
#undef MECH
#undef MECH_EPL
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void plan_cuda(const at::Tensor &ids, at::Tensor counts, at::Tensor offsets,
               at::Tensor block_hist, at::Tensor inverse, at::Tensor restore,
               at::Tensor combine_rows, at::Tensor tiles, at::Tensor total,
               int64_t block_m) {
  const c10::cuda::CUDAGuard guard(ids.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  const int routes = int(ids.numel()), E = int(counts.numel()),
            topk = int(ids.size(1));
  const int blocks = (routes + kRouteBlock - 1) / kRouteBlock;
  int *crows = combine_rows.numel() ? combine_rows.data_ptr<int>() : nullptr;
  if (routes && routes <= kSmallRoutes && E <= kSmallExperts) {
    // The whole plan in ONE launch, for every route count up to ``kSmallRoutes``. Same integers,
    // one kernel-launch floor instead of four. The CTA is 256 threads in the band the earlier
    // rounds measured and 1024 above it, because the emit walk is the term that scales and it
    // scales with the number of sub-chunk warps.
#define SMALL_PLAN(NT)                                                                  \
  do {                                                                                  \
    const int nw = small_plan_warps(E, NT);                                              \
    const int64_t bytes = small_plan_smem(routes, E, nw);                                \
    TORCH_CHECK(bytes <= 200000, "sm100 MoE plan: too many experts for one block's cursor"); \
    C10_CUDA_CHECK(cudaFuncSetAttribute(small_plan_kernel<NT>,                           \
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,      \
                                        int(bytes)));                                    \
    small_plan_kernel<NT><<<1, NT, size_t(bytes), stream>>>(                             \
        ids.data_ptr<int>(), counts.data_ptr<int>(), offsets.data_ptr<int>(),             \
        inverse.data_ptr<int>(), restore.data_ptr<int>(), crows, tiles.data_ptr<int>(),   \
        total.data_ptr<int>(), routes, E, topk, int(block_m), nw);                        \
  } while (0)
    if (routes <= kSmallRoutesOneWarpSet)
      SMALL_PLAN(256);
    else
      SMALL_PLAN(1024);
#undef SMALL_PLAN
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  if (routes) {
    hist_kernel<<<blocks, 256, 0, stream>>>(ids.data_ptr<int>(),
                                            block_hist.data_ptr<int>(), routes, E);
    bases_kernel<<<1, 256, 0, stream>>>(
        offsets.data_ptr<int>(), block_hist.data_ptr<int>(),
        tiles.data_ptr<int>(), total.data_ptr<int>(), E, blocks, routes, int(block_m));
    // Eight sub-chunk cursors where they fit, one where they do not. Which one runs is
    // scheduling: the ranks are a count over the flat route index either way, and the
    // ``dispatch`` battery phase gates both paths at five shapes.
    const int warps = std::max(1, std::min(8, E ? 8192 / E : 8));
    const size_t emit_smem = size_t(kRouteBlock + warps * E) * sizeof(int);
    TORCH_CHECK(emit_smem <= 200000u, "sm100 MoE plan: too many experts for one block's cursor");
    C10_CUDA_CHECK(cudaFuncSetAttribute(emit_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        int(emit_smem)));
    emit_kernel<<<blocks, 256, emit_smem, stream>>>(
        ids.data_ptr<int>(), block_hist.data_ptr<int>(), inverse.data_ptr<int>(),
        restore.data_ptr<int>(), crows, routes, E, topk, warps);
  } else {
    bases_kernel<<<1, 256, 0, stream>>>(
        offsets.data_ptr<int>(), block_hist.data_ptr<int>(),
        tiles.data_ptr<int>(), total.data_ptr<int>(), E, 0, routes, int(block_m));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void tiles_cuda(const at::Tensor &offsets, at::Tensor tiles, at::Tensor total,
                int64_t rows, int64_t block_m) {
  const c10::cuda::CUDAGuard guard(offsets.device());
  tiles_kernel<<<1, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      offsets.data_ptr<int>(), tiles.data_ptr<int>(), total.data_ptr<int>(),
      int(offsets.numel()) - 1, int(rows), int(block_m));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gather_cuda(const at::Tensor &x, const at::Tensor &p, const at::Tensor &inverse,
                 at::Tensor out, at::Tensor op, int64_t topk) {
  const c10::cuda::CUDAGuard guard(x.device());
  if (!out.size(0))
    return;
  auto stream = at::cuda::getCurrentCUDAStream();
  const int rows = int(out.size(0));
  // Which distribution runs is pure scheduling -- every row is written by exactly one owner and
  // the bytes are the token's bytes either way, which is what the battery checks.
  const bool per_warp = rows >= 4 * sm_count_for_score();
  const int grid = per_warp ? std::min((rows + 7) / 8, 8 * sm_count_for_score()) : rows;
#define GATHER(T)                                                                       \
  gather_kernel<T><<<grid, 256, 0, stream>>>(                                            \
      reinterpret_cast<const T *>(x.data_ptr()), p.data_ptr<float>(),                    \
      inverse.data_ptr<int>(), reinterpret_cast<T *>(out.data_ptr()),                    \
      op.data_ptr<float>(), rows, int(x.size(1)), int(topk), per_warp)
  if (x.scalar_type() == at::kBFloat16) {
    GATHER(__nv_bfloat16);
  } else {
    GATHER(__half);
  }
#undef GATHER
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void activation_cuda(const at::Tensor &gu, const at::Tensor &probs, at::Tensor out) {
  TORCH_CHECK(gu.is_cuda() && gu.is_contiguous() && out.is_contiguous() && gu.dim() == 2 &&
              out.dim() == 2 && gu.size(0) == out.size(0) && gu.size(1) == 2 * out.size(1),
              "sm100 MoE activation expects a contiguous [R, 2I] projection");
  const c10::cuda::CUDAGuard guard(gu.device());
  if (!out.numel())
    return;
  auto stream = at::cuda::getCurrentCUDAStream();
  const float *p = probs.numel() ? probs.data_ptr<float>() : nullptr;
  TORCH_CHECK(!p || probs.numel() == out.size(0), "one probability per row");
#define ACT(T)                                                                          \
  activation_kernel<T><<<(out.numel() + 255) / 256, 256, 0, stream>>>(                    \
      reinterpret_cast<const T *>(gu.data_ptr()), p,                                      \
      reinterpret_cast<T *>(out.data_ptr()), out.numel(), int(out.size(1)))
  if (gu.scalar_type() == at::kBFloat16) {
    ACT(__nv_bfloat16);
  } else {
    ACT(__half);
  }
#undef ACT
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void combine_cuda(const at::Tensor &rows, const at::Tensor &order, at::Tensor out) {
  const c10::cuda::CUDAGuard guard(rows.device());
  if (!out.numel())
    return;
  auto stream = at::cuda::getCurrentCUDAStream();
  const int hidden = int(out.size(1));
  const int topk = int(order.size(1));
  dim3 grid(out.size(0), (hidden + 1023) / 1024);
#define COMBINE_K(T, KT)                                                                \
  combine_kernel<T, KT><<<grid, 256, 0, stream>>>(                                       \
      rows.data_ptr<float>(), order.data_ptr<int>(),                                     \
      reinterpret_cast<T *>(out.data_ptr()), hidden, topk)
#define COMBINE(T)                                                                      \
  do {                                                                                  \
    switch (topk) {                                                                     \
    case 1: COMBINE_K(T, 1); break;                                                     \
    case 2: COMBINE_K(T, 2); break;                                                     \
    case 4: COMBINE_K(T, 4); break;                                                     \
    case 8: COMBINE_K(T, 8); break;                                                     \
    case 16: COMBINE_K(T, 16); break;                                                   \
    default: COMBINE_K(T, 0);                                                           \
    }                                                                                   \
  } while (0)
  if (out.scalar_type() == at::kBFloat16) {
    COMBINE(__nv_bfloat16);
  } else if (out.scalar_type() == at::kHalf) {
    COMBINE(__half);
  } else {
    COMBINE(float);
  }
#undef COMBINE
#undef COMBINE_K
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace moe100
