// SPDX-License-Identifier: Apache-2.0
// The per-row scalar program of the mHC split: 24 projected logits -> pre / post / comb (Sinkhorn).
// One warp holds one row. Lane layout of the results:
//   lanes 0..3   pre[lane]      = sigmoid(mix*scale0 + base) + eps
//   lanes 4..7   post[lane-4]   = 2 * sigmoid(mix*scale1 + base)
//   lanes 16..31 comb[lane-16]  (row-major [source][destination]), Sinkhorn-normalized
// All sums over four elements are the declared tree (e0 + e1) + (e2 + e3), obtained by xor shuffles, which
// is bitwise symmetric across the four lanes. Divisions are IEEE div.rn; exp is the CUDA expf.
#pragma once
#include "common.cuh"

namespace mhc {

#define FULL 0xffffffffu

__device__ __forceinline__ float sum4_row(float v) {   // over the 4 lanes of a comb row (xor 1, xor 2)
  v = __fadd_rn(v, __shfl_xor_sync(FULL, v, 1));
  return __fadd_rn(v, __shfl_xor_sync(FULL, v, 2));
}
__device__ __forceinline__ float sum4_col(float v) {   // over the 4 rows of a comb column (xor 4, xor 8)
  v = __fadd_rn(v, __shfl_xor_sync(FULL, v, 4));
  return __fadd_rn(v, __shfl_xor_sync(FULL, v, 8));
}

// mix_lane: the 24 mixes in lanes 0..23 (any value elsewhere). Returns pp (lanes 0..7) and comb (lanes 16..31).
__device__ __forceinline__ void tail_forward(int lane, float mix_lane, const float *scale, const float *base,
                                             float eps, int iters, float &pp, float &comb) {
  const float mix_c = __shfl_sync(FULL, mix_lane, 8 + (lane & 15));
  {
    const float a = __fadd_rn(__fmul_rn(mix_lane, scale[lane < 4 ? 0 : 1]), base[lane & 7]);
    const float s = sigmoid_rn(a);
    pp = lane < 4 ? __fadd_rn(s, eps) : __fmul_rn(2.0f, s);
  }
  const float a = __fadd_rn(__fmul_rn(mix_c, scale[2]), base[8 + (lane & 15)]);
  float m = fmaxf(a, __shfl_xor_sync(FULL, a, 1));
  m = fmaxf(m, __shfl_xor_sync(FULL, m, 2));
  const float e = expf(__fsub_rn(a, m));
  float p = __fadd_rn(__fdiv_rn(e, sum4_row(e)), eps);
  p = __fdiv_rn(p, __fadd_rn(sum4_col(p), eps));
  for (int it = 1; it < iters; ++it) {
    p = __fdiv_rn(p, __fadd_rn(sum4_row(p), eps));
    p = __fdiv_rn(p, __fadd_rn(sum4_col(p), eps));
  }
  comb = p;
}

// Two rows at once (see tail_kernel): mix_a / mix_b hold the 24 mixes of rows A / B in lanes 0..23. Returns pp_a / pp_b (lanes 0..7
// as in tail_forward) and comb: row A's Sinkhorn result in lanes 16..31, row B's in lanes 0..15. Every lane computes exactly what
// tail_forward computes for its own row, so the bits equal two separate tail_forward calls.
__device__ __forceinline__ void tail_forward_pair(int lane, float mix_a, float mix_b, const float *scale, const float *base,
                                                  float eps, int iters, float &pp_a, float &pp_b, float &comb) {
  const float mca = __shfl_sync(FULL, mix_a, 8 + (lane & 15));
  const float mcb = __shfl_sync(FULL, mix_b, 8 + (lane & 15));
  const float mix_c = lane >= 16 ? mca : mcb;
  {
    const float sc = scale[lane < 4 ? 0 : 1], bs = base[lane & 7];
    const float sa = sigmoid_rn(__fadd_rn(__fmul_rn(mix_a, sc), bs));
    const float sb = sigmoid_rn(__fadd_rn(__fmul_rn(mix_b, sc), bs));
    pp_a = lane < 4 ? __fadd_rn(sa, eps) : __fmul_rn(2.0f, sa);
    pp_b = lane < 4 ? __fadd_rn(sb, eps) : __fmul_rn(2.0f, sb);
  }
  const float a = __fadd_rn(__fmul_rn(mix_c, scale[2]), base[8 + (lane & 15)]);
  float m = fmaxf(a, __shfl_xor_sync(FULL, a, 1));
  m = fmaxf(m, __shfl_xor_sync(FULL, m, 2));
  const float e = expf(__fsub_rn(a, m));
  float p = __fadd_rn(__fdiv_rn(e, sum4_row(e)), eps);
  p = __fdiv_rn(p, __fadd_rn(sum4_col(p), eps));
  for (int it = 1; it < iters; ++it) {
    p = __fdiv_rn(p, __fadd_rn(sum4_row(p), eps));
    p = __fdiv_rn(p, __fadd_rn(sum4_col(p), eps));
  }
  comb = p;
}

}  // namespace mhc
