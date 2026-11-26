// SPDX-License-Identifier: Apache-2.0
// GLM Kimi Delta Attention on SM100 -- numerical recipe `sm100_kda_v1`, version 1.
//
// THIS HEADER IS THE NUMERICAL CONTRACT. Every kernel of the package evaluates each quantity through the
// definitions below, compiled with one profile (sm_100a, --fmad=false, --ftz=false: build_cache's
// NUMERICS_FLAGS). Everything not written here -- which warp, CTA, launch or tile evaluates a quantity,
// how many value columns a CTA owns, pipelining -- is execution and does not enter the bits.
//
// Geometry: D = 128 key channels = 128 value channels per head (GLM-5.3-Flash linear_head_dim), any
// head count. Chunks of C = 64 tokens anchored at ABSOLUTE SEQUENCE POSITION 0 (a call continuing an engine
// record resumes inside its open chunk, csrc/record.cuh); a chunk
// holds r <= 64 rows (only a sequence's last chunk may be partial). Sub-blocks of 16 rows; row j lies in
// block b(j) = j div 16 whose first row is n(j) = 16 b(j). S_c is the fp32 [K = 128][V = 128] state
// entering chunk c (S_0 = the initial state or +0).
//
// Notation: (+) (-) (*) RN fp32 operations, fma = one RN fused multiply-add, bf() exact bf16 -> fp32
// widening, rn() RN fp32 -> bf16, E(x) = ex2.approx.f32(x) (MUFU.EX2, subnormals preserved: a fixed
// hardware function of its fp32 input, so its bits are a property of the profile like expf's would be).
// MMA = mma.sync.m16n8k16.row.col.f32.bf16.bf16.f32:
//   CHAIN_n(a, b; seed) = acc <- seed; for s = 0 .. n/16 - 1 ascending: acc <- MMA(a[16s..], b[16s..], acc)
// so a contraction's bits depend only on its two operand vectors and its seed (placement invariance and
// the zero-slice facts are gated for this instruction on B200 by gdn_cpr_sm100/tests/test_mma_facts_gpu.py).
//
// Token preparation (per token and head; inputs q, k, v bf16, log decay g fp32 or bf16, beta bf16/fp32):
//   ss  = FOLD32(x^2): lane l of a warp owns elements 4l..4l+3, an ascending fma chain seeded +0, then the
//         xor butterfly with offsets 1, 2, 4, 8, 16 (fadd).
//   k^  = rn(bf(k) (*) rsqrt_rn(ss_k (+) 1e-6))
//   q^  = rn(bf(q) (*) (rsqrt_rn(ss_q (+) 1e-6) (*) 128^-1/2))
//   beta = widened input
//   G_j = G_{j-1} (+) (g_j (*) LOG2E) per channel, G_{-1} = +0 (chunk-local log2 decay, ascending rows)
//   L   = r - 1, the chunk's last row.
// Operand tiles (bf16, per row and channel):
//   kg_j = rn(k^_j (*) E(G_j))          qg_j = rn(q^_j (*) E(G_j))          kd_j = rn(k^_j (*) E(G_L (-) G_j))
//   kr_j = rn(k^_j (*) E(G_j (-) G_n(j)))     qr_j = rn(q^_j (*) E(G_j (-) G_n(j)))          (j >= 16)
//   ki^b_i = rn(k^_i (*) E(G_16b (-) G_i))                                          (b = 1, 2, 3; i < 16 b)
// Every exponent above is <= 0 for the rows it is used on, so no factor overflows for ANY finite log
// decay <= 0; this is why the decay reference is the row block's first row and why the diagonal blocks
// are not factored through a reference (a factored diagonal needs E(+sum g) and overflows for strong
// decay -- FLA's `safe_gate` restriction, which this recipe does not need).
// Intra-chunk products (i <= j):
//   b(i) < b(j):  KK_ji = CHAIN128(kr_j, ki^b(j)_i; +0)    KQ_ji = CHAIN128(qr_j, ki^b(j)_i; +0)
//   b(i) = b(j):  KK_ji = DIAG(k^_j, k^_i)                  KQ_ji = DIAG(q^_j, k^_i)
//     the diagonal block carries the DECAYED KEY as an ascending product of per-token factors (each <= 1,
//     so it underflows gracefully and never overflows):
//       delta_t = E(G_t (-) G_{t-1}),   T_ii = bf(k^_i),   T_ji = T_{j-1,i} (*) delta_j   (j > i, same block)
//     DIAG(a_j, k_i) = T8 over the eighths Q_p = {c : (c >> 2) & 7 == p}, p = 0..7, of
//       CH_p = ascending-c fma chain seeded +0 of  fma(bf(a_jc), T_ji,c, acc),
//       T8 = (((CH_0 (+) CH_1) (+) (CH_2 (+) CH_3)) (+) ((CH_4 (+) CH_5) (+) (CH_6 (+) CH_7))).
//   L_ji = KK_ji (*) beta_j  (i < j; +0 otherwise)      Mq_ji = rn(KQ_ji)  (i <= j; +0 otherwise)
// X = (I + L)^-1 by 16-row blocks, row j in block B (gdn_cpr_sm100's block recipe), with
//   CH4_c(a, b) = fma chain over i in [4c, 4c + 4) ascending seeded +0, T4 / T16 = adjacent-pairs trees:
//   X_jj = 1;  X_jm = +0 (m > j)
//   X_jm = -T4_{c in block B} CH4_c(L_j., X_.m)                               16 B <= m < j
//   P_jm = T16_{c = 0..15} [c < 4 B ? CH4_c(L_j., X_.m) : +0]                  m < 16 B
//   X_jm = -T4_{c in block B} CH4_c(Y_j., P_.m),  Y_ji = X_ji (i < j), 1 (i = j), +0 (i > j)
// WY rows (CHAIN64_i^b = the CHAIN over the i-blocks 0 .. b ascending, i.e. k-steps 0 .. b only):
//   xu_ji = rn(X_ji (*) beta_i)
//   u_j = rn(CHAIN64_i^b(j)(xu_j., v_i; +0))          w_j = rn(CHAIN64_i^b(j)(xu_j., kg_i; +0))
// Chunk step from S_c (the only part that depends on the state):
//   S^ = rn(S_c)
//   d_j  = rn(bf(u_j) (-) CHAIN128_c(w_j, S^; +0))
//   o_j  = rn(CHAIN64_i^b(j)(Mq_j., d_i; seed = CHAIN128_c(qg_j, S^; +0)))         [output, bf16]
//   S_{c+1} = CHAIN64_i^b(L)(kd_i^T, d_i; seed = S_c (*) E(G_L)[channel])           [fp32]
// THE CHAIN BOUNDS ARE A FUNCTION OF THE ROW BLOCK, never of the call: xu_ji, Mq_ji = +0 for i > j and
// kd_i, d_i of absent rows are +0, so a k-step past the bound only adds +0 products. Writing the bound into
// the program (rather than issuing those steps) is what lets a one-row decode step issue one k-step
// instead of four with no relation to argue: row j's chain has b(j) + 1 steps in every evaluator.
// (S_{c+1}'s bound depends on the chunk's row count r, which the recipe already fixes: a chunk boundary
// is part of the program, see "Relations" in README.md.)
// The final state of a sequence is S after its last (possibly partial) chunk.
//
// Absent rows (r <= j < 64 of a partial chunk) enter as k^ = q^ = v = beta = g = +0, so every operand they
// contribute is finite and every coefficient that multiplies them is +0: rows j < r are the same bits
// whether the chunk is evaluated with 64 rows of storage or r. Rows are NOT row-separable across the chunk
// boundary choice: a chunk boundary is part of the program (README "Relations").
#pragma once
#include "common.cuh"

namespace kda_sm100 {

constexpr float QSCALE = 0.08838834764831845f;   // 128^-1/2 rounded to fp32
constexpr float LOG2E = 1.4426950408889634f;     // log2(e) rounded to fp32

// ------------------------------------------------------------------------------ token preparation
__device__ __forceinline__ float inv_norm(float folded) { return __frsqrt_rn(fadd(folded, 1e-6f)); }

// FOLD32 of 4 elements per lane (lane l owns 4l..4l+3); every lane ends with the root's bits.
__device__ __forceinline__ float fold32(const float (&x)[4]) {
  float s = 0.f;
  #pragma unroll
  for (int e = 0; e < 4; ++e) s = ffma(x[e], x[e], s);
  #pragma unroll
  for (int off = 1; off < 32; off *= 2) s = fadd(s, __shfl_xor_sync(0xffffffffu, s, off));
  return s;
}

__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float cum_gate(float G_prev, float g) { return fadd(G_prev, fmul(g, LOG2E)); }
__device__ __forceinline__ float decay(float G_a, float G_b) { return ex2(fsub(G_a, G_b)); }
__device__ __forceinline__ float decay0(float G) { return ex2(G); }

// ------------------------------------------------------------------------------ diagonal products
// Eighth p's chain over its 16 channels c = 32 m + 4 p + e, m = 0..3, e = 0..3 (ascending c).
// T8 across the eight lanes of an octet (lane & 7 == p), adjacent pairs: every lane ends with the root.
__device__ __forceinline__ float oct_t8(float p) {
  p = fadd(p, __shfl_xor_sync(0xffffffffu, p, 1));
  p = fadd(p, __shfl_xor_sync(0xffffffffu, p, 2));
  return fadd(p, __shfl_xor_sync(0xffffffffu, p, 4));
}
// T4 across the four lanes of a quad (the backward's owner rows; not part of the forward program).
__device__ __forceinline__ float quad_t4(float p) {
  p = fadd(p, __shfl_xor_sync(0xffffffffu, p, 1));
  return fadd(p, __shfl_xor_sync(0xffffffffu, p, 2));
}

// ------------------------------------------------------------------------------ X solve
// -T4 over the 16 rows of one block: a[i], b[i] for the block's rows in ascending order.
__device__ __forceinline__ float x_block(const float (&a)[16], const float (&b)[16]) {
  float part[4];
  #pragma unroll
  for (int c = 0; c < 4; ++c) {
    part[c] = 0.f;
    #pragma unroll
    for (int e = 0; e < 4; ++e) part[c] = ffma(a[4 * c + e], b[4 * c + e], part[c]);
  }
  return -fadd(fadd(part[0], part[1]), fadd(part[2], part[3]));
}
// T16 over the 4-row chunks of rows [0, 16 nb), the chunks at or past 4 nb replaced by +0.
template <class AF, class BF>
__device__ __forceinline__ float p_entry(int nb, AF a, BF b) {
  float part[16];
  #pragma unroll
  for (int c = 0; c < 16; ++c) {
    float acc = 0.f;
    if (c < 4 * nb) {
      #pragma unroll
      for (int e = 0; e < 4; ++e) acc = ffma(a(4 * c + e), b(4 * c + e), acc);
    }
    part[c] = acc;
  }
  #pragma unroll
  for (int span = 1; span < 16; span *= 2)
    #pragma unroll
    for (int c = 0; c + span < 16; c += 2 * span) part[c] = fadd(part[c], part[c + span]);
  return part[0];
}

}  // namespace kda_sm100
