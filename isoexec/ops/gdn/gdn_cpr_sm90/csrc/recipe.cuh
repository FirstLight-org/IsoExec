// SPDX-License-Identifier: Apache-2.0
// Gated DeltaNet core, chunk-parallel with recurrent decode (CPR) -- arithmetic identity version 1.
//
// This header IS the numerical contract. Every kernel of the family (the two prefill passes and the
// fused decode step) evaluates each quantity through the definitions below and nothing else, compiled
// with one profile (sm_90a, --fmad=false, --ftz=false). The definitions are ROW-SEPARABLE: the value
// of every per-token quantity depends only on that token, the tokens before it in its chunk and the
// chunk's boundary state -- never on later tokens, batch, packing, split point or execution plan. That
// is what lets decode evaluate "row j" of the very same chunk arithmetic that prefill evaluates for
// all 64 rows at once, and get the same bits.
//
// Geometry. D = 128 (head_k_dim = head_v_dim). Chunks are C = 64 tokens ALIGNED TO ABSOLUTE SEQUENCE
// POSITION: token p of a sequence is row j = p mod 64 of chunk c = p div 64. S_c is the fp32 [V, K]
// state at the start of chunk c (S_0 = the initial state). Per (token, value head h, key head hq).
//
// Notation: (+) (-) (*) RN single operations, fma one RN fused multiply-add, bf() the exact bf16->fp32
// widening, rn() the RN narrowing fp32->bf16, e() = expf. MMA = mma.sync.m16n8k16.row.col.f32.bf16:
//   CHAIN64(a, b; seed)  = acc <- seed; for s = 0..3 ascending: acc <- MMA(a[16s:16s+16], b[16s:16s+16], acc)
//   CHAIN128(a, b; seed) = the same over s = 0..7 (128-wide operands)
// A contraction's bits depend only on its two operand vectors and its seed (verified: placement
// invariance of the MMA unit, invariance to what a zero operand faces, skipping a k16 step whose B
// column is +0, WGMMA == this chain, and the chain with its operand roles swapped == its transpose --
// tests/test_mma_facts_gpu.py).
//
// Token preparation (bf16 inputs; FOLD32 as in gdn_sm90: 32 fma-chains of 4 + adjacent-pairs tree):
//   k^ = rn(kf (*) rsqrt_rn(FOLD32(kf^2) (+) 1e-6))                      [bf16, the only key operand]
//   q^ = rn(qf (*) (rsqrt_rn(FOLD32(qf^2) (+) 1e-6) (*) 128^-1/2))       [bf16]
//   x = bf(a) (+) dt_bias;  sp = x <= 20 ? log1pf(e(x)) : x;  lg = -(e(A_log) (*) sp);  beta = 1 (/) (1 (+) e(-bf(b)))
// Chunk-local decay: G_j = G_{j-1} (+) lg_j with G_{-1} = +0.
//
// Row j of chunk c, from S_c (fp32) and rows i <= j of the same chunk:
//   KK_ij = CHAIN128(k^_i, k^_j; +0)    KQ_ij = CHAIN128(k^_i, q^_j; +0)
//   L_ji  = (KK_ij (*) beta_j) (*) e(G_j (-) G_i)                         (i < j)
//   X = (I + L)^-1 by 16-row blocks; row j lies in block B = j div 16; CH4_c(a, b) is the fma chain over
//   i in [4c, 4c + 4) ascending, seeded +0; T4 = ((c0 (+) c1) (+) (c2 (+) c3)) over c = 4B..4B+3; T16 is
//   the adjacent-pairs tree over c = 0..15:
//     X_jj = 1;  X_jm = +0 (m > j)
//     X_jm = -T4_c CH4_c(L_j., X_.m)                          m in [16B, j)   (the diagonal block)
//     P_jm =  T16_c [c < 4B ? CH4_c(L_j., X_.m) : +0]         m < 16B         (earlier blocks)
//     X_jm = -T4_c CH4_c(Y_j., P_.m),  Y_ji = X_ji (i < j), 1 (i = j), 0 (i > j)     m < 16B
//   (X_ba = -X_bb (L_b,<b X_<b,a): rows of P are carried in the open-chunk buffer like rows of X.)
//   xu_ji = rn(X_ji (*) beta_i)                   xw_ji = rn((X_ji (*) beta_i) (*) e(G_i))
//   u_j   = rn(CHAIN64_i(v_i, xu_ji; +0))         w_j   = rn(CHAIN64_i(k^_i, xw_ji; +0))           [X_ji = 0 for i > j]
//   (u = (I + L)^-1 diag(beta) V and w = (I + L)^-1 diag(beta e(G)) K: the per-row factors ride on the
//    64-long X row, so the 128-wide operands are the raw values and the normalized keys themselves)
//   S^    = rn(S_c)
//   d_j   = rn(bf(u_j) (-) CHAIN128_d(S^, w_j; +0))                                                     [delta, bf16]
//   M_ji  = rn(KQ_ij (*) e(G_j (-) G_i))  (i <= j, else 0)
//   o_j   = rn(CHAIN64_i(d_i, M_ji; seed = CHAIN128_d(S^, q^_j; +0) (*) e(G_j)))                        [output, bf16]
// Boundary (row 63 done):
//   S_{c+1} = CHAIN64_i(rn(bf(d_i) (*) e(G_63 (-) G_i)), k^_i; seed = S_c (*) e(G_63))            [fp32]
//
// The chunk is the unit of the recipe, not of execution: prefill evaluates rows [r_lo, r_hi) of a chunk
// with rows [0, r_lo) read back from the state's open-chunk buffer (k^, v, d, X and P rows, G, beta), and
// decode is the case r_hi = r_lo + 1. The boundary state is replaced exactly when row 63 completes,
// by whichever kernel completes it. Everything else -- which warp, CTA, tile, split or kernel evaluates
// a contraction -- is execution plan and free.
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace gdn_cpr {

constexpr int D = 128;   // head_k_dim == head_v_dim
constexpr int C = 64;    // chunk length (absolute-position aligned)
constexpr int NC = 32;   // FOLD32 chunks (== lanes per warp)
constexpr int CE = 4;    // FOLD32 elements per chunk

// ------------------------------------------------------------------------------ scalar operations
__device__ __forceinline__ float bf(uint16_t bits) { return __bfloat162float(__ushort_as_bfloat16(bits)); }
__device__ __forceinline__ uint16_t rn(float x) { return __bfloat16_as_ushort(__float2bfloat16_rn(x)); }
__device__ __forceinline__ float fadd(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ float fsub(float a, float b) { return __fsub_rn(a, b); }
__device__ __forceinline__ float fmul(float a, float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ float ffma(float a, float b, float c) { return __fmaf_rn(a, b, c); }
__device__ __forceinline__ uint32_t pack(uint16_t lo, uint16_t hi) { return uint32_t(lo) | (uint32_t(hi) << 16); }
__device__ __forceinline__ uint16_t lo16(uint32_t x) { return uint16_t(x & 0xffffu); }
__device__ __forceinline__ uint16_t hi16(uint32_t x) { return uint16_t(x >> 16); }

// ------------------------------------------------------------------------------ token preparation
// FOLD32: the 32 per-lane chunk partials of one warp merged by the adjacent-pairs tree (butterfly with
// ascending offsets; IEEE addition is commutative, so every lane ends with the root's bits).
__device__ __forceinline__ float fold32(float x) {
  #pragma unroll
  for (int off = 1; off < NC; off *= 2) x = fadd(x, __shfl_xor_sync(0xffffffffu, x, off));
  return x;
}

__device__ __forceinline__ float inv_norm(float folded_sum_sq) { return __frsqrt_rn(fadd(folded_sum_sq, 1e-6f)); }

// A warp normalizes R independent 128-wide rows at once; lane c owns elements 4c..4c+3 of each. The
// butterfly levels run over all rows before the next level, so the R folds overlap instead of
// queueing -- each row's fold is still exactly FOLD32. Returns, per row, the lane's 4 bf16 results
// packed as two words (elements 4c, 4c+1 | 4c+2, 4c+3). `scale` multiplies the inverse norm of queries.
// The optional inverse_norms keeps the unscaled inverse for trainer saves without repeating FOLD32.
template <int R>
__device__ __forceinline__ void l2norm_rows(const float (&x)[R][CE], float scale, bool is_query, uint2 (&out)[R], float *inverse_norms = nullptr) {
  float ss[R];
  #pragma unroll
  for (int r = 0; r < R; ++r) {
    ss[r] = 0.f;
    #pragma unroll
    for (int e = 0; e < CE; ++e) ss[r] = ffma(x[r][e], x[r][e], ss[r]);
  }
  #pragma unroll
  for (int off = 1; off < NC; off *= 2)
    #pragma unroll
    for (int r = 0; r < R; ++r) ss[r] = fadd(ss[r], __shfl_xor_sync(0xffffffffu, ss[r], off));
  #pragma unroll
  for (int r = 0; r < R; ++r) {
    float inv = inv_norm(ss[r]);
    if (inverse_norms) inverse_norms[r] = inv;
    if (is_query) inv = fmul(inv, scale);
    out[r] = make_uint2(pack(rn(fmul(x[r][0], inv)), rn(fmul(x[r][1], inv))),
                        pack(rn(fmul(x[r][2], inv)), rn(fmul(x[r][3], inv))));
  }
}
__device__ __forceinline__ uint2 l2norm_row(const float (&x)[CE], float scale, bool is_query) {
  float xr[1][CE] = {{x[0], x[1], x[2], x[3]}};
  uint2 out[1];
  l2norm_rows<1>(xr, scale, is_query, out);
  return out[0];
}

struct Gate {
  float lg;     // log decay (<= 0)
  float beta;
};

__device__ __forceinline__ Gate gate(float a, float b, float A_log, float dt_bias) {
  float x = fadd(a, dt_bias);
  float sp = x <= 20.0f ? log1pf(expf(x)) : x;
  return Gate{-fmul(expf(A_log), sp), __fdiv_rn(1.0f, fadd(1.0f, expf(-b)))};
}

// ------------------------------------------------------------------------------ chunk scalars
// Decay factors are e(G_a (-) G_b) (or e(G_j) for the chunk-start factor); an evaluator may compute one
// factor per row and reuse it -- the factor is a function of its two arguments only.
__device__ __forceinline__ float cum_gate(float G_prev, float lg) { return fadd(G_prev, lg); }
__device__ __forceinline__ float decay(float G_a, float G_b) { return expf(fsub(G_a, G_b)); }
__device__ __forceinline__ float decay0(float G) { return expf(G); }
__device__ __forceinline__ float l_entry(float kk, float beta_j, float dec_ji) { return fmul(fmul(kk, beta_j), dec_ji); }
__device__ __forceinline__ uint16_t m_entry(float kq, float dec_ji) { return rn(fmul(kq, dec_ji)); }
__device__ __forceinline__ uint16_t xu_entry(float x_ji, float beta_i) { return rn(fmul(x_ji, beta_i)); }
__device__ __forceinline__ uint16_t xw_entry(float x_ji, float beta_i, float gam_i) {
  return rn(fmul(fmul(x_ji, beta_i), gam_i));
}
__device__ __forceinline__ uint16_t delta_entry(uint16_t u, float sw) { return rn(fsub(bf(u), sw)); }
__device__ __forceinline__ uint16_t dtilde_entry(uint16_t d, float dec_last_i) { return rn(fmul(bf(d), dec_last_i)); }
__device__ __forceinline__ float out_seed(float sq, float gam_j) { return fmul(sq, gam_j); }
__device__ __forceinline__ float state_seed(float s, float gam_last) { return fmul(s, gam_last); }

// The two contraction shapes of X = (I + L)^-1 (see the header). Operands are passed for whole 4-row fold
// chunks; rows that are not part of a sum enter with a zero coefficient instead of being skipped.
//
// Zero extension: that is the same sum bit for bit. An fma chain seeded with +0 never holds -0 (an RN sum
// is -0 only when both addends are -0), so ffma(+0, x, p) == p for every finite x; the trees then see
// identical leaves. Evaluators therefore only need the operands of absent rows to be finite -- never
// NaN or Inf -- which they guarantee by zero-filling.
//
// -T4 over the 16 rows of one block: the diagonal-block entries and the earlier-block entries of X.
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

// T16 over the chunks of rows [0, 16 * nb) (nb = B, the number of earlier blocks): P_jm. a(i) = L_ji,
// b(i) = X_im for i < 16 * nb. Branch-free: the chains of chunks [0, MAXC) are all evaluated and those at
// or past 4 nb replaced by +0 (a select, so rows [16 nb, 4 MAXC) must be readable but may hold anything);
// chunks past MAXC are +0 by construction (callers guarantee 4 nb <= MAXC).
template <int MAXC, class AF, class BF>
__device__ __forceinline__ float p_entry(int nb, AF a, BF b) {
  float part[16];
  #pragma unroll
  for (int c = 0; c < 16; ++c) {
    float acc = 0.f;
    if (c < MAXC) {
      #pragma unroll
      for (int e = 0; e < 4; ++e) acc = ffma(a(4 * c + e), b(4 * c + e), acc);
    }
    part[c] = c < 4 * nb ? acc : 0.f;
  }
  #pragma unroll
  for (int span = 1; span < 16; span *= 2)
    #pragma unroll
    for (int c = 0; c + span < 16; c += 2 * span) part[c] = fadd(part[c], part[c + span]);
  return part[0];
}

// ------------------------------------------------------------------------------ tensor-core contractions
// One m16n8k16 bf16 MMA accumulating into d. Fragment layouts (g = lane / 4, t = lane % 4):
//   a[0] = A[g][2t..2t+1], a[1] = A[g+8][2t..], a[2] = A[g][2t+8..], a[3] = A[g+8][2t+8..]
//   b[0] = B[2t..2t+1][g],  b[1] = B[2t+8..2t+9][g]
//   d[0] = D[g][2t], d[1] = D[g][2t+1], d[2] = D[g+8][2t], d[3] = D[g+8][2t+1]
__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}


}  // namespace gdn_cpr
