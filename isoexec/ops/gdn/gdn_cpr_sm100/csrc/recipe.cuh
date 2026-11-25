// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

namespace gdn_cpr_sm100 {

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
__device__ __forceinline__ uint32_t rn2(float lo, float hi) {
  const __nv_bfloat162 p = __floats2bfloat162_rn(lo, hi);   // .x -> low half, .y -> high half
  uint32_t r;
  memcpy(&r, &p, 4);
  return r;
}
// q^ / k^: rn(x (*) inv) for two adjacent elements of a row, `inv` the row's (scaled) inverse norm.
__device__ __forceinline__ uint32_t norm_pair(float x0, float x1, float inv) {
  return rn2(fmul(x0, inv), fmul(x1, inv));
}
__device__ __forceinline__ uint16_t lo16(uint32_t x) { return uint16_t(x & 0xffffu); }
__device__ __forceinline__ uint16_t hi16(uint32_t x) { return uint16_t(x >> 16); }

// ------------------------------------------------------------------------------ token preparation
// FOLD32 lives in `l2norm_rows` below: the 32 per-lane chunk partials of one warp merged by the
// adjacent-pairs tree (butterfly with ascending offsets; IEEE addition is commutative, so every lane ends
// with the root's bits).
__device__ __forceinline__ float inv_norm(float folded_sum_sq) { return __frsqrt_rn(fadd(folded_sum_sq, 1e-6f)); }

template <int R, bool GUARD = true>
__device__ __forceinline__ void l2norm_rows(const float (&x)[R][CE], float scale, bool is_query, uint2 (&out)[R],
                                            float *inverse_norms = nullptr, int nrow = R) {
  float ss[R];
  #pragma unroll
  for (int r = 0; r < R; ++r)
    if (!GUARD || r < nrow) {
      ss[r] = 0.f;
      #pragma unroll
      for (int e = 0; e < CE; ++e) ss[r] = ffma(x[r][e], x[r][e], ss[r]);
    }
  #pragma unroll
  for (int off = 1; off < NC; off *= 2)
    #pragma unroll
    for (int r = 0; r < R; ++r)
      if (!GUARD || r < nrow) ss[r] = fadd(ss[r], __shfl_xor_sync(0xffffffffu, ss[r], off));
  #pragma unroll
  for (int r = 0; r < R; ++r)
    if (!GUARD || r < nrow) {
      float inv = inv_norm(ss[r]);
      if (inverse_norms) inverse_norms[r] = inv;
      if (is_query) inv = fmul(inv, scale);
      out[r] = make_uint2(norm_pair(x[r][0], x[r][1], inv), norm_pair(x[r][2], x[r][3], inv));
    }
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

__device__ __forceinline__ uint32_t m_entry2(bool live0, float kq0, float dec0, bool live1, float kq1,
                                             float dec1) {
  return rn2(live0 ? fmul(kq0, dec0) : 0.0f, live1 ? fmul(kq1, dec1) : 0.0f);
}
__device__ __forceinline__ uint32_t xu_entry2(float x0, float b0, float x1, float b1) {
  return rn2(fmul(x0, b0), fmul(x1, b1));
}
__device__ __forceinline__ uint32_t xw_entry2(float x0, float b0, float g0, float x1, float b1, float g1) {
  return rn2(fmul(fmul(x0, b0), g0), fmul(fmul(x1, b1), g1));
}
__device__ __forceinline__ uint32_t delta_entry2(uint16_t u0, float sw0, uint16_t u1, float sw1) {
  return rn2(fsub(bf(u0), sw0), fsub(bf(u1), sw1));
}
__device__ __forceinline__ uint32_t dtilde_entry2(uint32_t d, float dec_last_i) {
  return rn2(fmul(bf(lo16(d)), dec_last_i), fmul(bf(hi16(d)), dec_last_i));
}
// rn of a whole pair, used where no scaling precedes it (S^ = rn(S_c), o = rn(..), FLA's A save, and
// u / w = rn(CHAIN64), whose absent rows enter as +0.0f -- rn(+0.0f) IS the 0x0000 those elements held
// when they were selected away one at a time).
__device__ __forceinline__ uint32_t rn_pair(float a, float b) { return rn2(a, b); }
__device__ __forceinline__ float out_seed(float sq, float gam_j) { return fmul(sq, gam_j); }
__device__ __forceinline__ float state_seed(float s, float gam_last) { return fmul(s, gam_last); }

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

// T16 over the chunks of rows [0, 16 * nb): P_jm. a(i) = L_ji, b(i) = X_im for i < 16 * nb. Branch-free:
// the chains of chunks [0, MAXC) are all evaluated and those at or past 4 nb replaced by +0 (a select, so
// rows [16 nb, 4 MAXC) must be readable but may hold anything).
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

// ------------------------------------------------------------------------------ tensor-core contraction
// One m16n8k16 bf16 MMA accumulating into d. Fragment layouts (g = lane / 4, t = lane % 4):
//   a[0] = A[g][2t..2t+1], a[1] = A[g+8][2t..], a[2] = A[g][2t+8..], a[3] = A[g+8][2t+8..]
//   b[0] = B[2t..2t+1][g], b[1] = B[2t+8..2t+9][g]
//   d[0] = D[g][2t], d[1] = D[g][2t+1], d[2] = D[g+8][2t], d[3] = D[g+8][2t+1]
// Retained on sm_100a; the probe in tests/test_mma_facts_gpu.py is the gate.
__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void mma_p(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2],
                                      unsigned need) {
  asm volatile(
      "{\n\t"
      ".reg .pred pg;\n\t"
      "setp.ne.b32 pg, %10, 0;\n\t"
      "@pg mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n\t"
      "}"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(need));
}

}  // namespace gdn_cpr_sm100
