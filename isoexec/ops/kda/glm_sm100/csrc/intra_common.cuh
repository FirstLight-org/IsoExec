// SPDX-License-Identifier: Apache-2.0
// The state-independent chunk phases of the recipe as device functions, shared by the forward phase-1
// kernel (intra.cu) and the backward kernels' recompute (bwd.cu), so the backward differentiates exactly
// the program the forward evaluated. All functions are called by a whole 256-thread CTA; those that
// contain a barrier say so.
#pragma once
#include "record.cuh"
#include "recipe.cuh"

namespace kda_sm100 {

using T128 = Tile<64, 128>;
using T64 = Tile<64, 64>;
constexpr int LS = 68;   // fp32 row stride of the 64 x 64 scratch matrices (floats)
constexpr int NT = 256;  // threads of every chunk kernel

__device__ __forceinline__ float ldg_val(const void *p, int64_t idx, int is_bf16) {
  return is_bf16 ? bf(static_cast<const uint16_t *>(p)[idx]) : static_cast<const float *>(p)[idx];
}
// rn(x (*) E(Ga (-) Gb)) / rn(x (*) e) for the two bf16 of a fragment register.
__device__ __forceinline__ uint32_t scale2(uint32_t x, float2 Ga, float2 Gb) {
  return rn2(fmul(bf(lo16(x)), decay(Ga.x, Gb.x)), fmul(bf(hi16(x)), decay(Ga.y, Gb.y)));
}
__device__ __forceinline__ uint32_t scale2e(uint32_t x, float e0, float e1) {
  return rn2(fmul(bf(lo16(x)), e0), fmul(bf(hi16(x)), e1));
}

// (jj, ii) of the 136 pairs jj >= ii of a 16-row block, pair index jj (jj + 1) / 2 + ii.
struct PairTable {
  unsigned char jj[136], ii[136];
  constexpr PairTable() : jj(), ii() {
    int p = 0;
    for (int a = 0; a < 16; ++a)
      for (int b = 0; b <= a; ++b) {
        jj[p] = (unsigned char)a;
        ii[p] = (unsigned char)b;
        ++p;
      }
  }
};
__constant__ PairTable kPairs = PairTable();

// cp.async a [64][128] bf16 row block (rows >= r zero-filled) into a swizzled tile. No commit.
__device__ __forceinline__ void load_tile128(const T128 &dst, const uint16_t *src, int t0, int r, int64_t rowstride,
                                             int64_t off, int tid) {
  #pragma unroll
  for (int it = 0; it < 4; ++it) {
    const int idx = it * NT + tid, row = idx >> 4, ch = idx & 15;
    const bool valid = row < r;
    cp16(dst.saddr(row, ch), src + (int64_t(t0) + (valid ? row : 0)) * rowstride + off + ch * 8, valid);
  }
}
// cp.async rows [lo, hi) of a [64][128] bf16 block (token t0 + row) into a tile, rows from hi on are +0;
// rows below lo are not touched (the caller fills them from the record).
__device__ __forceinline__ void load_rows128(const T128 &dst, const uint16_t *src, int t0, int lo, int hi,
                                             int64_t rowstride, int64_t off, int tid) {
  #pragma unroll
  for (int it = 0; it < 4; ++it) {
    const int idx = it * NT + tid, row = idx >> 4, ch = idx & 15;
    const bool valid = row < hi;
    if (row >= lo) cp16(dst.saddr(row, ch), src + (int64_t(t0) + (valid ? row : lo)) * rowstride + off + ch * 8, valid);
  }
}
// cp.async rows [0, n) of a linear [64][128] bf16 record section into a tile (other rows untouched).
__device__ __forceinline__ void load_rec_rows128(const T128 &dst, const uint16_t *src, int n, int tid) {
  #pragma unroll
  for (int it = 0; it < 4; ++it) {
    const int idx = it * NT + tid, row = idx >> 4, ch = idx & 15;
    if (row < n) cp16(dst.saddr(row, ch), src + row * D + ch * 8, true);
  }
}

// The raw log decay rows: fp32 straight into G ([64][128] floats), bf16 into `raw` (64 x 256 B). No commit.
// Rows [lo, r) are this call's tokens; rows [0, lo) of G are left alone (the caller loads the record's G).
__device__ __forceinline__ void load_g(const void *g, int g_bf16, float *G, uint16_t *raw, int t0, int r,
                                       int64_t rowH, int64_t hoff, int tid, int lo = 0) {
  if (g_bf16) {
    #pragma unroll
    for (int it = 0; it < 4; ++it) {
      const int idx = it * NT + tid, row = idx >> 4, ch = idx & 15;
      const bool valid = row >= lo && row < r;
      cp16(smem_u32(raw) + uint32_t(idx * 16),
           static_cast<const uint16_t *>(g) + (int64_t(t0) + (valid ? row : lo)) * rowH + hoff + ch * 8, valid);
    }
  } else {
    #pragma unroll
    for (int it = 0; it < 8; ++it) {
      const int idx = it * NT + tid, row = idx >> 5, ch = idx & 31;
      const bool valid = row >= lo && row < r;
      if (row >= lo)
        cp16(smem_u32(G) + uint32_t(idx * 16),
             static_cast<const float *>(g) + (int64_t(t0) + (valid ? row : lo)) * rowH + hoff + ch * 4, valid);
    }
  }
}

// k^ / q^ in place (warp w: rows 8w..8w+7). `inv_out` (optional) receives the UNSCALED inverse norm of
// each row (for the l2 norm backward).
// Rows below `lo` (an open chunk's rows taken from the record, already k^) are left untouched.
__device__ __forceinline__ void l2norm_tile(const T128 &tt, bool query, float *inv_out, int warp, int lane,
                                            int lo = 0) {
  float x[8][4];
  #pragma unroll
  for (int rr = 0; rr < 8; ++rr) {
    const uint2 w2 = *reinterpret_cast<const uint2 *>(&tt.at(8 * warp + rr, 4 * lane));
    x[rr][0] = bf(lo16(w2.x));
    x[rr][1] = bf(hi16(w2.x));
    x[rr][2] = bf(lo16(w2.y));
    x[rr][3] = bf(hi16(w2.y));
  }
  float ss[8];
  #pragma unroll
  for (int rr = 0; rr < 8; ++rr) {
    ss[rr] = 0.f;
    #pragma unroll
    for (int e = 0; e < 4; ++e) ss[rr] = ffma(x[rr][e], x[rr][e], ss[rr]);
  }
  #pragma unroll
  for (int off = 1; off < 32; off *= 2)
    #pragma unroll
    for (int rr = 0; rr < 8; ++rr) ss[rr] = fadd(ss[rr], __shfl_xor_sync(0xffffffffu, ss[rr], off));
  #pragma unroll
  for (int rr = 0; rr < 8; ++rr) {
    float inv = inv_norm(ss[rr]);
    if (inv_out && lane == 0) inv_out[8 * warp + rr] = inv;
    if (query) inv = fmul(inv, QSCALE);
    uint2 o2;
    o2.x = rn2(fmul(x[rr][0], inv), fmul(x[rr][1], inv));
    o2.y = rn2(fmul(x[rr][2], inv), fmul(x[rr][3], inv));
    if (8 * warp + rr >= lo) *reinterpret_cast<uint2 *>(&tt.at(8 * warp + rr, 4 * lane)) = o2;
  }
}

// G chain of channel tid (tid < 128), in place over the raw rows (absent rows hold g = +0), continuing
// from G_{lo-1} when rows [0, lo) already hold the record's G.
__device__ __forceinline__ void g_chain(float *G, const uint16_t *raw, int g_bf16, int tid, int lo = 0) {
  if (tid < D) {
    float acc = lo > 0 ? G[(lo - 1) * D + tid] : 0.f;
    #pragma unroll 16
    for (int j = 0; j < 64; ++j) {
      if (j < lo) continue;
      const float gv = g_bf16 ? bf(raw[j * D + tid]) : G[j * D + tid];
      acc = cum_gate(acc, gv);
      G[j * D + tid] = acc;
    }
  }
}

// KK (fp32, into KK[j][i] for i < j of the off-diagonal block pairs) and, WITH_Q, Mq = rn(KQ) into Mq.
// Warp w < 6 owns block pair w of (1,0) (2,0) (2,1) (3,0) (3,1) (3,2); the decay-factored operands are
// formed in registers element by element (the recipe's rn(x (*) E(..)), same bits as staged tiles).
template <bool WITH_Q>
__device__ __forceinline__ void offdiag_products(const T128 &Kh, const T128 &Qh, const float *G, float *KK,
                                                 const T64 &Mq, int warp, int lane) {
  if (warp >= 6) return;
  const int g = lane >> 2, t = lane & 3;
  const int bj = warp == 0 ? 1 : (warp < 3 ? 2 : 3);
  const int bi = warp < 2 ? 0 : (warp == 2 ? 1 : warp - 3);
  const int j0 = 16 * bj, i0 = 16 * bi;
  float kk[2][4] = {}, kq[2][4] = {};
  #pragma unroll 2
  for (int ks = 0; ks < 8; ++ks) {
    const int c = 16 * ks + 2 * t;
    uint32_t ak[4], aq[4], b[4];
    frag_a(ak, Kh, j0, 16 * ks, lane);
    if (WITH_Q) frag_a(aq, Qh, j0, 16 * ks, lane);
    frag_b2(b, Kh, i0, 16 * ks, lane);
    const float2 Gn0 = *reinterpret_cast<const float2 *>(G + j0 * D + c);
    const float2 Gn1 = *reinterpret_cast<const float2 *>(G + j0 * D + c + 8);
    #pragma unroll
    for (int e = 0; e < 4; ++e) {   // A rows j0 + g (+8), channels c (+8)
      const int row = j0 + g + 8 * (e & 1);
      const float2 Gn = (e & 2) ? Gn1 : Gn0;
      const float2 Gr = *reinterpret_cast<const float2 *>(G + row * D + c + 8 * (e >> 1));
      const float e0 = decay(Gr.x, Gn.x), e1 = decay(Gr.y, Gn.y);
      ak[e] = scale2e(ak[e], e0, e1);
      if (WITH_Q) aq[e] = scale2e(aq[e], e0, e1);
    }
    #pragma unroll
    for (int e = 0; e < 4; ++e) {   // B rows i0 + g (+8 for the second n8 tile), channels c (+8)
      const int row = i0 + g + 8 * (e >> 1);
      const float2 Gn = (e & 1) ? Gn1 : Gn0;
      const float2 Gr = *reinterpret_cast<const float2 *>(G + row * D + c + 8 * (e & 1));
      b[e] = scale2(b[e], Gn, Gr);
    }
    mma(kk[0], ak, b[0], b[1]);
    mma(kk[1], ak, b[2], b[3]);
    if (WITH_Q) {
      mma(kq[0], aq, b[0], b[1]);
      mma(kq[1], aq, b[2], b[3]);
    }
  }
  #pragma unroll
  for (int nt = 0; nt < 2; ++nt)
    #pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int j = j0 + g + 8 * (e >> 1), i = i0 + 8 * nt + 2 * t + (e & 1);
      KK[j * LS + i] = kk[nt][e];
      if (WITH_Q) Mq.at(j, i) = rn(kq[nt][e]);
    }
}

// delta_t = E(G_t (-) G_{t-1}) for t = 1..63, in place over G (thread per channel, descending t so every
// G_{t-1} is read before it is overwritten). Row 0 keeps G_0 (no diagonal pair uses delta of a block's
// first row). Call after the last reader of G.
__device__ __forceinline__ void delta_in_place(float *G, int tid) {
  if (tid < D) {
    float hi = G[63 * D + tid];
    #pragma unroll 8
    for (int t = 63; t >= 1; --t) {
      const float lo = G[(t - 1) * D + tid];
      G[t * D + tid] = decay(hi, lo);
      hi = lo;
    }
  }
}

// The diagonal blocks: DIAG with the decayed key T_ji = T_{j-1,i} (*) delta_j over Dl = delta.
// Octet (warp w, lane >> 3) owns block w >> 1 and the column pair {ip, 15 - ip}, ip = 4 (w & 1) + octet,
// lane p = lane & 7 owning channel eighth p; it sweeps rows j = i .. 15 of each column, 17 steps for every
// octet (16 - ip + ip + 1), so the warp stays converged for the T8 shuffles.
template <bool WITH_Q>
__device__ __forceinline__ void diag_products(const T128 &Kh, const T128 &Qh, const float *Dl, float *KK,
                                              const T64 &Mq, int warp, int lane) {
  const int B = warp >> 1, ip = 4 * (warp & 1) + (lane >> 3), p = lane & 7;
  float T[16];
  int i = 0;
  #pragma unroll 1
  for (int s = 0; s < 17; ++s) {
    const bool first_col = s < 16 - ip;
    const int col = first_col ? ip : 15 - ip;
    const int j = 16 * B + (first_col ? ip + s : (15 - ip) + (s - (16 - ip)));
    if (s == 0 || s == 16 - ip) {   // a new column: T = k^_i
      i = 16 * B + col;
      #pragma unroll
      for (int m = 0; m < 4; ++m) {
        const uint2 w2 = *reinterpret_cast<const uint2 *>(&Kh.at(i, 32 * m + 4 * p));
        T[4 * m] = bf(lo16(w2.x)); T[4 * m + 1] = bf(hi16(w2.x));
        T[4 * m + 2] = bf(lo16(w2.y)); T[4 * m + 3] = bf(hi16(w2.y));
      }
    } else {
      #pragma unroll
      for (int m = 0; m < 4; ++m) {
        const float4 d4 = *reinterpret_cast<const float4 *>(Dl + j * D + 32 * m + 4 * p);
        T[4 * m] = fmul(T[4 * m], d4.x);
        T[4 * m + 1] = fmul(T[4 * m + 1], d4.y);
        T[4 * m + 2] = fmul(T[4 * m + 2], d4.z);
        T[4 * m + 3] = fmul(T[4 * m + 3], d4.w);
      }
    }
    float kk = 0.f, kq = 0.f;
    #pragma unroll
    for (int m = 0; m < 4; ++m) {
      const uint2 kw = *reinterpret_cast<const uint2 *>(&Kh.at(j, 32 * m + 4 * p));
      const float kj[4] = {bf(lo16(kw.x)), bf(hi16(kw.x)), bf(lo16(kw.y)), bf(hi16(kw.y))};
      float qj[4] = {0.f, 0.f, 0.f, 0.f};
      if (WITH_Q) {
        const uint2 qw = *reinterpret_cast<const uint2 *>(&Qh.at(j, 32 * m + 4 * p));
        qj[0] = bf(lo16(qw.x)); qj[1] = bf(hi16(qw.x)); qj[2] = bf(lo16(qw.y)); qj[3] = bf(hi16(qw.y));
      }
      #pragma unroll
      for (int e = 0; e < 4; ++e) {
        kk = ffma(kj[e], T[4 * m + e], kk);
        if (WITH_Q) kq = ffma(qj[e], T[4 * m + e], kq);
      }
    }
    kk = oct_t8(kk);
    if (WITH_Q) kq = oct_t8(kq);
    if (p == 0) {
      if (i < j) KK[j * LS + i] = kk;
      if (WITH_Q) Mq.at(j, i) = rn(kq);
    }
  }
}

// X = (I + L)^-1 with L_ji = KK_ji (*) beta_j evaluated on read (the recipe's L). Contains barriers.
// KK must be +0 for i >= j. Writes X (fp32, [64][LS]) for i <= j; P is [16][LS] scratch.
// `psave` (optional) receives the P rows of block `save_block` ([16][P_COLS] fp32, rows of that block) before
// the next round overwrites them: an open chunk's record needs them for its next rows.
__device__ __forceinline__ void x_solve(const float *KK, const float *beta, float *X, float *P, int tid,
                                        float *psave = nullptr, int save_block = -1) {
  if (tid < 64) {   // phase a: column m of diagonal block B
    const int B = tid >> 4, mm = tid & 15, m = 16 * B + mm;
    float xcol[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) xcol[i] = i == mm ? 1.f : 0.f;
    #pragma unroll
    for (int jj = 0; jj < 16; ++jj) {
      if (jj > mm) {
        const int j = 16 * B + jj;
        const float bj = beta[j];
        float lrow[16];
        #pragma unroll
        for (int i = 0; i < 16; ++i) lrow[i] = fmul(KK[j * LS + 16 * B + i], bj);
        xcol[jj] = x_block(lrow, xcol);
      }
    }
    #pragma unroll
    for (int jj = 0; jj < 16; ++jj) X[(16 * B + jj) * LS + m] = xcol[jj];
  }
  __syncthreads();
  #pragma unroll 1
  for (int B = 1; B < 4; ++B) {   // phase b: the earlier-block entries, block by block
    const int n = 16 * 16 * B;    // entries (j in block B, m < 16 B)
    for (int idx = tid; idx < n; idx += NT) {
      const int jj = idx / (16 * B), m = idx - jj * (16 * B), j = 16 * B + jj;
      const float bj = beta[j];
      P[jj * LS + m] = p_entry(B, [&](int i) { return fmul(KK[j * LS + i], bj); },
                               [&](int i) { return i < m ? 0.f : X[i * LS + m]; });
    }
    __syncthreads();
    if (psave && B == save_block)   // [16][P_COLS] rows: +0 past the block's earlier columns
      for (int idx = tid; idx < 16 * P_COLS; idx += NT) {
        const int jj = idx / P_COLS, m = idx - jj * P_COLS;
        psave[idx] = m < 16 * B ? P[jj * LS + m] : 0.f;
      }
    for (int idx = tid; idx < n; idx += NT) {
      const int jj = idx / (16 * B), m = idx - jj * (16 * B), j = 16 * B + jj;
      float y[16], pc[16];
      #pragma unroll
      for (int i = 0; i < 16; ++i) {
        y[i] = i < jj ? X[j * LS + 16 * B + i] : (i == jj ? 1.f : 0.f);
        pc[i] = P[i * LS + m];
      }
      X[j * LS + m] = x_block(y, pc);
    }
    __syncthreads();
  }
}

// xu = rn(X (*) beta_i) into a [64][64] bf16 tile (+0 above the diagonal).
__device__ __forceinline__ void make_xu(const float *X, const float *beta, const T64 &Xu, int tid) {
  for (int idx = tid; idx < 64 * 32; idx += NT) {
    const int j = idx >> 5, i0 = 2 * (idx & 31);
    const float x0 = i0 <= j ? X[j * LS + i0] : 0.f;
    const float x1 = i0 + 1 <= j ? X[j * LS + i0 + 1] : 0.f;
    *reinterpret_cast<uint32_t *>(&Xu.at(j, i0)) = rn2(fmul(x0, beta[i0]), fmul(x1, beta[i0 + 1]));
  }
}

}  // namespace kda_sm100
