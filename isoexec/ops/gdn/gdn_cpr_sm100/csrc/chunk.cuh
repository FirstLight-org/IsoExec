// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "params.cuh"
#include "tile.cuh"

namespace gdn_cpr_sm100 {

constexpr int THREADS = 256;

constexpr int LS = 68;   // L row stride in floats: 16-byte aligned rows, conflict-free fragment stores

template <int N, int NT, class F>
__device__ __forceinline__ void static_tasks_n(int tid, F body) {
  constexpr int IT = (N + NT - 1) / NT;
  #pragma unroll
  for (int it = 0; it < IT; ++it) {
    const int x = tid + it * NT;
    if (N % NT != 0 && x >= N) break;
    body(x);
  }
}
template <int N, class F>
__device__ __forceinline__ void static_tasks(int tid, F body) {
  static_tasks_n<N, THREADS>(tid, body);
}

struct GateTile {
  float G[C], beta[C], gam[C], lam[C];
};

struct CarryRef {
  float *X, *P, *G, *beta;
};

// Trainer mode: FLA's normalized rows (queries without the attention scale) and the inverse norms.
__device__ __forceinline__ void save_norms(const Params &p, const ChunkGeo &g, int hq, bool query,
                                           const float (&x)[C / 8][CE], const uint2 (&nrm)[C / 8],
                                           const float (&inverse_norms)[C / 8], int row0, int nrow,
                                           int lane) {
  #pragma unroll
  for (int rr = 0; rr < C / 8; ++rr) {
    if (rr >= nrow) break;
    const int i = row0 + 8 * rr;
    const float inv = inverse_norms[rr];
    const int64_t row = g.token(i) * p.hk + hq;
    uint2 o = nrm[rr];
    if (query)
      o = make_uint2(norm_pair(x[rr][0], x[rr][1], inv), norm_pair(x[rr][2], x[rr][3], inv));
    *reinterpret_cast<uint2 *>((query ? p.sv.qn : p.sv.kn) + row * D + 4 * lane) = o;
    if (lane == 0) (query ? p.sv.q_rstd : p.sv.k_rstd)[row] = inv;
  }
}

template <int NMT>
__device__ __forceinline__ void dots128_pair(float (&kk)[2][2][4], float (&kq)[2][2][4], const KTile &kt,
                                             const KTile &qt, int i0, int j0, int lane) {
  #pragma unroll
  for (int mt = 0; mt < NMT; ++mt)
    #pragma unroll
    for (int nt = 0; nt < 2; ++nt)
      #pragma unroll
      for (int e = 0; e < 4; ++e) {
        kk[mt][nt][e] = 0.f;
        kq[mt][nt][e] = 0.f;
      }
  #pragma unroll
  for (int ks = 0; ks < 8; ++ks) {
    uint32_t bk[4], bq[4];
    frag_b2(bk, kt, j0, 16 * ks, lane);
    frag_b2(bq, qt, j0, 16 * ks, lane);
    const uint32_t bk0[2] = {bk[0], bk[1]}, bk1[2] = {bk[2], bk[3]};
    const uint32_t bq0[2] = {bq[0], bq[1]}, bq1[2] = {bq[2], bq[3]};
    #pragma unroll
    for (int mt = 0; mt < NMT; ++mt) {
      uint32_t a[4];
      frag_a(a, kt, i0 + 16 * mt, 16 * ks, lane);
      mma(kk[mt][0], a, bk0);
      mma(kk[mt][1], a, bk1);
      mma(kq[mt][0], a, bq0);
      mma(kq[mt][1], a, bq1);
    }
  }
}

constexpr int KKQ_TILES = 16;                 // (i-block, j-block) pairs of a 64 x 64 product
constexpr int KKQ_FLOATS = KKQ_TILES * 32 * 8;

__device__ __forceinline__ int kkq_tile(int di0, int dj0) { return (dj0 >> 4) * 4 + (di0 >> 4); }

// One tile's eight accumulator floats for this lane, as two float4.
__device__ __forceinline__ void kkq_load(float (&acc)[2][4], const float *base, int di0, int dj0, int lane) {
  const float4 *src = reinterpret_cast<const float4 *>(base + (kkq_tile(di0, dj0) * 32 + lane) * 8);
  const float4 a = src[0], b = src[1];
  acc[0][0] = a.x; acc[0][1] = a.y; acc[0][2] = a.z; acc[0][3] = a.w;
  acc[1][0] = b.x; acc[1][1] = b.y; acc[1][2] = b.z; acc[1][3] = b.w;
}

__device__ __forceinline__ void kkq_store(const float (&acc)[2][4], float *base, int di0, int dj0, int lane) {
  float4 *dst = reinterpret_cast<float4 *>(base + (kkq_tile(di0, dj0) * 32 + lane) * 8);
  dst[0] = make_float4(acc[0][0], acc[0][1], acc[0][2], acc[0][3]);
  dst[1] = make_float4(acc[1][0], acc[1][1], acc[1][2], acc[1][3]);
}

// What launch A reads out of the (chunk, key head) hand-off instead of deriving it. `nullptr` in every
// form but the pre-kernel one, where the template parameter has already removed the code that would
// derive these.
struct PreKRef {
  const uint16_t *kn;      // the k^ tile image, all sixty-four rows as the pre-kernel left them
  const float *kk, *kq;    // KKQ_FLOATS each, in the fragment order above
};

template <int NKS, int NP>
__device__ __forceinline__ void chain64_cols(float (&acc)[2 * NP][4], const MTile &at, const KTile &bt,
                                             int j0, int col0, int lane) {
  #pragma unroll
  for (int nt = 0; nt < 2 * NP; ++nt)
    #pragma unroll
    for (int e = 0; e < 4; ++e) acc[nt][e] = 0.f;
  #pragma unroll
  for (int ks = 0; ks < NKS; ++ks) {
    uint32_t a[4];
    frag_a(a, at, j0, 16 * ks, lane);
    #pragma unroll
    for (int np = 0; np < NP; ++np) {
      uint32_t b[4];
      frag_b2_t(b, bt, col0 + 16 * np, 16 * ks, lane);
      const uint32_t b0[2] = {b[0], b[1]}, b1[2] = {b[2], b[3]};
      mma(acc[2 * np], a, b0);
      mma(acc[2 * np + 1], a, b1);
    }
  }
}

// Shared-memory layout of one CTA. Two phases share k^, q^, M, the gates, w, u and S^; the phase-1
// scratch (v, xu, xw, L, X, P) and the phase-2 delta tiles overlay each other. 174 KB, so one CTA per
// SM -- which is what the measured mma.sync ceiling wants anyway: eight warps saturate the four tensor
// pipes of an SM, and a second CTA adds nothing but contention.
struct Slab {
  uint16_t dl[C * D];   // Tile<64, 128>: d~ [i][v], then the staged outputs o [j][v]
};

struct Smem {
  uint16_t k_[C * D];
  uint16_t q_[C * D];
  uint16_t m_[C * C];
  uint16_t w_[C * D];
  uint16_t u_[C * D];
  uint16_t sh_[D * D];    // Tile<128,128>: S^ [v][d], the image the record stores
  // The deltas [i][v]. OUTSIDE the phase-1 / phase-2 union on purpose, at a cost of 16 KB that the
  // 228 KB an SM has to give leaves free: the open chunk's delta rows come from the record, and if this
  // tile shared storage with the phase-1 scratch that copy could not be issued until the u chain had
  // finished reading v -- which put it on the critical path (a decode step at chunk row 40 spent 3,000
  // cycles waiting for 10 KB). Here it rides in the same commit group as v, X, P and S^ and is free.
  uint16_t dt_[C * D];
  GateTile gt_;
  float lg_[C];
  union {
    struct {
      uint16_t v[C * D];
      uint16_t xu[C * C];
      uint16_t xw[C * C];
      float L[C * LS];
      float X[C * C];
      float P[16 * C];
    } wy;
    Slab slab;
  } r;

  __device__ __forceinline__ KTile k() { return KTile{k_}; }
  __device__ __forceinline__ KTile q() { return KTile{q_}; }
  __device__ __forceinline__ MTile m() { return MTile{m_}; }
  __device__ __forceinline__ KTile w() { return KTile{w_}; }
  __device__ __forceinline__ KTile u() { return KTile{u_}; }
  __device__ __forceinline__ ShTile sh() { return ShTile{sh_}; }
  __device__ __forceinline__ KTile dt() { return KTile{dt_}; }
  __device__ __forceinline__ KTile v() { return KTile{r.wy.v}; }
  __device__ __forceinline__ MTile xu() { return MTile{r.wy.xu}; }
  __device__ __forceinline__ MTile xw() { return MTile{r.wy.xw}; }
  __device__ __forceinline__ float *L() { return r.wy.L; }
  __device__ __forceinline__ float *X() { return r.wy.X; }
  __device__ __forceinline__ float *P() { return r.wy.P; }
  __device__ __forceinline__ GateTile &gt() { return gt_; }
  __device__ __forceinline__ Slab &slab() { return r.slab; }
};

template <bool ALL, int NMT, bool PREK, class SM>
__device__ __forceinline__ void kk_kq_tiles(SM &sm, const ChunkGeo &g, const KTile &kt, const KTile &qt,
                                            int dj0, int di0, int lane, int gq, int tq,
                                            const PreKRef &pk) {
  if (!ALL && !(dj0 < g.r_hi && dj0 + 16 > g.r_lo)) return;
  GateTile &gt = sm.gt();
  float kk[2][2][4], kq[2][2][4];
  if constexpr (PREK) {   // the pre-kernel already evaluated both products for this key head
    #pragma unroll
    for (int mt_ = 0; mt_ < NMT; ++mt_) {
      kkq_load(kk[mt_], pk.kk, di0 + 16 * mt_, dj0, lane);
      kkq_load(kq[mt_], pk.kq, di0 + 16 * mt_, dj0, lane);
    }
  } else {
    dots128_pair<NMT>(kk, kq, kt, qt, di0, dj0, lane);
  }
  const MTile mt = sm.m();
  #pragma unroll
  for (int mt_ = 0; mt_ < NMT; ++mt_) {
    uint32_t mw[4];
    #pragma unroll
    for (int nt = 0; nt < 2; ++nt)
      #pragma unroll
      for (int eh = 0; eh < 2; ++eh) {
        float kqv[2], decv[2];
        bool liv[2];
        #pragma unroll
        for (int el = 0; el < 2; ++el) {
          const int e = 2 * eh + el;
          const int i = di0 + 16 * mt_ + gq + eh * 8, j = dj0 + 8 * nt + 2 * tq + el;
          const float dec = decay(gt.G[j], gt.G[i]);
          sm.L()[j * LS + i] = i < j ? l_entry(kk[mt_][nt][e], gt.beta[j], dec) : 0.f;
          // M_ji's own i <= j, passed as a FLAG: `dec` overflows to +inf for i > j and
          // 0.0f * inf is NaN, so the gate has to be inside the multiply (recipe.cuh).
          liv[el] = i <= j && (ALL || (j >= g.r_lo && j < g.r_hi));
          kqv[el] = kq[mt_][nt][e];
          decv[el] = dec;
        }
        mw[2 * nt + eh] = m_entry2(liv[0], kqv[0], decv[0], liv[1], kqv[1], decv[1]);
      }
    store_acc_t(mt, di0 + 16 * mt_, dj0, mw, lane);
  }
}

template <bool ALL, bool PREK = false, class SM>
__device__ __forceinline__ void kk_kq_deal(SM &sm, const ChunkGeo &g, const KTile &kt, const KTile &qt,
                                           int warp, int lane, int gq, int tq,
                                           const PreKRef &pk = PreKRef{nullptr, nullptr, nullptr}) {
  const int nmt = warp < 4 ? 2 : (warp < 6 ? 1 : 0);
  if (nmt == 0) return;                       // warps 6 and 7 have no tile
  const int dj0 = warp < 2 ? 48 : (warp == 2 ? 32 : (warp == 3 ? 16 : (warp == 4 ? 0 : 32)));
  const int di0 = (warp == 1 || warp == 5) ? 32 : 0;
  if (nmt == 2) kk_kq_tiles<ALL, 2, PREK>(sm, g, kt, qt, dj0, di0, lane, gq, tq, pk);
  else kk_kq_tiles<ALL, 1, PREK>(sm, g, kt, qt, dj0, di0, lane, gq, tq, pk);
}

template <int BL, bool ALL, class SM>
__device__ __forceinline__ void x_earlier_round(SM &sm, const ChunkGeo &g, int tid) {
  constexpr int NQ = 4 * BL;                  // float4 column quads of this round
  const int lo = ALL ? 16 * BL : max(16 * BL, g.r_lo);
  const int hi = ALL ? 16 * BL + 16 : min(16 * BL + 16, g.r_hi);
  if (lo >= hi) return;                       // uniform over the CTA: this round has no live row
  const int j = lo + tid / NQ, m4 = 4 * (tid % NQ);
  if (tid < (16 * BL + 16 - lo) * NQ) {
    float pv[4] = {0.f, 0.f, 0.f, 0.f};
    if (ALL || j < hi) {
      float a01[4], a2[4];
      #pragma unroll
      for (int gp = 0; gp < 4; ++gp) {        // T4 of chunks 4 gp .. 4 gp + 3, folded in tree order
        float t[4] = {0.f, 0.f, 0.f, 0.f};
        if (4 * gp < NQ) {
          float lv[16];                       // L_j. over this group's four chunks, issued up front
          #pragma unroll
          for (int cc = 0; cc < 4; ++cc) {
            const float4 l4 = *reinterpret_cast<const float4 *>(sm.L() + j * LS + 16 * gp + 4 * cc);
            lv[4 * cc] = l4.x;
            lv[4 * cc + 1] = l4.y;
            lv[4 * cc + 2] = l4.z;
            lv[4 * cc + 3] = l4.w;
          }
          float part[4][4];
          #pragma unroll
          for (int cc = 0; cc < 4; ++cc) {
            const int c = 4 * gp + cc;
            #pragma unroll
            for (int x = 0; x < 4; ++x) part[cc][x] = 0.f;
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
              const float4 x4 = *reinterpret_cast<const float4 *>(sm.X() + (4 * c + e) * C + m4);
              const float av = lv[4 * cc + e];
              part[cc][0] = ffma(av, x4.x, part[cc][0]);
              part[cc][1] = ffma(av, x4.y, part[cc][1]);
              part[cc][2] = ffma(av, x4.z, part[cc][2]);
              part[cc][3] = ffma(av, x4.w, part[cc][3]);
            }
          }
          #pragma unroll
          for (int x = 0; x < 4; ++x)
            t[x] = fadd(fadd(part[0][x], part[1][x]), fadd(part[2][x], part[3][x]));
        }
        #pragma unroll
        for (int x = 0; x < 4; ++x) {
          if (gp == 0) a01[x] = t[x];              // span 4 pair (0, 4) is (T4_0, T4_1)
          else if (gp == 1) a01[x] = fadd(a01[x], t[x]);
          else if (gp == 2) a2[x] = t[x];          // span 4 pair (8, 12) is (T4_2, T4_3)
          else pv[x] = fadd(a01[x], fadd(a2[x], t[x]));   // span 8 pair (0, 8)
        }
      }
    }
    *reinterpret_cast<float4 *>(sm.P() + (j - 16 * BL) * C + m4) =
        make_float4(pv[0], pv[1], pv[2], pv[3]);
  }
  __syncthreads();
  if (tid < (hi - lo) * NQ) {
    float y[16];                               // Y_j.: X_ji (i < j), 1, then zeros
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
      const float4 y4 = *reinterpret_cast<const float4 *>(sm.X() + j * C + 16 * BL + 4 * c);
      y[4 * c] = y4.x;
      y[4 * c + 1] = y4.y;
      y[4 * c + 2] = y4.z;
      y[4 * c + 3] = y4.w;
    }
    float part[4][4];                          // x_block's own four fma chunks, four columns at a time
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
      #pragma unroll
      for (int x = 0; x < 4; ++x) part[c][x] = 0.f;
      #pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float4 p4 = *reinterpret_cast<const float4 *>(sm.P() + (4 * c + e) * C + m4);
        const float av = y[4 * c + e];
        part[c][0] = ffma(av, p4.x, part[c][0]);
        part[c][1] = ffma(av, p4.y, part[c][1]);
        part[c][2] = ffma(av, p4.z, part[c][2]);
        part[c][3] = ffma(av, p4.w, part[c][3]);
      }
    }
    float res[4];
    #pragma unroll
    for (int x = 0; x < 4; ++x)
      res[x] = -fadd(fadd(part[0][x], part[1][x]), fadd(part[2][x], part[3][x]));
    *reinterpret_cast<float4 *>(sm.X() + j * C + m4) = make_float4(res[0], res[1], res[2], res[3]);
  }
  __syncthreads();
}

template <bool SAVE, bool ALL, bool DEAL = true, bool WIDTHS = true, bool PREP = false, bool PREK = false,
          class SM = Smem>
__device__ void wy_chunk(const Params &p, SM &sm, const ChunkGeo &g, int h, int si, int so,
                        bool load_sh, CarryRef cy = CarryRef{nullptr, nullptr, nullptr, nullptr},
                        PreKRef pk = PreKRef{nullptr, nullptr, nullptr}) {
  // The pre-kernel form is only ever instantiated with the ten-tile deal, which is what the deployment
  // rule runs; the plan-0 deal touches all sixteen (i-block, j-block) tiles including the six the
  // ten-tile deal proves are +0, and giving it fragments would mean the pre-kernel publishing those six.
  static_assert(!PREK || DEAL, "the (chunk, key head) pre-kernel form needs the ten-tile deal");
  static_assert(!PREK || PREP, "the pre-kernel form exists only inside the prepared two-launch mode");
  static_assert(!PREK || !SAVE, "the trainer save path never takes the prepared mode");
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int gq = lane >> 2, tq = lane & 3;
  const int hq = p.key_head(h);
  const bool leader = p.key_leader(h);
  const bool keep_open = !PREP && !ALL && so >= 0 && g.last() && !g.complete();
  // The same predicate for the prepared mode, whose destination is the carry and not the record.
  // `!g.complete()` already implies `!ALL`, so the ALL instantiation compiles this away.
  const bool keep_carry = PREP && !ALL && so >= 0 && g.last() && !g.complete();
  const OpenRows open = open_rows(g, si, so);
  const KTile kt = sm.k(), qt = sm.q(), vt = sm.v();
  GateTile &gt = sm.gt();

  // ------------------------------------------------------------------ 1. rows into shared memory
  uint16_t a_raw = 0, b_raw = 0;
  float A_log = 0.f, dt_bias = 0.f;
  const bool gate_row = ALL ? tid < C : (tid >= g.r_lo && tid < g.r_hi);   // tid < C: row i = tid
  if (gate_row) {
    const int64_t t = g.token(tid);
    a_raw = p.a[t * p.as0 + h * p.as1];
    b_raw = p.b[t * p.bs0 + h * p.bs1];
    A_log = p.A_log_of(h);
    dt_bias = p.dt_bias_of(h);
  }
  // Rows [0, r_lo) come from the input slot's record. Its bf16 sections are byte images of these tiles
  // and its fp32 sections are plain row-major, so every source is the destination's own address map and
  // one 16-byte copy per chunk serves both. Rows the call does not evaluate are zero-filled by the same
  // instruction (`valid == false`), which is the recipe's zero extension on the load path.
  const int p_old = (!ALL && g.r_lo >= 16) ? (g.r_lo & 15) : 0;
  const int rmax = ALL ? C : ((g.r_hi + 15) & ~15);
  auto ld_k = [&](int x) {
    const int i = x >> 4, c = x & 15;
    const bool old = !ALL && i < g.r_lo, cur = ALL || (i >= g.r_lo && i < g.r_hi);
    const int64_t t = cur ? g.token(i) : 0;
    cp16(smem_u32(kt.chunk(i, c)),
         old ? (const void *)KTile{p.st.kb(si, h)}.chunk(i, c) : (const void *)(p.k + t * p.ks0 + hq * p.ks1 + c * 8),
         old || cur);
  };
  auto ld_q = [&](int x) {
    const int i = g.r_lo + (x >> 4), c = x & 15;
    cp16(smem_u32(qt.chunk(i, c)), p.q + g.token(i) * p.qs0 + hq * p.qs1 + c * 8, true);
  };
  auto ld_v = [&](int x) {
    const int i = x >> 4, c = x & 15;
    const bool old = !ALL && i < g.r_lo, cur = ALL || (i >= g.r_lo && i < g.r_hi);
    const int64_t t = cur ? g.token(i) : 0;
    cp16(smem_u32(vt.chunk(i, c)),
         old ? (const void *)KTile{p.st.vb(si, h)}.chunk(i, c) : (const void *)(p.v + t * p.vs0 + h * p.vs1 + c * 8),
         old || cur);
  };
  if constexpr (PREK) {
    // ALL SIXTY-FOUR ROWS OF THE PRE-KERNEL'S `k^` TILE, verbatim, stale rows included -- the same
    // argument the (chunk, value head) hand-off rests on. Rows in [r_hi, rmax) were zero-filled by the
    // pre-kernel's own load and rows at or past rmax are read by no chain, so which bits they hold does
    // not enter the arithmetic. `q^` is not loaded: its only reader was the KK / KQ products, and those
    // arrive as fragments.
    const KTile src{const_cast<uint16_t *>(pk.kn)};
    static_tasks<C * 16>(tid, [&](int x) {
      const int i = x >> 4, c = x & 15;
      cp16(smem_u32(kt.chunk(i, c)), src.chunk(i, c), true);
    });
    cp_commit();
    if (WIDTHS && ALL) static_tasks<C * 16>(tid, ld_v);
    else for (int x = tid; x < rmax * 16; x += THREADS) ld_v(x);
  } else if (WIDTHS && ALL) {
    static_tasks<C * 16>(tid, ld_k);
    static_tasks<C * 16>(tid, ld_q);
    cp_commit();
    static_tasks<C * 16>(tid, ld_v);
  } else {
    for (int x = tid; x < rmax * 16; x += THREADS) ld_k(x);
    for (int x = tid; x < (ALL ? C : g.r_hi - g.r_lo) * 16; x += THREADS) ld_q(x);
    cp_commit();
    for (int x = tid; x < rmax * 16; x += THREADS) ld_v(x);
  }
  for (int x = tid; x < (ALL ? 0 : g.r_lo * 16); x += THREADS) {
    const int i = x >> 4, c = x & 15;
    cp16(smem_u32(sm.X() + i * C + c * 4), p.st.xb(si, h) + i * C + c * 4, true);
  }
  for (int x = tid; x < p_old * 16; x += THREADS) {
    const int i = x >> 4, c = x & 15;
    cp16(smem_u32(sm.P() + i * C + c * 4), p.st.pb(si, h) + i * C + c * 4, true);
  }
  // THE DELTA AND S^ TILES ARE PHASE 2'S AND THEY DO NOT EXIST IN THE PREPARED MODE'S PHASE-1 LAYOUT.
  // Launch A evaluates phase 1 alone: the record's delta rows are pulled in by launch B's `load_hand`
  // and its S^ by launch B itself, so `PrepASmem` has neither tile and these two copies are compiled
  // out. That is 48 KB of the 82 KB launch A had to give back to get two CTAs per SM.
  if constexpr (!PREP) {
    // The open chunk's delta rows, which the M d chain of phase 2 reads exactly as the record left them.
    for (int x = tid; x < (ALL ? 0 : g.r_lo * 16); x += THREADS) {
      const int i = x >> 4, c = x & 15;
      cp16(smem_u32(sm.dt().chunk(i, c)), KTile{p.st.db(si, h)}.chunk(i, c), true);
    }
    // S^ straight out of the record: its section is the byte image of this tile, so the copy is the
    // destination's own address map. When the fp32 state is coming into registers instead, slab_step
    // writes every byte of this tile itself and nothing is moved here.
    auto ld_sh = [&](int x) {
      const int v = x >> 4, c = x & 15;
      cp16(smem_u32(sm.sh().chunk(v, c)), ShTile{p.st.sh(si, h)}.chunk(v, c), true);
    };
    if (load_sh && WIDTHS) static_tasks<D * 16>(tid, ld_sh);
    else if (load_sh)
      for (int x = tid; x < D * 16; x += THREADS) {
        const int v = x >> 4, c = x & 15;
        cp16(smem_u32(sm.sh().chunk(v, c)), ShTile{p.st.sh(si, h)}.chunk(v, c), true);
      }
  }
  cp_commit();
  if (tid < C) {
    const int i = tid;
    float G = 0.f, beta = 0.f, lg = 0.f;
    if (i < g.r_lo) {
      G = p.st.gb(si, h)[i];
      beta = p.st.bb(si, h)[i];
    } else if (gate_row) {
      const Gate gg = gate(bf(a_raw), bf(b_raw), A_log, dt_bias);
      lg = gg.lg;
      beta = gg.beta;
    }
    gt.G[i] = G;
    gt.beta[i] = beta;
    sm.lg_[i] = lg;
  }
  cp_wait<1>();   // k^ and q^ only; v, X, P and S^ are still in flight behind them
  __syncthreads();

  // ------------------------------------------------------------------ 2. k^, q^ and G
  // Each row's FOLD32 is a butterfly over the 32 lanes holding THAT row's 128 elements, so rows are
  // independent and WHICH warp folds a row does not enter the bits. The call's live rows [r_lo, r_hi) are
  // therefore dealt round robin -- warp w takes r_lo + 8 rr + w -- and a warp folds only the rows it was
  // dealt. A full chunk deals every warp eight rows, which is the shape this phase always had; a one-row
  // call deals one warp one row instead of making one warp fold eight and throw seven away.
  // Rows before r_lo arrive already normalized from the record and rows at or past r_hi are never folded.
  const int nlive = ALL ? C : g.r_hi - g.r_lo;
  const int mine = ALL ? C / 8 : (nlive > warp ? ((nlive - warp + 7) >> 3) : 0);
  #pragma unroll
  for (int which = 0; which < 2 && !PREK; ++which) {   // PREK: the pre-kernel normalized both rows
    if (mine == 0) break;
    const KTile t = which ? qt : kt;
    float x[C / 8][CE];
    uint2 *e[C / 8];
    #pragma unroll
    for (int rr = 0; rr < C / 8; ++rr) {
      if (!ALL && rr >= mine) break;
      e[rr] = reinterpret_cast<uint2 *>(t.chunk(g.r_lo + 8 * rr + warp, lane >> 1) + 4 * (lane & 1));
      const uint2 raw = *e[rr];
      x[rr][0] = bf(lo16(raw.x));
      x[rr][1] = bf(hi16(raw.x));
      x[rr][2] = bf(lo16(raw.y));
      x[rr][3] = bf(hi16(raw.y));
    }
    uint2 nrm[C / 8];
    float inverse_norms[C / 8];
    if (ALL || mine == C / 8) l2norm_rows<C / 8, false>(x, p.scale, which == 1, nrm, SAVE ? inverse_norms : nullptr);
    else l2norm_rows<C / 8, true>(x, p.scale, which == 1, nrm, SAVE ? inverse_norms : nullptr, mine);
    #pragma unroll
    for (int rr = 0; rr < C / 8; ++rr) {
      if (!ALL && rr >= mine) break;
      *e[rr] = nrm[rr];
    }
    if (SAVE && leader) save_norms(p, g, hq, which == 1, x, nrm, inverse_norms, g.r_lo + warp, mine, lane);
  }
  if (tid == 0) {   // G_i = G_{i-1} (+) lg_i: one serial chain, over the call's rows only
    float G = g.r_lo > 0 ? gt.G[g.r_lo - 1] : 0.f;
    if (ALL || (g.r_lo == 0 && g.r_hi == C)) {   // the whole chunk: one chain, all loads up front
      float lg[C];
      #pragma unroll
      for (int i = 0; i < C; ++i) lg[i] = sm.lg_[i];
      #pragma unroll
      for (int i = 0; i < C; ++i) {
        G = cum_gate(G, lg[i]);
        gt.G[i] = G;
      }
    } else
    for (int base = g.r_lo & ~7; base < g.r_hi; base += 8) {
      float lg[8];   // eight loads issued together, then an eight-long dependent chain
      #pragma unroll
      for (int t = 0; t < 8; ++t) lg[t] = sm.lg_[base + t];
      #pragma unroll
      for (int t = 0; t < 8; ++t) {
        const int i = base + t;
        if (i >= g.r_lo && i < g.r_hi) {
          G = cum_gate(G, lg[t]);
          gt.G[i] = G;
        }
      }
    }
  }
  fence_async_smem();
  __syncthreads();
  if (keep_open) {   // the output slot's open k^ rows, G and beta
    for (int x = tid; x < (open.jf - open.r_first) * 16; x += THREADS) {
      const int i = open.r_first + (x >> 4), c = x & 15;
      *reinterpret_cast<uint4 *>(KTile{p.st.kb(so, h)}.chunk(i, c)) =
          *reinterpret_cast<const uint4 *>(kt.chunk(i, c));
    }
    if (tid >= open.r_first && tid < open.jf) {
      p.st.gb(so, h)[tid] = gt.G[tid];
      p.st.bb(so, h)[tid] = gt.beta[tid];
    }
  }
  if (keep_carry && tid < C) {   // all sixty-four, as launch A's `publish` used to: launch B selects
    cy.G[tid] = gt.G[tid];
    cy.beta[tid] = gt.beta[tid];
  }

  // ------------------------------------------------------------------ 3. decay factors; KK -> L, KQ -> M
  if (tid < C) {
    const int i = tid;
    const bool present = ALL || i < g.r_hi;
    gt.gam[i] = present ? decay0(gt.G[i]) : 0.f;
    gt.lam[i] = present && (ALL || g.complete()) ? decay(gt.G[C - 1], gt.G[i]) : 0.f;
  }
  __syncthreads();
  if (DEAL) {
    kk_kq_deal<ALL, PREK>(sm, g, kt, qt, warp, lane, gq, tq, pk);
  } else {
    const int dj0 = 16 * (warp & 3), di0 = 32 * (warp >> 2);
    const bool dneed = (ALL || (dj0 < g.r_hi && dj0 + 16 > g.r_lo)) && di0 < dj0 + 16;
    // kk / kq are read only through the `dneed && ...` selects below, and a ternary evaluates only the
    // branch it takes, so they are never read uninitialized. Zero-initializing them instead costs ~80
    // registers, which is what a 249-register kernel cannot afford.
    float kk[2][2][4], kq[2][2][4];
    if (dneed) dots128_pair<2>(kk, kq, kt, qt, di0, dj0, lane);
    if (dneed) {
      const MTile mt = sm.m();
      #pragma unroll
      for (int mt_ = 0; mt_ < 2; ++mt_) {
        uint32_t mw[4];
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt)
          #pragma unroll
          for (int eh = 0; eh < 2; ++eh) {
            float kqv[2], decv[2];
            bool liv[2];
            #pragma unroll
            for (int el = 0; el < 2; ++el) {
              const int e = 2 * eh + el;
              const int i = di0 + 16 * mt_ + gq + eh * 8, j = dj0 + 8 * nt + 2 * tq + el;
              const float dec = decay(gt.G[j], gt.G[i]);
              sm.L()[j * LS + i] = i < j ? l_entry(kk[mt_][nt][e], gt.beta[j], dec) : 0.f;
              // M_ji's own i <= j, passed as a FLAG: `dec` overflows to +inf for i > j and
              // 0.0f * inf is NaN, so the gate has to be inside the multiply (recipe.cuh).
              liv[el] = i <= j && (ALL || (j >= g.r_lo && j < g.r_hi));
              kqv[el] = kq[mt_][nt][e];
              decv[el] = dec;
            }
            mw[2 * nt + eh] = m_entry2(liv[0], kqv[0], decv[0], liv[1], kqv[1], decv[1]);
          }
        store_acc_t(mt, di0 + 16 * mt_, dj0, mw, lane);
      }
    }
  }
  __syncthreads();

  // ------------------------------------------------------------------ 4. X = (I + L)^-1 by 16-row blocks
  cp_wait<0>();   // group 2: the X and P rows, v and S^
  __syncthreads();
  // X_jm = +0 for m at or past the row's OWN 16-column block, and this loop is the only writer of those
  // zeros: the diagonal solve below writes a row's own block and the earlier-block step writes only
  // columns beneath it, while the P sum `T16_c CH4_c(L_j., X_.m)` reads X row i at every column m < 16 bl
  // -- including the columns above row i's block. (The record's rows arrive with those zeros already: a
  // saved X row is all 64 columns.) The load path used to supply them by zero-filling every row it did
  // not read, which cost a 16 KB shared-memory fill per chunk to place at most 48 floats a row.
  for (int x = tid; x < nlive * 16; x += THREADS) {
    const int j = (ALL ? 0 : g.r_lo) + (x >> 4), m = 4 * (x & 15);
    if (m >= ((j & ~15) + 16))
      *reinterpret_cast<float4 *>(sm.X() + j * C + m) = make_float4(0.f, 0.f, 0.f, 0.f);
  }
  if (tid < C) {   // the four diagonal blocks at once: thread (block bq = tid >> 4, column m = tid)
    const int bq = tid >> 4, m = tid;
    float xb[16];
    #pragma unroll
    for (int r = 0; r < 16; ++r) xb[r] = sm.X()[(16 * bq + r) * C + m];
    #pragma unroll
    for (int r = 0; r < 16; ++r) {
      const int j = 16 * bq + r;
      if (!ALL && (j < g.r_lo || j >= g.r_hi)) continue;   // the record's row, already in xb[r]
      float l[16];
      #pragma unroll
      for (int c = 0; c < 4; ++c) {
        const float4 l4 = *reinterpret_cast<const float4 *>(sm.L() + j * LS + 16 * bq + 4 * c);
        l[4 * c] = l4.x;
        l[4 * c + 1] = l4.y;
        l[4 * c + 2] = l4.z;
        l[4 * c + 3] = l4.w;
      }
      // In a new row, L's diagonal and upper triangle and the still-unwritten xb tail are +0. Only the
      // nonzero-prefix FMAs are evaluated, retaining x_block's four-leaf tree; the first omitted product
      // still contributes an explicit +0 add, because a tiny negative partial may have rounded to -0.
      float part[4];
      #pragma unroll
      for (int c = 0; c < 4; ++c) {
        part[c] = 0.f;
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
          if (4 * c + e < r) part[c] = ffma(l[4 * c + e], xb[4 * c + e], part[c]);
          else if (4 * c + e == r) part[c] = fadd(part[c], 0.f);
        }
      }
      const float x = -fadd(fadd(part[0], part[1]), fadd(part[2], part[3]));
      xb[r] = m < j ? x : (m == j ? 1.f : 0.f);
    }
    #pragma unroll
    for (int r = 0; r < 16; ++r) {
      const int j = 16 * bq + r;
      if (ALL || (j >= g.r_lo && j < g.r_hi)) sm.X()[j * C + m] = xb[r];
    }
  }
  __syncthreads();
  if (WIDTHS) {   // block bl needs block bl - 1, so the three rounds are sequential by the recipe
    x_earlier_round<1, ALL>(sm, g, tid);
    x_earlier_round<2, ALL>(sm, g, tid);
    x_earlier_round<3, ALL>(sm, g, tid);
  } else
  for (int bl = 1; bl < 4; ++bl) {   // earlier-block entries of block bl: P, then X = -X_bb P
    const int lo = ALL ? 16 * bl : max(16 * bl, g.r_lo), hi = ALL ? 16 * bl + 16 : min(16 * bl + 16, g.r_hi);
    if (lo >= hi) continue;   // uniform over the CTA
    const int nq = 4 * bl;
    const int task = tid;
    const int j = lo + task / nq, m4 = 4 * (task % nq);
    if (task < (16 * bl + 16 - lo) * nq) {
      float pv[4] = {0.f, 0.f, 0.f, 0.f};
      if (ALL || j < hi) {
        float part[16][4];
        #pragma unroll
        for (int c = 0; c < 16; ++c) {
          #pragma unroll
          for (int x = 0; x < 4; ++x) part[c][x] = 0.f;
          if (c < nq) {
            const float4 l4 = *reinterpret_cast<const float4 *>(sm.L() + j * LS + 4 * c);
            const float lv[4] = {l4.x, l4.y, l4.z, l4.w};
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
              const float4 x4 = *reinterpret_cast<const float4 *>(sm.X() + (4 * c + e) * C + m4);
              part[c][0] = ffma(lv[e], x4.x, part[c][0]);
              part[c][1] = ffma(lv[e], x4.y, part[c][1]);
              part[c][2] = ffma(lv[e], x4.z, part[c][2]);
              part[c][3] = ffma(lv[e], x4.w, part[c][3]);
            }
          }
        }
        #pragma unroll
        for (int x = 0; x < 4; ++x) {   // T16 over the chunks, four columns at once
          #pragma unroll
          for (int span = 1; span < 16; span *= 2)
            #pragma unroll
            for (int c = 0; c + span < 16; c += 2 * span) part[c][x] = fadd(part[c][x], part[c + span][x]);
          pv[x] = part[0][x];
        }
      }
      *reinterpret_cast<float4 *>(sm.P() + (j - 16 * bl) * C + m4) = make_float4(pv[0], pv[1], pv[2], pv[3]);
    }
    __syncthreads();
    if (task < (hi - lo) * nq) {
      float y[16];
      #pragma unroll
      for (int c = 0; c < 4; ++c) {   // Y_j.: X_ji (i < j), 1, then zeros
        const float4 y4 = *reinterpret_cast<const float4 *>(sm.X() + j * C + 16 * bl + 4 * c);
        y[4 * c] = y4.x;
        y[4 * c + 1] = y4.y;
        y[4 * c + 2] = y4.z;
        y[4 * c + 3] = y4.w;
      }
      float pc[4][16];
      #pragma unroll
      for (int r = 0; r < 16; ++r) {
        const float4 p4 = *reinterpret_cast<const float4 *>(sm.P() + r * C + m4);
        pc[0][r] = p4.x;
        pc[1][r] = p4.y;
        pc[2][r] = p4.z;
        pc[3][r] = p4.w;
      }
      float res[4];
      #pragma unroll
      for (int x = 0; x < 4; ++x) res[x] = x_block(y, pc[x]);
      *reinterpret_cast<float4 *>(sm.X() + j * C + m4) = make_float4(res[0], res[1], res[2], res[3]);
    }
    __syncthreads();
  }
  if (keep_open) {   // the output slot keeps the open X rows and the open block's P rows
    for (int x = tid; x < (open.jf - open.r_first) * 16; x += THREADS) {
      const int i = open.r_first + (x >> 4), c = x & 15;
      *reinterpret_cast<float4 *>(p.st.xb(so, h) + i * C + c * 4) =
          *reinterpret_cast<const float4 *>(sm.X() + i * C + c * 4);
    }
    if ((open.jf & ~15) > 0) {
      const int pb0 = open.jf & ~15, r0 = max(open.r_first, pb0);
      for (int x = tid; x < (open.jf - r0) * 16; x += THREADS) {
        const int i = r0 + (x >> 4), c = x & 15;
        *reinterpret_cast<float4 *>(p.st.pb(so, h) + (i - pb0) * C + c * 4) =
            *reinterpret_cast<const float4 *>(sm.P() + (i - pb0) * C + c * 4);
      }
    }
  }

  // THE PREPARED MODE'S X AND P CARRY, HERE AND NOT IN `publish`. `u` and `w` overlay `L`/`X`/`P` in the
  // phase-1-only layout, so this is the last point at which the bytes still exist. Launch B reads only
  // the open rows out of it; all sixty-four are copied, exactly as `publish` used to, so the selection
  // rule stays in one place.
  if (keep_carry) {
    for (int x = tid; x < C * 16; x += THREADS) {
      const int i = x >> 4, c = x & 15;
      *reinterpret_cast<float4 *>(cy.X + i * C + c * 4) =
          *reinterpret_cast<const float4 *>(sm.X() + i * C + c * 4);
    }
    for (int x = tid; x < 16 * 16; x += THREADS) {
      const int i = x >> 4, c = x & 15;
      *reinterpret_cast<float4 *>(cy.P + i * C + c * 4) =
          *reinterpret_cast<const float4 *>(sm.P() + i * C + c * 4);
    }
  }

  // ------------------------------------------------------------------ 5. xu, xw rows
  {
    // Only the call's own rows: the u / w store selects a row outside [r_lo, r_hi) away to +0 whatever
    // its coefficients were, so those rows of xu / xw are never consumed and are left as they lie.
    const MTile xut = sm.xu(), xwt = sm.xw();
    // A full chunk is 512 tasks over 256 threads, exactly two. The trip count is therefore made
    // COMPILE-TIME: this loop measured 1,146 cycles for ninety-two instructions, because with a runtime
    // bound its two iterations ran one after the other and each waited out its own LDS.128 latency on two
    // warps per sub-partition. Unrolled, the two iterations' six independent shared loads issue together.
    auto row = [&](int x) {
      const int j = (ALL ? 0 : g.r_lo) + (x >> 3), c = x & 7;
      uint32_t pu[4], pw[4];
      const float *r = sm.X() + j * C + c * 8;
      #pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int i = 8 * c + 2 * e;
        pu[e] = xu_entry2(r[2 * e], gt.beta[i], r[2 * e + 1], gt.beta[i + 1]);
        pw[e] = xw_entry2(r[2 * e], gt.beta[i], gt.gam[i], r[2 * e + 1], gt.beta[i + 1], gt.gam[i + 1]);
      }
      if (SAVE) {   // FLA's A = (I + L)^-1 row j, bf16
        uint32_t pa[4];
        #pragma unroll
        for (int e = 0; e < 4; ++e) pa[e] = rn_pair(r[2 * e], r[2 * e + 1]);
        *reinterpret_cast<uint4 *>(p.sv.A + (g.token(j) * p.hv + h) * C + c * 8) =
            make_uint4(pa[0], pa[1], pa[2], pa[3]);
      }
      *reinterpret_cast<uint4 *>(xut.chunk(j, c)) = make_uint4(pu[0], pu[1], pu[2], pu[3]);
      *reinterpret_cast<uint4 *>(xwt.chunk(j, c)) = make_uint4(pw[0], pw[1], pw[2], pw[3]);
    };
    if (WIDTHS && ALL) static_tasks<C * 8>(tid, row);
    else for (int x = tid; x < nlive * 8; x += THREADS) row(x);
  }
  fence_async_smem();
  __syncthreads();

  {
    // The j-block a warp takes is dealt 0,1,2,3,3,2,1,0 rather than w & 3, so that warps w and w + 4 --
    // which share a sub-partition -- carry complementary k-step counts. `nks` is the j-block plus one, so
    // the old deal gave one sub-partition 1 + 1 steps and another 4 + 4; this one gives every
    // sub-partition five. All eight (j-block, column-half) pairs are still covered exactly once, and which
    // warp evaluates a contraction does not enter its bits.
    const int j0 = 16 * (warp < 4 ? warp : 7 - warp), col0 = 64 * (warp >> 2);
    const int jr = j0 + gq;
    const bool c0 = ALL || (jr >= g.r_lo && jr < g.r_hi), c1 = ALL || (jr + 8 >= g.r_lo && jr + 8 < g.r_hi);
    const bool jneed = ALL || (j0 < g.r_hi && j0 + 16 > g.r_lo);   // this warp's j block holds a live row
    #pragma unroll
    for (int which = 0; which < 2; ++which) {
      float acc[8][4];
      #pragma unroll
      for (int nt = 0; nt < 8; ++nt)
        #pragma unroll
        for (int e = 0; e < 4; ++e) acc[nt][e] = 0.f;
      if (jneed) switch (j0 >> 4) {   // NKS = (j0 >> 4) + 1, warp-uniform, so it is a template argument
        case 0: chain64_cols<1, 4>(acc, which ? sm.xu() : sm.xw(), which ? sm.v() : sm.k(), j0, col0, lane); break;
        case 1: chain64_cols<2, 4>(acc, which ? sm.xu() : sm.xw(), which ? sm.v() : sm.k(), j0, col0, lane); break;
        case 2: chain64_cols<3, 4>(acc, which ? sm.xu() : sm.xw(), which ? sm.v() : sm.k(), j0, col0, lane); break;
        default: chain64_cols<4, 4>(acc, which ? sm.xu() : sm.xw(), which ? sm.v() : sm.k(), j0, col0, lane); break;
      }
      if (!ALL && !jneed) continue;   // nothing reads u / w at a row this call does not evaluate
      const KTile dst = which ? sm.u() : sm.w();
      #pragma unroll
      for (int np = 0; np < 4; ++np) {
        uint32_t w[4];
        #pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          const float *a = acc[2 * np + hh];
          // Both elements of a pair share the row test (c0 is rows gq, c1 is rows gq + 8), so the pair
          // narrows in one instruction and a dead row enters it as +0.0f, whose rn IS 0x0000.
          w[2 * hh] = rn_pair(c0 ? a[0] : 0.0f, c0 ? a[1] : 0.0f);
          w[2 * hh + 1] = rn_pair(c1 ? a[2] : 0.0f, c1 ? a[3] : 0.0f);
        }
        store_acc(dst, j0, col0 + 16 * np, w, lane);
      }
    }
  }
  fence_async_smem();
  __syncthreads();

  if (SAVE) {   // the values and the w rows as FLA lays them out
    for (int x = tid; x < (ALL ? C : g.r_hi - g.r_lo) * 16; x += THREADS) {
      const int j = (ALL ? 0 : g.r_lo) + (x >> 4), c = x & 15;
      const int64_t row = g.token(j) * p.hv + h;
      *reinterpret_cast<uint4 *>(p.sv.v + row * D + c * 8) = *reinterpret_cast<const uint4 *>(vt.chunk(j, c));
      if (p.sv.w)
        *reinterpret_cast<uint4 *>(p.sv.w + row * D + c * 8) =
            *reinterpret_cast<const uint4 *>(sm.w().chunk(j, c));
    }
  }
}

}  // namespace gdn_cpr_sm100
