// SPDX-License-Identifier: Apache-2.0
// The pass-1 work of one (chunk, value head): every per-row quantity of recipe.cuh that does not depend on
// the boundary state -- k^, q^, gates, G, L, X = (I + L)^-1, u, w, M -- into shared-memory tiles, by 256
// threads. Two users:
//   wy.cu     the two-pass prefill's pass 1 (one CTA per (chunk, head)): the tiles then go to the workspace;
//   fused.cu  the fused prefill (one CTA walks a (sequence, head)'s chunks): the tiles stay in shared memory
//             for the chunk's state step and outputs, and the output slot's open-chunk rows of the
//             sequence's last chunk (k^, X, P rows; G, beta) go straight to the record.
//
// Rows [0, r_lo) of a continuing sequence's first chunk come from the input slot's open-chunk buffer
// (k^, v, X rows, P rows, G, beta); rows [r_lo, r_hi) are this call's tokens. SAVE (trainer mode) also
// writes the backward's tensors of prefill.cuh's SaveParams (qn, kn, inverse norms, v, A, w).
#pragma once
#include "prefill.cuh"
#include "tile.cuh"

namespace gdn_cpr {

constexpr int WY_THREADS = 256;
constexpr int LS = 68;   // L row stride in floats (16-byte aligned rows, conflict-free fragment stores)

// Trainer mode: FLA's normalized rows (queries without the scale) and inverse norms, rows 8 rr + warp.
__device__ __forceinline__ void save_norms(const PrefillParams &p, const ChunkGeo &g, int hq, bool query,
                                           const float (&x)[C / 8][CE], const uint2 (&nrm)[C / 8], const float (&inverse_norms)[C / 8], int warp, int lane) {
  #pragma unroll
  for (int rr = 0; rr < C / 8; ++rr) {
    const float inv = inverse_norms[rr];
    const int i = 8 * rr + warp;
    if (i < g.r_lo || i >= g.r_hi) continue;
    const int64_t row = g.token(i) * p.hk + hq;
    uint2 out = nrm[rr];
    if (query)
      out = make_uint2(pack(rn(fmul(x[rr][0], inv)), rn(fmul(x[rr][1], inv))),
                       pack(rn(fmul(x[rr][2], inv)), rn(fmul(x[rr][3], inv))));
    *reinterpret_cast<uint2 *>((query ? p.sv.qn : p.sv.kn) + row * D + 4 * lane) = out;
    if (lane == 0) (query ? p.sv.q_rstd : p.sv.k_rstd)[row] = inv;
  }
}

// The i-pair x j-pair block of a 64 x 64 product CHAIN128(k^_i, x_j) owned by warp w: rows
// i0 + 16 mt, columns j0 + 8 nt; four independent 8-step chains interleaved.
__device__ __forceinline__ void dots(float (&acc)[2][2][4], const Wide<64> &a_rows, const Wide<64> &b_rows, int i0,
                                     int j0, int lane) {
  #pragma unroll
  for (int mt = 0; mt < 2; ++mt)
    #pragma unroll
    for (int nt = 0; nt < 2; ++nt)
      #pragma unroll
      for (int e = 0; e < 4; ++e) acc[mt][nt][e] = 0.f;
  #pragma unroll
  for (int ks = 0; ks < 8; ++ks) {
    uint32_t b[4];
    frag_b2(b, b_rows, j0, 16 * ks, lane);
    const uint32_t b0[2] = {b[0], b[1]}, b1[2] = {b[2], b[3]};
    #pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
      uint32_t a[4];
      frag_a(a, a_rows, i0 + 16 * mt, 16 * ks, lane);
      mma(acc[mt][0], a, b0);
      mma(acc[mt][1], a, b1);
    }
  }
}

// Chunk g (of sequence g.n, dense chunk s) at value head h; si = the sequence's input slot. `sm` names
// the tiles (the layouts are the users'): k, q, v [i][d] Wide<64>; m, xu, xw Tile<64>; L, X, P fp32; w
// (Wide<64>), u (Tile<64>, 128 rows); gt; lg. With FUSED the M and xu tiles must not overlay q^ (it is read
// by the outputs), and w, u may overlay L, X, P as in pass 1. Ends with the tiles complete for every thread
// and visible to the async proxy.
template <bool SAVE, bool FUSED, class Sm>
__device__ __forceinline__ void wy_chunk(const PrefillParams &p, Sm &sm, const ChunkGeo &g, int s, int h, int si) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int gq = lane >> 2, tq = lane & 3;
  const int n = g.n, hq = p.key_head(h), so = p.slot_out_of(n);
  const bool leader = p.key_leader(h);
  const bool keep_open = so >= 0 && g.last() && !g.complete();   // the output slot keeps open rows of this chunk
  const OpenRows open = open_rows(g, si, so);
  const Wide<64> kt{sm.k()}, qt{sm.q()}, vt{sm.v()};
  GateTile &gt = sm.gt();

  // ---------------------------------------------------------------- 1. rows -> shared memory
  // The gate inputs first: scalar loads issued behind the row copies wait for them in the memory pipe.
  uint16_t a_raw = 0, b_raw = 0;
  float A_log = 0.f, dt_bias = 0.f;
  const bool gate_row = tid >= g.r_lo && tid < g.r_hi;   // tid < C: row i = tid of this call
  if (gate_row) {
    const int64_t t = g.token(tid);
    a_raw = p.a[t * p.as0 + h * p.as1];
    b_raw = p.b[t * p.bs0 + h * p.bs1];
    A_log = p.A_log_of(h);
    dt_bias = p.dt_bias_of(h);
  }
  for (int x = tid; x < C * 16; x += WY_THREADS) {
    const int i = x >> 4, c = x & 15;
    const uint16_t *ks = p.k, *qs = p.q, *vs = p.v;
    bool old = false, cur = false;
    if (i < g.r_lo) {   // v of old rows is value-major in the record: transposed below
      old = true;
      ks = p.st.kb_tile(si, h).chunk(i, c);
    } else if (i < g.r_hi) {
      cur = true;
      const int64_t t = g.token(i);
      ks = p.k + t * p.ks0 + hq * p.ks1 + c * 8;
      qs = p.q + t * p.qs0 + hq * p.qs1 + c * 8;
      vs = p.v + t * p.vs0 + h * p.vs1 + c * 8;
    }
    cp16(smem_u32(kt.chunk(i, c)), ks, old || cur);
    cp16(smem_u32(qt.chunk(i, c)), qs, cur);   // queries of earlier rows are never needed
  }
  for (int x = tid; x < g.r_lo * D; x += WY_THREADS) {   // the open chunk's v [v][i] -> rows [i][v]
    const int i = x % g.r_lo, v = x / g.r_lo;
    vt.at(i, v) = p.st.vb_tile(si, h).at(v, i);
  }
  // X and P rows of the open chunk from the buffer, zeros after them (the zero extension of recipe.cuh
  // needs finite operands; the zeros are plain stores, not copies)
  for (int x = tid; x < C * 16; x += WY_THREADS) {
    if (x < g.r_lo * 16) cp16(smem_u32(sm.X() + x * 4), p.st.xb(si, h) + x * 4, true);
    else *reinterpret_cast<float4 *>(sm.X() + x * 4) = make_float4(0.f, 0.f, 0.f, 0.f);
  }
  const int p_old = g.r_lo >= 16 ? g.r_lo & 15 : 0;   // P rows exist only past the first block
  for (int x = tid; x < 16 * 16; x += WY_THREADS) {
    if (x < p_old * 16) cp16(smem_u32(sm.P() + x * 4), p.st.pb(si, h) + x * 4, true);
    else *reinterpret_cast<float4 *>(sm.P() + x * 4) = make_float4(0.f, 0.f, 0.f, 0.f);
  }
  cp_commit();
  // the values are first read by the u chain (step 6): a second copy group, landing under steps 2-5, so
  // the rows every CTA of a wave needs first (k, q) are a smaller burst
  for (int x = tid; x < C * 16; x += WY_THREADS) {
    const int i = x >> 4, c = x & 15;
    if (i >= g.r_lo)
      cp16(smem_u32(vt.chunk(i, c)), i < g.r_hi ? p.v + g.token(i) * p.vs0 + h * p.vs1 + c * 8 : p.v, i < g.r_hi);
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
    sm.lg()[i] = lg;
  }
  cp_wait<1>();   // k, q, X, P (the values may still be in flight)
  __syncthreads();

  // ---------------------------------------------------------------- 2. k^, q^ (8 + 8 rows per warp, folds interleaved); G
  #pragma unroll
  for (int which = 0; which < 2; ++which) {
    const Wide<64> &t = which ? qt : kt;
    float x[C / 8][CE];
    uint2 *e[C / 8];
    #pragma unroll
    for (int rr = 0; rr < C / 8; ++rr) {
      e[rr] = reinterpret_cast<uint2 *>(t.chunk(8 * rr + warp, lane >> 1) + 4 * (lane & 1));
      const uint2 raw = *e[rr];
      x[rr][0] = bf(lo16(raw.x));
      x[rr][1] = bf(hi16(raw.x));
      x[rr][2] = bf(lo16(raw.y));
      x[rr][3] = bf(hi16(raw.y));
    }
    uint2 nrm[C / 8];
    float inverse_norms[C / 8];
    l2norm_rows<C / 8>(x, p.scale, which == 1, nrm, SAVE ? inverse_norms : nullptr);
    #pragma unroll
    for (int rr = 0; rr < C / 8; ++rr) {
      const int i = 8 * rr + warp;
      if (i >= g.r_lo && i < g.r_hi) *e[rr] = nrm[rr];
    }
    if (SAVE && leader) save_norms(p, g, hq, which == 1, x, nrm, inverse_norms, warp, lane);
  }
  if (tid == 0) {   // G_i = G_{i-1} (+) lg_i: one serial chain, operands staged in registers
    float lg[C], G = g.r_lo > 0 ? gt.G[g.r_lo - 1] : 0.f;
    #pragma unroll
    for (int i = 0; i < C; ++i) lg[i] = sm.lg()[i];
    #pragma unroll
    for (int i = 0; i < C; ++i)
      if (i >= g.r_lo && i < g.r_hi) {
        G = cum_gate(G, lg[i]);
        gt.G[i] = G;
      }
  }
  fence_async_smem();
  __syncthreads();
  if constexpr (!FUSED) {
    if (tid == 0 && leader) {   // the K and Q tiles are final
      bulk_store(p.k_tile(s, hq), sm.k(), TILE_BYTES);
      bulk_store(p.q_tile(s, hq), sm.q(), TILE_BYTES);
      bulk_commit();
    }
  } else if (keep_open) {   // the output slot's open k^ rows, G and beta (this CTA's head)
    for (int x = tid; x < (open.jf - open.r_first) * 16; x += WY_THREADS) {
      const int i = open.r_first + (x >> 4), c = x & 15;
      *reinterpret_cast<uint4 *>(p.st.kb_tile(so, h).chunk(i, c)) = *reinterpret_cast<const uint4 *>(kt.chunk(i, c));
    }
    if (tid >= open.r_first && tid < open.jf) {
      p.st.gb(so, h)[tid] = gt.G[tid];
      p.st.bb(so, h)[tid] = gt.beta[tid];
    }
  }

  // ---------------------------------------------------------------- 3. decay factors; KK -> L, KQ -> M
  if (tid < C) {
    const int i = tid;
    const bool present = i < g.r_hi;
    gt.gam[i] = present ? decay0(gt.G[i]) : 0.f;
    gt.lam[i] = present && g.complete() ? decay(gt.G[C - 1], gt.G[i]) : 0.f;
  }
  const int dj0 = 16 * (warp & 3), di0 = 32 * (warp >> 2);   // this warp's block of the 64 x 64 products
  const bool dneed = dj0 < g.r_hi && dj0 + 16 > g.r_lo && di0 < dj0 + 16;
  uint32_t mpk[2][2][2];   // this warp's M entries, packed; written once q^ is no longer read
  {
    float kk[2][2][4], kq[2][2][4];
    if (dneed) {
      dots(kk, kt, kt, di0, dj0, lane);
      dots(kq, kt, qt, di0, dj0, lane);
    }
    #pragma unroll
    for (int mt = 0; mt < 2; ++mt)
      #pragma unroll
      for (int nt = 0; nt < 2; ++nt)
        #pragma unroll
        for (int eh = 0; eh < 2; ++eh) {
          uint16_t mv[2];
          #pragma unroll
          for (int el = 0; el < 2; ++el) {
            const int e = 2 * eh + el;
            const int i = di0 + 16 * mt + gq + eh * 8, j = dj0 + 8 * nt + 2 * tq + el;
            const float dec = decay(gt.G[j], gt.G[i]);
            // L_ji = +0 for i >= j (the zero extension of recipe.cuh); M_ji only for this call's rows j
            sm.L()[j * LS + i] = (dneed && i < j) ? l_entry(kk[mt][nt][e], gt.beta[j], dec) : 0.f;
            mv[el] = (dneed && i <= j && j >= g.r_lo && j < g.r_hi) ? m_entry(kq[mt][nt][e], dec) : uint16_t(0);
          }
          mpk[mt][nt][eh] = pack(mv[0], mv[1]);
        }
  }
  if (!FUSED && tid == 0 && leader) bulk_wait_read<0>();   // the Q tile has left q^'s buffer (M overlays it)
  __syncthreads();
  {
    const Tile<64> mtile{sm.m()};
    #pragma unroll
    for (int mt = 0; mt < 2; ++mt)
      #pragma unroll
      for (int nt = 0; nt < 2; ++nt)
        #pragma unroll
        for (int eh = 0; eh < 2; ++eh) {
          const int i = di0 + 16 * mt + gq + eh * 8, j = dj0 + 8 * nt + 2 * tq;
          mtile.at(j, i) = lo16(mpk[mt][nt][eh]);
          mtile.at(j + 1, i) = hi16(mpk[mt][nt][eh]);
        }
  }

  // ---------------------------------------------------------------- 4. X = (I + L)^-1 by 16-row blocks
  if (tid < C) {   // diagonal blocks, all four at once: lane (block bq, column m = 16bq + r)
    const int bq = tid >> 4, m = tid;
    float xb[16];
    #pragma unroll
    for (int r = 0; r < 16; ++r) xb[r] = sm.X()[(16 * bq + r) * C + m];
    #pragma unroll
    for (int r = 0; r < 16; ++r) {   // branch-free: every row is evaluated, rows outside the call are not kept
      const int j = 16 * bq + r;
      float l[16];
      #pragma unroll
      for (int c = 0; c < 4; ++c) {
        const float4 l4 = *reinterpret_cast<const float4 *>(sm.L() + j * LS + 16 * bq + 4 * c);
        l[4 * c] = l4.x;
        l[4 * c + 1] = l4.y;
        l[4 * c + 2] = l4.z;
        l[4 * c + 3] = l4.w;
      }
      // In a new row, L's diagonal and upper triangle and the still-unwritten xb tail
      // are +0. Evaluate only the nonzero-prefix FMAs of x_block, retaining its
      // four-leaf tree. The first omitted +0 product still contributes an explicit
      // +0 add: a tiny negative partial may have rounded to -0. Later +0 adds are
      // idempotent. Rows already in the input record are evaluated but never kept.
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
      const float xj = m < j ? x : (m == j ? 1.f : 0.f);
      xb[r] = (j >= g.r_lo && j < g.r_hi) ? xj : xb[r];
    }
    #pragma unroll
    for (int r = 0; r < 16; ++r) {
      const int j = 16 * bq + r;
      if (j >= g.r_lo && j < g.r_hi) sm.X()[j * C + m] = xb[r];
    }
  }
  __syncthreads();
  for (int bl = 1; bl < 4; ++bl) {   // earlier-block entries of block bl: P, then X = -X_bb P
    const int lo = max(16 * bl, g.r_lo), hi = min(16 * bl + 16, g.r_hi);
    if (lo >= hi) continue;   // uniform
    const int nq = 4 * bl;    // column quads m < 16 bl
    const int task = tid;     // (row j, column quad): at most 16 * 12 = 192 tasks
    const int j = lo + task / nq, m4 = 4 * (task % nq);
    if (task < (16 * bl + 16 - lo) * nq) {
      float pv[4] = {0.f, 0.f, 0.f, 0.f};
      if (j < hi) {   // rows past this call stay finite zeros
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
        for (int x = 0; x < 4; ++x) {   // T16 over the chunks (p_entry of recipe.cuh, four columns at once)
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
  // the output slot's open 16-row block keeps its P rows (they are the last level solved)
  if constexpr (!FUSED) {
    if (keep_open && g.r_hi > 16 && (g.r_hi & 15))
      for (int x = tid; x < (g.r_hi & 15) * 16; x += WY_THREADS)
        *reinterpret_cast<float4 *>(p.fp_tile(n, h) + x * 4) = *reinterpret_cast<const float4 *>(sm.P() + x * 4);
  } else if (keep_open && (open.jf & ~15) > 0) {
    const int pb0 = open.jf & ~15, r0 = max(open.r_first, pb0);
    for (int x = tid; x < (open.jf - r0) * 16; x += WY_THREADS) {
      const int i = r0 + (x >> 4), c = x & 15;
      *reinterpret_cast<float4 *>(p.st.pb(so, h) + (i - pb0) * C + c * 4) = *reinterpret_cast<const float4 *>(sm.P() + (i - pb0) * C + c * 4);
    }
  }

  // ---------------------------------------------------------------- 5. xu, xw rows; the open-chunk X rows
  {
    const Tile<64> xut{sm.xu()}, xwt{sm.xw()};
    for (int x = tid; x < C * 8; x += WY_THREADS) {
      const int j = x >> 3, c = x & 7;
      uint32_t pu[4] = {0u, 0u, 0u, 0u}, pw[4] = {0u, 0u, 0u, 0u};
      if (j >= g.r_lo && j < g.r_hi) {
        const float *r = sm.X() + j * C + c * 8;
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int i = 8 * c + 2 * e;
          pu[e] = pack(xu_entry(r[2 * e], gt.beta[i]), xu_entry(r[2 * e + 1], gt.beta[i + 1]));
          pw[e] = pack(xw_entry(r[2 * e], gt.beta[i], gt.gam[i]), xw_entry(r[2 * e + 1], gt.beta[i + 1], gt.gam[i + 1]));
        }
        if (SAVE) {   // trainer mode: FLA's A = (I + L)^-1 row j, bf16
          uint32_t pa[4];
          #pragma unroll
          for (int e = 0; e < 4; ++e) pa[e] = pack(rn(r[2 * e]), rn(r[2 * e + 1]));
          *reinterpret_cast<uint4 *>(p.sv.A + (g.token(j) * p.hv + h) * C + c * 8) = make_uint4(pa[0], pa[1], pa[2], pa[3]);
        }
      }
      *reinterpret_cast<uint4 *>(xut.chunk(j, c)) = make_uint4(pu[0], pu[1], pu[2], pu[3]);
      *reinterpret_cast<uint4 *>(xwt.chunk(j, c)) = make_uint4(pw[0], pw[1], pw[2], pw[3]);
    }
  }
  if constexpr (!FUSED) {
    if (keep_open)
      for (int x = tid; x < g.r_hi * 16; x += WY_THREADS)
        *reinterpret_cast<float4 *>(p.fx_tile(n, h) + x * 4) = *reinterpret_cast<const float4 *>(sm.X() + x * 4);
  } else if (keep_open) {
    for (int x = tid; x < (open.jf - open.r_first) * 16; x += WY_THREADS) {
      const int i = open.r_first + (x >> 4), c = x & 15;
      *reinterpret_cast<float4 *>(p.st.xb(so, h) + i * C + c * 4) = *reinterpret_cast<const float4 *>(sm.X() + i * C + c * 4);
    }
  }
  cp_wait<0>();         // the values
  fence_async_smem();   // xu, xw (and v, k^) are read by the WGMMAs below
  __syncthreads();

  // ---------------------------------------------------------------- 6. w (warpgroup 0) and u (warpgroup 1)
  // Transposed products: W[j][d] = CHAIN64_i(xw[j][i], k^[i][d]) and U^T[j][v] = CHAIN64_i(xu[j][i], v[i][v])
  // (A = the xw / xu tile, B = the k^ / v tile read MN-major), each element the recipe's chain with the
  // operand roles swapped (tests/test_mma_facts_gpu.py). Rows j outside the call are zeros.
  {
    const int wg = warp >> 2, wgw = warp & 3;
    const Tile<64> xut{sm.xu()}, xwt{sm.xw()};
    const Wide<64> wt{sm.w()};
    const Tile<64> ut{sm.u()};
    float acc[64];   // (j = 16 wgw + gq (+8), column 8 nt + 2 tq (+1)): d for warpgroup 0, v for warpgroup 1
    wg_fence();
    #pragma unroll
    for (int s = 0; s < 4; ++s)
      wgmma_n128_ss<1>(acc, desc_kmajor(wg ? xut.base : xwt.base, 16 * s), desc_mnmajor(wg ? vt : kt, 16 * s), s > 0);
    wg_commit();
    wg_wait<0>();
    wg_pin(acc);
    const int j0r = 16 * wgw + gq;   // this thread's rows j0r, j0r + 8
    const bool c0 = j0r >= g.r_lo && j0r < g.r_hi, c1 = j0r + 8 >= g.r_lo && j0r + 8 < g.r_hi;
    const int mm = lane >> 3, rr = lane & 7;
    #pragma unroll
    for (int np = 0; np < 8; ++np) {   // n8 tiles 2np, 2np + 1 = columns 16np..16np+15
      uint32_t r4[4];
      #pragma unroll
      for (int h2 = 0; h2 < 2; ++h2) {
        const int nt = 2 * np + h2;
        r4[2 * h2] = c0 ? pack(rn(acc[4 * nt]), rn(acc[4 * nt + 1])) : 0u;
        r4[2 * h2 + 1] = c1 ? pack(rn(acc[4 * nt + 2]), rn(acc[4 * nt + 3])) : 0u;
      }
      if (wg == 0)   // W [j][d] as computed
        stsm_x4(smem_u32(wt.half(np >> 2).chunk(16 * wgw + rr + 8 * (mm & 1), 2 * (np & 3) + (mm >> 1))), r4[0], r4[1],
                r4[2], r4[3]);
      else           // U [v][j], transposed
        stsm_x4_t(smem_u32(ut.chunk(8 * (2 * np + (mm >> 1)) + rr, 2 * wgw + (mm & 1))), r4[0], r4[1], r4[2], r4[3]);
    }
  }
  fence_async_smem();
  __syncthreads();

  if (SAVE)   // trainer mode: the values and w rows as FLA lays them out
    for (int x = tid; x < (g.r_hi - g.r_lo) * 16; x += WY_THREADS) {
      const int j = g.r_lo + (x >> 4), c = x & 15;
      const int64_t row = g.token(j) * p.hv + h;
      *reinterpret_cast<uint4 *>(p.sv.v + row * D + c * 8) = *reinterpret_cast<const uint4 *>(vt.chunk(j, c));
      if (p.sv.w)
        *reinterpret_cast<uint4 *>(p.sv.w + row * D + c * 8) = *reinterpret_cast<const uint4 *>(Wide<64>{sm.w()}.chunk(j, c));
    }

}

}  // namespace gdn_cpr
