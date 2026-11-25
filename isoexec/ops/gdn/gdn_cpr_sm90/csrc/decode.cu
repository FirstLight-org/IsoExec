// SPDX-License-Identifier: Apache-2.0
// Engine decode: one token per sequence, one launch, no workspace, no host involvement.
//
// A decode step for a sequence at position p evaluates row j = p mod 64 of the chunk arithmetic of
// recipe.cuh -- exactly the row a prefill over the same tokens evaluates -- reading rows [0, j) from
// the slot's open-chunk buffer and appending row j to it. When row 63 completes, the same launch
// replaces the boundary state (the in-kernel resync): S_{c+1} = CHAIN64(d~, k^; S_c (*) e(G_63)).
// Every step reads the fp32 state S_c and rounds it to S^ = rn(S_c) as it loads its A fragments (the
// record keeps no separate image); the state is written at the boundary alone.
//
// Execution: a head is split over R CTAs (R = 1, 2, 4, 8), CTA r owning value rows [128r/R, 128(r+1)/R);
// four warps per CTA. The work that does not depend on the value row (q^, k^, the gate, the key
// products, the row of X, w_j) is repeated by every CTA of the head; CTA 0 appends the key-side rows.
// Each CTA reads and advances only its own copies of the position (state.cuh), and reads only rows
// [0, j) of the record while row j is appended (by CTA 0 for the key side, by each CTA for its value
// rows), so the CTAs of a head need no ordering between them. Sequences whose input and output slots are both negative are NULL lanes:
// they read and write nothing. A slot must not be one lane's output and another lane's input.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <type_traits>

#include "launch.h"
#include "state.cuh"
#include "tile.cuh"

namespace gdn_cpr {
namespace {

constexpr int DEC_THREADS = 128;

struct DecodeParams {
  const uint16_t *q, *k, *v, *a, *b;   // q/k [N, hk, 128], v [N, hv, 128], a/b [N, hv]
  int64_t qs0, qs1, ks0, ks1, vs0, vs1, as0, as1, bs0, bs1;
  const void *A_log, *dt_bias;
  int param_bf16;
  StateRef st;
  const int *slot_in, *slot_out;
  uint16_t *out;
  int64_t os0, os1;
  int n, hk, hv;
  float scale;
};

template <int VR>   // value rows of this CTA
struct DecSmem {
  uint16_t k[C * D];    // Wide<64>: k^ rows [0, j], zeros after (the image of the record's kb)
  uint16_t v[VR * C];   // Cols<VR>: v [v][i] of this CTA's value rows, zeros past column j
  uint16_t d[VR * C];   // Cols<VR>: delta [v][i] of this CTA's value rows, zeros past column j
  float X[C * C];       // X rows [0, j], zeros after
  float P[16 * C];      // P rows of the open 16-row block, zeros after row j
  float G[C], beta[C], kk[C], kq[C], L[C], dec[C], gam[C];
  uint16_t mrow[C], xu[C], xw[C];         // B-operand columns: M_j., xu_j., xw_j.
  uint16_t khat[D], qhat[D], wv[D];
  uint16_t uv[VR], vraw[VR];
  float seed[VR];
  float lg_j, beta_j;
  uint64_t bar[3];   // bulk copies: k^ (and G, beta) | this CTA's v and delta | X and P
};

// An A fragment whose k index is a chunk row i (k16 step at row i0): elements of rows past j become +0
// (a[q] holds k = i0 + 2t + 8 (q / 2) and the next row).
__device__ __forceinline__ void mask_rows(uint32_t (&a)[4], int i0, int j, int tq) {
  #pragma unroll
  for (int q = 0; q < 4; ++q) {
    const int i = i0 + 2 * tq + 8 * (q >> 1);
    a[q] &= (i <= j ? 0x0000FFFFu : 0u) | (i + 1 <= j ? 0xFFFF0000u : 0u);
  }
}

// B fragment of an n8 tile whose columns 0 and 1 are the bf16 vectors c0, c1 (other columns zero).
__device__ __forceinline__ void frag_b_cols(uint32_t (&b)[2], const uint16_t *c0, const uint16_t *c1, int k0, int lane) {
  const int g = lane >> 2, t = lane & 3;
  const uint16_t *c = g == 0 ? c0 : c1;
  if (g < 2 && c != nullptr) {
    b[0] = *reinterpret_cast<const uint32_t *>(c + k0 + 2 * t);
    b[1] = *reinterpret_cast<const uint32_t *>(c + k0 + 2 * t + 8);
  } else {
    b[0] = b[1] = 0u;
  }
}

template <int R>
__global__ __launch_bounds__(DEC_THREADS, 4) void decode_kernel(const __grid_constant__ DecodeParams p) {
  constexpr int VR = D / R, VT = VR / 16;   // value rows and 16-row value tiles of this CTA
  constexpr int VP = NPOS / R;              // the position copies this CTA owns
  extern __shared__ __align__(128) char smem_raw[];
  DecSmem<VR> &sm = *reinterpret_cast<DecSmem<VR> *>(smem_raw);
  const int r = blockIdx.x % R;
  const int head = blockIdx.x / R, seq = head / p.hv, h = head % p.hv;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, gq = lane >> 2, tq = lane & 3;
  const int hq = h / (p.hv / p.hk);
  const int v0 = r * VR;
  const Wide<C> kt{sm.k};
  const Cols<VR> vt{sm.v}, dt{sm.d};
  const int si = p.slot_in[seq], so = p.slot_out[seq];   // first: the slot -> position chase gates the rows
  // the token's own inputs do not depend on the slot: start them before the slot -> position chase
  uint2 raw_qk = make_uint2(0, 0);
  if (warp < 2)
    raw_qk = *reinterpret_cast<const uint2 *>((warp == 0 ? p.k + seq * p.ks0 + hq * p.ks1 : p.q + seq * p.qs0 + hq * p.qs1) + 4 * lane);
  if (tid >= 64 && tid - 64 < VR / 8)
    cp16(smem_u32(sm.vraw + (tid - 64) * 8), p.v + seq * p.vs0 + h * p.vs1 + v0 + (tid - 64) * 8, true);
  uint16_t a_raw = 0, b_raw = 0, A16 = 0, dt16 = 0;   // raw: nothing here waits for them
  float A32 = 0.f, dt32 = 0.f;
  if (tid == 64) {
    a_raw = p.a[seq * p.as0 + h * p.as1];
    b_raw = p.b[seq * p.bs0 + h * p.bs1];
    if (p.param_bf16) {
      A16 = static_cast<const uint16_t *>(p.A_log)[h];
      dt16 = static_cast<const uint16_t *>(p.dt_bias)[h];
    } else {
      A32 = static_cast<const float *>(p.A_log)[h];
      dt32 = static_cast<const float *>(p.dt_bias)[h];
    }
  }
  // Each barrier is initialized by the one thread that arms it and (with its warp, after a __syncwarp)
  // waits on it, so no CTA-wide barrier precedes the first loads.
  if (tid == 0) {
    mbar_init(&sm.bar[0], 1);
    mbar_init(&sm.bar[2], 1);
    fence_async_smem();
  } else if (tid == 64) {
    mbar_init(&sm.bar[1], 1);
    fence_async_smem();
  }
  __syncwarp();
  if (si < 0 && so < 0) {   // a NULL lane: nothing to read or write (the v row copy lands first)
    cp_commit();
    cp_wait<0>();
    return;
  }
  const int pos = si >= 0 ? p.st.pos(si, h)[r * VP] : 0;   // issued first: the position gates the rest
  // S^ of this warp's value tiles as A fragments (tile t = warp + 4u: rows 16t..16t+15 of the CTA),
  // issued before the position is known so that they land under the position load
  constexpr int WT = VT >= 4 ? VT / 4 : 1;   // value tiles per warp
  const bool wact = warp < VT;               // this warp owns value tiles
  uint32_t sa[WT][8][4];
  #pragma unroll
  for (int u = 0; u < WT; ++u) {
    const int row = v0 + 16 * (warp + 4 * u) + gq;
    #pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      if (wact && si >= 0) {   // the S^ fragment image (state.cuh): one 16-byte load per k16 step
        const uint4 f = *reinterpret_cast<const uint4 *>(p.st.sh(si, h) + (((row >> 4) * 8 + ks) * 32 + lane) * 8);
        sa[u][ks][0] = f.x;
        sa[u][ks][1] = f.y;
        sa[u][ks][2] = f.z;
        sa[u][ks][3] = f.w;
      } else {
        sa[u][ks][0] = sa[u][ks][1] = sa[u][ks][2] = sa[u][ks][3] = 0u;
      }
    }
  }
  const int j = pos & (C - 1), B = j >> 4;

  // ---------------------------------------------------------------- 0. loads (the rows present only)
  // One thread per side issues the bulk copies: the key side (rows [0, j) of k^ and X, the P rows of the
  // open block) and this CTA's value side (the column chunks of v and delta holding [0, j)). Rows past j
  // are never read as data, so nothing is cleared: the u, w and o chains skip their k16
  // steps past row j's 16-row block (a step whose B column is all +0 leaves the accumulator
  // bit-identical, tests/test_mma_facts_gpu.py) and select +0 for the operand elements past row j in that
  // block (mask_rows), which is exactly the zero extension of recipe.cuh; the X and P rows the row of X
  // reads are selected the same way; the key products and P chains of rows past j are selected away.
  const int pr = B > 0 ? (j & 15) : 0;  // P rows of the open block present
  const int nc = (j + 7) >> 3;          // value column chunks holding rows < j
  if (tid == 0) {
    mbar_expect(&sm.bar[0], 2 * j * 128);
    mbar_expect(&sm.bar[2], j * C * 4 + pr * C * 4);
    if (j > 0) {
      const Wide<C> rec = p.st.kb_tile(si, h);
      bulk_load(kt.half(0).base, rec.half(0).base, j * 128, &sm.bar[0]);
      bulk_load(kt.half(1).base, rec.half(1).base, j * 128, &sm.bar[0]);
      bulk_load(sm.X, p.st.xb(si, h), j * C * 4, &sm.bar[2]);
      bulk_load(sm.P, p.st.pb(si, h), pr * C * 4, &sm.bar[2]);
    }
  } else if (tid == 64) {
    mbar_expect(&sm.bar[1], 2 * nc * VR * 16);
    if (j > 0) {
      const Cols<D> rv = p.st.vb_tile(si, h), rd = p.st.db_tile(si, h);
      for (int s = 0; s < VR; s += 16) {   // per 16-row slab: its chunks [0, nc)
        bulk_load(vt.chunk(s, 0), rv.chunk(v0 + s, 0), nc * 256, &sm.bar[1]);
        bulk_load(dt.chunk(s, 0), rd.chunk(v0 + s, 0), nc * 256, &sm.bar[1]);
      }
    }
  }
  if (tid < 64) {
    cp4(smem_u32(sm.G + tid), tid < j ? static_cast<const void *>(p.st.gb(si, h) + tid) : p.k, tid < j);
    cp4(smem_u32(sm.beta + tid), tid < j ? static_cast<const void *>(p.st.bb(si, h) + tid) : p.k, tid < j);
  }
  cp_commit();

  // ---------------------------------------------------------------- 1. k^_j, q^_j, the gate (overlapping the loads)
  uint2 nrm = make_uint2(0, 0);
  if (warp < 2) {
    const uint2 raw = raw_qk;
    const float x[4] = {bf(lo16(raw.x)), bf(hi16(raw.x)), bf(lo16(raw.y)), bf(hi16(raw.y))};
    nrm = l2norm_row(x, p.scale, warp == 1);
    *reinterpret_cast<uint2 *>((warp == 0 ? sm.khat : sm.qhat) + 4 * lane) = nrm;
  } else if (tid == 64) {
    const Gate gt = gate(bf(a_raw), bf(b_raw), p.param_bf16 ? bf(A16) : A32, p.param_bf16 ? bf(dt16) : dt32);
    sm.lg_j = gt.lg;
    sm.beta_j = gt.beta;
  }
  if (tid < 64) cp_wait<0>();   // G, beta
  if (warp == 0) {              // the key-side rows
    mbar_wait(&sm.bar[0], 0);
    *reinterpret_cast<uint2 *>(kt.chunk(j, lane >> 1) + 4 * (lane & 1)) = nrm;
  }
  __syncthreads();

  // ---------------------------------------------------------------- 2. KK_.j, KQ_.j | the value side: v_j, rows past j
  const float G_j = cum_gate(j > 0 ? sm.G[j - 1] : 0.f, sm.lg_j), beta_j = sm.beta_j;
  if (tid == 0) {
    sm.G[j] = G_j;
    sm.beta[j] = beta_j;
  }
  if (16 * warp <= j) {   // CHAIN128 of key rows [16w, 16w + 16) against the columns (k^_j, q^_j)
    float acc[4] = {};
    #pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      uint32_t a[4], b[2];
      frag_a(a, kt, 16 * warp, 16 * ks, lane);
      frag_b_cols(b, sm.khat, sm.qhat, 16 * ks, lane);
      mma(acc, a, b);
    }
    if (tq == 0) {
      sm.kk[16 * warp + gq] = acc[0];
      sm.kq[16 * warp + gq] = acc[1];
      sm.kk[16 * warp + gq + 8] = acc[2];
      sm.kq[16 * warp + gq + 8] = acc[3];
    }
  }
  __syncthreads();

  // ---------------------------------------------------------------- 3. L_j., M_j., X_j. (blocked, recipe.cuh)
  if (tid < C) {
    const int i = tid;
    const float dec = decay(G_j, sm.G[i]);
    sm.L[i] = i < j ? l_entry(sm.kk[i], beta_j, dec) : 0.f;
    sm.mrow[i] = i <= j ? m_entry(sm.kq[i], dec) : uint16_t(0);
    sm.dec[i] = dec;   // at the boundary (j = 63): d~'s factor e(G_63 - G_i)
    sm.gam[i] = decay0(sm.G[i]);   // xw's factor e(G_i), before the rows of X arrive
    mbar_wait(&sm.bar[2], 0);   // X and P (published by the barrier below)
  }
  if (tid >= 64) {   // the value side, by the threads that fetched it (first needed by w_j, u_j)
    cp_wait<0>();   // v_j
    if (tid == 64) mbar_wait(&sm.bar[1], 0);
    asm volatile("bar.sync 1, 64;" ::: "memory");
    for (int x = tid - 64; x < VR; x += 64) vt.at(x, j) = sm.vraw[x];
  }
  __syncthreads();
  const int m = tid;
  if (m < 16 * B) {   // P_j. (earlier blocks) and the diagonal-block entries
    sm.P[(j & 15) * C + m] = p_entry<12>(B, [&](int i) { return sm.L[i]; }, [&](int i) { return sm.X[i * C + m]; });
  } else if (m < C) {
    float l[16], xb[16];
    #pragma unroll
    for (int rr = 0; rr < 16; ++rr) {
      l[rr] = m < 16 * B + 16 ? sm.L[16 * B + rr] : 0.f;                    // +0 for i >= j
      xb[rr] = m < 16 * B + 16 && 16 * B + rr < j ? sm.X[(16 * B + rr) * C + m] : 0.f;   // rows < j only
    }
    const float x = m < j ? x_block(l, xb) : (m == j ? 1.f : 0.f);   // final for m >= 16B
    sm.X[j * C + m] = x;
    sm.xu[m] = xu_entry(x, sm.beta[m]);   // +0 past j: X_jm = +0 and beta_m = +0 there
    sm.xw[m] = xw_entry(x, sm.beta[m], sm.gam[m]);
  }
  __syncthreads();
  if (B > 0) {   // earlier-block entries: -X_bb P (uniform: no barrier when there are none)
    if (m < 16 * B) {
      float y[16], pc[16];
      #pragma unroll
      for (int rr = 0; rr < 16; ++rr) {
        y[rr] = sm.X[j * C + 16 * B + rr];
        pc[rr] = rr <= (j & 15) ? sm.P[rr * C + m] : 0.f;   // rows past j mod 16 are absent
      }
      const float x = x_block(y, pc);
      sm.X[j * C + m] = x;
      sm.xu[m] = xu_entry(x, sm.beta[m]);
      sm.xw[m] = xw_entry(x, sm.beta[m], sm.gam[m]);
    }
    __syncthreads();
  }

  // ---------------------------------------------------------------- 4. w_j (all d), u_j (this CTA's value rows)
  // Steps past row j's block have all-zero B columns and are skipped exactly; the step count B + 1 is a
  // template constant per branch so that every fragment load of a chain issues before its first MMA.
  auto wu = [&](auto steps) {
    constexpr int NS = decltype(steps)::value;
    #pragma unroll
    for (int u = 0; u < 2; ++u) {   // w: 8 tiles of 16 d rows, two per warp
      const int d0 = 16 * (warp + 4 * u);
      uint32_t a[NS][4], b[NS][2];
      #pragma unroll
      for (int ks = 0; ks < NS; ++ks) {
        frag_a_t(a[ks], kt, d0, 16 * ks, lane);   // k^^T
        frag_b_cols(b[ks], sm.xw, nullptr, 16 * ks, lane);
      }
      mask_rows(a[NS - 1], 16 * (NS - 1), j, tq);
      float cw[4] = {};
      #pragma unroll
      for (int ks = 0; ks < NS; ++ks) mma(cw, a[ks], b[ks]);
      if (tq == 0) {
        sm.wv[d0 + gq] = rn(cw[0]);
        sm.wv[d0 + gq + 8] = rn(cw[2]);
      }
    }
    if (wact)
      #pragma unroll
      for (int u = 0; u < WT; ++u) {
        const int t0 = 16 * (warp + 4 * u);
        uint32_t a[NS][4], b[NS][2];
        #pragma unroll
        for (int ks = 0; ks < NS; ++ks) {
          frag_a(a[ks], vt, t0, 16 * ks, lane);   // v^T (value-major tile)
          frag_b_cols(b[ks], sm.xu, nullptr, 16 * ks, lane);
        }
        mask_rows(a[NS - 1], 16 * (NS - 1), j, tq);
        float cu[4] = {};
        #pragma unroll
        for (int ks = 0; ks < NS; ++ks) mma(cu, a[ks], b[ks]);
        if (tq == 0) {
          sm.uv[t0 + gq] = rn(cu[0]);
          sm.uv[t0 + gq + 8] = rn(cu[2]);
        }
      }
  };
  switch (B) {
    case 0: wu(std::integral_constant<int, 1>{}); break;
    case 1: wu(std::integral_constant<int, 2>{}); break;
    case 2: wu(std::integral_constant<int, 3>{}); break;
    default: wu(std::integral_constant<int, 4>{}); break;
  }
  __syncthreads();

  // Appending row j in place: the key-side rows and v_j are final now. The warps that own no value tiles
  // (split R >= 4) store them while the others run the S^ chain -- otherwise every thread stores its share
  // first; the deltas follow in step 5, so no barrier is left at the end of an interior step.
  const bool inplace = so >= 0 && so == si;
  constexpr int APPENDERS = VT < 4 ? 4 - VT : 0;   // warps without value tiles
  auto append_row = [&](int t, int nthreads) {   // key-side row j (leader CTA) and this CTA's v_j
    if (r == 0) {
      for (int x = t; x < 16; x += nthreads) {
        *reinterpret_cast<uint4 *>(p.st.kb_tile(so, h).chunk(j, x)) = *reinterpret_cast<const uint4 *>(kt.chunk(j, x));
        *reinterpret_cast<uint4 *>(p.st.xb(so, h) + j * C + x * 4) = *reinterpret_cast<const uint4 *>(sm.X + j * C + x * 4);
        if (B > 0)   // the P row of the open block: columns [0, 16B) (the rest is never read)
          *reinterpret_cast<uint4 *>(p.st.pb(so, h) + (j & 15) * C + x * 4) =
              4 * x < 16 * B ? *reinterpret_cast<const uint4 *>(sm.P + (j & 15) * C + x * 4) : make_uint4(0, 0, 0, 0);
      }
      if (t == 0) {
        p.st.gb(so, h)[j] = sm.G[j];
        p.st.bb(so, h)[j] = sm.beta[j];
      }
    }
    for (int x = t; x < VR; x += nthreads) p.st.vb_tile(so, h).at(v0 + x, j) = vt.at(x, j);
  };
  if (inplace) {
    if (APPENDERS == 0) append_row(tid, DEC_THREADS);
    else if (warp >= VT) append_row(tid - 32 * VT, 32 * APPENDERS);
  }

  // ---------------------------------------------------------------- 5. CHAIN128(S^, [w_j q^_j]) -> delta_j, output seed
  if (wact)
    #pragma unroll
    for (int u = 0; u < WT; ++u) {
      const int t0 = 16 * (warp + 4 * u);
      float acc[4] = {};
      #pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        uint32_t b[2];
        frag_b_cols(b, sm.wv, sm.qhat, 16 * ks, lane);
        mma(acc, sa[u][ks], b);
      }
      if (tq == 0) {
        const float gam = decay0(G_j);
        #pragma unroll
        for (int rr = 0; rr < 2; ++rr) {
          const int vr = t0 + gq + 8 * rr;
          const uint16_t d = delta_entry(sm.uv[vr], acc[2 * rr]);
          dt.at(vr, j) = d;
          if (inplace) p.st.db_tile(so, h).at(v0 + vr, j) = d;   // appended in place now
          sm.seed[vr] = out_seed(acc[2 * rr + 1], gam);
        }
      }
    }
  __syncwarp();   // row j of d and the seeds are this warp's own value rows

  // ---------------------------------------------------------------- 6. o_j = rn(CHAIN64(d, M_j.; seed))
  if (wact)
    #pragma unroll
    for (int u = 0; u < WT; ++u) {
      const int t0 = 16 * (warp + 4 * u);
      float acc[4] = {tq == 0 ? sm.seed[t0 + gq] : 0.f, 0.f, tq == 0 ? sm.seed[t0 + gq + 8] : 0.f, 0.f};
      #pragma unroll
      for (int ks = 0; ks <= 3; ++ks) {
        if (ks > B) break;
        uint32_t a[4], b[2];
        frag_a(a, dt, t0, 16 * ks, lane);
        if (ks == B) mask_rows(a, 16 * ks, j, tq);
        frag_b_cols(b, sm.mrow, nullptr, 16 * ks, lane);
        mma(acc, a, b);
      }
      if (tq == 0) {
        uint16_t *o = p.out + seq * p.os0 + h * p.os1 + v0;
        o[t0 + gq] = rn(acc[0]);
        o[t0 + gq + 8] = rn(acc[2]);
      }
    }

  const bool boundary = j == C - 1;
  if (so >= 0 && !(inplace && !boundary)) {
    __syncthreads();   // the append and the boundary read every warp's row j of v and d
    // ---------------------------------------------------------------- 7. append row j (and move rows [0, j) with the slot)
    if (inplace) {
      // row j was appended in steps 4-5
    } else if (r == 0) {   // rows of the value-independent sections: the leader writes them
      for (int x = tid; x < (j + 1) * 16; x += DEC_THREADS) {   // the slot moves: rows [0, j]
        const int i = x >> 4, c = x & 15;
        *reinterpret_cast<uint4 *>(p.st.kb_tile(so, h).chunk(i, c)) = *reinterpret_cast<const uint4 *>(kt.chunk(i, c));
        *reinterpret_cast<uint4 *>(p.st.xb(so, h) + i * C + c * 4) = *reinterpret_cast<const uint4 *>(sm.X + i * C + c * 4);
      }
      if (B > 0) {   // P rows of the open block: columns [0, 16B) (the rest is never read)
        for (int x = tid; x < ((j & 15) + 1) * 16; x += DEC_THREADS)
          *reinterpret_cast<uint4 *>(p.st.pb(so, h) + x * 4) =
              4 * (x & 15) < 16 * B ? *reinterpret_cast<const uint4 *>(sm.P + x * 4) : make_uint4(0, 0, 0, 0);
      }
      if (tid <= j) {
        p.st.gb(so, h)[tid] = sm.G[tid];
        p.st.bb(so, h)[tid] = sm.beta[tid];
      }
    }
    if (!inplace)
      for (int x = tid; x < VR * (j + 1); x += DEC_THREADS) {   // this CTA's value rows, columns [0, j]
        const int vr = x / (j + 1), i = x % (j + 1);
        p.st.vb_tile(so, h).at(v0 + vr, i) = vt.at(vr, i);
        p.st.db_tile(so, h).at(v0 + vr, i) = dt.at(vr, i);
      }
    // ---------------------------------------------------------------- 8. boundary: S_{c+1} = CHAIN64(d~, k^; S_c * e(G_63))
    if (boundary) {
      const float gam = decay0(G_j);
      if (wact)
        #pragma unroll
        for (int u = 0; u < WT; ++u) {
          const int t0 = 16 * (warp + 4 * u), row = v0 + t0 + gq;
          uint32_t ta[4][4];   // d~^T fragments: column (chunk row) 16ks + 2tq (+1), +8
          #pragma unroll
          for (int ks = 0; ks < 4; ++ks) {
            frag_a(ta[ks], dt, t0, 16 * ks, lane);
            #pragma unroll
            for (int q4 = 0; q4 < 4; ++q4) {
              const int i = 16 * ks + 2 * tq + (q4 >> 1) * 8;
              ta[ks][q4] = pack(dtilde_entry(lo16(ta[ks][q4]), sm.dec[i]), dtilde_entry(hi16(ta[ks][q4]), sm.dec[i + 1]));
            }
          }
          #pragma unroll
          for (int half = 0; half < 2; ++half) {   // 64 state columns at a time
            float acc[8][4];
            #pragma unroll
            for (int k2 = 0; k2 < 4; ++k2) {   // the decoded S_c fragment of k16 step 4 half + k2 = n8 tiles 2 k2, 2 k2 + 1
              float x[8];
              if (si >= 0) {
                s_load8(p.st, si, h, (((row >> 4) * 8 + 4 * half + k2) * 32 + lane) * 8, x);
              } else {
                #pragma unroll
                for (int e = 0; e < 8; ++e) x[e] = 0.f;
              }
              #pragma unroll
              for (int e = 0; e < 4; ++e) {
                acc[2 * k2][e] = state_seed(x[e], gam);
                acc[2 * k2 + 1][e] = state_seed(x[4 + e], gam);
              }
            }
            #pragma unroll
            for (int ks = 0; ks < 4; ++ks)
              #pragma unroll
              for (int np = 0; np < 4; ++np) {
                uint32_t bb[4];
                frag_b2_t(bb, kt, 64 * half + 16 * np, 16 * ks, lane);
                const uint32_t b0[2] = {bb[0], bb[1]}, b1[2] = {bb[2], bb[3]};
                mma(acc[2 * np], ta[ks], b0);
                mma(acc[2 * np + 1], ta[ks], b1);
              }
            #pragma unroll
            for (int k2 = 0; k2 < 4; ++k2) {   // S_{c+1}, encoded: this lane owns the fragment's 8 elements and tie byte
              const float x[8] = {acc[2 * k2][0], acc[2 * k2][1], acc[2 * k2][2], acc[2 * k2][3],
                                  acc[2 * k2 + 1][0], acc[2 * k2 + 1][1], acc[2 * k2 + 1][2], acc[2 * k2 + 1][3]};
              s_store8(p.st, so, h, (((row >> 4) * 8 + 4 * half + k2) * 32 + lane) * 8, x);
            }
          }
        }
    } else if (so != si) {   // the boundary state moves with the slot (a NULL input slot is the zero state)
      s_move_rows(p.st, si, so, h, v0, VR, tid, DEC_THREADS);
    }
  }
  if (so >= 0 && tid < VP) p.st.pos(so, h)[r * VP + tid] = pos + 1;   // this CTA's copies
}

template <int R>
void launch(const DecodeParams &p, cudaStream_t stream) {
  const int bytes = int(sizeof(DecSmem<D / R>));
  auto kernel = decode_kernel<R>;
  C10_CUDA_CHECK(allow_smem<decode_kernel<R>>(bytes));
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(unsigned(p.n * p.hv * R));
  cfg.blockDim = dim3(DEC_THREADS);
  cfg.dynamicSmemBytes = bytes;
  cfg.stream = stream;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, p));
}

}  // namespace
}  // namespace gdn_cpr

void gdn_cpr_decode(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, const at::Tensor &a,
                    const at::Tensor &b, const at::Tensor &A_log, const at::Tensor &dt_bias, at::Tensor state,
                    const at::Tensor &slot_in, const at::Tensor &slot_out, at::Tensor out, double scale, int64_t split) {
  using namespace gdn_cpr;
  const c10::cuda::CUDAGuard guard(q.device());
  const int n = int(q.size(0));
  if (n <= 0) return;
  const int hk = int(q.size(1)), hv = int(v.size(1));
  DecodeParams p{static_cast<const uint16_t *>(q.data_ptr()), static_cast<const uint16_t *>(k.data_ptr()),
                 static_cast<const uint16_t *>(v.data_ptr()), static_cast<const uint16_t *>(a.data_ptr()),
                 static_cast<const uint16_t *>(b.data_ptr()),
                 q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1),
                 a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                 A_log.data_ptr(), dt_bias.data_ptr(), int(A_log.scalar_type() == at::kBFloat16),
                 StateRef{static_cast<char *>(state.data_ptr()), hv},
                 slot_in.data_ptr<int>(), slot_out.data_ptr<int>(),
                 static_cast<uint16_t *>(out.data_ptr()), out.stride(0), out.stride(1),
                 n, hk, hv, float(scale)};
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  switch (split) {
    case 1: launch<1>(p, stream); break;
    case 2: launch<2>(p, stream); break;
    case 4: launch<4>(p, stream); break;
    case 8: launch<8>(p, stream); break;
    default: TORCH_CHECK(false, "gdn_cpr_sm90 decode: unsupported split ", split);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
