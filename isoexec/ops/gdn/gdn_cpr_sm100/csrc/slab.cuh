// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "chunk.cuh"

namespace gdn_cpr_sm100 {

constexpr int MT = C / 16;      // m16 tiles of the token index: four
using SlabT = KTile;            // [i][v], full 128 value columns

// Does a boundary chain or a slot move need the fp32 state in registers for this chunk?
__device__ __forceinline__ bool state_in_regs(const ChunkGeo &g, int si, int so, bool more) {
  return (g.complete() && (more || so >= 0)) || (so >= 0 && so != si);
}

// Element (accumulator order) of an A fragment read back from a tile a `store_acc` wrote:
// word 2 * nt + (e >> 1), low half for even e.
__device__ __forceinline__ uint16_t frag_at(const uint32_t (&f)[4], int nt, int e) {
  const uint32_t w = f[2 * nt + (e >> 1)];
  return (e & 1) ? hi16(w) : lo16(w);
}

__device__ __forceinline__ void out_seed_scale(float (&o)[C / 16][2][4], const GateTile &gt, int gq) {
  #pragma unroll
  for (int m = 0; m < C / 16; ++m) {
    const float gam0 = gt.gam[16 * m + gq], gam1 = gt.gam[16 * m + gq + 8];
    #pragma unroll
    for (int nt = 0; nt < 2; ++nt)
      #pragma unroll
      for (int e = 0; e < 4; ++e) o[m][nt][e] = out_seed(o[m][nt][e], (e >> 1) ? gam1 : gam0);
  }
}

template <bool PRED, int R, int CO>
__device__ __forceinline__ void chain_step(float (&d0)[4], float (&d1)[4], const Tile<R, CO> &at, int m0,
                                           int k0, int lane, const uint32_t (&b0)[2],
                                           const uint32_t (&b1)[2], unsigned need) {
  uint32_t a[4];
  if (!PRED) {
    if (need) {   // ALL folds this to a constant; a partial PREFILL chunk keeps the branch, see below
      frag_a(a, at, m0, k0, lane);
      mma(d0, a, b0);
      mma(d1, a, b1);
    }
  } else {
    frag_a_p(a, at, m0, k0, lane, need);
    mma_p(d0, a, b0, need);
    mma_p(d1, a, b1, need);
  }
}

// A sequence whose slot moves without consuming tokens: the record, copied (S^ as rn(S), as a commit
// writes it).
template <int NT = THREADS>
__device__ __forceinline__ void move_record(const Params &p, int h, int si, int so, int pos_in) {
  const int tid = threadIdx.x;
  for (int x = tid; x < D * D; x += NT) {
    const float s = si >= 0 ? p.st.S(si, h)[x] : 0.f;
    p.st.S(so, h)[x] = s;
    ShTile{p.st.sh(so, h)}.at(x / D, x % D) = rn(s);
  }
  const int jf = si >= 0 ? (pos_in & (C - 1)) : 0;
  for (int x = tid; x < jf * 16; x += NT) {
    const int i = x >> 4, c = x & 15;
    *reinterpret_cast<uint4 *>(KTile{p.st.kb(so, h)}.chunk(i, c)) =
        *reinterpret_cast<const uint4 *>(KTile{p.st.kb(si, h)}.chunk(i, c));
    *reinterpret_cast<uint4 *>(KTile{p.st.vb(so, h)}.chunk(i, c)) =
        *reinterpret_cast<const uint4 *>(KTile{p.st.vb(si, h)}.chunk(i, c));
    *reinterpret_cast<uint4 *>(KTile{p.st.db(so, h)}.chunk(i, c)) =
        *reinterpret_cast<const uint4 *>(KTile{p.st.db(si, h)}.chunk(i, c));
    *reinterpret_cast<float4 *>(p.st.xb(so, h) + i * C + c * 4) =
        *reinterpret_cast<const float4 *>(p.st.xb(si, h) + i * C + c * 4);
  }
  const int p_old = jf >= 16 ? (jf & 15) : 0;
  for (int x = tid; x < p_old * 16; x += NT) {
    const int i = x >> 4, c = x & 15;
    *reinterpret_cast<float4 *>(p.st.pb(so, h) + i * C + c * 4) =
        *reinterpret_cast<const float4 *>(p.st.pb(si, h) + i * C + c * 4);
  }
  if (tid < jf) {
    p.st.gb(so, h)[tid] = p.st.gb(si, h)[tid];
    p.st.bb(so, h)[tid] = p.st.bb(si, h)[tid];
  }
  if (tid < NPOS) p.st.pos(so, h)[tid] = jf == 0 && si < 0 ? 0 : pos_in;
}

struct SlabTiles {
  KTile kt, qt, wt, ut;
  MTile mt;
  ShTile sh;
  SlabT dt, dl;
  GateTile *gt;
  int wb;
};

template <bool SAVE, bool ALL, bool PRED, bool FUSE = false>
__device__ void slab_step(const Params &p, const SlabTiles &tl, const ChunkGeo &g, int h, int si, int so,
                          bool more, bool sh_cached, float (&ST)[8][2][4]) {
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int vw = warp + tl.wb;   // this warp's slab of sixteen value columns, across the whole head
  const int gq = lane >> 2, tq = lane & 3, v0 = 16 * vw, c0 = 2 * vw;
  const KTile kt = tl.kt, qt = tl.qt, wt = tl.wt, ut = tl.ut;
  const MTile mt = tl.mt;
  const ShTile sh = tl.sh;
  const SlabT dt = tl.dt, dl = tl.dl;
  const GateTile &gt = *tl.gt;
  const bool zero_start = g.k == 0 && si < 0;
  const bool need_state = g.complete() && (more || so >= 0);
  // Which m16 tiles of the token index hold a row of this call. For decode exactly one does.
  // `unsigned`, not `bool`: it is an operand of the `@p` guards below (`chain_step`), not a branch
  // condition, and one `setp` per distinct value is all ptxas emits for it.
  unsigned mneed[MT];
  #pragma unroll
  for (int m = 0; m < MT; ++m)
    mneed[m] = (ALL || (16 * m < g.r_hi && 16 * m + 16 > g.r_lo)) ? 1u : 0u;
  // Which m16 tiles of d / d~ have to be WRITTEN. Strictly fewer than the four:
  //   * a tile at or past r_hi is read by nothing -- the M d chain below stops at k-block m for row
  //     block m (M_ji = +0 for i > j), so no chain reaches a d row at or past 16 (mmax + 1);
  //   * a tile wholly before r_lo already holds the record's rows, copied in above byte for byte, so
  //     reading them back only to store them again is a round trip -- unless the boundary chain runs,
  //     which needs d~ = rn(d e(G_63 - G_i)) for those rows too.
  // A decode step writes ONE tile of four instead of four.
  const int mmax = (g.r_hi - 1) >> 4, mlo = need_state ? 0 : (g.r_lo >> 4);

  // ------------------------------------------------------------------ 0. the state, once per sequence
  // Unless S^ was cached the fp32 state comes in and S^ = rn(S) is written here; otherwise wy_chunk
  // prefetched the record's S^ image and the fp32 state is not read at all.
  if (g.k == 0 && !sh_cached) {
    #pragma unroll
    for (int dq = 0; dq < 8; ++dq)
      #pragma unroll
      for (int nt = 0; nt < 2; ++nt)
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int d = 16 * dq + gq + 8 * (e >> 1), v = v0 + 8 * nt + 2 * tq + (e & 1);
          ST[dq][nt][e] = si >= 0 ? p.st.S(si, h)[int64_t(v) * D + d] : 0.f;
        }
  }
  if (!sh_cached) {   // S^ = rn(S_c) from the register state, into this warp's own 16 rows of the tile
    #pragma unroll
    for (int dq = 0; dq < 8; ++dq) {
      uint32_t w[4];
      #pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        w[2 * nt] = rn_pair(ST[dq][nt][0], ST[dq][nt][1]);
        w[2 * nt + 1] = rn_pair(ST[dq][nt][2], ST[dq][nt][3]);
      }
      store_acc_t(sh, 16 * dq, v0, w, lane);
    }
  }
  // The open chunk's delta rows are already here: wy_chunk prefetched them into the dt tile with v, X,
  // P and S^, which it can do only because that tile sits outside the phase union (chunk.cuh).
  fence_async_smem();
  __syncwarp();

  // ------------------------------------------------------------------ 1. D^T = W S^T; 2. the deltas
  //                                              (and, under FUSE, the output seed o^T = q^ S^T beside it)
  float acc[MT][2][4], o[MT][2][4];
  #pragma unroll
  for (int m = 0; m < MT; ++m)
    #pragma unroll
    for (int nt = 0; nt < 2; ++nt)
      #pragma unroll
      for (int e = 0; e < 4; ++e) {
        acc[m][nt][e] = 0.f;
        if constexpr (FUSE) o[m][nt][e] = 0.f;
      }
  if (!zero_start) {
    #pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      uint32_t b[4];
      frag_b2(b, sh, v0, 16 * ks, lane);
      const uint32_t b0[2] = {b[0], b[1]}, b1[2] = {b[2], b[3]};
      #pragma unroll
      for (int m = 0; m < MT; ++m)
        chain_step<PRED>(acc[m][0], acc[m][1], wt, 16 * m, 16 * ks, lane, b0, b1, mneed[m]);
      if constexpr (FUSE) {
        #pragma unroll
        for (int m = 0; m < MT; ++m)
          chain_step<PRED>(o[m][0], o[m][1], qt, 16 * m, 16 * ks, lane, b0, b1, mneed[m]);
      }
    }
    if constexpr (FUSE) out_seed_scale(o, gt, gq);
  }
  // d~ is derived at store time rather than kept beside d: a second array of bits costs registers this
  // kernel does not have (it runs at the 255-register cap, and past it -- see README.md on the spill).
  #pragma unroll
  for (int m = 0; m < MT; ++m) {
    if (!ALL && (m < mlo || m > mmax)) continue;
    // Both fragment loads carry their own `@p`, so `uf` and `df` are asm outputs on every path and the
    // guard never becomes a branch. A fragment the predicate skipped is read below and SELECTED AWAY.
    uint32_t uf[4], df[4];
    if (!PRED) {
      if (mneed[m]) frag_a(uf, ut, 16 * m, v0, lane);                     // u, in the accumulator's order
      if (!ALL && 16 * m < g.r_lo) frag_a(df, dt, 16 * m, v0, lane);      // the record's rows, unchanged
    } else {
      frag_a_p(uf, ut, 16 * m, v0, lane, mneed[m]);
      frag_a_p(df, dt, 16 * m, v0, lane, 16 * m < g.r_lo ? 1u : 0u);
    }
    const float lam0 = gt.lam[16 * m + gq], lam1 = gt.lam[16 * m + gq + 8];
    uint32_t wd[4], wt[4];
    #pragma unroll
    for (int nt = 0; nt < 2; ++nt)
      #pragma unroll
      for (int eh = 0; eh < 2; ++eh) {
        const int e0 = 2 * eh, i = 16 * m + gq + 8 * eh;
        const float lam = eh ? lam1 : lam0;
        uint32_t d;
        if (!PRED) {
          if (!ALL && i < g.r_lo) d = pack(frag_at(df, nt, e0), frag_at(df, nt, e0 + 1));
          else if (!ALL && i >= g.r_hi) d = 0u;
          else d = delta_entry2(frag_at(uf, nt, e0), acc[m][nt][e0],
                                frag_at(uf, nt, e0 + 1), acc[m][nt][e0 + 1]);
        } else {
          d = delta_entry2(frag_at(uf, nt, e0), acc[m][nt][e0],
                           frag_at(uf, nt, e0 + 1), acc[m][nt][e0 + 1]);
          const uint32_t rec = pack(frag_at(df, nt, e0), frag_at(df, nt, e0 + 1));
          d = i < g.r_lo ? rec : (i >= g.r_hi ? 0u : d);
        }
        wd[2 * nt + eh] = d;
        wt[2 * nt + eh] = dtilde_entry2(d, lam);
      }
    store_acc(dt, 16 * m, v0, wd, lane);
    store_acc(dl, 16 * m, v0, wt, lane);
  }
  fence_async_smem();
  __syncwarp();

  // ------------------------------------------------------------------ 3. the boundary state
  if (g.complete() && (more || so >= 0)) {
    const float gam_last = gt.gam[C - 1];
    #pragma unroll
    for (int dq = 0; dq < 8; ++dq)
      #pragma unroll
      for (int nt = 0; nt < 2; ++nt)
        #pragma unroll
        for (int e = 0; e < 4; ++e) ST[dq][nt][e] = state_seed(ST[dq][nt][e], gam_last);
    #pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      uint32_t b[4];
      frag_b2_t(b, dl, v0, 16 * ks, lane);
      const uint32_t b0[2] = {b[0], b[1]}, b1[2] = {b[2], b[3]};
      #pragma unroll
      for (int dq = 0; dq < 8; ++dq) {
        uint32_t a[4];
        frag_a_t(a, kt, 16 * dq, 16 * ks, lane);
        mma(ST[dq][0], a, b0);
        mma(ST[dq][1], a, b1);
      }
    }
  }

  // ------------------------------------------------------------------ 4. outputs (seeded by the old S^)
  if constexpr (!FUSE) {
    #pragma unroll
    for (int m = 0; m < MT; ++m)
      #pragma unroll
      for (int nt = 0; nt < 2; ++nt)
        #pragma unroll
        for (int e = 0; e < 4; ++e) o[m][nt][e] = 0.f;
    if (!zero_start) {
      #pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        uint32_t b[4];
        frag_b2(b, sh, v0, 16 * ks, lane);
        const uint32_t b0[2] = {b[0], b[1]}, b1[2] = {b[2], b[3]};
        #pragma unroll
        for (int m = 0; m < MT; ++m)
          chain_step<PRED>(o[m][0], o[m][1], qt, 16 * m, 16 * ks, lane, b0, b1, mneed[m]);
      }
      out_seed_scale(o, gt, gq);
    }
  }
  // `m < ks` stops each row block's chain at its own k-block: M_ji is +0 for i > j, so for rows
  // 16m .. 16m + 15 every coefficient in k-block ks > m is +0 and the step is skippable, by the same
  // measured pair of facts the u / w prune rests on (chunk.cuh). It is the ROW BLOCK that fixes the
  // bound, not the call's length, so every call shape prunes identically and they agree bit for bit.
  // Twenty k-steps instead of thirty-two over a full chunk; one instead of four for a decode step.
  #pragma unroll
  for (int ks = 0; ks < 4; ++ks) {
    uint32_t b[4];
    frag_b2_t(b, dt, v0, 16 * ks, lane);
    const uint32_t b0[2] = {b[0], b[1]}, b1[2] = {b[2], b[3]};
    #pragma unroll
    for (int m = 0; m < MT; ++m) {
      if (m < ks) continue;   // compile-time: the row block's chain stops at its own k-block
      chain_step<PRED>(o[m][0], o[m][1], mt, 16 * m, 16 * ks, lane, b0, b1, mneed[m]);
    }
  }

  // ------------------------------------------------------------------ 5. trainer saves off the old S^
  if (SAVE && p.sv.h) {
    uint16_t *hd = p.sv.h + (int64_t(p.chunk_base[g.n] + g.k) * p.hv + h) * D * D;
    for (int x = lane; x < 16 * 16; x += 32) {
      const int v = v0 + (x >> 4), c = x & 15;
      *reinterpret_cast<uint4 *>(hd + v * D + c * 8) = *reinterpret_cast<const uint4 *>(sh.chunk(v, c));
    }
  }
  if (SAVE && p.sv.v_new) {
    for (int x = lane; x < (g.r_hi - g.r_lo) * 2; x += 32) {
      const int i = g.r_lo + (x >> 1), c = c0 + (x & 1);
      *reinterpret_cast<uint4 *>(p.sv.v_new + (g.token(i) * p.hv + h) * D + 8 * c) =
          *reinterpret_cast<const uint4 *>(dt.chunk(i, c));
    }
  }

  // ------------------------------------------------------------------ 6. the record, when the sequence ends
  if (!more && so >= 0) {
    if (!sh_cached) {   // the register state may differ from the slot's: S and its image go out
      #pragma unroll
      for (int dq = 0; dq < 8; ++dq) {   // this warp's own 16 rows of S^, which its chains have read
        uint32_t w[4];
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
          w[2 * nt] = rn_pair(ST[dq][nt][0], ST[dq][nt][1]);
          w[2 * nt + 1] = rn_pair(ST[dq][nt][2], ST[dq][nt][3]);
        }
        store_acc_t(sh, 16 * dq, v0, w, lane);
      }
      #pragma unroll
      for (int dq = 0; dq < 8; ++dq)
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt)
          #pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int d = 16 * dq + gq + 8 * (e >> 1), v = v0 + 8 * nt + 2 * tq + (e & 1);
            p.st.S(so, h)[int64_t(v) * D + d] = ST[dq][nt][e];
          }
      fence_async_smem();
      __syncwarp();
      for (int x = lane; x < 16 * 16; x += 32) {
        const int v = v0 + (x >> 4), c = x & 15;
        *reinterpret_cast<uint4 *>(ShTile{p.st.sh(so, h)}.chunk(v, c)) =
            *reinterpret_cast<const uint4 *>(sh.chunk(v, c));
      }
    }
    const OpenRows open = open_rows(g, si, so);
    for (int x = lane; x < (open.jf - open.r_first) * 2; x += 32) {
      const int i = open.r_first + (x >> 1), c = c0 + (x & 1);
      const uint16_t *vsrc = i < open.r_new ? KTile{p.st.vb(si, h)}.chunk(i, c)
                                           : p.v + g.token(i) * p.vs0 + h * p.vs1 + 8 * c;
      *reinterpret_cast<uint4 *>(KTile{p.st.vb(so, h)}.chunk(i, c)) =
          *reinterpret_cast<const uint4 *>(vsrc);
      *reinterpret_cast<uint4 *>(KTile{p.st.db(so, h)}.chunk(i, c)) =
          *reinterpret_cast<const uint4 *>(dt.chunk(i, c));
    }
  }

  // ------------------------------------------------------------------ 7. the outputs out
  #pragma unroll
  for (int m = 0; m < MT; ++m) {
    if (!ALL && !mneed[m]) continue;
    uint32_t w[4];
    #pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      w[2 * nt] = rn_pair(o[m][nt][0], o[m][nt][1]);
      w[2 * nt + 1] = rn_pair(o[m][nt][2], o[m][nt][3]);
    }
    store_acc(dl, 16 * m, v0, w, lane);
  }
  fence_async_smem();
  __syncwarp();
  for (int x = lane; x < (g.r_hi - g.r_lo) * 2; x += 32) {
    const int j = g.r_lo + (x >> 1), c = c0 + (x & 1);
    *reinterpret_cast<uint4 *>(p.out + g.token(j) * p.os0 + h * p.os1 + 8 * c) =
        *reinterpret_cast<const uint4 *>(dl.chunk(j, c));
  }
}

}  // namespace gdn_cpr_sm100
