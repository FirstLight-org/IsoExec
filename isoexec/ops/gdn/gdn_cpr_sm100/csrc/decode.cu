// SPDX-License-Identifier: Apache-2.0
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <atomic>
#include <type_traits>

#include "chunk.cuh"

namespace gdn_cpr_sm100 {
namespace {

//: Warps per decode CTA. Four is what the scalar phases want (`X_j.` is 64 threads of P entries plus 64
//: of the diagonal block, so 128 threads cover both) and what the chain phases want (8 d-tiles of `w`
//: and 8 value tiles of `u` / the delta chain deal 2 to a warp at R = 1). Eight was measured and is
//: slower: it doubles the barrier population without giving any phase more work to do.
constexpr int DEC_WARPS = 4;
constexpr int DEC_THREADS = 32 * DEC_WARPS;

// An A fragment whose k index is a CHUNK ROW: elements at rows past `j` become +0. The recipe's zero
// extension says a product with a +0 coefficient contributes +0 and therefore that the other operand
// only has to be FINITE; on this path the other operand is a shared-memory row the step never loaded, so
// it is made +0 here instead of on the load path. `a[q]` holds k = i0 + 2t + 8 (q >> 1) in its low half
// and the next row in its high half (`recipe.cuh`, the fragment layout).
__device__ __forceinline__ void mask_rows(uint32_t (&a)[4], int i0, int j, int tq) {
  #pragma unroll
  for (int q = 0; q < 4; ++q) {
    const int i = i0 + 2 * tq + 8 * (q >> 1);
    a[q] &= (i <= j ? 0x0000FFFFu : 0u) | (i + 1 <= j ? 0xFFFF0000u : 0u);
  }
}

__device__ __forceinline__ void frag_b_cols(uint32_t (&b)[2], const uint16_t *c0, const uint16_t *c1,
                                            int k0, int lane) {
  const int g = lane >> 2, t = lane & 3;
  const bool one = g == 1 && c1 != nullptr;
  const uint16_t *c = one ? c1 : c0;            // never null: the dead lanes read column 0 and discard it
  const uint32_t w0 = *reinterpret_cast<const uint32_t *>(c + k0 + 2 * t);
  const uint32_t w1 = *reinterpret_cast<const uint32_t *>(c + k0 + 2 * t + 8);
  const bool live = g == 0 || one;
  b[0] = live ? w0 : 0u;
  b[1] = live ? w1 : 0u;
}

// ------------------------------------------------------------------ the recurrent step's shared memory
// 72 KB, which is THREE CTAs PER SM against the chunk-form decode kernel's one at 190 KB. The three
// bf16 tiles are full width (all 128 columns) even when a CTA owns fewer value rows: the tile's address
// map is its swizzle and a column subset is not a smaller tile, so a narrow CTA simply leaves the other
// chunks unloaded and unread. At R = 1 -- the large-batch case, where occupancy is what matters -- all
// 128 columns are needed anyway.
struct DecSmem {
  uint16_t k_[C * D];     // Tile<64,128> [i][d]: k^ rows [0, j]
  uint16_t v_[C * D];     // Tile<64,128> [i][v]: v rows [0, j], this CTA's columns
  uint16_t d_[C * D];     // Tile<64,128> [i][v]: the deltas, this CTA's columns
  float X[C * C];         // rows [0, j] of X = (I + L)^-1
  float P[16 * C];        // rows of P of the open 16-row block, row 16B + r at r
  float G[C], beta[C], kk[C], kq[C], L[C], dec[C], gam[C];
  __align__(16) uint16_t mrow[C];      // B-operand columns over the chunk row index: M_j., xu_j., xw_j.
  __align__(16) uint16_t xu[C];
  __align__(16) uint16_t xw[C];
  __align__(16) uint16_t khat[D];      // B-operand columns over d: k^_j, q^_j, w_j
  __align__(16) uint16_t qhat[D];
  __align__(16) uint16_t wv[D];
  __align__(16) uint16_t ov[D];        // o_j, staged so the output store is one 16-byte store a chunk
  float lg_j, beta_j;

  __device__ __forceinline__ KTile k() { return KTile{k_}; }
  __device__ __forceinline__ KTile v() { return KTile{v_}; }
  __device__ __forceinline__ KTile d() { return KTile{d_}; }
};

template <int R>
__global__ __launch_bounds__(DEC_THREADS) void gdn_decode_rec(Params p) {
  constexpr int VR = D / R;             // value rows of this CTA
  constexpr int VT = VR / 16;           // its 16-row value tiles
  constexpr int WT = VT >= DEC_WARPS ? VT / DEC_WARPS : 1;   // value tiles per warp
  constexpr int VP = NPOS / R;          // the position copies this CTA owns
  constexpr int DT = (D / 16) / DEC_WARPS;                   // d-row tiles of `w` per warp
  static_assert(VR >= 16 && NPOS % R == 0, "the value split must leave whole tiles and pos copies");
  extern __shared__ __align__(1024) char raw[];
  DecSmem &sm = *reinterpret_cast<DecSmem *>(raw);
  const int r = blockIdx.x % R, item = blockIdx.x / R;
  const int seq = item / p.hv, h = item - seq * p.hv;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int gq = lane >> 2, tq = lane & 3;
  const int hq = p.key_head(h);
  const int v0 = r * VR;
  const KTile kt = sm.k(), vt = sm.v(), dt = sm.d();
  const int si = p.slot_in_of(seq), so = p.slot_out_of(seq);
  const bool wact = warp < VT;

  // The token's own inputs do not depend on the slot, so they are issued BEFORE the slot -> position
  // chase that gates everything read out of the record.
  uint2 raw_qk = make_uint2(0u, 0u);
  if (warp < 2)
    raw_qk = *reinterpret_cast<const uint2 *>(
        (warp == 0 ? p.k + int64_t(seq) * p.ks0 + hq * p.ks1 : p.q + int64_t(seq) * p.qs0 + hq * p.qs1) +
        4 * lane);
  uint16_t a_raw = 0, b_raw = 0, A16 = 0, dt16 = 0;
  float A32 = 0.f, dt32 = 0.f;
  if (tid == 64) {
    a_raw = p.a[int64_t(seq) * p.as0 + h * p.as1];
    b_raw = p.b[int64_t(seq) * p.bs0 + h * p.bs1];
    if (p.param_bf16) {
      A16 = static_cast<const uint16_t *>(p.A_log)[h];
      dt16 = static_cast<const uint16_t *>(p.dt_bias)[h];
    } else {
      A32 = static_cast<const float *>(p.A_log)[h];
      dt32 = static_cast<const float *>(p.dt_bias)[h];
    }
  }
  // A NULL input slot is the fresh (zero) state and reads nothing; a NULL output slot writes no record
  // byte. Neither is an early return: the chunk-form kernel computes and STORES the output row of a NULL
  // lane (from the zero state), and this kernel is required to be bit-identical to it everywhere.
  const int pos = si >= 0 ? p.st.pos(si, h)[r * VP] : 0;
  const int j = pos & (C - 1), B = j >> 4;

  // `S^` of this warp's value tiles as A fragments, straight from the record image into registers. The
  // row index does not depend on `pos`, so these are issued under the position load rather than after it.
  // Four 32-bit words per (tile, k16 step): row m0 + g and m0 + g + 8, chunks 2ks and 2ks + 1 of the
  // Tile image, at chunk position `c ^ (row & 7)` -- and (m0 + g + 8) & 7 == g, so both rows share the
  // XOR pattern (`tile.cuh`).
  uint32_t sa[WT][8][4];
  {
    const bool have = si >= 0 && wact;
    int off0[8], off1[8];
    #pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      off0[ks] = (((2 * ks) ^ gq) << 4) + 4 * tq;
      off1[ks] = (((2 * ks + 1) ^ gq) << 4) + 4 * tq;
    }
    #pragma unroll
    for (int u = 0; u < WT; ++u) {
      const int row = v0 + 16 * (warp + DEC_WARPS * u) + gq;
      const char *r0 = have ? reinterpret_cast<const char *>(p.st.sh(si, h)) + int64_t(row) * (D * 2)
                            : reinterpret_cast<const char *>(p.q);
      const char *r1 = r0 + (have ? 8 * (D * 2) : 0);
      #pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        if (have) {
          sa[u][ks][0] = *reinterpret_cast<const uint32_t *>(r0 + off0[ks]);
          sa[u][ks][1] = *reinterpret_cast<const uint32_t *>(r1 + off0[ks]);
          sa[u][ks][2] = *reinterpret_cast<const uint32_t *>(r0 + off1[ks]);
          sa[u][ks][3] = *reinterpret_cast<const uint32_t *>(r1 + off1[ks]);
        } else {
          sa[u][ks][0] = sa[u][ks][1] = sa[u][ks][2] = sa[u][ks][3] = 0u;
        }
      }
    }
  }

  // ---------------------------------------------------------------- 0. the rows present, and only those
  // Rows at or past `j` are never loaded and never cleared: the chains stop at block `B` and `mask_rows`
  // makes the operand elements past `j` inside that block +0.
  const int pr = B > 0 ? (j & 15) : 0;          // P rows of the open block that exist
  constexpr int VC = VR / 8;                    // this CTA's 16-byte column chunks of v and the deltas
  if (tid < C) {   // G and beta of rows [0, j); rows at or past j are +0, which is what zeroes xu / xw
    sm.G[tid] = (si >= 0 && tid < j) ? p.st.gb(si, h)[tid] : 0.f;
    sm.beta[tid] = (si >= 0 && tid < j) ? p.st.bb(si, h)[tid] : 0.f;
  }
  if (si >= 0) {
    const KTile rk{p.st.kb(si, h)}, rv{p.st.vb(si, h)}, rd{p.st.db(si, h)};
    const float *rx = p.st.xb(si, h), *rp = p.st.pb(si, h);
    for (int x = tid; x < j * 16; x += DEC_THREADS) {
      const int i = x >> 4, c = x & 15;
      cp16(smem_u32(kt.chunk(i, c)), rk.chunk(i, c), true);
      cp16(smem_u32(sm.X + int64_t(i) * C + c * 4), rx + int64_t(i) * C + c * 4, true);
    }
    for (int x = tid; x < j * VC; x += DEC_THREADS) {
      const int i = x / VC, c = v0 / 8 + (x % VC);
      cp16(smem_u32(vt.chunk(i, c)), rv.chunk(i, c), true);
      cp16(smem_u32(dt.chunk(i, c)), rd.chunk(i, c), true);
    }
    for (int x = tid; x < pr * 16; x += DEC_THREADS) {
      const int i = x >> 4, c = x & 15;
      cp16(smem_u32(sm.P + int64_t(i) * C + c * 4), rp + int64_t(i) * C + c * 4, true);
    }
  }
  for (int cc = tid; cc < VC; cc += DEC_THREADS)     // v_j, the token's own value row, into the tile
    cp16(smem_u32(vt.chunk(j, v0 / 8 + cc)),
         p.v + int64_t(seq) * p.vs0 + h * p.vs1 + v0 + cc * 8, true);
  cp_commit();

  // ---------------------------------------------------------------- 1. k^_j, q^_j, the gate (over the loads)
  if (warp < 2) {
    const float xr[1][CE] = {{bf(lo16(raw_qk.x)), bf(hi16(raw_qk.x)), bf(lo16(raw_qk.y)),
                              bf(hi16(raw_qk.y))}};
    uint2 nrm[1];
    l2norm_rows<1, false>(xr, p.scale, warp == 1, nrm);
    *reinterpret_cast<uint2 *>((warp == 0 ? sm.khat : sm.qhat) + 4 * lane) = nrm[0];
    if (warp == 0)   // row j of the k^ tile: the key products, w_j and the boundary all read it
      *reinterpret_cast<uint2 *>(kt.chunk(j, lane >> 1) + 4 * (lane & 1)) = nrm[0];
  } else if (tid == 64) {
    const Gate gt = gate(bf(a_raw), bf(b_raw), p.param_bf16 ? bf(A16) : A32,
                         p.param_bf16 ? bf(dt16) : dt32);
    sm.lg_j = gt.lg;
    sm.beta_j = gt.beta;
  }
  cp_wait<0>();
  __syncthreads();

  // ---------------------------------------------------------------- 2. KK_.j, KQ_.j
  const float G_j = cum_gate(j > 0 ? sm.G[j - 1] : 0.f, sm.lg_j), beta_j = sm.beta_j;
  if (tid == 0) {
    sm.G[j] = G_j;
    sm.beta[j] = beta_j;
  }
  if (16 * warp <= j) {   // CHAIN128 over d of key rows [16w, 16w + 16) against the columns (k^_j, q^_j)
    // EVERY FRAGMENT IS LOADED BEFORE THE FIRST MMA, and that is worth 1,400 cycles of a 9,600-cycle step.
    // The eight k16 steps accumulate into ONE accumulator, so they are strictly dependent -- their floor is
    // eight MMA latencies, about 200 cycles -- while their operand loads are entirely independent. Left
    // interleaved, the measured phase was 1,781 cycles: each step waited out its own `ldmatrix` and its two
    // `LDS.32`. Hoisted, the loads pipeline under the chain. It is the same shape `wu` below already had
    // and the same lesson this package records for the X solve and the l2 norm.
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    uint32_t a[8][4], b[8][2];
    #pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      frag_a(a[ks], kt, 16 * warp, 16 * ks, lane);
      frag_b_cols(b[ks], sm.khat, sm.qhat, 16 * ks, lane);
    }
    #pragma unroll
    for (int ks = 0; ks < 8; ++ks) mma(acc, a[ks], b[ks]);
    if (tq == 0) {
      sm.kk[16 * warp + gq] = acc[0];
      sm.kq[16 * warp + gq] = acc[1];
      sm.kk[16 * warp + gq + 8] = acc[2];
      sm.kq[16 * warp + gq + 8] = acc[3];
    }
  }
  __syncthreads();

  // ---------------------------------------------------------------- 3. L_j., M_j., the decays
  if (tid < C) {
    const int i = tid;
    const float dc = decay(G_j, sm.G[i]);
    sm.L[i] = i < j ? l_entry(sm.kk[i], beta_j, dc) : 0.f;
    sm.mrow[i] = i <= j ? m_entry(sm.kq[i], dc) : uint16_t(0);
    sm.dec[i] = dc;                  // at the boundary: d~'s factor e(G_63 - G_i)
    sm.gam[i] = decay0(sm.G[i]);     // xw's factor e(G_i)
  }
  __syncthreads();

  // ---------------------------------------------------------------- 4. X_j., xu_j., xw_j.
  const int m = tid;
  if (m < 16 * B) {
    sm.P[(j & 15) * C + m] =
        p_entry<12>(B, [&](int i) { return sm.L[i]; }, [&](int i) { return sm.X[i * C + m]; });
  } else if (m < C) {
    float l[16], xb[16];
    #pragma unroll
    for (int rr = 0; rr < 16; ++rr) {
      l[rr] = m < 16 * B + 16 ? sm.L[16 * B + rr] : 0.f;
      xb[rr] = (m < 16 * B + 16 && 16 * B + rr < j) ? sm.X[(16 * B + rr) * C + m] : 0.f;
    }
    const float x = m < j ? x_block(l, xb) : (m == j ? 1.f : 0.f);
    sm.X[j * C + m] = x;
    sm.xu[m] = xu_entry(x, sm.beta[m]);
    sm.xw[m] = xw_entry(x, sm.beta[m], sm.gam[m]);
  }
  __syncthreads();
  if (B > 0) {   // the earlier blocks: X_jm = -T4 CH4(Y_j., P_.m). Uniform over the CTA, so no split path.
    if (m < 16 * B) {
      float y[16], pc[16];
      #pragma unroll
      for (int rr = 0; rr < 16; ++rr) {
        y[rr] = sm.X[j * C + 16 * B + rr];
        pc[rr] = rr <= (j & 15) ? sm.P[rr * C + m] : 0.f;
      }
      const float x = x_block(y, pc);
      sm.X[j * C + m] = x;
      sm.xu[m] = xu_entry(x, sm.beta[m]);
      sm.xw[m] = xw_entry(x, sm.beta[m], sm.gam[m]);
    }
    __syncthreads();
  }

  // ---------------------------------------------------------------- 5. the key side of the record
  // It depends on nothing the value side computes, so it runs here rather than at the end, under the
  // w / u chains' latency. CTA 0 of the head owns it; the value sections follow the deltas.
  const int jf = (j + 1) & (C - 1);            // open rows the step leaves behind (0 when the chunk closes)
  const bool inplace = so >= 0 && so == si;
  const int r_first = inplace ? j : 0;         // in place, only row j is new; a slot move carries [0, j]
  if (so >= 0 && r == 0 && jf > r_first) {
    const KTile ok{p.st.kb(so, h)};
    for (int x = tid; x < (jf - r_first) * 16; x += DEC_THREADS) {
      const int i = r_first + (x >> 4), c = x & 15;
      *reinterpret_cast<uint4 *>(ok.chunk(i, c)) = *reinterpret_cast<const uint4 *>(kt.chunk(i, c));
      *reinterpret_cast<float4 *>(p.st.xb(so, h) + int64_t(i) * C + c * 4) =
          *reinterpret_cast<const float4 *>(sm.X + i * C + c * 4);
    }
    const int pb0 = jf & ~15;
    if (pb0 > 0) {
      const int lo = max(r_first, pb0);
      for (int x = tid; x < (jf - lo) * 16; x += DEC_THREADS) {
        const int i = lo + (x >> 4), c = x & 15;
        *reinterpret_cast<float4 *>(p.st.pb(so, h) + int64_t(i - pb0) * C + c * 4) =
            p_row_chunk(sm.P + (i - pb0) * C + c * 4, c, pb0);
      }
    }
    if (tid >= r_first && tid < jf) {
      p.st.gb(so, h)[tid] = sm.G[tid];
      p.st.bb(so, h)[tid] = sm.beta[tid];
    }
  }

  // ---------------------------------------------------------------- 6. w_j (all d), u_j (this CTA's rows)
  // The step count is `B + 1` and it is a TEMPLATE constant per branch, so every fragment load of a chain
  // issues before its first MMA and no runtime predicate sits between two dependent MMAs.
  auto chains = [&](auto steps) {
    constexpr int NS = decltype(steps)::value;
    uint16_t uvr[WT][2];
    #pragma unroll
    for (int u = 0; u < DT; ++u) {      // w_j = rn(CHAIN64_i(k^_i, xw_ji)), 8 tiles of 16 d rows
      const int d0 = 16 * (warp + DEC_WARPS * u);
      uint32_t a[NS][4], b[NS][2];
      #pragma unroll
      for (int ks = 0; ks < NS; ++ks) {
        frag_a_t(a[ks], kt, d0, 16 * ks, lane);
        frag_b_cols(b[ks], sm.xw, nullptr, 16 * ks, lane);
      }
      mask_rows(a[NS - 1], 16 * (NS - 1), j, tq);
      float cw[4] = {0.f, 0.f, 0.f, 0.f};
      #pragma unroll
      for (int ks = 0; ks < NS; ++ks) mma(cw, a[ks], b[ks]);
      if (tq == 0) {
        sm.wv[d0 + gq] = rn(cw[0]);
        sm.wv[d0 + gq + 8] = rn(cw[2]);
      }
    }
    if (wact)
      #pragma unroll
      for (int u = 0; u < WT; ++u) {   // u_j = rn(CHAIN64_i(v_i, xu_ji)), this CTA's value tiles
        const int t0 = 16 * (warp + DEC_WARPS * u);
        uint32_t a[NS][4], b[NS][2];
        #pragma unroll
        for (int ks = 0; ks < NS; ++ks) {
          frag_a_t(a[ks], vt, v0 + t0, 16 * ks, lane);
          frag_b_cols(b[ks], sm.xu, nullptr, 16 * ks, lane);
        }
        mask_rows(a[NS - 1], 16 * (NS - 1), j, tq);
        float cu[4] = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int ks = 0; ks < NS; ++ks) mma(cu, a[ks], b[ks]);
        if (tq == 0) {
          uvr[u][0] = rn(cu[0]);
          uvr[u][1] = rn(cu[2]);
        }
      }
    __syncthreads();   // `wv` is a B-operand column every warp reads

    // -------------------------------------------------------------- 7. d_j and the output seed
    // ONE k-loop for both: `CHAIN128_d(S^, w_j)` and `CHAIN128_d(S^, q^_j)` share the A operand `S^` and
    // differ only in the B column, so an n8 tile whose columns 0 and 1 are `w_j` and `q^_j` evaluates
    // both in the recipe's ascending k order with each accumulator seeing exactly its own addends.
    float sd[WT][2];
    if (wact)
      #pragma unroll
      for (int u = 0; u < WT; ++u) {
        const int t0 = 16 * (warp + DEC_WARPS * u);
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        uint32_t b[8][2];
        #pragma unroll
        for (int ks = 0; ks < 8; ++ks) frag_b_cols(b[ks], sm.wv, sm.qhat, 16 * ks, lane);
        #pragma unroll
        for (int ks = 0; ks < 8; ++ks) mma(acc, sa[u][ks], b[ks]);
        if (tq == 0) {
          const float gam_j = decay0(G_j);
          #pragma unroll
          for (int rr = 0; rr < 2; ++rr) {
            const int vr = t0 + gq + 8 * rr;
            dt.at(j, v0 + vr) = delta_entry(uvr[u][rr], acc[2 * rr]);
            sd[u][rr] = out_seed(acc[2 * rr + 1], gam_j);
          }
        }
      }
    fence_async_smem();
    __syncwarp();   // row j of the deltas and the seeds are this warp's own value columns

    // -------------------------------------------------------------- 8. o_j = rn(CHAIN64(d_i, M_ji; seed))
    if (wact)
      #pragma unroll
      for (int u = 0; u < WT; ++u) {
        const int t0 = 16 * (warp + DEC_WARPS * u);
        float acc[4] = {tq == 0 ? sd[u][0] : 0.f, 0.f, tq == 0 ? sd[u][1] : 0.f, 0.f};
        uint32_t a[NS][4], b[NS][2];
        #pragma unroll
        for (int ks = 0; ks < NS; ++ks) {
          frag_a_t(a[ks], dt, v0 + t0, 16 * ks, lane);
          frag_b_cols(b[ks], sm.mrow, nullptr, 16 * ks, lane);
        }
        mask_rows(a[NS - 1], 16 * (NS - 1), j, tq);
        #pragma unroll
        for (int ks = 0; ks < NS; ++ks) mma(acc, a[ks], b[ks]);
        if (tq == 0) {
          sm.ov[t0 + gq] = rn(acc[0]);
          sm.ov[t0 + gq + 8] = rn(acc[2]);
        }
      }
  };
  switch (B) {
    case 0: chains(std::integral_constant<int, 1>{}); break;
    case 1: chains(std::integral_constant<int, 2>{}); break;
    case 2: chains(std::integral_constant<int, 3>{}); break;
    default: chains(std::integral_constant<int, 4>{}); break;
  }
  __syncthreads();
  for (int cc = tid; cc < VC; cc += DEC_THREADS)
    *reinterpret_cast<uint4 *>(p.out + int64_t(seq) * p.os0 + h * p.os1 + v0 + cc * 8) =
        *reinterpret_cast<const uint4 *>(sm.ov + cc * 8);

  // ---------------------------------------------------------------- 9. the value side of the record
  if (so >= 0 && jf > r_first) {
    const KTile ov{p.st.vb(so, h)}, od{p.st.db(so, h)};
    for (int x = tid; x < (jf - r_first) * VC; x += DEC_THREADS) {
      const int i = r_first + x / VC, c = v0 / 8 + (x % VC);
      *reinterpret_cast<uint4 *>(ov.chunk(i, c)) = *reinterpret_cast<const uint4 *>(vt.chunk(i, c));
      *reinterpret_cast<uint4 *>(od.chunk(i, c)) = *reinterpret_cast<const uint4 *>(dt.chunk(i, c));
    }
  }

  // ---------------------------------------------------------------- 10. the boundary, or a state move
  if (so >= 0 && j == C - 1) {
    // S_{c+1} = CHAIN64_i(d~_i, k^_i; S_c (*) e(G_63)), evaluated with the value rows as the MMA's M and
    // the d columns as its N -- the transpose of `slab.cuh`'s orientation, bit for bit.
    const float gam_last = decay0(G_j);
    if (wact)
      #pragma unroll
      for (int u = 0; u < WT; ++u) {
        const int t0 = 16 * (warp + DEC_WARPS * u), row = v0 + t0 + gq;
        uint32_t ta[4][4];
        #pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
          uint32_t a[4];
          frag_a_t(a, dt, v0 + t0, 16 * ks, lane);
          #pragma unroll
          for (int q4 = 0; q4 < 4; ++q4) {
            const int i = 16 * ks + 2 * tq + 8 * (q4 >> 1);
            ta[ks][q4] = pack(dtilde_entry(lo16(a[q4]), sm.dec[i]),
                              dtilde_entry(hi16(a[q4]), sm.dec[i + 1]));
          }
        }
        const ShTile shw{p.st.sh(so, h)};
        #pragma unroll
        for (int half = 0; half < 2; ++half) {   // 64 state columns at a time: eight n8 accumulators
          float acc[8][4];
          #pragma unroll
          for (int nt = 0; nt < 8; ++nt) {
            const float *src = (si >= 0 ? p.st.S(si, h) : p.st.S(so, h)) + int64_t(row) * D + 64 * half +
                               8 * nt + 2 * tq;
            const float2 lo2 = si >= 0 ? *reinterpret_cast<const float2 *>(src) : make_float2(0.f, 0.f);
            const float2 hi2 = si >= 0 ? *reinterpret_cast<const float2 *>(src + 8 * D)
                                       : make_float2(0.f, 0.f);
            acc[nt][0] = state_seed(lo2.x, gam_last);
            acc[nt][1] = state_seed(lo2.y, gam_last);
            acc[nt][2] = state_seed(hi2.x, gam_last);
            acc[nt][3] = state_seed(hi2.y, gam_last);
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
          for (int nt = 0; nt < 8; ++nt) {
            const int n0 = 64 * half + 8 * nt;
            float *o = p.st.S(so, h) + int64_t(row) * D + n0 + 2 * tq;
            *reinterpret_cast<float2 *>(o) = make_float2(acc[nt][0], acc[nt][1]);
            *reinterpret_cast<float2 *>(o + 8 * D) = make_float2(acc[nt][2], acc[nt][3]);
            *reinterpret_cast<uint32_t *>(&shw.at(row, n0 + 2 * tq)) = rn2(acc[nt][0], acc[nt][1]);
            *reinterpret_cast<uint32_t *>(&shw.at(row + 8, n0 + 2 * tq)) = rn2(acc[nt][2], acc[nt][3]);
          }
        }
      }
  } else if (so >= 0 && so != si) {
    // The boundary state moves with the slot, exactly as the chunk-form kernel moves it: S is copied and
    // its image is rn(S), not a copy of the input image (the record's contract is sh == rn(S)).
    const ShTile shw{p.st.sh(so, h)};
    for (int x = tid; x < VR * (D / 2); x += DEC_THREADS) {
      const int vv = v0 + x / (D / 2), dd = 2 * (x % (D / 2));
      const float2 s = si >= 0 ? *reinterpret_cast<const float2 *>(p.st.S(si, h) + int64_t(vv) * D + dd)
                               : make_float2(0.f, 0.f);
      *reinterpret_cast<float2 *>(p.st.S(so, h) + int64_t(vv) * D + dd) = s;
      *reinterpret_cast<uint32_t *>(&shw.at(vv, dd)) = rn2(s.x, s.y);
    }
  }
  if (so >= 0 && tid < VP) p.st.pos(so, h)[r * VP + tid] = pos + 1;
}

template <int R>
void launch_rec(const Params &p, int items, cudaStream_t stream) {
  auto *fn = gdn_decode_rec<R>;
  static std::atomic<uint64_t> raised{0};
  int device = 0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  const uint64_t bit = uint64_t(1) << (device & 63);
  if (!(raised.load(std::memory_order_acquire) & bit)) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void *>(fn),
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, int(sizeof(DecSmem))));
    C10_CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void *>(fn),
                                        cudaFuncAttributePreferredSharedMemoryCarveout,
                                        cudaSharedmemCarveoutMaxShared));
    raised.fetch_or(bit, std::memory_order_release);
  }
  fn<<<items * R, DEC_THREADS, sizeof(DecSmem), stream>>>(p);
  C10_CUDA_CHECK(cudaGetLastError());
}

}  // namespace

int64_t decode_rec_smem_bytes() { return int64_t(sizeof(DecSmem)); }

// How many CTAs of the recurrent decode kernel an SM holds, from the hardware rather than from the
// arithmetic. `tests/test_composition_gpu.py` asserts it, because a growth in `DecSmem` that silently
// costs a CTA would show up only as a large-batch regression.
int64_t decode_rec_blocks_per_sm() {
  auto *fn = gdn_decode_rec<1>;
  static std::atomic<uint64_t> raised{0};
  int device = 0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  const uint64_t bit = uint64_t(1) << (device & 63);
  if (!(raised.load(std::memory_order_acquire) & bit)) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void *>(fn),
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, int(sizeof(DecSmem))));
    C10_CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void *>(fn),
                                        cudaFuncAttributePreferredSharedMemoryCarveout,
                                        cudaSharedmemCarveoutMaxShared));
    raised.fetch_or(bit, std::memory_order_release);
  }
  int blocks = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, reinterpret_cast<const void *>(fn),
                                                               DEC_THREADS, sizeof(DecSmem)));
  return blocks;
}

// `split` is the value-row split R, an EXECUTION choice on the call's shape (`ops.choose_decode_split`):
// every value produces the same bits and `tests/test_identity_gpu.py` gates 1 / 2 / 4 / 8 against each
// other and against the chunk-form kernel.
void decode_rec_launch(const Params &p, int split, int items, cudaStream_t stream) {
  if (items <= 0) return;
  switch (split) {
    case 1: launch_rec<1>(p, items, stream); break;
    case 2: launch_rec<2>(p, items, stream); break;
    case 4: launch_rec<4>(p, items, stream); break;
    case 8: launch_rec<8>(p, items, stream); break;
    default: TORCH_CHECK(false, "gdn_cpr_sm100: unsupported decode split ", split);
  }
}

}  // namespace gdn_cpr_sm100
