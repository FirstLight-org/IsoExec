// SPDX-License-Identifier: Apache-2.0
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <atomic>
#include <type_traits>

#include "slab.cuh"

namespace gdn_cpr_sm100 {

// ------------------------------------------------------------------ the hand-off, per (chunk, head)
// Every section is the byte image of the shared-memory tile that produced it and that consumes it, so
// both sides are the tile's own address map and the copy is `cp.async` in and 16-byte stores out.
// `k^` AND `q^` ARE LAST ON PURPOSE. They are the two sections the (chunk, KEY head) pre-kernel carries
// instead, so in that form the per-(chunk, value head) record is a PREFIX of this one and the allocation
// is `SHORT` rather than `BYTES` -- 40.5 KB against 72.5, which is what keeps the workspace from growing
// when a second buffer appears beside it.
struct Hand {
  static constexpr int64_t U = 0;                              // Tile<64,128> u  [j][v]
  static constexpr int64_t W = U + int64_t(C) * D * 2;         // Tile<64,128> w  [j][d]
  static constexpr int64_t M = W + int64_t(C) * D * 2;         // Tile<64,64>  M  [j][i]
  static constexpr int64_t GAM = M + int64_t(C) * C * 2;       // gam[64] fp32
  static constexpr int64_t LAM = GAM + int64_t(C) * 4;         // lam[64] fp32
  static constexpr int64_t SHORT = (LAM + int64_t(C) * 4 + 255) / 256 * 256;
  static constexpr int64_t K = SHORT;                          // Tile<64,128> k^ [i][d]
  static constexpr int64_t Q = K + int64_t(C) * D * 2;         // Tile<64,128> q^ [j][d]
  static constexpr int64_t BYTES = (Q + int64_t(C) * D * 2 + 255) / 256 * 256;
};

// The (chunk, KEY head) hand-off the pre-kernel publishes: the two normalized tiles and the two 64 x 64
// products in the accumulator's own fragment order (chunk.cuh, `kkq_store`).
struct HandK {
  static constexpr int64_t K = 0;                              // Tile<64,128> k^ [i][d]
  static constexpr int64_t Q = K + int64_t(C) * D * 2;         // Tile<64,128> q^ [j][d]
  static constexpr int64_t KK = Q + int64_t(C) * D * 2;        // KKQ_FLOATS fp32
  static constexpr int64_t KQ = KK + int64_t(KKQ_FLOATS) * 4;
  static constexpr int64_t BYTES = (KQ + int64_t(KKQ_FLOATS) * 4 + 255) / 256 * 256;
};

// The open-chunk carry, per (SEQUENCE, head) rather than per chunk: only the last chunk of a sequence
// produces it, and only when that chunk is incomplete.
struct Carry {
  static constexpr int64_t X = 0;                              // [64 j][64 m] fp32
  static constexpr int64_t P = X + int64_t(C) * C * 4;         // [16][64] fp32
  static constexpr int64_t G = P + int64_t(16) * C * 4;        // [64] fp32
  static constexpr int64_t BETA = G + int64_t(C) * 4;          // [64] fp32
  static constexpr int64_t BYTES = (BETA + int64_t(C) * 4 + 255) / 256 * 256;
};

struct ChunkRec {
  int n;      // sequence of this chunk slot, -1 past the pack's own chunks
  int k;      // chunk index inside that sequence
  int cs;     // p.cu[n]
  int len;    // p.cu[n + 1] - cs
  int j0;     // pos_in mod 64
  int si;     // input slot, -1 for NULL / stateless
  int so;     // output slot
  int pad;
};

// Not a member of `Params`: the one-launch kernels must keep the launch signature they had, so this
// rides as a second kernel argument of the prepared kernels alone.
struct PrepRef {
  char *hand;             // [maxch, hv, hstride]
  char *handk;            // [maxch, hk, HandK::BYTES], nullptr without the pre-kernel
  char *carry;            // [nseq, hv, Carry::BYTES]
  const ChunkRec *tab;    // [maxch]
  const int *base;        // [nseq + 1] first chunk slot of each sequence, then the total
  int maxch;
  int64_t hstride;        // Hand::SHORT with the pre-kernel, Hand::BYTES without
  __device__ __forceinline__ char *hd(int slot, int hv, int h) const {
    return hand + (int64_t(slot) * hv + h) * hstride;
  }
  __device__ __forceinline__ char *hdk(int slot, int hk, int hq) const {
    return handk + (int64_t(slot) * hk + hq) * HandK::BYTES;
  }
};

namespace {

struct PrepASmem {
  uint16_t k_[C * D];
  uint16_t q_[C * D];
  uint16_t m_[C * C];
  __align__(16) GateTile gt_;
  float lg_[C];
  uint16_t v_[C * D];
  uint16_t xu_[C * C];
  uint16_t xw_[C * C];
  union {
    struct {
      float L[C * LS];
      float X[C * C];
      float P[16 * C];
    } x;
    struct {
      uint16_t u[C * D];
      uint16_t w[C * D];
    } o;
  } t;

  __device__ __forceinline__ KTile k() { return KTile{k_}; }
  __device__ __forceinline__ KTile q() { return KTile{q_}; }
  __device__ __forceinline__ MTile m() { return MTile{m_}; }
  __device__ __forceinline__ KTile w() { return KTile{t.o.w}; }
  __device__ __forceinline__ KTile u() { return KTile{t.o.u}; }
  __device__ __forceinline__ KTile v() { return KTile{v_}; }
  __device__ __forceinline__ MTile xu() { return MTile{xu_}; }
  __device__ __forceinline__ MTile xw() { return MTile{xw_}; }
  __device__ __forceinline__ float *L() { return t.x.L; }
  __device__ __forceinline__ float *X() { return t.x.X; }
  __device__ __forceinline__ float *P() { return t.x.P; }
  __device__ __forceinline__ GateTile &gt() { return gt_; }
};
static_assert(sizeof(PrepASmem) == 112896, "launch A's layout must stay under 114 KB for two CTAs/SM");

// ------------------------------------------------------------------ the chunk table
// `nch` needs the input slot's position, which lives in device memory, so the host cannot build this
// table and cannot even know how many chunks the call has. It bounds the grid instead
// (`T / 64 + 2 nseq`, which `chunks_of` cannot exceed) and the slots past the pack's own chunks carry
// `cseq = -1` and their CTAs return. One CTA: the per-sequence reads are issued in parallel (one round
// trip, not `nseq` dependent ones) and only the prefix sum is serial.
__global__ void prep_plan_kernel(Params p, ChunkRec *tab, int *base, int maxch) {
  extern __shared__ int shm[];                 // nchs[nseq], then the prefix sum bs[nseq + 1]
  int *nchs = shm, *bs = shm + p.nseq;
  const int tid = threadIdx.x;
  for (int n = tid; n < p.nseq; n += THREADS) {
    const int si = p.slot_in_of(n);
    const int pos_in = si >= 0 ? p.st.pos(si, 0)[0] : 0;
    nchs[n] = chunks_of(p.cu[n + 1] - p.cu[n], pos_in & (C - 1));
  }
  __syncthreads();
  if (tid == 0) {
    int idx = 0;
    for (int n = 0; n < p.nseq; ++n) {
      bs[n] = idx;
      idx += nchs[n];
    }
    bs[p.nseq] = idx;
  }
  __syncthreads();
  for (int n = tid; n <= p.nseq; n += THREADS) base[n] = bs[n];   // launch B walks a sequence, not a chunk
  const int tot = bs[p.nseq];
  for (int x = tid; x < maxch; x += THREADS) {
    ChunkRec r{-1, 0, 0, 0, 0, -1, -1, 0};
    if (x < tot) {
      int lo = 0, hi = p.nseq;                 // the last n with bs[n] <= x
      while (lo + 1 < hi) {
        const int mid = (lo + hi) >> 1;
        if (bs[mid] <= x) lo = mid; else hi = mid;
      }
      r.n = lo;
      r.k = x - bs[lo];
      r.cs = p.cu[lo];
      r.len = p.cu[lo + 1] - r.cs;
      r.si = p.slot_in_of(lo);
      r.so = p.slot_out_of(lo);
      r.j0 = r.si >= 0 ? (p.st.pos(r.si, 0)[0] & (C - 1)) : 0;
    }
    tab[x] = r;
  }
}

struct PreKSmem {
  uint16_t k_[C * D];
  uint16_t q_[C * D];
  __device__ __forceinline__ KTile k() { return KTile{k_}; }
  __device__ __forceinline__ KTile q() { return KTile{q_}; }
};

__global__ __launch_bounds__(THREADS) void gdn_prep_k(Params p, PrepRef pr) {
  extern __shared__ __align__(1024) char raw[];
  PreKSmem &sm = *reinterpret_cast<PreKSmem *>(raw);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int gq = lane >> 2, tq = lane & 3;
  const int item = blockIdx.x;
  const int slot = item / p.hk, hq = item - slot * p.hk;
  const int4 r0 = reinterpret_cast<const int4 *>(pr.tab)[2 * slot];
  const int4 r1 = reinterpret_cast<const int4 *>(pr.tab)[2 * slot + 1];   // issued before the exit test
  const int n = r0.x;
  if (n < 0) return;
  const int si = r1.y;
  const ChunkGeo g = chunk_geo(n, r0.y, r0.z, r0.w, r1.x);
  const int hlead = hq * (p.hv / p.hk);          // the key leader's value head: whose record holds k^
  const int rmax = (g.r_lo == 0 && g.r_hi == C) ? C : ((g.r_hi + 15) & ~15);
  const KTile kt = sm.k(), qt = sm.q();

  // ------------------------------------------------------------------ 1. rows in
  for (int x = tid; x < rmax * 16; x += THREADS) {
    const int i = x >> 4, c = x & 15;
    const bool old = i < g.r_lo, cur = i >= g.r_lo && i < g.r_hi;
    const int64_t t = cur ? g.token(i) : 0;
    cp16(smem_u32(kt.chunk(i, c)),
         old ? (const void *)KTile{p.st.kb(si, hlead)}.chunk(i, c)
             : (const void *)(p.k + t * p.ks0 + hq * p.ks1 + c * 8),
         old || cur);
  }
  for (int x = tid; x < (g.r_hi - g.r_lo) * 16; x += THREADS) {
    const int i = g.r_lo + (x >> 4), c = x & 15;
    cp16(smem_u32(qt.chunk(i, c)), p.q + g.token(i) * p.qs0 + hq * p.qs1 + c * 8, true);
  }
  cp_commit();
  cp_wait<0>();
  __syncthreads();

  // ------------------------------------------------------------------ 2. k^, q^ (FOLD32), the live rows dealt round robin
  const int nlive = g.r_hi - g.r_lo;
  const int mine = nlive > warp ? ((nlive - warp + 7) >> 3) : 0;
  #pragma unroll
  for (int which = 0; which < 2; ++which) {
    if (mine == 0) break;
    const KTile t = which ? qt : kt;
    float x[C / 8][CE];
    uint2 *e[C / 8];
    #pragma unroll
    for (int rr = 0; rr < C / 8; ++rr) {
      if (rr >= mine) break;
      e[rr] = reinterpret_cast<uint2 *>(t.chunk(g.r_lo + 8 * rr + warp, lane >> 1) + 4 * (lane & 1));
      const uint2 rw = *e[rr];
      x[rr][0] = bf(lo16(rw.x));
      x[rr][1] = bf(hi16(rw.x));
      x[rr][2] = bf(lo16(rw.y));
      x[rr][3] = bf(hi16(rw.y));
    }
    uint2 nrm[C / 8];
    if (mine == C / 8) l2norm_rows<C / 8, false>(x, p.scale, which == 1, nrm, nullptr);
    else l2norm_rows<C / 8, true>(x, p.scale, which == 1, nrm, nullptr, mine);
    #pragma unroll
    for (int rr = 0; rr < C / 8; ++rr) {
      if (rr >= mine) break;
      *e[rr] = nrm[rr];
    }
  }
  fence_async_smem();
  __syncthreads();

  // ------------------------------------------------------------------ 3. publish the tiles, then the products
  char *hd = pr.hdk(slot, p.hk, hq);
  {
    const KTile hk{reinterpret_cast<uint16_t *>(hd + HandK::K)},
        hqt{reinterpret_cast<uint16_t *>(hd + HandK::Q)};
    static_tasks<C * 16>(tid, [&](int x) {       // all sixty-four rows, verbatim, stale rows included
      const int i = x >> 4, c = x & 15;
      *reinterpret_cast<uint4 *>(hk.chunk(i, c)) = *reinterpret_cast<const uint4 *>(kt.chunk(i, c));
      *reinterpret_cast<uint4 *>(hqt.chunk(i, c)) = *reinterpret_cast<const uint4 *>(qt.chunk(i, c));
    });
  }
  float *kkb = reinterpret_cast<float *>(hd + HandK::KK), *kqb = reinterpret_cast<float *>(hd + HandK::KQ);
  // THE TEN-TILE DEAL, exactly the one launch A runs (chunk.cuh, `kk_kq_deal`), so the warp that computes
  // a tile is the warp shape the products were measured on. Warps 6 and 7 own no tile under it, and here
  // they are what ZEROES the six (i-block > j-block) tiles the deal proves are +0 -- so a reader that asks
  // for one gets finite bits rather than whatever the workspace allocation held.
  {
    const int nmt = warp < 4 ? 2 : (warp < 6 ? 1 : 0);
    if (nmt > 0) {
      const int dj0 = warp < 2 ? 48 : (warp == 2 ? 32 : (warp == 3 ? 16 : (warp == 4 ? 0 : 32)));
      const int di0 = (warp == 1 || warp == 5) ? 32 : 0;
      float kk[2][2][4], kq[2][2][4];
      if (nmt == 2) dots128_pair<2>(kk, kq, kt, qt, di0, dj0, lane);
      else dots128_pair<1>(kk, kq, kt, qt, di0, dj0, lane);
      #pragma unroll
      for (int mt_ = 0; mt_ < 2; ++mt_) {
        if (mt_ >= nmt) break;
        kkq_store(kk[mt_], kkb, di0 + 16 * mt_, dj0, lane);
        kkq_store(kq[mt_], kqb, di0 + 16 * mt_, dj0, lane);
      }
    } else {                                     // the six dead tiles, 3 per idle warp
      const int dead[6][2] = {{16, 0}, {32, 0}, {48, 0}, {32, 16}, {48, 16}, {48, 32}};
      #pragma unroll
      for (int t = 0; t < 6; ++t)
        if ((t & 1) == (warp - 6)) {
          const float z[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
          kkq_store(z, kkb, dead[t][0], dead[t][1], lane);
          kkq_store(z, kqb, dead[t][0], dead[t][1], lane);
        }
    }
  }
  (void)gq;
  (void)tq;
}

// ------------------------------------------------------------------ launch A: publish
template <bool PREK, class SM>
__device__ __forceinline__ void publish(const Params &p, const PrepRef &pr, SM &sm, const ChunkGeo &g,
                                        int h, int slot, int so) {
  const int tid = threadIdx.x;
  char *hd = pr.hd(slot, p.hv, h);
  const KTile kt = sm.k(), qt = sm.q(), ut = sm.u(), wt = sm.w();
  const MTile mt = sm.m();
  const KTile hu{reinterpret_cast<uint16_t *>(hd + Hand::U)}, hw{reinterpret_cast<uint16_t *>(hd + Hand::W)};
  const MTile hm{reinterpret_cast<uint16_t *>(hd + Hand::M)};
  // ALL SIXTY-FOUR ROWS, including the ones no chain writes. A tile's unwritten rows are read by chains
  // whose results are discarded (the q^ argument of chunk.cuh) or selected away (u, w), so carrying them
  // verbatim is what makes the two modes bit-identical instead of merely equal where it is checked.
  // With the pre-kernel, `k^` and `q^` are ALREADY published per (chunk, KEY head) and launch B reads them
  // from there, so only `u`, `w`, `M` and the two gate vectors cross here: 40.5 KB rather than 72.5.
  if constexpr (PREK) {
    static_tasks<C * 16>(tid, [&](int x) {
      const int i = x >> 4, c = x & 15;
      *reinterpret_cast<uint4 *>(hu.chunk(i, c)) = *reinterpret_cast<const uint4 *>(ut.chunk(i, c));
      *reinterpret_cast<uint4 *>(hw.chunk(i, c)) = *reinterpret_cast<const uint4 *>(wt.chunk(i, c));
    });
  } else {
    const KTile hk{reinterpret_cast<uint16_t *>(hd + Hand::K)},
        hq{reinterpret_cast<uint16_t *>(hd + Hand::Q)};
    static_tasks<C * 16>(tid, [&](int x) {
      const int i = x >> 4, c = x & 15;
      *reinterpret_cast<uint4 *>(hk.chunk(i, c)) = *reinterpret_cast<const uint4 *>(kt.chunk(i, c));
      *reinterpret_cast<uint4 *>(hq.chunk(i, c)) = *reinterpret_cast<const uint4 *>(qt.chunk(i, c));
      *reinterpret_cast<uint4 *>(hu.chunk(i, c)) = *reinterpret_cast<const uint4 *>(ut.chunk(i, c));
      *reinterpret_cast<uint4 *>(hw.chunk(i, c)) = *reinterpret_cast<const uint4 *>(wt.chunk(i, c));
    });
  }
  static_tasks<C * 8>(tid, [&](int x) {
    const int i = x >> 3, c = x & 7;
    *reinterpret_cast<uint4 *>(hm.chunk(i, c)) = *reinterpret_cast<const uint4 *>(mt.chunk(i, c));
  });
  if (tid < C) {
    reinterpret_cast<float *>(hd + Hand::GAM)[tid] = sm.gt().gam[tid];
    reinterpret_cast<float *>(hd + Hand::LAM)[tid] = sm.gt().lam[tid];
  }
  // THE CARRY IS NOT WRITTEN HERE ANY MORE. `u` and `w` overlay `L`/`X`/`P` in `PrepASmem`, so by this
  // point the X and P bytes the carry needs are gone; `wy_chunk` writes all four carry sections at the
  // point its `keep_open` would have written the record (chunk.cuh, `CarryRef`).
}

template <int PLAN, int FORM>
__global__ __launch_bounds__(THREADS) void gdn_prep_a(Params p, PrepRef pr) {
  constexpr bool DEAL = (PLAN & 1) != 0, WIDTHS = (PLAN & 2) != 0;
  constexpr bool R4 = FORM == 0;      // round 4: the one-launch layout, one CTA per SM, the deep prologue
  constexpr bool PREK = FORM == 2;    // the (chunk, KEY head) pre-kernel carries k^, q^, KK and KQ
  using ASmem = std::conditional_t<R4, Smem, PrepASmem>;
  extern __shared__ __align__(1024) char raw[];
  ASmem &sm = *reinterpret_cast<ASmem *>(raw);
  const int item = blockIdx.x;
  const int slot = item / p.hv, h = item - slot * p.hv;
  int n, ck, cs, len, j0, si, so;
  if constexpr (R4) {                             // round 4: three dependent global loads, in order
    n = pr.tab[slot].n;
    if (n < 0) return;
    ck = pr.tab[slot].k;
    cs = p.cu[n];
    len = p.cu[n + 1] - cs;
    si = p.slot_in_of(n);
    so = p.slot_out_of(n);
    j0 = (si >= 0 ? p.st.pos(si, h)[0] : 0) & (C - 1);
  } else {                                        // TWO INDEPENDENT 16-BYTE LOADS: `ChunkRec` above
    // BOTH HALVES ARE ISSUED BEFORE THE EXIT TEST, and that is the fifth review's five-line item. The
    // previous form read `r0`, tested `n < 0`, and only then issued `r1` -- so what this file called two
    // independent `LDG.E.128` was two DEPENDENT round trips, the second waiting on a branch. The table
    // entry exists for every grid slot (the plan kernel writes all `maxch` of them), so reading the
    // second half unconditionally is safe, and it is ~0.3 us of a ~15 us CTA: ~2% of launch A.
    const int4 r0 = reinterpret_cast<const int4 *>(pr.tab)[2 * slot];
    const int4 r1 = reinterpret_cast<const int4 *>(pr.tab)[2 * slot + 1];
    n = r0.x;
    if (n < 0) return;                            // a grid slot past the pack's own chunks
    ck = r0.y;
    cs = r0.z;
    len = r0.w;
    j0 = r1.x;
    si = r1.y;
    so = r1.z;
  }
  const ChunkGeo g = chunk_geo(n, ck, cs, len, j0);
  char *cyb = pr.carry + (int64_t(n) * p.hv + h) * Carry::BYTES;
  const CarryRef cy{reinterpret_cast<float *>(cyb + Carry::X), reinterpret_cast<float *>(cyb + Carry::P),
                    reinterpret_cast<float *>(cyb + Carry::G), reinterpret_cast<float *>(cyb + Carry::BETA)};
  PreKRef pk{nullptr, nullptr, nullptr};
  if constexpr (PREK) {
    const char *hd = pr.hdk(slot, p.hk, p.key_head(h));
    pk = PreKRef{reinterpret_cast<const uint16_t *>(hd + HandK::K),
                 reinterpret_cast<const float *>(hd + HandK::KK),
                 reinterpret_cast<const float *>(hd + HandK::KQ)};
  }
  if (g.r_lo == 0 && g.r_hi == C)
    wy_chunk<false, true, DEAL, WIDTHS, true, PREK>(p, sm, g, h, si, so, false, cy, pk);
  else
    wy_chunk<false, false, DEAL, WIDTHS, true, PREK>(p, sm, g, h, si, so, false, cy, pk);
  __syncthreads();
  publish<PREK>(p, pr, sm, g, h, slot, so);
}

// ------------------------------------------------------------------ launch B: the recurrence
// LAUNCH B'S SHARED MEMORY IS ITS OWN, AND IT IS DOUBLE-BUFFERED. A CTA here runs no phase 1, so the
// 70 KB the one-launch layout gives the X solve's scratch is free -- and what launch B is waiting on is
// exactly the 72 KB of hand-off it has to pull in per chunk, measured at ~1.4 us of a 4.9 us chunk with
// one buffer. Two buffers put chunk k + 1's `cp.async` in flight while chunk k is evaluated, which is
// the whole reason this layout exists; 210 KB of the 227 KB an SM will give one CTA.
struct PrepSmem {
  static constexpr int HB = 4 * C * D + C * C;      // one buffer: k^, q^, u, w and M, in bf16 elements
  uint16_t hb[2][HB];
  uint16_t sh_[D * D];
  uint16_t dt_[C * D];
  uint16_t dl_[C * D];
  __align__(16) GateTile gt_[2];
  __device__ __forceinline__ KTile k(int b) { return KTile{hb[b]}; }
  __device__ __forceinline__ KTile q(int b) { return KTile{hb[b] + C * D}; }
  __device__ __forceinline__ KTile u(int b) { return KTile{hb[b] + 2 * C * D}; }
  __device__ __forceinline__ KTile w(int b) { return KTile{hb[b] + 3 * C * D}; }
  __device__ __forceinline__ MTile m(int b) { return MTile{hb[b] + 4 * C * D}; }
  __device__ __forceinline__ ShTile sh() { return ShTile{sh_}; }
  __device__ __forceinline__ KTile dt() { return KTile{dt_}; }
  __device__ __forceinline__ KTile dl() { return KTile{dl_}; }
  __device__ __forceinline__ SlabTiles tiles(int b, int wb) {
    return SlabTiles{k(b), q(b), w(b), u(b), m(b), sh(), dt(), dl(), &gt_[b], wb};
  }
};

template <bool PREK, int SPLIT>
__device__ __forceinline__ void load_hand(const Params &p, const PrepRef &pr, PrepSmem &sm,
                                          const ChunkGeo &g, int h, int slot, int si, int b, int wb,
                                          bool load_sh) {
  constexpr int NT = THREADS / SPLIT, NW = 8 / SPLIT;
  const int tid = threadIdx.x;
  const char *hd = pr.hd(slot, p.hv, h);
  const MTile hm{const_cast<uint16_t *>(reinterpret_cast<const uint16_t *>(hd + Hand::M))};
  uint16_t *dst = sm.hb[b];
  const uint16_t *kqs = reinterpret_cast<const uint16_t *>(
      PREK ? pr.hdk(slot, p.hk, p.key_head(h)) + HandK::K : hd + Hand::K);
  static_tasks_n<2 * C * 16, NT>(tid, [&](int x) {    // k^ and q^: 2 tiles x 64 rows x 16 chunks
    const int t = x >> 10, i = (x >> 4) & (C - 1), c = x & 15;
    const KTile s{const_cast<uint16_t *>(kqs + t * C * D)}, d{dst + t * C * D};
    cp16(smem_u32(d.chunk(i, c)), s.chunk(i, c), true);
  });
  const uint16_t *uws = reinterpret_cast<const uint16_t *>(hd + Hand::U);
  static_tasks_n<C * 16, NT>(tid, [&](int x) {        // w: an A operand over d, every sibling loads it all
    const int i = (x >> 4) & (C - 1), c = x & 15;
    const KTile s{const_cast<uint16_t *>(uws + C * D)}, d{dst + 3 * C * D};
    cp16(smem_u32(d.chunk(i, c)), s.chunk(i, c), true);
  });
  static_tasks_n<C * 2 * NW, NT>(tid, [&](int x) {    // u: THIS CTA's value columns only
    const int i = x / (2 * NW), c = 2 * wb + (x % (2 * NW));
    const KTile s{const_cast<uint16_t *>(uws)}, d{dst + 2 * C * D};
    cp16(smem_u32(d.chunk(i, c)), s.chunk(i, c), true);
  });
  const MTile mt{dst + 4 * C * D};
  static_tasks_n<C * 8, NT>(tid, [&](int x) {
    const int i = x >> 3, c = x & 7;
    cp16(smem_u32(mt.chunk(i, c)), hm.chunk(i, c), true);
  });
  if (tid < 32) {                                     // gam and lam, 512 B, as two 16-byte copies a lane
    const int which = tid >> 4, off = (tid & 15) * 16;
    cp16(smem_u32(reinterpret_cast<char *>(which ? sm.gt_[b].lam : sm.gt_[b].gam) + off),
         (which ? hd + Hand::LAM : hd + Hand::GAM) + off, true);
  }
  for (int x = tid; x < g.r_lo * 2 * NW; x += NT) {    // the open chunk's delta rows, chunk 0 only
    const int i = x / (2 * NW), c = 2 * wb + (x % (2 * NW));
    cp16(smem_u32(sm.dt().chunk(i, c)), KTile{p.st.db(si, h)}.chunk(i, c), true);
  }
  if (load_sh) static_tasks_n<16 * NW * 16, NT>(tid, [&](int x) {   // S^: this CTA's value ROWS
    const int v = 16 * wb + (x >> 4), c = x & 15;
    cp16(smem_u32(sm.sh().chunk(v, c)), ShTile{p.st.sh(si, h)}.chunk(v, c), true);
  });
  cp_commit();
}

template <bool PRED, bool PREK, bool FUSE, int SPLIT>
__global__ __launch_bounds__(THREADS / SPLIT) void gdn_prep_b(Params p, PrepRef pr) {
  constexpr int NT = THREADS / SPLIT, NW = 8 / SPLIT;
  extern __shared__ __align__(1024) char raw[];
  PrepSmem &sm = *reinterpret_cast<PrepSmem *>(raw);
  const int tid = threadIdx.x;
  const int sp = blockIdx.x % SPLIT, item = blockIdx.x / SPLIT;
  const int wb = sp * NW;
  const int nn = item / p.hv, h = item - nn * p.hv;
  const int n = p.order ? p.order[nn] : nn;
  const int cs = p.cu[n], len = p.cu[n + 1] - cs;
  const int si = p.slot_in_of(n), so = p.slot_out_of(n);
  // Launch B pays this two-deep chain ONCE PER SEQUENCE, not once per chunk, and it needs the fp32
  // `pos_in` and not only `j0` (it writes `pos_in + len` back), so it reads its own head's copy.
  const int pos_in = si >= 0 ? p.st.pos(si, h)[0] : 0;
  const int j0 = pos_in & (C - 1);
  const int nch = chunks_of(len, j0);
  if (nch == 0) {
    if (so >= 0 && so != si && sp == 0) move_record<NT>(p, h, si, so, pos_in);
    return;
  }
  const int b0 = pr.base[n];
  float ST[8][2][4];
  {
    const ChunkGeo g0 = chunk_geo(n, 0, cs, len, j0);
    load_hand<PREK, SPLIT>(p, pr, sm, g0, h, b0, si, 0, wb,
                           !state_in_regs(g0, si, so, nch > 1) && si >= 0);
  }
  for (int k = 0; k < nch; ++k) {
    const ChunkGeo g = chunk_geo(n, k, cs, len, j0);
    const bool more = k + 1 < nch;
    const bool sh_cached = k == 0 && !state_in_regs(g, si, so, more) && si >= 0;
    if (more) {   // chunk k + 1's hand-off goes in flight now and is waited on one iteration later
      const ChunkGeo gn = chunk_geo(n, k + 1, cs, len, j0);
      load_hand<PREK, SPLIT>(p, pr, sm, gn, h, b0 + k + 1, si, (k + 1) & 1, wb, false);
      cp_wait<1>();
    } else {
      cp_wait<0>();
    }
    __syncthreads();
    const SlabTiles tl = sm.tiles(k & 1, wb);
    if (g.r_lo == 0 && g.r_hi == C)
      slab_step<false, true, false, FUSE>(p, tl, g, h, si, so, more, sh_cached, ST);
    else
      slab_step<false, false, PRED, FUSE>(p, tl, g, h, si, so, more, sh_cached, ST);
    __syncthreads();
  }
  // What `wy_chunk`'s `keep_open` writes in the one-launch kernel: the open k^ rows (still in this CTA's
  // shared memory from the last chunk) and the X, P, G and beta launch A left in the carry. All of it is
  // value-INDEPENDENT, so CTA 0 of the head writes it and the siblings skip it.
  const ChunkGeo gl = chunk_geo(n, nch - 1, cs, len, j0);
  if (so >= 0 && !gl.complete() && sp == 0) {
    const OpenRows open = open_rows(gl, si, so);
    const char *cy = pr.carry + (int64_t(n) * p.hv + h) * Carry::BYTES;
    const KTile kt = sm.k((nch - 1) & 1);
    for (int x = tid; x < (open.jf - open.r_first) * 16; x += NT) {
      const int i = open.r_first + (x >> 4), c = x & 15;
      *reinterpret_cast<uint4 *>(KTile{p.st.kb(so, h)}.chunk(i, c)) =
          *reinterpret_cast<const uint4 *>(kt.chunk(i, c));
      *reinterpret_cast<float4 *>(p.st.xb(so, h) + i * C + c * 4) =
          *reinterpret_cast<const float4 *>(reinterpret_cast<const float *>(cy + Carry::X) + i * C + c * 4);
    }
    if (tid >= open.r_first && tid < open.jf) {
      p.st.gb(so, h)[tid] = reinterpret_cast<const float *>(cy + Carry::G)[tid];
      p.st.bb(so, h)[tid] = reinterpret_cast<const float *>(cy + Carry::BETA)[tid];
    }
    if ((open.jf & ~15) > 0) {
      const int pb0 = open.jf & ~15, r0 = max(open.r_first, pb0);
      for (int x = tid; x < (open.jf - r0) * 16; x += NT) {
        const int i = r0 + (x >> 4), c = x & 15;
        *reinterpret_cast<float4 *>(p.st.pb(so, h) + (i - pb0) * C + c * 4) =
            p_row_chunk(reinterpret_cast<const float *>(cy + Carry::P) + (i - pb0) * C + c * 4, c, pb0);
      }
    }
  }
  if (so >= 0 && sp == 0 && tid < NPOS) p.st.pos(so, h)[tid] = pos_in + len;
}

// The dynamic shared-memory opt-in is per (kernel, device); see csrc/core.cu on why the bitmask lives
// inside the function template rather than beside it.
template <class FN>
void raise_smem(FN *fn, std::atomic<uint64_t> &raised, size_t bytes) {
  int device = 0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  const uint64_t bit = uint64_t(1) << (device & 63);
  if (!(raised.load(std::memory_order_acquire) & bit)) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void *>(fn),
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, int(bytes)));
    raised.fetch_or(bit, std::memory_order_release);
  }
}

// TWO CTAs PER SM NEEDS THE CARVEOUT ASKED FOR, NOT ONLY THE FOOTPRINT. The driver sizes the unified
// L1 / shared split from the per-block request, and 110.25 KB lands on a configuration that fits exactly
// one CTA; `cudaSharedmemCarveoutMaxShared` gives the SM its full 228 KB and 2 x 110.25 fits. Both
// attributes are per (kernel, device), so they ride the same bitmask as the opt-in.
template <int PLAN, int FORM = 1>
void launch_a(const Params &p, const PrepRef &pr, int items, cudaStream_t stream) {
  static std::atomic<uint64_t> raised{0};
  auto *fn = gdn_prep_a<PLAN, FORM>;
  constexpr size_t SM_BYTES = FORM == 0 ? sizeof(Smem) : sizeof(PrepASmem);
  int device = 0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  const uint64_t bit = uint64_t(1) << (device & 63);
  if (!(raised.load(std::memory_order_acquire) & bit)) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void *>(fn),
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, int(SM_BYTES)));
    C10_CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void *>(fn),
                                        cudaFuncAttributePreferredSharedMemoryCarveout,
                                        cudaSharedmemCarveoutMaxShared));
    raised.fetch_or(bit, std::memory_order_release);
  }
  fn<<<items, THREADS, SM_BYTES, stream>>>(p, pr);
  C10_CUDA_CHECK(cudaGetLastError());
}

template <bool PRED, bool PREK, bool FUSE, int SPLIT>
void launch_b(const Params &p, const PrepRef &pr, int items, cudaStream_t stream) {
  static std::atomic<uint64_t> raised{0};
  auto *fn = gdn_prep_b<PRED, PREK, FUSE, SPLIT>;
  raise_smem(fn, raised, sizeof(PrepSmem));
  fn<<<items * SPLIT, THREADS / SPLIT, sizeof(PrepSmem), stream>>>(p, pr);
  C10_CUDA_CHECK(cudaGetLastError());
}

void launch_k(const Params &p, const PrepRef &pr, int items, cudaStream_t stream) {
  if (items <= 0) return;
  static std::atomic<uint64_t> raised{0};
  auto *fn = gdn_prep_k;
  raise_smem(fn, raised, sizeof(PreKSmem));
  fn<<<items, THREADS, sizeof(PreKSmem), stream>>>(p, pr);
  C10_CUDA_CHECK(cudaGetLastError());
}

}  // namespace

int64_t prep_hand_bytes() { return Hand::BYTES; }
int64_t prep_hand_short_bytes() { return Hand::SHORT; }
int64_t prep_handk_bytes() { return HandK::BYTES; }
int64_t prep_b_smem_bytes() { return int64_t(sizeof(PrepSmem)); }
int64_t prep_carry_bytes() { return Carry::BYTES; }
int64_t prep_a_smem_bytes() { return int64_t(sizeof(PrepASmem)); }
int64_t prep_table_ints() { return int64_t(sizeof(ChunkRec) / sizeof(int)); }

// What the hardware says about the occupancy this pass went after, rather than what the arithmetic says:
// `cudaOccupancyMaxActiveBlocksPerMultiprocessor` on the shipped launch-A instantiation, after the opt-in
// and the carveout. `tests/test_contract_cpu.py` has no GPU; the battery asserts it is 2.
int64_t prep_a_blocks_per_sm() {
  auto *fn = gdn_prep_a<3, 1>;
  static std::atomic<uint64_t> raised{0};
  int device = 0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  const uint64_t bit = uint64_t(1) << (device & 63);
  if (!(raised.load(std::memory_order_acquire) & bit)) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void *>(fn),
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        int(sizeof(PrepASmem))));
    C10_CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void *>(fn),
                                        cudaFuncAttributePreferredSharedMemoryCarveout,
                                        cudaSharedmemCarveoutMaxShared));
    raised.fetch_or(bit, std::memory_order_release);
  }
  int blocks = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, reinterpret_cast<const void *>(fn),
                                                               THREADS, sizeof(PrepASmem)));
  return blocks;
}

// Three launches on one stream: the chunk table, phase 1 over every (chunk, head), then the recurrence
// over every (sequence, head). `plan` is the phase-1 execution plan launch A runs (bits 0 and 1, exactly
// as in the one-launch kernel); the prepared mode itself is bit 3 and `ops.py` has stripped it.
void prep_launch(const Params &p, char *hand, char *handk, char *carry, int *table, int *base, int maxch,
                 int64_t hstride, int plan, cudaStream_t stream) {
  PrepRef pr{hand, handk, carry, reinterpret_cast<const ChunkRec *>(table), base, maxch, hstride};
  if (p.nseq <= 0 || pr.maxch <= 0) return;
  // The plan kernel's shared memory is `2 nseq + 1` ints with no opt-in, so it cannot be launched past
  // ~6,100 sequences. `ops._use_prepared` bounds `3 nseq hv <= 2 SMs`, i.e. nseq <= 98, so this is only
  // reachable through a FORCED `plan = 11`; the fourth review named the silent launch failure and this
  // is the check rather than the silence.
  TORCH_CHECK(size_t(2 * p.nseq + 1) * sizeof(int) <= 48u * 1024u,
              "gdn_cpr_sm100: the prepared mode's plan kernel cannot hold ", p.nseq,
              " sequences in 48 KB of shared memory; the selection rule never asks for this many");
  prep_plan_kernel<<<1, THREADS, size_t(2 * p.nseq + 1) * sizeof(int), stream>>>(
      p, const_cast<ChunkRec *>(pr.tab), const_cast<int *>(pr.base), pr.maxch);
  C10_CUDA_CHECK(cudaGetLastError());
  // WHICH FORM OF LAUNCH A. The SHIPPED form is the two-CTA-per-SM layout with no pre-kernel. Bit 4 forces
  // ROUND 4's form (the one-launch shared-memory layout at one CTA per SM and the three-deep prologue) and
  // bit 5 the (chunk, KEY head) PRE-KERNEL, which is built and gated and not shipped because it measured a
  // net loss (ops.py::_use_prekernel). Whether the pre-kernel runs is read off `pr.handk`, which `ops.py`
  // allocates only for that plan, so the workspace shape and the form cannot disagree.
  const int items_a = pr.maxch * p.hv;
  const bool r4 = (plan & 16) != 0;
  const bool prek = pr.handk != nullptr && !r4;
  if (prek) launch_k(p, pr, pr.maxch * p.hk, stream);
  if (r4) {
    if ((plan & 3) == 0) launch_a<0, 0>(p, pr, items_a, stream);
    else if ((plan & 3) == 1) launch_a<1, 0>(p, pr, items_a, stream);
    else launch_a<3, 0>(p, pr, items_a, stream);
  } else if (prek) {
    launch_a<3, 2>(p, pr, items_a, stream);
  } else {
    if ((plan & 3) == 0) launch_a<0, 1>(p, pr, items_a, stream);
    else if ((plan & 3) == 1) launch_a<1, 1>(p, pr, items_a, stream);
    else launch_a<3, 1>(p, pr, items_a, stream);
  }
  // Bit 6 switches the FUSED delta / seed chain of launch B (slab.cuh, `FUSE`); the shipped value is
  // whichever the measurement wants and the other stays in the build as its before column.
  // Round 4's launch B was unfused, so bit 4 implies it: the before column must be the whole of round 4.
  const bool fuse = (plan & 64) == 0 && !r4;
  // BITS 7-8: launch B's value-column split across CTAs. Code 0 is unsplit (and every non-shipped form of
  // launch A keeps it, so a before column is the whole of its pass), 1 is two-way and 2 is four-way.
  // `ops.py` resolves the deployment rule into the code, so the kernel never reads an SM count.
  const int sc = (plan >> 7) & 3;
  const int items_b = p.nseq * p.hv;
  if (prek) {
    if (fuse) launch_b<false, true, true, 1>(p, pr, items_b, stream);
    else launch_b<false, true, false, 1>(p, pr, items_b, stream);
  } else if (!fuse) {
    launch_b<false, false, false, 1>(p, pr, items_b, stream);
  } else if (sc == 1) {
    launch_b<false, false, true, 2>(p, pr, items_b, stream);
  } else if (sc == 2) {
    launch_b<false, false, true, 4>(p, pr, items_b, stream);
  } else {
    launch_b<false, false, true, 1>(p, pr, items_b, stream);
  }
}

}  // namespace gdn_cpr_sm100
