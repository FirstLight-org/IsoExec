// SPDX-License-Identifier: Apache-2.0
// The fully fused prefill: one CTA walks a (sequence, value head)'s chunks in order and does everything
// of a chunk on chip -- the pass-1 work (wy_chunk.cuh), then the state step and the outputs. Only inputs,
// final outputs, requested trainer saves and engine records cross global memory. The host chooses an
// execution mode from measured shape crossovers; the two-pass prefill (wy.cu, sweep.cu) retains chunk
// parallelism for packs with few long sequences. Bit for bit the same function:
// every element is the same recipe.cuh chain as in the two passes, on the same tiles --
//   pass-1 tiles      wy_chunk.cuh, shared with wy.cu;
//   state step        the sweep.cu chains, with 64 value rows per warpgroup: D^T = W S^T, the deltas,
//                     S^T += k^T d~^T;
//   outputs           transposed: o^T[j][v] = CHAIN64_i(M[j][i], d[v][i]; CHAIN128(q^_j, S^_v) (*) e(G_j)),
//                     A = q^ / M, B = S^ / the deltas (tests/test_mma_facts_gpu.py: a swapped-operand
//                     WGMMA is the transpose of the recipe's chain, seeded or not).
//
// CTA: 256 threads, one per SM. Each warpgroup owns 64 value rows and keeps its fp32 boundary
// state in registers throughout the sequence. The two warpgroups perform one m64n64 state/output step
// each after the 8-warp pass-1 work. No intermediate state or operand tiles leave the chip. The grid has
// one CTA per (sequence, head), ordered longest first; CUDA schedules new items as CTAs finish.
//
// The prepared execution mode uses the same state/output step after a separate parallel WY pass.
// Two shared-memory operand buffers stream that pass's K/Q/W/U/M/gate tiles with bulk copies while
// the current chunk computes. The state and deltas still stay on chip: it has no sweep/output
// publication workspace. PREPARED selects only these loads and the key-side record writes; the
// state/output arithmetic below is shared by both modes.
#include <c10/cuda/CUDAException.h>
#include <type_traits>

#include "launch.h"
#include "wy_chunk.cuh"

namespace gdn_cpr {
namespace {

constexpr int FVR = 64;          // one value-row slab per warpgroup
constexpr int FNA = FVR / 2;     // accumulator registers of an m64 x 64
constexpr int SLABS = D / FVR;   // 2: one per warpgroup

struct Slab {   // one warpgroup's state-step tiles (every member a multiple of 1024 bytes)
  uint16_t shat[FVR * D];   // Wide<64>: S^ [v][d] (then the outputs o^T [j][v], staged)
  uint16_t dt[FVR * C];     // Tile<64>: deltas [v][i]
  uint16_t dtl[FVR * C];    // Tile<64>: d~ [v][i]
};

struct FusedSmem {
  uint16_t k_[C * D];
  uint16_t q_[C * D];
  uint16_t m_[C * C];
  union {
    struct {
      uint16_t v[C * D];
      uint16_t xu[C * C];
      uint16_t xw[C * C];
    } a;
    Slab slab[2];
  } phase;
  union {
    struct {
      float L[C * LS];
      float X[C * C];
      float P[16 * C];
    } a;
    struct {
      uint16_t w[C * D];
      uint16_t u[D * C];
    } b;
  } r;
  GateTile gt_;
  float lg_[C];
  __device__ uint16_t *k() { return k_; }
  __device__ uint16_t *q() { return q_; }
  __device__ uint16_t *m() { return m_; }
  __device__ uint16_t *xu() { return phase.a.xu; }
  __device__ uint16_t *v() { return phase.a.v; }
  __device__ uint16_t *xw() { return phase.a.xw; }
  __device__ float *L() { return r.a.L; }
  __device__ float *X() { return r.a.X; }
  __device__ float *P() { return r.a.P; }
  __device__ uint16_t *w() { return r.b.w; }
  __device__ uint16_t *u() { return r.b.u; }
  __device__ GateTile &gt() { return gt_; }
  __device__ float *lg() { return lg_; }
  __device__ Slab &slab(int wg) { return phase.slab[wg]; }
};

// The prepared mode double-buffers the completed WY tiles while each warpgroup
// keeps its boundary state in registers and evaluates the same slab_step as fused.
struct PreparedTiles {
  uint16_t k[C * D], q[C * D], w[C * D], u[D * C], m[C * C];
  GateTile gt;
};
struct PreparedSmem {
  PreparedTiles prep[2];
  Slab slabs[2];
  int parity;
  uint64_t full[2];
  __device__ uint16_t *k() { return prep[parity].k; }
  __device__ uint16_t *q() { return prep[parity].q; }
  __device__ uint16_t *m() { return prep[parity].m; }
  __device__ uint16_t *w() { return prep[parity].w; }
  __device__ uint16_t *u() { return prep[parity].u; }
  __device__ GateTile &gt() { return prep[parity].gt; }
  __device__ Slab &slab(int wg) { return slabs[wg]; }
};
__device__ void load_prepared(const PrefillParams &p, PreparedTiles &a, int s, int h, uint64_t *bar) {
  if (threadIdx.x != 0) return;
  const int hq = p.key_head(h);
  mbar_expect(bar, 4 * TILE_BYTES + MT_BYTES + GT_BYTES);
  bulk_load(a.k, p.k_tile(s, hq), TILE_BYTES, bar);
  bulk_load(a.q, p.q_tile(s, hq), TILE_BYTES, bar);
  bulk_load(a.w, p.w_tile(s, h), TILE_BYTES, bar);
  bulk_load(a.u, p.u_tile(s, h), TILE_BYTES, bar);
  bulk_load(a.m, p.m_tile(s, h), MT_BYTES, bar);
  bulk_load(&a.gt, p.gt_tile(s, h), GT_BYTES, bar);
}

__device__ __forceinline__ void wg_sync(int wg) { asm volatile("bar.sync %0, 128;" ::"r"(1 + wg) : "memory"); }

// The state step and outputs of value rows [64 vs, 64 vs + 64) of chunk g, by one warpgroup. `more`: a
// later chunk of the sequence follows (ST remains in registers between chunks);
// else, with an output slot, the boundary state and the open-chunk value rows go to the record.
template <bool SAVE, class Sm>
__device__ void slab_step(const PrefillParams &p, Sm &sm, Slab &sl, const ChunkGeo &g, int h, int si, int so,
                          int vs, bool more, float (&ST)[2][FNA]) {
  const int wtid = threadIdx.x & 127, wg = threadIdx.x >> 7, warp = wtid >> 5, lane = wtid & 31;
  const int gq = lane >> 2, tq = lane & 3, v0 = FVR * vs;
  const Wide<64> wt{sm.w()}, kt{sm.k()}, qt{sm.q()};
  const Tile<64> ut{sm.u() + v0 * C}, mt{sm.m()};
  const Wide<FVR> shat{sl.shat};
  const Tile<64> dt{sl.dt}, dtl{sl.dtl};
  const GateTile &gt = sm.gt();
  // The fp32 boundary state stays in registers while this CTA walks its chunks.
  if (g.k == 0) {
    #pragma unroll
    for (int db = 0; db < 2; ++db)
      #pragma unroll
      for (int e = 0; e < FNA; ++e) {
        const int d = 64 * db + 16 * warp + gq + 8 * ((e >> 1) & 1), v = v0 + 8 * (e >> 2) + 2 * tq + (e & 1);
        ST[db][e] = si >= 0 ? s_load(p.st, si, h, s_index(v, d)) : 0.f;
      }
  }

  // 2. S^ = rn(S_c) -> [v][d]
  #pragma unroll
  for (int db = 0; db < 2; ++db) store_transposed(shat.half(db), 0, ST[db], warp, lane);
  fence_async_smem();
  wg_sync(wg);

  // 3. D^T[i][v] = CHAIN128_d(W[i][d], S^[v][d])
  float acc[FNA];
  const bool zero_start = g.k == 0 && si < 0;
  if (zero_start) {
    #pragma unroll
    for (int e = 0; e < FNA; ++e) acc[e] = 0.f;
  } else {
    wg_fence();
    #pragma unroll
    for (int s = 0; s < 8; ++s)
      wgmma_ss<FVR, 0>(acc, desc_kmajor(wt.half(s >> 2).base, 16 * (s & 3)), desc_kmajor(shat.half(s >> 2).base, 16 * (s & 3)),
                       s > 0);
    wg_commit();
  }
  uint16_t ub[FNA];   // this thread's u entries (i = 16 warp + gq (+8), v = 8 nt + 2 tq (+1))
  #pragma unroll
  for (int e = 0; e < FNA; ++e) ub[e] = ut.at(8 * (e >> 2) + 2 * tq + (e & 1), 16 * warp + gq + 8 * ((e >> 1) & 1));
  const float lam0 = gt.lam[16 * warp + gq], lam1 = gt.lam[16 * warp + gq + 8];
  wg_wait<0>();
  wg_pin(acc);

  // 4. deltas -> dt (the outputs' B, the record's open rows) ; d~ -> dtl (the boundary chain's B)
  uint16_t dbits[FNA], tbits[FNA];
  #pragma unroll
  for (int e = 0; e < FNA; ++e) dbits[e] = delta_entry(ub[e], acc[e]);
  #pragma unroll
  for (int hi = 0; hi < 2; ++hi) {   // the input slot's deltas before r_lo, zeros past r_hi (edge chunks only)
    const int i = 16 * warp + gq + 8 * hi;
    if (i < g.r_lo || i >= g.r_hi)
      #pragma unroll
      for (int e = 2 * hi; e < FNA; e += (e & 1) ? 3 : 1)
        dbits[e] = i < g.r_lo ? p.st.db_tile(si, h).at(v0 + 8 * (e >> 2) + 2 * tq + (e & 1), i) : uint16_t(0);
  }
  #pragma unroll
  for (int e = 0; e < FNA; ++e) tbits[e] = dtilde_entry(dbits[e], (e >> 1) & 1 ? lam1 : lam0);
  store_transposed_bits(dt, 0, dbits, warp, lane);
  store_transposed_bits(dtl, 0, tbits, warp, lane);
  fence_async_smem();
  wg_sync(wg);

  // 5. S_{c+1}^T = CHAIN64_i(k^[i][d], d~[v][i]; S_c^T (*) e(G_63)), when anything reads it
  if (g.complete() && (more || so >= 0)) {
    const float gam_last = gt.gam[C - 1];
    #pragma unroll
    for (int db = 0; db < 2; ++db)
      #pragma unroll
      for (int e = 0; e < FNA; ++e) ST[db][e] = state_seed(ST[db][e], gam_last);
    wg_pin(ST[0]);
    wg_pin(ST[1]);
    wg_fence();
    #pragma unroll
    for (int s = 0; s < 4; ++s)
      #pragma unroll
      for (int db = 0; db < 2; ++db)
        wgmma_ss<FVR, 1>(ST[db], desc_mnmajor_a(kt.half(db).base, 16 * s), desc_kmajor(dtl.base, 16 * s), true);
    wg_commit();
    wg_wait<0>();
    wg_pin(ST[0]);
    wg_pin(ST[1]);
  }

  // 6. the state onward: to the next chunk, or (last chunk) to the output slot
  if (!more && so >= 0) {
    #pragma unroll
    for (int db = 0; db < 2; ++db)
      #pragma unroll
      for (int e = 0; e < FNA; ++e) {
        const int d = 64 * db + 16 * warp + gq + 8 * ((e >> 1) & 1), v = v0 + 8 * (e >> 2) + 2 * tq + (e & 1);
        s_store(p.st, so, h, s_index(v, d), ST[db][e]);
      }
    // the open chunk's value-major rows [r_first, jf) of these value rows, whole 16-byte chunks (rows past
    // jf are free, rows before r_first equal the slot's own)
    const OpenRows open = open_rows(g, si, so);
    if (open.jf > open.r_first) {
      const int c0 = open.r_first >> 3, nc = ((open.jf + 7) >> 3) - c0;
      for (int x = wtid; x < FVR * nc; x += 128) {
        const int v = x % FVR, c = c0 + x / FVR;
        uint16_t vv[8];
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int i = 8 * c + e;
          vv[e] = i < open.r_new ? p.st.vb_tile(si, h).at(v0 + v, i)
                : i < open.jf    ? p.v[g.token(i) * p.vs0 + h * p.vs1 + v0 + v]
                                 : uint16_t(0);
        }
        *reinterpret_cast<uint4 *>(p.st.vb_tile(so, h).chunk(v0 + v, c)) =
            make_uint4(pack(vv[0], vv[1]), pack(vv[2], vv[3]), pack(vv[4], vv[5]), pack(vv[6], vv[7]));
        *reinterpret_cast<uint4 *>(p.st.db_tile(so, h).chunk(v0 + v, c)) = *reinterpret_cast<const uint4 *>(dt.chunk(v, c));
      }
    }
  }

  // 7. outputs, transposed: o^T[j][v] = CHAIN64_i(M[j][i], d[v][i]; CHAIN128_d(q^[j][d], S^[v][d]) (*) e(G_j))
  float o[FNA];   // (j = 16 warp + gq (+8), v = 8 nt + 2 tq (+1))
  if (zero_start) {
    #pragma unroll
    for (int e = 0; e < FNA; ++e) o[e] = 0.f;
  } else {
    wg_fence();
    #pragma unroll
    for (int s = 0; s < 8; ++s)
      wgmma_ss<FVR, 0>(o, desc_kmajor(qt.half(s >> 2).base, 16 * (s & 3)), desc_kmajor(shat.half(s >> 2).base, 16 * (s & 3)),
                       s > 0);
    wg_commit();
    wg_wait<0>();
    wg_pin(o);
    #pragma unroll
    for (int e = 0; e < FNA; ++e) o[e] = out_seed(o[e], gt.gam[16 * warp + gq + 8 * ((e >> 1) & 1)]);
  }
  wg_pin(o);
  wg_fence();
  #pragma unroll
  for (int s = 0; s < 4; ++s) wgmma_ss<FVR, 0>(o, desc_kmajor(mt.base, 16 * s), desc_kmajor(dt.base, 16 * s), true);
  wg_commit();
  if (SAVE && p.sv.h) {   // trainer mode: S^ rows (FLA's h, value-major) and the deltas (FLA's v_new)
    uint16_t *hd = p.sv.h + ((int64_t(p.chunk0[g.n] + g.k) * p.hv + h) * D + v0) * D;
    for (int x = wtid; x < FVR * 16; x += 128)
      *reinterpret_cast<uint4 *>(hd + (x >> 4) * D + (x & 15) * 8) = *reinterpret_cast<const uint4 *>(shat.chunk(x >> 4, x & 15));
    for (int x = wtid; x < C * (FVR / 8); x += 128) {
      const int i = x / (FVR / 8), vg = x % (FVR / 8);
      if (i < g.r_lo || i >= g.r_hi) continue;
      uint32_t w4[4];
      #pragma unroll
      for (int e = 0; e < 4; ++e) w4[e] = pack(dt.at(8 * vg + 2 * e, i), dt.at(8 * vg + 2 * e + 1, i));
      *reinterpret_cast<uint4 *>(p.sv.v_new + (g.token(i) * p.hv + h) * D + v0 + 8 * vg) = make_uint4(w4[0], w4[1], w4[2], w4[3]);
    }
  }
  wg_wait<0>();
  wg_pin(o);
  wg_sync(wg);   // every warp's chains and saves have read S^: its tile takes the outputs
  uint32_t *ob = reinterpret_cast<uint32_t *>(sl.shat);   // [64 j][64 v] bf16, row-major
  #pragma unroll
  for (int e = 0; e < FNA; e += 2) {
    const int j = 16 * warp + gq + 8 * ((e >> 1) & 1), v = 8 * (e >> 2) + 2 * tq;
    ob[(j * FVR + v) >> 1] = pack(rn(o[e]), rn(o[e + 1]));
  }
  wg_sync(wg);
  for (int x = wtid; x < (g.r_hi - g.r_lo) * (FVR / 8); x += 128) {
    const int j = g.r_lo + x / (FVR / 8), c = x % (FVR / 8);
    *reinterpret_cast<uint4 *>(p.out + g.token(j) * p.os0 + h * p.os1 + v0 + 8 * c) =
        *reinterpret_cast<const uint4 *>(sl.shat + j * FVR + 8 * c);
  }
  wg_sync(wg);   // the staged outputs are out: the next chunk may reuse these tiles
}

// A sequence without tokens whose slot moves: the record, copied.
__device__ void move_record(const PrefillParams &p, int n, int h, int si, int so) {
  const int tid = threadIdx.x, jf = p.pos_in(n) & (C - 1);
  s_move_rows(p.st, si, so, h, 0, D, tid, WY_THREADS);
  if (si >= 0) {
    for (int x = tid; x < jf * 16; x += WY_THREADS) {
      const int i = x >> 4, c = x & 15;
      *reinterpret_cast<uint4 *>(p.st.kb_tile(so, h).chunk(i, c)) = *reinterpret_cast<const uint4 *>(p.st.kb_tile(si, h).chunk(i, c));
      *reinterpret_cast<float4 *>(p.st.xb(so, h) + i * C + c * 4) = *reinterpret_cast<const float4 *>(p.st.xb(si, h) + i * C + c * 4);
      if (i < 16) *reinterpret_cast<float4 *>(p.st.pb(so, h) + i * C + c * 4) = *reinterpret_cast<const float4 *>(p.st.pb(si, h) + i * C + c * 4);
    }
    for (int x = tid; x < D * 8; x += WY_THREADS) {
      const int v = x >> 3, c = x & 7;
      *reinterpret_cast<uint4 *>(p.st.vb_tile(so, h).chunk(v, c)) = *reinterpret_cast<const uint4 *>(p.st.vb_tile(si, h).chunk(v, c));
      *reinterpret_cast<uint4 *>(p.st.db_tile(so, h).chunk(v, c)) = *reinterpret_cast<const uint4 *>(p.st.db_tile(si, h).chunk(v, c));
    }
    if (tid < jf) {
      p.st.gb(so, h)[tid] = p.st.gb(si, h)[tid];
      p.st.bb(so, h)[tid] = p.st.bb(si, h)[tid];
    }
  }
  if (tid < NPOS) p.st.pos(so, h)[tid] = p.pos_in(n);
}

// The key-side record rows from parallel WY. slab_step writes the value-side rows.
__device__ void write_prepared_open(const PrefillParams &p, PreparedSmem &sm, const ChunkGeo &g,
                                    int h, int si, int so) {
  if (so < 0 || !g.last() || g.complete()) return;
  const int tid = threadIdx.x, n = g.n;
  const OpenRows open = open_rows(g, si, so);
  const Wide<64> kt{sm.k()};
  const int pb0 = open.jf & ~15;
  for (int x = tid; x < (open.jf - open.r_first) * 16; x += WY_THREADS) {
    const int i = open.r_first + (x >> 4), c = x & 15;
    *reinterpret_cast<uint4 *>(p.st.kb_tile(so, h).chunk(i, c)) = *reinterpret_cast<const uint4 *>(kt.chunk(i, c));
    *reinterpret_cast<float4 *>(p.st.xb(so, h) + i * C + c * 4) =
        *reinterpret_cast<const float4 *>(p.fx_tile(n, h) + i * C + c * 4);
    if (i >= pb0 && pb0 > 0)
      *reinterpret_cast<float4 *>(p.st.pb(so, h) + (i - pb0) * C + c * 4) =
          *reinterpret_cast<const float4 *>(p.fp_tile(n, h) + (i - pb0) * C + c * 4);
  }
  if (tid >= open.r_first && tid < open.jf) {
    p.st.gb(so, h)[tid] = sm.gt().G[tid];
    p.st.bb(so, h)[tid] = sm.gt().beta[tid];
  }
}

// FULL is a stateless pack whose chunk bound equals T / C, with T divisible by C.
// The preceding plan verifies the actual device lengths fit that bound. Since each
// ceil(len / C) >= len / C, equality forces every nonempty sequence to have whole
// chunks; empty sequences return below. An underestimated bound traps in the plan,
// an overestimate disables FULL, and every stateful call uses the general kernel.
// Separate instantiations remove record/tail branches without duplicating a running
// kernel's instruction body.
template <bool SAVE, bool PREPARED, bool FULL = false>
__global__ __launch_bounds__(WY_THREADS, 1) void fused_kernel(const __grid_constant__ PrefillParams p) {
  extern __shared__ __align__(1024) char smem_raw[];
  using Sm = std::conditional_t<PREPARED, PreparedSmem, FusedSmem>;
  Sm &sm = *reinterpret_cast<Sm *>(smem_raw);
  const int tid = threadIdx.x, wg = tid >> 7;
  float state[SLABS / 2][2][FNA];
  {
    const int it = blockIdx.x;
    const int n = p.order[it / p.hv], h = it % p.hv;
    const int cs = p.cu[n], len = p.cu[n + 1] - cs;
    const int si = FULL ? -1 : p.slot_in_of(n), so = FULL ? -1 : p.slot_out_of(n);
    if (len == 0) {
      if (so != si && so >= 0) move_record(p, n, h, si, so);
      return;
    }
    const int j0 = FULL ? 0 : p.pos_in(n) & (C - 1), nch = chunks_of(len, j0);
    if constexpr (PREPARED) {
      if (tid == 0) {
        mbar_init(&sm.full[0], 1);
        mbar_init(&sm.full[1], 1);
        fence_async_smem();
      }
      __syncthreads();
      load_prepared(p, sm.prep[0], p.chunk0[n], h, &sm.full[0]);
    }
    for (int k = 0; k < nch; ++k) {
      ChunkGeo g = chunk_geo(n, k, cs, len, j0);
      if constexpr (FULL) { g.r_lo = 0; g.r_hi = C; }
      if constexpr (PREPARED) {
        mbar_wait(&sm.full[k & 1], (k >> 1) & 1);
        if (tid == 0) sm.parity = k & 1;
        fence_async_smem();
        __syncthreads();
        // The previous iteration's final CTA barrier released the other buffer.
        if (k + 1 < nch)
          load_prepared(p, sm.prep[(k + 1) & 1], p.chunk0[n] + k + 1, h, &sm.full[(k + 1) & 1]);
        write_prepared_open(p, sm, g, h, si, so);
      } else {
        wy_chunk<SAVE, true>(p, sm, g, p.chunk0[n] + k, h, si);
        __syncthreads();   // v / xu / xw are free for the slabs
      }
      #pragma unroll
      for (int x = 0; x < SLABS / 2; ++x) slab_step<SAVE>(p, sm, sm.slab(wg), g, h, si, so, (SLABS / 2) * wg + x, k + 1 < nch, state[x]);
      __syncthreads();   // the chunk's tiles are no longer read
    }
    if (so >= 0 && tid < NPOS) p.st.pos(so, h)[tid] = p.pos_in(n) + len;
  }
}

}  // namespace

void launch_fused(const PrefillParams &p, cudaStream_t stream) {
  const int bytes = int(sizeof(FusedSmem));
  const int64_t ctas = int64_t(p.nseq) * p.hv;
  if (ctas <= 0) return;
  if (p.full_chunks) {
    if (p.sv.kn) {
      C10_CUDA_CHECK((allow_smem<fused_kernel<true, false, true>>(bytes)));
      fused_kernel<true, false, true><<<unsigned(ctas), WY_THREADS, bytes, stream>>>(p);
    } else {
      C10_CUDA_CHECK((allow_smem<fused_kernel<false, false, true>>(bytes)));
      fused_kernel<false, false, true><<<unsigned(ctas), WY_THREADS, bytes, stream>>>(p);
    }
  } else {
    if (p.sv.kn) {
      C10_CUDA_CHECK((allow_smem<fused_kernel<true, false, false>>(bytes)));
      fused_kernel<true, false, false><<<unsigned(ctas), WY_THREADS, bytes, stream>>>(p);
    } else {
      C10_CUDA_CHECK((allow_smem<fused_kernel<false, false, false>>(bytes)));
      fused_kernel<false, false, false><<<unsigned(ctas), WY_THREADS, bytes, stream>>>(p);
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Parallel WY has completed on this stream; this kernel consumes its prepared tiles.
void launch_prepared(const PrefillParams &p, cudaStream_t stream) {
  const int bytes = int(sizeof(PreparedSmem));
  const int64_t ctas = int64_t(p.nseq) * p.hv;
  if (ctas <= 0) return;
  if (p.full_chunks) {
    if (p.sv.kn) {
      C10_CUDA_CHECK((allow_smem<fused_kernel<true, true, true>>(bytes)));
      fused_kernel<true, true, true><<<unsigned(ctas), WY_THREADS, bytes, stream>>>(p);
    } else {
      C10_CUDA_CHECK((allow_smem<fused_kernel<false, true, true>>(bytes)));
      fused_kernel<false, true, true><<<unsigned(ctas), WY_THREADS, bytes, stream>>>(p);
    }
  } else {
    if (p.sv.kn) {
      C10_CUDA_CHECK((allow_smem<fused_kernel<true, true, false>>(bytes)));
      fused_kernel<true, true, false><<<unsigned(ctas), WY_THREADS, bytes, stream>>>(p);
    } else {
      C10_CUDA_CHECK((allow_smem<fused_kernel<false, true, false>>(bytes)));
      fused_kernel<false, true, false><<<unsigned(ctas), WY_THREADS, bytes, stream>>>(p);
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace gdn_cpr
