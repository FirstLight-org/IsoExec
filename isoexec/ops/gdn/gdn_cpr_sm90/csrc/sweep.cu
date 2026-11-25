// SPDX-License-Identifier: Apache-2.0
// Prefill pass 2. Two ordered launches, each with one kind of work:
//
// Sweep roles, per (sequence or sequence group, value head, 16 or 32 value rows): walk the chunks carrying the
// fp32 boundary state in registers, and for every chunk evaluate
//   d_i = rn(bf(u_i) (-) CHAIN128(S^, w_i; +0))                          (deltas)
//   S_{c+1} = CHAIN64(d~, k^; S_c (*) e(G_63))                           (boundary state, complete chunks)
// with S^ = rn(S_c), handing S^ and the deltas of every chunk to the workspace; then write the boundary
// state to the output slot. This is the only sequential work of a prefill, so nothing else rides on it.
// A pack of many short sequences groups several into one role, one after another (sweep_cta).
//
// Output units, per (chunk, value head, 64 value rows), all in parallel once the sweeps have passed
// the chunk:
//   o_j = rn(CHAIN64(d_i, M_ji; CHAIN128(S^, q^_j; +0) (*) e(G_j)))      (outputs)
// and, for a sequence's last chunk, the rest of the output slot: open-chunk rows and position.
//
// Each sweep CTA owns a role. The output grid launches on the same stream after all sweeps finish.
// Neither grid waits for another CTA, and shared storage never changes between sweep and output roles.
//
// The sweep evaluates the transposed products, so that the value rows are the WGMMA N dimension and a
// chain step costs an m64n32 instead of an m64n128 (the chains are latency-bound per step): the state
// is held as S^T[d][v] (two m64n32 accumulators), the deltas as D^T[i][v] = W S^T (A = W, B = S^ from
// shared memory), and S_{c+1}^T = k^T d~^T + seed (A = the k^ tile read transposed, B = d~). Each
// element's chain -- operands, k16 steps, seed -- is the one recipe.cuh defines; an MMA's result with
// its operands swapped is the transpose, bit for bit (tests/test_mma_facts_gpu.py).
//
// Sweep execution: one consumer warpgroup, a loader warp (W, k^, this CTA's u rows and the gates of each
// chunk, bulk copies into a double-buffered stage) and a publisher warp (S^ and the deltas of each chunk,
// from double-buffered staging tiles to the workspace).
#include <c10/cuda/CUDAException.h>

#include <algorithm>
#include <type_traits>

#include "launch.h"
#include "prefill.cuh"
#include "tile.cuh"

namespace gdn_cpr {
namespace {

constexpr int CONSUMERS = 128;
constexpr int SW_THREADS = CONSUMERS + 64;   // + a loader warp and a publisher warp
constexpr int OROWS = 64;                    // value rows of an output unit
static_assert(ZERO_BYTES == OROWS * D * 2, "the zero S^ operand is an output unit's");
constexpr int GROUP_MAX_AVG = 8;             // grouped packs: at most this many chunks per sequence on average
constexpr int GROUP_CHAIN = 64;              // grouped packs: a role's chain of about this many chunks at most
// VR = value rows of a sweep CTA (32 or 16: the chains are m64n32 / m64n16), a per-call execution choice

template <int VR>
struct SwStage {   // every member a multiple of 1024 bytes: 128-byte-swizzle tiles stay aligned
  uint16_t w[C * D];    // Wide<64>: W [i][d]
  uint16_t k[C * D];    // Wide<64>: k^ [i][d]
  uint16_t u[VR * C];   // Tile<64>: this CTA's rows of U [v][i]
  GateTile gt;
};

template <int VR>
struct SwSmem {   // sweep CTAs
  SwStage<VR> stage[2];
  uint16_t shat[2][VR * D];   // Wide<32>: S^ [v][d] (the B operand of the delta chain), then to the workspace
  uint16_t dt[2][VR * C];     // Tile<64>: deltas [v][i], on their way to the workspace
  uint16_t dtl[VR * C];       // Tile<64>: d~ [v][i] (the B operand of the boundary chain)
  uint16_t dold[VR * C];      // Tile<64>: the input slot's deltas [v][i], i < r_lo of the first chunk
  uint64_t full[2], empty[2], pub_full[2], pub_empty[2];
};


__device__ __forceinline__ void consumer_sync() { asm volatile("bar.sync 1, %0;" ::"n"(CONSUMERS) : "memory"); }


// A sequence the sweep has nothing to do for: no slot to move, and a boundary state that is known without
// it -- zero, from no input slot through one open chunk (the output units need nothing from the sweep for
// a zero-start chunk, and write the output slot of such a sequence themselves: out_commit).
__device__ __forceinline__ bool sweep_idle(int si, int so, int len, int nch) {
  return (len == 0 && so == si) || (si < 0 && nch == 1 && (so < 0 || len < C));
}

// One sequence's geometry for a sweep role.
struct SeqGeo {
  int n, cs, len, si, so, pos_in, j0, nch, base;
  __device__ __forceinline__ bool idle() const { return sweep_idle(si, so, len, nch); }
};
__device__ __forceinline__ SeqGeo seq_geo(const PrefillParams &p, int n) {
  SeqGeo s;
  s.n = n;
  s.cs = p.cu[n];
  s.len = p.cu[n + 1] - s.cs;
  s.si = p.slot_in_of(n);
  s.so = p.slot_out_of(n);
  s.pos_in = p.pos_in(n);
  s.j0 = s.pos_in & (C - 1);
  s.nch = chunks_of(s.len, s.j0);
  s.base = p.chunk0[n];
  return s;
}

// Who writes a sequence's output slot besides its boundary state (the open-chunk rows and the position):
// the sweep, where the commit overlaps the last chunk's output units -- or those output units, where the
// sweep's next sequence would wait for it (grouped roles) or there is no sweep (sweep_idle).
__device__ __forceinline__ bool units_commit(const PrefillParams &p, int si, int so, int len, int nch) {
  return len > 0 && (p.groups < p.nseq || sweep_idle(si, so, len, nch));
}

// The output slot's open-chunk rows [r_first, jf) of sequence n, whose last chunk is `g`, and its position:
// one writer's share, of `parts` -- the row-major sections (k^, X, P rows; G, beta) by row modulo parts,
// the value-major ones (v, deltas) by the writer's value rows [v0, v0 + rows). `dt` holds the last chunk's
// deltas of those rows (rows [0, r_lo) being the input slot's), `gt` its gates. Value-major sections move
// in whole 16-byte chunks: rows past jf are free (state.cuh), rows before r_first equal the slot's own.
template <int THREADS>
__device__ void write_open_rows(const PrefillParams &p, const ChunkGeo &g, int h, int part, int parts, int v0, int rows,
                                const Tile<64> &dt, const GateTile &gt) {
  const int tid = threadIdx.x, n = g.n;
  const int si = p.slot_in_of(n), so = p.slot_out_of(n);
  const OpenRows open = open_rows(g, si, so);
  const int jf = open.jf, r_new = open.r_new, r_first = open.r_first;
  if (part == 0 && tid < NPOS) p.st.pos(so, h)[tid] = p.pos_in(n) + g.len;
  if (jf <= r_first) return;
  const Wide<64> kt{p.k_tile(p.chunk0[n] + g.k, p.key_head(h))};
  const float *fx = p.fx_tile(n, h), *fp = p.fp_tile(n, h);
  const int pb0 = jf & ~15;   // the open 16-row block starts here
  for (int x = tid; x < (jf - r_first) * 16; x += THREADS) {
    const int i = r_first + (x >> 4), c = x & 15;
    if (i % parts != part) continue;
    *reinterpret_cast<uint4 *>(p.st.kb_tile(so, h).chunk(i, c)) = *reinterpret_cast<const uint4 *>(kt.chunk(i, c));
    *reinterpret_cast<float4 *>(p.st.xb(so, h) + i * C + c * 4) = *reinterpret_cast<const float4 *>(fx + i * C + c * 4);
    if (i >= pb0 && pb0 > 0)
      *reinterpret_cast<float4 *>(p.st.pb(so, h) + (i - pb0) * C + c * 4) = *reinterpret_cast<const float4 *>(fp + (i - pb0) * C + c * 4);
  }
  const int c0 = r_first >> 3, nc = ((jf + 7) >> 3) - c0;   // the 8-row chunks of [r_first, jf)
  for (int x = tid; x < rows * nc; x += THREADS) {   // v fastest: the value reads coalesce
    const int v = x % rows, c = c0 + x / rows;
    uint16_t vv[8];
    #pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int i = 8 * c + e;
      vv[e] = i < r_new ? p.st.vb_tile(si, h).at(v0 + v, i)   // predates the call
            : i < jf    ? p.v[g.token(i) * p.vs0 + h * p.vs1 + v0 + v]
                        : uint16_t(0);
    }
    *reinterpret_cast<uint4 *>(p.st.vb_tile(so, h).chunk(v0 + v, c)) =
        make_uint4(pack(vv[0], vv[1]), pack(vv[2], vv[3]), pack(vv[4], vv[5]), pack(vv[6], vv[7]));
    *reinterpret_cast<uint4 *>(p.st.db_tile(so, h).chunk(v0 + v, c)) = *reinterpret_cast<const uint4 *>(dt.chunk(v, c));
  }
  if (tid >= r_first && tid < jf && tid % parts == part) {
    p.st.gb(so, h)[tid] = gt.G[tid];
    p.st.bb(so, h)[tid] = gt.beta[tid];
  }
}

// Commit a sequence's output slot, this CTA's value rows: S and its bf16 image from the final S^T; the
// open-chunk rows and the position unless the output units write them (units_commit), from the deltas of
// the last chunk in staging buffer `last_buf`; or, for a slot that moves without new tokens, a copy.
template <int VR>
__device__ void commit(const PrefillParams &p, SwSmem<VR> &sm, const SeqGeo &sg, int h, int vh, const float (&ST)[2][VR / 2],
                       int last_buf) {
  constexpr int NVH = D / VR, NA = VR / 2;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, gq = lane >> 2, tq = lane & 3;
  const int v0 = VR * vh, si = sg.si, so = sg.so;
  #pragma unroll
  for (int db = 0; db < 2; ++db)   // S [v][d] fp32, from S^T[d][v] (element-wise)
    #pragma unroll
    for (int e = 0; e < NA; ++e) {
      const int d = 64 * db + 16 * warp + gq + 8 * ((e >> 1) & 1), v = v0 + 8 * (e >> 2) + 2 * tq + (e & 1);
      s_store(p.st, so, h, s_index(v, d), ST[db][e]);
    }
  if (sg.len > 0) {
    if (units_commit(p, si, so, sg.len, sg.nch)) return;
    consumer_sync();   // every warp's deltas of the last chunk are in their staging tile
    const ChunkGeo gl = chunk_geo(sg.n, sg.nch - 1, sg.cs, sg.len, sg.j0);
    write_open_rows<CONSUMERS>(p, gl, h, vh, NVH, v0, VR, Tile<64>{sm.dt[last_buf]}, *p.gt_tile(sg.base + sg.nch - 1, h));
    return;
  }
  // the slot moves unchanged (so != si here): its open rows [0, jf), row-major sections split over the
  // slabs by row, value-major ones by column
  if (si >= 0) {
    const int jf = sg.j0;
    for (int x = tid; x < jf * 16; x += CONSUMERS) {
      const int i = x >> 4, c = x & 15;
      if (i % NVH != vh) continue;
      *reinterpret_cast<uint4 *>(p.st.kb_tile(so, h).chunk(i, c)) = *reinterpret_cast<const uint4 *>(p.st.kb_tile(si, h).chunk(i, c));
      *reinterpret_cast<float4 *>(p.st.xb(so, h) + i * C + c * 4) = *reinterpret_cast<const float4 *>(p.st.xb(si, h) + i * C + c * 4);
      if (i < 16) *reinterpret_cast<float4 *>(p.st.pb(so, h) + i * C + c * 4) = *reinterpret_cast<const float4 *>(p.st.pb(si, h) + i * C + c * 4);
    }
    for (int x = tid; x < VR * 8; x += CONSUMERS) {
      const int v = v0 + (x >> 3), c = x & 7;
      *reinterpret_cast<uint4 *>(p.st.vb_tile(so, h).chunk(v, c)) = *reinterpret_cast<const uint4 *>(p.st.vb_tile(si, h).chunk(v, c));
      *reinterpret_cast<uint4 *>(p.st.db_tile(so, h).chunk(v, c)) = *reinterpret_cast<const uint4 *>(p.st.db_tile(si, h).chunk(v, c));
    }
    if (tid < jf && tid % NVH == vh) {
      p.st.gb(so, h)[tid] = p.st.gb(si, h)[tid];
      p.st.bb(so, h)[tid] = p.st.bb(si, h)[tid];
    }
  }
  if (vh == 0 && tid < NPOS) p.st.pos(so, h)[tid] = sg.pos_in;
}

// The chains of sequence group g (sequences g, g + groups, g + 2 groups, ...) at value head h, value rows
// [VR vh, VR vh + VR), one after another through one pipeline. A pack of many sequences groups them so
// that a role's fixed cost -- barrier setup, the first stage's latency, the drain -- is paid per group, not
// per sequence; the groups are strided so that the roles, running side by side, finish the sequences in
// order (sequences g < groups first, and so on), the order the output units take them.
template <int VR>
__device__ void sweep_cta(const PrefillParams &p, char *smem_raw, int g, int h, int vh) {
  constexpr int NVH = D / VR, NA = VR / 2;   // sweep CTAs per head; accumulator registers of m64 x VR
  SwSmem<VR> &sm = *reinterpret_cast<SwSmem<VR> *>(smem_raw);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, gq = lane >> 2, tq = lane & 3;
  const int hq = p.key_head(h), v0 = VR * vh;
  const SeqGeo first = seq_geo(p, g);   // loads in flight under the barrier setup
  auto geo = [&](int n) { return n == g ? first : seq_geo(p, n); };

  if (tid == 0) {
    for (int b = 0; b < 2; ++b) {
      mbar_init(&sm.full[b], 1);
      mbar_init(&sm.empty[b], CONSUMERS);
      mbar_init(&sm.pub_full[b], 1);
      mbar_init(&sm.pub_empty[b], 1);
    }
    fence_async_smem();
  }
  __syncthreads();

  // ---------------------------------------------------------------- loader and publisher warps
  // Both walk the role's chunks in order; t counts them across the sequences (the stage and staging
  // buffers alternate on t).
  if (warp == CONSUMERS / 32) {
    if (lane == 0)
      for (int n = g, t = 0; n < p.nseq; n += p.groups) {
        const SeqGeo sg = geo(n);
        if (sg.idle()) continue;
        for (int k = 0; k < sg.nch; ++k, ++t) {
          const int b = t & 1, slot = sg.base + k;
          if (t >= 2) mbar_wait(&sm.empty[b], ((t - 2) >> 1) & 1);
          SwStage<VR> &st = sm.stage[b];
          mbar_expect(&sm.full[b], 2 * TILE_BYTES + VR * C * 2 + GT_BYTES);
          bulk_load(st.w, p.w_tile(slot, h), TILE_BYTES, &sm.full[b]);
          bulk_load(st.k, p.k_tile(slot, hq), TILE_BYTES, &sm.full[b]);
          bulk_load(st.u, p.u_tile(slot, h) + v0 * C, VR * C * 2, &sm.full[b]);
          bulk_load(&st.gt, p.gt_tile(slot, h), GT_BYTES, &sm.full[b]);
        }
      }
    return;
  }
  if (warp == CONSUMERS / 32 + 1) {   // publisher: chunk k's S^ and deltas -> the workspace
    // Release staging tiles once copies have read them. Drain outstanding stores whenever no staged
    // chunk is waiting, and before advancing to the next sequence or finishing the kernel.
    if (lane == 0)
      for (int n = g, t = 0; n < p.nseq; n += p.groups) {
        const SeqGeo sg = geo(n);
        if (sg.idle()) continue;
        for (int k = 0, pending = 0; k < sg.nch || pending > 0;) {
          const bool staged = k < sg.nch && mbar_test(&sm.pub_full[t & 1], (t >> 1) & 1);
          if (k < sg.nch && (staged || pending == 0)) {
            const int b = t & 1;
            mbar_wait(&sm.pub_full[b], (t >> 1) & 1);
            if (!p.zero_start(n, k)) {   // (else the output units do without)
              bulk_store(p.sh_tile(sg.base + k, h, v0), sm.shat[b], VR * D * 2);
              bulk_store(p.d_tile(sg.base + k, h, v0), sm.dt[b], VR * C * 2);
              bulk_commit();
              bulk_wait_read<0>();
            }
            mbar_arrive(&sm.pub_empty[b]);   // the staging tiles may be refilled
            ++k, ++t, ++pending;
          } else {
            bulk_wait<0>();                  // chunks [0, k) have landed
            pending = 0;
          }
        }
      }
    return;
  }

  // ---------------------------------------------------------------- consumer warpgroup
  const Tile<64> dold{sm.dold}, dtl{sm.dtl};
  int t = 0;   // the role's chunk count so far
  for (int n = g; n < p.nseq; n += p.groups) {
    const SeqGeo sg = geo(n);
    if (sg.idle()) continue;
    const int si = sg.si, so = sg.so, len = sg.len, j0 = sg.j0, nch = sg.nch;
    // ST[db][4 nt + e]: S^T[d][v], d = 64 db + 16 warp + gq (+8 for e >= 2), v = v0 + 8 nt + 2 tq (+1 for odd e)
    float ST[2][NA];
    #pragma unroll
    for (int db = 0; db < 2; ++db)
      #pragma unroll
      for (int e = 0; e < NA; ++e) {
        const int d = 64 * db + 16 * warp + gq + 8 * ((e >> 1) & 1), v = v0 + 8 * (e >> 2) + 2 * tq + (e & 1);
        ST[db][e] = si >= 0 ? s_load(p.st, si, h, s_index(v, d)) : 0.f;
      }
    consumer_sync();   // the previous sequence's steps have read dold
    if (j0 > 0 && si >= 0 && len > 0)   // the input slot's deltas of the open chunk, this CTA's value rows
      for (int x = tid; x < VR * 8; x += CONSUMERS) {
        const int v = x >> 3, c = x & 7;
        *reinterpret_cast<uint4 *>(dold.chunk(v, c)) = *reinterpret_cast<const uint4 *>(p.st.db_tile(si, h).chunk(v0 + v, c));
      }
    consumer_sync();

    auto step = [&](auto stage_index, int k, int t) {   // chunk k of the sequence, step t of the role
      constexpr int b = decltype(stage_index)::value;
      const int ph = (t >> 1) & 1;
      const ChunkGeo cg = chunk_geo(n, k, sg.cs, len, j0);
      SwStage<VR> &st = sm.stage[b];
      const Wide<64> wt{st.w}, kt{st.k};
      const Wide<VR> shat{sm.shat[b]};
      const Tile<64> ut{st.u}, dt{sm.dt[b]};

      // 1. S^ = rn(S_c) -> [v][d] (the delta chain's B operand, and the output units' S^)
      if (t >= 2) mbar_wait(&sm.pub_empty[b], ph ^ 1);   // the publisher has read chunk t - 2's staging tiles
      #pragma unroll
      for (int db = 0; db < 2; ++db) store_transposed(shat.half(db), 0, ST[db], warp, lane);
      fence_async_smem();
      consumer_sync();
      mbar_wait(&sm.full[b], ph);

      // 2. D^T[i][v] = CHAIN128_d(W[i][d], S^[v][d]) (the recipe's CHAIN128(S^, w_i), transposed)
      float acc[NA];
      wg_fence();
      #pragma unroll
      for (int s = 0; s < 8; ++s)
        wgmma_ss<VR, 0>(acc, desc_kmajor(wt.half(s >> 2).base, 16 * (s & 3)), desc_kmajor(shat.half(s >> 2).base, 16 * (s & 3)),
                        s > 0);
      wg_commit();
      // meanwhile: the u entries and gates of this thread's elements (i = 16 warp + gq (+8), v = 8 nt + 2 tq (+1))
      uint16_t ub[NA];
      #pragma unroll
      for (int e = 0; e < NA; ++e) ub[e] = ut.at(8 * (e >> 2) + 2 * tq + (e & 1), 16 * warp + gq + 8 * ((e >> 1) & 1));
      const float lam0 = st.gt.lam[16 * warp + gq], lam1 = st.gt.lam[16 * warp + gq + 8];
      wg_wait<0>();
      wg_pin(acc);

      // 3. deltas -> their staging tile (for the output units) ; d~ -> its tile (the boundary chain's B)
      uint16_t dbits[NA], tbits[NA];
      #pragma unroll
      for (int e = 0; e < NA; ++e) dbits[e] = delta_entry(ub[e], acc[e]);
      #pragma unroll
      for (int hi = 0; hi < 2; ++hi) {   // this thread's two chunk rows i: the input slot's deltas before r_lo,
        const int i = 16 * warp + gq + 8 * hi;   // zeros past r_hi (edge chunks only)
        if (i < cg.r_lo || i >= cg.r_hi)
          #pragma unroll
          for (int e = 2 * hi; e < NA; e += (e & 1) ? 3 : 1)   // e with (e >> 1) & 1 == hi
            dbits[e] = i < cg.r_lo ? dold.at(8 * (e >> 2) + 2 * tq + (e & 1), i) : uint16_t(0);
      }
      #pragma unroll
      for (int e = 0; e < NA; ++e) tbits[e] = dtilde_entry(dbits[e], (e >> 1) & 1 ? lam1 : lam0);
      store_transposed_bits(dt, 0, dbits, warp, lane);
      store_transposed_bits(dtl, 0, tbits, warp, lane);
      fence_async_smem();
      consumer_sync();
      if (tid == 0) mbar_arrive(&sm.pub_full[b]);

      // 4. S_{c+1}^T[d][v] = CHAIN64_i(k^[i][d], d~[v][i]; S_c^T (*) e(G_63)): two d blocks, interleaved
      if (cg.complete()) {
        const float gam_last = st.gt.gam[C - 1];
        #pragma unroll
        for (int db = 0; db < 2; ++db)
          #pragma unroll
          for (int e = 0; e < NA; ++e) ST[db][e] = state_seed(ST[db][e], gam_last);
        wg_pin(ST[0]);
        wg_pin(ST[1]);
        wg_fence();
        #pragma unroll
        for (int s = 0; s < 4; ++s)   // k16 step s = chunk rows 16s..16s+15
          #pragma unroll
          for (int db = 0; db < 2; ++db)
            wgmma_ss<VR, 1>(ST[db], desc_mnmajor_a(kt.half(db).base, 16 * s), desc_kmajor(dtl.base, 16 * s), true);
        wg_commit();
        wg_wait<0>();
        wg_pin(ST[0]);
        wg_pin(ST[1]);
      }
      // Every reader releases the stage directly. The loader waits for all CONSUMERS arrivals
      // before its next asynchronous writes, including the generic gate reads above.
      mbar_arrive(&sm.empty[b]);
    };
    // Step t runs on stage t & 1, a compile-time index: a sequence that starts on stage 1 takes that step
    // alone, then pairs (a runtime stage index per step costs the long chains ~3%, measured).
    int k = 0;
    if ((t & 1) && k < nch) step(std::integral_constant<int, 1>{}, k++, t++);
    for (; k + 1 < nch; k += 2, t += 2) {
      step(std::integral_constant<int, 0>{}, k, t);
      step(std::integral_constant<int, 1>{}, k + 1, t + 1);
    }
    if (k < nch) step(std::integral_constant<int, 0>{}, k++, t++);
    if (so >= 0) commit<VR>(p, sm, sg, h, vh, ST, (t - 1) & 1);
  }
}

// Output unit u = (chunk, value head, value half), in the order the sweeps produce the chunks. The roles
// take the sequences in rounds, k-major within each round. The plan computes that permutation once;
// every output unit now only looks up its chunk. Above OUTPUT_ORDER_MAX_SEQS the order stays dense.
__device__ int output_chunk(const PrefillParams &p, int total, int u, int &h, int &half) {
  constexpr int HALVES = D / OROWS;
  const int per = p.hv * HALVES;
  const int grp = u / per;
  h = (u % per) / HALVES, half = u % HALVES;
  if (grp >= total) return -1;
  if (p.nseq > OUTPUT_ORDER_MAX_SEQS || p.groups == 1) return grp;
  return p.order[grp];
}

struct OutBuf {   // one output unit's operands (every member a multiple of 1024 bytes but the last)
  uint16_t sh[OROWS * D];   // Wide<64>: S^ [v][d]
  uint16_t q[C * D];        // Wide<64>: q^ [j][d]
  uint16_t d[OROWS * C];    // Tile<64>: deltas [v][i]
  uint16_t m[C * C];        // Tile<64>: M [j][i]
  GateTile gt;
};

enum : int { CMD_UNIT, CMD_END };

struct OutSmem {   // output units: a producer lane fills two buffers ahead of the consumer warpgroup
  OutBuf buf[2];
  uint16_t o[C * OROWS];   // Tile<64>: outputs [j][v]
  uint64_t full[2], empty[2];
  int cmd[2][4];           // kind, chunk slot, head, value half
};

// One producer lane fills the next free buffer. The preceding sweep launch completed every
// nonzero S^ and delta operand, so no output CTA waits for another CTA's progress.
template <int VR>
__device__ void out_producer(const PrefillParams &p, OutSmem &sm) {
  constexpr int OSLABS = OROWS / VR;
  const int total = p.chunk0[p.nseq];
  for (int i = 0;;) {
    const int b = i & 1;
    const int u = atomicAdd(p.unit_next, 1);
    int h = 0, half = 0;
    const int s = output_chunk(p, total, u, h, half);
    if (s < 0) {
      if (i >= 2) mbar_wait(&sm.empty[b], ((i - 2) >> 1) & 1);
      sm.cmd[b][0] = CMD_END;
      mbar_arrive(&sm.full[b]);
      return;
    }
    const int n = p.seq_of_chunk(s);
    const ChunkGeo g = chunk_geo(n, s - p.chunk0[n], p.cu[n], p.cu[n + 1] - p.cu[n], p.pos_in(n) & (C - 1));
    const bool zero = p.zero_start(n, g.k);   // S^ = 0, deltas = u: nothing from the sweep
    if (i >= 2) mbar_wait(&sm.empty[b], ((i - 2) >> 1) & 1);
    sm.cmd[b][0] = CMD_UNIT, sm.cmd[b][1] = s, sm.cmd[b][2] = h, sm.cmd[b][3] = half;
    OutBuf &ob = sm.buf[b];
    const Wide<64> sht{ob.sh};
    const Tile<64> dtile{ob.d};
    mbar_expect(&sm.full[b], TILE_BYTES + MT_BYTES + GT_BYTES + OROWS * D * 2 + OROWS * C * 2);
    bulk_load(ob.q, p.q_tile(s, p.key_head(h)), TILE_BYTES, &sm.full[b]);   // pass-1 tiles: complete before this grid
    bulk_load(ob.m, p.m_tile(s, h), MT_BYTES, &sm.full[b]);
    bulk_load(&ob.gt, p.gt_tile(s, h), GT_BYTES, &sm.full[b]);
    if (zero) {
      bulk_load(ob.sh, p.zeros, OROWS * D * 2, &sm.full[b]);
      bulk_load(ob.d, p.u_tile(s, h) + OROWS * half * C, OROWS * C * 2, &sm.full[b]);
      ++i;
      continue;
    }
    #pragma unroll
    for (int sl = 0; sl < OSLABS; ++sl) {   // the sweep slabs of these value rows, once they have passed
      const int vh = half * OSLABS + sl;
      const Wide<VR> src{p.sh_tile(s, h, VR * vh)};
      #pragma unroll
      for (int hf = 0; hf < 2; ++hf) bulk_load(sht.half(hf).base + sl * VR * 64, src.half(hf).base, VR * 64 * 2, &sm.full[b]);
      bulk_load(dtile.base + sl * VR * C, p.d_tile(s, h, VR * vh), VR * C * 2, &sm.full[b]);
    }
    ++i;
  }
}

// Trainer mode: an output unit's S^ rows (FLA's h, value-major) and deltas (FLA's v_new, token-major),
// by the consumer warpgroup from the unit's operand buffer.
// The dense chunk order of a stateless pack is FLA's (each sequence's ceil(len / 64) chunks back to back).
__device__ void save_unit(const PrefillParams &p, const OutBuf &ob, const ChunkGeo &g, int chunk, int h, int v0) {
  const int tid = threadIdx.x;
  const Wide<64> sht{const_cast<uint16_t *>(ob.sh)};
  const Tile<64> dt{const_cast<uint16_t *>(ob.d)};
  uint16_t *hd = p.sv.h + ((int64_t(chunk) * p.hv + h) * D + v0) * D;
  for (int x = tid; x < OROWS * 16; x += CONSUMERS)
    *reinterpret_cast<uint4 *>(hd + (x >> 4) * D + (x & 15) * 8) = *reinterpret_cast<const uint4 *>(sht.chunk(x >> 4, x & 15));
  for (int x = tid; x < C * (OROWS / 8); x += CONSUMERS) {
    const int i = x >> 3, vg = x & 7;
    if (i < g.r_lo || i >= g.r_hi) continue;
    uint32_t w[4];
    #pragma unroll
    for (int e = 0; e < 4; ++e) w[e] = pack(dt.at(8 * vg + 2 * e, i), dt.at(8 * vg + 2 * e + 1, i));
    *reinterpret_cast<uint4 *>(p.sv.v_new + (g.token(i) * p.hv + h) * D + v0 + 8 * vg) = make_uint4(w[0], w[1], w[2], w[3]);
  }
}

// The consumer warpgroup: o_j of each unit the producer posts, until it posts something else (returned).
__device__ int out_consumer(const PrefillParams &p, OutSmem &sm) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, tq = lane & 3;
  for (int i = 0;; ++i) {
    const int b = i & 1;
    mbar_wait(&sm.full[b], (i >> 1) & 1);
    if (sm.cmd[b][0] != CMD_UNIT) return b;
    const int s = sm.cmd[b][1], h = sm.cmd[b][2], half = sm.cmd[b][3];
    const OutBuf &ob = sm.buf[b];
    const Wide<64> sht{const_cast<uint16_t *>(ob.sh)}, qt{const_cast<uint16_t *>(ob.q)};
    float o[32];   // CHAIN128(S^, q^_j)
    wg_fence();
    #pragma unroll
    for (int k = 0; k < 8; ++k)
      wgmma_n64_ss(o, desc_kmajor(sht.half(k >> 2).base, 16 * (k & 3)), desc_kmajor(qt.half(k >> 2).base, 16 * (k & 3)), k > 0);
    wg_commit();
    wg_wait<0>();
    wg_pin(o);
    #pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      const float2 gam = *reinterpret_cast<const float2 *>(ob.gt.gam + 8 * nt + 2 * tq);
      o[4 * nt] = out_seed(o[4 * nt], gam.x);
      o[4 * nt + 1] = out_seed(o[4 * nt + 1], gam.y);
      o[4 * nt + 2] = out_seed(o[4 * nt + 2], gam.x);
      o[4 * nt + 3] = out_seed(o[4 * nt + 3], gam.y);
    }
    wg_pin(o);
    wg_fence();
    #pragma unroll
    for (int k = 0; k < 4; ++k)   // k16 step k = chunk rows 16k..16k+15
      wgmma_n64_ss(o, desc_kmajor(ob.d, 16 * k), desc_kmajor(ob.m, 16 * k), true);
    wg_commit();
    wg_wait<0>();
    wg_pin(o);
    const int n = p.seq_of_chunk(s);
    const ChunkGeo g = chunk_geo(n, s - p.chunk0[n], p.cu[n], p.cu[n + 1] - p.cu[n], p.pos_in(n) & (C - 1));
    const int v0 = OROWS * half;
    if (p.sv.h)   // trainer mode: this unit's S^ rows and deltas in FLA's layouts (h, v_new)
      save_unit(p, ob, g, s, h, v0);
    if (g.last() && p.slot_out_of(n) >= 0) {   // stateful: this unit's share of the output slot (commit)
      const int si = p.slot_in_of(n), so = p.slot_out_of(n);
      if (sweep_idle(si, so, g.len, g.nch)) {   // S = 0, this unit's value rows (a contiguous run)
        s_move_rows(p.st, -1, so, h, v0, OROWS, tid, CONSUMERS);   // whole 16-row tiles: zeroed
      }
      if (units_commit(p, si, so, g.len, g.nch))
        write_open_rows<CONSUMERS>(p, g, h, half, D / OROWS, v0, OROWS, Tile<64>{const_cast<uint16_t *>(ob.d)}, ob.gt);
    }
    mbar_arrive(&sm.empty[b]);   // every reader releases the buffer before the next async overwrite
    const Tile<64> ot{sm.o};
    consumer_sync();   // the previous unit's outputs have left the tile
    #pragma unroll
    for (int np = 0; np < 4; ++np)   // n8 tiles 2np, 2np + 1 = outputs j in 16np..16np+15
      stsm_x4_t(smem_u32(ot.chunk(16 * np + (lane & 7) + ((lane >> 4) << 3), 2 * warp + ((lane >> 3) & 1))),
                pack(rn(o[8 * np]), rn(o[8 * np + 1])), pack(rn(o[8 * np + 2]), rn(o[8 * np + 3])),
                pack(rn(o[8 * np + 4]), rn(o[8 * np + 5])), pack(rn(o[8 * np + 6]), rn(o[8 * np + 7])));
    consumer_sync();
    for (int x = tid; x < (g.r_hi - g.r_lo) * 8; x += CONSUMERS) {
      const int j = g.r_lo + (x >> 3), c = x & 7;
      *reinterpret_cast<uint4 *>(p.out + int64_t(g.token(j)) * p.os0 + h * p.os1 + v0 + c * 8) =
          *reinterpret_cast<const uint4 *>(ot.chunk(j, c));
    }
  }
}

// One sweep role per CTA; invalidate its barriers after all role threads finish.
template <int VR>
__device__ void run_sweep(const PrefillParams &p, char *smem_raw, int r) {
  constexpr int NVH = D / VR;
  sweep_cta<VR>(p, smem_raw, r / (p.hv * NVH), (r / NVH) % p.hv, r % NVH);
  fence_async_smem();
  __syncthreads();
  if (threadIdx.x == 0) {
    SwSmem<VR> &sm = *reinterpret_cast<SwSmem<VR> *>(smem_raw);
    for (int b = 0; b < 2; ++b) {
      mbar_inval(&sm.full[b]);
      mbar_inval(&sm.empty[b]);
      mbar_inval(&sm.pub_full[b]);
      mbar_inval(&sm.pub_empty[b]);
    }
  }
  __syncthreads();
}

// Separate launch lifetimes keep sweep and output TMA destinations disjoint. Stream order
// completes all state operands before outputs; both kernels have only CTA-local waits.
template <int VR>
__global__ __launch_bounds__(SW_THREADS, 2) void state_kernel(const __grid_constant__ PrefillParams p) {
  extern __shared__ __align__(1024) char raw[];
  run_sweep<VR>(p, raw, blockIdx.x);
}

template <int VR>
__global__ __launch_bounds__(CONSUMERS + 32, 2) void output_kernel(const __grid_constant__ PrefillParams p) {
  extern __shared__ __align__(1024) char raw[];
  OutSmem &sm = *reinterpret_cast<OutSmem *>(raw);
  const int tid = threadIdx.x;
  if (tid == 0) {
    for (int b = 0; b < 2; ++b) {
      mbar_init(&sm.full[b], 1);
      mbar_init(&sm.empty[b], CONSUMERS);
    }
    fence_async_smem();
  }
  __syncthreads();
  if (tid == CONSUMERS) out_producer<VR>(p, sm);
  else if (tid < CONSUMERS) out_consumer(p, sm);
  fence_async_smem();
  __syncthreads();
  if (tid == 0) for (int b = 0; b < 2; ++b) {
    mbar_inval(&sm.full[b]);
    mbar_inval(&sm.empty[b]);
  }
}

template <int VR>
void launch(const PrefillParams &p, int64_t max_chunks, int per_sm, cudaStream_t stream) {
  PrefillParams q = p;
  q.units = int(max_chunks * p.hv * (D / OROWS));
  const int roles = p.groups * p.hv * (D / VR);
  const int64_t outputs = std::min<int64_t>(per_sm * int64_t(device_sms()), q.units);
  if (roles > 0) {
    C10_CUDA_CHECK(allow_smem<state_kernel<VR>>(sizeof(SwSmem<VR>)));
    state_kernel<VR><<<roles, SW_THREADS, sizeof(SwSmem<VR>), stream>>>(q);
  }
  if (outputs > 0) {
    C10_CUDA_CHECK(allow_smem<output_kernel<VR>>(sizeof(OutSmem)));
    output_kernel<VR><<<unsigned(outputs), CONSUMERS + 32, sizeof(OutSmem), stream>>>(q);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

// Sequence groups (sweep roles per head and slab), from the host's chunk bound. A role per sequence while
// the roles fit the SMs (one independent CTA per role) or the sequences
// are long (a longer chain would cost more than a late start). A pack of many short sequences is grouped
// down to the roles the SMs take at once: a role's fixed cost -- barrier setup, the first stage's
// latency, the drain -- is then paid per group rather than per sequence, and every chain starts at once;
// chains stay within about GROUP_CHAIN chunks. Any grouping gives the same bits.
int prefill_groups(int nseq, int64_t max_chunks, int roles_per_group, int sms) {
  if (int64_t(nseq) * roles_per_group <= sms || max_chunks > int64_t(GROUP_MAX_AVG) * nseq) return nseq;
  const int64_t groups = std::max<int64_t>(sms / roles_per_group, (max_chunks + GROUP_CHAIN - 1) / GROUP_CHAIN);
  return int(std::clamp<int64_t>(groups, 1, nseq));
}

// The sweep's value rows per CTA, an execution choice (every width is bitwise identical): 16 rows (8 CTAs
// per head, a shorter chain step) while the sweeps fit the machine side by side, else 32.
// `rows` 16 or 32 forces a width, 0 chooses. The schedule is a test hook (ops.py):
// `on_demand` is a compatibility no-op after separating the launches. `ctas_per_sm` sizes the output
// grid (2 are resident; more start only as others finish), `one_group` puts every
// sequence of a head and slab in one role.
void launch_sweep(const PrefillParams &p0, int64_t max_chunks, int rows, bool on_demand, int ctas_per_sm,
                  bool one_group, cudaStream_t stream) {
  PrefillParams p = p0;
  (void)on_demand;
  if (rows == 0) rows = int64_t(p.nseq) * p.hv * (D / 16) <= 128 ? 16 : 32;
  TORCH_CHECK(rows == 16 || rows == 32, "gdn_cpr_sm90: sweep rows must be 16 or 32, got ", rows);
  p.groups = one_group ? 1 : prefill_groups(p.nseq, max_chunks, p.hv * (D / rows), device_sms());
  TORCH_CHECK(ctas_per_sm >= 2, "gdn_cpr_sm90: the sweep grid needs at least 2 CTAs per SM");
  if (rows == 16) launch<16>(p, max_chunks, ctas_per_sm, stream);
  else launch<32>(p, max_chunks, ctas_per_sm, stream);
}

}  // namespace gdn_cpr
