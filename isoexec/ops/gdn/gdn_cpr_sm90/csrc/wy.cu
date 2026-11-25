// SPDX-License-Identifier: Apache-2.0
// Prefill pass 1 of the two-pass prefill, one CTA per (chunk slot, value head), all chunks in parallel:
// wy_chunk.cuh's work, its tiles written to the workspace as prefill.cuh lays them out.
#include <c10/cuda/CUDAException.h>

#include "launch.h"
#include "wy_chunk.cuh"

namespace gdn_cpr {
namespace {

struct WySmem {
  uint16_t k_[C * D];    // Wide<64>: raw keys, then k^ (the K tile)
  union {
    uint16_t q[C * D];   // Wide<64>: raw queries, then q^ (the Q tile) -- dead after the KQ products
    struct {
      uint16_t m[C * C];   // Tile<64>: M [j][i]
      uint16_t xu[C * C];  // Tile<64>: xu [j][i] = rn(X_ji (*) beta_i)
    } mx;
  } qm;
  uint16_t v_[C * D];    // Wide<64>: values v [i][d]
  uint16_t xw_[C * C];   // Tile<64>: xw [j][i] = rn((X_ji (*) beta_i) (*) e(G_i))
  union {
    struct {
      float L[C * LS];   // L [j][i]
      float X[C * C];    // X [j][m]
      float P[16 * C];   // P rows of the 16-row block being solved (row 16B + r at r)
    } a;
    struct {
      uint16_t w[C * D];   // Wide<64>: the W tile
      uint16_t u[D * C];   // Tile<64> (128 rows): the U tile, value-major
    } b;
  } r;
  GateTile gt_;          // the GT tile
  float lg_[C];
  __device__ uint16_t *k() { return k_; }
  __device__ uint16_t *q() { return qm.q; }
  __device__ uint16_t *m() { return qm.mx.m; }
  __device__ uint16_t *xu() { return qm.mx.xu; }
  __device__ uint16_t *v() { return v_; }
  __device__ uint16_t *xw() { return xw_; }
  __device__ float *L() { return r.a.L; }
  __device__ float *X() { return r.a.X; }
  __device__ float *P() { return r.a.P; }
  __device__ uint16_t *w() { return r.b.w; }
  __device__ uint16_t *u() { return r.b.u; }
  __device__ GateTile &gt() { return gt_; }
  __device__ float *lg() { return lg_; }
};

// CTA (s, h): chunk s of the pack (dense order, prefill.cuh), value head h. The grid is the host's bound on
// the chunks; the plan's chunk count is known on the device only, and a chunk past it owns no sequence.
// SAVE: trainer mode, also write the backward's tensors (p.sv); a separate instantiation.
template <bool SAVE, bool FULL>
__global__ __launch_bounds__(WY_THREADS, 2) void wy_kernel(const __grid_constant__ PrefillParams p) {
  extern __shared__ __align__(1024) char smem_raw[];
  WySmem &sm = *reinterpret_cast<WySmem *>(smem_raw);
  const int s = blockIdx.x, h = blockIdx.y;
  if (p.seq_of[s] < 0) return;
  const int n = p.seq_of_chunk(s);
  const int cs = p.cu[n], len = p.cu[n + 1] - cs;
  const int si = FULL ? -1 : p.slot_in_of(n);
  // the input position from the record, not the plan's copy: one dependent load less, and no record is
  // written before pass 2's commit (a fresh sequence needs no load at all)
  const int pos_in = si >= 0 ? *p.st.pos(si, h) : 0;
  ChunkGeo g = chunk_geo(n, s - p.chunk0[n], cs, len, pos_in & (C - 1));
  if constexpr (FULL) { g.r_lo = 0; g.r_hi = C; }
  wy_chunk<SAVE, false>(p, sm, g, s, h, si);

  // ---------------------------------------------------------------- 7. tiles -> workspace
  if (threadIdx.x == 0) {
    bulk_store(p.w_tile(s, h), sm.w(), TILE_BYTES);
    bulk_store(p.u_tile(s, h), sm.u(), TILE_BYTES);
    bulk_store(p.m_tile(s, h), sm.m(), MT_BYTES);
    bulk_store(p.gt_tile(s, h), &sm.gt(), GT_BYTES);
    bulk_commit();
    bulk_wait_read<0>();   // the tiles have left shared memory: the next item may overwrite it
  }
}

}  // namespace

void launch_wy(const PrefillParams &p, int64_t max_chunks, cudaStream_t stream) {
  const int bytes = int(sizeof(WySmem));
  if (max_chunks <= 0) return;
  const dim3 grid(unsigned(max_chunks), unsigned(p.hv));
  // The preceding device plan rejects underestimated bounds. With no state and
  // T divisible by C, an accepted bound of T / C forces sum ceil(len / C) == T / C:
  // every nonempty sequence has whole chunks. Empty sequences contribute nothing;
  // an overestimated bound or any state pool selects the general kernel.
  const bool full = !p.st.base && p.T % C == 0 && max_chunks == p.T / C;
  if (full) {
    if (p.sv.kn) {
      C10_CUDA_CHECK((allow_smem<wy_kernel<true, true>>(bytes)));
      wy_kernel<true, true><<<grid, WY_THREADS, bytes, stream>>>(p);
    } else {
      C10_CUDA_CHECK((allow_smem<wy_kernel<false, true>>(bytes)));
      wy_kernel<false, true><<<grid, WY_THREADS, bytes, stream>>>(p);
    }
  } else {
    if (p.sv.kn) {
      C10_CUDA_CHECK((allow_smem<wy_kernel<true, false>>(bytes)));
      wy_kernel<true, false><<<grid, WY_THREADS, bytes, stream>>>(p);
    } else {
      C10_CUDA_CHECK((allow_smem<wy_kernel<false, false>>(bytes)));
      wy_kernel<false, false><<<grid, WY_THREADS, bytes, stream>>>(p);
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace gdn_cpr
