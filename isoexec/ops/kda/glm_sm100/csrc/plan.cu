// SPDX-License-Identifier: Apache-2.0
// The chunk table of a packed call (integer bookkeeping only; exact and order-free).
#include <cuda_runtime.h>

#include "common.cuh"
#include "params.cuh"

namespace kda_sm100 {

namespace {
__device__ __forceinline__ int block_excl_scan(int v, int *warp_sum, int &total) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  int incl = v;
  #pragma unroll
  for (int off = 1; off < 32; off *= 2) {
    const int y = __shfl_up_sync(0xffffffffu, incl, off);
    if (lane >= off) incl += y;
  }
  if (lane == 31) warp_sum[warp] = incl;
  __syncthreads();
  if (warp == 0) {
    int w = warp_sum[lane];
    #pragma unroll
    for (int off = 1; off < 32; off *= 2) {
      const int y = __shfl_up_sync(0xffffffffu, w, off);
      if (lane >= off) w += y;
    }
    warp_sum[lane] = w;
  }
  __syncthreads();
  const int excl = (warp > 0 ? warp_sum[warp - 1] : 0) + incl - v;
  total = warp_sum[31];
  __syncthreads();
  return excl;
}
}  // namespace

// One block: exclusive scans of per-sequence chunk counts and workspace rows in tiles of 1024 sequences,
// then every sequence writes its own chunk rows.
__global__ void __launch_bounds__(1024) plan_kernel(Plan p) {
  __shared__ int warp_sum[32];
  __shared__ int carry_c, carry_w;
  const int tid = threadIdx.x;
  if (tid == 0) carry_c = carry_w = 0;
  __syncthreads();
  for (int base = 0; base < p.nseq; base += 1024) {
    const int s = base + tid;
    const int len = s < p.nseq ? p.cu[s + 1] - p.cu[s] : 0;
    int pos_in = 0;
    if (s < p.nseq && p.slot_in && p.rec.base) {
      const int si = p.slot_in[s];
      pos_in = si >= 0 ? *p.rec.pos(si, 0) : 0;
    }
    const int j0 = pos_in % CL;
    const int nch = len > 0 ? (j0 + len + CL - 1) / CL : 0;
    const int nws = len > 0 ? j0 + len : 0;
    int tc, tw;
    const int ec = block_excl_scan(nch, warp_sum, tc) + carry_c;
    const int ew = block_excl_scan(nws, warp_sum, tw) + carry_w;
    if (s < p.nseq) {
      p.seq_chunk0[s] = ec;
      p.seq_pos[s] = pos_in;
      const int t0 = p.cu[s] - j0;
      for (int q = 0; q < nch; ++q) {
        p.chunk_tok[ec + q] = t0 + q * CL;
        p.chunk_lo[ec + q] = q == 0 ? j0 : 0;
        p.chunk_rows[ec + q] = min(CL, j0 + len - q * CL);
        p.chunk_seq[ec + q] = s;
        p.chunk_wb[ec + q] = ew + q * CL;
      }
    }
    __syncthreads();
    if (tid == 0) {
      carry_c += tc;
      carry_w += tw;
    }
    __syncthreads();
  }
  if (tid == 0) {
    p.seq_chunk0[p.nseq] = carry_c;
    p.count[0] = carry_c;
  }
}

void plan_launch(const Plan &p, cudaStream_t stream) { plan_kernel<<<1, 1024, 0, stream>>>(p); }

// The chunk length this build was compiled with (read back by _extension.py and the schedule tests).
int chunk_len() { return CL; }

}  // namespace kda_sm100
