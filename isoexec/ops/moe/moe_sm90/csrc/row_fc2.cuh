// SPDX-License-Identifier: Apache-2.0
// 128x64: two row warpgroups share B, with room for two resident CTAs/SM.
#pragma once
#include "tma.cuh"
#include "wgmma.cuh"

__device__ __forceinline__ void row_stage(uint16_t *sm, const CUtensorMap *xm,
                                          const CUtensorMap *wm, uint64_t *bar,
                                          int row, int col, int expert, int k) {
  if (!threadIdx.x) {
    barrier_expect(bar, 24576);
    tma2(sm, xm, bar, k, row);
    tma3(sm + 8192, wm, bar, k, col, expert);
  }
}

template <class T, int L, int LK>
__global__ __launch_bounds__(256, 2) void row_fc2(
    void *output, const int *tiles, const int *total, const float *probs, int n,
    int k, bool narrow, const __grid_constant__ CUtensorMap xm,
    const __grid_constant__ CUtensorMap wm) {
  constexpr int SS = 12288, BLOCKS = (LK + 63) / 64, ITERATIONS = BLOCKS * L;
  int cols = (n + 63) / 64, tile = blockIdx.x / cols;
  if (tile >= *total)
    return;
  int col = blockIdx.x % cols * 64, expert = tiles[3 * tile],
      first = tiles[3 * tile + 1], end = tiles[3 * tile + 2];
  end = min(end, first + 128);
  extern __shared__ __align__(1024) unsigned char storage[];
  auto *sm = reinterpret_cast<uint16_t *>(storage);
  __shared__ uint64_t bars[3];
  if (threadIdx.x < 3)
    barrier_init(bars + threadIdx.x);
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  row_stage(sm, &xm, &wm, bars, first, col, expert, 0);
  int wg = threadIdx.x / 128, tid = threadIdx.x % 128;
  float *high = reinterpret_cast<float *>(sm + 3 * SS) + wg * 4096;
  float d[32], pending[32];
  uint32_t pair[16];
#pragma unroll 1
  for (int leaf = 0; leaf < L; ++leaf) {
#pragma unroll
    for (int j = 0; j < 32; ++j)
      d[j] = 0;
    fence_acc(d);
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
#pragma unroll
    for (int b = 0; b < BLOCKS; ++b) {
      int iter = leaf * BLOCKS + b, next = iter + 1;
      if (next < ITERATIONS)
        row_stage(sm + next % 3 * SS, &xm, &wm, bars + next % 3, first, col,
                  expert, next / BLOCKS * LK + next % BLOCKS * 64);
      barrier_wait(bars + iter % 3, iter / 3 & 1);
      __syncthreads();
      auto *a = sm + iter % 3 * SS + wg * 4096;
      auto *bt = sm + iter % 3 * SS + 8192;
#pragma unroll
      for (int ki = 0; ki < 64; ki += 16)
        if (b * 64 + ki < LK)
          mma<T>(d, descriptor(a + ki), descriptor(bt + ki), b != 0 || ki != 0);
      asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
      if (b == BLOCKS - 1)
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
      else
        asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
      __syncthreads();
    }
    fence_acc(d);
    if (L > 1 && !(leaf & 1)) {
#pragma unroll
      for (int j = 0; j < 16; ++j)
        pair[j] = pack_leaf<T>(d[2 * j], d[2 * j + 1]);
    } else {
#pragma unroll
      for (int j = 0; j < 32; ++j) {
        d[j] = rounded<T>(d[j]);
        if constexpr (L > 1)
          d[j] = __fadd_rn(unpack_leaf<T>(pair[j / 2], j % 2), d[j]);
        if constexpr (L >= 4)
          if ((leaf & 3) == 3)
            d[j] = __fadd_rn(pending[j], d[j]);
        if constexpr (L == 8)
          if (leaf == 7)
            d[j] = __fadd_rn(high[j * 128 + tid], d[j]);
        if (leaf != L - 1) {
          if ((leaf & 3) == 1)
            pending[j] = d[j];
          else
            high[j * 128 + tid] = d[j];
        }
      }
    }
  }
  int lane = threadIdx.x % 32, warp = tid / 32;
#pragma unroll
  for (int j = 0; j < 32; j += 2) {
    int row = first + wg * 64 + warp * 16 + lane / 4 + (j % 4 / 2) * 8;
    int nc = col + (j / 4) * 8 + (lane % 4) * 2;
    if (row < end && nc < n) {
      auto *dst = static_cast<float *>(output) + int64_t(row) * n + nc;
      if (!(n & 1) && nc + 1 < n)
        *reinterpret_cast<float2 *>(dst) = make_float2(d[j], d[j + 1]);
      else {
        dst[0] = d[j];
        if (nc + 1 < n)
          dst[1] = d[j + 1];
      }
    }
  }
}
