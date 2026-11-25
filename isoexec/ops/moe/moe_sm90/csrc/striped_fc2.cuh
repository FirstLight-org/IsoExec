// SPDX-License-Identifier: Apache-2.0
// Exact 96-wide leaves: three K32 TMA strips, one WGMMA group per leaf.
#pragma once
#include "tma.cuh"
#include "wgmma.cuh"

__device__ __forceinline__ uint64_t descriptor64(const void *p) {
  uint64_t a = __cvta_generic_to_shared(p);
  return ((a & 0x3ffff) >> 4) | (uint64_t(1) << 16) | (uint64_t(32) << 32) |
         (uint64_t(2) << 62);
}

template <int ROWS, int BN>
__device__ __forceinline__ void
striped_load(uint16_t *sm, const CUtensorMap *xm, const CUtensorMap *wm,
             uint64_t *bar, int row, int col, int expert, int leaf) {
  constexpr int SS = (ROWS + BN) * 32;
  if (!threadIdx.x) {
    barrier_expect(bar, 3 * SS * 2);
#pragma unroll
    for (int b = 0; b < 3; ++b) {
      tma2(sm + b * SS, xm, bar, leaf * 96 + b * 32, row);
      tma3(sm + b * SS + ROWS * 32, wm, bar, leaf * 96 + b * 32, col, expert);
    }
  }
}

template <class T, int L, int ROWS, int BN>
__global__ __launch_bounds__(ROWS * 2, BN == 128 ? 1 : 2) void striped_fc2(
    void *output, const int *tiles, const int *total, const float *probs, int n,
    int k, bool narrow, const __grid_constant__ CUtensorMap xm,
    const __grid_constant__ CUtensorMap wm) {
  constexpr int SS = (ROWS + BN) * 32, LS = 3 * SS;
  int cols = (n + BN - 1) / BN, tile = blockIdx.x / cols;
  if (tile >= *total)
    return;
  int col = blockIdx.x % cols * BN, expert = tiles[3 * tile],
      first = tiles[3 * tile + 1], end = tiles[3 * tile + 2];
  end = min(end, first + ROWS);
  extern __shared__ __align__(1024) unsigned char storage[];
  auto *sm = reinterpret_cast<uint16_t *>(storage);
  __shared__ uint64_t bars[2];
  if (threadIdx.x < 2)
    barrier_init(bars + threadIdx.x);
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  striped_load<ROWS, BN>(sm, &xm, &wm, bars, first, col, expert, 0);
  int wg = threadIdx.x / 128, tid = threadIdx.x % 128;
  float *high = reinterpret_cast<float *>(sm + 2 * LS) + wg * (BN * 64);
  float d[BN / 64][32], pending[BN / 64][32];
  uint32_t pair[BN / 64][16];
#pragma unroll 1
  for (int leaf = 0; leaf < L; ++leaf) {
    int slot = leaf & 1;
    if (leaf + 1 < L)
      striped_load<ROWS, BN>(sm + (1 - slot) * LS, &xm, &wm, bars + (1 - slot),
                             first, col, expert, leaf + 1);
#pragma unroll
    for (int half = 0; half < BN / 64; ++half) {
#pragma unroll
      for (int j = 0; j < 32; ++j)
        d[half][j] = 0;
      fence_acc(d[half]);
    }
    barrier_wait(bars + slot, leaf / 2 & 1);
    __syncthreads();
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
#pragma unroll
    for (int b = 0; b < 3; ++b) {
      auto *a = sm + slot * LS + b * SS + wg * 2048;
      auto *bt = sm + slot * LS + b * SS + ROWS * 32;
      if constexpr (BN == 128) {
        mma_glu<T>(d[0], d[1], descriptor64(a), descriptor64(bt), b != 0);
        mma_glu<T>(d[0], d[1], descriptor64(a + 16), descriptor64(bt + 16), 1);
      } else {
        mma<T>(d[0], descriptor64(a), descriptor64(bt), b != 0);
        mma<T>(d[0], descriptor64(a + 16), descriptor64(bt + 16), 1);
      }
    }
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
#pragma unroll
    for (int half = 0; half < BN / 64; ++half) {
      fence_acc(d[half]);
      if (L > 1 && !(leaf & 1)) {
#pragma unroll
        for (int j = 0; j < 16; ++j)
          pair[half][j] = pack_leaf<T>(d[half][2 * j], d[half][2 * j + 1]);
      } else {
#pragma unroll
        for (int j = 0; j < 32; ++j) {
          d[half][j] = rounded<T>(d[half][j]);
          if constexpr (L > 1)
            d[half][j] =
                __fadd_rn(unpack_leaf<T>(pair[half][j / 2], j % 2), d[half][j]);
          if constexpr (L >= 4)
            if ((leaf & 3) == 3)
              d[half][j] = __fadd_rn(pending[half][j], d[half][j]);
          if constexpr (L == 8)
            if (leaf == 7)
              d[half][j] =
                  __fadd_rn(high[(half * 32 + j) * 128 + tid], d[half][j]);
          if (leaf != L - 1) {
            if ((leaf & 3) == 1)
              pending[half][j] = d[half][j];
            else
              high[(half * 32 + j) * 128 + tid] = d[half][j];
          }
        }
      }
    }
    __syncthreads();
  }
  int lane = threadIdx.x % 32, warp = tid / 32;
#pragma unroll
  for (int half = 0; half < BN / 64; ++half) {
#pragma unroll
    for (int j = 0; j < 32; j += 2) {
      int row = first + wg * 64 + warp * 16 + lane / 4 + (j % 4 / 2) * 8;
      int nc = col + half * 64 + (j / 4) * 8 + (lane % 4) * 2;
      if (row < end && nc < n) {
        auto *dst = static_cast<float *>(output) + int64_t(row) * n + nc;
        if (!(n & 1) && nc + 1 < n)
          *reinterpret_cast<float2 *>(dst) =
              make_float2(d[half][j], d[half][j + 1]);
        else {
          dst[0] = d[half][j];
          if (nc + 1 < n)
            dst[1] = d[half][j + 1];
        }
      }
    }
  }
}
