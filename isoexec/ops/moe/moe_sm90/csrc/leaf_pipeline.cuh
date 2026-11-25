// SPDX-License-Identifier: Apache-2.0
// Two independent accumulator banks: WGMMA for leaf i overlaps the FP32 tree
// fold of leaf i-1 and TMA for leaf i+1, with one commit group per leaf.
#pragma once
#include "tma.cuh"
#include "wgmma.cuh"

template <class T, int L, bool SHARED_HIGH>
__device__ __forceinline__ void fold_leaf(float (&d)[32], int leaf,
                                          uint32_t (&pair)[16], float *pending,
                                          float *high) {
  // Pack rounded even leaves; pending holds completed left subtrees in FP32.
  // Fold each right subtree into its left sibling without reassociating adds.
  if (L > 1 && (leaf & 1) == 0) {
#pragma unroll
    for (int j = 0; j < 16; ++j)
      pair[j] = pack_leaf<T>(d[2 * j], d[2 * j + 1]);
  } else {
#pragma unroll
    for (int j = 0; j < 32; ++j) {
      float v = rounded<T>(d[j]);
      if constexpr (L > 1)
        v = __fadd_rn(unpack_leaf<T>(pair[j / 2], j % 2), v);
      if constexpr (L >= 4)
        if ((leaf & 3) == 3)
          v = __fadd_rn(pending[j], v);
      if constexpr (L == 8)
        if ((leaf & 7) == 7)
          v = __fadd_rn(
              SHARED_HIGH ? high[j * 128 + threadIdx.x] : pending[32 + j], v);
      if (leaf == L - 1)
        d[j] = v;
      else if ((leaf & 3) == 1)
        pending[j] = v;
      else if constexpr (L == 8) {
        if constexpr (SHARED_HIGH)
          high[j * 128 + threadIdx.x] = v;
        else
          pending[32 + j] = v;
      }
    }
  }
}

template <int LK>
__device__ __forceinline__ void
load_leaf(uint16_t *sm, const CUtensorMap *xm, const CUtensorMap *wm,
          uint64_t *bar, int row, int col, int expert, int leaf) {
  constexpr int BLOCKS = (LK + 63) / 64, SS = 8192;
  if (!threadIdx.x) {
    barrier_expect(bar, BLOCKS * SS * 2);
#pragma unroll
    for (int b = 0; b < BLOCKS; ++b) {
      tma2(sm + b * SS, xm, bar, leaf * LK + b * 64, row);
      tma3(sm + b * SS + 4096, wm, bar, leaf * LK + b * 64, col, expert);
    }
  }
}

template <class T, int L, int LK>
__global__ __launch_bounds__(128) void leaf_pipeline(
    void *output, const int *tiles, const int *total, const float *probs, int n,
    int k, bool narrow, int row_parts, const __grid_constant__ CUtensorMap xm,
    const __grid_constant__ CUtensorMap wm) {
  constexpr int BLOCKS = (LK + 63) / 64, SS = 8192, LS = BLOCKS * SS;
  int cols = (n + 63) / 64, tile = blockIdx.x / cols;
  if (tile >= *total)
    return;
  int col = blockIdx.x % cols * 64, expert = tiles[3 * tile],
      first = tiles[3 * tile + 1], end = tiles[3 * tile + 2];
  extern __shared__ __align__(1024) unsigned char storage[];
  auto *sm = reinterpret_cast<uint16_t *>(storage);
  __shared__ uint64_t bar[2];
  if (threadIdx.x < 2)
    barrier_init(bar + threadIdx.x);
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  load_leaf<LK>(sm, &xm, &wm, bar, first, col, expert, 0);
  if constexpr (L > 1)
    load_leaf<LK>(sm + LS, &xm, &wm, bar + 1, first, col, expert, 1);
  float acc[L > 1 ? 2 : 1][32];
  uint32_t pair[16];
  float pending[L == 8 && LK != 192 ? 64 : 32];
  // At K192, place the oldest subtree in shared memory to avoid register
  // spills.
  auto *high = reinterpret_cast<float *>(sm + 2 * LS);
#pragma unroll
  for (int leaf = 0; leaf < L; ++leaf) {
    const int slot = leaf & 1;
    float(&d)[32] = acc[slot];
#pragma unroll
    for (int j = 0; j < 32; ++j)
      d[j] = 0;
    fence_acc(d);
    barrier_wait(bar + slot, (leaf / 2) & 1);
    __syncthreads();
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
#pragma unroll
    for (int b = 0; b < BLOCKS; ++b) {
      auto *a = sm + slot * LS + b * SS;
      auto *wtile = a + 4096;
#pragma unroll
      for (int ki = 0; ki < 64; ki += 16)
        if (b * 64 + ki < LK)
          mma<T>(d, descriptor(a + ki), descriptor(wtile + ki),
                 b != 0 || ki != 0);
    }
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
    if (leaf == L - 1)
      asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    else
      asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
    // The previous leaf is complete; its operand buffer is free. Launch the
    // next transfer before doing the previous leaf's scalar tree arithmetic.
    if (leaf > 0 && leaf + 1 < L)
      load_leaf<LK>(sm + (1 - slot) * LS, &xm, &wm, bar + (1 - slot), first,
                    col, expert, leaf + 1);
    if (leaf > 0) {
      fence_acc(acc[1 - slot]);
      fold_leaf<T, L, LK == 192>(acc[1 - slot], leaf - 1, pair, pending, high);
    }
    if (leaf == L - 1) {
      fence_acc(d);
      fold_leaf<T, L, LK == 192>(d, leaf, pair, pending, high);
    }
  }
  auto &d = acc[(L - 1) & 1];
  int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
#pragma unroll
  for (int j = 0; j < 32; ++j) {
    int row = first + warp * 16 + lane / 4 + (j % 4 / 2) * 8;
    int nc = col + (j / 4) * 8 + (lane % 4) * 2 + j % 2;
    if (row < end && nc < n) {
      if (narrow)
        static_cast<T *>(output)[int64_t(row) * n + nc] = cast<T>(d[j]);
      else
        static_cast<float *>(output)[int64_t(row) * n + nc] = d[j];
    }
  }
}
