// SPDX-License-Identifier: Apache-2.0
// Two row warpgroups share a B tile: 128x128 raw FC1 or 128x64 fused gate/up.
#pragma once
#include "tma.cuh"
#include "wgmma.cuh"

template <bool GLU, int ROWS>
__device__ __forceinline__ void
wide_stage(uint16_t *sm, const CUtensorMap *xm, const CUtensorMap *wm,
           uint64_t *bar, int row, int col, int n, int expert, int k) {
  if (!threadIdx.x) {
    barrier_expect(bar, (ROWS + 128) * 64 * 2);
    tma2(sm, xm, bar, k, row);
    tma3(sm + ROWS * 64, wm, bar, k, col, expert);
    if constexpr (GLU)
      tma3(sm + (ROWS + 64) * 64, wm, bar, k, col + n, expert);
  }
}

template <class T, bool GLU, int ROWS>
__global__ __launch_bounds__(ROWS * 2, ROWS == 256 ? 1 : 2) void wide_fc1(
    void *output, const int *tiles, const int *total, const float *probs, int n,
    int k, bool narrow, const __grid_constant__ CUtensorMap xm,
    const __grid_constant__ CUtensorMap wm) {
  constexpr int BN = GLU ? 64 : 128, SS = (ROWS + 128) * 64;
  int cols = (n + BN - 1) / BN, tile = blockIdx.x / cols;
  if (tile >= *total)
    return;
  int col = blockIdx.x % cols * BN, expert = tiles[3 * tile],
      first = tiles[3 * tile + 1], end = tiles[3 * tile + 2];
  end = min(end, first + ROWS);
  extern __shared__ __align__(1024) unsigned char storage[];
  auto *sm = reinterpret_cast<uint16_t *>(storage);
  __shared__ uint64_t bars[3];
  if (threadIdx.x < 3)
    barrier_init(bars + threadIdx.x);
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  wide_stage<GLU, ROWS>(sm, &xm, &wm, bars, first, col, n, expert, 0);
  int blocks = (k + 63) / 64, wg = threadIdx.x / 128;
  float d[32], u[32];
#pragma unroll
  for (int j = 0; j < 32; ++j) {
    d[j] = 0;
    u[j] = 0;
  }
  fence_acc(d);
  fence_acc(u);
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
  for (int b = 0; b < blocks; ++b) {
    if (b + 1 < blocks)
      wide_stage<GLU, ROWS>(sm + (b + 1) % 3 * SS, &xm, &wm, bars + (b + 1) % 3,
                            first, col, n, expert, (b + 1) * 64);
    barrier_wait(bars + b % 3, b / 3 & 1);
    __syncthreads();
    auto *a = sm + b % 3 * SS + wg * 4096;
    auto *bt = sm + b % 3 * SS + ROWS * 64;
#pragma unroll
    for (int ki = 0; ki < 64; ki += 16)
      if (b * 64 + ki < k)
        mma_glu<T>(d, u, descriptor(a + ki), descriptor(bt + ki),
                   b != 0 || ki != 0);
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
    if (b == blocks - 1)
      asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    else
      asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
    __syncthreads();
  }
  fence_acc(d);
  fence_acc(u);
  int lane = threadIdx.x % 32, warp = threadIdx.x % 128 / 32;
#pragma unroll
  for (int j = 0; j < 32; ++j) {
    int row = first + wg * 64 + warp * 16 + lane / 4 + (j % 4 / 2) * 8;
    int nc = col + (j / 4) * 8 + (lane % 4) * 2 + j % 2;
    if (row < end) {
      if constexpr (GLU) {
        if (nc < n) {
          float g = rounded<T>(d[j]), up = rounded<T>(u[j]);
          float act = rounded<T>(__fdividef(g, __fadd_rn(1.f, expf(-g))));
          float h = rounded<T>(__fmul_rn(act, up));
          if (probs)
            h = rounded<T>(__fmul_rn(h, probs[row]));
          static_cast<T *>(output)[int64_t(row) * n + nc] = cast<T>(h);
        }
      } else {
        if (nc < n)
          static_cast<T *>(output)[int64_t(row) * n + nc] = cast<T>(d[j]);
        if (nc + 64 < n)
          static_cast<T *>(output)[int64_t(row) * n + nc + 64] = cast<T>(u[j]);
      }
    }
  }
}
