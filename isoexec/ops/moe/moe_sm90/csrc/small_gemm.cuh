// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "tma.cuh"
#include "wgmma.cuh"

template <bool GLU>
__device__ __forceinline__ void
small_stage(uint16_t *sm, const CUtensorMap *xm, const CUtensorMap *wm,
            uint64_t *bar, int row, int col, int n, int expert, int k) {
  if (!threadIdx.x) {
    barrier_expect(bar, (1024 + 4096 * (GLU ? 2 : 1)) * 2);
    tma2(sm, xm, bar, k, row);
    tma3(sm + 1024, wm, bar, k, col, expert);
    if constexpr (GLU)
      tma3(sm + 5120, wm, bar, k, col + n, expert);
  }
}

template <class T, int L, bool GLU, int LK = 0>
__global__ __launch_bounds__(128) void small_gemm(
    void *output, const int *tiles, const int *total, const float *probs, int n,
    int k, bool narrow, const __grid_constant__ CUtensorMap xm,
    const __grid_constant__ CUtensorMap wm) {
  constexpr int SS = 1024 + 4096 * (GLU ? 2 : 1);
  constexpr int ST = !GLU && L == 1 && LK == 64 ? 1 : 3;
  int cols = (n + 63) / 64, tile = blockIdx.x / cols;
  if (tile >= *total)
    return;
  int col = blockIdx.x % cols * 64, expert = tiles[3 * tile],
      first = tiles[3 * tile + 1], end = min(tiles[3 * tile + 2], first + 16);
  extern __shared__ __align__(1024) unsigned char storage[];
  auto *sm = reinterpret_cast<uint16_t *>(storage);
  __shared__ uint64_t bars[ST];
  if (threadIdx.x < ST)
    barrier_init(bars + threadIdx.x);
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  int leaf_k = LK ? LK : k / L, blocks = (leaf_k + 63) / 64;
  small_stage<GLU>(sm, &xm, &wm, bars, first, col, n, expert, 0);
  float d[8], u[8], p1[8], p2[8];
  uint32_t p0[4];
#pragma unroll
  for (int leaf = 0; leaf < L; ++leaf) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      d[j] = 0;
      if constexpr (GLU)
        u[j] = 0;
    }
    fence_acc(d);
    if constexpr (GLU)
      fence_acc(u);
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
    for (int b = 0; b < blocks; ++b) {
      int iter = leaf * blocks + b, next = iter + 1;
      if (next < blocks * L)
        small_stage<GLU>(sm + next % ST * SS, &xm, &wm, bars + next % ST, first,
                         col, n, expert,
                         next / blocks * leaf_k + next % blocks * 64);
      barrier_wait(bars + iter % ST, iter / ST & 1);
      __syncthreads();
      auto *a = sm + iter % ST * SS;
      if ((leaf_k & 15) && b == blocks - 1) {
        int kk = leaf_k % 64;
        for (int r = threadIdx.x; r < 16 + 64 * (GLU ? 2 : 1); r += 128)
          *reinterpret_cast<uint4 *>(a + r * 64 + (kk ^ ((r & 7) * 8))) =
              make_uint4(0, 0, 0, 0);
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncthreads();
      }
#pragma unroll
      for (int ki = 0; ki < 64; ki += 16)
        if (b * 64 + ki < leaf_k) {
          mma_small<T>(d, descriptor(a + 1024 + ki), descriptor(a + ki),
                       b != 0 || ki != 0);
          if constexpr (GLU)
            mma_small<T>(u, descriptor(a + 5120 + ki), descriptor(a + ki),
                         b != 0 || ki != 0);
        }
      asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
      if (b == blocks - 1)
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
      else
        asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
      __syncthreads();
    }
    fence_acc(d);
    if constexpr (GLU)
      fence_acc(u);
    if constexpr (!GLU) {
      if (L > 1 && !(leaf & 1)) {
#pragma unroll
        for (int j = 0; j < 4; ++j)
          p0[j] = pack_leaf<T>(d[2 * j], d[2 * j + 1]);
      } else {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          d[j] = rounded<T>(d[j]);
          if constexpr (L > 1)
            d[j] = __fadd_rn(unpack_leaf<T>(p0[j / 2], j % 2), d[j]);
          if constexpr (L >= 4)
            if ((leaf & 3) == 3)
              d[j] = __fadd_rn(p1[j], d[j]);
          if constexpr (L == 8)
            if (leaf == 7)
              d[j] = __fadd_rn(p2[j], d[j]);
          if (leaf != L - 1) {
            if ((leaf & 3) == 1)
              p1[j] = d[j];
            else
              p2[j] = d[j];
          }
        }
      }
    }
  }
  int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    int row = first + (j / 4) * 8 + (lane % 4) * 2 + j % 2;
    int nc = col + warp * 16 + lane / 4 + (j % 4 / 2) * 8;
    if (row < end && nc < n) {
      if constexpr (GLU) {
        float g = rounded<T>(d[j]), up = rounded<T>(u[j]);
        float h = rounded<T>(__fdividef(g, __fadd_rn(1.f, expf(-g))));
        h = rounded<T>(__fmul_rn(h, up));
        if (probs)
          h = rounded<T>(__fmul_rn(h, probs[row]));
        static_cast<T *>(output)[int64_t(row) * n + nc] = cast<T>(h);
      } else {
        if (narrow)
          static_cast<T *>(output)[int64_t(row) * n + nc] = cast<T>(d[j]);
        else
          static_cast<float *>(output)[int64_t(row) * n + nc] = d[j];
      }
    }
  }
}
