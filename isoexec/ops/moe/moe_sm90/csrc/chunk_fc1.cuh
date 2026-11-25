// Stage several K64 tiles per transfer group for H2048/I64 decode.
// Grouping transfers leaves the ascending k16 accumulation order unchanged.
#pragma once
#include "tma.cuh"
#include "wgmma.cuh"

template <int CH>
__device__ __forceinline__ void
chunk_stage(uint16_t *sm, const CUtensorMap *xm, const CUtensorMap *wm,
            uint64_t *bar, int first, int expert, int base) {
  if (!threadIdx.x) {
    barrier_expect(bar, CH * 9216 * 2);
#pragma unroll
    for (int c = 0; c < CH; ++c) {
      auto *a = sm + c * 9216;
      tma2(a, xm, bar, base + c * 64, first);
      tma3(a + 1024, wm, bar, base + c * 64, 0, expert);
      tma3(a + 5120, wm, bar, base + c * 64, 64, expert);
    }
  }
}

template <class T, int CH>
__global__
__launch_bounds__(128) void chunk_fc1(T *output, const int *tiles,
                                      const int *total, const float *probs,
                                      const __grid_constant__ CUtensorMap xm,
                                      const __grid_constant__ CUtensorMap wm) {
  // Each chunk holds A[16,64], gate[64,64], and up[64,64] in 16-bit elements.
  constexpr int SS = 9216 * CH, BLOCKS = 2048 / (64 * CH);
  int tile = blockIdx.x;
  if (tile >= *total)
    return;
  int first = tiles[3 * tile + 1], end = min(tiles[3 * tile + 2], first + 16),
      expert = tiles[3 * tile];
  extern __shared__ __align__(1024) unsigned char storage[];
  auto *sm = reinterpret_cast<uint16_t *>(storage);
  __shared__ uint64_t bar[3];
  if (threadIdx.x < 3)
    barrier_init(bar + threadIdx.x);
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  chunk_stage<CH>(sm, &xm, &wm, bar, first, expert, 0);
  float d[8] = {}, u[8] = {};
  fence_acc(d);
  fence_acc(u);
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
  for (int b = 0; b < BLOCKS; ++b) {
    if (b + 1 < BLOCKS)
      chunk_stage<CH>(sm + (b + 1) % 3 * SS, &xm, &wm, bar + (b + 1) % 3, first,
                      expert, (b + 1) * 64 * CH);
    barrier_wait(bar + b % 3, b / 3 & 1);
    __syncthreads();
#pragma unroll
    for (int c = 0; c < CH; ++c) {
      auto *a = sm + b % 3 * SS + c * 9216;
#pragma unroll
      for (int ki = 0; ki < 64; ki += 16) {
        mma_small<T>(d, descriptor(a + 1024 + ki), descriptor(a + ki),
                     b != 0 || c != 0 || ki != 0);
        mma_small<T>(u, descriptor(a + 5120 + ki), descriptor(a + ki),
                     b != 0 || c != 0 || ki != 0);
      }
    }
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
    if (b == BLOCKS - 1)
      asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    else
      asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
    __syncthreads();
  }
  fence_acc(d);
  fence_acc(u);
  int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    int row = first + (j / 4) * 8 + (lane % 4) * 2 + j % 2;
    int col = warp * 16 + lane / 4 + (j % 4 / 2) * 8;
    if (row < end) {
      float g = rounded<T>(d[j]), up = rounded<T>(u[j]);
      float h = rounded<T>(__fdividef(g, __fadd_rn(1.f, expf(-g))));
      h = rounded<T>(__fmul_rn(h, up));
      if (probs)
        h = rounded<T>(__fmul_rn(h, probs[row]));
      output[int64_t(row) * 64 + col] = cast<T>(h);
    }
  }
}
