// Gather routed token rows into shared memory while staging expert weights.
// This avoids materializing the expanded activation matrix in global memory.
#pragma once
#include "tma.cuh"
#include "wgmma.cuh"

template <int R, int CH>
__device__ __forceinline__ void
gathered_stage(uint16_t *sm, const uint16_t *x, const int (&tokens)[R / 16],
               int stride, const CUtensorMap *wm, uint64_t *bar, int expert,
               int base) {
  constexpr int A = R * 64, SS = (R + 128) * 64;
  if (!threadIdx.x) {
    barrier_expect(bar, CH * 8192 * 2);
#pragma unroll
    for (int c = 0; c < CH; ++c) {
      tma3(sm + c * SS + A, wm, bar, base + c * 64, 0, expert);
      tma3(sm + c * SS + A + 4096, wm, bar, base + c * 64, 64, expert);
    }
  }
  int kk = threadIdx.x % 8 * 8;
#pragma unroll
  for (int c = 0; c < CH; ++c) {
#pragma unroll
    for (int j = 0; j < R / 16; ++j) {
      int row = threadIdx.x / 8 + j * 16, token = tokens[j];
      copy16(sm + c * SS + row * 64 + (kk ^ ((row & 7) * 8)),
             x + int64_t(token < 0 ? 0 : token) * stride + base + c * 64 + kk,
             token < 0 ? 0 : 16);
    }
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
}

template <class T, int R, int CH, class I>
__global__ __launch_bounds__(128) void gathered_fc1(
    const uint16_t *x, int stride, const int64_t *sorted, const I *ids,
    int topk, T *output, int *combine_rows, const int *tiles, const int *total,
    const float *probs, const __grid_constant__ CUtensorMap wm) {
  constexpr int A = R * 64, SS = (R + 128) * 64, ST = SS * CH,
                BLOCKS = 2048 / (64 * CH), D = R == 16 ? 8 : 32;
  int tile = blockIdx.x;
  if (tile >= *total)
    return;
  int first = tiles[3 * tile + 1], end = min(tiles[3 * tile + 2], first + R),
      expert = tiles[3 * tile];
  int tokens[R / 16];
#pragma unroll
  for (int j = 0; j < R / 16; ++j) {
    int row = first + threadIdx.x / 8 + j * 16;
    tokens[j] = row < end ? int(sorted[row]) : -1;
  }
  if (combine_rows && threadIdx.x < R && first + threadIdx.x < end) {
    int row = first + threadIdx.x, token = sorted[row], slot = 0;
    for (int j = 0; j < topk; ++j)
      slot += ids[int64_t(token) * topk + j] < expert;
    combine_rows[int64_t(token) * topk + slot] = row;
  }
  extern __shared__ __align__(1024) unsigned char storage[];
  auto *sm = reinterpret_cast<uint16_t *>(storage);
  __shared__ uint64_t bar[3];
  if (threadIdx.x < 3)
    barrier_init(bar + threadIdx.x);
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  gathered_stage<R, CH>(sm, x, tokens, stride, &wm, bar, expert, 0);
  float d[D] = {}, u[D] = {};
  fence_acc(d);
  fence_acc(u);
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
  for (int b = 0; b < BLOCKS; ++b) {
    if (b + 1 < BLOCKS)
      gathered_stage<R, CH>(sm + (b + 1) % 3 * ST, x, tokens, stride, &wm,
                            bar + (b + 1) % 3, expert, (b + 1) * CH * 64);
    barrier_wait(bar + b % 3, b / 3 & 1);
    if (b + 1 < BLOCKS)
      asm volatile("cp.async.wait_group 1;" ::: "memory");
    else
      asm volatile("cp.async.wait_group 0;" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
#pragma unroll
    for (int c = 0; c < CH; ++c) {
      auto *a = sm + b % 3 * ST + c * SS;
#pragma unroll
      for (int ki = 0; ki < 64; ki += 16) {
        bool accum = b != 0 || c != 0 || ki != 0;
        if constexpr (R == 16) {
          mma_small<T>(d, descriptor(a + A + ki), descriptor(a + ki), accum);
          mma_small<T>(u, descriptor(a + A + 4096 + ki), descriptor(a + ki),
                       accum);
        } else
          mma_glu<T>(d, u, descriptor(a + ki), descriptor(a + A + ki), accum);
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
  for (int j = 0; j < D; ++j) {
    int row, col;
    if constexpr (R == 16) {
      row = first + (j / 4) * 8 + (lane % 4) * 2 + j % 2;
      col = warp * 16 + lane / 4 + (j % 4 / 2) * 8;
    } else {
      row = first + warp * 16 + lane / 4 + (j % 4 / 2) * 8;
      col = (j / 4) * 8 + (lane % 4) * 2 + j % 2;
    }
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
