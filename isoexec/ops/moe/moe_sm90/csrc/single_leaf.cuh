// Single K64 leaf. Wider output tiles amortize TMA/CTA setup while keeping
// exactly the same four ascending k16 WGMMA atoms and BF16/FP16 leaf rounding.
#pragma once
#include "tma.cuh"
#include "wgmma.cuh"

template <class T, int R, int GROUPS, bool COALESCE, bool NARROW = false>
__global__ __launch_bounds__(128) void single_leaf(
    void *out, const int *tiles, const int *total, int n,
    const __grid_constant__ CUtensorMap xm,
    const __grid_constant__ CUtensorMap wm) {
  auto *output = static_cast<float *>(out);
  auto *compact = static_cast<T *>(out);
  constexpr int N = 64 * GROUPS, A = R * 64, D = R == 16 ? 8 : 32;
  int cols = (n + N - 1) / N, tile = blockIdx.x / cols;
  if (tile >= *total)
    return;
  int first = tiles[3 * tile + 1], end = min(tiles[3 * tile + 2], first + R);
  int expert = tiles[3 * tile], col = blockIdx.x % cols * N;
  extern __shared__ __align__(1024) unsigned char storage[];
  auto *sm = reinterpret_cast<uint16_t *>(storage);
  __shared__ uint64_t bar;
  if (!threadIdx.x)
    barrier_init(&bar);
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  if (!threadIdx.x) {
    barrier_expect(&bar, (A + N * 64) * 2);
    tma2(sm, &xm, &bar, 0, first);
    tma3(sm + A, &wm, &bar, 0, col, expert);
  }
  float d[GROUPS][D];
#pragma unroll
  for (int g = 0; g < GROUPS; ++g) {
#pragma unroll
    for (int j = 0; j < D; ++j)
      d[g][j] = 0;
    fence_acc(d[g]);
  }
  barrier_wait(&bar, 0);
  __syncthreads();
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
#pragma unroll
  for (int ki = 0; ki < 64; ki += 16) {
#pragma unroll
    for (int g = 0; g < GROUPS; ++g) {
      if constexpr (R == 16)
        mma_small<T>(d[g], descriptor(sm + A + g * 4096 + ki),
                     descriptor(sm + ki), ki != 0);
      else
        mma<T>(d[g], descriptor(sm + ki), descriptor(sm + A + g * 4096 + ki),
               ki != 0);
    }
  }
  asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
  int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  if constexpr (COALESCE)
    __syncthreads();
  auto *scratch = reinterpret_cast<float *>(sm);
#pragma unroll
  for (int g = 0; g < GROUPS; ++g) {
    fence_acc(d[g]);
#pragma unroll
    for (int j = 0; j < D; ++j) {
      int row, nc;
      if constexpr (R == 16) {
        row = first + (j / 4) * 8 + (lane % 4) * 2 + j % 2;
        nc = col + g * 64 + warp * 16 + lane / 4 + (j % 4 / 2) * 8;
      } else {
        row = first + warp * 16 + lane / 4 + (j % 4 / 2) * 8;
        nc = col + g * 64 + (j / 4) * 8 + (lane % 4) * 2 + j % 2;
      }
      if constexpr (COALESCE)
        scratch[(row - first) * N + nc - col] = rounded<T>(d[g][j]);
      else if (row < end && nc < n) {
        if constexpr (NARROW)
          compact[int64_t(row) * n + nc] = cast<T>(d[g][j]);
        else
          output[int64_t(row) * n + nc] = rounded<T>(d[g][j]);
      }
    }
  }
  if constexpr (COALESCE) {
    __syncthreads();
    for (int v = threadIdx.x; v < R * N / 4; v += 128) {
      int row = first + v * 4 / N, nc = col + v * 4 % N;
      if (row < end && nc + 3 < n &&
          (NARROW ? (uintptr_t(compact + int64_t(row) * n + nc) & 7) == 0
                  : (uintptr_t(output + int64_t(row) * n + nc) & 15) == 0)) {
        float4 value = reinterpret_cast<float4 *>(scratch)[v];
        if constexpr (NARROW)
          *reinterpret_cast<uint2 *>(compact + int64_t(row) * n + nc) =
              make_uint2(pack_leaf<T>(value.x, value.y),
                         pack_leaf<T>(value.z, value.w));
        else
          *reinterpret_cast<float4 *>(output + int64_t(row) * n + nc) = value;
      } else if (row < end) {
#pragma unroll
        for (int j = 0; j < 4; ++j)
          if (nc + j < n) {
            if constexpr (NARROW)
              compact[int64_t(row) * n + nc + j] = cast<T>(scratch[v * 4 + j]);
            else
              output[int64_t(row) * n + nc + j] = scratch[v * 4 + j];
          }
      }
    }
  }
}
