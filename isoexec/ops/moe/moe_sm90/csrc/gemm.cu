// SPDX-License-Identifier: Apache-2.0
#include "chunk_fc1.cuh"
#include "gathered_fc1.cuh"
#include "leaf_pipeline.cuh"
#include "row_fc2.cuh"
#include "single_leaf.cuh"
#include "small_gemm.cuh"
#include "striped_fc2.cuh"
#include "tma.cuh"
#include "wgmma.cuh"
#include "wide_fc1.cuh"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

namespace {
constexpr int BM = 64, BN = 64, BK = 64, STAGES = 3;

template <class T, bool GLU>
__device__ __forceinline__ void
stage(T *sm, const CUtensorMap *xm, const CUtensorMap *wm, uint64_t *bar,
      int first, int col, int n, int expert, int base) {
  if (threadIdx.x % 128 == 0) {
    barrier_expect(bar, (BM + BN * (GLU ? 2 : 1)) * BK * 2);
    tma2(sm, xm, bar, base, first);
    tma3(sm + BM * BK, wm, bar, base, col, expert);
    if constexpr (GLU)
      tma3(sm + (BM + BN) * BK, wm, bar, base, col + n, expert);
  }
}

template <class T, int LEAVES, bool GLU, int LEAF_K = 0>
__global__ __launch_bounds__(128) void grouped(
    void *__restrict__ output, const int *__restrict__ tiles,
    const int *__restrict__ total, const float *__restrict__ probs, int n,
    int k, bool narrow, int row_parts, const __grid_constant__ CUtensorMap xm,
    const __grid_constant__ CUtensorMap wm) {
  int columns = (n + BN - 1) / BN;
  int tile = blockIdx.x / (columns * row_parts);
  if (tile >= *total)
    return;
  int expert = tiles[tile * 3], first = tiles[tile * 3 + 1],
      end = tiles[tile * 3 + 2];
  first += (blockIdx.x / columns % row_parts) * 64;
  end = min(end, first + 64);
  if (first >= end)
    return;
  int col = (blockIdx.x % columns) * BN;
  constexpr int SS = (BM + BN * (GLU ? 2 : 1)) * BK;
  extern __shared__ __align__(1024) unsigned char storage[];
  T *sm = reinterpret_cast<T *>(storage);
  __shared__ uint64_t barriers[STAGES];
  uint64_t *bar = barriers;
  if (threadIdx.x < STAGES)
    barrier_init(barriers + threadIdx.x);
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  constexpr int LEVELS = LEAVES == 8   ? 3
                         : LEAVES == 4 ? 2
                         : LEAVES == 2 ? 1
                                       : 0;
  uint32_t leaf_pair[16];
  float pending[LEVELS > 1 ? LEVELS - 1 : 1][32];
  float d[32], u[32];
  const int leaf_k = LEAF_K ? LEAF_K : k / LEAVES;
  const int blocks = (leaf_k + BK - 1) / BK;
  const int iterations = blocks * LEAVES;
#pragma unroll
  for (int s = 0; s < 1; ++s) {
    int leaf = s / blocks, kb = s % blocks;
    if (s < iterations)
      stage<T, GLU>(sm + s * SS, &xm, &wm, bar + s, first, col, n, expert,
                    leaf * leaf_k + kb * BK);
  }
#pragma unroll
  for (int leaf = 0; leaf < LEAVES; ++leaf) {
#pragma unroll
    for (int j = 0; j < 32; ++j) {
      d[j] = 0;
      if constexpr (GLU)
        u[j] = 0;
    }
    fence_acc(d);
    if constexpr (GLU)
      fence_acc(u);
    for (int kb = 0; kb < blocks; ++kb) {
      int iter = leaf * blocks + kb;
      // Three buffers permit one outstanding WGMMA group while TMA fills the
      // next tile: buffer iter-2 is free after wait_group 1 on iter-1.
      int next = iter + 1;
      int nl = next / blocks, nk = next % blocks;
      if (next < iterations)
        stage<T, GLU>(sm + (next % STAGES) * SS, &xm, &wm, bar + next % STAGES,
                      first, col, n, expert, nl * leaf_k + nk * BK);
      barrier_wait(bar + iter % STAGES, (iter / STAGES) & 1);
      __syncthreads();
      T *a = sm + (iter % STAGES) * SS;
      T *b = a + BM * BK;
      if ((leaf_k & 15) && kb == blocks - 1) {
        // TMA knows tensor bounds, not logical leaf bounds. Clear the eight
        // lanes in a partial final k16 atom before they can reach the dot.
        int kk = leaf_k % BK;
        for (int r = threadIdx.x % 128; r < BM + BN * (GLU ? 2 : 1); r += 128)
          *reinterpret_cast<uint4 *>(a + r * BK + (kk ^ ((r & 7) * 8))) =
              make_uint4(0, 0, 0, 0);
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncthreads();
      }
      if (kb == 0)
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
#pragma unroll
      for (int ki = 0; ki < BK; ki += 16) {
        if (kb * BK + ki < leaf_k) {
          if constexpr (GLU)
            mma_glu<T>(d, u, descriptor(a + ki), descriptor(b + ki),
                       kb != 0 || ki != 0);
          else
            mma<T>(d, descriptor(a + ki), descriptor(b + ki),
                   kb != 0 || ki != 0);
        }
      }
      asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
      if (kb == blocks - 1) {
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
        fence_acc(d);
        if constexpr (GLU)
          fence_acc(u);
      } else {
        asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
      }
      __syncthreads();
    }
    if constexpr (!GLU) {
      if (LEAVES > 1 && (leaf & 1) == 0) {
#pragma unroll
        for (int j = 0; j < 16; ++j)
          leaf_pair[j] = pack_leaf<T>(d[j * 2], d[j * 2 + 1]);
      } else {
#pragma unroll
        for (int j = 0; j < 32; ++j) {
          float v = rounded<T>(d[j]);
          if constexpr (LEAVES > 1)
            v = __fadd_rn(unpack_leaf<T>(leaf_pair[j / 2], j % 2), v);
#pragma unroll
          for (int level = 1; level < LEVELS; ++level)
            if ((leaf & ((1 << (level + 1)) - 1)) == ((1 << (level + 1)) - 1))
              v = __fadd_rn(pending[level - 1][j], v);
          if (leaf == LEAVES - 1)
            d[j] = v;
          else {
#pragma unroll
            for (int level = 1; level < LEVELS; ++level)
              if ((leaf & ((1 << (level + 1)) - 1)) == ((1 << level) - 1))
                pending[level - 1][j] = v;
          }
        }
      }
    }
  }
  // PTX accumulator layout: each lane owns two columns in each 8-column group,
  // and two rows (separated by eight) in its warp's sixteen-row stripe.
  int lane = threadIdx.x % 32, warp = threadIdx.x % 128 / 32;

  __syncthreads();
#pragma unroll
  for (int j = 0; j < 32; ++j) {
    int row = first + warp * 16 + lane / 4 + (j % 4 / 2) * 8;
    int nc = col + (j / 4) * 8 + (lane % 4) * 2 + j % 2;
    if (row < end && nc < n) {
      if constexpr (GLU) {
        float g = rounded<T>(d[j]), up = rounded<T>(u[j]);
        float act = rounded<T>(__fdividef(g, __fadd_rn(1.f, expf(-g))));
        float h = rounded<T>(__fmul_rn(act, up));
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

template <class T, bool GLU, int LEAF_K = 0>
void launch(const at::Tensor &x, const at::Tensor &w, const at::Tensor &tiles,
            const at::Tensor &total, const at::Tensor &p, at::Tensor &out,
            int leaves) {
  dim3 grid(tiles.size(0) * ((out.size(1) + BN - 1) / BN));
  const auto stream = at::cuda::getCurrentCUDAStream();
  CUtensorMap xm, wm;
  uint64_t xd[2] = {uint64_t(x.size(1)), uint64_t(x.size(0))};
  uint64_t xs[1] = {uint64_t(x.stride(0) * 2)};
  bool small = x.size(0) <= w.size(0) * 16;
  bool wide = (GLU || out.scalar_type() != at::kFloat || LEAF_K > 0) &&
              x.size(0) > w.size(0) * 64;
  bool striped =
      !GLU && out.scalar_type() == at::kFloat && LEAF_K == 96 && !small;
  uint32_t stage_k = striped ? 32 : BK;
  uint32_t xb[2] = {stage_k, uint32_t(small  ? 16
                                      : wide ? 128
                                             : BM)},
           xe[2] = {1, 1};
  uint64_t wd[3] = {uint64_t(w.size(2)), uint64_t(w.size(1)),
                    uint64_t(w.size(0))};
  uint64_t ws[2] = {uint64_t(w.stride(1) * 2), uint64_t(w.stride(0) * 2)};
  uint32_t wb[3] = {stage_k,
                    uint32_t(wide && (GLU || out.scalar_type() != at::kFloat)
                                 ? (GLU ? 64 : 128)
                                 : BN),
                    1},
           we[3] = {1, 1, 1};
  auto encode = [striped](CUtensorMap *map, int rank, void *ptr, uint64_t *dims,
                          uint64_t *strides, uint32_t *box, uint32_t *element) {
    auto status = cuTensorMapEncodeTiled(
        map, CU_TENSOR_MAP_DATA_TYPE_UINT16, rank, ptr, dims, strides, box,
        element, CU_TENSOR_MAP_INTERLEAVE_NONE,
        striped ? CU_TENSOR_MAP_SWIZZLE_64B : CU_TENSOR_MAP_SWIZZLE_128B,
        striped ? CU_TENSOR_MAP_L2_PROMOTION_L2_128B
                : CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(status == CUDA_SUCCESS,
                "TMA descriptor rejected layout: ", int(status));
  };
  encode(&xm, 2, x.data_ptr(), xd, xs, xb, xe);
  encode(&wm, 3, w.data_ptr(), wd, ws, wb, we);
  if constexpr (GLU) {
    if (small && x.size(1) == 2048 && out.size(1) == 64 && x.size(0) <= 256) {
#define CHUNK(C)                                                               \
  do {                                                                         \
    auto kernel = chunk_fc1<T, C>;                                             \
    constexpr int BYTES = 3 * C * 9216 * 2;                                    \
    C10_CUDA_CHECK(cudaFuncSetAttribute(                                       \
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, BYTES));          \
    kernel<<<tiles.size(0), 128, BYTES, stream>>>(                             \
        static_cast<T *>(out.data_ptr()), tiles.data_ptr<int>(),               \
        total.data_ptr<int>(), p.numel() ? p.data_ptr<float>() : nullptr, xm,  \
        wm);                                                                   \
  } while (0)
      if (x.size(0) <= 64) {
        CHUNK(4);
      } else {
        CHUNK(2);
      }
#undef CHUNK
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      return;
    }
  }
  if (striped) {
#define STRIPED(L, R, N)                                                       \
  do {                                                                         \
    constexpr int BYTES = 2 * 3 * (R + N) * 32 * 2 + (L == 8 ? R * N * 4 : 0); \
    auto kernel = striped_fc2<T, L, R, N>;                                     \
    C10_CUDA_CHECK(cudaFuncSetAttribute(                                       \
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, BYTES));          \
    kernel<<<dim3(tiles.size(0) * ((out.size(1) + N - 1) / N)), R * 2, BYTES,  \
             stream>>>(out.data_ptr(), tiles.data_ptr<int>(),                  \
                       total.data_ptr<int>(), nullptr, out.size(1), x.size(1), \
                       false, xm, wm);                                         \
  } while (0)
#define STRIPED_L(L)                                                           \
  do {                                                                         \
    if (wide) {                                                                \
      STRIPED(L, 128, 64);                                                     \
    } else {                                                                   \
      STRIPED(L, 64, 64);                                                      \
    }                                                                          \
  } while (0)
    switch (leaves) {
    case 1:
      STRIPED_L(1);
      break;
    case 2:
      STRIPED_L(2);
      break;
    case 4:
      STRIPED_L(4);
      break;
    case 8:
      STRIPED_L(8);
      break;
    }
#undef STRIPED_L
#undef STRIPED
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  if (small) {
#define SMALL(L)                                                               \
  do {                                                                         \
    constexpr int BYTES = (!GLU && L == 1 && LEAF_K == 64 ? 1 : 3) *           \
                          (1024 + 4096 * (GLU ? 2 : 1)) * 2;                   \
    auto kernel = small_gemm<T, L, GLU, LEAF_K>;                               \
    C10_CUDA_CHECK(cudaFuncSetAttribute(                                       \
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, BYTES));          \
    kernel<<<grid, 128, BYTES, stream>>>(                                      \
        out.data_ptr(), tiles.data_ptr<int>(), total.data_ptr<int>(),          \
        p.numel() ? p.data_ptr<float>() : nullptr, out.size(1), x.size(1),     \
        out.scalar_type() != at::kFloat, xm, wm);                              \
  } while (0)
    if constexpr (GLU) {
      SMALL(1);
    } else
      switch (leaves) {
      case 1:
        SMALL(1);
        break;
      case 2:
        SMALL(2);
        break;
      case 4:
        SMALL(4);
        break;
      case 8:
        SMALL(8);
        break;
      }
#undef SMALL
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  if (wide && (GLU || out.scalar_type() != at::kFloat)) {
    auto kernel = wide_fc1<T, GLU, 128>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304));
    dim3 wide_grid(tiles.size(0) *
                   ((out.size(1) + (GLU ? 64 : 128) - 1) / (GLU ? 64 : 128)));
    kernel<<<wide_grid, 256, 98304, stream>>>(
        out.data_ptr(), tiles.data_ptr<int>(), total.data_ptr<int>(),
        p.numel() ? p.data_ptr<float>() : nullptr, out.size(1), x.size(1), true,
        xm, wm);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  if constexpr (!GLU && LEAF_K > 0) {
    if (wide) {
      dim3 wide_grid(tiles.size(0) * ((out.size(1) + 63) / 64));
#define WIDE(L)                                                                \
  do {                                                                         \
    constexpr int BYTES = 73728 + (L == 8 ? 32768 : 0);                        \
    auto kernel = row_fc2<T, L, LEAF_K>;                                       \
    C10_CUDA_CHECK(cudaFuncSetAttribute(                                       \
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, BYTES));          \
    kernel<<<wide_grid, 256, BYTES, stream>>>(                                 \
        out.data_ptr(), tiles.data_ptr<int>(), total.data_ptr<int>(), nullptr, \
        out.size(1), x.size(1), false, xm, wm);                                \
  } while (0)
      switch (leaves) {
      case 1:
        WIDE(1);
        break;
      case 2:
        WIDE(2);
        break;
      case 4:
        WIDE(4);
        break;
      case 8:
        WIDE(8);
        break;
      }
#undef WIDE
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      return;
    }
  }
  int row_parts = x.size(0) > w.size(0) * 64 ? 2 : 1;
  grid.x *= row_parts;
#define RUN(L)                                                                 \
  do {                                                                         \
    constexpr bool FULL_LEAF = !GLU && LEAF_K > 0;                             \
    constexpr int SHMEM =                                                      \
        FULL_LEAF ? (L == 1 ? 1 : 2) * ((LEAF_K + 63) / 64) * 8192 * 2 +       \
                        (L == 8 && LEAF_K == 192 ? 16384 : 0)                  \
                  : STAGES * (BM + BN * (GLU ? 2 : 1)) * BK * 2;               \
    auto kernel = [] {                                                         \
      if constexpr (FULL_LEAF)                                                 \
        return leaf_pipeline<T, L, LEAF_K>;                                    \
      else                                                                     \
        return grouped<T, L, GLU, LEAF_K>;                                     \
    }();                                                                       \
    C10_CUDA_CHECK(cudaFuncSetAttribute(                                       \
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM));          \
    kernel<<<grid, 128, SHMEM, stream>>>(                                      \
        out.data_ptr(), tiles.data_ptr<int>(), total.data_ptr<int>(),          \
        p.numel() ? p.data_ptr<float>() : nullptr, out.size(1), x.size(1),     \
        out.scalar_type() != at::kFloat, row_parts, xm, wm);                   \
  } while (0)
  if constexpr (GLU) {
    RUN(1);
  } else
    switch (leaves) {
    case 1:
      RUN(1);
      break;
    case 2:
      RUN(2);
      break;
    case 4:
      RUN(4);
      break;
    case 8:
      RUN(8);
      break;
    }
#undef RUN
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <class T, int R, int CH, class I>
void launch_gathered(const at::Tensor &x, const at::Tensor &w,
                     const at::Tensor &tiles, const at::Tensor &total,
                     const at::Tensor &sorted, const at::Tensor &ids,
                     const at::Tensor &probs, at::Tensor &out,
                     at::Tensor &rows) {
  CUtensorMap wm;
  uint64_t wd[3] = {2048, 128, uint64_t(w.size(0))},
           ws[2] = {uint64_t(w.stride(1) * 2), uint64_t(w.stride(0) * 2)};
  uint32_t wb[3] = {64, 64, 1}, we[3] = {1, 1, 1};
  TORCH_CHECK(cuTensorMapEncodeTiled(
                  &wm, CU_TENSOR_MAP_DATA_TYPE_UINT16, 3, w.data_ptr(), wd, ws,
                  wb, we, CU_TENSOR_MAP_INTERLEAVE_NONE,
                  CU_TENSOR_MAP_SWIZZLE_128B,
                  CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                  CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS);
  auto kernel = gathered_fc1<T, R, CH, I>;
  constexpr int BYTES = 3 * CH * (R + 128) * 64 * 2;
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, BYTES));
  kernel<<<tiles.size(0), 128, BYTES, at::cuda::getCurrentCUDAStream()>>>(
      static_cast<const uint16_t *>(x.data_ptr()), x.stride(0),
      sorted.data_ptr<int64_t>(), ids.data_ptr<I>(), ids.size(1),
      static_cast<T *>(out.data_ptr()),
      rows.numel() ? rows.data_ptr<int>() : nullptr, tiles.data_ptr<int>(),
      total.data_ptr<int>(), probs.numel() ? probs.data_ptr<float>() : nullptr, wm);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <class T, int R, int G, bool COALESCE, bool NARROW = false>
void launch_single(const at::Tensor &x, const at::Tensor &w,
                   const at::Tensor &tiles, const at::Tensor &total,
                   at::Tensor &out) {
  CUtensorMap xm, wm;
  uint64_t xd[2] = {64, uint64_t(x.size(0))},
           xs[1] = {uint64_t(x.stride(0) * 2)};
  uint64_t wd[3] = {64, uint64_t(w.size(1)), uint64_t(w.size(0))};
  uint64_t ws[2] = {uint64_t(w.stride(1) * 2), uint64_t(w.stride(0) * 2)};
  uint32_t xb[2] = {64, R}, xe[2] = {1, 1}, wb[3] = {64, 64 * G, 1},
           we[3] = {1, 1, 1};
  TORCH_CHECK(cuTensorMapEncodeTiled(
                  &xm, CU_TENSOR_MAP_DATA_TYPE_UINT16, 2, x.data_ptr(), xd, xs,
                  xb, xe, CU_TENSOR_MAP_INTERLEAVE_NONE,
                  CU_TENSOR_MAP_SWIZZLE_128B,
                  CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                  CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS);
  TORCH_CHECK(cuTensorMapEncodeTiled(
                  &wm, CU_TENSOR_MAP_DATA_TYPE_UINT16, 3, w.data_ptr(), wd, ws,
                  wb, we, CU_TENSOR_MAP_INTERLEAVE_NONE,
                  CU_TENSOR_MAP_SWIZZLE_128B,
                  CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                  CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS);
  constexpr int INPUT_BYTES = (R + 64 * G) * 64 * 2;
  constexpr int OUTPUT_BYTES = R * 64 * G * 4;
  constexpr int BYTES =
      COALESCE && OUTPUT_BYTES > INPUT_BYTES ? OUTPUT_BYTES : INPUT_BYTES;
  auto kernel = single_leaf<T, R, G, COALESCE, NARROW>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, BYTES));
  kernel<<<tiles.size(0) * ((out.size(1) + 64 * G - 1) / (64 * G)), 128, BYTES,
           at::cuda::getCurrentCUDAStream()>>>(
      out.data_ptr(), tiles.data_ptr<int>(), total.data_ptr<int>(), out.size(1),
      xm, wm);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <class T>
void launch_fc2(const at::Tensor &x, const at::Tensor &w,
                const at::Tensor &tiles, const at::Tensor &total,
                const at::Tensor &p, at::Tensor &out, int leaves) {
  // Single K64 root: no leaf pipeline is needed. Wider output tiles amortize
  // CTA setup; cooperative stores matter when a tile has many token rows.
  if (leaves == 1 && x.size(1) == 64 && x.size(0) <= w.size(0) * 64) {
    if (out.scalar_type() == at::kFloat) {
      if (x.size(0) <= w.size(0) * 16)
        launch_single<T, 16, 2, false, false>(x, w, tiles, total, out);
      else
        launch_single<T, 64, 2, true, false>(x, w, tiles, total, out);
    } else {
      if (x.size(0) <= w.size(0) * 16)
        launch_single<T, 16, 2, false, true>(x, w, tiles, total, out);
      else
        launch_single<T, 64, 2, true, true>(x, w, tiles, total, out);
    }
    return;
  }
  switch (x.size(1) / leaves) {
  case 64:
    launch<T, false, 64>(x, w, tiles, total, p, out, leaves);
    break;
  case 96:
    launch<T, false, 96>(x, w, tiles, total, p, out, leaves);
    break;
  case 128:
    launch<T, false, 128>(x, w, tiles, total, p, out, leaves);
    break;
  case 192:
    launch<T, false, 192>(x, w, tiles, total, p, out, leaves);
    break;
  default:
    launch<T, false, 0>(x, w, tiles, total, p, out, leaves);
  }
}
} // namespace

void gemm_cuda(const at::Tensor &x, const at::Tensor &w,
               const at::Tensor &tiles, const at::Tensor &total,
               const at::Tensor &probs, at::Tensor out, int64_t leaves,
               bool glu) {
  const c10::cuda::CUDAGuard guard(x.device());
  if (!x.size(0) || !out.size(1))
    return;
  // Autograd may enter on a fresh host thread. PyTorch's device guard can
  // defer context activation, while driver-side TMA encoding requires one.
  CUcontext context;
  TORCH_CHECK(cuCtxGetCurrent(&context) == CUDA_SUCCESS);
  if (!context)
    C10_CUDA_CHECK(cudaSetDevice(x.get_device()));
  if (x.scalar_type() == at::kBFloat16) {
    if (glu)
      launch<__nv_bfloat16, true>(x, w, tiles, total, probs, out, 1);
    else
      launch_fc2<__nv_bfloat16>(x, w, tiles, total, probs, out, leaves);
  } else {
    if (glu)
      launch<__half, true>(x, w, tiles, total, probs, out, 1);
    else
      launch_fc2<__half>(x, w, tiles, total, probs, out, leaves);
  }
}

void gathered_fc1_cuda(const at::Tensor &x, const at::Tensor &w,
                       const at::Tensor &tiles, const at::Tensor &total,
                       const at::Tensor &sorted, const at::Tensor &ids,
                       const at::Tensor &probs, at::Tensor out,
                       at::Tensor rows) {
  const c10::cuda::CUDAGuard guard(x.device());
  C10_CUDA_CHECK(cudaSetDevice(x.get_device()));
  int chunks = out.size(0) <= 64 ? 4 : out.size(0) <= 256 ? 2 : 1;
#define GFC1(T, I)                                                             \
  do {                                                                         \
    if (out.size(0) <= w.size(0) * 16) {                                       \
      if (chunks == 4)                                                         \
        launch_gathered<T, 16, 4, I>(x, w, tiles, total, sorted, ids, probs,   \
                                     out, rows);                               \
      else if (chunks == 2)                                                    \
        launch_gathered<T, 16, 2, I>(x, w, tiles, total, sorted, ids, probs,   \
                                     out, rows);                               \
      else                                                                     \
        launch_gathered<T, 16, 1, I>(x, w, tiles, total, sorted, ids, probs,   \
                                     out, rows);                               \
    } else {                                                                   \
      launch_gathered<T, 64, 1, I>(x, w, tiles, total, sorted, ids, probs,     \
                                   out, rows);                                 \
    }                                                                          \
  } while (0)
  if (x.scalar_type() == at::kBFloat16) {
    if (ids.scalar_type() == at::kLong) {
      GFC1(__nv_bfloat16, int64_t);
    } else {
      GFC1(__nv_bfloat16, int);
    }
  } else {
    if (ids.scalar_type() == at::kLong) {
      GFC1(__half, int64_t);
    } else {
      GFC1(__half, int);
    }
  }
#undef GFC1
}
