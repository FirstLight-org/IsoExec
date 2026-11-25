// SPDX-License-Identifier: Apache-2.0
#include "wgmma.cuh"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <assert.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cub/block/block_scan.cuh>

#include "compact_combine.cuh"
#include "counts_plan.cuh"
#include "small_routes.cuh"

namespace {
__global__ void plan_kernel(const int *offsets, int *tiles, int *total,
                            int experts, int rows) {
  using Scan = cub::BlockScan<int, 256>;
  __shared__ Scan::TempStorage tmp;
  __shared__ int carry;
  if (!threadIdx.x)
    carry = 0;
  __syncthreads();
  for (int base = 0; base < experts; base += 256) {
    int e = base + threadIdx.x;
    int start = e < experts ? offsets[e] : 0;
    int end = e < experts ? offsets[e + 1] : 0;
    assert(start >= 0 && end >= start && end <= rows);
    int bm = rows <= int64_t(experts) * 16   ? 16
             : rows <= int64_t(experts) * 64 ? 64
                                             : 128;
    int count = (end - start + bm - 1) / bm, prefix, aggregate;
    Scan(tmp).ExclusiveSum(count, prefix, aggregate);
    int tile = carry + prefix;
    for (int j = 0; j < count; ++j) {
      tiles[(tile + j) * 3] = e;
      tiles[(tile + j) * 3 + 1] = start + j * bm;
      tiles[(tile + j) * 3 + 2] = end;
    }
    __syncthreads();
    if (!threadIdx.x)
      carry += aggregate;
    __syncthreads();
  }
  if (!threadIdx.x) {
    assert(offsets[0] == 0 && offsets[experts] == rows);
    *total = carry;
  }
}

__global__ void histogram(const int *ids, int *counts, int routes,
                          int experts) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < routes;
       i += gridDim.x * blockDim.x) {
    int e = ids[i];
    assert(e >= 0 && e < experts);
    // Warp aggregation avoids 32 atomics when a decode batch chooses the same
    // expert.
    unsigned peers = __match_any_sync(__activemask(), e);
    int leader = __ffs(peers) - 1;
    if ((threadIdx.x & 31) == leader)
      atomicAdd(counts + e, __popc(peers));
  }
}

__global__ void prefix_kernel(const int *counts, int *offsets, int experts) {
  using Scan = cub::BlockScan<int, 256>;
  __shared__ Scan::TempStorage tmp;
  __shared__ int carry;
  if (!threadIdx.x)
    carry = 0;
  __syncthreads();
  for (int base = 0; base < experts; base += 256) {
    int e = base + threadIdx.x, prefix, aggregate;
    Scan(tmp).ExclusiveSum(e < experts ? counts[e] : 0, prefix, aggregate);
    if (e < experts)
      offsets[e] = carry + prefix;
    __syncthreads();
    if (!threadIdx.x)
      carry += aggregate;
    __syncthreads();
  }
  if (!threadIdx.x)
    offsets[experts] = carry;
}

__global__ void scatter_map(const int *ids, const int *offsets, int *cursors,
                            int *inverse, int *restore, int routes) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < routes;
       i += gridDim.x * blockDim.x) {
    int e = ids[i];
    unsigned peers = __match_any_sync(__activemask(), e);
    int leader = __ffs(peers) - 1, base = 0;
    if ((threadIdx.x & 31) == leader)
      base = atomicAdd(cursors + e, __popc(peers));
    base = __shfl_sync(peers, base, leader);
    unsigned lower = (1u << (threadIdx.x & 31)) - 1;
    int row = offsets[e] + base + __popc(peers & lower);
    inverse[row] = i;
    restore[i] = row;
  }
}

template <class T>
__global__ void gather_kernel(const T *x, const float *p, const int *inverse,
                              T *out, float *op, int routes, int hidden,
                              int topk) {
  int row = blockIdx.x;
  int route = inverse[row];
  for (int i = threadIdx.x; i < hidden / 8; i += blockDim.x)
    reinterpret_cast<uint4 *>(out + int64_t(row) * hidden)[i] =
        reinterpret_cast<const uint4 *>(x + int64_t(route / topk) * hidden)[i];
  if (!threadIdx.x)
    op[row] = p[route];
}

template <class T>
__global__ void combine_kernel(const float *rows, const int *ids,
                               const int *restore, T *out, int hidden,
                               int topk) {
  int token = blockIdx.x, col = blockIdx.y * 256 + threadIdx.x;
  __shared__ int order[32];
  if (threadIdx.x < topk) {
    int slot = threadIdx.x, e = ids[token * topk + slot], rank = 0;
    for (int j = 0; j < topk; ++j) {
      int other = ids[token * topk + j];
      rank += other < e || (other == e && j < slot);
    }
    order[rank] = restore[token * topk + slot];
  }
  __syncthreads();
  if (col < hidden) {
    // Start with the first row, preserving -0 for the top-k=1 case.
    float v = rows[int64_t(order[0]) * hidden + col];
    for (int j = 1; j < topk; ++j)
      v = __fadd_rn(v, rows[int64_t(order[j]) * hidden + col]);
    out[int64_t(token) * hidden + col] = cast<T>(v);
  }
}
} // namespace

void plan_cuda(const at::Tensor &offsets, at::Tensor tiles, at::Tensor total,
               int64_t rows) {
  const c10::cuda::CUDAGuard guard(offsets.device());
  plan_kernel<<<1, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      offsets.data_ptr<int>(), tiles.data_ptr<int>(), total.data_ptr<int>(),
      offsets.numel() - 1, rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void route_cuda(const at::Tensor &ids, at::Tensor counts, at::Tensor offsets,
                at::Tensor cursors, at::Tensor inverse, at::Tensor restore) {
  const c10::cuda::CUDAGuard guard(ids.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(
      cudaMemsetAsync(counts.data_ptr(), 0, counts.nbytes(), stream));
  C10_CUDA_CHECK(
      cudaMemsetAsync(cursors.data_ptr(), 0, cursors.nbytes(), stream));
  int routes = ids.numel(), experts = counts.numel(),
      blocks = std::min(256, (routes + 255) / 256);
  if (routes)
    histogram<<<blocks, 256, 0, stream>>>(
        ids.data_ptr<int>(), counts.data_ptr<int>(), routes, experts);
  prefix_kernel<<<1, 256, 0, stream>>>(counts.data_ptr<int>(),
                                       offsets.data_ptr<int>(), experts);
  if (routes)
    scatter_map<<<blocks, 256, 0, stream>>>(
        ids.data_ptr<int>(), offsets.data_ptr<int>(), cursors.data_ptr<int>(),
        inverse.data_ptr<int>(), restore.data_ptr<int>(), routes);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gather_cuda(const at::Tensor &x, const at::Tensor &p,
                 const at::Tensor &inverse, at::Tensor out, at::Tensor op,
                 int64_t topk) {
  const c10::cuda::CUDAGuard guard(x.device());
  if (!out.size(0))
    return;
#define GATHER(T)                                                              \
  gather_kernel<T><<<out.size(0), 128, 0, at::cuda::getCurrentCUDAStream()>>>( \
      reinterpret_cast<const T *>(x.data_ptr()), p.data_ptr<float>(),          \
      inverse.data_ptr<int>(), reinterpret_cast<T *>(out.data_ptr()),          \
      op.data_ptr<float>(), out.size(0), x.size(1), topk)
  if (x.scalar_type() == at::kBFloat16) {
    GATHER(__nv_bfloat16);
  } else {
    GATHER(__half);
  }
#undef GATHER
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void combine_cuda(const at::Tensor &rows, const at::Tensor &ids,
                  const at::Tensor &restore, at::Tensor out) {
  const c10::cuda::CUDAGuard guard(rows.device());
  if (!out.numel())
    return;
  dim3 grid(out.size(0), (out.size(1) + 255) / 256);
#define COMBINE(T)                                                             \
  combine_kernel<T><<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(       \
      rows.data_ptr<float>(), ids.data_ptr<int>(), restore.data_ptr<int>(),    \
      reinterpret_cast<T *>(out.data_ptr()), out.size(1), ids.size(1))
  if (out.scalar_type() == at::kBFloat16) {
    COMBINE(__nv_bfloat16);
  } else if (out.scalar_type() == at::kHalf) {
    COMBINE(__half);
  } else {
    COMBINE(float);
  }
#undef COMBINE
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

namespace {
template <class T>
__global__ void activation_kernel(const T *gu, const float *probs, T *out,
                                  int64_t size, int n) {
  int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= size)
    return;
  int64_t row = i / n, col = i % n;
  float g, up;
  if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    g = __bfloat162float(gu[row * 2 * n + col]);
    up = __bfloat162float(gu[row * 2 * n + n + col]);
  } else {
    g = __half2float(gu[row * 2 * n + col]);
    up = __half2float(gu[row * 2 * n + n + col]);
  }
  float h = rounded<T>(__fdividef(g, __fadd_rn(1.f, expf(-g))));
  h = rounded<T>(__fmul_rn(h, up));
  out[i] = cast<T>(__fmul_rn(h, probs[row]));
}
} // namespace

// Fused native epilogue baseline, with exactly the same pointwise rounding as
// FC1.
void activation_cuda(const at::Tensor &gu, const at::Tensor &probs,
                     at::Tensor out) {
  TORCH_CHECK(gu.is_cuda() && gu.device() == probs.device() &&
              gu.device() == out.device());
  TORCH_CHECK(gu.is_contiguous() && probs.is_contiguous() &&
              out.is_contiguous());
  TORCH_CHECK(gu.dim() == 2 && out.dim() == 2 && probs.dim() == 1 &&
              probs.scalar_type() == at::kFloat);
  TORCH_CHECK(
      gu.scalar_type() == out.scalar_type() &&
      (gu.scalar_type() == at::kHalf || gu.scalar_type() == at::kBFloat16));
  TORCH_CHECK(gu.size(0) == out.size(0) && gu.size(1) == 2 * out.size(1) &&
              probs.numel() == out.size(0));
  const c10::cuda::CUDAGuard guard(gu.device());
  if (!out.numel())
    return;
#define ACT(T)                                                                 \
  activation_kernel<T><<<(out.numel() + 255) / 256, 256, 0,                    \
                         at::cuda::getCurrentCUDAStream()>>>(                  \
      reinterpret_cast<const T *>(gu.data_ptr()), probs.data_ptr<float>(),     \
      reinterpret_cast<T *>(out.data_ptr()), out.numel(), out.size(1))
  if (gu.scalar_type() == at::kBFloat16) {
    ACT(__nv_bfloat16);
  } else {
    ACT(__half);
  }
#undef ACT
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void counts_plan_cuda(const at::Tensor &counts, at::Tensor offsets,
                      at::Tensor tiles, at::Tensor total, int64_t rows) {
  const c10::cuda::CUDAGuard guard(counts.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  if (counts.scalar_type() == at::kLong)
    counts_plan_kernel<<<1, 256, 0, stream>>>(
        counts.data_ptr<int64_t>(), offsets.data_ptr<int>(),
        tiles.data_ptr<int>(), total.data_ptr<int>(), counts.numel(), rows);
  else
    counts_plan_kernel<<<1, 256, 0, stream>>>(
        counts.data_ptr<int>(), offsets.data_ptr<int>(), tiles.data_ptr<int>(),
        total.data_ptr<int>(), counts.numel(), rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

namespace {
template <class T> __device__ float root_float(T x) {
  if constexpr (std::is_same_v<T, __nv_bfloat16>)
    return __bfloat162float(x);
  else
    return __half2float(x);
}
template <class T>
__global__ void combine_root_rows_kernel(const T *input, const int *rows,
                                         T *output, int h, int k) {
  int token = blockIdx.x, col = blockIdx.y * 256 + threadIdx.x;
  if (col < h) {
    float acc = root_float(input[int64_t(rows[token * k]) * h + col]);
    for (int j = 1; j < k; ++j)
      acc = __fadd_rn(
          acc, root_float(input[int64_t(rows[token * k + j]) * h + col]));
    output[int64_t(token) * h + col] = cast<T>(acc);
  }
}
} // namespace
void combine_root_rows_cuda(const at::Tensor &input, const at::Tensor &rows,
                            at::Tensor output) {
  const c10::cuda::CUDAGuard guard(input.device());
  if (!output.size(0) || !output.size(1))
    return;
  auto stream = at::cuda::getCurrentCUDAStream();
  if (output.size(0) >= 128 && output.size(1) % 8 == 0 &&
      input.data_ptr() != nullptr && uintptr_t(input.data_ptr()) % 16 == 0) {
    dim3 vector_grid(output.size(0), (output.size(1) + 1023) / 1024);
    if (input.scalar_type() == at::kBFloat16)
      compact_combine_vector<<<vector_grid, 128, 0, stream>>>(
          static_cast<const __nv_bfloat16 *>(input.data_ptr()),
          rows.data_ptr<int>(), static_cast<__nv_bfloat16 *>(output.data_ptr()),
          output.size(1), rows.size(1));
    else
      compact_combine_vector<<<vector_grid, 128, 0, stream>>>(
          static_cast<const __half *>(input.data_ptr()), rows.data_ptr<int>(),
          static_cast<__half *>(output.data_ptr()), output.size(1),
          rows.size(1));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  dim3 grid(output.size(0), (output.size(1) + 255) / 256);
  if (input.scalar_type() == at::kBFloat16)
    combine_root_rows_kernel<<<grid, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16 *>(input.data_ptr()),
        rows.data_ptr<int>(), static_cast<__nv_bfloat16 *>(output.data_ptr()),
        output.size(1), rows.size(1));
  else
    combine_root_rows_kernel<<<grid, 256, 0, stream>>>(
        static_cast<const __half *>(input.data_ptr()), rows.data_ptr<int>(),
        static_cast<__half *>(output.data_ptr()), output.size(1), rows.size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void small_routes_cuda(const at::Tensor &ids, const at::Tensor &probs,
                       at::Tensor sorted, at::Tensor pp, at::Tensor rows,
                       at::Tensor offsets, at::Tensor tiles, at::Tensor total) {
  const c10::cuda::CUDAGuard guard(ids.device());
  auto stream = at::cuda::getCurrentCUDAStream();
#define SMALL_ROUTES(I, R)                                                     \
  small_routes_kernel<I, R><<<1, 256, 0, stream>>>(                            \
      ids.data_ptr<I>(), probs.data_ptr<float>(), ids.numel(),                 \
      offsets.numel() - 1, ids.size(1), sorted.data_ptr<int64_t>(),            \
      pp.data_ptr<float>(), rows.data_ptr<int>(), offsets.data_ptr<int>(),     \
      tiles.data_ptr<int>(), total.data_ptr<int>())
  if (ids.numel() <= 64) {
    if (ids.scalar_type() == at::kLong) {
      SMALL_ROUTES(int64_t, false);
    } else {
      SMALL_ROUTES(int, false);
    }
  } else {
    if (ids.scalar_type() == at::kLong) {
      SMALL_ROUTES(int64_t, true);
    } else {
      SMALL_ROUTES(int, true);
    }
  }
#undef SMALL_ROUTES
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void route_metadata_cuda(const at::Tensor &ids, const at::Tensor &probs,
                         const at::Tensor &inverse, at::Tensor sorted,
                         at::Tensor pp, at::Tensor rows) {
  const c10::cuda::CUDAGuard guard(ids.device());
  if (!ids.numel())
    return;
  route_metadata_kernel<<<(ids.numel() + 255) / 256, 256, 0,
                          at::cuda::getCurrentCUDAStream()>>>(
      ids.data_ptr<int>(), probs.data_ptr<float>(), inverse.data_ptr<int>(),
      sorted.data_ptr<int64_t>(), pp.data_ptr<float>(), rows.data_ptr<int>(),
      ids.numel(), ids.size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
