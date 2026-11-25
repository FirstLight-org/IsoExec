// SPDX-License-Identifier: Apache-2.0
// Fused glue kernels for the native Flash models. Every kernel is either pure data movement (row scatter,
// embedding expand), an order-free selection with a total tie-break (argmax) or a single IEEE product
// rounded once (row scale). No kernel reduces floating-point values, so there is no tree to declare, no
// atomic, and no output bit depends on the grid shape, the row count or the SM count.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

namespace fused_glue {

constexpr int kMaxTensors = 4;

struct ScatterArgs {
  const char *src[kMaxTensors];
  char *dst[kMaxTensors];
  int64_t row_bytes[kMaxTensors];
  int n_tensors;
};

// dst_t[rows[n], cols[n] + col_offset, :] = src_t[n, :] byte for byte, 16 bytes per thread.
__global__ void __launch_bounds__(128) scatter_rows_kernel(ScatterArgs a, const int64_t *__restrict__ rows,
                                                           const int64_t *__restrict__ cols, int64_t col_offset,
                                                           int64_t B, int64_t L) {
  const int64_t n = blockIdx.x;
  const int t = blockIdx.y;
  const int64_t r = rows[n], c = cols[n] + col_offset;
  if (r < 0 || r >= B || c < 0 || c >= L) __trap(); // torch's index_put asserts the same condition
  const int64_t rb = a.row_bytes[t];
  const char *s = a.src[t] + n * rb;
  char *d = a.dst[t] + (r * L + c) * rb;
  for (int64_t i = (int64_t)threadIdx.x * 16; i < rb; i += (int64_t)blockDim.x * 16)
    *reinterpret_cast<uint4 *>(d + i) = *reinterpret_cast<const uint4 *>(s + i);
}

void scatter_rows_cuda(const std::vector<at::Tensor> &srcs, const std::vector<at::Tensor> &dsts,
                       const at::Tensor &rows, const at::Tensor &cols, int64_t col_offset) {
  TORCH_CHECK(srcs.size() == dsts.size() && !srcs.empty() && (int)srcs.size() <= kMaxTensors,
              "scatter_rows: 1..4 (src, dst) pairs");
  TORCH_CHECK(rows.is_cuda() && cols.is_cuda() && rows.scalar_type() == at::kLong && cols.scalar_type() == at::kLong &&
                  rows.is_contiguous() && cols.is_contiguous() && rows.numel() == cols.numel(),
              "scatter_rows: contiguous int64 CUDA rows/cols of equal length");
  const int64_t N = rows.numel();
  const at::cuda::CUDAGuard guard(rows.device());
  ScatterArgs a{};
  a.n_tensors = (int)srcs.size();
  int64_t B = dsts[0].size(0), L = dsts[0].size(1);
  for (int t = 0; t < a.n_tensors; ++t) {
    const at::Tensor &s = srcs[t];
    const at::Tensor &d = dsts[t];
    TORCH_CHECK(s.is_cuda() && d.is_cuda() && s.is_contiguous() && d.is_contiguous() && s.dim() >= 1 && d.dim() >= 3 - 1,
                "scatter_rows: contiguous CUDA tensors");
    TORCH_CHECK(s.scalar_type() == d.scalar_type() && s.size(0) == N, "scatter_rows: dtype / row count mismatch");
    TORCH_CHECK(d.size(0) == B && d.size(1) == L, "scatter_rows: every destination shares [B, L]");
    const int64_t rb = s.numel() / std::max<int64_t>(N, 1) * s.element_size();
    TORCH_CHECK(N == 0 || (d.numel() / (B * L) * d.element_size() == rb && rb % 16 == 0),
                "scatter_rows: row bytes must match and be a multiple of 16");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(s.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(d.data_ptr()) % 16 == 0,
                "scatter_rows: 16-byte aligned storage");
    a.src[t] = static_cast<const char *>(s.data_ptr());
    a.dst[t] = static_cast<char *>(d.data_ptr());
    a.row_bytes[t] = rb;
  }
  if (N == 0) return;
  dim3 grid((unsigned)N, (unsigned)a.n_tensors);
  scatter_rows_kernel<<<grid, 128, 0, at::cuda::getCurrentCUDAStream()>>>(a, rows.data_ptr<int64_t>(),
                                                                          cols.data_ptr<int64_t>(), col_offset, B, L);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// out[n, s, :] = weight[ids[n], :] for every stream s; optionally pre[n, s] = (s == 0 ? 1.f : 0.f).
__global__ void __launch_bounds__(256) embed_expand_kernel(const char *__restrict__ w, const int64_t *__restrict__ ids,
                                                           char *__restrict__ out, float *__restrict__ pre,
                                                           int64_t V, int64_t row_bytes, int S) {
  const int64_t n = blockIdx.x;
  const int64_t id = ids[n];
  if (id < 0 || id >= V) __trap();
  const char *src = w + id * row_bytes;
  for (int s = 0; s < S; ++s) {
    char *dst = out + ((int64_t)n * S + s) * row_bytes;
    for (int64_t i = (int64_t)threadIdx.x * 16; i < row_bytes; i += (int64_t)blockDim.x * 16)
      *reinterpret_cast<uint4 *>(dst + i) = *reinterpret_cast<const uint4 *>(src + i);
  }
  if (pre != nullptr && threadIdx.x < S) pre[(int64_t)n * S + threadIdx.x] = threadIdx.x == 0 ? 1.0f : 0.0f;
}

void embed_expand_cuda(const at::Tensor &weight, const at::Tensor &ids, at::Tensor out, at::Tensor pre) {
  TORCH_CHECK(weight.is_cuda() && ids.is_cuda() && out.is_cuda() && weight.is_contiguous() && out.is_contiguous() &&
                  ids.is_contiguous() && ids.scalar_type() == at::kLong && weight.dim() == 2 && out.dim() == 3,
              "embed_expand: contiguous CUDA weight [V, H], int64 ids [N], out [N, S, H]");
  const int64_t N = ids.numel(), V = weight.size(0), H = weight.size(1);
  const int S = (int)out.size(1);
  TORCH_CHECK(out.size(0) == N && out.size(2) == H && out.scalar_type() == weight.scalar_type(), "embed_expand: shape / dtype");
  const int64_t rb = H * weight.element_size();
  TORCH_CHECK(rb % 16 == 0 && S >= 1 && S <= 256, "embed_expand: row bytes % 16 and 1 <= S <= 256");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(weight.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0,
              "embed_expand: 16-byte aligned storage");
  float *pre_ptr = nullptr;
  if (pre.defined() && pre.numel() > 0) {
    TORCH_CHECK(pre.is_cuda() && pre.is_contiguous() && pre.scalar_type() == at::kFloat && pre.numel() == N * S,
                "embed_expand: pre must be contiguous float32 [N, S]");
    pre_ptr = pre.data_ptr<float>();
  }
  if (N == 0) return;
  const at::cuda::CUDAGuard guard(weight.device());
  embed_expand_kernel<<<(unsigned)N, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      static_cast<const char *>(weight.data_ptr()), ids.data_ptr<int64_t>(), static_cast<char *>(out.data_ptr()), pre_ptr, V, rb, S);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T> struct Up;
template <> struct Up<float> { __device__ static float f(float v) { return v; } };
template <> struct Up<__nv_bfloat16> { __device__ static float f(__nv_bfloat16 v) { return __bfloat162float(v); } };
template <> struct Up<__half> { __device__ static float f(__half v) { return __half2float(v); } };

// torch's GreaterOrNan: does (a, ia) beat (b, ib)? NaN beats everything, equal values keep the lower index.
// A total order, so the winner is independent of the reduction tree.
__device__ __forceinline__ bool beats(float a, int ia, float b, int ib) {
  if (isnan(a)) return isnan(b) ? ia < ib : true;
  if (isnan(b)) return false;
  return a == b ? ia < ib : a > b;
}

// Two launches, no atomics: (1) grid (R, S) blocks each scan a contiguous slice of one row in increasing index order
// (16-byte vector loads) and write their best (value bits, index) to a scratch slot; (2) one warp per row folds the S
// slots. The winner is the maximum under a total order (value, then lowest index; NaN above everything), so it does
// not depend on S, the slice boundaries or the fold order: S is chosen from R only to fill the machine.
template <typename T, bool VECTOR>
__global__ void __launch_bounds__(256) argmax_partial_kernel(const T *__restrict__ x, int *__restrict__ scratch,
                                                              int64_t V, int S) {
  constexpr int VEC = VECTOR ? 16 / (int)sizeof(T) : 1;
  const T *row = x + (int64_t)blockIdx.x * V;
  const int64_t chunk = ((V + S - 1) / S + VEC - 1) / VEC * VEC; // slice length, a multiple of VEC
  const int64_t lo = (int64_t)blockIdx.y * chunk;
  const int64_t hi = min(V, lo + chunk);
  float best = 0.f;
  int bi = -1;
  if constexpr (VECTOR) {
#pragma unroll 4
    for (int64_t i = lo + (int64_t)threadIdx.x * VEC; i < hi; i += (int64_t)blockDim.x * VEC) {
      alignas(16) T v[VEC];
      *reinterpret_cast<uint4 *>(v) = *reinterpret_cast<const uint4 *>(row + i);
#pragma unroll
      for (int e = 0; e < VEC; ++e) {
        const float f = Up<T>::f(v[e]);
        if (bi < 0 || beats(f, (int)(i + e), best, bi)) { best = f; bi = (int)(i + e); }
      }
    }
  } else {
    for (int64_t i = lo + threadIdx.x; i < hi; i += blockDim.x) {
      const float f = Up<T>::f(row[i]);
      if (bi < 0 || beats(f, (int)i, best, bi)) { best = f; bi = (int)i; }
    }
  }
  for (int off = 16; off > 0; off >>= 1) {
    const float ov = __shfl_down_sync(0xffffffffu, best, off);
    const int oi = __shfl_down_sync(0xffffffffu, bi, off);
    if (oi >= 0 && (bi < 0 || beats(ov, oi, best, bi))) { best = ov; bi = oi; }
  }
  __shared__ float sv[8];
  __shared__ int si[8];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (lane == 0) { sv[warp] = best; si[warp] = bi; }
  __syncthreads();
  if (warp == 0) {
    const int nw = blockDim.x >> 5;
    best = lane < nw ? sv[lane] : 0.f;
    bi = lane < nw ? si[lane] : -1;
    for (int off = 16; off > 0; off >>= 1) {
      const float ov = __shfl_down_sync(0xffffffffu, best, off);
      const int oi = __shfl_down_sync(0xffffffffu, bi, off);
      if (oi >= 0 && (bi < 0 || beats(ov, oi, best, bi))) { best = ov; bi = oi; }
    }
    if (lane == 0) {
      int *slot = scratch + ((int64_t)blockIdx.x * S + blockIdx.y) * 2;
      slot[0] = __float_as_int(best);
      slot[1] = bi;
    }
  }
}

__global__ void __launch_bounds__(32) argmax_final_kernel(const int *__restrict__ scratch, int64_t *__restrict__ out, int S) {
  const int *base = scratch + (int64_t)blockIdx.x * S * 2;
  float best = 0.f;
  int bi = -1;
  for (int s = threadIdx.x; s < S; s += 32) {
    const int oi = base[s * 2 + 1];
    const float ov = __int_as_float(base[s * 2]);
    if (oi >= 0 && (bi < 0 || beats(ov, oi, best, bi))) { best = ov; bi = oi; }
  }
  for (int off = 16; off > 0; off >>= 1) {
    const float ov = __shfl_down_sync(0xffffffffu, best, off);
    const int oi = __shfl_down_sync(0xffffffffu, bi, off);
    if (oi >= 0 && (bi < 0 || beats(ov, oi, best, bi))) { best = ov; bi = oi; }
  }
  if (threadIdx.x == 0) out[blockIdx.x] = bi;
}

template <typename T> void launch_argmax(const at::Tensor &x, at::Tensor &out, at::Tensor &scratch, int S, cudaStream_t s) {
  const int64_t R = x.size(0), V = x.size(1);
  const T *p = reinterpret_cast<const T *>(x.data_ptr());
  const bool vec = (V * (int64_t)sizeof(T)) % 16 == 0 && reinterpret_cast<uintptr_t>(p) % 16 == 0;
  dim3 grid((unsigned)R, (unsigned)S);
  if (vec) argmax_partial_kernel<T, true><<<grid, 256, 0, s>>>(p, scratch.data_ptr<int>(), V, S);
  else argmax_partial_kernel<T, false><<<grid, 256, 0, s>>>(p, scratch.data_ptr<int>(), V, S);
  argmax_final_kernel<<<(unsigned)R, 32, 0, s>>>(scratch.data_ptr<int>(), out.data_ptr<int64_t>(), S);
}

void argmax_rows_cuda(const at::Tensor &x, at::Tensor out, at::Tensor scratch) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() == 2 && x.size(1) >= 1 && x.size(1) < (1LL << 31),
              "argmax_rows: contiguous CUDA [R, V], V < 2^31");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kLong && out.numel() == x.size(0), "argmax_rows: int64 out [R]");
  const int64_t R = x.size(0);
  TORCH_CHECK(scratch.is_cuda() && scratch.scalar_type() == at::kInt && scratch.is_contiguous() && R > 0 &&
                  scratch.numel() % (2 * R) == 0, "argmax_rows: int32 scratch [R, S, 2]");
  const int S = (int)(scratch.numel() / (2 * R));
  TORCH_CHECK(S >= 1 && S <= 1024, "argmax_rows: 1 <= S <= 1024");
  const at::cuda::CUDAGuard guard(x.device());
  auto s = at::cuda::getCurrentCUDAStream();
  switch (x.scalar_type()) {
  case at::kFloat: launch_argmax<float>(x, out, scratch, S, s); break;
  case at::kBFloat16: launch_argmax<__nv_bfloat16>(x, out, scratch, S, s); break;
  case at::kHalf: launch_argmax<__half>(x, out, scratch, S, s); break;
  default: TORCH_CHECK(false, "argmax_rows: float32 / bfloat16 / float16");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// out[r, h] = bf16(float(w[h]) * float(y[r, h])): torch's bf16 broadcast product, one rounding.
__global__ void __launch_bounds__(256) mul_rows_kernel(const __nv_bfloat16 *__restrict__ y, const __nv_bfloat16 *__restrict__ w,
                                                       __nv_bfloat16 *__restrict__ out, int64_t total, int64_t H) {
  for (int64_t i = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) * 8; i < total; i += (int64_t)gridDim.x * blockDim.x * 8) {
    const int64_t h = i % H;
    alignas(16) __nv_bfloat16 yv[8], wv[8], ov[8];
    *reinterpret_cast<uint4 *>(yv) = *reinterpret_cast<const uint4 *>(y + i);
    *reinterpret_cast<uint4 *>(wv) = *reinterpret_cast<const uint4 *>(w + h);
#pragma unroll
    for (int e = 0; e < 8; ++e) ov[e] = __float2bfloat16_rn(__bfloat162float(wv[e]) * __bfloat162float(yv[e]));
    *reinterpret_cast<uint4 *>(out + i) = *reinterpret_cast<const uint4 *>(ov);
  }
}

void mul_rows_cuda(const at::Tensor &y, const at::Tensor &w, at::Tensor out) {
  TORCH_CHECK(y.is_cuda() && w.is_cuda() && out.is_cuda() && y.is_contiguous() && w.is_contiguous() && out.is_contiguous() &&
                  y.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16 && out.scalar_type() == at::kBFloat16,
              "mul_rows: contiguous CUDA bf16");
  const int64_t H = w.numel(), total = y.numel();
  TORCH_CHECK(H % 8 == 0 && y.size(-1) == H && out.numel() == total, "mul_rows: H % 8 == 0 and y[..., H]");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(y.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0, "mul_rows: 16-byte aligned storage");
  if (total == 0) return;
  const at::cuda::CUDAGuard guard(y.device());
  int64_t blocks = std::min<int64_t>((total / 8 + 255) / 256, 4096);
  mul_rows_kernel<<<(unsigned)blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16 *>(y.data_ptr()), reinterpret_cast<const __nv_bfloat16 *>(w.data_ptr()),
      reinterpret_cast<__nv_bfloat16 *>(out.data_ptr()), total, H);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace fused_glue
