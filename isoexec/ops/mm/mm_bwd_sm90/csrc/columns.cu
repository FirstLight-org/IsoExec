// Gradient of split_columns: assemble the column blocks' gradients into one [T, sum(w_i)] buffer.
//
//   out[r, off_i + c] = rn_bf16(add.rn(fl32(g_i[r, c]), +0))
//
// exactly what ``core/linear.py::_SplitColumns.backward`` computes with one ``torch.add(g_i, 0,
// out=...)`` per block (the ``+ 0`` maps -0 to +0, as autograd's slice accumulation does), in one
// launch. Each block may be any row-strided bf16 view with unit column stride.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include <vector>

namespace isoexec_mm_bwd_sm90 {
namespace {

constexpr int kMaxBlocks = 8;

struct Blocks {
  const __nv_bfloat16 *src[kMaxBlocks];
  int64_t ld[kMaxBlocks];
  int64_t off[kMaxBlocks + 1];  // cumulative column offsets
  int vec[kMaxBlocks];          // 1: 16-byte vectorisable block
  int n;
};

__device__ __forceinline__ __nv_bfloat16 plus0(__nv_bfloat16 v) {
  return __float2bfloat16_rn(__fadd_rn(__bfloat162float(v), 0.0f));
}

// grid.y = block index; grid.x strides over that block's elements (8 per thread when vectorisable).
__global__ void assemble_columns(Blocks b, __nv_bfloat16 *__restrict__ out, int64_t rows, int64_t ldo) {
  const int i = blockIdx.y;
  const int64_t w = b.off[i + 1] - b.off[i];
  const __nv_bfloat16 *src = b.src[i];
  const int64_t ld = b.ld[i];
  __nv_bfloat16 *dst = out + b.off[i];
  const int64_t stride = int64_t(blockDim.x) * gridDim.x;
  if (b.vec[i]) {
    const int64_t w8 = w / 8, n8 = rows * w8;
    for (int64_t e = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; e < n8; e += stride) {
      const int64_t r = e / w8, c = (e - r * w8) * 8;
      uint4 v = *reinterpret_cast<const uint4 *>(src + r * ld + c);
      __nv_bfloat16 *h = reinterpret_cast<__nv_bfloat16 *>(&v);
#pragma unroll
      for (int j = 0; j < 8; ++j) h[j] = plus0(h[j]);
      *reinterpret_cast<uint4 *>(dst + r * ldo + c) = v;
    }
  } else {
    const int64_t n = rows * w;
    for (int64_t e = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; e < n; e += stride) {
      const int64_t r = e / w, c = e - r * w;
      dst[r * ldo + c] = plus0(src[r * ld + c]);
    }
  }
}

}  // namespace
}  // namespace isoexec_mm_bwd_sm90

using namespace isoexec_mm_bwd_sm90;

// grads: the column blocks' gradients in column order, each [T, w_i] bf16 with unit column stride.
at::Tensor assemble_columns(std::vector<at::Tensor> const &grads) {
  TORCH_CHECK(!grads.empty() && grads.size() <= size_t(kMaxBlocks), "assemble_columns: 1..8 blocks");
  const int64_t rows = grads[0].size(0);
  Blocks b{};
  b.n = int(grads.size());
  b.off[0] = 0;
  for (int i = 0; i < b.n; ++i) {
    auto const &g = grads[i];
    TORCH_CHECK(g.is_cuda() && g.scalar_type() == at::kBFloat16 && g.dim() == 2 && g.size(0) == rows,
                "assemble_columns: [T, w_i] bf16 CUDA blocks with a common T");
    TORCH_CHECK(g.stride(1) == 1 || g.size(1) <= 1, "assemble_columns: unit column stride");
    b.src[i] = reinterpret_cast<const __nv_bfloat16 *>(g.data_ptr());
    b.ld[i] = g.stride(0);
    b.off[i + 1] = b.off[i] + g.size(1);
  }
  const int64_t width = b.off[b.n];
  at::Tensor out = at::empty({rows, width}, grads[0].options().memory_format(at::MemoryFormat::Contiguous));
  if (rows == 0 || width == 0) return out;
  int64_t most = 0;
  for (int i = 0; i < b.n; ++i) {
    const int64_t w = b.off[i + 1] - b.off[i];
    b.vec[i] = w % 8 == 0 && b.ld[i] % 8 == 0 && b.off[i] % 8 == 0 && width % 8 == 0 &&
               (reinterpret_cast<uintptr_t>(b.src[i]) & 15u) == 0;
    const int64_t work = b.vec[i] ? rows * (w / 8) : rows * w;
    most = work > most ? work : most;
  }
  const at::cuda::CUDAGuard guard(grads[0].device());
  const int threads = 256;
  int64_t gx = (most + threads - 1) / threads;
  gx = gx < 2048 ? gx : 2048;
  assemble_columns<<<dim3(unsigned(gx), unsigned(b.n)), threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      b, reinterpret_cast<__nv_bfloat16 *>(out.data_ptr()), rows, width);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
