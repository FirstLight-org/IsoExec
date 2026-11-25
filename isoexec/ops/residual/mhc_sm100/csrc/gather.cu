// SPDX-License-Identifier: Apache-2.0
// Row gather of several tensors in ONE launch: dst_t[r] = src_t[index[r]] for every tensor t (a byte copy, no
// arithmetic). The CED decoder's segment gather (the handoff streams, h, pre, post, comb at the decoder rows) in
// place of one index_select per tensor. Grid: (rows, tensors); a CTA copies one row of one tensor in 16-byte
// words when the row bytes and both bases are 16-byte aligned, else bytes. Out-of-range indices are refused on
// the host side by the caller's contract (DecoderRows); the kernel writes nothing for them.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <vector>

namespace mhc_gather {

constexpr int kMaxTensors = 8;

struct Args {
  const char *src[kMaxTensors];
  char *dst[kMaxTensors];
  int64_t row_bytes[kMaxTensors];
};

__global__ void gather_rows_kernel(const int64_t *__restrict__ index, int64_t src_rows, Args a) {
  const int64_t r = blockIdx.x;
  const int t = blockIdx.y;
  const int64_t s = index[r];
  if (s < 0 || s >= src_rows) return;
  const int64_t nb = a.row_bytes[t];
  const char *src = a.src[t] + s * nb;
  char *dst = a.dst[t] + r * nb;
  if (((nb | reinterpret_cast<uintptr_t>(src) | reinterpret_cast<uintptr_t>(dst)) & 15) == 0) {
    const int4 *s4 = reinterpret_cast<const int4 *>(src);
    int4 *d4 = reinterpret_cast<int4 *>(dst);
    for (int64_t i = threadIdx.x; i < nb / 16; i += blockDim.x) d4[i] = s4[i];
  } else {
    for (int64_t i = threadIdx.x; i < nb; i += blockDim.x) dst[i] = src[i];
  }
}

}  // namespace mhc_gather

namespace mhc_host {
void gather_rows_cuda(const at::Tensor &index, const std::vector<at::Tensor> &srcs, std::vector<at::Tensor> &dsts) {
  TORCH_CHECK(index.scalar_type() == at::kLong && index.dim() == 1 && index.is_contiguous() && index.is_cuda(),
              "index: int64 [R] CUDA");
  TORCH_CHECK(!srcs.empty() && srcs.size() == dsts.size() && srcs.size() <= mhc_gather::kMaxTensors,
              "1..8 source / destination pairs");
  const int64_t R = index.size(0);
  const int64_t T = srcs[0].size(0);
  mhc_gather::Args a{};
  for (size_t t = 0; t < srcs.size(); ++t) {
    const auto &s = srcs[t];
    auto &d = dsts[t];
    TORCH_CHECK(s.is_cuda() && d.is_cuda() && s.device() == index.device() && d.device() == index.device());
    TORCH_CHECK(s.is_contiguous() && d.is_contiguous() && s.scalar_type() == d.scalar_type(), "contiguous, same dtype");
    TORCH_CHECK(s.dim() >= 1 && s.size(0) == T && d.dim() == s.dim() && d.size(0) == R, "rows: src [T, ...], dst [R, ...]");
    for (int64_t k = 1; k < s.dim(); ++k) TORCH_CHECK(s.size(k) == d.size(k), "row shapes differ");
    a.src[t] = reinterpret_cast<const char *>(s.data_ptr());
    a.dst[t] = reinterpret_cast<char *>(d.data_ptr());
    a.row_bytes[t] = T ? s.numel() / T * s.element_size() : 0;
  }
  if (!R) return;
  const c10::cuda::CUDAGuard guard(index.device());
  mhc_gather::gather_rows_kernel<<<dim3(unsigned(R), unsigned(srcs.size())), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      index.data_ptr<int64_t>(), T, a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace mhc_host
