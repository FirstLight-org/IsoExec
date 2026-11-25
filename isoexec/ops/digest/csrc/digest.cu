// SPDX-License-Identifier: Apache-2.0
// Multi-tensor parameter digest: one launch over every tensor of a list, integer terms only.
//
// Each tensor is read as 16-bit words w_i (i = word index inside the tensor). Per tensor:
//   s1 = sum w_i,  s2 = sum w_i^2,  s3 = sum mix(i, w_i)          (uint64, modulo 2^64)
// Integer addition is associative and commutative, so the per-tensor result is the same whatever the
// grid, block or atomic order: the digest is deterministic and needs no fixed reduction schedule.
// s3 is position-sensitive (a permutation of the words changes it). The host sees one [n, 3] buffer.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>

namespace digest {
namespace {

// One chunk: a contiguous run of 16-bit words of one tensor.
struct Chunk {
  uint64_t ptr;      // byte address of the chunk's first word
  int64_t words;     // 16-bit words in the chunk
  int64_t base;      // index of the chunk's first word inside its tensor
  int64_t tensor;    // output row
};
static_assert(sizeof(Chunk) == 32, "Chunk is four int64 on the host");

__device__ __forceinline__ uint32_t mix(uint64_t i, uint32_t w) {
  uint32_t h = uint32_t(i) * 0x9E3779B1u ^ uint32_t(i >> 32) * 0x85EBCA77u ^ w * 0xC2B2AE3Du;
  h ^= h >> 15;
  h *= 0x27D4EB2Fu;
  h ^= h >> 13;
  return h;
}

struct Acc {
  uint64_t s1 = 0, s2 = 0, s3 = 0;
  __device__ __forceinline__ void add(uint64_t i, uint32_t w) {
    s1 += w;
    s2 += uint64_t(w * w);  // w < 2^16: w*w < 2^32, exact in uint32
    s3 += mix(i, w);
  }
};

constexpr int kThreads = 256;

__global__ __launch_bounds__(kThreads) void digest_kernel(const Chunk *__restrict__ chunks,
                                                          unsigned long long *__restrict__ out) {
  const Chunk c = chunks[blockIdx.x];
  Acc a;
  const int64_t n = c.words;
  if ((c.ptr & 15) == 0) {
    const uint4 *p = reinterpret_cast<const uint4 *>(c.ptr);
    const int64_t n8 = n / 8;
    for (int64_t v = threadIdx.x; v < n8; v += kThreads) {
      uint4 q = __ldg(p + v);
      uint32_t r[4] = {q.x, q.y, q.z, q.w};
      int64_t i = c.base + v * 8;
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        a.add(i + 2 * k, r[k] & 0xffffu);
        a.add(i + 2 * k + 1, r[k] >> 16);
      }
    }
    const uint16_t *t = reinterpret_cast<const uint16_t *>(c.ptr);
    for (int64_t j = n8 * 8 + threadIdx.x; j < n; j += kThreads)
      a.add(c.base + j, t[j]);
  } else {
    const uint16_t *t = reinterpret_cast<const uint16_t *>(c.ptr);
    for (int64_t j = threadIdx.x; j < n; j += kThreads)
      a.add(c.base + j, t[j]);
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    a.s1 += __shfl_xor_sync(0xffffffffu, a.s1, o);
    a.s2 += __shfl_xor_sync(0xffffffffu, a.s2, o);
    a.s3 += __shfl_xor_sync(0xffffffffu, a.s3, o);
  }
  __shared__ uint64_t sh[3][kThreads / 32];
  int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  if (lane == 0) {
    sh[0][warp] = a.s1;
    sh[1][warp] = a.s2;
    sh[2][warp] = a.s3;
  }
  __syncthreads();
  if (threadIdx.x < 3) {
    uint64_t s = 0;
    for (int w = 0; w < kThreads / 32; ++w)
      s += sh[threadIdx.x][w];
    atomicAdd(out + c.tensor * 3 + threadIdx.x, (unsigned long long)s);
  }
}

}  // namespace

// chunks: device int64 [n_chunks, 4] (Chunk rows); out: device int64 [n_tensors, 3], zeroed here.
void run(const at::Tensor &chunks, at::Tensor out) {
  TORCH_CHECK(chunks.is_cuda() && chunks.scalar_type() == at::kLong && chunks.dim() == 2 && chunks.size(1) == 4 &&
                  chunks.is_contiguous(),
              "digest: chunks must be a contiguous CUDA int64 [n, 4]");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kLong && out.is_contiguous() && out.size(-1) == 3,
              "digest: out must be a contiguous CUDA int64 [n, 3]");
  const at::cuda::CUDAGuard guard(chunks.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cudaMemsetAsync(out.data_ptr(), 0, out.numel() * sizeof(int64_t), stream));
  int64_t n = chunks.size(0);
  if (n == 0)
    return;
  TORCH_CHECK(n < (int64_t(1) << 31), "digest: too many chunks");
  digest_kernel<<<dim3(unsigned(n)), kThreads, 0, stream>>>(
      reinterpret_cast<const Chunk *>(chunks.data_ptr<int64_t>()),
      reinterpret_cast<unsigned long long *>(out.data_ptr<int64_t>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace digest

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &digest::run); }
