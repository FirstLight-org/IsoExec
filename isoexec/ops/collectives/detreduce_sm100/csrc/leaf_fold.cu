// SPDX-License-Identifier: Apache-2.0
// The BF16-leaf fold of the V4.1 wo_b program, local form: out[i] = BF16_RNE(tree(planes[0][i] .. planes[P-1][i])),
// planes BF16 in leaf order (rank-major: rank r's 8/W leaves are planes [r 8/W, (r+1) 8/W)), each widened to FP32
// exactly, `tree` the balanced adjacent-pair FP32 tree (P a power of two, at most 16; lower index left; every add
// __fadd_rn, no multiply, so nothing to contract), ONE rounding at the root. It is the arithmetic the symmetric-memory
// transport applies to the peers' leaf planes; here the planes are local (TP = 1, the NCCL realization after an
// all-gather of leaf planes, and the single-process simulation of every world size).
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

namespace detreduce_leaf {

constexpr int kMaxPlanes = 16;

template <int P>
__device__ __forceinline__ float tree(const float (&v)[P]) {
  float t[P];
#pragma unroll
  for (int i = 0; i < P; ++i) t[i] = v[i];
#pragma unroll
  for (int w = P; w > 1; w >>= 1) {
#pragma unroll
    for (int i = 0; i < w / 2; ++i) t[i] = __fadd_rn(t[2 * i], t[2 * i + 1]);
  }
  return t[0];
}

template <int P>
__global__ void leaf_fold_kernel(const __nv_bfloat16 *__restrict__ planes, int64_t plane_stride, int64_t n,
                                 __nv_bfloat16 *__restrict__ out) {
  // 8 elements per thread through 16-byte loads (n % 8 == 0 and 16-byte aligned planes, checked by the host)
  const int64_t vecs = n / 8;
  for (int64_t v = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; v < vecs; v += int64_t(gridDim.x) * blockDim.x) {
    float acc[8][P];
#pragma unroll
    for (int p = 0; p < P; ++p) {
      const uint4 raw = reinterpret_cast<const uint4 *>(planes + p * plane_stride)[v];
      const __nv_bfloat16 *b = reinterpret_cast<const __nv_bfloat16 *>(&raw);
#pragma unroll
      for (int j = 0; j < 8; ++j) acc[j][p] = __bfloat162float(b[j]);
    }
    uint4 res;
    __nv_bfloat16 *r = reinterpret_cast<__nv_bfloat16 *>(&res);
#pragma unroll
    for (int j = 0; j < 8; ++j) r[j] = __float2bfloat16_rn(tree<P>(acc[j]));
    reinterpret_cast<uint4 *>(out)[v] = res;
  }
}

}  // namespace detreduce_leaf

namespace detreduce {
// planes: BF16 [P, n] (row stride >= n, 16-byte aligned rows); out: BF16 [n] contiguous. P in {1, 2, 4, 8, 16}.
void leaf_fold_bf16(const at::Tensor &planes, at::Tensor &out) {
  TORCH_CHECK(planes.is_cuda() && planes.scalar_type() == at::kBFloat16 && planes.dim() == 2 && planes.stride(1) == 1,
              "planes: BF16 [P, n] CUDA, unit column stride");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kBFloat16 && out.is_contiguous() && out.numel() == planes.size(1),
              "out: contiguous BF16 [n]");
  const int64_t P = planes.size(0), n = planes.size(1);
  TORCH_CHECK(P >= 1 && P <= detreduce_leaf::kMaxPlanes && (P & (P - 1)) == 0, "P a power of two <= 16");
  TORCH_CHECK(n % 8 == 0 && planes.stride(0) % 8 == 0 && (reinterpret_cast<uintptr_t>(planes.data_ptr()) & 15) == 0 &&
              (reinterpret_cast<uintptr_t>(out.data_ptr()) & 15) == 0, "n % 8 == 0, 16-byte aligned planes and out");
  if (!n) return;
  const c10::cuda::CUDAGuard guard(planes.device());
  const int threads = 256;
  const int64_t vecs = n / 8;
  const int blocks = int(std::min<int64_t>((vecs + threads - 1) / threads, 148 * 8));
  auto s = at::cuda::getCurrentCUDAStream();
  const auto *pp = reinterpret_cast<const __nv_bfloat16 *>(planes.data_ptr());
  auto *op = reinterpret_cast<__nv_bfloat16 *>(out.data_ptr());
  const int64_t ps = planes.stride(0);
  switch (P) {
    case 1: detreduce_leaf::leaf_fold_kernel<1><<<blocks, threads, 0, s>>>(pp, ps, n, op); break;
    case 2: detreduce_leaf::leaf_fold_kernel<2><<<blocks, threads, 0, s>>>(pp, ps, n, op); break;
    case 4: detreduce_leaf::leaf_fold_kernel<4><<<blocks, threads, 0, s>>>(pp, ps, n, op); break;
    case 8: detreduce_leaf::leaf_fold_kernel<8><<<blocks, threads, 0, s>>>(pp, ps, n, op); break;
    default: detreduce_leaf::leaf_fold_kernel<16><<<blocks, threads, 0, s>>>(pp, ps, n, op); break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace detreduce
