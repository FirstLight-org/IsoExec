// SPDX-License-Identifier: Apache-2.0
// Trainer backward epilogue (integration.py). FLA's chunked backward kernels produce dq and dk per value
// head (and a second dk from the WY backward); the key-head gradients are their group sums, and the
// l2norm backward maps them to the raw q, k:
//   dy = sum over the value heads g of key head hq (dq_hv; dk_hv + dk2_hv)          (fp32)
//   dx = dy (*) r - (sum_d dy y) (*) y (*) r      with y = the normalized row, r = its inverse norm
// One warp per (token, key head) row and operand, one pass over the inputs. No determinism claim is
// needed here, but the sums are fixed-order anyway.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

#include <tuple>

namespace {

__device__ __forceinline__ void load4(const __nv_bfloat16 *p, float (&x)[4]) {
  const uint2 raw = *reinterpret_cast<const uint2 *>(p);
  const __nv_bfloat162 a = *reinterpret_cast<const __nv_bfloat162 *>(&raw.x), b = *reinterpret_cast<const __nv_bfloat162 *>(&raw.y);
  x[0] = __low2float(a), x[1] = __high2float(a), x[2] = __low2float(b), x[3] = __high2float(b);
}

// rows = T * hk; row r = (t, hq) of the key-head tensors, value head rows r * G .. r * G + G - 1.
__global__ void qk_finish_kernel(const __nv_bfloat16 *dq_hv, const __nv_bfloat16 *dk_hv, const __nv_bfloat16 *dk2_hv,
                                 const __nv_bfloat16 *qn, const __nv_bfloat16 *kn, const float *q_rstd,
                                 const float *k_rstd, __nv_bfloat16 *dq, __nv_bfloat16 *dk, int64_t rows, int G) {
  const int lane = threadIdx.x & 31;
  const int64_t task = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;   // 2 rows x {q, k}
  if (task >= 2 * rows) return;
  const bool key = task & 1;
  const int64_t r = task >> 1;
  float dy[4] = {0.f, 0.f, 0.f, 0.f}, y[4], x[4];
  for (int g = 0; g < G; ++g) {
    const int64_t off = (r * G + g) * 128 + 4 * lane;
    load4((key ? dk_hv : dq_hv) + off, x);
    for (int e = 0; e < 4; ++e) dy[e] += x[e];
    if (key) {
      load4(dk2_hv + off, x);
      for (int e = 0; e < 4; ++e) dy[e] += x[e];
    }
  }
  load4((key ? kn : qn) + r * 128 + 4 * lane, y);
  float s = dy[0] * y[0] + dy[1] * y[1] + dy[2] * y[2] + dy[3] * y[3];
  for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
  const float rs = (key ? k_rstd : q_rstd)[r];
  __nv_bfloat162 out[2];
  for (int h = 0; h < 2; ++h)
    out[h] = __floats2bfloat162_rn(dy[2 * h] * rs - s * y[2 * h] * rs, dy[2 * h + 1] * rs - s * y[2 * h + 1] * rs);
  *reinterpret_cast<uint2 *>((key ? dk : dq) + r * 128 + 4 * lane) = *reinterpret_cast<const uint2 *>(out);
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> gdn_cpr_bwd_qk_finish(const at::Tensor &dq_hv, const at::Tensor &dk_hv,
                                                         const at::Tensor &dk2_hv, const at::Tensor &qn,
                                                         const at::Tensor &kn, const at::Tensor &q_rstd,
                                                         const at::Tensor &k_rstd) {
  const c10::cuda::CUDAGuard guard(qn.device());
  for (const auto *t : {&dq_hv, &dk_hv, &dk2_hv, &qn, &kn, &q_rstd, &k_rstd})
    TORCH_CHECK(t->is_cuda() && t->device() == qn.device() && t->is_contiguous(),
                "gdn_cpr_sm90: bwd_qk_finish: contiguous tensors on one CUDA device");
  for (const auto *t : {&dq_hv, &dk_hv, &dk2_hv, &qn, &kn})
    TORCH_CHECK(t->scalar_type() == at::kBFloat16 && t->dim() == 3 && t->size(-1) == 128,
                "gdn_cpr_sm90: bwd_qk_finish: bf16 [T, heads, 128] gradients and rows");
  TORCH_CHECK(kn.sizes() == qn.sizes(), "gdn_cpr_sm90: bwd_qk_finish: qn and kn must be [T, hk, 128] alike");
  for (const auto *t : {&q_rstd, &k_rstd})
    TORCH_CHECK(t->scalar_type() == at::kFloat && t->dim() == 2 && t->size(0) == qn.size(0) && t->size(1) == qn.size(1),
                "gdn_cpr_sm90: bwd_qk_finish: inverse norms must be fp32 [T, hk]");
  const int64_t T = qn.size(0), hk = qn.size(1), rows = T * hk;
  for (const auto *t : {&dq_hv, &dk_hv, &dk2_hv})
    TORCH_CHECK(t->size(0) == T && hk > 0 && t->size(1) % hk == 0 && t->sizes() == dq_hv.sizes(),
                "gdn_cpr_sm90: bwd_qk_finish: value-head gradients must be [T, hk * G, 128]");
  const int G = hk > 0 ? int(dq_hv.size(1) / hk) : 0;
  at::Tensor dq = at::empty_like(qn), dk = at::empty_like(kn);
  if (rows > 0) {
    const int threads = 256;
    const int64_t blocks = (2 * rows * 32 + threads - 1) / threads;
    qk_finish_kernel<<<unsigned(blocks), threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16 *>(dq_hv.data_ptr()), reinterpret_cast<const __nv_bfloat16 *>(dk_hv.data_ptr()),
        reinterpret_cast<const __nv_bfloat16 *>(dk2_hv.data_ptr()), reinterpret_cast<const __nv_bfloat16 *>(qn.data_ptr()),
        reinterpret_cast<const __nv_bfloat16 *>(kn.data_ptr()), q_rstd.data_ptr<float>(), k_rstd.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16 *>(dq.data_ptr()), reinterpret_cast<__nv_bfloat16 *>(dk.data_ptr()), rows, G);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {dq, dk};
}
