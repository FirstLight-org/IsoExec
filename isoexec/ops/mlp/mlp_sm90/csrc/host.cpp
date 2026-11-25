// SPDX-License-Identifier: Apache-2.0
// Host entry for the SM90 fused mlp.act: silu(g) * (u + 0.0) in one launch. No device sync (capturable).
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

namespace mlp_sm90 {

void launch_silu_mul(const __nv_bfloat16 *g, const __nv_bfloat16 *u, __nv_bfloat16 *out, int64_t n,
                     cudaStream_t stream);

bool launch_silu_mul_rows(const __nv_bfloat16 *g, const __nv_bfloat16 *u, __nv_bfloat16 *out, int64_t rows,
                          int64_t cols, int64_t ldg, int64_t ldu, cudaStream_t stream);

bool launch_silu_mul_bwd_rows(const __nv_bfloat16 *dy, const __nv_bfloat16 *g, const __nv_bfloat16 *u,
                              __nv_bfloat16 *dg, __nv_bfloat16 *du, int64_t rows, int64_t cols, int64_t ldy,
                              int64_t ldg, int64_t ldu, cudaStream_t stream);

void launch_silu_mul_bwd(const __nv_bfloat16 *dy, const __nv_bfloat16 *g, const __nv_bfloat16 *u,
                         __nv_bfloat16 *dg, __nv_bfloat16 *du, int64_t n, cudaStream_t stream);

using at::Tensor;

Tensor silu_mul(const Tensor &g, const Tensor &u) {
  TORCH_CHECK(g.is_cuda() && u.is_cuda(), "mlp_sm90.silu_mul: g and u must be CUDA tensors");
  TORCH_CHECK(g.scalar_type() == at::kBFloat16 && u.scalar_type() == at::kBFloat16,
              "mlp_sm90.silu_mul: g and u must be bfloat16, got ", g.dtype(), "/", u.dtype());
  TORCH_CHECK(g.sizes() == u.sizes(), "mlp_sm90.silu_mul: g and u must have the same shape, got ", g.sizes(),
              " vs ", u.sizes());
  const c10::cuda::CUDAGuard guard(g.device());
  // Column views of one packed [rows, 2I] GEMM output: read them in place instead of copying.
  if (g.dim() == 2 && g.stride(1) == 1 && u.stride(1) == 1 && !(g.is_contiguous() && u.is_contiguous())) {
    Tensor out = at::empty(g.sizes(), g.options().memory_format(at::MemoryFormat::Contiguous));
    if (launch_silu_mul_rows(reinterpret_cast<const __nv_bfloat16 *>(g.data_ptr()),
                             reinterpret_cast<const __nv_bfloat16 *>(u.data_ptr()),
                             reinterpret_cast<__nv_bfloat16 *>(out.data_ptr()), g.size(0), g.size(1), g.stride(0),
                             u.stride(0), at::cuda::getCurrentCUDAStream().stream()))
      return out;
  }
  Tensor gc = g.contiguous();
  Tensor uc = u.contiguous();
  Tensor out = at::empty_like(gc, at::MemoryFormat::Contiguous);
  launch_silu_mul(reinterpret_cast<const __nv_bfloat16 *>(gc.data_ptr()),
                  reinterpret_cast<const __nv_bfloat16 *>(uc.data_ptr()),
                  reinterpret_cast<__nv_bfloat16 *>(out.data_ptr()), gc.numel(),
                  at::cuda::getCurrentCUDAStream().stream());
  return out;
}

// The eager VJP of silu(g) * (u + 0.0) in one launch: (dg, du), both contiguous bf16.
std::tuple<Tensor, Tensor> silu_mul_bwd(const Tensor &dy, const Tensor &g, const Tensor &u) {
  TORCH_CHECK(dy.is_cuda() && g.is_cuda() && u.is_cuda(), "mlp_sm90.silu_mul_bwd: CUDA tensors required");
  TORCH_CHECK(dy.scalar_type() == at::kBFloat16 && g.scalar_type() == at::kBFloat16 &&
                  u.scalar_type() == at::kBFloat16,
              "mlp_sm90.silu_mul_bwd: bfloat16 operands required");
  TORCH_CHECK(dy.sizes() == g.sizes() && g.sizes() == u.sizes(), "mlp_sm90.silu_mul_bwd: shapes must match");
  const c10::cuda::CUDAGuard guard(g.device());
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  auto opts = g.options().memory_format(at::MemoryFormat::Contiguous);
  Tensor dg = at::empty(g.sizes(), opts), du = at::empty(g.sizes(), opts);
  auto P = [](const Tensor &t) { return reinterpret_cast<const __nv_bfloat16 *>(t.data_ptr()); };
  auto W = [](Tensor &t) { return reinterpret_cast<__nv_bfloat16 *>(t.data_ptr()); };
  if (g.dim() == 2 && dy.stride(1) == 1 && g.stride(1) == 1 && u.stride(1) == 1 &&
      launch_silu_mul_bwd_rows(P(dy), P(g), P(u), W(dg), W(du), g.size(0), g.size(1), dy.stride(0), g.stride(0),
                               u.stride(0), stream))
    return {dg, du};
  Tensor dyc = dy.contiguous(), gc = g.contiguous(), uc = u.contiguous();
  launch_silu_mul_bwd(P(dyc), P(gc), P(uc), W(dg), W(du), gc.numel(), stream);
  return {dg, du};
}

} // namespace mlp_sm90

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.attr("RECIPE_VERSION") = 1;
  m.def("silu_mul", &mlp_sm90::silu_mul, py::arg("g"), py::arg("u"));
  m.def("silu_mul_bwd", &mlp_sm90::silu_mul_bwd, py::arg("dy"), py::arg("g"), py::arg("u"));
}
