// SPDX-License-Identifier: Apache-2.0
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <tuple>

namespace mlp_sm100 {

void launch_silu_mul(const __nv_bfloat16 *g, const __nv_bfloat16 *u, __nv_bfloat16 *out, int64_t n,
                     int64_t width, cudaStream_t stream);

void launch_silu_mul_rows(const __nv_bfloat16 *g, int64_t sg, const __nv_bfloat16 *u, int64_t su,
                          __nv_bfloat16 *out, int64_t rows, int64_t width, cudaStream_t stream);

using at::Tensor;

// [rows, width] with unit column stride, a 16-byte aligned base and 16-byte aligned rows: read in place.
static bool strided_rows_ok(const Tensor &t) {
  return t.dim() == 2 && t.stride(1) == 1 && (reinterpret_cast<uintptr_t>(t.data_ptr()) & 15u) == 0 &&
         (t.size(0) <= 1 || (t.stride(0) % 8) == 0);
}

Tensor silu_mul(const Tensor &g, const Tensor &u) {
  TORCH_CHECK(g.is_cuda() && u.is_cuda(), "mlp_sm100.silu_mul: g and u must be CUDA tensors");
  TORCH_CHECK(g.scalar_type() == at::kBFloat16 && u.scalar_type() == at::kBFloat16,
              "mlp_sm100.silu_mul: g and u must be bfloat16 (HELD_DTYPE), got ", g.dtype(), "/", u.dtype());
  TORCH_CHECK(g.sizes() == u.sizes(), "mlp_sm100.silu_mul: g and u must have the same shape, got ",
              g.sizes(), " vs ", u.sizes());
  const c10::cuda::CUDAGuard guard(g.device());
  // Column windows of one wider row (the packed gate|up output): no copies, same per-element program.
  if (!(g.is_contiguous() && u.is_contiguous()) && g.dim() == 2 && g.size(1) % 8 == 0 && strided_rows_ok(g) &&
      strided_rows_ok(u)) {
    Tensor out = at::empty(g.sizes(), g.options().memory_format(at::MemoryFormat::Contiguous));
    const int64_t rows = g.size(0), width = g.size(1);
    launch_silu_mul_rows(reinterpret_cast<const __nv_bfloat16 *>(g.data_ptr()), rows > 1 ? g.stride(0) : width,
                         reinterpret_cast<const __nv_bfloat16 *>(u.data_ptr()), rows > 1 ? u.stride(0) : width,
                         reinterpret_cast<__nv_bfloat16 *>(out.data_ptr()), rows, width,
                         at::cuda::getCurrentCUDAStream().stream());
    return out;
  }
  Tensor gc = g.contiguous();
  Tensor uc = u.contiguous();
  Tensor out = at::empty_like(gc);
  int64_t n = gc.numel();
  auto stream = at::cuda::getCurrentCUDAStream();
  int64_t width = gc.dim() ? gc.size(-1) : n;
  launch_silu_mul(reinterpret_cast<const __nv_bfloat16 *>(gc.data_ptr()),
                  reinterpret_cast<const __nv_bfloat16 *>(uc.data_ptr()),
                  reinterpret_cast<__nv_bfloat16 *>(out.data_ptr()), n, width, stream.stream());
  return out;
}

} // namespace mlp_sm100

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  using namespace mlp_sm100;
  m.attr("RECIPE_VERSION") = 1;
  m.def("silu_mul", &silu_mul, py::arg("g"), py::arg("u"));
}
