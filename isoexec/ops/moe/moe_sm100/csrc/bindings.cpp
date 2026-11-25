// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>

namespace moe100 {
void score_cuda(const at::Tensor &, const at::Tensor &, at::Tensor);
void mechanics_cuda(const at::Tensor &, const c10::optional<at::Tensor> &, at::Tensor,
                    at::Tensor, double);
void plan_cuda(const at::Tensor &, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
               at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t);
void tiles_cuda(const at::Tensor &, at::Tensor, at::Tensor, int64_t, int64_t);
void gather_cuda(const at::Tensor &, const at::Tensor &, const at::Tensor &, at::Tensor,
                 at::Tensor, int64_t);
void activation_cuda(const at::Tensor &, const at::Tensor &, at::Tensor);
void combine_cuda(const at::Tensor &, const at::Tensor &, at::Tensor);
void fc1_glu_cuda(const at::Tensor &, const at::Tensor &, const at::Tensor &,
                  const at::Tensor &, const at::Tensor &, at::Tensor);
void fc1_cuda(const at::Tensor &, const at::Tensor &, const at::Tensor &,
              const at::Tensor &, at::Tensor);
void fc2_cuda(const at::Tensor &, const at::Tensor &, const at::Tensor &,
              const at::Tensor &, at::Tensor, int64_t);
} // namespace moe100

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("score", &moe100::score_cuda);
  m.def("mechanics", &moe100::mechanics_cuda);
  m.def("plan", &moe100::plan_cuda);
  m.def("tiles", &moe100::tiles_cuda);
  m.def("gather", &moe100::gather_cuda);
  m.def("activation", &moe100::activation_cuda);
  m.def("combine", &moe100::combine_cuda);
  m.def("fc1_glu", &moe100::fc1_glu_cuda);
  m.def("fc1", &moe100::fc1_cuda);
  m.def("fc2", &moe100::fc2_cuda);
}
