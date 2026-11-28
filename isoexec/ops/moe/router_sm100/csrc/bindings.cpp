// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>

namespace rt100 {
void select_cuda(const at::Tensor &, const c10::optional<at::Tensor> &,
                 const c10::optional<at::Tensor> &, int64_t, int64_t, int64_t, int64_t, int64_t,
                 double, at::Tensor, at::Tensor);
void select_tree_cuda(const at::Tensor &, const c10::optional<at::Tensor> &,
                      const c10::optional<at::Tensor> &, int64_t, int64_t, int64_t, int64_t, int64_t,
                      double, at::Tensor, at::Tensor);
void fused_cuda(const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor> &,
                const c10::optional<at::Tensor> &, int64_t, int64_t, int64_t, int64_t, int64_t,
                double, at::Tensor, at::Tensor, const c10::optional<at::Tensor> &, int64_t);
void tile_override_cuda(int64_t, int64_t, int64_t);
void bwd_select_cuda(const at::Tensor &, const c10::optional<at::Tensor> &,
                     const c10::optional<at::Tensor> &, int64_t, int64_t, int64_t, int64_t,
                     int64_t, double, const at::Tensor &, at::Tensor, at::Tensor);
void dx_cuda(const at::Tensor &, const at::Tensor &, const at::Tensor &, at::Tensor);
void dw_cuda(const at::Tensor &, const at::Tensor &, const at::Tensor &, int64_t, at::Tensor,
             at::Tensor, at::Tensor);
} // namespace rt100

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("select", &rt100::select_cuda);
  m.def("select_tree", &rt100::select_tree_cuda);
  m.def("fused", &rt100::fused_cuda);
  m.def("tile_override", &rt100::tile_override_cuda);
  m.def("bwd_select", &rt100::bwd_select_cuda);
  m.def("dx", &rt100::dx_cuda);
  m.def("dw", &rt100::dw_cuda);
}
