// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>

void gemm_cuda(const at::Tensor &, const at::Tensor &, const at::Tensor &,
               const at::Tensor &, const at::Tensor &, at::Tensor, int64_t,
               bool);
void plan_cuda(const at::Tensor &, at::Tensor, at::Tensor, int64_t);
void route_cuda(const at::Tensor &, at::Tensor, at::Tensor, at::Tensor,
                at::Tensor, at::Tensor);
void gather_cuda(const at::Tensor &, const at::Tensor &, const at::Tensor &,
                 at::Tensor, at::Tensor, int64_t);
void combine_cuda(const at::Tensor &, const at::Tensor &, const at::Tensor &,
                  at::Tensor);
void activation_cuda(const at::Tensor &, const at::Tensor &, at::Tensor);

void counts_plan_cuda(const at::Tensor &, at::Tensor, at::Tensor, at::Tensor,
                      int64_t);

void combine_root_rows_cuda(const at::Tensor &, const at::Tensor &, at::Tensor);

void gathered_fc1_cuda(const at::Tensor &, const at::Tensor &,
                       const at::Tensor &, const at::Tensor &,
                       const at::Tensor &, const at::Tensor &,
                       const at::Tensor &, at::Tensor, at::Tensor);

void small_routes_cuda(const at::Tensor &, const at::Tensor &, at::Tensor,
                       at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                       at::Tensor);
void route_metadata_cuda(const at::Tensor &, const at::Tensor &,
                         const at::Tensor &, at::Tensor, at::Tensor,
                         at::Tensor);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemm", &gemm_cuda);
  m.def("small_routes", &small_routes_cuda);
  m.def("route_metadata", &route_metadata_cuda);
  m.def("gathered_fc1", &gathered_fc1_cuda);
  m.def("plan", &plan_cuda);
  m.def("counts_plan", &counts_plan_cuda);
  m.def("combine_root_rows", &combine_root_rows_cuda);
  m.def("route", &route_cuda);
  m.def("gather", &gather_cuda);
  m.def("combine", &combine_cuda);
  m.def("activation", &activation_cuda);
}
