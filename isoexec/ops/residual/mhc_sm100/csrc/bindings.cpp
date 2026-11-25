// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>

namespace mhc_host {
std::vector<at::Tensor> split_forward_cuda(const at::Tensor &, const at::Tensor &, const at::Tensor &,
                                           const at::Tensor &, const c10::optional<at::Tensor> &, int64_t, double,
                                           double, int64_t, bool, int64_t, int64_t);
std::vector<at::Tensor> split_norm_forward_cuda(const at::Tensor &, const at::Tensor &, const at::Tensor &,
                                                const at::Tensor &, const at::Tensor &, double, double, int64_t, double,
                                                const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &,
                                                const c10::optional<at::Tensor> &);
at::Tensor merge_forward_cuda(const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, bool);
at::Tensor collapse_forward_cuda(const at::Tensor &, const at::Tensor &);
at::Tensor collapse_backward_dx_cuda(const at::Tensor &, const at::Tensor &);
at::Tensor dcolx_cuda(const at::Tensor &, const at::Tensor &);
std::vector<at::Tensor> merge_backward_cuda(const at::Tensor &, const at::Tensor &, const at::Tensor &,
                                            const at::Tensor &, const at::Tensor &, bool);
std::vector<at::Tensor> split_backward_cuda(const at::Tensor &, const at::Tensor &, const at::Tensor &,
                                            const at::Tensor &, const at::Tensor &, const at::Tensor &,
                                            const c10::optional<at::Tensor> &, const at::Tensor &, const at::Tensor &,
                                            const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &,
                                            double, int64_t, int64_t);
void deepseek_rows_cuda(const at::Tensor &, const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &,
                        const c10::optional<at::Tensor> &, const at::Tensor &, const at::Tensor &, double,
                        const c10::optional<at::Tensor> &, at::Tensor &, const c10::optional<at::Tensor> &,
                        const c10::optional<at::Tensor> &, int64_t);
void deepseek_rows_cuda(const at::Tensor &, const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &,
                        const c10::optional<at::Tensor> &, const at::Tensor &, const at::Tensor &, double,
                        const c10::optional<at::Tensor> &, at::Tensor &, const c10::optional<at::Tensor> &,
                        const c10::optional<at::Tensor> &, int64_t, const c10::optional<at::Tensor> &);
at::Tensor split_proj_cuda(const at::Tensor &, const at::Tensor &, int64_t, int64_t);
void gather_rows_cuda(const at::Tensor &, const std::vector<at::Tensor> &, std::vector<at::Tensor> &);
std::vector<at::Tensor> split_tail_cuda(const at::Tensor &, const at::Tensor &, const at::Tensor &, double, double, int64_t);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("split_forward", &mhc_host::split_forward_cuda);
  m.def("gather_rows", &mhc_host::gather_rows_cuda);
  m.def("split_proj", &mhc_host::split_proj_cuda, pybind11::arg("x"), pybind11::arg("fn"), pybind11::arg("groups"),
        pybind11::arg("pieces") = 3);
  m.def("split_tail", &mhc_host::split_tail_cuda);
  m.def("split_norm_forward", &mhc_host::split_norm_forward_cuda);
  m.def("split_backward", &mhc_host::split_backward_cuda);
  m.def("collapse_forward", &mhc_host::collapse_forward_cuda);
  m.def("collapse_backward_dx", &mhc_host::collapse_backward_dx_cuda);
  m.def("dcolx", &mhc_host::dcolx_cuda);
  m.def("merge_forward", &mhc_host::merge_forward_cuda);
  m.def("merge_backward", &mhc_host::merge_backward_cuda);
  m.def("deepseek_rows", static_cast<void (*)(const at::Tensor &, const c10::optional<at::Tensor> &,
      const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &, const at::Tensor &, const at::Tensor &, double,
      const c10::optional<at::Tensor> &, at::Tensor &, const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &,
      int64_t)>(&mhc_host::deepseek_rows_cuda));
  // plus the activation scales of h in DeepGEMM's packed layout (no separate packing launch)
  m.def("deepseek_rows_packed", static_cast<void (*)(const at::Tensor &, const c10::optional<at::Tensor> &,
      const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &, const at::Tensor &, const at::Tensor &, double,
      const c10::optional<at::Tensor> &, at::Tensor &, const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &,
      int64_t, const c10::optional<at::Tensor> &)>(&mhc_host::deepseek_rows_cuda));
}
