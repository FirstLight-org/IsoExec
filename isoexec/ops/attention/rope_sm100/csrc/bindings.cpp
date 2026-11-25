// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>

#include <optional>
#include <vector>

void table_cuda(const at::Tensor &, at::Tensor, int64_t);
std::vector<at::Tensor> rope_apply(const at::Tensor &,
                                   const std::optional<at::Tensor> &,
                                   const at::Tensor &, const at::Tensor &,
                                   int64_t, int64_t, bool, bool, bool, bool);
std::vector<at::Tensor>
qknorm_rope_forward(const at::Tensor &, const at::Tensor &, const at::Tensor &,
                    const at::Tensor &, const std::optional<at::Tensor> &,
                    const std::optional<at::Tensor> &, int64_t, int64_t,
                    double, bool, bool, bool);
std::vector<at::Tensor>
qknorm_rope_backward(const at::Tensor &, const at::Tensor &, const at::Tensor &,
                     const at::Tensor &, const at::Tensor &, const at::Tensor &,
                     const at::Tensor &, const std::optional<at::Tensor> &,
                     const std::optional<at::Tensor> &, int64_t, int64_t, bool,
                     bool, int64_t);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("table", &table_cuda);
  m.def("rope", &rope_apply);
  m.def("qknorm_rope_forward", &qknorm_rope_forward);
  m.def("qknorm_rope_backward", &qknorm_rope_backward);
}
