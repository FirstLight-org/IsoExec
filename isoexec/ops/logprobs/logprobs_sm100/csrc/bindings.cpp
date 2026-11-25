// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>

void leaf_states_cuda(const at::Tensor &x, int64_t leaves_local, at::Tensor mleaf, at::Tensor sleaf,
                      const c10::optional<at::Tensor> &temperature, const c10::optional<at::Tensor> &aleaf);
void entropy_rows_cuda(const at::Tensor &mleaf, const at::Tensor &sleaf, const at::Tensor &aleaf, at::Tensor ent);
void merge_rows_cuda(const at::Tensor &mleaf, const at::Tensor &sleaf, at::Tensor rM, at::Tensor rL,
                     const c10::optional<at::Tensor> &lse, const c10::optional<at::Tensor> &sampled,
                     const c10::optional<at::Tensor> &x, const c10::optional<at::Tensor> &target,
                     const c10::optional<at::Tensor> &sampled_raw, int64_t vocab,
                     const c10::optional<at::Tensor> &temperature);
void write_rows_cuda(at::Tensor out, const at::Tensor &x, const at::Tensor &rM, const at::Tensor &rL,
                     const c10::optional<at::Tensor> &temperature);
at::Tensor full_rows_cuda(const at::Tensor &x, const c10::optional<at::Tensor> &temperature);
void grad_rows_cuda(at::Tensor dx, const at::Tensor &x, const at::Tensor &lse, const at::Tensor &gout,
                    const at::Tensor &local_target, int64_t leaves_local);

std::tuple<int, int> schedule_cuda();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("schedule", &schedule_cuda);
  m.def("leaf_states", &leaf_states_cuda, py::arg("x"), py::arg("leaves_local"), py::arg("mleaf"), py::arg("sleaf"),
        py::arg("temperature"), py::arg("aleaf") = py::none());
  m.def("entropy_rows", &entropy_rows_cuda);
  m.def("merge_rows", &merge_rows_cuda);
  m.def("write_rows", &write_rows_cuda);
  m.def("full_rows", &full_rows_cuda);
  m.def("grad_rows", &grad_rows_cuda);
}
