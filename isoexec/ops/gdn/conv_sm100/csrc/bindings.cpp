// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>

void varlen_fwd_cuda(at::Tensor out, const at::Tensor &x, const at::Tensor &w,
                     const c10::optional<at::Tensor> &bias, const at::Tensor &cu,
                     const c10::optional<at::Tensor> &state, const c10::optional<at::Tensor> &idx,
                     const c10::optional<at::Tensor> &has, int64_t null_block, int64_t silu_act,
                     int64_t tile, int64_t threads, int64_t cpt, int64_t recipe);

at::Tensor varlen_bwd_cuda(at::Tensor dx, const at::Tensor &x, const at::Tensor &dy,
                           const at::Tensor &w, const c10::optional<at::Tensor> &bias,
                           const at::Tensor &cu, bool silu_act, int64_t tile, int64_t threads,
                           int64_t cpt);

void fold_dwdb_cuda(const at::Tensor &part, const c10::optional<at::Tensor> &dw,
                    const c10::optional<at::Tensor> &db, int64_t groups, int64_t variant);

void update_fwd_cuda(at::Tensor out, const at::Tensor &x, const at::Tensor &w,
                     const c10::optional<at::Tensor> &bias, at::Tensor state, const at::Tensor &idx,
                     int64_t null_block, int64_t silu_act, int64_t threads, int64_t cpt,
                     int64_t recipe);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("varlen_fwd", &varlen_fwd_cuda);
  m.def("varlen_bwd", &varlen_bwd_cuda);
  m.def("fold_dwdb", &fold_dwdb_cuda, py::arg("part"), py::arg("dw"), py::arg("db"), py::arg("groups"),
        py::arg("variant") = 0);
  m.def("update_fwd", &update_fwd_cuda);
}
