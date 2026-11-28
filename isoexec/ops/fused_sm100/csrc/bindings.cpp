// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>

namespace fused_glue {
void scatter_rows_cuda(const std::vector<at::Tensor> &srcs, const std::vector<at::Tensor> &dsts,
                       const at::Tensor &rows, const at::Tensor &cols, int64_t col_offset);
void embed_expand_cuda(const at::Tensor &weight, const at::Tensor &ids, at::Tensor out, at::Tensor pre);
void argmax_rows_cuda(const at::Tensor &x, at::Tensor out, at::Tensor scratch);
void mul_rows_cuda(const at::Tensor &y, const at::Tensor &w, at::Tensor out);
} // namespace fused_glue

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("scatter_rows", &fused_glue::scatter_rows_cuda);
  m.def("embed_expand", &fused_glue::embed_expand_cuda);
  m.def("argmax_rows", &fused_glue::argmax_rows_cuda);
  m.def("mul_rows", &fused_glue::mul_rows_cuda);
}
