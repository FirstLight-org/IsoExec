// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>

void sample_cuda(const at::Tensor &x, const c10::optional<at::Tensor> &temperature, int64_t seed,
                 int64_t offset, int64_t split, bool prune, const c10::optional<at::Tensor> &row_max,
                 at::Tensor pmax, at::Tensor pval, at::Tensor pidx, at::Tensor token);
void rank_cuda(const at::Tensor &x, const at::Tensor &rM, const at::Tensor &rL, const at::Tensor &lp_tok,
               const at::Tensor &token, const c10::optional<at::Tensor> &temperature, int64_t split,
               int64_t k, at::Tensor pcnt, at::Tensor ptop, at::Tensor ptopi, at::Tensor out_idx,
               at::Tensor out_lp, at::Tensor out_rank);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sample", &sample_cuda);
  m.def("rank", &rank_cuda);
}
