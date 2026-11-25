// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>

void varlen_fwd_cuda(at::Tensor out, const at::Tensor &x, const at::Tensor &w,
                     const c10::optional<at::Tensor> &bias, const at::Tensor &cu,
                     const c10::optional<at::Tensor> &state, const c10::optional<at::Tensor> &idx,
                     const c10::optional<at::Tensor> &has, int64_t null_block, int64_t silu_act,
                     int64_t tile, int64_t threads);

at::Tensor varlen_bwd_cuda(at::Tensor dx, const at::Tensor &x, const at::Tensor &dy,
                           const at::Tensor &w, const c10::optional<at::Tensor> &bias,
                           const at::Tensor &cu, bool silu_act, int64_t tile, int64_t threads);

void update_fwd_cuda(at::Tensor out, const at::Tensor &x, const at::Tensor &w,
                     const c10::optional<at::Tensor> &bias, at::Tensor state, const at::Tensor &idx,
                     int64_t null_block, int64_t silu_act, int64_t threads);

at::Tensor varlen_bwd2_cuda(at::Tensor dx, const at::Tensor &x, const at::Tensor &dy, const at::Tensor &w,
                            const c10::optional<at::Tensor> &bias, const at::Tensor &cu, bool silu_act,
                            int64_t tile, int64_t tb);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("varlen_fwd", &varlen_fwd_cuda);
  m.def("varlen_bwd", &varlen_bwd_cuda);
  m.def("varlen_bwd2", &varlen_bwd2_cuda);
  m.def("update_fwd", &update_fwd_cuda);
}
