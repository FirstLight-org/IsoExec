// SPDX-License-Identifier: Apache-2.0
// The vocab-parallel sampling head's extension (logprobs_sm90/_shard_extension.py; kernels in shard_sample.cu).
#include <torch/extension.h>

void shard_pass_cuda(const at::Tensor &x, int64_t leaves_local, int64_t col0, const at::Tensor &req_of_row,
                     const at::Tensor &req_temp, const at::Tensor &req_seed, const at::Tensor &row_pos,
                     at::Tensor part, const c10::optional<at::Tensor> &temperature);
void shard_finish_cuda(const at::Tensor &gathered, int64_t world, int64_t leaves_local, const at::Tensor &x,
                       at::Tensor sampled, at::Tensor logprob, at::Tensor count,
                       const c10::optional<at::Tensor> &temperature, int64_t nan_col);
at::Tensor gumbel_probe_cuda(const at::Tensor &x, const at::Tensor &seed, const at::Tensor &pos, const at::Tensor &col);
void gumbel_range_probe_cuda(int64_t first, at::Tensor out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("shard_pass", &shard_pass_cuda);
  m.def("shard_finish", &shard_finish_cuda);
  m.def("gumbel_probe", &gumbel_probe_cuda);
  m.def("gumbel_range_probe", &gumbel_range_probe_cuda);
}
