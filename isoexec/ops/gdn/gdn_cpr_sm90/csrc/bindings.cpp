// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>

void gdn_cpr_prefill(const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &,
                     const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &,
                     const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &,
                     const c10::optional<at::Tensor> &, at::Tensor, double, int64_t, int64_t,
                     const std::vector<at::Tensor> &, int64_t, int64_t);
void gdn_cpr_decode(const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &,
                    const at::Tensor &, const at::Tensor &, const at::Tensor &, at::Tensor, const at::Tensor &,
                    const at::Tensor &, at::Tensor, double, int64_t);
std::map<std::string, int64_t> gdn_cpr_record_layout();
std::tuple<at::Tensor, at::Tensor> gdn_cpr_bwd_qk_finish(const at::Tensor &, const at::Tensor &, const at::Tensor &,
                                                         const at::Tensor &, const at::Tensor &, const at::Tensor &,
                                                         const at::Tensor &);

namespace gdn_cpr {
void decode_rows(const at::Tensor &slots, const std::vector<at::Tensor> &maps, std::vector<at::Tensor> rows32s,
                 std::vector<at::Tensor> lane32s, const std::vector<at::Tensor> &clocks,
                 const std::vector<at::Tensor> &last_useds);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prefill", &gdn_cpr_prefill);
  m.def("decode", &gdn_cpr_decode);
  m.def("record_layout", &gdn_cpr_record_layout);   // byte offsets inside one (slot, head) state record
  m.def("bwd_qk_finish", &gdn_cpr_bwd_qk_finish);   // trainer backward epilogue (integration.py)
  m.def("decode_rows", &gdn_cpr::decode_rows);       // one step's rows, NULL lanes and LRU stamps for every pool
}
