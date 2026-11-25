// SPDX-License-Identifier: BSD-3-Clause
#include <torch/extension.h>

#include <tuple>
#include <vector>

namespace gqa_sm100 {

std::tuple<at::Tensor, at::Tensor> forward(
        const at::Tensor& q, const at::Tensor& k_cache, const at::Tensor& v_cache,
        const at::Tensor& cu_seqlens_q, const at::Tensor& seqused_k,
        const c10::optional<at::Tensor>& block_table,
        const c10::optional<at::Tensor>& cu_seqlens_k,
        double softmax_scale, int64_t seg, int64_t max_seqlen_q, int64_t max_seqlen_k,
        int64_t granularity, int64_t waves_override, int64_t max_run_override, int64_t leaf,
        int64_t decode_split, const c10::optional<at::Tensor>& out);


#ifdef GQA_PLAN_BINDING
void set_plan_single(bool single);
std::vector<at::Tensor> last_plan();
#endif


}  // namespace gqa_sm100

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gqa_sm100::forward,
          "Deterministic SM100 causal GQA forward (segments anchored at key 0, canonical tree)",
          py::arg("q"), py::arg("k_cache"), py::arg("v_cache"), py::arg("cu_seqlens_q"),
          py::arg("seqused_k"), py::arg("block_table"), py::arg("cu_seqlens_k"),
          py::arg("softmax_scale"), py::arg("seg"), py::arg("max_seqlen_q"),
          py::arg("max_seqlen_k"), py::arg("granularity"), py::arg("waves_override"),
          py::arg("max_run_override"), py::arg("leaf") = 0, py::arg("decode_split") = 0,
          py::arg("out") = py::none());
#ifdef GQA_PLAN_BINDING
    m.def("set_plan_single", &gqa_sm100::set_plan_single, "test only: single-block plan");
    m.def("last_plan", &gqa_sm100::last_plan, "test only: (items, combine_tiles, counts)");
#endif
}
