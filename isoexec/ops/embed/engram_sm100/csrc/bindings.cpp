// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>

namespace engram {
void hash_cuda(const at::Tensor&, const at::Tensor&, const c10::optional<at::Tensor>&,
               const c10::optional<at::Tensor>&, const at::Tensor&, const at::Tensor&, const at::Tensor&, int64_t,
               at::Tensor, at::Tensor, const c10::optional<at::Tensor>&, const c10::optional<at::Tensor>&,
               const c10::optional<at::Tensor>&, const c10::optional<at::Tensor>&);
void lookup_fp8_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor, at::Tensor);
void lookup_fp8_quant_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor, at::Tensor, at::Tensor);
void quantize_table_cuda(const at::Tensor&, const c10::optional<at::Tensor>&, const c10::optional<at::Tensor>&,
                         const c10::optional<at::Tensor>&, at::Tensor, at::Tensor, at::Tensor);
void lookup_master_cuda(const at::Tensor&, const at::Tensor&, at::Tensor, at::Tensor);
void gate_fwd_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                   const c10::optional<at::Tensor>&, at::Tensor, at::Tensor, double, double);
void gate_bwd_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                   const c10::optional<at::Tensor>&, const at::Tensor&, at::Tensor, at::Tensor, at::Tensor,
                   at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, double, double);
void table_grad_cuda(const at::Tensor&, const at::Tensor&, int64_t, at::Tensor, at::Tensor, at::Tensor, at::Tensor);
void scatter_rows_cuda(at::Tensor, const at::Tensor&, const at::Tensor&, const at::Tensor&);
}  // namespace engram

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("hash", &engram::hash_cuda, pybind11::arg("ids"), pybind11::arg("tmap"), pybind11::arg("mask"),
        pybind11::arg("state"), pybind11::arg("primes"), pybind11::arg("offsets"), pybind11::arg("mult"),
        pybind11::arg("pad_id"), pybind11::arg("out"), pybind11::arg("err"), pybind11::arg("new_state"),
        pybind11::arg("magic"), pybind11::arg("raw_state") = pybind11::none(), pybind11::arg("seg_start") = pybind11::none());
  m.def("lookup_fp8", &engram::lookup_fp8_cuda);
  m.def("lookup_fp8_quant", &engram::lookup_fp8_quant_cuda);
  m.def("quantize_table", &engram::quantize_table_cuda);
  m.def("lookup_master", &engram::lookup_master_cuda);
  m.def("gate_fwd", &engram::gate_fwd_cuda);
  m.def("gate_bwd", &engram::gate_bwd_cuda);
  m.def("table_grad", &engram::table_grad_cuda);
  m.def("scatter_rows", &engram::scatter_rows_cuda);
}
