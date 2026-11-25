// Torch binding of the sm90 projection-backward kernels: the fp32 wgrad-accumulate GEMM and the
// split-column gradient assembly.
#include <torch/extension.h>

#include <string>
#include <vector>

std::vector<std::string> wgrad_config_names();
void wgrad_accumulate(int64_t cid, torch::Tensor const &dy, torch::Tensor const &x, torch::Tensor &main_grad,
                      int64_t swizzle, int64_t raster);
at::Tensor assemble_columns(std::vector<at::Tensor> const &grads);
void wgrad_grouped_accumulate(torch::Tensor const &dy, torch::Tensor const &b, torch::Tensor const &ends,
                              torch::Tensor const &meta, torch::Tensor const &mg);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.attr("RECIPE_VERSION") = 1;
  m.def("config_names", &wgrad_config_names);
  m.def("accumulate", &wgrad_accumulate, py::arg("cid"), py::arg("dy"), py::arg("x"), py::arg("main_grad"),
        py::arg("swizzle") = 1, py::arg("raster") = 0);
  m.def("assemble_columns", &assemble_columns, py::arg("grads"));
  m.def("grouped_accumulate", &wgrad_grouped_accumulate, py::arg("dy"), py::arg("b"), py::arg("ends"), py::arg("meta"),
        py::arg("mg"));
}
