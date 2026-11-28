// SPDX-License-Identifier: MIT
// IsoExec dsa_sm100: the device primitives of the fp32 emulation, compiled under the SAME profile as
// the kernels (tests only).  Each is written exactly as its kernel writes it: exp2f(x); x / y with y a
// runtime value (the root's O / L'); logf(L) + M / (float)M_LOG2E (the root's lse expression).
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <math_constants.h>
#include <cmath>

__global__ void k_exp2f(const float *x, float *y, int64_t n) {
    const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i < n) y[i] = exp2f(x[i]);
}
__global__ void k_div(const float *a, const float *b, float *y, int64_t n) {
    const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] / b[i];
}
__global__ void k_lse(const float *L, const float *M, float *y, int64_t n) {
    const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i < n) y[i] = logf(L[i]) + M[i] / (float)M_LOG2E;
}

static void launch(int64_t n, auto &&f) {
    if (n == 0) return;
    const int th = 256;
    f(dim3((unsigned)((n + th - 1) / th)), dim3(th), c10::cuda::getCurrentCUDAStream());
}

torch::Tensor exp2f_tensor(const torch::Tensor &x) {
    auto y = torch::empty_like(x);
    launch(x.numel(), [&](dim3 g, dim3 b, cudaStream_t s) { k_exp2f<<<g, b, 0, s>>>(x.data_ptr<float>(), y.data_ptr<float>(), x.numel()); });
    return y;
}
torch::Tensor div_tensor(const torch::Tensor &a, const torch::Tensor &b) {
    auto y = torch::empty_like(a);
    launch(a.numel(), [&](dim3 g, dim3 bl, cudaStream_t s) { k_div<<<g, bl, 0, s>>>(a.data_ptr<float>(), b.data_ptr<float>(), y.data_ptr<float>(), a.numel()); });
    return y;
}
torch::Tensor lse_tensor(const torch::Tensor &L, const torch::Tensor &M) {
    auto y = torch::empty_like(L);
    launch(L.numel(), [&](dim3 g, dim3 b, cudaStream_t s) { k_lse<<<g, b, 0, s>>>(L.data_ptr<float>(), M.data_ptr<float>(), y.data_ptr<float>(), L.numel()); });
    return y;
}

torch::Tensor mma_probe(const torch::Tensor &a0, const torch::Tensor &b0, const torch::Tensor &a1,
                        const torch::Tensor &b1, int64_t steps);
torch::Tensor mma_probe_ws(const torch::Tensor &a0, const torch::Tensor &b0, const torch::Tensor &a1,
                           const torch::Tensor &b1, int64_t steps);
torch::Tensor mma_chain(const torch::Tensor &a, const torch::Tensor &b, int64_t n, int64_t mode);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mma_probe", &mma_probe);
    m.def("mma_probe_ws", &mma_probe_ws);
    m.def("mma_chain", &mma_chain);
    m.def("exp2f", &exp2f_tensor);
    m.def("div", &div_tensor);
    m.def("lse", &lse_tensor);
}
