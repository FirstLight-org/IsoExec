#pragma once

#include <format>
#include <variant>

#include <pybind11/pybind11.h>
#include <torch/python.h>

#include "../utils/exception.hpp"
#include "../utils/layout.hpp"
#include "../utils/compatibility.hpp"
#include "gemm.hpp"

// IsoExec (pruned to BF16): only `bhr,hdr->bhd` on the SM100 BF16 GEMM family is vendored; the other
// expressions, `fp8_einsum`, and the SM90 / cuBLASLt / bmk_bnk_mn implementations are not part of this copy
#include "../jit_kernels/impls/sm100_bf16_gemm.hpp"

namespace deep_gemm::einsum {

static void bhr_hdr_bhd(const torch::Tensor& A, const torch::Tensor& B, const torch::Tensor& D,
                        const std::optional<torch::Tensor>& C,
                        const std::shared_ptr<EpilogueClass>& epilogue_class) {
    const auto [b , h  , r ] = get_shape<3>(A);
    const auto [h_, d  , r_] = get_shape<3>(B);
    const auto [b_, h__, d_] = get_shape<3>(D);
    DG_HOST_ASSERT(b == b_ and h == h_ and r == r_ and d == d_ and h == h__);

    DG_HOST_ASSERT(A.scalar_type() == torch::kBFloat16 and A.stride(2) == 1);
    DG_HOST_ASSERT(B.scalar_type() == torch::kBFloat16 and B.stride(2) == 1);
    DG_HOST_ASSERT(D.stride(2) == 1);

    // Early return for trivial cases
    if (h == 0 or gemm::early_return(b, d, r, D, C))
        return;

    const auto resolved_epilogue_class = resolve_epilogue_class(epilogue_class, C, D);

    // Dispatch implementation (IsoExec: SM100 DeepGEMM kernel only; no cuBLASLt or SM90 dispatch)
    const auto arch_major = jit->device.get_arch_major();
    if (arch_major == 10) {
        sm100_bf16_bhr_hdr_bhd(A, B, C, D, b, h, r, d, "nk", resolved_epilogue_class);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}

static void einsum(const std::string& expr,
                   const torch::Tensor& a,
                   const torch::Tensor& b,
                   const torch::Tensor& d,
                   const std::optional<torch::Tensor>& c,
                   const std::shared_ptr<EpilogueClass>& epilogue_class) {
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(b.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16 or d.scalar_type() == torch::kFloat);
    if (c.has_value()) {
        DG_HOST_ASSERT(d.scalar_type() == c->scalar_type());
    }

    // Some hardcoded Einstein sum kernels
    // TODO: support any expression
    // TODO: canonicalize expression
    if (expr == "bhr,hdr->bhd") {
        bhr_hdr_bhd(a, b, d, c, epilogue_class);
    } else {
        DG_HOST_UNREACHABLE(std::format("Unsupported einsum expression: {}", expr));
    }
}

static void register_apis(pybind11::module_& m) {
    m.def("einsum", &einsum,
          py::arg("expr"), py::arg("a"), py::arg("b"),
          py::arg("d"), py::arg("c") = std::nullopt,
          py::arg("epilogue") = nullptr);
}

} // namespace deep_gemm::einsum
