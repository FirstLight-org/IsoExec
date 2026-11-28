#pragma once

#include <cmath>
#include <unordered_map>
#include <c10/cuda/CUDAGraphsC10Utils.h>

#include "../utils/compatibility.hpp"
#include "../jit_kernels/impls/sm100_bf16_mega_gate.hpp"
#include "../utils/layout.hpp"

namespace deep_gemm::mega_gate {

static const torch::Tensor& get_score_barriers(const torch::TensorOptions& options) {
    const auto stream = at::cuda::getCurrentCUDAStream();
    DG_HOST_ASSERT(options.device() == stream.device());
    static std::unordered_map<c10::cuda::CUDAStream, torch::Tensor> score_barriers_by_stream;
    auto& score_barriers = score_barriers_by_stream[stream];
    if (not score_barriers.defined()) {
        // Warm up each stream before capture so one-time zeroing is not replayed with the graph.
        DG_HOST_ASSERT(c10::cuda::currentStreamCaptureStatusMayInitCtx() == c10::cuda::CaptureStatus::None);
        score_barriers = torch::zeros(
            {mega_gate_layout::kNumMaxTokenBlocks, mega_gate_layout::kScoreBarrierLineBytes},
            options.dtype(torch::kByte));
    }
    return score_barriers;
}

// IsoExec: experts per warp lane of the router's selection (`router_sm100` `pick_epl`): the smallest served EPL with
// 32 * EPL >= E
static int get_num_experts_per_lane(const int& num_experts) {
    constexpr int kServed[6] = {2, 4, 8, 12, 16, 32};
    for (const auto& epl: kServed)
        if (32 * epl >= num_experts)
            return epl;
    return 0;
}

/**
 * IsoExec router program on the BF16 MegaGate: the gate GEMM with IsoExec's router selection in one launch, every
 * output bit equal to `isoexec.ops.moe.router_sm100.route()` (the 4-segment logits tree, then `select_warp`).
 *
 * Args:
 *     x                              [T,H]               BF16, H % 256 == 0 (4 segments of whole 64-wide K blocks)
 *     weight                         [E,H]               BF16, E <= 512, E % 4 == 0
 *     bias                           [E] FP32 | None     (selection only; required by mode 0)
 *     mask                           [T] bool | None     (false -> id -1, weight 0 in every slot)
 *     out_idx                        [T,K+S] int32       routed slots in ascending expert id, then E..E+S-1
 *     out_w                          [T,K+S] FP32        unbiased normalized weights x routed_scale, shared 1.0
 *     select_mode                    0 = sigmoid (grouped), 1 = sqrt-softplus
 *     num_topk                       K in {1, 2, 4, 6, 8, 16}
 *     num_shared_experts             S (mode 1 only)
 *     num_groups, num_topk_groups    the group-limited selection (mode 0), 1 / 1 otherwise
 *     routed_scaling_factor          float
 *     forced_num_split_k             0, or 1 / 2 / 4: restrict the ownership (tests; never the bits)
 *     forced_num_mma_ctas            0, or 1 / 2: restrict the ownership (tests; never the bits)
 */
static void bf16_mega_gate_isoexec(const torch::Tensor& x,
                                   const torch::Tensor& weight,
                                   const std::optional<torch::Tensor>& bias,
                                   const std::optional<torch::Tensor>& mask,
                                   const torch::Tensor& out_idx,
                                   const torch::Tensor& out_w,
                                   const int& select_mode,
                                   const int& num_topk,
                                   const int& num_shared_experts,
                                   const int& num_groups,
                                   const int& num_topk_groups,
                                   const float& routed_scaling_factor,
                                   const int& forced_num_split_k,
                                   const int& forced_num_mma_ctas) {
    constexpr int kHiddenAlignment = static_cast<int>(mega_gate_layout::kNumSegments * mega_gate_layout::BLOCK_K);
    constexpr int kNumMaxRoutedExperts = 512, kNumMaxSlots = 32;
    const auto [num_tokens, hidden] = get_shape<2>(x);
    const auto num_routed_experts = static_cast<int>(weight.size(0));
    DG_HOST_ASSERT(hidden > 0 and hidden % kHiddenAlignment == 0);
    DG_HOST_ASSERT(num_routed_experts >= 2 and num_routed_experts <= kNumMaxRoutedExperts and
                   num_routed_experts % static_cast<int>(mega_gate_layout::kNumExpertsPerLaneVector) == 0);

    const auto device = x.device();
    const auto check_dense = [&](const torch::Tensor& tensor, const at::IntArrayRef& shape, const torch::ScalarType& dtype) {
        DG_HOST_ASSERT(tensor.sizes() == shape and tensor.scalar_type() == dtype);
        DG_HOST_ASSERT(tensor.is_contiguous());
        DG_HOST_ASSERT(tensor.device() == device);
    };
    check_dense(x, {num_tokens, hidden}, torch::kBFloat16);
    check_dense(weight, {num_routed_experts, hidden}, torch::kBFloat16);

    // The router's geometry (`router_sm100` `check_geometry`)
    DG_HOST_ASSERT(select_mode == 0 or select_mode == 1);
    DG_HOST_ASSERT(num_topk == 1 or num_topk == 2 or num_topk == 4 or num_topk == 6 or num_topk == 8 or num_topk == 16);
    DG_HOST_ASSERT(num_topk <= num_routed_experts);
    DG_HOST_ASSERT(num_shared_experts >= 0 and num_topk + num_shared_experts <= kNumMaxSlots);
    DG_HOST_ASSERT(num_groups >= 1 and num_topk_groups >= 1 and num_topk_groups <= num_groups);
    DG_HOST_ASSERT(std::isfinite(routed_scaling_factor));
    const auto num_experts_per_lane = get_num_experts_per_lane(num_routed_experts);
    DG_HOST_ASSERT(num_experts_per_lane > 0);
    if (select_mode == 0) {
        DG_HOST_ASSERT(bias.has_value() and num_shared_experts == 0 and not mask.has_value());
        DG_HOST_ASSERT(num_routed_experts % num_groups == 0);
        if (num_topk_groups < num_groups) {
            const auto group_size = num_routed_experts / num_groups;
            const auto lanes_per_group = group_size / num_experts_per_lane;
            DG_HOST_ASSERT(group_size >= 2 and group_size % num_experts_per_lane == 0 and
                           (lanes_per_group & (lanes_per_group - 1)) == 0);
            DG_HOST_ASSERT(num_topk <= num_topk_groups * group_size);
        }
    } else {
        DG_HOST_ASSERT(num_groups == 1 and num_topk_groups == 1);
    }
    DG_HOST_ASSERT(forced_num_split_k == 0 or forced_num_split_k == 1 or forced_num_split_k == 2 or forced_num_split_k == 4);
    DG_HOST_ASSERT(forced_num_mma_ctas == 0 or forced_num_mma_ctas == 1 or forced_num_mma_ctas == 2);

    if (mask.has_value())
        check_dense(mask.value(), {num_tokens}, torch::kBool);
    if (bias.has_value())
        check_dense(bias.value(), {num_routed_experts}, torch::kFloat);
    const auto num_slots = num_topk + num_shared_experts;
    check_dense(out_idx, {num_tokens, num_slots}, torch::kInt);
    check_dense(out_w, {num_tokens, num_slots}, torch::kFloat);

    if (num_tokens == 0)
        return;
    DG_HOST_ASSERT(num_tokens <= static_cast<int>(mega_gate_layout::kNumMaxTokens));

    const auto arch_major = jit->device.get_arch_major();
    DG_HOST_ASSERT(arch_major == 10);
    sm100_bf16_mega_gate(x, weight, bias, mask, out_idx, out_w,
                         select_mode, num_experts_per_lane, num_shared_experts, num_groups, num_topk_groups,
                         routed_scaling_factor, get_score_barriers(x.options()),
                         num_tokens, hidden, num_routed_experts, num_topk,
                         forced_num_split_k, forced_num_mma_ctas);
}

// IsoExec: the launch plan the heuristic picks for a shape (ownership only), for tests and benchmarks
static std::tuple<int, int, int, int, int, int, int> get_mega_gate_isoexec_config(const int& num_tokens, const int& hidden,
                                                                                  const int& num_routed_experts,
                                                                                  const int& forced_num_split_k,
                                                                                  const int& forced_num_mma_ctas) {
    const auto config = get_mega_gate_config(num_tokens, hidden, num_routed_experts, runtime->get_num_sms(),
                                             forced_num_split_k, forced_num_mma_ctas);
    return {config.num_expert_groups, config.num_mma_ctas, config.num_split_k, config.block_tokens,
            config.num_launch_sms, config.num_gate_threads, config.num_stages};
}

static void register_apis(pybind11::module_& m) {
    m.def("bf16_mega_gate_isoexec", &bf16_mega_gate_isoexec,
          py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("mask"),
          py::arg("out_idx"), py::arg("out_w"),
          py::arg("select_mode"), py::arg("num_topk"), py::arg("num_shared_experts"),
          py::arg("num_groups"), py::arg("num_topk_groups"), py::arg("routed_scaling_factor"),
          py::arg("forced_num_split_k") = 0, py::arg("forced_num_mma_ctas") = 0);
    m.def("get_mega_gate_isoexec_config", &get_mega_gate_isoexec_config,
          py::arg("num_tokens"), py::arg("hidden"), py::arg("num_routed_experts"),
          py::arg("forced_num_split_k") = 0, py::arg("forced_num_mma_ctas") = 0);
}

} // namespace deep_gemm::mega_gate
