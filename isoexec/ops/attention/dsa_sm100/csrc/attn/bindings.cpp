// SPDX-License-Identifier: MIT
// IsoExec dsa_sm100: torch bindings of the sparse attention.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cmath>

#include "params.h"

namespace {

void sparse_fwd_impl(const torch::Tensor &q, const torch::Tensor &cache, const torch::Tensor &block_table,
                const c10::optional<torch::Tensor> &bt_row, const torch::Tensor &idx,
                const c10::optional<torch::Tensor> &topk_length, double softmax_scale, int64_t seg,
                torch::Tensor &out, torch::Tensor &lse, const c10::optional<torch::Tensor> &partials, int64_t recipe, const c10::optional<torch::Tensor> &sink,
                const float *fused_sink = nullptr) {
    using namespace dsa_sm100::attn;
    TORCH_CHECK(q.scalar_type() == at::kBFloat16 && q.dim() == 3 && (q.size(2) == 512 || q.size(2) == 576) &&
                q.stride(2) == 1, "q: bf16 [T, h_q, 512 | 576]");
    const int64_t dqk = q.size(2);
    TORCH_CHECK(q.stride(1) % 8 == 0 && q.stride(0) % 8 == 0, "q rows must be 16-byte aligned");
    const bool kv_bf16 = cache.scalar_type() == at::kBFloat16;
    if (kv_bf16) {
        TORCH_CHECK(cache.dim() == 2 && cache.size(1) == dqk && cache.is_contiguous() &&
                    (reinterpret_cast<uintptr_t>(cache.data_ptr()) & 127) == 0, "kv: bf16 [S, D_QK] (D_QK = q's), 128-byte aligned");
    } else {
        TORCH_CHECK(dqk == 512, "D_QK = 576 is built for a bf16 kv only (the 656-byte fp8 reader is a gap)");
        TORCH_CHECK(cache.scalar_type() == at::kByte && cache.is_contiguous() && cache.size(-1) == 528 &&
                    cache.size(1) == 64, "cache: uint8 [pages, 64, 1, 528]");
        TORCH_CHECK(block_table.scalar_type() == at::kInt && block_table.stride(1) == 1);
    }
    TORCH_CHECK(idx.scalar_type() == at::kInt && idx.dim() == 2 && idx.stride(1) == 1 && idx.size(0) == q.size(0));
    const int64_t K = idx.size(1);
    const int64_t k_cap = dqk == 512 ? 2112 : 2048;   // 2112: glm_kpool_sm100's 2051-slot lists (bf16 kv or fp8 cache)
    TORCH_CHECK(K % 64 == 0 && K > 0 && K <= k_cap, "K must be a multiple of the 64-slot tile, at most 2048 (2112 at D_QK 512)");
    TORCH_CHECK(seg % 64 == 0 && K % seg == 0, "the group size must hold whole tiles and divide K");
    const int64_t G = K / seg;
    TORCH_CHECK(G >= 1 && G <= 8, "the combine folds at most 8 groups");
    TORCH_CHECK((G == 1 && !sink.has_value()) || partials.has_value(), "G > 1 needs a partials buffer");
    TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.is_contiguous() && out.dim() == 3 && out.size(0) == q.size(0) &&
                out.size(1) == q.size(1) && out.size(2) == 512, "out: bf16 [T, h_q, 512]");
    TORCH_CHECK(lse.scalar_type() == at::kFloat && lse.is_contiguous() && lse.numel() == q.size(0) * q.size(1));
    LeafParams p{};
    p.q = q.data_ptr();
    p.q_tok_stride = q.stride(0);
    p.q_head_stride = q.stride(1);
    p.num_tokens = (int)q.size(0);
    p.h_q = (int)q.size(1);
    p.cache = cache.data_ptr();
    p.num_pages = cache.size(0);
    p.kv_bf16 = kv_bf16 ? 1 : 0;
    p.kv_rows = kv_bf16 ? cache.size(0) : cache.size(0) * 64;
    // bf16 kv: flat rows, or (execution mode) paged 64-row blocks when a bt_row map is given
    const bool bf16_paged = kv_bf16 && bt_row.has_value();
    if (bf16_paged) {
        TORCH_CHECK(block_table.scalar_type() == at::kInt && block_table.stride(1) == 1 && cache.size(0) % 64 == 0,
                    "paged bf16 kv: int32 block table, whole 64-row pages");
    }
    p.block_table = (kv_bf16 && !bf16_paged) ? nullptr : block_table.data_ptr<int>();
    p.bt_stride = (kv_bf16 && !bf16_paged) ? 0 : block_table.stride(0);
    p.bt_row = nullptr;
    if (kv_bf16 && !bf16_paged) {
        // no block table
    } else if (bt_row.has_value()) {
        TORCH_CHECK(bt_row->scalar_type() == at::kInt && bt_row->is_contiguous() && bt_row->numel() == q.size(0));
        p.bt_row = bt_row->data_ptr<int>();
    } else {
        TORCH_CHECK(block_table.size(0) == q.size(0), "one block-table row per token unless bt_row is given");
    }
    p.idx = idx.data_ptr<int>();
    TORCH_CHECK(idx.stride(0) % 2 == 0 && idx.stride(1) == 1 && (reinterpret_cast<uintptr_t>(idx.data_ptr()) & 7) == 0,
                "idx rows must be contiguous and 8-byte aligned (the translators load int2)");
    p.idx_stride = idx.stride(0);
    p.topk = (int)K;
    p.topk_length = nullptr;
    if (topk_length.has_value()) {
        TORCH_CHECK(topk_length->scalar_type() == at::kInt && topk_length->is_contiguous() &&
                    topk_length->numel() == q.size(0));
        p.topk_length = topk_length->data_ptr<int>();
    }
    TORCH_CHECK(recipe == 1 || recipe == 2, "recipe: 1 (sm100_dsa_sparse_segtree_v1) or 2 (sm100_dsa_sparse_v2)");
    p.recipe = (int)recipe;
    p.dqk = (int)dqk;
    p.seg = (int)seg;
    p.scale_log2 = float(softmax_scale * M_LOG2E);      // mla_sm100's host-side product, one rounding
    p.out = out.data_ptr();
    p.lse = lse.data_ptr<float>();
    p.partials = nullptr;
    p.fold = 0;
    p.sink = fused_sink;          // only sparse_sink_fused_fwd sets it (G == 1, no partials, no root launch)
    if (G > 1 || sink.has_value()) {
        TORCH_CHECK(partials->scalar_type() == at::kFloat && partials->is_contiguous() &&
                    partials->numel() >= q.size(0) * G * q.size(1) * 520, "partials: fp32 [T, G, h_q, 520]");
        p.partials = partials->data_ptr<float>();
    }
    if (sink.has_value()) {
        TORCH_CHECK(kv_bf16, "sink path requires BF16 KV");
        TORCH_CHECK(sink->is_cuda() && sink->device() == q.device() && sink->scalar_type() == at::kFloat &&
                    sink->is_contiguous() && sink->dim() == 1 && sink->size(0) == q.size(1), "sink: fp32 [H] on q device");
    }
    if (p.num_tokens == 0) return;
    auto stream = c10::cuda::getCurrentCUDAStream();
    launch_leaf_h64(p, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (sink.has_value()) {
        launch_sink_combine(p.partials, (int)G, p.h_q, (int)(q.size(0) * q.size(1)), out.data_ptr(),
                            lse.data_ptr<float>(), sink->data_ptr<float>(), stream);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else if (G > 1) {
        launch_combine(p.partials, (int)G, p.h_q, (int)(q.size(0) * q.size(1)), out.data_ptr(), lse.data_ptr<float>(), stream);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

void sparse_fwd(const torch::Tensor &q, const torch::Tensor &cache, const torch::Tensor &block_table,
                const c10::optional<torch::Tensor> &bt_row, const torch::Tensor &idx,
                const c10::optional<torch::Tensor> &topk_length, double softmax_scale, int64_t seg,
                torch::Tensor &out, torch::Tensor &lse, const c10::optional<torch::Tensor> &partials, int64_t recipe) {
    sparse_fwd_impl(q, cache, block_table, bt_row, idx, topk_length, softmax_scale, seg, out, lse, partials, recipe, c10::nullopt);
}
// sm100_dsa_sparse_sink_v1 with the root fused into the G == 1 leaf epilogue: no fp32 partials, one launch.
void sparse_sink_fused_fwd(const torch::Tensor &q, const torch::Tensor &kv, const torch::Tensor &idx,
                           double softmax_scale, torch::Tensor &out, torch::Tensor &lse, int64_t recipe,
                           const torch::Tensor &sink) {
    TORCH_CHECK(kv.scalar_type() == at::kBFloat16, "the fused sink root serves a BF16 kv");
    TORCH_CHECK(sink.is_cuda() && sink.device() == q.device() && sink.scalar_type() == at::kFloat &&
                sink.is_contiguous() && sink.dim() == 1 && sink.size(0) == q.size(1), "sink: fp32 [H] on q device");
    auto dummy = torch::empty({1, 1}, q.options().dtype(at::kInt));
    sparse_fwd_impl(q, kv, dummy, c10::nullopt, idx, c10::nullopt, softmax_scale, idx.size(1), out, lse,
                    c10::nullopt, recipe, c10::nullopt, sink.data_ptr<float>());
}

void sparse_sink_fwd(const torch::Tensor &q, const torch::Tensor &cache, const torch::Tensor &block_table,
                const c10::optional<torch::Tensor> &bt_row, const torch::Tensor &idx,
                const c10::optional<torch::Tensor> &topk_length, double softmax_scale, int64_t seg,
                torch::Tensor &out, torch::Tensor &lse, const c10::optional<torch::Tensor> &partials, int64_t recipe, const torch::Tensor &sink) {
    sparse_fwd_impl(q, cache, block_table, bt_row, idx, topk_length, softmax_scale, seg, out, lse, partials, recipe, sink);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_sink_fwd", &sparse_sink_fwd, "DSA single-round learned-sink root");
    m.def("sparse_sink_fused_fwd", &sparse_sink_fused_fwd, "learned-sink root fused into the G == 1 leaf");
    m.def("sink_grad", &dsa_sm100::attn::sink_grad, "owned head sink gradient, no atomics");
    m.def("sparse_fwd", &sparse_fwd, "deterministic DSA sparse attention (leaf + root)");
}
