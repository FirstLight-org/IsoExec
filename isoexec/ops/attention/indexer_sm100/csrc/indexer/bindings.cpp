// SPDX-License-Identifier: MIT
// IsoExec sparse_sm100: torch bindings of the deterministic lightning indexer.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include "indexer.h"

namespace {

void indexer_impl(const torch::Tensor &q, const torch::Tensor &q_sf, const torch::Tensor &k,
                 const c10::optional<torch::Tensor> &k_sf, const torch::Tensor &weights,
                 const c10::optional<torch::Tensor> &ks, const torch::Tensor &ke,
                 const c10::optional<torch::Tensor> &block_table, torch::Tensor &out,
                 int64_t qb, int64_t span, int64_t grid, int64_t tree_leaves, int64_t tree_ieee_levels,
                 const c10::optional<torch::Tensor> &bt_rows = c10::nullopt) {
    using namespace sparse_sm100::indexer;
    TORCH_CHECK(q.is_cuda() && q.is_contiguous() && q.element_size() == 1, "q: e4m3 [rows, 64, 128]");
    const int64_t rows = q.numel() / (64 * 128);
    TORCH_CHECK(q_sf.scalar_type() == at::kInt && q_sf.is_contiguous() && q_sf.numel() == rows * 64);
    TORCH_CHECK(weights.scalar_type() == at::kBFloat16 && weights.stride(1) == 1 && weights.size(0) == rows &&
                weights.size(1) == 64);
    TORCH_CHECK(ke.scalar_type() == at::kInt && ke.is_contiguous() && ke.numel() == rows);
    TORCH_CHECK((out.scalar_type() == at::kBFloat16 || (tree_leaves > 0 && out.scalar_type() == at::kFloat)) &&
                out.stride(1) == 1 && out.size(0) == rows);
    TORCH_CHECK(tree_leaves == 0 || (tree_leaves <= 8 && (tree_leaves & (tree_leaves - 1)) == 0),
                "tree leaves: a power of two <= 8 (leaves of 4 heads)");
    TORCH_CHECK(tree_ieee_levels >= 0 && (tree_ieee_levels == 0 || (tree_leaves > 0 && (1 << tree_ieee_levels) <= tree_leaves)),
                "tree IEEE levels: at most log2(tree leaves)");
    IndexerParams p{};
    IndexerTensors t{};
    p.rows = (int)rows;
    p.qb = (int)qb;
    p.num_qblocks = (int)((rows + qb - 1) / qb);
    // span < 0: the kernel chooses each launch's span (power of two <= -span) from the live windows
    TORCH_CHECK(span != 0 && span >= -16, "span: 1..16 tiles, or -max for the in-kernel choice (max <= 16)");
    p.span_auto = span < 0 ? (int)(-span) : 0;
    if (span < 0) span = 1;
    p.span = (int)span;
    p.ks = nullptr;
    if (ks.has_value()) {
        TORCH_CHECK(ks->scalar_type() == at::kInt && ks->is_contiguous() && ks->numel() == rows);
        p.ks = ks->data_ptr<int>();
    }
    p.ke = ke.data_ptr<int>();
    p.weights = reinterpret_cast<const __nv_bfloat16 *>(weights.data_ptr());
    p.w_stride = weights.stride(0);
    p.tree_leaves = (int)tree_leaves;
    p.tree_ieee_levels = (int)tree_ieee_levels;
    p.out_f32 = out.scalar_type() == at::kFloat ? out.data_ptr<float>() : nullptr;
    p.out = out.scalar_type() == at::kFloat ? nullptr : reinterpret_cast<__nv_bfloat16 *>(out.data_ptr());
    p.out_stride = out.stride(0);
    const int64_t width = out.size(1);
    const int64_t tiles = (width + 127) / 128;
    p.splits = (int)((tiles + span - 1) / span);
    t.q = q.data_ptr();
    t.q_sf = q_sf.data_ptr();
    if (block_table.has_value()) {
        // page_rows = k.size(1) entries per page: 64 (default) or 128 (vLLM's ratio-1 index pages); the
        // block table holds one entry per page_rows keys
        const int64_t page_rows = k.dim() == 4 ? k.size(1) : 0;
        TORCH_CHECK(k.scalar_type() == at::kByte && k.dim() == 4 && (page_rows == 64 || page_rows == 128) &&
                    k.size(3) == 132 && k.stride(1) == 132 && k.stride(3) == 1 &&
                    (k.size(2) == 1 || k.stride(2) == 132) && k.stride(0) % 16 == 0 &&
                    k.stride(0) >= page_rows * 132, "paged cache: uint8 [pages, 64 or 128, 1, 132]");
        TORCH_CHECK(block_table->scalar_type() == at::kInt && block_table->stride(1) == 1 &&
                    (bt_rows.has_value() || block_table->size(0) == rows));
        p.bt_rows = nullptr;
        if (bt_rows.has_value()) {
            // row -> block-table row map (a packed step's token -> sequence): addressing only
            TORCH_CHECK(bt_rows->scalar_type() == at::kInt && bt_rows->is_contiguous() && bt_rows->numel() == rows,
                        "bt_rows: int32 [rows]");
            p.bt_rows = bt_rows->data_ptr<int>();
        }
        TORCH_CHECK(qb == 1, "the paged (decode) indexer takes one query row per block");
        TORCH_CHECK(span >= 1 && span <= 16, "paged items span at most 16 tiles (32 pages: one per lane)");
        p.paged = true;
        p.block_table = block_table->data_ptr<int>();
        p.bt_stride = block_table->stride(0);
        t.k = k.data_ptr();
        p.page_rows = (int)page_rows;
        t.k_sf = static_cast<uint8_t *>(k.data_ptr()) + page_rows * 128;
        t.num_pages = k.size(0);
        t.page_stride = k.stride(0);
    } else {
        TORCH_CHECK(k.is_contiguous() && k.element_size() == 1 && k.dim() == 2 && k.size(1) == 128);
        TORCH_CHECK(k_sf.has_value() && k_sf->scalar_type() == at::kInt && k_sf->numel() == k.size(0));
        p.paged = false;
        t.k = k.data_ptr();
        t.k_sf = k_sf->data_ptr();
        t.num_keys = k.size(0);
    }
    if (rows == 0) return;
    launch(p, t, (int)grid, c10::cuda::getCurrentCUDAStream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void indexer_fwd(const torch::Tensor &q, const torch::Tensor &q_sf, const torch::Tensor &k,
                 const c10::optional<torch::Tensor> &k_sf, const torch::Tensor &weights,
                 const c10::optional<torch::Tensor> &ks, const torch::Tensor &ke,
                 const c10::optional<torch::Tensor> &block_table, torch::Tensor &out,
                 int64_t qb, int64_t span, int64_t grid) {
    indexer_impl(q, q_sf, k, k_sf, weights, ks, ke, block_table, out, qb, span, grid, 0, 0);
}

void indexer_tree_fwd(const torch::Tensor &q, const torch::Tensor &q_sf, const torch::Tensor &k,
                      const c10::optional<torch::Tensor> &k_sf, const torch::Tensor &weights,
                      const c10::optional<torch::Tensor> &ks, const torch::Tensor &ke,
                      const c10::optional<torch::Tensor> &block_table, torch::Tensor &out,
                      int64_t qb, int64_t span, int64_t grid, int64_t tree_leaves) {
    indexer_impl(q, q_sf, k, k_sf, weights, ks, ke, block_table, out, qb, span, grid, tree_leaves, 0);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.attr("SPAN_AUTO") = true;   // span < 0: the kernel picks the item span from the live windows
    m.def("indexer_fwd", &indexer_fwd, "deterministic DSA lightning indexer");
    m.def("indexer_tree_fwd", &indexer_tree_fwd, "sm100_dsa_indexer_tree_v1: leaf-group head tree (all FTZ adds)");
    m.def("indexer_tree_ieee_fwd", [](const torch::Tensor &q, const torch::Tensor &q_sf, const torch::Tensor &k,
                                      const c10::optional<torch::Tensor> &k_sf, const torch::Tensor &weights,
                                      const c10::optional<torch::Tensor> &ks, const torch::Tensor &ke,
                                      const c10::optional<torch::Tensor> &block_table, torch::Tensor &out,
                                      int64_t qb, int64_t span, int64_t grid, int64_t tree_leaves, int64_t tree_ieee_levels) {
              indexer_impl(q, q_sf, k, k_sf, weights, ks, ke, block_table, out, qb, span, grid, tree_leaves, tree_ieee_levels);
          },
          "sm100_dsa_indexer_tree_v1 with its top tree_ieee_levels levels in IEEE adds (the TP leaf_tree completion)");
    m.def("indexer_tree_ieee_rows_fwd", &indexer_impl,
          "indexer_tree_ieee_fwd with a query row -> block-table row map (bt_rows; paged only)");
}
