// SPDX-License-Identifier: MIT
// IsoExec dsa_sm100: torch bindings of the deterministic top-k.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include "topk.h"

namespace {

void topk_rows_fwd(const torch::Tensor &scores, const c10::optional<torch::Tensor> &ks,
                   const torch::Tensor &ke, torch::Tensor &out, int64_t k, int64_t cluster,
                   int64_t threads, int64_t stash_keys, int64_t algo, int64_t n_lo, int64_t n_hi,
                   int64_t cmax_width, int64_t solo_below, const c10::optional<torch::Tensor> &rows_out,
                   const c10::optional<torch::Tensor> &rows_bt, const c10::optional<torch::Tensor> &rows_seq,
                   int64_t rows_per) {
    TORCH_CHECK(scores.is_cuda() && scores.dim() == 2 && scores.stride(1) == 1, "scores [rows, W] row-major");
    TORCH_CHECK(scores.scalar_type() == at::kBFloat16 || scores.scalar_type() == at::kFloat,
                "scores must be bf16 or fp32");
    TORCH_CHECK(ke.scalar_type() == at::kInt && ke.is_contiguous() && ke.numel() == scores.size(0));
    TORCH_CHECK(out.scalar_type() == at::kInt && out.is_contiguous() && out.size(0) == scores.size(0) &&
                out.size(1) == k);
    TORCH_CHECK(stash_keys % 8 == 0 && stash_keys >= 0);
    const bool fp32 = scores.scalar_type() == at::kFloat;
    dsa_sm100::topk::TopkParams p{};
    p.scores = scores.data_ptr();
    p.ks = nullptr;
    if (ks.has_value()) {
        TORCH_CHECK(ks->scalar_type() == at::kInt && ks->is_contiguous() && ks->numel() == scores.size(0));
        p.ks = ks->data_ptr<int>();
    }
    p.ke = ke.data_ptr<int>();
    p.out = out.data_ptr<int>();
    p.stride = scores.stride(0);
    p.rows = (int)scores.size(0);
    p.k = (int)k;
    p.stash_keys = (int)stash_keys;
    {   // chunks of the widest per-CTA key range of this launch (from the row width: no host sync)
        const int64_t width = cmax_width > 0 ? std::min<int64_t>(cmax_width, scores.size(1)) : scores.size(1);
        const int64_t chunks = (width + 7) / 8 + 1;
        p.cmax_chunks = (int)(((chunks + cluster - 1) / cluster + 8 + 7) / 8 * 8);
        if (solo_below > 0) {   // a solo row's whole window sits on rank 0
            const int64_t solo_chunks = (std::min<int64_t>(solo_below, scores.size(1)) + 7) / 8 + 1;
            p.cmax_chunks = std::max<int>(p.cmax_chunks, (int)((solo_chunks + 8 + 7) / 8 * 8));
        }
    }
    TORCH_CHECK(solo_below >= 0 && (solo_below == 0 || (cluster > 1 && algo == 0 && !fp32)),
                "solo rows: the bf16 cluster kernel only");
    p.solo_below = (int)solo_below;
    p.rows_out = nullptr; p.rows_bt = nullptr; p.rows_seq = nullptr; p.rows_bt_stride = 0; p.rows_per = 64;
    if (rows_out.has_value()) {
        TORCH_CHECK(algo == 0 && !fp32, "the rows output is written by the bf16 kernel");
        TORCH_CHECK(rows_out->scalar_type() == at::kInt && rows_out->is_contiguous() && rows_out->size(0) == scores.size(0) &&
                    rows_out->size(1) == k, "rows_out: int32 [rows, k]");
        TORCH_CHECK(rows_bt.has_value() && rows_bt->scalar_type() == at::kInt && rows_bt->dim() == 2 && rows_bt->stride(1) == 1);
        TORCH_CHECK(rows_per >= 64 || rows_per == 1);
        p.rows_out = rows_out->data_ptr<int>();
        p.rows_bt = rows_bt->data_ptr<int>();
        p.rows_bt_stride = rows_bt->stride(0);
        p.rows_per = (int)rows_per;
        if (rows_seq.has_value()) {
            TORCH_CHECK(rows_seq->scalar_type() == at::kInt && rows_seq->is_contiguous() && rows_seq->numel() == scores.size(0));
            p.rows_seq = rows_seq->data_ptr<int>();
        }
    }
    TORCH_CHECK(n_lo >= 0 && n_hi > n_lo && n_hi <= INT32_MAX);
    p.n_lo = (int)n_lo;
    p.n_hi = (int)n_hi;
    const int64_t esz = scores.element_size();
    p.vec_ok = ((reinterpret_cast<uintptr_t>(scores.data_ptr()) % 16) == 0) && ((p.stride * esz) % 16 == 0);
    if (p.rows == 0) return;
    TORCH_CHECK(dsa_sm100::topk::launch(p, fp32, (int)cluster, (int)threads, (int)algo, c10::cuda::getCurrentCUDAStream()),
                "no top-k instantiation for cluster=", cluster, " threads=", threads, " algo=", algo);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void topk_band_fwd(const torch::Tensor &scores, const c10::optional<torch::Tensor> &ks,
                   const torch::Tensor &ke, torch::Tensor &out, int64_t k, int64_t cluster,
                   int64_t threads, int64_t stash_keys, int64_t algo, int64_t n_lo, int64_t n_hi,
                   int64_t cmax_width) {
    topk_rows_fwd(scores, ks, ke, out, k, cluster, threads, stash_keys, algo, n_lo, n_hi, cmax_width, 0,
                  c10::nullopt, c10::nullopt, c10::nullopt, 64);
}

void topk_fwd(const torch::Tensor &scores, const c10::optional<torch::Tensor> &ks,
              const torch::Tensor &ke, torch::Tensor &out, int64_t k, int64_t cluster,
              int64_t threads, int64_t stash_keys, int64_t algo) {
    topk_band_fwd(scores, ks, ke, out, k, cluster, threads, stash_keys, algo, 0, INT32_MAX, 0);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("topk_fwd", &topk_fwd, "deterministic DSA top-k");
    m.def("topk_band_fwd", &topk_band_fwd,
          "deterministic DSA top-k of the rows whose window length is in [n_lo, n_hi) (cmax_width > 0: the widest window)");
    m.def("topk_rows_fwd", &topk_rows_fwd,
          "topk_band_fwd + solo rows below solo_below on the cluster kernel + the selected ids as cache rows (rows_out)");
}
