// SPDX-License-Identifier: MIT
// IsoExec sparse_sm100: torch bindings of the index-score kernel on DeepGEMM's SM100 MQA logits (mqa.cu).
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include "mqa/mqa.h"

namespace {

using namespace sparse_sm100::mqa;

//: rows of one paged schedule: its shared memory holds two words per row (request starts, work prefix)
constexpr int64_t MAX_PAGED_ROWS = (232448 / 4 - 1024 / 32 - 1) / 2;

void check_q(const torch::Tensor &q, const torch::Tensor &q_sf, const torch::Tensor &w, int64_t &rows, int64_t &heads) {
    TORCH_CHECK(q.is_cuda() && q.element_size() == 1 && q.dim() == 3 && q.size(2) == HEAD_DIM && q.is_contiguous(),
                "q: e4m3 codes [rows, heads, 128], contiguous");
    rows = q.size(0);
    heads = q.size(1);
    TORCH_CHECK(heads == 32 || heads == 64, "index heads: 32 or 64 (one Q tile of 128 (row, head) pairs)");
    TORCH_CHECK(q_sf.scalar_type() == at::kInt && q_sf.is_contiguous() && q_sf.numel() == rows * heads,
                "q_sf: int32 [rows, heads] (four equal UE8M0 bytes)");
    TORCH_CHECK(w.scalar_type() == at::kBFloat16 && w.dim() == 2 && w.size(0) == rows && w.size(1) == heads &&
                    w.stride(1) == 1 && (w.stride(0) * 2) % 16 == 0,
                "weights: bf16 [rows, heads], 16-byte aligned rows");
    for (const auto *t : {&q, &q_sf, &w})
        TORCH_CHECK((reinterpret_cast<uintptr_t>(t->data_ptr()) & 15) == 0, "16-byte aligned operands (TMA)");
}

void paged_schedule_fwd(const torch::Tensor &ke, const torch::Tensor &indices, torch::Tensor &meta) {
    TORCH_CHECK(ke.is_cuda() && ke.scalar_type() == at::kInt && ke.is_contiguous() && ke.dim() == 1, "ke: int32 [rows]");
    const int64_t rows = ke.numel();
    TORCH_CHECK(rows > 0 && rows <= MAX_PAGED_ROWS, "paged index scores: 1..", MAX_PAGED_ROWS, " rows per schedule");
    TORCH_CHECK(indices.scalar_type() == at::kInt && indices.is_contiguous() && indices.numel() == rows &&
                    (reinterpret_cast<uintptr_t>(indices.data_ptr()) & 7) == 0,
                "indices: int32 [rows], 8-byte aligned (read two at a time)");
    TORCH_CHECK(meta.scalar_type() == at::kInt && meta.is_contiguous() && meta.numel() == 2 * (NUM_SMS + 1),
                "meta: int32 [NUM_SMS + 1, 2]");
    const c10::cuda::CUDAGuard guard(ke.device());
    paged_schedule((int)rows, ke.data_ptr<int>(), indices.data_ptr<int>(), meta.data_ptr<int>(),
                   c10::cuda::getCurrentCUDAStream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void paged_fwd(const torch::Tensor &q, const torch::Tensor &q_sf, const torch::Tensor &cache, const torch::Tensor &w,
               const torch::Tensor &ke, const torch::Tensor &block_table, const torch::Tensor &indices,
               const torch::Tensor &meta, torch::Tensor &out) {
    int64_t rows, heads;
    check_q(q, q_sf, w, rows, heads);
    const int64_t page_kv = cache.dim() == 4 ? cache.size(1) : 0;
    TORCH_CHECK(cache.scalar_type() == at::kByte && cache.dim() == 4 && (page_kv == 64 || page_kv == 128) &&
                    cache.size(2) == 1 && cache.size(3) == 132 && cache.stride(1) == 132 && cache.stride(3) == 1 &&
                    cache.stride(0) % 16 == 0 && cache.stride(0) >= page_kv * 132 &&
                    (reinterpret_cast<uintptr_t>(cache.data_ptr()) & 15) == 0,
                "paged index keys: uint8 [pages, 64 or 128, 1, 132]");
    TORCH_CHECK(ke.scalar_type() == at::kInt && ke.is_contiguous() && ke.numel() == rows, "ke: int32 [rows]");
    TORCH_CHECK(block_table.scalar_type() == at::kInt && block_table.dim() == 2 && block_table.stride(1) == 1,
                "block_table: int32 [sequences, pages]");
    TORCH_CHECK(indices.scalar_type() == at::kInt && indices.is_contiguous() && indices.numel() == rows,
                "indices: int32 [rows] (row -> block-table row)");
    TORCH_CHECK(meta.scalar_type() == at::kInt && meta.is_contiguous() && meta.numel() == 2 * (NUM_SMS + 1),
                "meta: int32 [NUM_SMS + 1, 2] (paged_schedule of the same ke and indices)");
    TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.dim() == 2 && out.size(0) == rows && out.stride(1) == 1,
                "out: bf16 [rows, width]");
    if (rows == 0) return;
    const c10::cuda::CUDAGuard guard(q.device());
    PagedArgs a{};
    a.rows = (int)rows;
    a.heads = (int)heads;
    a.page_kv = (int)page_kv;
    a.q = q.data_ptr();
    a.q_sf = q_sf.data_ptr<int>();
    a.cache = cache.data_ptr<uint8_t>();
    a.num_pages = cache.size(0);
    a.page_stride = cache.stride(0);
    a.weights = w.data_ptr();
    a.w_stride = w.stride(0);
    a.ke = ke.data_ptr<int>();
    a.block_table = block_table.data_ptr<int>();
    a.bt_stride = block_table.stride(0);
    a.indices = indices.data_ptr<int>();
    a.meta = meta.data_ptr<int>();
    a.out = out.data_ptr();
    a.out_stride = out.stride(0);
    paged_logits(a, c10::cuda::getCurrentCUDAStream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void contig_fwd(const torch::Tensor &q, const torch::Tensor &q_sf, const torch::Tensor &k, const torch::Tensor &k_sf,
                const torch::Tensor &w, const c10::optional<torch::Tensor> &ks, const torch::Tensor &ke,
                torch::Tensor &out) {
    int64_t rows, heads;
    check_q(q, q_sf, w, rows, heads);
    TORCH_CHECK(k.is_cuda() && k.element_size() == 1 && k.dim() == 2 && k.size(1) == HEAD_DIM && k.stride(1) == 1 &&
                    k.stride(0) == HEAD_DIM && (reinterpret_cast<uintptr_t>(k.data_ptr()) & 15) == 0,
                "k: e4m3 codes [keys, 128], dense rows, 16-byte aligned");
    TORCH_CHECK(k_sf.scalar_type() == at::kInt && k_sf.is_contiguous() && k_sf.numel() == k.size(0) &&
                    (reinterpret_cast<uintptr_t>(k_sf.data_ptr()) & 15) == 0,
                "k_sf: int32 [keys], 16-byte aligned");
    TORCH_CHECK(ke.scalar_type() == at::kInt && ke.is_contiguous() && ke.numel() == rows, "ke: int32 [rows]");
    if (ks.has_value())
        TORCH_CHECK(ks->scalar_type() == at::kInt && ks->is_contiguous() && ks->numel() == rows, "ks: int32 [rows]");
    const int64_t block_q = 128 / heads;
    // the last Q block's spare rows are written (with the last row's window): their storage must exist
    const int64_t padded = (rows + block_q - 1) / block_q * block_q;
    TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.dim() == 2 && out.size(0) == rows && out.stride(1) == 1 &&
                    (int64_t)out.storage().nbytes() >= (out.storage_offset() + padded * out.stride(0)) * 2,
                "out: bf16 [rows, width] whose storage holds the rows rounded up to a whole Q block (alloc_logits)");
    TORCH_CHECK(k.size(0) > 0, "no keys");
    if (rows == 0) return;
    const c10::cuda::CUDAGuard guard(q.device());
    ContigArgs a{};
    a.rows = (int)rows;
    a.heads = (int)heads;
    a.num_keys = (int)k.size(0);
    a.q = q.data_ptr();
    a.q_sf = q_sf.data_ptr<int>();
    a.weights = w.data_ptr();
    a.w_stride = w.stride(0);
    a.k = k.data_ptr();
    a.k_sf = k_sf.data_ptr<int>();
    a.ks = ks.has_value() ? ks->data_ptr<int>() : nullptr;
    a.ke = ke.data_ptr<int>();
    a.out = out.data_ptr();
    a.out_stride = out.stride(0);
    logits(a, c10::cuda::getCurrentCUDAStream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

//: sequences of one key_ranges launch: its shared memory holds one word per sequence (48 KiB without opt-in)
constexpr int64_t MAX_GATHER_SEQS = 8192;

void key_ranges_fwd(const torch::Tensor &ke, const torch::Tensor &rows_seq, int64_t width, torch::Tensor &ks,
                    torch::Tensor &ke_out, torch::Tensor &offsets) {
    TORCH_CHECK(ke.is_cuda() && ke.scalar_type() == at::kInt && ke.is_contiguous() && ke.dim() == 1,
                "ke: int32 [rows]");
    const int64_t rows = ke.numel();
    TORCH_CHECK(rows_seq.scalar_type() == at::kInt && rows_seq.is_contiguous() && rows_seq.numel() == rows,
                "rows_seq: int32 [rows]");
    for (const auto *t : {&ks, &ke_out})
        TORCH_CHECK(t->scalar_type() == at::kInt && t->is_contiguous() && t->numel() == rows, "ks / ke: int32 [rows]");
    const int64_t seqs = offsets.numel() - 1;
    TORCH_CHECK(offsets.scalar_type() == at::kInt && offsets.is_contiguous() && seqs >= 1 && seqs <= MAX_GATHER_SEQS,
                "offsets: int32 [seqs + 1], 1 <= seqs <= ", MAX_GATHER_SEQS);
    TORCH_CHECK(width >= 0 && width < (int64_t(1) << 31) / seqs, "width x seqs must fit int32 key offsets");
    if (rows == 0) return;
    const c10::cuda::CUDAGuard guard(ke.device());
    key_ranges((int)rows, (int)seqs, (int)width, ke.data_ptr<int>(), rows_seq.data_ptr<int>(), ks.data_ptr<int>(),
               ke_out.data_ptr<int>(), offsets.data_ptr<int>(), c10::cuda::getCurrentCUDAStream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gather_keys_fwd(const torch::Tensor &cache, const torch::Tensor &block_table, const torch::Tensor &offsets,
                     torch::Tensor &codes, torch::Tensor &sf) {
    const int64_t page_kv = cache.dim() == 4 ? cache.size(1) : 0;
    TORCH_CHECK(cache.is_cuda() && cache.scalar_type() == at::kByte && cache.dim() == 4 &&
                    (page_kv == 64 || page_kv == 128) && cache.size(2) == 1 && cache.size(3) == 132 &&
                    cache.stride(1) == 132 && cache.stride(3) == 1 && cache.stride(0) % 16 == 0 &&
                    cache.stride(0) >= page_kv * 132 && (reinterpret_cast<uintptr_t>(cache.data_ptr()) & 15) == 0,
                "paged index keys: uint8 [pages, 64 or 128, 1, 132]");
    const int64_t seqs = offsets.numel() - 1;
    TORCH_CHECK(offsets.scalar_type() == at::kInt && offsets.is_contiguous() && seqs >= 1 && seqs <= MAX_GATHER_SEQS,
                "offsets: int32 [seqs + 1] (key_ranges)");
    TORCH_CHECK(block_table.scalar_type() == at::kInt && block_table.dim() == 2 && block_table.stride(1) == 1 &&
                    block_table.size(0) >= seqs,
                "block_table: int32 [>= seqs, pages]");
    TORCH_CHECK(codes.scalar_type() == at::kByte && codes.dim() == 2 && codes.size(1) == HEAD_DIM &&
                    codes.is_contiguous() && (reinterpret_cast<uintptr_t>(codes.data_ptr()) & 15) == 0,
                "codes: uint8 [capacity, 128], contiguous, 16-byte aligned");
    TORCH_CHECK(sf.scalar_type() == at::kInt && sf.is_contiguous() && sf.numel() == codes.size(0),
                "sf: int32 [capacity]");
    const c10::cuda::CUDAGuard guard(cache.device());
    GatherArgs a{};
    a.cache = cache.data_ptr<uint8_t>();
    a.page_stride = cache.stride(0);
    a.page_kv = (int)page_kv;
    a.block_table = block_table.data_ptr<int>();
    a.bt_stride = block_table.stride(0);
    a.seqs = (int)seqs;
    a.offsets = offsets.data_ptr<int>();
    a.capacity = codes.size(0);
    a.codes = codes.data_ptr<uint8_t>();
    a.sf = sf.data_ptr<int>();
    gather_keys(a, c10::cuda::getCurrentCUDAStream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.attr("NUM_SMS") = NUM_SMS;
    m.attr("SPLIT_KV") = SPLIT_KV;
    m.attr("MAX_PAGED_ROWS") = MAX_PAGED_ROWS;
    m.def("paged_schedule", &paged_schedule_fwd, "the paged index-score schedule (DeepGEMM paged MQA logits metadata)");
    m.def("paged_fwd", &paged_fwd, "index scores over paged FP8 index keys (DeepGEMM SM100 paged MQA logits)");
    m.def("fwd", &contig_fwd, "index scores over contiguous FP8 index keys (DeepGEMM SM100 MQA logits)");
    m.attr("MAX_GATHER_SEQS") = MAX_GATHER_SEQS;
    m.def("key_ranges", &key_ranges_fwd, "per-row key ranges [ks, ke) of the rows' sequences packed contiguous");
    m.def("gather_keys", &gather_keys_fwd, "the rows' sequences' paged index keys packed contiguous (key_ranges)");
    m.def("smem_bytes",[](int64_t heads) { return (int64_t)paged_smem_bytes((int)heads); });
}
