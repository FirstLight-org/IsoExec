// SPDX-License-Identifier: MIT
// Torch bindings of the IsoExec mla_sm100 forward: plan kernel, Blackwell leaf kernel, tree combine.
// Shapes and capacities are validated in fwd.py; this file checks what protects memory.
#include <cmath>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cutlass/bfloat16.h>

#include "params.h"
#include "leaf.h"
#include "plan.h"
#include "combine.h"

namespace {

using namespace mla_sm100::fwd;

void check_i32(const at::Tensor &t, const char *name) {
    TORCH_CHECK(t.is_cuda() && t.dtype() == torch::kInt32 && t.is_contiguous(), name,
                " must be a contiguous CUDA int32 tensor");
}

bool is_sm100a() {
    auto *prop = at::cuda::getCurrentDeviceProperties();
    return prop->major == 10 && prop->minor == 0;
}

void plan_cuda(const at::Tensor &cu_seqlens_q, const at::Tensor &seqused_k,
               at::Tensor items, at::Tensor heads, at::Tensor combine_rows, at::Tensor counts,
               at::Tensor scratch, int64_t h_q, int64_t seg_tokens, int64_t workers,
               int64_t granularity, int64_t target, int64_t item_overhead, int64_t fold_overhead,
               int64_t slot_cap, int64_t mb_cap, int64_t cost_cap, bool reverse, int64_t smem_ints) {
    TORCH_CHECK(is_sm100a(), "mla_sm100 requires an SM100a device");
    check_i32(cu_seqlens_q, "cu_seqlens_q");
    check_i32(seqused_k, "seqused_k");
    check_i32(items, "items");
    check_i32(heads, "heads");
    check_i32(combine_rows, "combine_rows");
    check_i32(counts, "counts");
    check_i32(scratch, "scratch");
    const int64_t b = seqused_k.size(0);
    TORCH_CHECK(cu_seqlens_q.numel() == b + 1, "cu_seqlens_q must be [B+1]");
    TORCH_CHECK(items.dim() == 2 && items.size(1) == ITEM_FIELDS, "items must be [cap, 8]");
    TORCH_CHECK(combine_rows.dim() == 2 && combine_rows.size(1) == COMBINE_FIELDS,
                "combine_rows must be [cap, 4]");
    TORCH_CHECK(counts.numel() >= NUM_COUNTS, "counts too small");
    TORCH_CHECK(seg_tokens > 0 && seg_tokens % 64 == 0, "seg_tokens must be a positive multiple of 64");
    TORCH_CHECK(workers > 0 && heads.numel() >= workers,
                "workers must be positive and heads must be [workers]");
    TORCH_CHECK(mb_cap >= 0 && scratch.numel() >= 9 * mb_cap + 3 * workers,
                "scratch must be [9*mb_cap + 3*workers]");
    TORCH_CHECK(granularity >= GRAN_AUTO && granularity <= GRAN_MIXED, "unknown granularity");
    at::cuda::CUDAGuard guard{(char)cu_seqlens_q.get_device()};
    PlanParams params = {
        (int)b, (int)h_q, (int)seg_tokens, (int)workers,
        (int)items.size(0), (int)std::min<int64_t>(slot_cap, 2147483647), (int)combine_rows.size(0),
        (int)mb_cap, (int)granularity, (int)std::min<int64_t>(target, 2147483647),
        (int)item_overhead, (int)fold_overhead, (int)cost_cap, (int)reverse, (int)smem_ints,
        cu_seqlens_q.data_ptr<int>(), seqused_k.data_ptr<int>(),
        (Item *)items.data_ptr<int>(), heads.data_ptr<int>(), combine_rows.data_ptr<int>(),
        counts.data_ptr<int>(), scratch.data_ptr<int>(), at::cuda::getCurrentCUDAStream().stream(),
    };
    run_mla_sm100_plan_kernel(params);
}

void fwd_cuda(const at::Tensor &q, const at::Tensor &kv_cache, const at::Tensor &block_table,
              const at::Tensor &cu_seqlens_q, const at::Tensor &seqused_k, const at::Tensor &items,
              const at::Tensor &heads, const at::Tensor &counts, const at::Tensor &combine_rows,
              at::Tensor out, at::Tensor lse, at::Tensor partials, at::Tensor stack,
              double softmax_scale, int64_t h_q, int64_t seg_tokens, int64_t workers,
              int64_t combine_entries, int64_t max_n, int64_t combine_config, int64_t dbg,
              int64_t recipe) {
    TORCH_CHECK(is_sm100a(), "mla_sm100 requires an SM100a device");
    TORCH_CHECK(recipe == 1 || recipe == 2,
                "recipe must be 1 (sm100_absorbed_segtree) or 2 (sm100_absorbed_segtree_v2)");
    TORCH_CHECK(q.is_cuda() && q.dtype() == torch::kBFloat16 && q.is_contiguous(),
                "q must be contiguous CUDA bf16");
    TORCH_CHECK(q.dim() == 3 && q.size(1) == h_q && q.size(2) == 576, "q must be [T, h_q, 576]");
    TORCH_CHECK(kv_cache.is_cuda() && kv_cache.dtype() == torch::kBFloat16, "kv_cache must be CUDA bf16");
    TORCH_CHECK(kv_cache.dim() == 4 && kv_cache.size(1) == 64 && kv_cache.size(2) == 1 &&
                kv_cache.size(3) == 576, "kv_cache must be [num_blocks, 64, 1, 576]");
    TORCH_CHECK(kv_cache.stride(-1) == 1, "kv_cache must have a contiguous last dimension");
    check_i32(block_table, "block_table");
    check_i32(cu_seqlens_q, "cu_seqlens_q");
    check_i32(seqused_k, "seqused_k");
    check_i32(items, "items");
    check_i32(heads, "heads");
    check_i32(counts, "counts");
    check_i32(combine_rows, "combine_rows");
    TORCH_CHECK(out.is_cuda() && out.dtype() == torch::kBFloat16 && out.is_contiguous() &&
                out.numel() == q.size(0) * h_q * 512, "out must be contiguous bf16 [T, h_q, 512]");
    TORCH_CHECK(lse.is_cuda() && lse.dtype() == torch::kFloat32 && lse.is_contiguous() &&
                lse.numel() == q.size(0) * h_q, "lse must be contiguous float32 [T, h_q]");
    TORCH_CHECK(partials.is_cuda() && partials.dtype() == torch::kFloat32 && partials.is_contiguous() &&
                partials.dim() == 2 && partials.size(0) >= 64 && partials.size(1) == PARTIAL_WIDTH,
                "partials must be contiguous float32 [>= 64, PARTIAL_WIDTH]");
    TORCH_CHECK(workers > 0 && heads.numel() >= workers,
                "workers must be positive and heads must be [workers]");
    TORCH_CHECK(stack.is_cuda() && stack.dtype() == torch::kFloat32 && stack.is_contiguous() &&
                stack.dim() == 4 && stack.size(0) >= workers && stack.size(1) >= 1 &&
                stack.size(2) == 64 && stack.size(3) == PARTIAL_WIDTH,
                "stack must be contiguous float32 [>= workers, depth, 64, PARTIAL_WIDTH]");
    TORCH_CHECK((1LL << stack.size(1)) >= max_n, "stack depth must cover the deepest tree");
    for (const at::Tensor *t : std::initializer_list<const at::Tensor *>{&q, &kv_cache, &out, &partials, &stack}) {
        TORCH_CHECK(((uintptr_t)t->data_ptr() & 15) == 0, "tensor data must be 16-byte aligned");
    }
    const int64_t total_rows = q.size(0) * h_q;
    if (total_rows == 0) return;

    at::cuda::CUDAGuard guard{(char)q.get_device()};
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    const bool pdl = combine_entries > 0 && !(dbg & 32);

    FwdParams params;
    params.h_q = (int)h_q;
    params.seg_tokens = (int)seg_tokens;
    params.total_rows = (int)total_rows;
    params.workers = (int)workers;
    params.pdl = pdl;
    params.recipe = (int)recipe;
    params.stack_depth = (int)stack.size(1);
    params.scale_softmax_log2 = float(softmax_scale * M_LOG2E);
    params.items = (Item *)items.data_ptr<int>();
    params.heads = heads.data_ptr<int>();
    params.counts = counts.data_ptr<int>();
    params.cu_seqlens_q = cu_seqlens_q.data_ptr<int>();
    params.seqused_k = seqused_k.data_ptr<int>();
    params.block_table = block_table.data_ptr<int>();
    params.block_table_stride = block_table.stride(0);
    params.out = out.data_ptr();
    params.lse = lse.data_ptr<float>();
    params.partials = partials.data_ptr<float>();
    params.stack = stack.data_ptr<float>();
    params.stream = stream;

    LeafTensors tensors = {
        q.data_ptr(), total_rows,
        kv_cache.data_ptr(), kv_cache.stride(1), kv_cache.stride(2), kv_cache.stride(0),
        kv_cache.size(0),
        out.data_ptr(), partials.data_ptr(), partials.size(0),
    };
    if (!(dbg & 16)) {
        run_mla_sm100_leaf_kernel<cutlass::bfloat16_t>(params, tensors);
    }

    if (combine_entries > 0) {
        CombineParams cparams = {
            (int)pdl, (int)((dbg >> 16) & 255), combine_rows.data_ptr<int>(), counts.data_ptr<int>(),
            params.partials, out.data_ptr(), lse.data_ptr<float>(), stream,
        };
        run_mla_sm100_combine_kernel<cutlass::bfloat16_t>(cparams, (int)combine_entries, (int)max_n,
                                                         (int)combine_config);
    }
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "IsoExec mla_sm100 forward (deterministic absorbed MLA on Blackwell)";
    m.def("plan", &plan_cuda);
    m.def("fwd", &fwd_cuda);
}
