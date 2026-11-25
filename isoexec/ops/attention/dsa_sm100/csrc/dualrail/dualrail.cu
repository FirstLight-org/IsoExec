// SPDX-License-Identifier: MIT
// IsoExec dsa_sm100: host launch + torch binding of the dual-rail sparse-attention leaf (leaf.cuh).
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <cstdlib>

#include "leaf.cuh"
#include "split.cuh"
#include "persist.cuh"

namespace dsa_sm100::dualrail {

static CUtensorMap bf16_kv_map(const torch::Tensor &kv) {
    CUtensorMap m;
    const cuuint64_t dims[2] = {512, (cuuint64_t)kv.size(0)};
    const cuuint64_t strides[1] = {1024};
    const cuuint32_t box[2] = {64, 1};
    const cuuint32_t estr[2] = {1, 1};
    const CUresult r = cuTensorMapEncodeTiled(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, kv.data_ptr(), dims, strides, box, estr,
                                              CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                              CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled (bf16 kv) failed: ", int(r));
    return m;
}

// The record rows of the fp8 cache as the 2-D gather map {token words, rows} with box height 1 (every slot is one
// TMA row gather; an invalid slot's coordinate -1 is out of bounds, a zero row). Whole 64-token pages give the map
// they always had; a cache view of any row count (a pipeline stage whose allocation has no whole page after its
// last record, engine.record_rows) gives the same map bounded at its last row: the same bytes at every valid
// coordinate, the same zero fill outside. Host-side extent only; no kernel instruction changes.
template<typename F, typename L>
static CUtensorMap quant_kv_map(const torch::Tensor &kv, const char *name) {
    TORCH_CHECK(kv.numel() % F::BYTES_PER_TOKEN == 0, name, ": whole ", (int)F::BYTES_PER_TOKEN, "-byte records");
    const int64_t tokens = kv.numel() / F::BYTES_PER_TOKEN;
    TORCH_CHECK(tokens > 0, name, ": empty cache");
    if (tokens % 64 == 0)
        return make_kv_quant_part_tensor_map<KVCachePart<F, 1, 0>, L::template RAW_TOKEN_SMEM_STRIDE<F>>(
            name, kv.data_ptr(), (int)(tokens / 64), (int64_t)64 * F::BYTES_PER_TOKEN, F::BYTES_PER_TOKEN);
    return make_kv_quant_part_tensor_map<KVCachePart<F, 1, 0>, L::template RAW_TOKEN_SMEM_STRIDE<F>>(
        name, kv.data_ptr(), (int)tokens, (int64_t)F::BYTES_PER_TOKEN, F::BYTES_PER_TOKEN);
}

// Fused RoPE (fused_rope.cuh): cos / sin fp32 [P, 32] contiguous, pos int32 [R]; q / o select the Q RoPE prologue and
// the output's inverse-RoPE epilogue.
static RopeParams rope_params(const c10::optional<torch::Tensor> &cosv, const c10::optional<torch::Tensor> &sinv,
                              const c10::optional<torch::Tensor> &pos, bool q, bool o, int64_t rows) {
    RopeParams rp{};
    if (!q && !o) return rp;
    TORCH_CHECK(cosv && sinv && pos, "fused rope needs cos, sin and positions");
    TORCH_CHECK(cosv->scalar_type() == at::kFloat && cosv->is_contiguous() && cosv->dim() == 2 && cosv->size(1) == 32);
    TORCH_CHECK(sinv->scalar_type() == at::kFloat && sinv->is_contiguous() && sinv->sizes() == cosv->sizes());
    TORCH_CHECK(pos->scalar_type() == at::kInt && pos->is_contiguous() && pos->numel() == rows, "rope positions: int32 [rows]");
    rp.cosv = cosv->data_ptr<float>();
    rp.sinv = sinv->data_ptr<float>();
    rp.pos = pos->data_ptr<int>();
    rp.q = q;
    rp.o = o;
    return rp;
}

//: Programmatic-dependent launch (the kernel waits for its predecessor in griddepcontrol.wait before touching
//: memory): launch latency only, no bit. ISOEXEC_SMALL_PDL=0 launches plainly.
template<typename K, typename... Args>
static void launch_pdl(K kernel, dim3 grid, dim3 block, size_t smem, cudaStream_t stream, const Args &...args) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    const char *e = std::getenv("ISOEXEC_SMALL_PDL");
    cfg.attrs = attr;
    cfg.numAttrs = (e && e[0] == '0') ? 0 : 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, args...));
}

template<bool KV_BF16, bool EXTRA>
static void launch(const Params &p, const torch::Tensor &kv, const c10::optional<torch::Tensor> &extra_kv, cudaStream_t stream) {
    using L = Leaf<KV_BF16, EXTRA>;
    auto shape_Q = make_shape(p.h_q, L::D_Q, p.rows);
    auto tma_Q = cute::make_tma_copy(SM90_TMA_LOAD{},
        make_tensor(make_gmem_ptr((bf16 *)p.q), make_layout(shape_Q, make_stride(p.q_head_stride, _1{}, p.q_row_stride))),
        typename L::SmemLayoutQ_SW128{});
    auto shape_O = make_shape(p.h_q, L::D_V, p.rows);
    auto tma_O = cute::make_tma_copy(SM90_TMA_STORE{},
        make_tensor(make_gmem_ptr((bf16 *)p.out), make_layout(shape_O, make_stride(p.out_head_stride, _1{}, p.out_row_stride))),
        typename L::SmemLayoutOBuf_TMA{});
    CUtensorMap kv_map = KV_BF16 ? bf16_kv_map(kv) : quant_kv_map<typename L::OrigKVFormat, L>(kv, "kv cache");
    CUtensorMap extra_map{};
    if constexpr (EXTRA) extra_map = quant_kv_map<typename L::ExtraKVFormat, L>(*extra_kv, "extra kv cache");
    typename L::template TmaParams<decltype(shape_Q), decltype(tma_Q), decltype(shape_O), decltype(tma_O)> tp = {
        shape_Q, tma_Q, shape_O, tma_O, kv_map, extra_map};
    auto kernel = &dualrail_kernel<L, decltype(tp), KV_BF16, EXTRA>;
    constexpr size_t smem = sizeof(typename L::SharedMemoryPlan);
    static_assert(smem <= 227 * 1024);
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    launch_pdl(kernel, dim3(p.rows, 1, 1), dim3(L::NUM_THREADS, 1, 1), smem, stream, p, tp);
}

// The persistent plan (persist.cuh): grid = rows, CTAs take further rows through cluster launch control. Same recipe as
// launch<KV_BF16, EXTRA>, same bits (tests/test_dualrail_persist_gpu.py).
template<bool KV_BF16, bool EXTRA>
static void launch_persist(const Params &p, const torch::Tensor &kv, const c10::optional<torch::Tensor> &extra_kv, cudaStream_t stream) {
    using L = Persist<KV_BF16, EXTRA>;
    auto shape_Q = make_shape(p.h_q, L::D_Q, p.rows);
    auto tma_Q = cute::make_tma_copy(SM90_TMA_LOAD{},
        make_tensor(make_gmem_ptr((bf16 *)p.q), make_layout(shape_Q, make_stride(p.q_head_stride, _1{}, p.q_row_stride))),
        typename L::SmemLayoutQ{});
    auto shape_O = make_shape(p.h_q, L::D_V, p.rows);
    auto tma_O = cute::make_tma_copy(SM90_TMA_STORE{},
        make_tensor(make_gmem_ptr((bf16 *)p.out), make_layout(shape_O, make_stride(p.out_head_stride, _1{}, p.out_row_stride))),
        typename L::template SmemLayoutOTiles<1>{});
    using LL = Leaf<false, EXTRA>;             // fp8: the one-CTA leaf's raw-record maps (P2)
    CUtensorMap kv_map = KV_BF16 ? bf16_kv_map(kv) : quant_kv_map<typename LL::OrigKVFormat, LL>(kv, "kv cache");
    CUtensorMap extra_map{};
    if constexpr (EXTRA) extra_map = quant_kv_map<typename LL::ExtraKVFormat, LL>(*extra_kv, "extra kv cache");
    static_assert(LL::template RAW_TOKEN_SMEM_STRIDE<typename LL::OrigKVFormat> == L::RAW_STRIDE);
    typename L::template TmaParams<decltype(shape_Q), decltype(tma_Q), decltype(shape_O), decltype(tma_O)> tp = {
        shape_Q, tma_Q, shape_O, tma_O, kv_map, extra_map};
    auto kernel = &dualrail_persist_kernel<L, decltype(tp), KV_BF16, EXTRA>;
    constexpr size_t smem = sizeof(typename L::SharedMemoryPlan);
    static_assert(smem <= 227 * 1024);
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    launch_pdl(kernel, dim3(p.rows, 1, 1), dim3(L::NUM_THREADS, 1, 1), smem, stream, p, tp);
}

// The split plan (split.cuh): qk_kernel over (row, tile chunk), then pv_kernel over (row, column CTA), the latter launched
// programmatically dependent (PDL) on the former. Same recipe as launch<false, EXTRA>, same bits.
template<bool EXTRA, int NC>
static void launch_split(const Params &p, const SplitParams &sp, const torch::Tensor &kv, const c10::optional<torch::Tensor> &extra_kv,
                         cudaStream_t stream) {
    using L = Leaf<false, EXTRA>;
    auto shape_Q = make_shape(p.h_q, L::D_Q, p.rows);
    auto tma_Q = cute::make_tma_copy(SM90_TMA_LOAD{},
        make_tensor(make_gmem_ptr((bf16 *)p.q), make_layout(shape_Q, make_stride(p.q_head_stride, _1{}, p.q_row_stride))),
        typename L::SmemLayoutQ_SW128{});
    auto shape_O = make_shape(p.h_q, L::D_V, p.rows);
    auto tma_O = cute::make_tma_copy(SM90_TMA_STORE{},
        make_tensor(make_gmem_ptr((bf16 *)p.out), make_layout(shape_O, make_stride(p.out_head_stride, _1{}, p.out_row_stride))),
        typename L::SmemLayoutOBuf_TMA{});
    CUtensorMap kv_map = quant_kv_map<typename L::OrigKVFormat, L>(kv, "kv cache");
    CUtensorMap extra_map{};
    if constexpr (EXTRA) extra_map = quant_kv_map<typename L::ExtraKVFormat, L>(*extra_kv, "extra kv cache");
    using TP = typename L::template TmaParams<decltype(shape_Q), decltype(tma_Q), decltype(shape_O), decltype(tma_O)>;
    TP tp = {shape_Q, tma_Q, shape_O, tma_O, kv_map, extra_map};

    using QP = QK<EXTRA>;
    auto k1 = &qk_kernel<QP, TP, EXTRA>;
    constexpr size_t smem1 = sizeof(typename QP::SharedMemoryPlan);
    static_assert(smem1 <= 227 * 1024);
    C10_CUDA_CHECK(cudaFuncSetAttribute(k1, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem1));
    launch_pdl(k1, dim3(p.rows, (sp.T + sp.tpc - 1) / sp.tpc, 1), dim3(384, 1, 1), smem1, stream, p, sp, tp);

    using PP = PV<NC>;
    auto k2 = &pv_kernel<PP, TP>;
    constexpr size_t smem2 = std::max(sizeof(typename PP::SharedMemoryPlan), (size_t)120 * 1024);   // one CTA per SM
    static_assert(smem2 <= 227 * 1024);
    C10_CUDA_CHECK(cudaFuncSetAttribute(k2, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem2));
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(p.rows, NC, 1);
    cfg.blockDim = dim3(256, 1, 1);
    cfg.dynamicSmemBytes = smem2;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k2, p, sp, tp));
}

// q bf16 [R, H, 512]; kv: bf16 [S, 512] or the V4.1 fp8 cache (uint8, 528 B tokens, any row count); idx int32 [R, K]
// (token rows, -1 pads, K % 64 == 0); extra_kv: a second V4.1 fp8 cache (the compressed KV, uint8, 528 B tokens, any row count) with extra_idx [R, K2];
// sink fp32 [H]; out bf16 [R, H, 512]; lse fp32 [R, H]. H is served in 64-head blocks (fewer padded in the CTA).
void dualrail_fwd(const torch::Tensor &q, const torch::Tensor &kv, const torch::Tensor &idx,
                  const c10::optional<torch::Tensor> &topk_length, const c10::optional<torch::Tensor> &extra_kv,
                  const c10::optional<torch::Tensor> &extra_idx, const c10::optional<torch::Tensor> &extra_topk_length,
                  const c10::optional<torch::Tensor> &sink, double softmax_scale, torch::Tensor &out, torch::Tensor &lse,
                  const c10::optional<torch::Tensor> &rope_cos, const c10::optional<torch::Tensor> &rope_sin,
                  const c10::optional<torch::Tensor> &rope_pos, bool rope_q, bool rope_o) {
    TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::kBFloat16 && q.dim() == 3 && q.size(2) == 512 && q.stride(2) == 1);
    TORCH_CHECK(q.stride(0) % 8 == 0 && q.stride(1) % 8 == 0 && (reinterpret_cast<uintptr_t>(q.data_ptr()) & 15) == 0);
    TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.sizes() == q.sizes() && out.stride(2) == 1 && out.stride(1) == 512);
    TORCH_CHECK(lse.scalar_type() == at::kFloat && lse.dim() == 2 && lse.size(0) == q.size(0) && lse.size(1) == q.size(1) && lse.stride(1) == 1);
    TORCH_CHECK(idx.scalar_type() == at::kInt && idx.dim() == 2 && idx.size(0) == q.size(0) && idx.stride(1) == 1 &&
                idx.size(1) % 64 == 0 && idx.size(1) > 0 && idx.stride(0) % 2 == 0, "idx: int32 [R, K], K % 64 == 0");
    const bool kv_bf16 = kv.scalar_type() == at::kBFloat16;
    if (kv_bf16) {
        TORCH_CHECK(kv.dim() == 2 && kv.size(1) == 512 && kv.is_contiguous() && (reinterpret_cast<uintptr_t>(kv.data_ptr()) & 127) == 0);
    } else {
        TORCH_CHECK(kv.scalar_type() == at::kByte && kv.is_contiguous() && kv.numel() % 528 == 0, "kv: V4.1 fp8 cache, whole 528-byte records");
    }
    const bool has_extra = extra_kv.has_value();
    if (has_extra) {
        TORCH_CHECK(!kv_bf16, "the fp8 extra cache pairs with the fp8 cache");
        TORCH_CHECK(extra_kv->scalar_type() == at::kByte && extra_kv->is_contiguous() && extra_kv->numel() % 528 == 0);
        TORCH_CHECK(extra_idx.has_value() && extra_idx->scalar_type() == at::kInt && extra_idx->dim() == 2 &&
                    extra_idx->size(0) == q.size(0) && extra_idx->stride(1) == 1 && extra_idx->size(1) % 64 == 0);
    }
    if (sink.has_value()) TORCH_CHECK(sink->scalar_type() == at::kFloat && sink->is_contiguous() && sink->numel() == q.size(1));
    if (topk_length.has_value()) TORCH_CHECK(topk_length->scalar_type() == at::kInt && topk_length->numel() == q.size(0));
    c10::cuda::CUDAGuard guard(q.device());
    const int R = (int)q.size(0), H = (int)q.size(1);
    if (R == 0) return;
    const RopeParams rp = rope_params(rope_cos, rope_sin, rope_pos, rope_q, rope_o, R);
    TORCH_CHECK(!(rope_q || rope_o) || !kv_bf16, "fused rope: engine (fp8 cache) launches only");
    auto stream = c10::cuda::getCurrentCUDAStream();
    for (int h0 = 0; h0 < H; h0 += 64) {
        Params p{};
        p.q = reinterpret_cast<const bf16 *>(q.data_ptr<at::BFloat16>()) + (int64_t)h0 * q.stride(1);
        p.q_row_stride = q.stride(0);
        p.q_head_stride = q.stride(1);
        p.rows = R;
        p.h_q = std::min(64, H - h0);
        p.topk = (int)idx.size(1);
        p.idx = idx.data_ptr<int>();
        p.idx_stride = idx.stride(0);
        p.topk_length = topk_length ? topk_length->data_ptr<int>() : nullptr;
        p.extra_topk = has_extra ? (int)extra_idx->size(1) : 0;
        p.extra_idx = has_extra ? extra_idx->data_ptr<int>() : nullptr;
        p.extra_idx_stride = has_extra ? extra_idx->stride(0) : 0;
        p.extra_topk_length = (has_extra && extra_topk_length) ? extra_topk_length->data_ptr<int>() : nullptr;
        p.sink = sink ? sink->data_ptr<float>() + h0 : nullptr;
        p.scale_log2 = float(softmax_scale * M_LOG2E);
        p.out = reinterpret_cast<bf16 *>(out.data_ptr<at::BFloat16>()) + (int64_t)h0 * out.stride(1);
        p.out_row_stride = out.stride(0);
        p.out_head_stride = out.stride(1);
        p.lse = lse.data_ptr<float>() + h0;
        p.lse_row_stride = lse.stride(0);
        p.rope = rp;
        if (kv_bf16) launch<true, false>(p, kv, extra_kv, stream);
        else if (has_extra) launch<false, true>(p, kv, extra_kv, stream);
        else launch<false, false>(p, kv, extra_kv, stream);
    }
}


// The persistent execution plan of the same recipe (persist.cuh): arguments as dualrail_fwd, plus: every row has at least
// two tiles (K >= 128, no topk_length), list rows 16-byte aligned (stride % 4 == 0); fused rope on fp8 launches.
void dualrail_persist_fwd(const torch::Tensor &q, const torch::Tensor &kv, const torch::Tensor &idx,
                          const c10::optional<torch::Tensor> &extra_kv, const c10::optional<torch::Tensor> &extra_idx,
                          const c10::optional<torch::Tensor> &extra_topk_length, const c10::optional<torch::Tensor> &sink,
                          double softmax_scale, torch::Tensor &out, torch::Tensor &lse,
                          const c10::optional<torch::Tensor> &rope_cos, const c10::optional<torch::Tensor> &rope_sin,
                          const c10::optional<torch::Tensor> &rope_pos, bool rope_q, bool rope_o) {
    TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::kBFloat16 && q.dim() == 3 && q.size(2) == 512 && q.stride(2) == 1);
    TORCH_CHECK(q.stride(0) % 8 == 0 && q.stride(1) % 8 == 0 && (reinterpret_cast<uintptr_t>(q.data_ptr()) & 15) == 0);
    TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.sizes() == q.sizes() && out.stride(2) == 1 && out.stride(1) == 512);
    TORCH_CHECK(lse.scalar_type() == at::kFloat && lse.dim() == 2 && lse.size(0) == q.size(0) && lse.size(1) == q.size(1) && lse.stride(1) == 1);
    TORCH_CHECK(idx.scalar_type() == at::kInt && idx.dim() == 2 && idx.size(0) == q.size(0) && idx.stride(1) == 1 &&
                idx.size(1) % 64 == 0 && idx.size(1) >= 128 && idx.stride(0) % 4 == 0 &&
                (reinterpret_cast<uintptr_t>(idx.data_ptr()) & 15) == 0,
                "persistent plan: idx int32 [R, K], K % 64 == 0, K >= 128, 16-byte aligned rows");
    const bool kv_bf16 = kv.scalar_type() == at::kBFloat16;
    if (kv_bf16) {
        TORCH_CHECK(kv.dim() == 2 && kv.size(1) == 512 && kv.is_contiguous() && (reinterpret_cast<uintptr_t>(kv.data_ptr()) & 127) == 0);
    } else {
        TORCH_CHECK(kv.scalar_type() == at::kByte && kv.is_contiguous() && kv.numel() % 528 == 0 &&
                    (reinterpret_cast<uintptr_t>(kv.data_ptr()) & 15) == 0, "kv: V4.1 fp8 cache, whole 528-byte records");
    }
    const bool has_extra = extra_kv.has_value();
    if (has_extra) {
        TORCH_CHECK(!kv_bf16, "the fp8 extra cache pairs with the fp8 cache");
        TORCH_CHECK(extra_kv->scalar_type() == at::kByte && extra_kv->is_contiguous() && extra_kv->numel() % 528 == 0 &&
                    (reinterpret_cast<uintptr_t>(extra_kv->data_ptr()) & 15) == 0);
        TORCH_CHECK(extra_idx.has_value() && extra_idx->scalar_type() == at::kInt && extra_idx->dim() == 2 &&
                    extra_idx->size(0) == q.size(0) && extra_idx->stride(1) == 1 && extra_idx->size(1) % 64 == 0 &&
                    extra_idx->stride(0) % 4 == 0 && (reinterpret_cast<uintptr_t>(extra_idx->data_ptr()) & 15) == 0);
    }
    if (sink.has_value()) TORCH_CHECK(sink->scalar_type() == at::kFloat && sink->is_contiguous() && sink->numel() == q.size(1));
    if (extra_topk_length.has_value()) TORCH_CHECK(extra_topk_length->scalar_type() == at::kInt && extra_topk_length->numel() == q.size(0));
    c10::cuda::CUDAGuard guard(q.device());
    const int R = (int)q.size(0), H = (int)q.size(1);
    if (R == 0) return;
    const RopeParams rp = rope_params(rope_cos, rope_sin, rope_pos, rope_q, rope_o, R);
    TORCH_CHECK(!(rope_q || rope_o) || !kv_bf16, "fused rope: engine (fp8 cache) launches only");
    auto stream = c10::cuda::getCurrentCUDAStream();
    for (int h0 = 0; h0 < H; h0 += 64) {
        Params p{};
        p.q = reinterpret_cast<const bf16 *>(q.data_ptr<at::BFloat16>()) + (int64_t)h0 * q.stride(1);
        p.q_row_stride = q.stride(0);
        p.q_head_stride = q.stride(1);
        p.rows = R;
        p.h_q = std::min(64, H - h0);
        p.topk = (int)idx.size(1);
        p.idx = idx.data_ptr<int>();
        p.idx_stride = idx.stride(0);
        p.topk_length = nullptr;
        p.rope = rp;
        p.extra_topk = has_extra ? (int)extra_idx->size(1) : 0;
        p.extra_idx = has_extra ? extra_idx->data_ptr<int>() : nullptr;
        p.extra_idx_stride = has_extra ? extra_idx->stride(0) : 0;
        p.extra_topk_length = (has_extra && extra_topk_length) ? extra_topk_length->data_ptr<int>() : nullptr;
        p.sink = sink ? sink->data_ptr<float>() + h0 : nullptr;
        p.scale_log2 = float(softmax_scale * M_LOG2E);
        p.out = reinterpret_cast<bf16 *>(out.data_ptr<at::BFloat16>()) + (int64_t)h0 * out.stride(1);
        p.out_row_stride = out.stride(0);
        p.out_head_stride = out.stride(1);
        p.lse = lse.data_ptr<float>() + h0;
        p.lse_row_stride = lse.stride(0);
        if (kv_bf16) launch_persist<true, false>(p, kv, extra_kv, stream);
        else if (has_extra) launch_persist<false, true>(p, kv, extra_kv, stream);
        else launch_persist<false, false>(p, kv, extra_kv, stream);
    }
}

// The split execution plan (split.cuh) of the same recipe; fp8 kv (+ optional fp8 extra), H <= 64. p_scr fp32 [R, T, 4096],
// v_scr uint8 [R, T, 65536] scratch (T = idx.size(1)/64 + extra_idx.size(1)/64); tpc tiles per qk CTA; nc column CTAs.
void dualrail_split_fwd(const torch::Tensor &q, const torch::Tensor &kv, const torch::Tensor &idx,
                        const c10::optional<torch::Tensor> &topk_length, const c10::optional<torch::Tensor> &extra_kv,
                        const c10::optional<torch::Tensor> &extra_idx, const c10::optional<torch::Tensor> &extra_topk_length,
                        const c10::optional<torch::Tensor> &sink, double softmax_scale, torch::Tensor &out, torch::Tensor &lse,
                        torch::Tensor &p_scr, torch::Tensor &v_scr, int64_t tpc, int64_t nc,
                        const c10::optional<torch::Tensor> &rope_cos, const c10::optional<torch::Tensor> &rope_sin,
                        const c10::optional<torch::Tensor> &rope_pos, bool rope_q, bool rope_o) {
    TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::kBFloat16 && q.dim() == 3 && q.size(2) == 512 && q.stride(2) == 1);
    TORCH_CHECK(q.stride(0) % 8 == 0 && q.stride(1) % 8 == 0 && (reinterpret_cast<uintptr_t>(q.data_ptr()) & 15) == 0);
    TORCH_CHECK(q.size(1) <= 64, "split plan: one 64-head block");
    TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.sizes() == q.sizes() && out.stride(2) == 1 && out.stride(1) == 512);
    TORCH_CHECK(lse.scalar_type() == at::kFloat && lse.dim() == 2 && lse.size(0) == q.size(0) && lse.size(1) == q.size(1) && lse.stride(1) == 1);
    TORCH_CHECK(idx.scalar_type() == at::kInt && idx.dim() == 2 && idx.size(0) == q.size(0) && idx.stride(1) == 1 &&
                idx.size(1) % 64 == 0 && idx.size(1) > 0 && idx.stride(0) % 2 == 0, "idx: int32 [R, K], K % 64 == 0");
    TORCH_CHECK(kv.scalar_type() == at::kByte && kv.is_contiguous() && kv.numel() % 528 == 0, "split plan: V4.1 fp8 cache");
    const bool has_extra = extra_kv.has_value();
    if (has_extra) {
        TORCH_CHECK(extra_kv->scalar_type() == at::kByte && extra_kv->is_contiguous() && extra_kv->numel() % 528 == 0);
        TORCH_CHECK(extra_idx.has_value() && extra_idx->scalar_type() == at::kInt && extra_idx->dim() == 2 &&
                    extra_idx->size(0) == q.size(0) && extra_idx->stride(1) == 1 && extra_idx->size(1) % 64 == 0);
    }
    if (sink.has_value()) TORCH_CHECK(sink->scalar_type() == at::kFloat && sink->is_contiguous() && sink->numel() == q.size(1));
    if (topk_length.has_value()) TORCH_CHECK(topk_length->scalar_type() == at::kInt && topk_length->numel() == q.size(0));
    const int R = (int)q.size(0), H = (int)q.size(1);
    const int T = (int)(idx.size(1) / 64 + (has_extra ? extra_idx->size(1) / 64 : 0));
    TORCH_CHECK(p_scr.is_cuda() && p_scr.scalar_type() == at::kFloat && p_scr.is_contiguous() && p_scr.numel() >= (int64_t)R * T * 4096);
    TORCH_CHECK(v_scr.is_cuda() && v_scr.scalar_type() == at::kByte && v_scr.is_contiguous() && v_scr.numel() >= (int64_t)R * T * 65536);
    TORCH_CHECK((reinterpret_cast<uintptr_t>(p_scr.data_ptr()) & 127) == 0 && (reinterpret_cast<uintptr_t>(v_scr.data_ptr()) & 127) == 0);
    TORCH_CHECK(tpc >= 1 && tpc <= T && (nc == 2 || nc == 4 || nc == 8));
    c10::cuda::CUDAGuard guard(q.device());
    if (R == 0) return;
    auto stream = c10::cuda::getCurrentCUDAStream();
    Params p{};
    p.q = reinterpret_cast<const bf16 *>(q.data_ptr<at::BFloat16>());
    p.q_row_stride = q.stride(0);
    p.q_head_stride = q.stride(1);
    p.rows = R;
    p.h_q = H;
    p.topk = (int)idx.size(1);
    p.idx = idx.data_ptr<int>();
    p.idx_stride = idx.stride(0);
    p.topk_length = topk_length ? topk_length->data_ptr<int>() : nullptr;
    p.extra_topk = has_extra ? (int)extra_idx->size(1) : 0;
    p.extra_idx = has_extra ? extra_idx->data_ptr<int>() : nullptr;
    p.extra_idx_stride = has_extra ? extra_idx->stride(0) : 0;
    p.extra_topk_length = (has_extra && extra_topk_length) ? extra_topk_length->data_ptr<int>() : nullptr;
    p.sink = sink ? sink->data_ptr<float>() : nullptr;
    p.scale_log2 = float(softmax_scale * M_LOG2E);
    p.out = reinterpret_cast<bf16 *>(out.data_ptr<at::BFloat16>());
    p.out_row_stride = out.stride(0);
    p.out_head_stride = out.stride(1);
    p.lse = lse.data_ptr<float>();
    p.lse_row_stride = lse.stride(0);
    p.rope = rope_params(rope_cos, rope_sin, rope_pos, rope_q, rope_o, R);
    SplitParams sp{p_scr.data_ptr<float>(), v_scr.data_ptr<uint8_t>(), T, (int)tpc};
    auto go = [&]<int NC>() {
        if (has_extra) launch_split<true, NC>(p, sp, kv, extra_kv, stream);
        else launch_split<false, NC>(p, sp, kv, extra_kv, stream);
    };
    if (nc == 2) go.template operator()<2>();
    else if (nc == 4) go.template operator()<4>();
    else go.template operator()<8>();
}

}  // namespace dsa_sm100::dualrail

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    namespace py = pybind11;
    m.def("dualrail_fwd", &dsa_sm100::dualrail::dualrail_fwd,
          "sm100_dsa_sparse_dualrail_v1: one CTA per query row x 64 heads, dual-rail QK, deterministic (no split)",
          py::arg("q"), py::arg("kv"), py::arg("idx"), py::arg("topk_length"), py::arg("extra_kv"), py::arg("extra_idx"),
          py::arg("extra_topk_length"), py::arg("sink"), py::arg("softmax_scale"), py::arg("out"), py::arg("lse"),
          py::arg("rope_cos") = py::none(), py::arg("rope_sin") = py::none(), py::arg("rope_pos") = py::none(),
          py::arg("rope_q") = false, py::arg("rope_o") = false);
    m.def("dualrail_persist_fwd", &dsa_sm100::dualrail::dualrail_persist_fwd,
          "sm100_dsa_sparse_dualrail_v1, persistent execution plan (CLC rows, FlashMLA prefill schedule): bit-identical to dualrail_fwd",
          py::arg("q"), py::arg("kv"), py::arg("idx"), py::arg("extra_kv"), py::arg("extra_idx"), py::arg("extra_topk_length"),
          py::arg("sink"), py::arg("softmax_scale"), py::arg("out"), py::arg("lse"),
          py::arg("rope_cos") = py::none(), py::arg("rope_sin") = py::none(), py::arg("rope_pos") = py::none(),
          py::arg("rope_q") = false, py::arg("rope_o") = false);
    m.def("dualrail_split_fwd", &dsa_sm100::dualrail::dualrail_split_fwd,
          "sm100_dsa_sparse_dualrail_v1, split execution plan (qk tiles x pv columns): bit-identical to dualrail_fwd",
          py::arg("q"), py::arg("kv"), py::arg("idx"), py::arg("topk_length"), py::arg("extra_kv"), py::arg("extra_idx"),
          py::arg("extra_topk_length"), py::arg("sink"), py::arg("softmax_scale"), py::arg("out"), py::arg("lse"),
          py::arg("p_scr"), py::arg("v_scr"), py::arg("tpc"), py::arg("nc"),
          py::arg("rope_cos") = py::none(), py::arg("rope_sin") = py::none(), py::arg("rope_pos") = py::none(),
          py::arg("rope_q") = false, py::arg("rope_o") = false);
}
