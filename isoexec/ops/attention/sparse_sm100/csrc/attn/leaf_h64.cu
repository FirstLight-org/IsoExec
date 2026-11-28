// SPDX-License-Identifier: MIT
// IsoExec sparse_sm100: host launch of the h_q = 64-per-CTA sparse-attention leaf.
#include <cuda.h>
#include <cstdio>
#include <cstdlib>

#include "leaf.cuh"

namespace sparse_sm100::attn {

static CUtensorMap make_raw_map(const void *cache, int64_t num_pages) {
    // the cache as [num_tokens, 132] uint32 words (528 B rows); box (144 words, 1 row): a gather4 writes
    // four 576-byte rows, the 12 words past each token zero-filled
    CUtensorMap m;
    const cuuint64_t dims[2] = {132, (cuuint64_t)num_pages * 64};
    const cuuint64_t strides[1] = {528};
    const cuuint32_t box[2] = {144, 1};
    const cuuint32_t estr[2] = {1, 1};
    CUresult r = cuTensorMapEncodeTiled(&m, CU_TENSOR_MAP_DATA_TYPE_UINT32, 2, const_cast<void *>(cache), dims, strides,
                                        box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                                        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS) {
        const char *msg = nullptr;
        cuGetErrorString(r, &msg);
        fprintf(stderr, "cuTensorMapEncodeTiled (raw kv) failed: %s\n", msg ? msg : "?");
        abort();
    }
    return m;
}

static CUtensorMap make_bf16_map(const void *kv, int64_t rows, int dqk = 512) {
    // the bf16 kv [rows, 512]; box (64 elements = 128 B, 1 row), SWIZZLE_128B: a gather4 writes four rows of
    // one 64-dim chunk in the UMMA K-major SW128 atom's layout (8 rows x 128 B, 16-byte unit XOR row % 8)
    CUtensorMap m;
    const cuuint64_t dims[2] = {(cuuint64_t)dqk, (cuuint64_t)rows};
    const cuuint64_t strides[1] = {(cuuint64_t)dqk * 2};
    const cuuint32_t box[2] = {64, 1};
    const cuuint32_t estr[2] = {1, 1};
    CUresult r = cuTensorMapEncodeTiled(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void *>(kv), dims, strides,
                                        box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS) {
        const char *msg = nullptr;
        cuGetErrorString(r, &msg);
        fprintf(stderr, "cuTensorMapEncodeTiled (bf16 kv) failed: %s\n", msg ? msg : "?");
        abort();
    }
    return m;
}

template<typename T, bool BF16KV, bool kLazy>
static void launch_t(const LeafParams &p, const CUtensorMap &tm, cudaStream_t stream) {
    auto kernel = sparse_leaf_kernel<T, BF16KV, kLazy>;
    const size_t smem = sizeof(typename T::SharedStorage) + 1024;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    dim3 grid(p.num_tokens, (p.h_q + T::H - 1) / T::H, p.topk / p.seg);   // z: the group (G CTAs per row)
    kernel<<<grid, T::NUM_THREADS, smem, stream>>>(p, tm);
}

void launch_leaf_h64(const LeafParams &p, cudaStream_t stream) {
    const bool lazy = p.recipe == 2;
    if (p.kv_bf16 && p.dqk == 576) {
        const CUtensorMap tm = make_bf16_map(p.cache, p.kv_rows, 576);
        if (lazy) launch_t<T576, true, true>(p, tm, stream); else launch_t<T576, true, false>(p, tm, stream);
    } else if (p.kv_bf16 && p.topk > T64B::MAX_K) {
        // glm_kpool_sm100 (lists of up to 2112 slots); the same leaf with a larger translation table
        const CUtensorMap tm = make_bf16_map(p.cache, p.kv_rows);
        if (lazy) launch_t<T64B_K2112, true, true>(p, tm, stream); else launch_t<T64B_K2112, true, false>(p, tm, stream);
    } else if (p.kv_bf16) {
        const CUtensorMap tm = make_bf16_map(p.cache, p.kv_rows);
        if (lazy) launch_t<T64B, true, true>(p, tm, stream); else launch_t<T64B, true, false>(p, tm, stream);
    } else if (p.topk > T64::MAX_K) {
        // glm_kpool_sm100's fp8 ABS recipe (2051-slot lists); the fp8 leaf with a larger translation table
        const CUtensorMap tm = make_raw_map(p.cache, p.num_pages);
        if (lazy) launch_t<T64_K2112, false, true>(p, tm, stream); else launch_t<T64_K2112, false, false>(p, tm, stream);
    } else {
        const CUtensorMap tm = make_raw_map(p.cache, p.num_pages);
        if (lazy) launch_t<T64, false, true>(p, tm, stream); else launch_t<T64, false, false>(p, tm, stream);
    }
}

}  // namespace sparse_sm100::attn
