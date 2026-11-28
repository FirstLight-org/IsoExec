// SPDX-License-Identifier: MIT
// IsoExec sparse_sm100: the index-score kernel (recipe sm100_dsa_index_logits_dg_v1) = DeepGEMM's SM100 FP8 MQA logits
// kernels (isoexec/ops/_vendor/deep_gemm, upstream 5081f85; PROVENANCE.md there), instantiated ahead of time here
// instead of by DeepGEMM's JIT. No JIT, no DeepGEMM runtime: this TU holds the template instantiations, the
// tensor maps and the launches that upstream's csrc/jit_kernels/impls/sm100_mqa_logits.hpp generates.
//
// Numerical program per (query row t, key j) -- upstream's, unchanged (impls/sm100_mqa_logits.cuh):
//   d[t,h,j] = block-scaled tcgen05 MMA (kind::mxf8f6f4, UE8M0 per 32 dims), e4m3 Q x e4m3 K, FP32 accumulator
//              over the 128 dims in four ascending k-steps;
//   r = cvt.rn.relu.bf16x2(d)  (BF16 relu, per head);
//   four BF16 fma.rn chains over the heads h = 4i + c (c = 0..3, i ascending), the chains (0,1) and (2,3) packed
//   in two bf16x2 registers; then hadd2 (c0 + c2, c1 + c3); then low + high: the BF16 logit, stored once by the
//   one thread that owns key j's TMEM lane.
// Which CTA computes which key is execution only (the schedule assigns whole keys; no value crosses threads).
#include <cuda.h>
#include <cuda_runtime.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>

#include <deep_gemm/impls/sm100_mqa_logits.cuh>

#include "mqa/mqa.h"

namespace sparse_sm100::mqa {

namespace {

using qk_t = cutlass::float_e4m3_t;
constexpr int kStages = 3;              // Q, KV and TMEM stages (upstream FP8 configuration)
constexpr int kSplitsPerChunk = 8;
constexpr int kSpecThreads = 128;
constexpr int kMathThreads = (SPLIT_KV / 128) * 128;
constexpr int kThreads = kSpecThreads + kMathThreads;
constexpr size_t kMaxSmem = 232448;     // SM100 opt-in shared memory per block

template <int H> struct Geo {
    static constexpr int BLOCK_Q = 128 / H;
    static constexpr int UMMA_N = BLOCK_Q * H;       // == align(BLOCK_Q * H, 8) for H in {32, 64}
    using Smem = deep_gemm::layout::MQALogitsSharedStorage<H, HEAD_DIM, BLOCK_Q, SPLIT_KV, UMMA_N, kStages, kStages,
                                                           kStages, qk_t>;
    static constexpr int W_ELEMS = Smem::kNumWeightElementsPerRow;
};

template <int H, int PAGE>
constexpr auto paged_kernel = &deep_gemm::sm100_paged_mqa_logits<H, HEAD_DIM, PAGE, Geo<H>::BLOCK_Q, Geo<H>::UMMA_N,
                                                                  kStages, kStages, kStages, SPLIT_KV, kSplitsPerChunk,
                                                                  kSpecThreads, kMathThreads, qk_t>;
template <int H>
constexpr auto contig_kernel = &deep_gemm::sm100_mqa_logits<H, HEAD_DIM, false, Geo<H>::BLOCK_Q, SPLIT_KV,
                                                            Geo<H>::UMMA_N, kStages, kStages, kStages, NUM_SMS,
                                                            kSpecThreads, kMathThreads, qk_t>;
//: upstream's host: the paged schedule is built for BLOCK_Q = 4 whatever the head count ("tuned for H32")
template <int THREADS>
constexpr auto meta_kernel = &deep_gemm::sched::sm100_paged_mqa_logits_metadata<SPLIT_KV, NUM_SMS, 4, THREADS>;

CUtensorMap make_map(CUtensorMapDataType dt, int rank, const void *base, const uint64_t *dims,
                     const uint64_t *strides_bytes, const uint32_t *box, CUtensorMapSwizzle swz) {
    CUtensorMap m;
    const uint32_t estr[5] = {1, 1, 1, 1, 1};
    const CUresult r = cuTensorMapEncodeTiled(&m, dt, rank, const_cast<void *>(base), dims, strides_bytes, box, estr,
                                              CU_TENSOR_MAP_INTERLEAVE_NONE, swz, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                              CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS) {
        const char *msg = nullptr;
        cuGetErrorString(r, &msg);
        fprintf(stderr, "sparse_sm100 mqa: cuTensorMapEncodeTiled failed: %s\n", msg ? msg : "?");
        abort();
    }
    return m;
}

//: upstream's maps for Q (e4m3 [rows * H, 128], SW128, box BLOCK_Q * H rows), Q scales ([rows, H] int32) and
//: weights ([rows, H] bf16, box of the 16-byte padded row)
template <int H>
void q_maps(int rows, const void *q, const int32_t *q_sf, const void *w, int64_t w_stride, CUtensorMap &tq,
            CUtensorMap &tqs, CUtensorMap &tw) {
    const uint64_t qd[2] = {HEAD_DIM, (uint64_t)rows * H}, qs[1] = {HEAD_DIM};
    const uint32_t qb[2] = {HEAD_DIM, (uint32_t)(Geo<H>::BLOCK_Q * H)};
    tq = make_map(CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, q, qd, qs, qb, CU_TENSOR_MAP_SWIZZLE_128B);
    const uint64_t sd[2] = {(uint64_t)H, (uint64_t)rows}, ss[1] = {(uint64_t)H * 4};
    const uint32_t sb[2] = {(uint32_t)H, (uint32_t)Geo<H>::BLOCK_Q};
    tqs = make_map(CU_TENSOR_MAP_DATA_TYPE_INT32, 2, q_sf, sd, ss, sb, CU_TENSOR_MAP_SWIZZLE_NONE);
    const uint64_t wd[2] = {(uint64_t)H, (uint64_t)rows}, ws[1] = {(uint64_t)w_stride * 2};
    const uint32_t wb[2] = {(uint32_t)Geo<H>::W_ELEMS, (uint32_t)Geo<H>::BLOCK_Q};
    tw = make_map(CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, w, wd, ws, wb, CU_TENSOR_MAP_SWIZZLE_NONE);
}

template <auto kKernel, typename... Args>
void launch(int grid, int threads, size_t smem, size_t smem_cap, cudaStream_t stream, Args... args) {
    // the shared-memory attribute (smem_cap >= every smem of this kernel) once per kernel and device (per context)
    static std::atomic<uint64_t> attr_devices{0};
    int dev = 0;
    cudaGetDevice(&dev);
    const uint64_t bit = uint64_t(1) << (dev & 63);
    if (!(attr_devices.load(std::memory_order_relaxed) & bit)) {
        cudaFuncSetAttribute(kKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_cap);
        attr_devices.fetch_or(bit);
    }
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(grid);
    cfg.blockDim = dim3(threads);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;   // launch structure only
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, kKernel, args...);
}

template <int H, int PAGE>
void paged_impl(const PagedArgs &a, cudaStream_t stream) {
    CUtensorMap tq, tqs, tw;
    q_maps<H>(a.rows, a.q, a.q_sf, a.weights, a.w_stride, tq, tqs, tw);
    const uint64_t kd[3] = {HEAD_DIM, (uint64_t)PAGE, (uint64_t)a.num_pages};
    const uint64_t ks[2] = {HEAD_DIM, (uint64_t)a.page_stride};
    const uint32_t kb[3] = {HEAD_DIM, (uint32_t)PAGE, 1};
    const CUtensorMap tk = make_map(CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, a.cache, kd, ks, kb, CU_TENSOR_MAP_SWIZZLE_128B);
    const uint64_t fd[2] = {(uint64_t)PAGE, (uint64_t)a.num_pages}, fs[1] = {(uint64_t)a.page_stride};
    const uint32_t fb[2] = {(uint32_t)PAGE, 1};
    const CUtensorMap tks = make_map(CU_TENSOR_MAP_DATA_TYPE_INT32, 2, a.cache + PAGE * HEAD_DIM, fd, fs, fb,
                                     CU_TENSOR_MAP_SWIZZLE_NONE);
    launch<paged_kernel<H, PAGE>>(NUM_SMS, kThreads, sizeof(typename Geo<H>::Smem), sizeof(typename Geo<H>::Smem), stream,
           (uint32_t)a.rows, (uint32_t)a.out_stride, (uint32_t)a.bt_stride,
           reinterpret_cast<const uint32_t *>(a.ke), reinterpret_cast<nv_bfloat16 *>(a.out),
           reinterpret_cast<const uint32_t *>(a.block_table), reinterpret_cast<const uint32_t *>(a.indices),
           reinterpret_cast<const uint32_t *>(a.meta), tq, tqs, tk, tks, tw);
}

template <int H>
void contig_impl(const ContigArgs &a, cudaStream_t stream) {
    CUtensorMap tq, tqs, tw;
    q_maps<H>(a.rows, a.q, a.q_sf, a.weights, a.w_stride, tq, tqs, tw);
    const uint64_t kd[2] = {HEAD_DIM, (uint64_t)a.num_keys}, ks[1] = {HEAD_DIM};
    const uint32_t kb[2] = {HEAD_DIM, 128};      // SPLIT_KV % 256 != 0: 128-token TMA tiles (upstream)
    const CUtensorMap tk = make_map(CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, a.k, kd, ks, kb, CU_TENSOR_MAP_SWIZZLE_128B);
    // scales: one int32 per key; upstream maps (align(num_keys, 4), 1) -- here (num_keys, 1), so no word past the
    // last key is read (the TMA fills out-of-range words with zero either way)
    const uint64_t fd[2] = {(uint64_t)a.num_keys, 1}, fs[1] = {((uint64_t)a.num_keys * 4 + 15) / 16 * 16};
    const uint32_t fb[2] = {128, 1};
    const CUtensorMap tks = make_map(CU_TENSOR_MAP_DATA_TYPE_INT32, 2, a.k_sf, fd, fs, fb, CU_TENSOR_MAP_SWIZZLE_NONE);
    launch<contig_kernel<H>>(NUM_SMS, kThreads, sizeof(typename Geo<H>::Smem), sizeof(typename Geo<H>::Smem), stream,
           (uint32_t)a.rows, (uint32_t)a.num_keys, (uint32_t)a.out_stride,
           reinterpret_cast<const uint32_t *>(a.ks), reinterpret_cast<const uint32_t *>(a.ke),
           static_cast<const uint32_t *>(nullptr), reinterpret_cast<nv_bfloat16 *>(a.out), tq, tqs, tk, tks, tw);
}

}  // namespace

static_assert(sizeof(Geo<32>::Smem) <= kMaxSmem && sizeof(Geo<64>::Smem) <= kMaxSmem, "shared memory");

size_t paged_smem_bytes(int heads) { return heads == 64 ? sizeof(Geo<64>::Smem) : sizeof(Geo<32>::Smem); }

void paged_schedule(int rows, const int32_t *ke, const int32_t *indices, int32_t *meta, cudaStream_t stream) {
    // upstream's host: 256 / 512 / 1024 threads by the row count; shared memory = request starts, work prefix,
    // warp sums and the request count
    const int threads = rows <= 512 ? 256 : (rows <= 2048 ? 512 : 1024);
    const size_t smem = (2 * (size_t)rows + threads / 32 + 1) * sizeof(int);
    if (smem > kMaxSmem) {
        fprintf(stderr, "sparse_sm100 mqa: paged schedule of %d rows exceeds shared memory\n", rows);
        abort();
    }
    const auto *k = reinterpret_cast<const uint32_t *>(ke);
    const auto *ix = reinterpret_cast<const uint32_t *>(indices);
    auto *m = reinterpret_cast<uint32_t *>(meta);
    if (threads == 256)      launch<meta_kernel<256>>(1, 256, smem, kMaxSmem, stream, (uint32_t)rows, k, ix, m);
    else if (threads == 512) launch<meta_kernel<512>>(1, 512, smem, kMaxSmem, stream, (uint32_t)rows, k, ix, m);
    else                     launch<meta_kernel<1024>>(1, 1024, smem, kMaxSmem, stream, (uint32_t)rows, k, ix, m);
}

void paged_logits(const PagedArgs &a, cudaStream_t stream) {
    if (a.heads == 32) {
        if (a.page_kv == 64) paged_impl<32, 64>(a, stream); else paged_impl<32, 128>(a, stream);
    } else {
        if (a.page_kv == 64) paged_impl<64, 64>(a, stream); else paged_impl<64, 128>(a, stream);
    }
}

void logits(const ContigArgs &a, cudaStream_t stream) {
    if (a.heads == 32) contig_impl<32>(a, stream); else contig_impl<64>(a, stream);
}

}  // namespace sparse_sm100::mqa
