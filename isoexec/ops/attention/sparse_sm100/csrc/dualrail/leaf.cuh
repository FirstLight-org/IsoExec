// SPDX-License-Identifier: MIT
// IsoExec sparse_sm100: the dual-rail sparse-attention leaf (recipes sm100_sparse_dualrail_v1 and its sink form).
//
// Derived from FlashMLA 395cb3a csrc/cuda_kernels/sm100/decode/sparse/head64/{config.h,kernel.cuh} (MIT, Copyright (c)
// 2025 DeepSeek; dualrail/LICENSE-FlashMLA.txt, dualrail/PROVENANCE.md). Changes, each named in PROVENANCE.md:
//   * the split-KV path is REMOVED (scheduler metadata, partial outputs, combine): one CTA = one query row x 64 heads,
//     every row runs its whole list -- nothing depends on the batch, its composition, or the SM count;
//   * the lazy-rescale decision is PER ROW (FlashMLA: __any_sync over a warp, i.e. 32 heads decide together), so a
//     head's bits never depend on the other heads of the CTA (ranks with < 64 heads == the 64-head run per head);
//   * a bf16-kv mode: the K/V tile is gathered straight into the SW128 tile (no raw buffer, no dequant arithmetic);
//   * flat token coordinates: idx[r, p] is the token's row in the cache (the caller resolves any paging);
//   * outputs [rows, h_q, 512] / lse [rows, h_q] with IsoExec strides; no PDL trigger.
//
// Numerical program (README "Dual-rail recipe"): per 64-slot tile of the list (list order), S = rail0 + rail1 where
// rail r is ONE tcgen05.mma .ws K16 chain over the 256 dims of 64-dim chunks {r, r+2, r+4, r+6} (ascending, zero
// start), both summed once in fp32 (commutative); invalid slots -> -inf; per row: m_t = max(S) * c (c = scale*log2 e,
// host fp32); lazy max: if m_t - m > 6 then m' = max(m_t, m), a = exp2f(m - m') else m' = m, a = 1;
// P = bf16(exp2f(fma(S, c, -m'))); l_half = fma(l_half, a, (sum of the half's even-index P terms) + (odd-index terms))
// with each thread's float2 chain ascending over its 32 slots; O = O*a (exact for a = 1) then O += P V (one .ws K16
// chain over the tile's 64 slots); end: l = l_half0 + l_half1, out = bf16(O * (l == 0 ? 0 : 1 / (l + exp2f(sink -
// m'))) with __fdividef), lse = fma(m', ln 2, logf(l + exp2f(sink - m'))) (sink-inclusive; an empty row: out 0, lse = sink,
// +inf without a sink). exp2f / __fdividef / logf as compiled under the
// sparse_sm100 profile (identical for every caller of this leaf: engine decode and trainer replay run the same kernel).
#pragma once

#include <cuda.h>
#include <cuda_fp8.h>
#include <math_constants.h>
#include <cutlass/barrier.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cute/tensor.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>

#include <kerutils/kerutils.cuh>

#include "cuda_kernels/defines.h"
#include "cuda_kernels/kv_cache_format.h"
#include "cuda_kernels/sm100/helpers.h"
#include "cuda_kernels/sm100/common_subroutine.h"
#include "cuda_kernels/sm100/dequant_utils.cuh"
#include "cuda_kernels/sm100/kv_cache_utils.cuh"
#include "fused_rope.cuh"

namespace sparse_sm100::dualrail {

using namespace cute;
using cutlass::arch::NamedBarrier;
using cutlass::arch::fence_view_async_shared;
using namespace sm100;

struct Params {
    const bf16 *q;                  // [rows, h_q, 512]
    int64_t q_row_stride, q_head_stride;
    int rows, h_q;                  // h_q <= 64 (a launch serves one 64-head block)
    int topk, extra_topk;           // list widths (multiples of 64)
    const int *idx;                 // [rows, idx_stride]: token rows (flat), -1 pads
    int64_t idx_stride;
    const int *extra_idx;           // [rows, extra_idx_stride] (fp8 extra cache)
    int64_t extra_idx_stride;
    const int *topk_length, *extra_topk_length;   // optional [rows]
    const float *sink;              // optional [h_q] (natural units)
    float scale_log2;               // softmax scale * log2(e) (host fp32)
    bf16 *out;                      // [rows, h_q, 512]
    int64_t out_row_stride, out_head_stride;
    float *lse;                     // [rows, h_q] contiguous rows of lse_row_stride
    int64_t lse_row_stride;
    RopeParams rope;                // optional fused Q RoPE / output inverse RoPE (fused_rope.cuh)
};

enum NamedBarriers : uint32_t { wg0_sync = 1, wg0_warp02_sync = 2, wg0_warp13_sync = 3 };

// KV_BF16: the cache is a bf16 [S, 512]; else the V4.1 fp8 cache (and, EXTRA, a second V4.1 fp8 cache: the compressed
// KV, read through its own tensor map and list).
template<bool KV_BF16, bool EXTRA>
struct Leaf {
    static constexpr int MMA_M = 64;
    static constexpr int D_Q = 512, D_V = 512, D_FP8 = 512;
    using OrigKVFormat = KVCacheFormat<ModelType::V41>;
    using ExtraKVFormat = KVCacheFormat<ModelType::V41>;
    static constexpr int B_TOPK = 64;
    static constexpr int NUM_BUFS = KV_BF16 ? 3 : 2;    // bf16: no raw buffers, so a third K/V tile fits
    static constexpr int NUM_INDEX_BUFS = 4;
    template<typename F> static constexpr int RAW_TOKEN_SMEM_STRIDE = F::QUANT_BYTES + 64;
    static constexpr int RAW_BLOCK_SMEM_BYTES = KV_BF16 ? 128 : B_TOPK * std::max(RAW_TOKEN_SMEM_STRIDE<OrigKVFormat>, RAW_TOKEN_SMEM_STRIDE<ExtraKVFormat>);
    template<typename F> using Dequantizer = KVBlockDequantizer<F, D_FP8, B_TOPK, RAW_TOKEN_SMEM_STRIDE<F>, RAW_TOKEN_SMEM_STRIDE<F>>;
    static constexpr int NUM_THREADS = 128 * 3;
    static constexpr float MAX_INIT_VAL = -1e30f;
    static constexpr float RESCALE_THRESHOLD = 6.0f;

    struct tmem_cols { static constexpr int O = 0, Q = 256, P = 400; };

    template<int NUM_TILES>
    using SmemLayoutQTiles = decltype(coalesce(tile_to_shape(UMMA::Layout_K_SW128_Atom<bf16>{}, Shape<Int<MMA_M>, Int<NUM_TILES*64>>{}, Step<_1, _2>{}), Shape<_1, _1>{}));
    using SmemLayoutQ_SW128 = SmemLayoutQTiles<D_Q/64>;
    using SmemLayoutOBuf = decltype(tile_to_shape(UMMA::Layout_K_SW128_Atom<bf16>{}, Shape<Int<MMA_M>, Int<D_V>>{}));
    using SmemLayoutOBuf_TMA = decltype(tile_to_shape(UMMA::Layout_K_SW128_Atom<bf16>{}, Shape<Int<MMA_M>, Int<64>>{}));
    using SmemLayoutS = decltype(tile_to_shape(UMMA::Layout_K_INTER_Atom<bf16>{}, Shape<Int<MMA_M>, Int<B_TOPK>>{}, Step<_1, _2>{}));
    template<int NUM_TILES>
    using SmemLayoutKTiles_SW128 = decltype(coalesce(tile_to_shape(UMMA::Layout_K_SW128_Atom<bf16>{}, Shape<Int<B_TOPK>, Int<64*NUM_TILES>>{}, Step<_1, _2>{}), Shape<_1, _1>{}));
    template<int NUM_TILES>
    using SmemLayoutKTiles_DualGemm_SW128 = decltype(coalesce(tile_to_shape(UMMA::Layout_K_SW128_Atom<bf16>{}, Shape<Int<B_TOPK*2>, Int<64*NUM_TILES>>{}, Step<_1, _2>{}), Shape<_1, _1>{}));
    template<int NUM_TILES>
    using SmemLayoutKTilesTransposed_SW128 = decltype(composition(SmemLayoutKTiles_SW128<NUM_TILES>{}, Layout<Shape<Int<64*NUM_TILES>, Int<B_TOPK>>, Stride<Int<B_TOPK>, _1>>{}));

    struct SharedMemoryPlan {
        union {
            struct {
                array_aligned<bf16, cosize_v<SmemLayoutQ_SW128>> q;
                array_aligned<bf16, cosize_v<SmemLayoutOBuf>> o_buf;
            } qo;
            struct {
                struct { alignas(1024) bf16 quant_part[B_TOPK*D_FP8]; } dequant[NUM_BUFS];
                array_aligned<uint8_t, RAW_BLOCK_SMEM_BYTES, 128> raw_quant[NUM_BUFS];
            } kv;
        } u;
        union {
            float p_exchange_buf[4][32 * (B_TOPK/(128/MMA_M))];
            array_aligned<bf16, cosize_v<SmemLayoutS>> s;
        } s_p;
        CUTE_ALIGNAS(16) float rowwise_max_buf[128];
        char is_token_valid[NUM_INDEX_BUFS][B_TOPK/8];
        int tma_coord[NUM_INDEX_BUFS][B_TOPK];
        array_aligned<uint32_t, 1> tmem_start_addr;
        transac_bar_t bar_q_tma, bar_q_utccp, bar_q_roped;
        transac_bar_t bar_quant_part_dequant_ready[NUM_BUFS];
        transac_bar_t bar_raw_ready[NUM_BUFS], bar_raw_free[NUM_BUFS];
        transac_bar_t bar_valid_coord_ready[NUM_INDEX_BUFS], bar_valid_coord_free[NUM_INDEX_BUFS];
        transac_bar_t bar_qk_done[NUM_BUFS], bar_so_ready[NUM_BUFS], bar_sv_done[NUM_BUFS];
    };

    using TiledMMA_P = decltype(make_tiled_mma(SM100_MMA_F16BF16_WS_TS_NOELECT<bf16, bf16, float, MMA_M, B_TOPK*2, UMMA::Major::K, UMMA::Major::K>{}));
    static constexpr int PV_GEMM_N = 256;
    using TiledMMA_O = decltype(make_tiled_mma(SM100_MMA_F16BF16_WS_SS_NOELECT<bf16, bf16, float, MMA_M, PV_GEMM_N, UMMA::Major::K, UMMA::Major::MN>{}));

    template<typename Shape_Q, typename TMA_Q, typename Shape_O, typename TMA_O>
    struct TmaParams {
        Shape_Q shape_Q; TMA_Q tma_Q;
        Shape_O shape_O; TMA_O tma_O;
        CUtensorMap kv_map;         // fp8: raw rows (uint32 words); bf16: [S, 512] box (64, 1) SW128
        CUtensorMap extra_kv_map;
    };
};

template<typename L, typename TmaParam, bool KV_BF16, bool EXTRA>
__global__ void __launch_bounds__(L::NUM_THREADS, 1, 1)
dualrail_kernel(__grid_constant__ const Params params, __grid_constant__ const TmaParam tma_params) {
    // PDL (launch structure only): no global memory is touched before the previous grid completed. The
    // successor is released only at this grid's completion (no early trigger: a multi-wave launch keeps its SMs).
    asm volatile("griddepcontrol.wait;" ::: "memory");
    using OrigKVFormat = typename L::OrigKVFormat;
    using ExtraKVFormat = typename L::ExtraKVFormat;
    constexpr int MMA_M = L::MMA_M, B_TOPK = L::B_TOPK, NUM_BUFS = L::NUM_BUFS, NUM_INDEX_BUFS = L::NUM_INDEX_BUFS;
    constexpr int D_Q = L::D_Q, D_V = L::D_V, PV_GEMM_N = L::PV_GEMM_N;
    using tmem_cols = typename L::tmem_cols;
    const int row = blockIdx.x;
    const int warpgroup_idx = cutlass::canonical_warp_group_idx();
    const int idx_in_warpgroup = threadIdx.x % 128;
    const int warp_idx = cutlass::canonical_warp_idx_sync();
    const int lane_idx = threadIdx.x % 32;

    extern __shared__ char wksp_buf[];
    typename L::SharedMemoryPlan &plan = *reinterpret_cast<typename L::SharedMemoryPlan *>(wksp_buf);

    if (warp_idx == 0 && elect_one_sync()) {
        cute::prefetch_tma_descriptor(tma_params.tma_Q.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_O.get_tma_descriptor());
        cute::prefetch_tma_descriptor(&tma_params.kv_map);
    }
    if (warp_idx == 0) {
        if (elect_one_sync()) {
            plan.bar_q_tma.init(1);
            plan.bar_q_utccp.init(1);
            plan.bar_q_roped.init(32);
            for (int i = 0; i < NUM_BUFS; ++i) {
                plan.bar_quant_part_dequant_ready[i].init(128);
                plan.bar_raw_ready[i].init(1);
                plan.bar_raw_free[i].init(128);
                plan.bar_qk_done[i].init(1);
                plan.bar_so_ready[i].init(128);
                plan.bar_sv_done[i].init(1);
            }
            for (int i = 0; i < NUM_INDEX_BUFS; ++i) {
                plan.bar_valid_coord_ready[i].init(32);
                plan.bar_valid_coord_free[i].init(128 + 1);
            }
            cutlass::arch::fence_barrier_init();
        }
        cute::TMEM::Allocator1Sm().allocate(512, plan.tmem_start_addr.data());
        KU_TRAP_ONLY_DEVICE_ASSERT(plan.tmem_start_addr.data()[0] == 0);
        cute::TMEM::Allocator1Sm().release_allocation_lock();
    }
    __syncthreads();

    // the row's whole list: no split, no scheduler
    const int topk_length = params.topk_length ? __ldg(params.topk_length + row) : params.topk;
    const int orig_topk_padded = max(ku::ceil(topk_length, (int)B_TOPK), (int)B_TOPK);
    const int extra_topk_length = EXTRA ? (params.extra_topk_length ? __ldg(params.extra_topk_length + row) : params.extra_topk) : 0;
    const int num_orig_kv_blocks = orig_topk_padded / B_TOPK;
    const int end_block_idx = num_orig_kv_blocks + ku::ceil(extra_topk_length, (int)B_TOPK) / B_TOPK;
    constexpr bool bar_phase_batch_rel = 0;

    struct RingState {
        int buf_idx = 0;
        bool bar_phase = 0;
        int index_buf_idx = 0;
        bool index_bar_phase = 0;
        CUTE_DEVICE void update() {
            bar_phase ^= (buf_idx == NUM_BUFS - 1);
            buf_idx = (buf_idx + 1) % NUM_BUFS;
            index_bar_phase ^= (index_buf_idx == NUM_INDEX_BUFS - 1);
            index_buf_idx = (index_buf_idx + 1) % NUM_INDEX_BUFS;
        }
    };
    RingState rs;

    if (warpgroup_idx == 0) {
        // ======================= scale & exp warpgroup (softmax, O correction, epilogue) =======================
        cutlass::arch::warpgroup_reg_alloc<224>();
        constexpr int B_EPI = 64;
        Tensor sO = make_tensor(make_smem_ptr(plan.u.qo.o_buf.data()), typename L::SmemLayoutOBuf{});
        bf16 *sO_bases[B_EPI/8];
        CUTE_UNROLL
        for (int i = 0; i < B_EPI/8; ++i)
            sO_bases[i] = &sO(idx_in_warpgroup % MMA_M, (idx_in_warpgroup / MMA_M) * (PV_GEMM_N / (128 / MMA_M)) + i * 8);
        const float2 scale = float2{params.scale_log2, params.scale_log2};
        bf16 *sS_base = plan.s_p.s.data() + lane_idx*8 + (warp_idx&1)*(MMA_M/2)*8 + (warp_idx/2)*MMA_M*(B_TOPK/2);
        // padding rows (head >= h_q) read no sink; their output is never stored
        const float attn_sink = (params.sink == nullptr || idx_in_warpgroup % MMA_M >= params.h_q)
            ? -CUDART_INF_F : __ldg(params.sink + (idx_in_warpgroup % MMA_M)) * CUDART_L2E_F;
        constexpr int NUM_ELEMS_PER_THREAD = B_TOPK / (128 / MMA_M);

        float mi = L::MAX_INIT_VAL;
        float li = 0.0f;
        float real_mi = -CUDART_INF_F;

        CUTE_NO_UNROLL
        for (int block_idx = 0; block_idx < end_block_idx; ++block_idx) {
            NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
            plan.bar_valid_coord_ready[rs.index_buf_idx].wait(rs.index_bar_phase);
            plan.bar_qk_done[rs.buf_idx].wait(rs.bar_phase);
            ku::tcgen05_after_thread_sync();

            float p[NUM_ELEMS_PER_THREAD];
            retrieve_mask_and_reduce_p<MMA_M, NUM_ELEMS_PER_THREAD, NamedBarriers::wg0_warp02_sync, NamedBarriers::wg0_warp13_sync>(
                tmem_cols::P, plan.is_token_valid[rs.index_buf_idx], warp_idx, lane_idx, [&]() {}, plan.s_p.p_exchange_buf, p);

            float cur_pi_max = get_max<NUM_ELEMS_PER_THREAD>(p);
            cur_pi_max *= params.scale_log2;
            plan.rowwise_max_buf[idx_in_warpgroup] = cur_pi_max;
            NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
            plan.bar_valid_coord_free[rs.index_buf_idx].arrive();
            cur_pi_max = max(cur_pi_max, plan.rowwise_max_buf[idx_in_warpgroup ^ 64]);
            real_mi = max(real_mi, cur_pi_max);
            // IsoExec: the lazy-rescale decision is the ROW's (identical in the row's two threads)
            const bool row_scale = cur_pi_max - mi > L::RESCALE_THRESHOLD;
            float new_max, scale_for_old;
            if (!row_scale) {
                scale_for_old = 1.0f;
                new_max = mi;
            } else {
                new_max = max(cur_pi_max, mi);
                scale_for_old = exp2f(mi - new_max);
            }
            mi = new_max;

            __nv_bfloat162 s[NUM_ELEMS_PER_THREAD/2];
            float2 neg_new_max = float2{-new_max, -new_max};
            float2 cur_sum = float2{0.0f, 0.0f};
            CUTE_UNROLL
            for (int i = 0; i < NUM_ELEMS_PER_THREAD/2; i += 1) {
                float2 d = ku::float2_fma(float2{p[i*2], p[i*2+1]}, scale, neg_new_max);
                d.x = exp2f(d.x);
                d.y = exp2f(d.y);
                cur_sum = ku::float2_add(cur_sum, d);
                s[i] = __float22bfloat162_rn(d);
            }
            li = fma(li, scale_for_old, (cur_sum.x + cur_sum.y));

            CUTE_UNROLL
            for (int i = 0; i < NUM_ELEMS_PER_THREAD/8; i += 1)
                *(uint128_t *)(sS_base + MMA_M*8*i) = *(uint128_t *)(s + i*4);

            // O *= a: the TMEM loads / stores are warp-collective, so a warp runs the correction when any of its rows
            // rescales; a row that does not multiplies by exactly 1.0f (an identity on every float)
            if (block_idx != 0 && __any_sync(0xffffffff, row_scale)) {
                ku::tcgen05_after_thread_sync();
                rescale_O<D_V / (128/MMA_M), 64, tmem_cols::O>(scale_for_old);
                ku::tcgen05_before_thread_sync();
            }
            fence_view_async_shared();
            plan.bar_so_ready[rs.buf_idx].arrive();
            if (block_idx != end_block_idx - 1) rs.update();
        }

        if (real_mi == -CUDART_INF_F) {
            li = 0.0f;
            mi = -CUDART_INF_F;
        }
        // FlashMLA #241 (395cb3a): the partner thread (idx ^ 64, another warp) may still be reading this thread's last
        // row max from rowwise_max_buf; wait for it before the buffer is reused for the li exchange
        NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
        plan.rowwise_max_buf[idx_in_warpgroup] = li;
        NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
        li += plan.rowwise_max_buf[idx_in_warpgroup ^ 64];

        // denominator with the zero-value sink (sink = -inf without one); lse is sink-inclusive (sparse_sm100 README.md): an
        // empty row has out = 0 and lse = sink (+inf without a sink)
        const float denom = li + exp2f(attn_sink - mi);
        if (idx_in_warpgroup < params.h_q) {
            float cur_lse;
            if (li == 0.0f) cur_lse = params.sink ? __ldg(params.sink + idx_in_warpgroup) : +CUDART_INF_F;
            else cur_lse = fma(mi, CUDART_LN2_F, logf(denom));
            params.lse[(int64_t)row * params.lse_row_stride + idx_in_warpgroup] = cur_lse;
        }

        plan.bar_sv_done[rs.buf_idx].wait(rs.bar_phase);
        ku::tcgen05_after_thread_sync();

        static constexpr int O_FOLD = 128 / MMA_M;
        Tensor tma_gO = flat_divide(tma_params.tma_O.get_tma_tensor(tma_params.shape_O)(_, _, row), Shape<Int<MMA_M>, Int<64>>{})(_, _, _0{}, _);
        auto thr_tma = tma_params.tma_O.get_slice(_0{});
        Tensor tma_sO = flat_divide(sO, Shape<Int<MMA_M>, Int<64>>{})(_, _, _0{}, _);
        const float o_scale = li == 0.0f ? 0.0f : __fdividef(1.0f, denom);
        const float2 o_scale_float2 = {o_scale, o_scale};
        float2 o[B_EPI/2];
        __nv_bfloat162 o_bf16[B_EPI/2];
        CUTE_UNROLL
        for (int i = 0; i < (D_V/O_FOLD) / B_EPI; ++i) {
            ku::tmem_ld_32dp32bNx<B_EPI>(tmem_cols::O + i*B_EPI, o);
            cutlass::arch::fence_view_async_tmem_load();
            CUTE_UNROLL
            for (int j = 0; j < B_EPI/2; ++j) {
                o[j] = ku::float2_mul(o[j], o_scale_float2);
                o_bf16[j] = __float22bfloat162_rn(o[j]);
            }
            const int col_base = (i >= PV_GEMM_N/O_FOLD/B_EPI ? D_V/2 : 0) + (i*B_EPI % (PV_GEMM_N/O_FOLD));
            if (params.rope.o)       // fused inverse RoPE of the bf16 output's tail (warp-uniform column range)
                rope_o_chunk(o_bf16, (idx_in_warpgroup / MMA_M) * (PV_GEMM_N / (128 / MMA_M)) + col_base, params.rope, row);
            CUTE_UNROLL
            for (int j = 0; j < B_EPI / 8; ++j)
                *(__int128_t *)(sO_bases[j] + col_base*MMA_M) = *(__int128_t *)(&o_bf16[j*4]);
            fence_view_async_shared();
            NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
            if (warp_idx < O_FOLD && elect_one_sync()) {
                const int copy_block_idx = warp_idx*(PV_GEMM_N/(128/MMA_M)/B_EPI) + col_base/B_EPI;
                cute::copy(tma_params.tma_O, thr_tma.partition_S(tma_sO(_, _, copy_block_idx)), thr_tma.partition_D(tma_gO(_, _, copy_block_idx)));
            }
        }
        cute::tma_store_arrive();
        cute::tma_store_wait<0>();
        if (warp_idx == 0) cute::TMEM::Allocator1Sm().free(0, 512);
    } else if (warpgroup_idx == 1) {
        cutlass::arch::warpgroup_reg_dealloc<72>();
        const int warp_idx = cutlass::canonical_warp_idx_sync();
        if (warp_idx == 4 && elect_one_sync()) {
            // ================================ UMMA warp ================================
            {
                Tensor gQ = tma_params.tma_Q.get_tma_tensor(tma_params.shape_Q)(_, _, row);
                Tensor sQ = make_tensor(make_smem_ptr(plan.u.qo.q.data()), typename L::SmemLayoutQ_SW128{});
                ku::launch_tma_copy(tma_params.tma_Q, gQ, sQ, plan.bar_q_tma, TMA::CacheHintSm90::EVICT_FIRST);
            }
            plan.bar_q_tma.arrive_and_expect_tx(MMA_M*D_Q*sizeof(bf16));
            plan.bar_q_tma.wait(bar_phase_batch_rel);
            if (params.rope.q) plan.bar_q_roped.wait(bar_phase_batch_rel);   // warp 6 rotated the tile's tail
            ku::tcgen05_after_thread_sync();
            {
                UMMA::SmemDescriptor sQ_desc = UMMA::make_umma_desc<UMMA::Major::K>(make_tensor(
                    make_smem_ptr(plan.u.qo.q.data()), tile_to_shape(UMMA::Layout_K_SW128_Atom<bf16>{}, Shape<Int<MMA_M*2>, Int<64>>{})));
                #pragma unroll (D_Q/128)
                for (int tile_idx = 0; tile_idx < D_Q/128; ++tile_idx) {
                    CUTE_UNROLL
                    for (int subtile_idx = 0; subtile_idx < 64/16; ++subtile_idx)
                        SM100_UTCCP_128dp256bit_1cta::copy(sQ_desc + (tile_idx*(MMA_M*128) + subtile_idx*16) * 2 / 16,
                                                           tmem_cols::Q + tile_idx*32 + subtile_idx*8);
                }
            }
            ku::umma_arrive_noelect(plan.bar_q_utccp);
            typename L::TiledMMA_P tiled_mma_P{};
            typename L::TiledMMA_O tiled_mma_O{};
            Tensor tP = partition_fragment_C(tiled_mma_P, Shape<Int<MMA_M>, _128>{});
            Tensor tO = partition_fragment_C(tiled_mma_O, Shape<Int<MMA_M>, Int<D_V>>{});
            tP.data().get() = tmem_cols::P;
            tO.data().get() = tmem_cols::O;
            plan.bar_q_utccp.wait(bar_phase_batch_rel);
            ku::tcgen05_after_thread_sync();
            CUTE_NO_UNROLL
            for (int block_idx = 0; block_idx < end_block_idx; ++block_idx) {
                // QK, dual rail: |0|2|4|6| over |1|3|5|7| (64-dim chunks), one .ws chain per rail
                if constexpr (KV_BF16) plan.bar_raw_ready[rs.buf_idx].wait(rs.bar_phase);   // the gathered tile itself
                else plan.bar_quant_part_dequant_ready[rs.buf_idx].wait(rs.bar_phase);
                ku::tcgen05_after_thread_sync();
                Tensor tQ = tiled_mma_P.get_slice(_0{}).make_fragment_A(partition_shape_A(tiled_mma_P, Shape<Int<MMA_M>, Int<D_Q/2>>{}));
                tQ.data().get() = tmem_cols::Q;
                Tensor sK = make_tensor(make_smem_ptr(plan.u.kv.dequant[rs.buf_idx].quant_part), typename L::template SmemLayoutKTiles_DualGemm_SW128<D_Q/64/2>{});
                ku::utcmma_ts(tiled_mma_P, tQ, sK, tP, true);
                ku::umma_arrive_noelect(plan.bar_qk_done[rs.buf_idx]);
                // PV
                plan.bar_so_ready[rs.buf_idx].wait(rs.bar_phase);
                ku::tcgen05_after_thread_sync();
                Tensor sS = make_tensor(make_smem_ptr(plan.s_p.s.data()), typename L::SmemLayoutS{});
                Tensor sV = make_tensor(make_smem_ptr(plan.u.kv.dequant[rs.buf_idx].quant_part), typename L::template SmemLayoutKTilesTransposed_SW128<D_V/64>{});
                ku::utcmma_ss(tiled_mma_O, sS, sV, tO, block_idx == 0);
                ku::umma_arrive_noelect(plan.bar_sv_done[rs.buf_idx]);
                rs.update();
            }
        } else if (warp_idx == 5) {
            // ================================ KV producer ================================
            if constexpr (KV_BF16) {
                // bf16 kv: the dequant warpgroup issues the gathers (below)
            } else if (elect_one_sync()) {
                plan.bar_q_utccp.wait(bar_phase_batch_rel);
                for_each_kv_block<OrigKVFormat, ExtraKVFormat>(0, end_block_idx, num_orig_kv_blocks, [&]<typename F>(int block_idx, bool is_extra_block) {
                    plan.bar_valid_coord_ready[rs.index_buf_idx].wait(rs.index_bar_phase);
                    plan.bar_raw_free[rs.buf_idx].wait(rs.bar_phase ^ 1);
                    const CUtensorMap *tensor_map = is_extra_block ? &tma_params.extra_kv_map : &tma_params.kv_map;
                    int4 cur_indices = *(int4 *)(plan.tma_coord[rs.index_buf_idx] + 0);
                    int4 nxt_cur_indices;
                    CUTE_UNROLL
                    for (int r = 0; r < B_TOPK; r += 4) {
                        if (r + 4 < B_TOPK) nxt_cur_indices = *(int4 *)(plan.tma_coord[rs.index_buf_idx] + r + 4);
                        ku::tma_gather4(tensor_map, plan.bar_raw_ready[rs.buf_idx],
                                        plan.u.kv.raw_quant[rs.buf_idx].data() + L::template RAW_TOKEN_SMEM_STRIDE<F> * r, 0, cur_indices,
                                        (int64_t)TMA::CacheHintSm90::EVICT_FIRST);
                        cur_indices = nxt_cur_indices;
                    }
                    plan.bar_raw_ready[rs.buf_idx].arrive_and_expect_tx(B_TOPK * L::template RAW_TOKEN_SMEM_STRIDE<F>);
                    plan.bar_valid_coord_free[rs.index_buf_idx].arrive();
                    rs.update();
                });
            }
        } else if (warp_idx == 6) {
            // ======================= fused Q RoPE (fused_rope.cuh), before the UTCCP into TMEM =======================
            if (params.rope.q) {
                plan.bar_q_tma.wait(bar_phase_batch_rel);
                Tensor sQ = make_tensor(make_smem_ptr(plan.u.qo.q.data()), typename L::SmemLayoutQ_SW128{});
                rope_q_tile(sQ, params.h_q, params.rope, row, lane_idx);
                fence_view_async_shared();          // generic-proxy writes -> the tcgen05.cp reads
                plan.bar_q_roped.arrive();
            }
        } else if (warp_idx == 7) {
            // ======================= indices -> TMA coordinates + validity (flat token rows) =======================
            const int *indices = params.idx + params.idx_stride * row;
            const int *extra_indices = EXTRA ? params.extra_idx + params.extra_idx_stride * row : nullptr;
            auto load_block_indices = [&](int block_idx) -> int2 {
                if (block_idx < num_orig_kv_blocks) return __ldg((const int2 *)(indices + block_idx*B_TOPK + lane_idx*2));
                return __ldg((const int2 *)(extra_indices + (block_idx - num_orig_kv_blocks)*B_TOPK + lane_idx*2));
            };
            int2 my_indices = load_block_indices(0);
            CUTE_NO_UNROLL
            for (int block_idx = 0; block_idx < end_block_idx; ++block_idx) {
                const bool has_next = block_idx + 1 < end_block_idx;
                const int2 next_indices = has_next ? load_block_indices(block_idx + 1) : int2{};
                const bool is_extra = block_idx >= num_orig_kv_blocks;
                const int abs_pos = (is_extra ? block_idx - num_orig_kv_blocks : block_idx) * B_TOPK + lane_idx*2;
                const int lim = is_extra ? extra_topk_length : topk_length;
                plan.bar_valid_coord_free[rs.index_buf_idx].wait(rs.index_bar_phase ^ 1);
                int tma_coords[2];
                char valid_mask = 0;
                CUTE_UNROLL
                for (int i = 0; i < 2; ++i) {
                    const int cur_idx = i == 0 ? my_indices.x : my_indices.y;
                    const bool ok = cur_idx >= 0 && abs_pos + i < lim;
                    valid_mask |= ok << i;
                    tma_coords[i] = ok ? cur_idx : -1;          // -1: TMA zero-fills the row
                }
                valid_mask <<= lane_idx % 4 * 2;
                valid_mask |= __shfl_xor_sync(0xFFFFFFFF, valid_mask, 0x1);
                valid_mask |= __shfl_xor_sync(0xFFFFFFFF, valid_mask, 0x2);
                *(int2 *)(plan.tma_coord[rs.index_buf_idx] + lane_idx*2) = *(int2 *)tma_coords;
                if (lane_idx % 4 == 0) plan.is_token_valid[rs.index_buf_idx][lane_idx/4] = valid_mask;
                plan.bar_valid_coord_ready[rs.index_buf_idx].arrive();
                rs.update();
                if (has_next) my_indices = next_indices;
            }
        }
    } else {
        // ================================ dequant warpgroup ================================
        cutlass::arch::warpgroup_reg_alloc<208>();
        plan.bar_q_utccp.wait(bar_phase_batch_rel);
        if constexpr (KV_BF16) {
            // bf16 kv: the tile's 128 gather4s (8 64-dim chunks x 16 four-row groups), one per thread, straight into the
            // SW128 tile once PV of the buffer's previous tile has read it (the UMMA warp waits the gather's transactions)
            const int c = idx_in_warpgroup / 16, g = idx_in_warpgroup % 16;
            for (int block_idx = 0; block_idx < end_block_idx; ++block_idx) {
                plan.bar_valid_coord_ready[rs.index_buf_idx].wait(rs.index_bar_phase);
                plan.bar_sv_done[rs.buf_idx].wait(rs.bar_phase ^ 1);
                if (idx_in_warpgroup == 0) plan.bar_raw_ready[rs.buf_idx].arrive_and_expect_tx(B_TOPK * D_Q * 2);
                const int4 rr = *(const int4 *)(plan.tma_coord[rs.index_buf_idx] + 4 * g);
                NamedBarrier::arrive_and_wait(128, 5);      // every coordinate read; the expect_tx precedes completion
                if (idx_in_warpgroup == 0) plan.bar_valid_coord_free[rs.index_buf_idx].arrive();
                const uint32_t dst = cute::cast_smem_ptr_to_uint(plan.u.kv.dequant[rs.buf_idx].quant_part);
                const uint32_t bar = cute::cast_smem_ptr_to_uint(&plan.bar_raw_ready[rs.buf_idx]);
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4, %5, %6}], [%7];\n"
                    :: "r"(dst + c * (B_TOPK * 128) + g * 4 * 128), "l"(&tma_params.kv_map), "r"(c * 64),
                       "r"(rr.x), "r"(rr.y), "r"(rr.z), "r"(rr.w), "r"(bar) : "memory");
                rs.update();                                 // run ahead: the UMMA warp waits the tile's transactions
            }
        } else {
            const typename L::template Dequantizer<OrigKVFormat> dequant_orig(idx_in_warpgroup);
            const typename L::template Dequantizer<ExtraKVFormat> dequant_extra(idx_in_warpgroup);
            const uint8_t *raw_0 = plan.u.kv.raw_quant[0].data(), *raw_1 = plan.u.kv.raw_quant[1].data();
            const uint32_t dst_0 = cute::cast_smem_ptr_to_uint(plan.u.kv.dequant[0].quant_part);
            const uint32_t dst_1 = cute::cast_smem_ptr_to_uint(plan.u.kv.dequant[1].quant_part);
            for_each_kv_block<OrigKVFormat, ExtraKVFormat>(0, end_block_idx, num_orig_kv_blocks, [&]<typename F>(int, bool) {
                plan.bar_raw_ready[rs.buf_idx].wait(rs.bar_phase);
                plan.bar_sv_done[rs.buf_idx].wait(rs.bar_phase ^ 1);
                const auto *dequant = [&] {
                    if constexpr (std::is_same_v<F, OrigKVFormat>) return &dequant_orig; else return &dequant_extra;
                }();
                const uint8_t *raw = rs.buf_idx == 0 ? raw_0 : raw_1;
                dequant->run(raw, raw + F::QUANT_BYTES, rs.buf_idx == 0 ? dst_0 : dst_1, [] {});
                cutlass::arch::fence_view_async_shared();
                plan.bar_quant_part_dequant_ready[rs.buf_idx].arrive();
                plan.bar_raw_free[rs.buf_idx].arrive();
                rs.update();
            });
        }
    }
}

}  // namespace sparse_sm100::dualrail
