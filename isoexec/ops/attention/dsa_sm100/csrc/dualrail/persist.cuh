// SPDX-License-Identifier: MIT
// IsoExec dsa_sm100: the dual-rail leaf's PERSISTENT execution plan (prefill chunks, trainer rows, large decode batches).
//
// Derived from FlashMLA 395cb3a csrc/cuda_kernels/sm100/prefill/sparse/fwd/head64/{config.h,phase1.cuh} and
// csrc/cuda_kernels/utils.h (RingBufferState) (MIT, Copyright (c) 2025 DeepSeek; dualrail/LICENSE-FlashMLA.txt,
// dualrail/PROVENANCE.md "persist.cuh"). Upstream's persistent schedule is kept: one 384-thread CTA per SM takes rows
// through cluster launch control (CLC), keeps TMEM / barriers / its 3-slot K/V ring across rows, loads the next row's Q
// into the ring slot after the row's last block and overlaps the next row's K/V loads and first QKs with the epilogue.
// Changes, each named in PROVENANCE.md:
//   P1 the lazy-rescale decision is PER ROW (upstream: __any_sync over a warp, 32 heads decide together), exactly as the
//      one-CTA leaf (leaf.cuh change 2): a head's bits never depend on the CTA's other heads;
//   P2 the K/V source: KV_BF16 = upstream's TMA gather4 of bf16 [S, 512] rows; otherwise the V4.1 fp8 528 B records of
//      one or two caches (window | compressed, the leaf's two lists in list order): warp 11 (idle upstream) runs ahead
//      over the ring and TMA-gathers a tile's 64 raw records (the one-CTA leaf's raw map and coordinates) into the TOP
//      of the tile's own slot as soon as the slot is free; the producer warpgroup reads them into registers, syncs, and
//      writes the swizzled bf16 tile over the same slot with the leaf's conversion (fp8x2_to_bf16x2_with_scale). No
//      separate raw buffer, so upstream's 3-slot ring fits, and the load latency is hidden by the ring depth;
//   P3 the leaf's tile structure and validity rule per row: window tiles max(ceil(len / 64), 1), then the extra tiles;
//      slot valid iff index >= 0 and position < length; an invalid slot's K/V row is zero (the TMA zero fill, as the leaf);
//      no TMA skipping of all-invalid tiles (a stale slot could change the sign of a zero product);
//   P4 the leaf's epilogue: sink-inclusive lse (empty row: lse = sink, +inf without one), out = bf16(O * (l == 0 ? 0 :
//      1 / (l + exp2f(sink - m)))) with O always read from TMEM, no max_logits; IsoExec q / out / lse strides; flat
//      token rows; the output store drained before the CTA exits;
//   P5 PDL: griddepcontrol.wait before any global access (launch structure only);
//   P6 the leaf's fused RoPE (fused_rope.cuh, fp8 launches): warp 10 rotates the Q tile's tail in its ring slot before
//      the UTCCP (barrier bar_q_roped), the epilogue applies the inverse rotation to the bf16 output chunks.
// Numerical program: identical, per row, to leaf.cuh (same per-tile subroutines, same dual-rail QK chain order -- the
// two accumulating halves {0,2}|{1,3} then {4,6}|{5,7} continue one K16 chain --, same PV chain, same rounding points);
// relation tests/test_dualrail_persist_gpu.py: persistent plan == one-CTA leaf, bit for bit.
#pragma once

#include "leaf.cuh"

namespace dsa_sm100::dualrail {

// FlashMLA csrc/cuda_kernels/utils.h, verbatim
struct RingBufferState {
    uint32_t cur_block_idx = 0u;
    __device__ __forceinline__ void update() { cur_block_idx += 1; }
    template<uint32_t NUM_STAGES>
    __device__ __forceinline__ std::pair<uint32_t, bool> get() const {
        uint32_t stage_idx = cur_block_idx % NUM_STAGES;
        bool phase = (cur_block_idx / NUM_STAGES) & 1;
        return {stage_idx, phase};
    }
    __device__ __forceinline__ RingBufferState offset_by(const int offset) const {
        RingBufferState s;
        s.cur_block_idx = static_cast<uint32_t>(static_cast<int>(cur_block_idx) + offset);
        return s;
    }
};

template<bool KV_BF16, bool EXTRA>
struct Persist {
    static_assert(!(KV_BF16 && EXTRA));
    static constexpr int MMA_M = 64;
    static constexpr int D_Q = 512, D_V = 512;
    static constexpr int B_TOPK = 64;
    static constexpr int NUM_BUFS = 3;
    static constexpr int NUM_THREADS = 128 * 3;
    static constexpr int NUM_WORKER_THREADS = 128 + 128 + 1 + B_TOPK/8 + 1 + 32 + 32;   // + warp 10 (P6), warp 11 (P2)
    static constexpr int B_EPI = 64, B_EPI_SB = 256;
    static constexpr float MAX_INIT_VAL = -1e30f;
    static constexpr float RESCALE_THRESHOLD = 6.0f;
    // P2: a tile's raw records in the top of its slot, the one-CTA leaf's raw rows (RAW_TOKEN_SMEM_STRIDE: 512 code bytes +
    // 64; the 16 scale bytes at 512, the rest of the box zero-filled)
    static constexpr int RAW_STRIDE = 512 + 64;
    static constexpr int RAW_OFFSET = B_TOPK*D_V*2 - B_TOPK*RAW_STRIDE;
    static_assert(RAW_OFFSET % 128 == 0 && RAW_OFFSET >= 0);

    struct tmem_cols { static constexpr int O = 0, Q = 256, P = 400; };

    using SmemLayoutQ = decltype(coalesce(tile_to_shape(UMMA::Layout_K_SW128_Atom<bf16>{}, Shape<Int<MMA_M>, Int<D_V>>{}, Step<_1, _2>{}), Shape<_1, _1>{}));
    template<int NUM_TILES>
    using SmemLayoutOTiles = decltype(coalesce(tile_to_shape(UMMA::Layout_K_SW128_Atom<bf16>{}, Shape<Int<MMA_M>, Int<B_EPI*NUM_TILES>>{}, Step<_1, _2>{}), Shape<_1, _1>{}));
    using SmemLayoutO = SmemLayoutOTiles<D_V/B_EPI>;
    template<int NUM_TILES>
    using SmemLayoutKTiles = decltype(coalesce(tile_to_shape(UMMA::Layout_K_SW128_Atom<bf16>{}, Shape<Int<B_TOPK>, Int<64*NUM_TILES>>{}, Step<_1, _2>{}), Shape<_1, _1>{}));
    using SmemLayoutK = SmemLayoutKTiles<8>;
    using SmemLayoutV = decltype(coalesce(composition(SmemLayoutK{}, Layout<Shape<Int<D_V>, Int<B_TOPK>>, Stride<Int<B_TOPK>, _1>>{}), Shape<_1, _1>{}));
    using SmemLayoutK_TiledMMA = decltype(coalesce(tile_to_shape(UMMA::Layout_K_SW128_Atom<bf16>{}, Shape<Int<B_TOPK*2>, Int<D_V/2>>{}, Step<_1, _2>{}), Shape<_1, _1>{}));
    using SmemLayoutS = decltype(coalesce(tile_to_shape(UMMA::Layout_K_INTER_Atom<bf16>{}, Shape<Int<MMA_M>, Int<B_TOPK>>{}, Step<_1, _2>{}), Shape<_1, _1>{}));

    struct SharedMemoryPlan {
        array_aligned<bf16, B_TOPK*D_V> qko_slots[NUM_BUFS];
        float p_exchange_buf[4][32 * (B_TOPK/(128/MMA_M))];
        bf16 s[MMA_M*B_TOPK];
        char is_k_valid[NUM_BUFS][B_TOPK/8];
        transac_bar_t bar_prologue_q, bar_prologue_utccp, bar_q_roped;
        transac_bar_t bar_qk_done[NUM_BUFS];
        transac_bar_t bar_sv_done[NUM_BUFS];
        transac_bar_t bar_kv_ready[NUM_BUFS][2];
        transac_bar_t bar_raw_ready[NUM_BUFS];      // P2: the tile's raw records landed in its slot
        transac_bar_t bar_p_free;
        transac_bar_t bar_so_ready;
        transac_bar_t bar_o_write_back_done;
        transac_bar_t bar_o_write_back_done_waited;
        transac_bar_t bar_k_valid_ready[NUM_BUFS], bar_k_valid_free[NUM_BUFS];
        transac_bar_t bar_clc_full, bar_clc_empty;
        array_aligned<uint32_t, 1> tmem_start_addr;
        float rowwise_max_buf[128], rowwise_li_buf[128];
        ku::CLCResponseObj clc_response_obj;
    };

    static constexpr int QK_MRGEMM_N = (128/64)*B_TOPK;
    static constexpr int QK_MRGEMM_K = D_V / (128/64);
    using TiledMMA_P = decltype(make_tiled_mma(SM100_MMA_F16BF16_WS_TS_NOELECT<bf16, bf16, float, MMA_M, QK_MRGEMM_N, UMMA::Major::K, UMMA::Major::K>{}));
    using TiledMMA_O = decltype(make_tiled_mma(SM100_MMA_F16BF16_WS_SS_NOELECT<bf16, bf16, float, MMA_M, 256, UMMA::Major::K, UMMA::Major::MN>{}));

    template<typename Shape_Q, typename TMA_Q, typename Shape_O, typename TMA_O>
    struct TmaParams {
        Shape_Q shape_Q; TMA_Q tma_Q;
        Shape_O shape_O; TMA_O tma_O;
        CUtensorMap kv_map;         // KV_BF16: [S, 512] box (64, 1) SW128; fp8: the leaf's raw-record map
        CUtensorMap extra_map;      // fp8 extra (compressed) cache: the leaf's raw-record map
    };
};

template<typename L, typename TmaParam, bool KV_BF16, bool EXTRA>
__global__ void __launch_bounds__(L::NUM_THREADS, 1, 1)
dualrail_persist_kernel(__grid_constant__ const Params params, __grid_constant__ const TmaParam tma_params) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1200))
    asm volatile("griddepcontrol.wait;" ::: "memory");     // P5
    constexpr int MMA_M = L::MMA_M, B_TOPK = L::B_TOPK, NUM_BUFS = L::NUM_BUFS;
    constexpr int D_V = L::D_V, B_EPI = L::B_EPI, B_EPI_SB = L::B_EPI_SB;
    using tmem_cols = typename L::tmem_cols;

    const int warp_idx = cutlass::canonical_warp_idx_sync();
    const int lane_idx = threadIdx.x % 32;
    const int warpgroup_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
    const int idx_in_warpgroup = threadIdx.x % 128;

    extern __shared__ char wksp_buf[];
    typename L::SharedMemoryPlan &plan = *reinterpret_cast<typename L::SharedMemoryPlan *>(wksp_buf);

    if (warp_idx == 0 && elect_one_sync()) {
        cute::prefetch_tma_descriptor(tma_params.tma_Q.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_O.get_tma_descriptor());
        cute::prefetch_tma_descriptor(&tma_params.kv_map);
        if constexpr (EXTRA) cute::prefetch_tma_descriptor(&tma_params.extra_map);
        plan.bar_prologue_q.init(1);
        plan.bar_prologue_utccp.init(1);
        plan.bar_q_roped.init(32);
        plan.bar_clc_full.init(1);
        plan.bar_clc_empty.init(L::NUM_WORKER_THREADS);
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 1 && elect_one_sync()) {
        CUTE_UNROLL
        for (int i = 0; i < NUM_BUFS; ++i) {
            plan.bar_qk_done[i].init(1);
            plan.bar_sv_done[i].init(1);
            // bf16: the MMA warp's expect_tx + the TMA bytes; fp8 (P2): the 128 producer threads' arrivals
            plan.bar_kv_ready[i][0].init(KV_BF16 ? 1 : 128);
            plan.bar_kv_ready[i][1].init(KV_BF16 ? 1 : 128);
            plan.bar_raw_ready[i].init(1);
            plan.bar_k_valid_ready[i].init(B_TOPK/8);
            plan.bar_k_valid_free[i].init(128);
        }
        plan.bar_p_free.init(128);
        plan.bar_so_ready.init(128);
        plan.bar_o_write_back_done.init(128);
        plan.bar_o_write_back_done_waited.init(KV_BF16 ? 4 : 4 + 1);     // fp8: + the raw issuer (warp 11)
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 2) {
        cute::TMEM::Allocator1Sm().allocate(512, plan.tmem_start_addr.data());
        KU_TRAP_ONLY_DEVICE_ASSERT(plan.tmem_start_addr.data()[0] == 0);
        cute::TMEM::Allocator1Sm().release_allocation_lock();
    }
    __syncthreads();

    // P3: the one-CTA leaf's tile structure of a row (window tiles, then the extra tiles)
    struct OuterloopArgs {
        bool outer_loop_phase;
        int row;
        int num_k_blocks, num_orig_blocks;
        int topk_length, extra_topk_length;
    };

    auto issue_q_tma = [&](int row, int slot) {
        Tensor gQ = tma_params.tma_Q.get_tma_tensor(tma_params.shape_Q)(_, _, row);
        Tensor sQ = make_tensor(make_smem_ptr(plan.qko_slots[slot].data()), typename L::SmemLayoutQ{});
        ku::launch_tma_copy(tma_params.tma_Q, gQ, sQ, plan.bar_prologue_q, TMA::CacheHintSm90::EVICT_FIRST);
    };
    auto issue_q_utccp = [&](bool phase, int slot) {
        plan.bar_prologue_q.arrive_and_expect_tx(MMA_M*D_V*sizeof(bf16));
        plan.bar_prologue_q.wait(phase);
        if (params.rope.q) plan.bar_q_roped.wait(phase);         // P6: warp 10 rotated the tile's tail
        ku::tcgen05_after_thread_sync();
        UMMA::SmemDescriptor sQ_desc = UMMA::make_umma_desc<UMMA::Major::K>(make_tensor(
            make_smem_ptr(plan.qko_slots[slot].data()), tile_to_shape(UMMA::Layout_K_SW128_Atom<bf16>{}, Shape<Int<MMA_M*2>, Int<64>>{})));
        CUTE_UNROLL
        for (int tile_idx = 0; tile_idx < D_V/64/2; ++tile_idx) {
            CUTE_UNROLL
            for (int subtile_idx = 0; subtile_idx < 4; ++subtile_idx)
                SM100_UTCCP_128dp256bit_1cta::copy(sQ_desc + (tile_idx*(MMA_M*128*2) + subtile_idx*32) / 16,
                                                   tmem_cols::Q + tile_idx*32 + subtile_idx*8);
        }
        ku::umma_arrive_noelect(plan.bar_prologue_utccp);
    };

    auto run_outer_loop = [&](auto loop_body) {
        int outer_loop_phase = false;
        ku::CLCResult next_job = {true, (int)blockIdx.x, 0, 0};
        CUTE_NO_UNROLL
        while (next_job.is_valid) {
            const int row = next_job.x;
            const int topk_length = params.topk_length ? __ldg(params.topk_length + row) : params.topk;
            const int extra_topk_length = EXTRA ? (params.extra_topk_length ? __ldg(params.extra_topk_length + row) : params.extra_topk) : 0;
            const int num_orig = max(ku::ceil(topk_length, (int)B_TOPK), (int)B_TOPK) / B_TOPK;
            const int num_k_blocks = num_orig + ku::ceil(extra_topk_length, (int)B_TOPK) / B_TOPK;
            KU_TRAP_ONLY_DEVICE_ASSERT(num_k_blocks >= 2);      // the host admits rows of >= 2 tiles only
            OuterloopArgs args = {(bool)outer_loop_phase, row, num_k_blocks, num_orig, topk_length, extra_topk_length};
            loop_body(args);
            plan.bar_clc_full.wait(outer_loop_phase);
            next_job = ku::get_clc_query_response<true>(plan.clc_response_obj);
            outer_loop_phase ^= 1;
            plan.bar_clc_empty.arrive();
        }
    };

    // list slot block k of a row: (list pointer, position base, length) -- window list, then the extra list
    struct BlockRef { const int *ids; int pos0, lim; bool extra; };
    auto block_ref = [&](const OuterloopArgs &a, int k) -> BlockRef {
        if (!EXTRA || k < a.num_orig_blocks)
            return {params.idx + params.idx_stride * a.row + k * B_TOPK, k * B_TOPK, a.topk_length, false};
        const int e = k - a.num_orig_blocks;
        return {params.extra_idx + params.extra_idx_stride * a.row + e * B_TOPK, e * B_TOPK, a.extra_topk_length, true};
    };

    RingBufferState rs;

    if (warpgroup_idx == 0) {
        // ======================= scale & exp warpgroup (softmax, O correction, epilogue) =======================
        bf16 *sS_base = plan.s + lane_idx*8 + (warp_idx&1)*(MMA_M/2)*8 + (warp_idx/2)*MMA_M*(B_TOPK/2);
        static constexpr int NUM_ELEMS_PER_THREAD = B_TOPK / (128/MMA_M);
        // padding rows (head >= h_q) read no sink; their output is never stored
        const float attn_sink = (params.sink == nullptr || idx_in_warpgroup % MMA_M >= params.h_q)
            ? -CUDART_INF_F : __ldg(params.sink + (idx_in_warpgroup % MMA_M)) * CUDART_L2E_F;

        run_outer_loop([&](const OuterloopArgs &args) {
            float mi = L::MAX_INIT_VAL;
            float li = 0.0f;
            float real_mi = -CUDART_INF_F;

            cute::tma_store_wait<0>();
            plan.bar_o_write_back_done.arrive();

            CUTE_NO_UNROLL
            for (int k = 0; k < args.num_k_blocks; ++k) {
                NamedBarrier::arrive_and_wait(64, NamedBarriers::wg0_warp02_sync + (warp_idx&1));
                auto [buf_idx, bar_phase] = rs.get<NUM_BUFS>();
                plan.bar_qk_done[buf_idx].wait(bar_phase);
                plan.bar_k_valid_ready[buf_idx].wait(bar_phase);
                ku::tcgen05_after_thread_sync();

                float p[NUM_ELEMS_PER_THREAD];
                retrieve_mask_and_reduce_p<MMA_M, NUM_ELEMS_PER_THREAD, NamedBarriers::wg0_warp02_sync, NamedBarriers::wg0_warp13_sync>(
                    tmem_cols::P, plan.is_k_valid[buf_idx], warp_idx, lane_idx, [&]() { plan.bar_p_free.arrive(); },
                    plan.p_exchange_buf, p);
                plan.bar_k_valid_free[buf_idx].arrive();

                float cur_pi_max = get_max<NUM_ELEMS_PER_THREAD>(p);
                cur_pi_max *= params.scale_log2;
                plan.rowwise_max_buf[idx_in_warpgroup] = cur_pi_max;
                NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
                cur_pi_max = max(cur_pi_max, plan.rowwise_max_buf[idx_in_warpgroup^64]);
                real_mi = max(real_mi, cur_pi_max);
                // P1: the lazy-rescale decision is the ROW's (identical in the row's two threads)
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

                nv_bfloat162 s[NUM_ELEMS_PER_THREAD/2];
                const float cur_sum = get_s_from_p<NUM_ELEMS_PER_THREAD>(s, p, params.scale_log2, new_max);
                li = fma(li, scale_for_old, cur_sum);

                if (k > 0) {
                    auto [last_sv_buf_idx, last_sv_bar_phase] = rs.offset_by(-1).get<NUM_BUFS>();
                    plan.bar_sv_done[last_sv_buf_idx].wait(last_sv_bar_phase);
                }
                CUTE_UNROLL
                for (int i = 0; i < NUM_ELEMS_PER_THREAD/8; i += 1)
                    *(uint128_t *)(sS_base + MMA_M*8*i) = *(uint128_t *)(s + i*4);

                // O *= a: the TMEM loads / stores are warp-collective, so a warp runs the correction when any of its rows
                // rescales; a row that does not multiplies by exactly 1.0f (an identity on every float)
                if (k > 0 && __any_sync(0xffffffff, row_scale)) {
                    ku::tcgen05_after_thread_sync();
                    rescale_O<D_V/(128/MMA_M), 32, tmem_cols::O>(scale_for_old);
                    ku::tcgen05_before_thread_sync();
                }
                fence_view_async_shared();
                plan.bar_so_ready.arrive();
                rs.update();
            }

            plan.bar_o_write_back_done_waited.wait(args.outer_loop_phase);

            if (real_mi == -CUDART_INF_F) {
                li = 0.0f;
                mi = -CUDART_INF_F;
            }
            plan.rowwise_li_buf[idx_in_warpgroup] = li;
            NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
            li += plan.rowwise_li_buf[idx_in_warpgroup^64];

            // P4: the leaf's epilogue (dsa_sm100 README.md: sink-inclusive lse, an empty row has out 0, lse = sink)
            const float denom = li + exp2f(attn_sink - mi);
            if (idx_in_warpgroup < params.h_q) {
                float cur_lse;
                if (li == 0.0f) cur_lse = params.sink ? __ldg(params.sink + idx_in_warpgroup) : +CUDART_INF_F;
                else cur_lse = fma(mi, CUDART_LN2_F, logf(denom));
                params.lse[(int64_t)args.row * params.lse_row_stride + idx_in_warpgroup] = cur_lse;
            }
            const float o_scale = li == 0.0f ? 0.0f : __fdividef(1.0f, denom);
            const float2 o_scale_float2 = make_float2(o_scale, o_scale);

            auto [o_slot_idx, __] = rs.offset_by(+2).get<NUM_BUFS>();
            Tensor sO = make_tensor(make_smem_ptr(plan.qko_slots[o_slot_idx].data()), typename L::SmemLayoutO{});
            Tensor tma_gO = flat_divide(tma_params.tma_O.get_tma_tensor(tma_params.shape_O)(_, _, args.row), Shape<Int<MMA_M>, Int<B_EPI>>{})(_, _, _0{}, _);
            Tensor sO_divided = flat_divide(sO, Shape<Int<MMA_M>, Int<B_EPI>>{})(_, _, _0{}, _);
            auto thr_tma = tma_params.tma_O.get_slice(_0{});
            bf16 *sO_addrs[B_EPI/8];
            CUTE_UNROLL
            for (int i = 0; i < B_EPI/8; ++i) sO_addrs[i] = &sO(idx_in_warpgroup%MMA_M, i*8);

            {
                auto [last_sv_buf_idx, last_sv_bar_phase] = rs.offset_by(-1).get<NUM_BUFS>();
                plan.bar_sv_done[last_sv_buf_idx].wait(last_sv_bar_phase);
                ku::tcgen05_after_thread_sync();
            }

            static constexpr int NUM_EPI_SB = D_V/B_EPI_SB;
            static constexpr int NUM_TMA_PARTS = 128 / MMA_M;
            float2 o[B_EPI/2];
            CUTE_UNROLL
            for (int c = 0; c < NUM_EPI_SB; ++c) {
                CUTE_UNROLL
                for (int kk = 0; kk < B_EPI_SB/B_EPI/NUM_TMA_PARTS; ++kk) {
                    ku::tmem_ld_32dp32bNx<B_EPI>(tmem_cols::O + c*(B_EPI_SB/NUM_TMA_PARTS) + kk*B_EPI, o);
                    cutlass::arch::fence_view_async_tmem_load();
                    CUTE_UNROLL
                    for (int i = 0; i < B_EPI/8; ++i) {
                        nv_bfloat162 o_bf16[4];
                        CUTE_UNROLL
                        for (int j = 0; j < 4; ++j) {
                            o[i*4+j] = ku::float2_mul(o[i*4+j], o_scale_float2);
                            o_bf16[j] = __float22bfloat162_rn(o[i*4+j]);
                        }
                        const int col0 = c*B_EPI_SB + (idx_in_warpgroup/MMA_M)*(B_EPI_SB/(128/MMA_M)) + kk*B_EPI;
                        if (params.rope.o)       // P6: inverse RoPE of the bf16 output's tail (warp-uniform columns)
                            rope_o_chunk(o_bf16, col0 + i*8, params.rope, args.row);
                        bf16 *sO_ptr = sO_addrs[i] + col0*MMA_M;
                        ku::st_shared(sO_ptr, *(__int128_t *)(o_bf16));
                    }
                    fence_view_async_shared();
                    NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
                    if (warp_idx < NUM_TMA_PARTS && elect_one_sync()) {
                        const int epi_chunk_idx = c*(B_EPI_SB/B_EPI) + ((B_EPI_SB/B_EPI)/NUM_TMA_PARTS)*warp_idx + kk;
                        cute::copy(tma_params.tma_O, thr_tma.partition_S(sO_divided(_, _, epi_chunk_idx)),
                                   thr_tma.partition_D(tma_gO(_, _, epi_chunk_idx)));
                    }
                }
            }
            cute::tma_store_arrive();
        });
        cute::tma_store_wait<0>();              // P4: the last row's store drained before the CTA exits
        // free at the allocated base (R3: read from the slot, as every allocation site does; a 512-column
        // allocation can only start at column 0, asserted at allocation, so the instruction's operand is unchanged)
        if (warp_idx == 3) cute::TMEM::Allocator1Sm().free(plan.tmem_start_addr.data()[0], 512);
    } else if (warpgroup_idx == 1) {
        // ================================ K/V producer warpgroup ================================
        if constexpr (KV_BF16) {
            // upstream's producer: per warp 16 rows of the tile as 4 gather4s x 8 64-dim chunks, two halves (P3: invalid
            // slots get coordinate -1, a zero row by the TMA out-of-bounds fill; no skipping)
            const int pwarp = cutlass::canonical_warp_idx_sync() - 4;
            constexpr int NUM_WARPS = 4, NUM_LOCAL_ROWS_PER_WARP = (B_TOPK/4)/NUM_WARPS;
            run_outer_loop([&](const OuterloopArgs &args) {
                if (elect_one_sync()) {
                    CUTE_NO_UNROLL
                    for (int k = 0; k < args.num_k_blocks; ++k) {
                        const BlockRef br = block_ref(args, k);
                        int4 indices[NUM_LOCAL_ROWS_PER_WARP];
                        CUTE_UNROLL
                        for (int lr = 0; lr < NUM_LOCAL_ROWS_PER_WARP; ++lr) {
                            const int s0 = (lr*NUM_WARPS + pwarp) * 4;
                            int4 v = __ldg((const int4 *)(br.ids + s0));
                            v.x = (v.x >= 0 && br.pos0 + s0 + 0 < br.lim) ? v.x : -1;
                            v.y = (v.y >= 0 && br.pos0 + s0 + 1 < br.lim) ? v.y : -1;
                            v.z = (v.z >= 0 && br.pos0 + s0 + 2 < br.lim) ? v.z : -1;
                            v.w = (v.w >= 0 && br.pos0 + s0 + 3 < br.lim) ? v.w : -1;
                            indices[lr] = v;
                        }
                        if (k == 1) plan.bar_prologue_utccp.wait(args.outer_loop_phase);       // Q shares the slot
                        else if (k == 2) plan.bar_o_write_back_done.wait(args.outer_loop_phase);  // O staging shares it
                        auto [buf_idx, bar_phase] = rs.get<NUM_BUFS>();
                        plan.bar_sv_done[buf_idx].wait(bar_phase^1);
                        bf16 *sK_base = plan.qko_slots[buf_idx].data() + pwarp*4*64;
                        CUTE_UNROLL
                        for (int part_idx = 0; part_idx < 2; ++part_idx) {
                            CUTE_UNROLL
                            for (int lr = 0; lr < NUM_LOCAL_ROWS_PER_WARP; ++lr) {
                                CUTE_UNROLL
                                for (int local_col = part_idx*(D_V/2/64); local_col < (part_idx+1)*(D_V/2/64); ++local_col)
                                    ku::tma_gather4(&tma_params.kv_map, plan.bar_kv_ready[buf_idx][part_idx],
                                                    sK_base + lr*(4*NUM_WARPS)*64 + local_col*(B_TOPK*64), local_col*64,
                                                    indices[lr], (int64_t)TMA::CacheHintSm90::EVICT_LAST);
                            }
                        }
                        rs.update();
                    }
                    if (args.num_k_blocks <= 2) plan.bar_o_write_back_done.wait(args.outer_loop_phase);
                    plan.bar_o_write_back_done_waited.arrive();
                }
                __syncwarp();
            });
        } else {
            // P2: thread (g, i) = (idx / 8, idx % 8) converts tile rows g + 16 lr (lr < 4), 8 elements [64 s + 8 i, +8) per
            // 64-dim chunk s with the chunk's scale, from the raw rows warp 11 gathered into the top of the slot; all reads
            // precede every write (the bf16 tile overlaps the raw rows); the destination is KVBlockDequantizer's
            const int g = idx_in_warpgroup / 8, ii = idx_in_warpgroup % 8;
            const uint32_t dst_off = (g / 8) * (8 * 128) + (g % 8) * 128 + ((ii ^ (g % 8)) * 16);
            run_outer_loop([&](const OuterloopArgs &args) {
                CUTE_NO_UNROLL
                for (int k = 0; k < args.num_k_blocks; ++k) {
                    auto [buf_idx, bar_phase] = rs.get<NUM_BUFS>();
                    plan.bar_raw_ready[buf_idx].wait(bar_phase);
                    const uint8_t *raw = reinterpret_cast<const uint8_t *>(plan.qko_slots[buf_idx].data()) + L::RAW_OFFSET;
                    uint2 words[4][8];
                    uint32_t scales[4][4];
                    CUTE_UNROLL
                    for (int lr = 0; lr < 4; ++lr) {
                        const uint8_t *rec = raw + (g + 16 * lr) * L::RAW_STRIDE;
                        CUTE_UNROLL
                        for (int s = 0; s < 8; ++s) words[lr][s] = *(const uint2 *)(rec + s * 64 + ii * 8);
                        const uint4 sc = *(const uint4 *)(rec + 512);
                        scales[lr][0] = sc.x; scales[lr][1] = sc.y; scales[lr][2] = sc.z; scales[lr][3] = sc.w;
                    }
                    NamedBarrier::arrive_and_wait(128, 4);       // every raw row read before the slot is overwritten
                    const uint32_t dst = cute::cast_smem_ptr_to_uint(plan.qko_slots[buf_idx].data()) + dst_off;
                    CUTE_UNROLL
                    for (int part_idx = 0; part_idx < 2; ++part_idx) {
                        CUTE_UNROLL
                        for (int lr = 0; lr < 4; ++lr) {
                            CUTE_UNROLL
                            for (int s = part_idx * 4; s < part_idx * 4 + 4; ++s) {
                                const int scale_idx = 2 * s + (ii >= 4 ? 1 : 0);
                                const uint8_t sbyte = (uint8_t)(scales[lr][scale_idx / 4] >> (8 * (scale_idx % 4)));
                                __nv_fp8_e8m0 se;
                                *(uint8_t *)&se = sbyte;
                                ku::nvbf16x2 data_bf16[4];
                                CUTE_UNROLL
                                for (int i = 0; i < 4; ++i)
                                    data_bf16[i] = fp8x2_to_bf16x2_with_scale(((ku::nve4m3x2 *)&words[lr][s])[i], se);
                                asm volatile("st.weak.shared::cta.b128 [%0], %1;\n"
                                             :: "r"(dst + lr * 16 * 128 + s * B_TOPK * 128), "q"(*(__int128_t *)data_bf16));
                            }
                        }
                        cutlass::arch::fence_view_async_shared();
                        plan.bar_kv_ready[buf_idx][part_idx].arrive();
                    }
                    rs.update();
                }
                if (args.num_k_blocks <= 2) plan.bar_o_write_back_done.wait(args.outer_loop_phase);
                __syncwarp();
                if (elect_one_sync()) plan.bar_o_write_back_done_waited.arrive();
                __syncwarp();
            });
        }
    } else {
        if (warp_idx == 8 && elect_one_sync()) {
            // ================================ UMMA warp ================================
            typename L::TiledMMA_P tiled_mma_P{};
            typename L::TiledMMA_O tiled_mma_O{};
            Tensor tP = partition_fragment_C(tiled_mma_P, Shape<Int<MMA_M>, Int<B_TOPK*2>>{});
            Tensor tQ_part0 = tiled_mma_P.get_slice(_0{}).make_fragment_A(partition_shape_A(tiled_mma_P, Shape<Int<MMA_M>, Int<L::QK_MRGEMM_K/2>>{}));
            Tensor tQ_part1 = tiled_mma_P.get_slice(_0{}).make_fragment_A(partition_shape_A(tiled_mma_P, Shape<Int<MMA_M>, Int<L::QK_MRGEMM_K/2>>{}));
            Tensor tO = partition_fragment_C(tiled_mma_O, Shape<Int<MMA_M>, Int<D_V>>{});
            tP.data().get() = tmem_cols::P;
            tQ_part0.data().get() = tmem_cols::Q;
            tQ_part1.data().get() = tmem_cols::Q + 64;
            tO.data().get() = tmem_cols::O;

            run_outer_loop([&](const OuterloopArgs &args) {
                // Q into the slot after the row's first block (free: see upstream's note on the last O gemm)
                const int q_slot_idx = rs.offset_by(+1).get<NUM_BUFS>().first;
                issue_q_tma(args.row, q_slot_idx);
                issue_q_utccp(args.outer_loop_phase, q_slot_idx);

                CUTE_NO_UNROLL
                for (int k = 0; k < args.num_k_blocks+1; ++k) {
                    if (k < args.num_k_blocks) {
                        auto [buf_idx, bar_phase] = rs.get<NUM_BUFS>();
                        Tensor sK = make_tensor(make_smem_ptr(plan.qko_slots[buf_idx].data()), typename L::SmemLayoutK_TiledMMA{});
                        auto [__, binary_bar_phase] = rs.get<1>();
                        plan.bar_p_free.wait(binary_bar_phase^1);
                        ku::tcgen05_after_thread_sync();
                        if (k == 0) plan.bar_prologue_utccp.wait(args.outer_loop_phase);
                        Tensor sK_divided = flat_divide(sK, Tile<Int<B_TOPK*2>, Int<D_V/4>>{})(_, _, _0{}, _);
                        CUTE_UNROLL
                        for (int kv_part_idx = 0; kv_part_idx < 2; ++kv_part_idx) {
                            if constexpr (KV_BF16) plan.bar_kv_ready[buf_idx][kv_part_idx].arrive_and_expect_tx(B_TOPK*D_V/2*sizeof(bf16));
                            plan.bar_kv_ready[buf_idx][kv_part_idx].wait(bar_phase);
                            ku::tcgen05_after_thread_sync();
                            // P += Q K^T: rail r over 64-dim chunks {r, r+2} then {r+4, r+6}, one K16 chain
                            ku::utcmma_ts(tiled_mma_P, kv_part_idx ? tQ_part1 : tQ_part0, sK_divided(_, _, kv_part_idx), tP, kv_part_idx == 0);
                        }
                        ku::umma_arrive_noelect(plan.bar_qk_done[buf_idx]);
                    }
                    if (k > 0) {
                        auto [buf_idx, bar_phase] = rs.offset_by(-1).get<NUM_BUFS>();
                        Tensor sS = make_tensor(make_smem_ptr(plan.s), typename L::SmemLayoutS{});
                        Tensor sV = make_tensor(make_smem_ptr(plan.qko_slots[buf_idx].data()), typename L::SmemLayoutV{});
                        auto [__, binary_bar_phase] = rs.offset_by(-1).get<1>();
                        plan.bar_so_ready.wait(binary_bar_phase);
                        ku::tcgen05_after_thread_sync();
                        ku::utcmma_ss(tiled_mma_O, sS, sV, tO, k == 1);
                        ku::umma_arrive_noelect(plan.bar_sv_done[buf_idx]);
                    }
                    rs.update();
                }
                rs = rs.offset_by(-1);
            });
        } else if (warp_idx == 9) {
            if (lane_idx < B_TOPK/8) {
                // validity masks (P3: the leaf's rule), 8 slots per lane
                run_outer_loop([&](const OuterloopArgs &args) {
                    CUTE_NO_UNROLL
                    for (int k = 0; k < args.num_k_blocks; ++k) {
                        const BlockRef br = block_ref(args, k);
                        const int4 a = __ldg((const int4 *)(br.ids + lane_idx*8));
                        const int4 b = __ldg((const int4 *)(br.ids + lane_idx*8 + 4));
                        const int v[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
                        char mask = 0;
                        CUTE_UNROLL
                        for (int i = 0; i < 8; ++i) mask |= (char)((v[i] >= 0 && br.pos0 + lane_idx*8 + i < br.lim) << i);
                        auto [buf_idx, bar_phase] = rs.get<NUM_BUFS>();
                        plan.bar_k_valid_free[buf_idx].wait(bar_phase^1);
                        plan.is_k_valid[buf_idx][lane_idx] = mask;
                        plan.bar_k_valid_ready[buf_idx].arrive();
                        rs.update();
                    }
                });
            } else if (lane_idx == B_TOPK/8) {
                // CLC producer: work assignment only (a row's program never depends on which CTA runs it)
                run_outer_loop([&](const OuterloopArgs &args) {
                    plan.bar_clc_empty.wait(args.outer_loop_phase^1);
                    ku::issue_clc_query(plan.bar_clc_full, plan.clc_response_obj);
                    plan.bar_clc_full.arrive_and_expect_tx(sizeof(plan.clc_response_obj));
                });
            }
        } else if (warp_idx == 10) {
            // P6: the fused Q RoPE of the row's Q tile in its ring slot (the slot after the row's first block, as the MMA
            // warp's issue_q_tma), between the Q TMA and the UTCCP
            run_outer_loop([&](const OuterloopArgs &args) {
                if (params.rope.q) {
                    const int q_slot_idx = rs.offset_by(+1).get<NUM_BUFS>().first;
                    plan.bar_prologue_q.wait(args.outer_loop_phase);
                    Tensor sQ = make_tensor(make_smem_ptr(plan.qko_slots[q_slot_idx].data()), typename L::SmemLayoutQ{});
                    rope_q_tile(sQ, params.h_q, params.rope, args.row, lane_idx);
                    fence_view_async_shared();      // generic-proxy writes -> the tcgen05.cp reads
                    plan.bar_q_roped.arrive();
                }
                for (int k = 0; k < args.num_k_blocks; ++k) rs.update();
            });
        } else if (warp_idx == 11) {
            // P2: the raw-record issuer (fp8): per tile, once its slot is free, lanes 0-15 each TMA-gather 4 records
            // (coordinate -1 for an invalid slot: the TMA zero fill, the one-CTA leaf's rule) into the slot's top
            run_outer_loop([&](const OuterloopArgs &args) {
                if constexpr (!KV_BF16) {
                    CUTE_NO_UNROLL
                    for (int k = 0; k < args.num_k_blocks; ++k) {
                        const BlockRef br = block_ref(args, k);
                        int4 c = make_int4(-1, -1, -1, -1);
                        if (lane_idx < 16) {
                            c = __ldg((const int4 *)(br.ids + lane_idx * 4));
                            const int p0 = br.pos0 + lane_idx * 4;
                            c.x = (c.x >= 0 && p0 + 0 < br.lim) ? c.x : -1;
                            c.y = (c.y >= 0 && p0 + 1 < br.lim) ? c.y : -1;
                            c.z = (c.z >= 0 && p0 + 2 < br.lim) ? c.z : -1;
                            c.w = (c.w >= 0 && p0 + 3 < br.lim) ? c.w : -1;
                        }
                        if (k == 1) plan.bar_prologue_utccp.wait(args.outer_loop_phase);       // Q shares the slot
                        else if (k == 2) plan.bar_o_write_back_done.wait(args.outer_loop_phase);  // O staging shares it
                        auto [buf_idx, bar_phase] = rs.get<NUM_BUFS>();
                        plan.bar_sv_done[buf_idx].wait(bar_phase^1);
                        if (lane_idx == 0) plan.bar_raw_ready[buf_idx].arrive_and_expect_tx(B_TOPK * L::RAW_STRIDE);
                        __syncwarp();
                        if (lane_idx < 16) {
                            uint8_t *dst = reinterpret_cast<uint8_t *>(plan.qko_slots[buf_idx].data()) + L::RAW_OFFSET
                                           + lane_idx * 4 * L::RAW_STRIDE;
                            ku::tma_gather4(br.extra ? &tma_params.extra_map : &tma_params.kv_map, plan.bar_raw_ready[buf_idx],
                                            dst, 0, c, (int64_t)TMA::CacheHintSm90::EVICT_LAST);
                        }
                        rs.update();
                    }
                    if (args.num_k_blocks <= 2) plan.bar_o_write_back_done.wait(args.outer_loop_phase);
                    if (lane_idx == 0) plan.bar_o_write_back_done_waited.arrive();
                }
            });
        }
    }
#endif
}

}  // namespace dsa_sm100::dualrail
