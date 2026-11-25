#ifndef ISOEXEC_DUALRAIL_TRIGGER_AFTER_ALLOC
#define ISOEXEC_DUALRAIL_TRIGGER_AFTER_ALLOC 1
#endif
// SPDX-License-Identifier: MIT
// IsoExec dsa_sm100: the dual-rail leaf's SPLIT execution plan (small decode batches). Same recipe, same bits.
//
// The one-CTA plan (leaf.cuh) runs a row's whole list in one CTA: at decode batch B only B CTAs are busy. This plan
// spreads the row over more CTAs WITHOUT changing a single arithmetic step of the per-row program:
//
//   qk_kernel  (grid rows x ceil(T / tpc)): the leaf's own gather -> dequant -> dual-rail QK -> retrieve_mask_and_reduce_p
//              for tiles [c*tpc, (c+1)*tpc) of the row. The per-tile QK result is independent of every other tile (a
//              fresh accumulator per tile), so computing tiles in different CTAs gives the same S. It publishes, per tile,
//              (a) each softmax thread's 32 fp32 S values exactly as they sit in that thread's registers after
//              retrieve_mask_and_reduce_p (masked, rail0 + rail1), and (b) the tile's dequantized bf16 K/V tile, byte for
//              byte as it sits in the leaf's SW128 shared-memory buffer.
//   pv_kernel  (grid rows x NC, PDL-dependent on qk_kernel): output columns are independent in O = sum_t a.. P_t V_t, so
//              CTA c owns columns [c*512/NC, (c+1)*512/NC). Every CTA runs the leaf's softmax-thread program VERBATIM on
//              the published registers (same lazy-max decisions, same exp2f / bf16 roundings, same l chain -- redundantly,
//              so every CTA has bit-identical P and m, l), the same O *= a multiplies on its columns, and the PV MMA
//              (.ws, M = 64, K = the tile's 64 slots as 4 K16 steps accumulated in list order) on its columns of the same
//              bf16 tile bytes. The epilogue is the leaf's, restricted to the CTA's columns; lse comes from CTA 0.
//
// Nothing depends on the batch: the per-row program (tile = 64 list slots, list order, rails, lazy max threshold, every
// rounding point) is the recipe's; tpc / NC / the plan choice only move work between CTAs. No atomics, no reductions
// across CTAs (the published S and V are copied, never combined).
#pragma once

#include "leaf.cuh"

namespace dsa_sm100::dualrail {

struct SplitParams {
    float *p_scr;       // [rows, T, 8, 128, 4]: per tile, thread i's p[4*j4 .. 4*j4+3] at (j4*128 + i) float4
    uint8_t *v_scr;     // [rows, T, 64 KiB]: per tile, the dequantized SW128 bf16 tile bytes
    int T;              // tiles per row held by the scratch (>= every row's tile count)
    int tpc;            // tiles per qk CTA
};

static constexpr int P_TILE_FLOATS = 4096;
static constexpr int V_TILE_BYTES = 64 * 512 * 2;

CUTE_DEVICE void bulk_s2g(void *gdst, const void *ssrc, uint32_t bytes) {
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n"
                 :: "l"(gdst), "r"(cute::cast_smem_ptr_to_uint(ssrc)), "r"(bytes) : "memory");
}
CUTE_DEVICE void bulk_g2s(void *sdst, const void *gsrc, uint32_t bytes, transac_bar_t &bar) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                 :: "r"(cute::cast_smem_ptr_to_uint(sdst)), "l"(gsrc), "r"(bytes), "r"(cute::cast_smem_ptr_to_uint(&bar))
                 : "memory");
}

// ===================================================== qk_kernel =====================================================
template<bool EXTRA>
struct QK {
    using L = Leaf<false, EXTRA>;
    static constexpr int NUM_RAW = 2, NUM_INDEX_BUFS = 4;
    struct SharedMemoryPlan {
        struct { alignas(1024) bf16 data[cosize_v<typename L::SmemLayoutQ_SW128>]; } q;
        struct { alignas(1024) bf16 quant_part[64 * 512]; } dequant;
        array_aligned<uint8_t, L::RAW_BLOCK_SMEM_BYTES, 128> raw_quant[NUM_RAW];
        float p_exchange_buf[4][32 * 32];
        char is_token_valid[NUM_INDEX_BUFS][8];
        int tma_coord[NUM_INDEX_BUFS][64];
        array_aligned<uint32_t, 1> tmem_start_addr;
        transac_bar_t bar_q_tma, bar_q_utccp, bar_q_roped;
        transac_bar_t bar_raw_ready[NUM_RAW], bar_raw_free[NUM_RAW];
        transac_bar_t bar_dq_ready, bar_dq_free, bar_qk_done, bar_p_free;
        transac_bar_t bar_valid_coord_ready[NUM_INDEX_BUFS], bar_valid_coord_free[NUM_INDEX_BUFS];
    };
};

template<typename Q, typename TmaParam, bool EXTRA>
__global__ void __launch_bounds__(384, 1, 1)
qk_kernel(__grid_constant__ const Params params, __grid_constant__ const SplitParams sp, __grid_constant__ const TmaParam tma_params) {
    // PDL (launch structure only): wait for the previous grid. pv_kernel may be scheduled only once this CTA OWNS its
    // TMEM (trigger after the allocate below, ISOEXEC_DUALRAIL_TRIGGER_AFTER_ALLOC, default on): pv_kernel allocates all
    // 512 columns before its own griddepcontrol.wait, so a pv CTA scheduled on this SM between an early trigger and this
    // CTA's allocate takes the whole TMEM and waits for this grid, which can then never allocate (silent deadlock).
    asm volatile("griddepcontrol.wait;" ::: "memory");
#if !ISOEXEC_DUALRAIL_TRIGGER_AFTER_ALLOC
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
    using L = typename Q::L;
    using OrigKVFormat = typename L::OrigKVFormat;
    using ExtraKVFormat = typename L::ExtraKVFormat;
    constexpr int MMA_M = L::MMA_M, B_TOPK = L::B_TOPK, D_Q = L::D_Q;
    using tmem_cols = typename L::tmem_cols;
    const int row = blockIdx.x;
    const int warpgroup_idx = cutlass::canonical_warp_group_idx();
    const int idx_in_warpgroup = threadIdx.x % 128;
    const int warp_idx = cutlass::canonical_warp_idx_sync();
    const int lane_idx = threadIdx.x % 32;

    // the row's tiles exactly as the leaf counts them
    const int topk_length = params.topk_length ? __ldg(params.topk_length + row) : params.topk;
    const int orig_topk_padded = max(ku::ceil(topk_length, (int)B_TOPK), (int)B_TOPK);
    const int extra_topk_length = EXTRA ? (params.extra_topk_length ? __ldg(params.extra_topk_length + row) : params.extra_topk) : 0;
    const int num_orig_kv_blocks = orig_topk_padded / B_TOPK;
    const int end_block_idx = num_orig_kv_blocks + ku::ceil(extra_topk_length, (int)B_TOPK) / B_TOPK;
    const int t0 = blockIdx.y * sp.tpc, t1 = min(t0 + sp.tpc, end_block_idx);
    if (t0 >= t1) return;

    extern __shared__ char wksp_buf[];
    typename Q::SharedMemoryPlan &plan = *reinterpret_cast<typename Q::SharedMemoryPlan *>(wksp_buf);

    if (warp_idx == 0 && elect_one_sync()) {
        cute::prefetch_tma_descriptor(tma_params.tma_Q.get_tma_descriptor());
        cute::prefetch_tma_descriptor(&tma_params.kv_map);
        if constexpr (EXTRA) cute::prefetch_tma_descriptor(&tma_params.extra_kv_map);
    }
    if (warp_idx == 0) {
        if (elect_one_sync()) {
            plan.bar_q_tma.init(1);
            plan.bar_q_utccp.init(1);
            plan.bar_q_roped.init(32);
            for (int i = 0; i < Q::NUM_RAW; ++i) {
                plan.bar_raw_ready[i].init(1);
                plan.bar_raw_free[i].init(128);
            }
            plan.bar_dq_ready.init(128);
            plan.bar_dq_free.init(2);           // the QK MMA's commit + the tile store's read completion
            plan.bar_qk_done.init(1);
            plan.bar_p_free.init(128);
            for (int i = 0; i < Q::NUM_INDEX_BUFS; ++i) {
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
#if ISOEXEC_DUALRAIL_TRIGGER_AFTER_ALLOC
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
    const int n_tiles = t1 - t0;

    if (warpgroup_idx == 0) {
        // ============== the leaf's retrieve_mask_and_reduce_p, then the registers go to the scratch ==============
        cutlass::arch::warpgroup_reg_alloc<224>();
        constexpr int NUM_ELEMS_PER_THREAD = B_TOPK / (128 / MMA_M);
        CUTE_NO_UNROLL
        for (int n = 0; n < n_tiles; ++n) {
            NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
            plan.bar_valid_coord_ready[n % Q::NUM_INDEX_BUFS].wait((n / Q::NUM_INDEX_BUFS) & 1);
            plan.bar_qk_done.wait(n & 1);
            ku::tcgen05_after_thread_sync();
            float p[NUM_ELEMS_PER_THREAD];
            retrieve_mask_and_reduce_p<MMA_M, NUM_ELEMS_PER_THREAD, NamedBarriers::wg0_warp02_sync, NamedBarriers::wg0_warp13_sync>(
                tmem_cols::P, plan.is_token_valid[n % Q::NUM_INDEX_BUFS], warp_idx, lane_idx,
                [&]() { plan.bar_p_free.arrive(); }, plan.p_exchange_buf, p);
            plan.bar_valid_coord_free[n % Q::NUM_INDEX_BUFS].arrive();
            float4 *dst = reinterpret_cast<float4 *>(sp.p_scr + ((int64_t)row * sp.T + t0 + n) * P_TILE_FLOATS);
            CUTE_UNROLL
            for (int j4 = 0; j4 < NUM_ELEMS_PER_THREAD / 4; ++j4)
                dst[j4 * 128 + idx_in_warpgroup] = *reinterpret_cast<float4 *>(p + 4 * j4);
        }
        if (warp_idx == 0) cute::TMEM::Allocator1Sm().free(0, 512);
    } else if (warpgroup_idx == 1) {
        cutlass::arch::warpgroup_reg_dealloc<72>();
        const int warp_idx = cutlass::canonical_warp_idx_sync();
        if (warp_idx == 4 && elect_one_sync()) {
            // ================================ UMMA warp (QK only) ================================
            {
                Tensor gQ = tma_params.tma_Q.get_tma_tensor(tma_params.shape_Q)(_, _, row);
                Tensor sQ = make_tensor(make_smem_ptr(plan.q.data), typename L::SmemLayoutQ_SW128{});
                ku::launch_tma_copy(tma_params.tma_Q, gQ, sQ, plan.bar_q_tma, TMA::CacheHintSm90::EVICT_FIRST);
            }
            plan.bar_q_tma.arrive_and_expect_tx(MMA_M * D_Q * sizeof(bf16));
            plan.bar_q_tma.wait(0);
            if (params.rope.q) plan.bar_q_roped.wait(0);       // warp 6 rotated the tile's tail (the leaf's step)
            ku::tcgen05_after_thread_sync();
            {
                UMMA::SmemDescriptor sQ_desc = UMMA::make_umma_desc<UMMA::Major::K>(make_tensor(
                    make_smem_ptr(plan.q.data), tile_to_shape(UMMA::Layout_K_SW128_Atom<bf16>{}, Shape<Int<MMA_M*2>, Int<64>>{})));
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
            Tensor tP = partition_fragment_C(tiled_mma_P, Shape<Int<MMA_M>, _128>{});
            tP.data().get() = tmem_cols::P;
            plan.bar_q_utccp.wait(0);
            ku::tcgen05_after_thread_sync();
            CUTE_NO_UNROLL
            for (int n = 0; n < n_tiles; ++n) {
                plan.bar_dq_ready.wait(n & 1);
                if (n > 0) plan.bar_p_free.wait((n - 1) & 1);     // the previous tile's S left TMEM
                ku::tcgen05_after_thread_sync();
                Tensor tQ = tiled_mma_P.get_slice(_0{}).make_fragment_A(partition_shape_A(tiled_mma_P, Shape<Int<MMA_M>, Int<D_Q/2>>{}));
                tQ.data().get() = tmem_cols::Q;
                Tensor sK = make_tensor(make_smem_ptr(plan.dequant.quant_part), typename L::template SmemLayoutKTiles_DualGemm_SW128<D_Q/64/2>{});
                ku::utcmma_ts(tiled_mma_P, tQ, sK, tP, true);
                ku::umma_arrive_noelect(plan.bar_qk_done);
                ku::umma_arrive_noelect(plan.bar_dq_free);
            }
        } else if (warp_idx == 5) {
            // ================================ KV producer (gathers) ================================
            if (elect_one_sync()) {
                for_each_kv_block<OrigKVFormat, ExtraKVFormat>(t0, t1, num_orig_kv_blocks, [&]<typename F>(int block_idx, bool is_extra_block) {
                    const int n = block_idx - t0;
                    const int ib = n % Q::NUM_INDEX_BUFS, rb = n % Q::NUM_RAW;
                    plan.bar_valid_coord_ready[ib].wait((n / Q::NUM_INDEX_BUFS) & 1);
                    plan.bar_raw_free[rb].wait(((n / Q::NUM_RAW) & 1) ^ 1);
                    const CUtensorMap *tensor_map = is_extra_block ? &tma_params.extra_kv_map : &tma_params.kv_map;
                    int4 cur_indices = *(int4 *)(plan.tma_coord[ib] + 0);
                    int4 nxt_cur_indices;
                    CUTE_UNROLL
                    for (int r = 0; r < B_TOPK; r += 4) {
                        if (r + 4 < B_TOPK) nxt_cur_indices = *(int4 *)(plan.tma_coord[ib] + r + 4);
                        ku::tma_gather4(tensor_map, plan.bar_raw_ready[rb],
                                        plan.raw_quant[rb].data() + L::template RAW_TOKEN_SMEM_STRIDE<F> * r, 0, cur_indices,
                                        (int64_t)TMA::CacheHintSm90::EVICT_FIRST);
                        cur_indices = nxt_cur_indices;
                    }
                    plan.bar_raw_ready[rb].arrive_and_expect_tx(B_TOPK * L::template RAW_TOKEN_SMEM_STRIDE<F>);
                    plan.bar_valid_coord_free[ib].arrive();
                });
            }
        } else if (warp_idx == 6) {
            // the fused Q RoPE first (leaf.cuh warp 6), then the tile copies
            if (params.rope.q) {
                plan.bar_q_tma.wait(0);
                Tensor sQ = make_tensor(make_smem_ptr(plan.q.data), typename L::SmemLayoutQ_SW128{});
                rope_q_tile(sQ, params.h_q, params.rope, row, lane_idx);
                fence_view_async_shared();
                plan.bar_q_roped.arrive();
            }
            // ======================== the dequantized tile -> v_scr (bytes, swizzle included) ========================
            if (elect_one_sync()) {
                for (int n = 0; n < n_tiles; ++n) {
                    plan.bar_dq_ready.wait(n & 1);
                    uint8_t *dst = sp.v_scr + ((int64_t)row * sp.T + t0 + n) * V_TILE_BYTES;
                    CUTE_UNROLL
                    for (int k = 0; k < 4; ++k)
                        bulk_s2g(dst + k * (V_TILE_BYTES / 4), (const uint8_t *)plan.dequant.quant_part + k * (V_TILE_BYTES / 4), V_TILE_BYTES / 4);
                    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
                    asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory");
                    plan.bar_dq_free.arrive();
                }
                asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory");
            }
        } else if (warp_idx == 7) {
            // ======================= indices -> TMA coordinates + validity (the leaf's code) =======================
            const int *indices = params.idx + params.idx_stride * row;
            const int *extra_indices = EXTRA ? params.extra_idx + params.extra_idx_stride * row : nullptr;
            auto load_block_indices = [&](int block_idx) -> int2 {
                if (block_idx < num_orig_kv_blocks) return __ldg((const int2 *)(indices + block_idx*B_TOPK + lane_idx*2));
                return __ldg((const int2 *)(extra_indices + (block_idx - num_orig_kv_blocks)*B_TOPK + lane_idx*2));
            };
            int2 my_indices = load_block_indices(t0);
            CUTE_NO_UNROLL
            for (int block_idx = t0; block_idx < t1; ++block_idx) {
                const int n = block_idx - t0, ib = n % Q::NUM_INDEX_BUFS;
                const bool has_next = block_idx + 1 < t1;
                const int2 next_indices = has_next ? load_block_indices(block_idx + 1) : int2{};
                const bool is_extra = block_idx >= num_orig_kv_blocks;
                const int abs_pos = (is_extra ? block_idx - num_orig_kv_blocks : block_idx) * B_TOPK + lane_idx*2;
                const int lim = is_extra ? extra_topk_length : topk_length;
                plan.bar_valid_coord_free[ib].wait(((n / Q::NUM_INDEX_BUFS) & 1) ^ 1);
                int tma_coords[2];
                char valid_mask = 0;
                CUTE_UNROLL
                for (int i = 0; i < 2; ++i) {
                    const int cur_idx = i == 0 ? my_indices.x : my_indices.y;
                    const bool ok = cur_idx >= 0 && abs_pos + i < lim;
                    valid_mask |= ok << i;
                    tma_coords[i] = ok ? cur_idx : -1;
                }
                valid_mask <<= lane_idx % 4 * 2;
                valid_mask |= __shfl_xor_sync(0xFFFFFFFF, valid_mask, 0x1);
                valid_mask |= __shfl_xor_sync(0xFFFFFFFF, valid_mask, 0x2);
                *(int2 *)(plan.tma_coord[ib] + lane_idx*2) = *(int2 *)tma_coords;
                if (lane_idx % 4 == 0) plan.is_token_valid[ib][lane_idx/4] = valid_mask;
                plan.bar_valid_coord_ready[ib].arrive();
                if (has_next) my_indices = next_indices;
            }
        }
    } else {
        // ================================ dequant warpgroup (the leaf's) ================================
        cutlass::arch::warpgroup_reg_alloc<208>();
        const typename L::template Dequantizer<OrigKVFormat> dequant_orig(idx_in_warpgroup);
        const typename L::template Dequantizer<ExtraKVFormat> dequant_extra(idx_in_warpgroup);
        const uint32_t dst = cute::cast_smem_ptr_to_uint(plan.dequant.quant_part);
        for_each_kv_block<OrigKVFormat, ExtraKVFormat>(t0, t1, num_orig_kv_blocks, [&]<typename F>(int block_idx, bool) {
            const int n = block_idx - t0, rb = n % Q::NUM_RAW;
            plan.bar_raw_ready[rb].wait((n / Q::NUM_RAW) & 1);
            plan.bar_dq_free.wait((n & 1) ^ 1);
            const auto *dequant = [&] {
                if constexpr (std::is_same_v<F, OrigKVFormat>) return &dequant_orig; else return &dequant_extra;
            }();
            const uint8_t *raw = plan.raw_quant[rb].data();
            dequant->run(raw, raw + F::QUANT_BYTES, dst, [] {});
            cutlass::arch::fence_view_async_shared();
            plan.bar_dq_ready.arrive();
            plan.bar_raw_free[rb].arrive();
        });
    }
}

// ===================================================== pv_kernel =====================================================
template<int NC>
struct PV {
    using L = Leaf<false, false>;
    static constexpr int MMA_M = 64, B_TOPK = 64, D_V = 512;
    static constexpr int PVN = D_V / NC, HALF = PVN / 2, B_EPI = HALF < 64 ? HALF : 64, STAGES = 3;
    static constexpr int V_BYTES = V_TILE_BYTES / NC;
    static_assert(NC == 2 || NC == 4 || NC == 8);
    using SmemLayoutV = typename L::template SmemLayoutKTilesTransposed_SW128<PVN / 64>;
    using SmemLayoutOBuf = decltype(tile_to_shape(UMMA::Layout_K_SW128_Atom<bf16>{}, Shape<Int<MMA_M>, Int<PVN>>{}));
    using TiledMMA_O = decltype(make_tiled_mma(SM100_MMA_F16BF16_WS_SS_NOELECT<bf16, bf16, float, MMA_M, PVN, UMMA::Major::K, UMMA::Major::MN>{}));
    struct Stage {
        alignas(1024) bf16 v[64 * PVN];
        alignas(128) float4 p[P_TILE_FLOATS / 4];
    };
    struct SharedMemoryPlan {
        Stage stage[STAGES];
        struct { alignas(1024) bf16 data[cosize_v<typename L::SmemLayoutS>]; } s[2];
        struct { alignas(1024) bf16 data[cosize_v<SmemLayoutOBuf>]; } o;
        CUTE_ALIGNAS(16) float rowwise_max_buf[128];
        array_aligned<uint32_t, 1> tmem_start_addr;
        transac_bar_t bar_stage_full[STAGES], bar_stage_free[STAGES];
        transac_bar_t bar_so_ready[2], bar_sv_done[2];   // indexed by the S buffer: no phase aliasing (see pv_kernel)
    };
};

template<typename P, typename TmaParam>
__global__ void __launch_bounds__(256, 1, 1)
pv_kernel(__grid_constant__ const Params params, __grid_constant__ const SplitParams sp, __grid_constant__ const TmaParam tma_params) {
    using L = typename P::L;
    constexpr int MMA_M = P::MMA_M, B_TOPK = P::B_TOPK, PVN = P::PVN, HALF = P::HALF, B_EPI = P::B_EPI, STAGES = P::STAGES;
    const int row = blockIdx.x, col_cta = blockIdx.y;
    const int warpgroup_idx = cutlass::canonical_warp_group_idx();
    const int idx_in_warpgroup = threadIdx.x % 128;
    const int warp_idx = cutlass::canonical_warp_idx_sync();
    const int lane_idx = threadIdx.x % 32;

    extern __shared__ char wksp_buf[];
    typename P::SharedMemoryPlan &plan = *reinterpret_cast<typename P::SharedMemoryPlan *>(wksp_buf);

    if (warp_idx == 0 && elect_one_sync()) cute::prefetch_tma_descriptor(tma_params.tma_O.get_tma_descriptor());
    if (warp_idx == 0) {
        if (elect_one_sync()) {
            for (int i = 0; i < STAGES; ++i) {
                plan.bar_stage_full[i].init(1);
                plan.bar_stage_free[i].init(128 + 1);   // the softmax threads read p + the PV MMA read v
            }
            plan.bar_so_ready[0].init(128);
            plan.bar_so_ready[1].init(128);
            plan.bar_sv_done[0].init(1);
            plan.bar_sv_done[1].init(1);
            cutlass::arch::fence_barrier_init();
        }
        cute::TMEM::Allocator1Sm().allocate(512, plan.tmem_start_addr.data());
        KU_TRAP_ONLY_DEVICE_ASSERT(plan.tmem_start_addr.data()[0] == 0);
        cute::TMEM::Allocator1Sm().release_allocation_lock();
    }
    __syncthreads();

    // the row's tile count exactly as the leaf counts it (fp8 extra cache iff extra_idx is set)
    const bool has_extra = params.extra_idx != nullptr;
    const int topk_length = params.topk_length ? __ldg(params.topk_length + row) : params.topk;
    const int orig_topk_padded = max(ku::ceil(topk_length, (int)B_TOPK), (int)B_TOPK);
    const int extra_topk_length = has_extra ? (params.extra_topk_length ? __ldg(params.extra_topk_length + row) : params.extra_topk) : 0;
    const int num_orig_kv_blocks = orig_topk_padded / B_TOPK;
    const int end_block_idx = num_orig_kv_blocks + ku::ceil(extra_topk_length, (int)B_TOPK) / B_TOPK;

    if (warpgroup_idx == 0) {
        // ======================= the leaf's scale & exp warpgroup, on the published registers =======================
        Tensor sO = make_tensor(make_smem_ptr(plan.o.data), typename P::SmemLayoutOBuf{});
        const float2 scale = float2{params.scale_log2, params.scale_log2};
        const float attn_sink = (params.sink == nullptr || idx_in_warpgroup % MMA_M >= params.h_q)
            ? -CUDART_INF_F : __ldg(params.sink + (idx_in_warpgroup % MMA_M)) * CUDART_L2E_F;
        constexpr int NUM_ELEMS_PER_THREAD = B_TOPK / (128 / MMA_M);
        const int s_off = lane_idx*8 + (warp_idx&1)*(MMA_M/2)*8 + (warp_idx/2)*MMA_M*(B_TOPK/2);

        float mi = L::MAX_INIT_VAL;
        float li = 0.0f;
        float real_mi = -CUDART_INF_F;

        CUTE_NO_UNROLL
        for (int block_idx = 0; block_idx < end_block_idx; ++block_idx) {
            const int st = block_idx % STAGES, sb = block_idx & 1;
            NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
            plan.bar_stage_full[st].wait((block_idx / STAGES) & 1);
            float p[NUM_ELEMS_PER_THREAD];
            CUTE_UNROLL
            for (int j4 = 0; j4 < NUM_ELEMS_PER_THREAD / 4; ++j4)
                *reinterpret_cast<float4 *>(p + 4 * j4) = plan.stage[st].p[j4 * 128 + idx_in_warpgroup];
            plan.bar_stage_free[st].arrive();

            // ---- from here to the S store: leaf.cuh verbatim ----
            float cur_pi_max = get_max<NUM_ELEMS_PER_THREAD>(p);
            cur_pi_max *= params.scale_log2;
            plan.rowwise_max_buf[idx_in_warpgroup] = cur_pi_max;
            NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
            cur_pi_max = max(cur_pi_max, plan.rowwise_max_buf[idx_in_warpgroup ^ 64]);
            real_mi = max(real_mi, cur_pi_max);
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
            // ---- the S buffer sb was last read by PV(block_idx - 2) ----
            if (block_idx >= 2) plan.bar_sv_done[sb].wait(((block_idx - 2) / 2) & 1);
            bf16 *sS_base = plan.s[sb].data + s_off;
            CUTE_UNROLL
            for (int i = 0; i < NUM_ELEMS_PER_THREAD/8; i += 1)
                *(uint128_t *)(sS_base + MMA_M*8*i) = *(uint128_t *)(s + i*4);

            // O *= a on this CTA's columns, after PV(block_idx - 1) (the leaf's condition, warp-collective)
            if (block_idx != 0 && __any_sync(0xffffffff, row_scale)) {
                plan.bar_sv_done[(block_idx - 1) & 1].wait(((block_idx - 1) / 2) & 1);
                ku::tcgen05_after_thread_sync();
                rescale_O<HALF, B_EPI, 0>(scale_for_old);
                ku::tcgen05_before_thread_sync();
            }
            fence_view_async_shared();
            plan.bar_so_ready[sb].arrive();
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

        const float denom = li + exp2f(attn_sink - mi);
        if (col_cta == 0 && idx_in_warpgroup < params.h_q) {
            float cur_lse;
            if (li == 0.0f) cur_lse = params.sink ? __ldg(params.sink + idx_in_warpgroup) : +CUDART_INF_F;
            else cur_lse = fma(mi, CUDART_LN2_F, logf(denom));
            params.lse[(int64_t)row * params.lse_row_stride + idx_in_warpgroup] = cur_lse;
        }

        const int last = end_block_idx - 1;
        plan.bar_sv_done[last & 1].wait((last / 2) & 1);
        ku::tcgen05_after_thread_sync();

        const float o_scale = li == 0.0f ? 0.0f : __fdividef(1.0f, denom);
        const float2 o_scale_float2 = {o_scale, o_scale};
        const int o_row = idx_in_warpgroup % MMA_M, o_col0 = (idx_in_warpgroup / MMA_M) * HALF;
        float2 o[B_EPI/2];
        __nv_bfloat162 o_bf16[B_EPI/2];
        CUTE_UNROLL
        for (int i = 0; i < HALF / B_EPI; ++i) {
            ku::tmem_ld_32dp32bNx<B_EPI>(i*B_EPI, o);
            cutlass::arch::fence_view_async_tmem_load();
            CUTE_UNROLL
            for (int j = 0; j < B_EPI/2; ++j) {
                o[j] = ku::float2_mul(o[j], o_scale_float2);
                o_bf16[j] = __float22bfloat162_rn(o[j]);
            }
            if (params.rope.o)       // the leaf's fused inverse RoPE, on this CTA's columns
                rope_o_chunk(o_bf16, col_cta * PVN + o_col0 + i*B_EPI, params.rope, row);
            CUTE_UNROLL
            for (int j = 0; j < B_EPI / 8; ++j)
                *(__int128_t *)(&sO(o_row, o_col0 + i*B_EPI + j*8)) = *(__int128_t *)(&o_bf16[j*4]);
        }
        fence_view_async_shared();
        NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
        if (warp_idx == 0 && elect_one_sync()) {
            Tensor tma_gO = flat_divide(tma_params.tma_O.get_tma_tensor(tma_params.shape_O)(_, _, row), Shape<Int<MMA_M>, Int<64>>{})(_, _, _0{}, _);
            auto thr_tma = tma_params.tma_O.get_slice(_0{});
            Tensor tma_sO = flat_divide(sO, Shape<Int<MMA_M>, Int<64>>{})(_, _, _0{}, _);
            CUTE_UNROLL
            for (int b = 0; b < PVN / 64; ++b)
                cute::copy(tma_params.tma_O, thr_tma.partition_S(tma_sO(_, _, b)), thr_tma.partition_D(tma_gO(_, _, col_cta * (PVN / 64) + b)));
            cute::tma_store_arrive();
            cute::tma_store_wait<0>();
        }
        __syncwarp();
        if (warp_idx == 0) cute::TMEM::Allocator1Sm().free(0, 512);
    } else {
        if (warp_idx == 4 && elect_one_sync()) {
            // ============================ UMMA warp: PV on this CTA's columns ============================
            typename P::TiledMMA_O tiled_mma_O{};
            Tensor tO = partition_fragment_C(tiled_mma_O, Shape<Int<MMA_M>, Int<PVN>>{});
            tO.data().get() = 0;
            CUTE_NO_UNROLL
            for (int block_idx = 0; block_idx < end_block_idx; ++block_idx) {
                const int st = block_idx % STAGES, sb = block_idx & 1;
                plan.bar_so_ready[sb].wait((block_idx / 2) & 1);
                plan.bar_stage_full[st].wait((block_idx / STAGES) & 1);
                ku::tcgen05_after_thread_sync();
                Tensor sS = make_tensor(make_smem_ptr(plan.s[sb].data), typename L::SmemLayoutS{});
                Tensor sV = make_tensor(make_smem_ptr(plan.stage[st].v), typename P::SmemLayoutV{});
                ku::utcmma_ss(tiled_mma_O, sS, sV, tO, block_idx == 0);
                ku::umma_arrive_noelect(plan.bar_sv_done[sb]);
                ku::umma_arrive_noelect(plan.bar_stage_free[st]);
            }
        } else if (warp_idx == 5 && elect_one_sync()) {
            // ============================ loader: the published S registers + V columns ============================
            asm volatile("griddepcontrol.wait;" ::: "memory");
            CUTE_NO_UNROLL
            for (int block_idx = 0; block_idx < end_block_idx; ++block_idx) {
                const int st = block_idx % STAGES;
                plan.bar_stage_free[st].wait(((block_idx / STAGES) & 1) ^ 1);
                plan.bar_stage_full[st].arrive_and_expect_tx(P_TILE_FLOATS * 4 + P::V_BYTES);
                const int64_t tile = (int64_t)row * sp.T + block_idx;
                bulk_g2s(plan.stage[st].p, sp.p_scr + tile * P_TILE_FLOATS, P_TILE_FLOATS * 4, plan.bar_stage_full[st]);
                bulk_g2s(plan.stage[st].v, sp.v_scr + tile * V_TILE_BYTES + col_cta * P::V_BYTES, P::V_BYTES, plan.bar_stage_full[st]);
            }
        }
    }
}

}  // namespace dsa_sm100::dualrail
