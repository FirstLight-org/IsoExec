// SPDX-License-Identifier: MIT
// Derived from DeepSeek FlashMLA (https://github.com/deepseek-ai/FlashMLA; csrc/kernels/sm90/decode/dense/splitkv_mla.cuh, HEAD ba89a34),
// Copyright (c) 2025 DeepSeek, MIT License.
//
// FlashMLA tile arithmetic over position-anchored segment runs, folded with canonical tree merges.
// Packed query offsets and logical causal borders determine published rows.
// Partial rows publish fp32 (O, L, M); roots round once with L'=1 for zero/NaN L.
#include <cutlass/cutlass.h>

#include "utils.h"

#include "params.h"
#include "leaf.h"
#include "config.h"
#include "traits.h"

using namespace cute;
using cutlass::arch::NamedBarrier;

namespace mla_sm90::fwd {

// Here we use MAX_INIT_VAL_SM to initialize sM, and MAX_INIT_VAL for masking
// The reason is that, we need to calculate new_max = max(sM(row_idx), cur_max*scale_softmax_log2)
// so we must guarantee that MAX_INIT_VAL*scale_softmax_log2 < MAX_INIT_VAL_SM
static constexpr float MAX_INIT_VAL_SM = -1e30f;
static constexpr float MAX_INIT_VAL = -1e33f;

__forceinline__ __device__ int get_AorC_row_idx(int local_row_idx, int idx_in_warpgroup) {
    // In the layout of fragment A and fragment C during WGMMA, data each thread holds resides in two particular rows. This function converts the local_row_idx (0~2) to the actual row_idx
    // You may refer to this link for the detailed layout: https://docs.nvidia.com/cuda/parallel-thread-execution/#wgmma-64n16-a
    int row_idx = (idx_in_warpgroup/32)*16 + local_row_idx*8 + (idx_in_warpgroup%32/4);
    return row_idx;
}

// Launch TMA copy for a range of KV tile
// A tile has a shape of PAGE_BLOCK_SIZE (64) x 64
template<
    int START_HEAD_DIM_TILE_IDX,
    int END_HEAD_DIM_TILE_IDX,
    typename TMA_K_OneTile,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1
>
__forceinline__ __device__ void launch_kv_tiles_copy_tma(
    Tensor<Engine0, Layout0> const &gKV,	// (PAGE_BLOCK_SIZE, HEAD_DIM_K)
    Tensor<Engine1, Layout1> &sKV,	// (PAGE_BLOCK_SIZE, HEAD_DIM_K), swizzled
    TMA_K_OneTile &tma_K,
    TMABarrier* barriers_K,
    int idx_in_warpgroup
) {
    if (idx_in_warpgroup == 0) {
        auto thr_tma = tma_K.get_slice(_0{});
        Tensor cur_gKV = thr_tma.partition_S(gKV)(_, _0{}, Int<START_HEAD_DIM_TILE_IDX>{});
        Tensor cur_sKV = thr_tma.partition_D(sKV)(_, _0{}, Int<START_HEAD_DIM_TILE_IDX>{});
        cute::copy(tma_K.with(reinterpret_cast<typename TMABarrier::ValueType &>(barriers_K[START_HEAD_DIM_TILE_IDX]), 0, cute::TMA::CacheHintSm90::EVICT_FIRST), cur_gKV, cur_sKV);
        if constexpr (START_HEAD_DIM_TILE_IDX+1 < END_HEAD_DIM_TILE_IDX) {
            launch_kv_tiles_copy_tma<START_HEAD_DIM_TILE_IDX+1, END_HEAD_DIM_TILE_IDX>(gKV, sKV, tma_K, barriers_K, idx_in_warpgroup);
        }
    }
}

// Prefetch some KV tiles
// Currently this is not used because it leads to performance degradation
template<
    int START_HEAD_DIM_TILE_IDX,
    int END_HEAD_DIM_TILE_IDX,
    typename TMA_K_OneTile,
    typename Engine0, typename Layout0
>
__forceinline__ __device__ void prefetch_kv_tiles(
    Tensor<Engine0, Layout0> const &gKV,	// (PAGE_BLOCK_SIZE, HEAD_DIM_K)
    TMA_K_OneTile &tma_K,
    int idx_in_warpgroup
) {
    if (idx_in_warpgroup == 0) {
        auto thr_tma = tma_K.get_slice(_0{});
        Tensor cur_gKV = thr_tma.partition_S(gKV)(_, _0{}, Int<START_HEAD_DIM_TILE_IDX>{});
        cute::prefetch(tma_K, cur_gKV);
        if constexpr (START_HEAD_DIM_TILE_IDX+1 < END_HEAD_DIM_TILE_IDX) {
            prefetch_kv_tiles<START_HEAD_DIM_TILE_IDX+1, END_HEAD_DIM_TILE_IDX>(gKV, tma_K, idx_in_warpgroup);
        }
    }
}

// Adapted from https://github.com/Dao-AILab/flash-attention/blob/cdaf2de6e95cb05400959b5ab984f66e4c7df317/hopper/utils.h
// * Copyright (c) 2024, Tri Dao.
template <bool zero_init=false, int wg_wait=0, bool arrive=true, bool commit=true, typename Tensor0, typename Tensor1, typename Tensor2, typename TiledMma>
__forceinline__ __device__ void gemm(TiledMma &tiled_mma, Tensor0 const &tCrA, Tensor1 const &tCrB, Tensor2 &tCrC) {
    constexpr bool Is_RS = !cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value;
    // Need to cast away const on tCrA since warpgroup_fence_operand doesn't take const
    if constexpr (Is_RS) { cute::warpgroup_fence_operand(const_cast<Tensor0 &>(tCrA)); }
    warpgroup_fence_operand(tCrC);
    if constexpr (arrive) {
        warpgroup_arrive();
    }
    if constexpr (zero_init) {
        tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
        // Unroll the K mode manually to set scale D to 1
        CUTLASS_PRAGMA_UNROLL
        for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
            cute::gemm(tiled_mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
            tiled_mma.accumulate_ = GMMA::ScaleOut::One;
        }
    } else {
        // Unroll the K mode manually to set scale D to 1
        CUTLASS_PRAGMA_UNROLL
        for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
            cute::gemm(tiled_mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
            tiled_mma.accumulate_ = GMMA::ScaleOut::One;
        }
    }
    if constexpr (commit) {
        warpgroup_commit_batch();
    }
    if constexpr (wg_wait >= 0) { warpgroup_wait<wg_wait>(); }
    warpgroup_fence_operand(tCrC);
    if constexpr (Is_RS) { warpgroup_fence_operand(const_cast<Tensor0 &>(tCrA)); }
}


// Wait for one KV-tile to be ready, and then calculate P += Q K^T for one Q-tile (BLOCK_SIZE_Mx64) and one KV-tile (PAGE_BLOCK_SIZEx64)
// The Q-tile should be in shared memory
template<
    typename TiledMMA,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1,
    typename Engine2, typename Layout2
> 
__forceinline__ __device__ void qkt_gemm_one_tile_sQ(
    TiledMMA &tiled_mma,
    Tensor<Engine0, Layout0> const &thr_mma_sQ_tile,	// (MMA, 1, 4)
    Tensor<Engine1, Layout1> const &thr_mma_sKV_tile,	// (MMA, 1, 4)
    Tensor<Engine2, Layout2> &rP,	// ((2, 2, 8), 1, 1)
    TMABarrier* barrier,
    bool &cur_phase,
    int idx_in_warpgroup
) {
    if (idx_in_warpgroup == 0) {
        barrier->arrive_and_expect_tx(64*64*2);
    }
    barrier->wait(cur_phase ? 1 : 0);

    warpgroup_fence_operand(rP);
    warpgroup_arrive();
    cute::gemm(tiled_mma, thr_mma_sQ_tile(_, _, _0{}), thr_mma_sKV_tile(_, _, _0{}), rP);
    tiled_mma.accumulate_ = GMMA::ScaleOut::One;
    cute::gemm(tiled_mma, thr_mma_sQ_tile(_, _, _1{}), thr_mma_sKV_tile(_, _, _1{}), rP);
    cute::gemm(tiled_mma, thr_mma_sQ_tile(_, _, _2{}), thr_mma_sKV_tile(_, _, _2{}), rP);
    cute::gemm(tiled_mma, thr_mma_sQ_tile(_, _, _3{}), thr_mma_sKV_tile(_, _, _3{}), rP);
    warpgroup_commit_batch();
    warpgroup_fence_operand(rP);
}

template<
    typename TiledMMA,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1,
    typename Engine2, typename Layout2
> 
__forceinline__ __device__ void  qkt_gemm_one_tile_rQ(
    TiledMMA &tiled_mma,
    Tensor<Engine0, Layout0> const &thr_mma_rQ_tile,	// (MMA, 1, 4)
    Tensor<Engine1, Layout1> const &thr_mma_sKV_tile,	// (MMA, 1, 4)
    Tensor<Engine2, Layout2> &rP,	// ((2, 2, 8), 1, 1)
    TMABarrier* barrier,
    bool &cur_phase,
    int idx_in_warpgroup
) {
    if (idx_in_warpgroup == 0) {
        barrier->arrive_and_expect_tx(64*64*2);
    }
    barrier->wait(cur_phase ? 1 : 0);

    warpgroup_fence_operand(const_cast<Tensor<Engine0, Layout0> &>(thr_mma_rQ_tile));
    warpgroup_fence_operand(rP);
    warpgroup_arrive();
    cute::gemm(tiled_mma, thr_mma_rQ_tile(_, _, _0{}), thr_mma_sKV_tile(_, _, _0{}), rP);
    tiled_mma.accumulate_ = GMMA::ScaleOut::One;
    cute::gemm(tiled_mma, thr_mma_rQ_tile(_, _, _1{}), thr_mma_sKV_tile(_, _, _1{}), rP);
    cute::gemm(tiled_mma, thr_mma_rQ_tile(_, _, _2{}), thr_mma_sKV_tile(_, _, _2{}), rP);
    cute::gemm(tiled_mma, thr_mma_rQ_tile(_, _, _3{}), thr_mma_sKV_tile(_, _, _3{}), rP);
    warpgroup_commit_batch();
    warpgroup_fence_operand(rP);
    warpgroup_fence_operand(const_cast<Tensor<Engine0, Layout0> &>(thr_mma_rQ_tile));
}

// Pipeline 64-column Q/K tiles: wait for each TMA tile immediately before its Q K^T GEMM.
template<
    typename T, // Traits
    int PHASE_IDX,	// See comments in the code
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1,
    typename Engine2, typename Layout2,
    typename Engine3, typename Layout3
> 
__forceinline__ __device__ void warpgroup_cooperative_qkt_gemm(
    Tensor<Engine0, Layout0> &sQ,	// (BLOCK_SIZE_M, HEAD_DIM_K)
    Tensor<Engine1, Layout1> &sKV,	// (PAGE_BLOCK_SIZE, HEAD_DIM_K)
    Tensor<Engine2, Layout2> &rP,	// ((2, 2, 8), 1, 1)
    Tensor<Engine3, Layout3> &rQ8,	// The 8-th tile of Q. We store it separately to leave some room for storing sP1
    TMABarrier* barriers,
    bool &cur_phase,
    int idx_in_warpgroup
) {
    Tensor sQ_tiled = flat_divide(sQ, Shape<Int<T::BLOCK_SIZE_M>, _64>{})(_, _, _0{}, _);	// (BLOCK_SIZE_M, 64, 9)
    Tensor sKV_tiled = flat_divide(sKV, Shape<Int<T::PAGE_BLOCK_SIZE>, _64>{})(_, _, _0{}, _);	// (PAGE_BLOCK_SIZE, 64, 9)
    TiledMMA tiled_mma_sQ = (typename T::TiledMMA_QK_sQ){};
    ThrMMA thr_mma_sQ = tiled_mma_sQ.get_slice(idx_in_warpgroup);
    Tensor thr_mma_sQ_tiled = thr_mma_sQ.partition_fragment_A(sQ_tiled);	// (MMA, 1, 4, 9)
    Tensor thr_mma_sKV_tiled = thr_mma_sQ.partition_fragment_B(sKV_tiled);	// (MMA, 1, 4, 9)
    TiledMMA tiled_mma_rQ = (typename T::TiledMMA_QK_rQ){};

    #define QKT_GEMM_ONE_TILE(TILE_IDX) \
        if constexpr(TILE_IDX != 8) { \
            qkt_gemm_one_tile_sQ(tiled_mma_sQ, thr_mma_sQ_tiled(_, _, _, Int<TILE_IDX>{}), thr_mma_sKV_tiled(_, _, _, Int<TILE_IDX>{}), rP, barriers + TILE_IDX, cur_phase, idx_in_warpgroup); \
        } else { \
            qkt_gemm_one_tile_rQ(tiled_mma_rQ, rQ8, thr_mma_sKV_tiled(_, _, _, Int<TILE_IDX>{}), rP, barriers + TILE_IDX, cur_phase, idx_in_warpgroup); \
        }

    if constexpr (PHASE_IDX == 0) {
        // In PHASE-0, warpgroup 0 calculates Q K^T for the first 4 tiles
        tiled_mma_sQ.accumulate_ = GMMA::ScaleOut::Zero;
        tiled_mma_rQ.accumulate_ = GMMA::ScaleOut::One;
        QKT_GEMM_ONE_TILE(0);
        QKT_GEMM_ONE_TILE(1);
        QKT_GEMM_ONE_TILE(2);
        QKT_GEMM_ONE_TILE(3);
    } else if constexpr (PHASE_IDX == 1) {
        // In PHASE-1, warpgroup 1 calculates Q K^T for all the 9 tiles
        tiled_mma_sQ.accumulate_ = GMMA::ScaleOut::Zero;
        tiled_mma_rQ.accumulate_ = GMMA::ScaleOut::One;
        QKT_GEMM_ONE_TILE(4);
        QKT_GEMM_ONE_TILE(5);
        QKT_GEMM_ONE_TILE(6);
        QKT_GEMM_ONE_TILE(7);
        QKT_GEMM_ONE_TILE(8);
        QKT_GEMM_ONE_TILE(0);
        QKT_GEMM_ONE_TILE(1);
        QKT_GEMM_ONE_TILE(2);
        QKT_GEMM_ONE_TILE(3);
        cur_phase ^= 1;
    } else {
        // In PHASE-2, warpgroup 0 calculates Q K^T for the last 5 tiles
        static_assert(PHASE_IDX == 2);
        tiled_mma_sQ.accumulate_ = GMMA::ScaleOut::One;
        tiled_mma_rQ.accumulate_ = GMMA::ScaleOut::One;
        QKT_GEMM_ONE_TILE(4);
        QKT_GEMM_ONE_TILE(5);
        QKT_GEMM_ONE_TILE(6);
        QKT_GEMM_ONE_TILE(7);
        QKT_GEMM_ONE_TILE(8);
        cur_phase ^= 1;
    }
}


template<
    typename T,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1,
    typename Engine2, typename Layout2
> 
__forceinline__ __device__ void warpgroup_cooperative_qkt_gemm_no_pipeline(
    Tensor<Engine0, Layout0> &sQ,	// (BLOCK_SIZE_M, HEAD_DIM_K)
    Tensor<Engine1, Layout1> &sKV,	// (BLOCK_SIZE_M, HEAD_DIM_K)
    Tensor<Engine2, Layout2> &rP,	// ((2, 2, 8), 1, 1)
    int idx_in_warpgroup
) {
    TiledMMA tiled_mma = (typename T::TiledMMA_QK_sQ){};
    ThrMMA thr_mma = tiled_mma.get_slice(idx_in_warpgroup);
    Tensor thr_mma_sQ = thr_mma.partition_fragment_A(sQ);	// (MMA, 1, 576/16=36)
    Tensor thr_mma_sKV = thr_mma.partition_fragment_B(sKV);	// (MMA, 1, 576/16=36)
    gemm<true, -1>(tiled_mma, thr_mma_sQ, thr_mma_sKV, rP);
}


// Compute O += PV, where P resides in register
template<
    typename T,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1,
    typename Engine2, typename Layout2
> 
__forceinline__ __device__ void warpgroup_cooperative_pv_gemm_localP(
    Tensor<Engine0, Layout0> &rP,	// ((2, 2, 8), 1, 1), fragment A layout
    Tensor<Engine1, Layout1> &sKV_half,	// (HEAD_DIM_V/2, PAGE_BLOCK_SIZE)
    Tensor<Engine2, Layout2> &rO,	// ((2, 2, 32), 1, 1)
    int idx_in_warpgroup
) {
    TiledMMA tiled_mma = (typename T::TiledMMA_PV_LocalP){};
    ThrMMA thr_mma = tiled_mma.get_slice(idx_in_warpgroup);
    Tensor rP_retiled = make_tensor(rP.data(), Layout<
        Shape<Shape<_2, _2, _2>, _1, _4>,
        Stride<Stride<_1, _2, _4>, _0, _8>
    >{});
    Tensor thr_mma_sKV_half = thr_mma.partition_fragment_B(sKV_half);	// (MMA, 1, 64/16=4)
    gemm<false, -1>(tiled_mma, rP_retiled, thr_mma_sKV_half, rO);
}


// Compute O += PV, where P resides in shared memory
template<
    typename T,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1,
    typename Engine2, typename Layout2
> 
__forceinline__ __device__ void warpgroup_cooperative_pv_gemm_remoteP(
    Tensor<Engine0, Layout0> &sP,
    Tensor<Engine1, Layout1> &sKV_half,	// (HEAD_DIM_V/2, PAGE_BLOCK_SIZE)
    Tensor<Engine2, Layout2> &rO,	// ((2, 2, 32), 1, 1)
    int idx_in_warpgroup
) {
    TiledMMA tiled_mma = (typename T::TiledMMA_PV_RemoteP){};
    ThrMMA thr_mma = tiled_mma.get_slice(idx_in_warpgroup);
    Tensor thr_mma_sP = thr_mma.partition_fragment_A(sP);
    Tensor thr_mma_sKV_half = thr_mma.partition_fragment_B(sKV_half);	// (MMA, 1, 64/16=4)
    gemm<false, -1>(tiled_mma, thr_mma_sP, thr_mma_sKV_half, rO);
}


template<
    typename T,
    bool DO_OOB_FILLING,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1,
    typename Engine2, typename Layout2,
    typename Engine3, typename Layout3,
    typename Engine4, typename Layout4
>
__forceinline__ __device__ void wg0_bunch_0(
    Tensor<Engine0, Layout0> &rPb,	// ((2, 2, 8), 1, 1)
    Tensor<Engine1, Layout1> &rP0,	// ((2, 2, 8), 1, 1)
    Tensor<Engine2, Layout2> &rO0,	// ((2, 2, 32), 1, 1)
    Tensor<Engine3, Layout3> &sScale0,	// (BLOCK_SIZE_M)
    Tensor<Engine4, Layout4> &sM,	// (BLOCK_SIZE_M)
    float rL[2],
    int rRightBorderForQSeq[2],
    float scale_softmax_log2,
    int start_token_idx,
    int idx_in_warpgroup
) {
    // This piece of code is tightly coupled [Accumulate's layout](https://docs.nvidia.com/cuda/parallel-thread-execution/_images/wgmma-64N16-D.png)
    CUTLASS_PRAGMA_UNROLL
    for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
        int row_idx = get_AorC_row_idx(local_row_idx, idx_in_warpgroup);

        // Mask, and get row-wise max
        float cur_max = MAX_INIT_VAL;
        CUTLASS_PRAGMA_UNROLL
        for (int i = local_row_idx ? 2 : 0; i < size(rP0); i += 4) {
            if constexpr (DO_OOB_FILLING) {
                int token_idx = start_token_idx + (i/4)*8 + idx_in_warpgroup%4*2;
                rP0(i) = token_idx < rRightBorderForQSeq[local_row_idx] ? rP0(i) : MAX_INIT_VAL;
                rP0(i+1) = token_idx+1 < rRightBorderForQSeq[local_row_idx] ? rP0(i+1) : MAX_INIT_VAL;
            }
            cur_max = max(cur_max, max(rP0(i), rP0(i+1)));
        }
        cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 1));
        cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 2));
        
        // Update sM and sL
        cur_max *= scale_softmax_log2;
        float new_max = max(sM(row_idx), cur_max);
        float scale_for_old = exp2f(sM(row_idx) - new_max);
        __syncwarp();   // Make sure all reads have finished before updating sM
        if (idx_in_warpgroup%4 == 0) {
            sScale0(row_idx) = scale_for_old;
            sM(row_idx) = new_max;
        }
        
        // Scale-O
        CUTLASS_PRAGMA_UNROLL
        for (int i = local_row_idx ? 2 : 0; i < size(rO0); i += 4) {
            rO0(i) *= scale_for_old;
            rO0(i+1) *= scale_for_old;
        }

        // Scale, exp, and get row-wise expsum
        float cur_sum = 0;
        CUTLASS_PRAGMA_UNROLL
        for (int i = local_row_idx ? 2 : 0; i < size(rP0); i += 4) {
            rP0(i) = exp2f(rP0(i)*scale_softmax_log2 - new_max);
            rP0(i+1) = exp2f(rP0(i+1)*scale_softmax_log2 - new_max);
            rPb(i) = (typename T::InputT)rP0(i);
            rPb(i+1) = (typename T::InputT)rP0(i+1);
            cur_sum += rP0(i) + rP0(i+1);
        }
        rL[local_row_idx] = rL[local_row_idx]*scale_for_old + cur_sum;
    }
}


template<
    typename T,
    bool IS_BLK0_LAST,
    bool IS_BLK1_LAST,
    bool IS_BLK2_LAST,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1,
    typename Engine2, typename Layout2,
    typename Engine3, typename Layout3,
    typename Engine4, typename Layout4,
    typename Engine5, typename Layout5
>
__forceinline__ __device__ void wg1_bunch_0(
    Tensor<Engine0, Layout0> &rP1b,	// ((2, 2, 8), 1, 1)
    Tensor<Engine1, Layout1> &sScale1,	// (BLOCK_SIZE_M)
    Tensor<Engine2, Layout2> &rO1,	// ((2, 2, 32), 1, 1)
    Tensor<Engine3, Layout3> &sM,	// (BLOCK_SIZE_M)
    float rL[2],
    int rRightBorderForQSeq[2],
    Tensor<Engine4, Layout4> const &sScale0,	// (BLOCK_SIZE_M)
    Tensor<Engine5, Layout5> &rP1,	// ((2, 2, 8), 1, 1)
    float scale_softmax_log2,
    int start_token_idx,
    int idx_in_warpgroup
) {
    CUTLASS_PRAGMA_UNROLL
    for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
        int row_idx = get_AorC_row_idx(local_row_idx, idx_in_warpgroup);

        // Mask, and get row-wise max
        float cur_max = MAX_INIT_VAL;
        CUTLASS_PRAGMA_UNROLL
        for (int i = local_row_idx ? 2 : 0; i < size(rP1); i += 4) {
            if constexpr (IS_BLK1_LAST || IS_BLK2_LAST) {
                // Need to apply the mask when either this block is the last one, or
                // the next block is the last one (because of the causal mask)
                int token_idx = start_token_idx + (i/4)*8 + idx_in_warpgroup%4*2;
                rP1(i) = token_idx < rRightBorderForQSeq[local_row_idx] ? rP1(i) : MAX_INIT_VAL;
                rP1(i+1) = token_idx+1 < rRightBorderForQSeq[local_row_idx] ? rP1(i+1) : MAX_INIT_VAL;
            } else if constexpr (IS_BLK0_LAST) {
                rP1(i) = rP1(i+1) = MAX_INIT_VAL;
            }
            cur_max = max(cur_max, max(rP1(i), rP1(i+1)));
        }
        cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 1));
        cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 2));
        cur_max *= scale_softmax_log2;

        float old_max = sM(row_idx);
        float new_max = max(old_max, cur_max);
        float scale_for_old = exp2f(old_max - new_max);
        __syncwarp();
        if (idx_in_warpgroup%4 == 0) {
            sM(row_idx) = new_max;
            sScale1(row_idx) = scale_for_old;
        }

        // Scale, exp, and get row-wise expsum
        float cur_sum = 0;
        if constexpr (!IS_BLK0_LAST) {
            CUTLASS_PRAGMA_UNROLL
            for (int i = local_row_idx ? 2 : 0; i < size(rP1); i += 4) {
                rP1(i) = exp2f(rP1(i)*scale_softmax_log2 - new_max);
                rP1(i+1) = exp2f(rP1(i+1)*scale_softmax_log2 - new_max);
                rP1b(i) = (typename T::InputT)rP1(i);
                rP1b(i+1) = (typename T::InputT)rP1(i+1);
                cur_sum += rP1(i) + rP1(i+1);
            }
        }

        // Scale O
        float cur_scale_for_o1 = scale_for_old * sScale0(row_idx);
        CUTLASS_PRAGMA_UNROLL
        for (int i = local_row_idx ? 2 : 0; i < size(rO1); i += 4) {
            rO1(i) *= cur_scale_for_o1;
            rO1(i+1) *= cur_scale_for_o1;
        }

        // Update rL
        rL[local_row_idx] = rL[local_row_idx]*cur_scale_for_o1 + cur_sum;
    }
}


// Save rPb (64x64, bfloat16/half) to sP using the stmatrix instruction
template<
    typename T,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1
>
__forceinline__ __device__ void save_rPb_to_sP(
    Tensor<Engine0, Layout0> &rPb,
    Tensor<Engine1, Layout1> &sP,
    int idx_in_warpgroup
) {
    auto r2s_copy = make_tiled_copy_C(
        Copy_Atom<SM90_U32x4_STSM_N, typename T::InputT>{},
        (typename T::TiledMMA_QK_sQ){}
    );
    ThrCopy thr_copy = r2s_copy.get_slice(idx_in_warpgroup);
    Tensor thr_copy_rPb = thr_copy.retile_S(rPb);
    Tensor thr_copy_sP = thr_copy.partition_D(sP);
    cute::copy(r2s_copy, thr_copy_rPb, thr_copy_sP);
}


// Retrieve rPb (64x64, bfloat16/half) from sP using the ldmatrix instruction
template<
    typename T,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1
>
__forceinline__ __device__ void retrieve_rP_from_sP(
    Tensor<Engine0, Layout0> &rPb,
    Tensor<Engine1, Layout1> const &sP,
    int idx_in_warpgroup
) {
    TiledCopy s2r_copy = make_tiled_copy_A(
        Copy_Atom<SM75_U32x4_LDSM_N, typename T::InputT>{},
        (typename T::TiledMMA_PV_LocalP){}
    );
    ThrCopy thr_copy = s2r_copy.get_slice(idx_in_warpgroup);
    Tensor thr_copy_sP = thr_copy.partition_S(sP);
    Tensor thr_copy_rPb = thr_copy.retile_D(rPb);
    cute::copy(s2r_copy, thr_copy_sP, thr_copy_rPb);
}


// Rescale rP0 and save the result to rPb
template<
    typename T,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1,
    typename Engine2, typename Layout2
>
__forceinline__ __device__ void wg0_scale_rP0(
    Tensor<Engine0, Layout0> const &sScale1,	// (BLOCK_M)
    Tensor<Engine1, Layout1> const &rP0,		// ((2, 2, 8), 1, 1)
    Tensor<Engine2, Layout2> &rPb,		// ((2, 2, 8), 1, 1)
    int idx_in_warpgroup
) {
    CUTLASS_PRAGMA_UNROLL
    for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
        int row_idx = get_AorC_row_idx(local_row_idx, idx_in_warpgroup);
        float scale_factor = sScale1(row_idx);
        CUTLASS_PRAGMA_UNROLL
        for (int i = local_row_idx ? 2 : 0; i < size(rP0); i += 4) {
            rPb(i) = (typename T::InputT)(rP0(i)*scale_factor);
            rPb(i+1) = (typename T::InputT)(rP0(i+1)*scale_factor);
        }
    }
}


// Rescale rO0 according to sScale1
template<
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1
>
__forceinline__ __device__ void wg0_rescale_rO0(
    Tensor<Engine0, Layout0> &rO0,
    Tensor<Engine1, Layout1> &sScale1,
    float rL[2],
    int idx_in_warpgroup
) {
    CUTLASS_PRAGMA_UNROLL
    for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
        int row_idx = get_AorC_row_idx(local_row_idx, idx_in_warpgroup);
        float scale_factor = sScale1(row_idx);
        CUTLASS_PRAGMA_UNROLL
        for (int i = local_row_idx ? 2 : 0; i < size(rO0); i += 4) {
            rO0(i) *= scale_factor;
            rO0(i+1) *= scale_factor;
        }
        rL[local_row_idx] *= scale_factor;
    }
}


// Fill out-of-bound V with 0.0
// We must fill it since it may contain NaN, which may propagate to the final result
template<
    typename T,
    typename Engine0, typename Layout0
>
__forceinline__ __device__ void fill_oob_V(
    Tensor<Engine0, Layout0> &sV,	// tile_to_shape(GMMA::Layout_MN_SW128_Atom<InputT>{}, Shape<Int<HALF_HEAD_DIM>, Int<T::PAGE_BLOCK_SIZE>>{}, LayoutRight{} );
    int valid_window_size,
    int idx_in_warpgroup
) {
    Tensor sV_int64 = make_tensor(
        make_smem_ptr((int64_t*)(sV.data().get().get())),
        tile_to_shape(
            GMMA::Layout_MN_SW128_Atom<cute::int64_t>{},
            Shape<Int<256/(64/16)>, Int<T::PAGE_BLOCK_SIZE>>{},
            LayoutRight{}
        )
    );
    valid_window_size = max(valid_window_size, 0);
    int head_dim_size = size<0>(sV_int64);	// 128%head_dim_size == 0 should holds
    for (int token_idx = valid_window_size + (idx_in_warpgroup/head_dim_size); token_idx < size<1>(sV); token_idx += (128/head_dim_size)) {
        sV_int64(idx_in_warpgroup%head_dim_size, token_idx) = 0;
    }
}

template<
    typename T,
    bool IS_R,
    typename Engine0, typename Layout0
>
__forceinline__ __device__ auto get_half_V(
    Tensor<Engine0, Layout0> &sK
) {
    Tensor sV = make_tensor(sK.data(), (typename T::SmemLayoutV){});
    return flat_divide(sV, Shape<Int<T::HEAD_DIM_V/2>, Int<T::PAGE_BLOCK_SIZE>>{})(_, _, Int<(int)IS_R>{}, _0{});
}

template<
    typename T,
    bool IS_BLK0_LAST,	// "BLK0" means block_idx+0, "BLK1" means block_idx+1, ...
    bool IS_BLK1_LAST,
    typename TMAParams,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1,
    typename Engine2, typename Layout2,
    typename Engine3, typename Layout3,
    typename Engine4, typename Layout4,
    typename Engine5, typename Layout5,
    typename Engine6, typename Layout6,
    typename Engine7, typename Layout7,
    typename Engine8, typename Layout8,
    typename Engine9, typename Layout9,
    typename Engine10, typename Layout10,
    typename Engine11, typename Layout11
>
__forceinline__ __device__ void wg0_subroutine(
    Tensor<Engine0, Layout0> &tma_gK,
    Tensor<Engine1, Layout1> &sQ,
    Tensor<Engine2, Layout2> &sK0,
    Tensor<Engine3, Layout3> &sK1,
    Tensor<Engine4, Layout4> &sP0,
    Tensor<Engine5, Layout5> &sP1,
    Tensor<Engine6, Layout6> &sM,
    Tensor<Engine7, Layout7> &sScale0,
    Tensor<Engine8, Layout8> &sScale1,
    Tensor<Engine9, Layout9> &rQ8,
    Tensor<Engine10, Layout10> &rP0,
    Tensor<Engine11, Layout11> &rO0,
    float rL[2],
    int rRightBorderForQSeq[2],
    TMABarrier barriers_K0[9],
    TMABarrier barriers_K1[9],
    bool &cur_phase_K0,
    const TMAParams &tma_params,
    const FwdParams &params,
    int* block_table_ptr,
    int seqlen_k,
    int block_idx,
    int end_block_idx,
    int idx_in_warpgroup
) {
    int start_token_idx = block_idx * T::PAGE_BLOCK_SIZE;
    #define GET_BLOCK_INDEX(block_idx) ((block_idx) >= end_block_idx ? 0 : __ldg(block_table_ptr + (block_idx)))
    int nxt_block0_index = GET_BLOCK_INDEX(block_idx+2);
    int nxt_block1_index = GET_BLOCK_INDEX(block_idx+3);

    Tensor sV0L = get_half_V<T, 0>(sK0);
    Tensor sV1L = get_half_V<T, 0>(sK1);

    Tensor rPb = make_tensor<T::InputT>(Shape<Shape<_2, _2, _2>, _1, _4>{});
    // Calc P0 = softmax(P0)
    wg0_bunch_0<T, IS_BLK0_LAST||IS_BLK1_LAST>(rPb, rP0, rO0, sScale0, sM, rL, rRightBorderForQSeq, params.scale_softmax_log2, start_token_idx, idx_in_warpgroup);
    NamedBarrier::arrive(T::NUM_THREADS, NamedBarriers::sScale0Ready);

    // Issue rO0 += rPb @ sV0L
    if constexpr (IS_BLK0_LAST) {
        fill_oob_V<T>(sV0L, seqlen_k-start_token_idx, idx_in_warpgroup);
        cutlass::arch::fence_view_async_shared();
    }
    warpgroup_cooperative_pv_gemm_localP<T>(rPb, sV0L, rO0, idx_in_warpgroup);

    // Wait for rO0, launch TMA for the next V0L
    cute::warpgroup_wait<0>();
    
    // Wait for warpgroup 1, rescale P0, notify warpgroup 1
    NamedBarrier::arrive_and_wait(T::NUM_THREADS, NamedBarriers::sScale1Ready);
    if constexpr (!IS_BLK0_LAST && !IS_BLK1_LAST) {
        launch_kv_tiles_copy_tma<0, 4>(tma_gK(_, _, nxt_block0_index), sK0, tma_params.tma_K, barriers_K0, idx_in_warpgroup);
    }
    wg0_scale_rP0<T>(sScale1, rP0, rPb, idx_in_warpgroup);
    save_rPb_to_sP<T>(rPb, sP0, idx_in_warpgroup);
    cutlass::arch::fence_view_async_shared();
    NamedBarrier::arrive(T::NUM_THREADS, NamedBarriers::sP0Ready);
    
    // Wait for warpgroup 1, rescale O0, issue rO0 += rPb @ sV1L
    if constexpr (!IS_BLK0_LAST) {
        if constexpr (IS_BLK1_LAST) {
            fill_oob_V<T>(sV1L, seqlen_k-start_token_idx-T::PAGE_BLOCK_SIZE, idx_in_warpgroup);
            cutlass::arch::fence_view_async_shared();
        }
        NamedBarrier::arrive_and_wait(T::NUM_THREADS, NamedBarriers::rO1sP0sV0RIssued);
        wg0_rescale_rO0(rO0, sScale1, rL, idx_in_warpgroup);
        warpgroup_cooperative_pv_gemm_remoteP<T>(sP1, sV1L, rO0, idx_in_warpgroup);
    }
    
    // Issue P0 = Q @ K0^T
    // Since TMAs for these 4 tiles are launched right after rO0 += rPb @ sV0L finishes, they should have already finished. Therefore, we issue the first 4 tiles to fill the pipeline.
    if constexpr (!IS_BLK0_LAST && !IS_BLK1_LAST) {
        warpgroup_cooperative_qkt_gemm<T, 0>(sQ, sK0, rP0, rQ8, barriers_K0, cur_phase_K0, idx_in_warpgroup);
    }

    // Wait for rO0 += rPb @ sV1L, launch TMA
    if (!IS_BLK0_LAST && !IS_BLK1_LAST && __builtin_expect(block_idx+3 < end_block_idx, true)) {
        cute::warpgroup_wait<4>();
        launch_kv_tiles_copy_tma<0, 4>(tma_gK(_, _, nxt_block1_index), sK1, tma_params.tma_K, barriers_K1, idx_in_warpgroup);
    }
    
    // Issue P0 = Q @ K0^T
    if constexpr (!IS_BLK0_LAST && !IS_BLK1_LAST) {
        warpgroup_cooperative_qkt_gemm<T, 2>(sQ, sK0, rP0, rQ8, barriers_K0, cur_phase_K0, idx_in_warpgroup);
    }
    
    // Wait for P0 = Q @ K0^T
    cute::warpgroup_wait<0>();
}


template<
    typename T,
    bool IS_BLK0_LAST,
    bool IS_BLK1_LAST,
    bool IS_BLK2_LAST,
    typename TMAParams,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1,
    typename Engine2, typename Layout2,
    typename Engine3, typename Layout3,
    typename Engine4, typename Layout4,
    typename Engine5, typename Layout5,
    typename Engine6, typename Layout6,
    typename Engine7, typename Layout7,
    typename Engine8, typename Layout8,
    typename Engine9, typename Layout9,
    typename Engine10, typename Layout10,
    typename Engine11, typename Layout11
>
__forceinline__ __device__ void wg1_subroutine(
    Tensor<Engine0, Layout0> &tma_gK,
    Tensor<Engine1, Layout1> &sQ,
    Tensor<Engine2, Layout2> &sK0,
    Tensor<Engine3, Layout3> &sK1,
    Tensor<Engine4, Layout4> &sP0,
    Tensor<Engine5, Layout5> &sP1,
    Tensor<Engine6, Layout6> &sM,
    Tensor<Engine7, Layout7> &sScale0,
    Tensor<Engine8, Layout8> &sScale1,
    Tensor<Engine9, Layout9> &rQ8,
    Tensor<Engine10, Layout10> &rP1,
    Tensor<Engine11, Layout11> &rO1,
    float rL[2],
    int rRightBorderForQSeq[2],
    TMABarrier barriers_K0[9],
    TMABarrier barriers_K1[9],
    bool &cur_phase_K1,
    const TMAParams &tma_params,
    const FwdParams &params,
    int* block_table_ptr,
    int seqlen_k,
    int block_idx,
    int end_block_idx,
    int idx_in_warpgroup
) {
    int start_token_idx = block_idx * T::PAGE_BLOCK_SIZE;
    int nxt_block0_index = GET_BLOCK_INDEX(block_idx+2);
    int nxt_block1_index = GET_BLOCK_INDEX(block_idx+3);

    Tensor rP1b = make_tensor<T::InputT>(Shape<Shape<_2, _2, _2>, _1, _4>{});
    
    Tensor sV0R = get_half_V<T, 1>(sK0);
    Tensor sV1R = get_half_V<T, 1>(sK1);

    // Wait for rP1 and warpgroup 0, run bunch 1, notify warpgroup 0
    NamedBarrier::arrive_and_wait(T::NUM_THREADS, NamedBarriers::sScale0Ready);
    wg1_bunch_0<T, IS_BLK0_LAST, IS_BLK1_LAST, IS_BLK2_LAST>(rP1b, sScale1, rO1, sM, rL, rRightBorderForQSeq, sScale0, rP1, params.scale_softmax_log2, start_token_idx+T::PAGE_BLOCK_SIZE, idx_in_warpgroup);
    NamedBarrier::arrive(T::NUM_THREADS, NamedBarriers::sScale1Ready);

    // Save rPb to sP, and issue rO1 += rP1b @ sV1R
    // We do this after notifying warpgroup 1, since both "saving rPb to sP" and "issuing" WGMMA are high-latency operations
    if constexpr (!IS_BLK0_LAST) {
        save_rPb_to_sP<T>(rP1b, sP1, idx_in_warpgroup);
    }
    if constexpr (!IS_BLK0_LAST) {
        if constexpr (IS_BLK1_LAST) {
            fill_oob_V<T>(sV1R, seqlen_k-start_token_idx-T::PAGE_BLOCK_SIZE, idx_in_warpgroup);
            cutlass::arch::fence_view_async_shared();
        }
        warpgroup_cooperative_pv_gemm_localP<T>(rP1b, sV1R, rO1, idx_in_warpgroup);
        if constexpr (!IS_BLK1_LAST) {
            // We use this proxy for making sP1 visible to the async proxy
            // We skip it if IS_BLK1_LAST, since in that case we have already put a fence
            cutlass::arch::fence_view_async_shared();
        }
    }
    
    // Wait for sP0, issue rO1 += sP0 @ sV0R, notify warpgroup 0
    NamedBarrier::arrive_and_wait(T::NUM_THREADS, NamedBarriers::sP0Ready);
    if constexpr (IS_BLK0_LAST) {
        fill_oob_V<T>(sV0R, seqlen_k-start_token_idx, idx_in_warpgroup);
        cutlass::arch::fence_view_async_shared();
    }
    warpgroup_cooperative_pv_gemm_remoteP<T>(sP0, sV0R, rO1, idx_in_warpgroup);
    if constexpr (!IS_BLK0_LAST) {
        NamedBarrier::arrive(T::NUM_THREADS, NamedBarriers::rO1sP0sV0RIssued);
    }
    
    // Wait for rO1 += rP1b @ sV1R, launch TMA for the next V1R
    if constexpr (!IS_BLK0_LAST && !IS_BLK1_LAST && !IS_BLK2_LAST) {
        cute::warpgroup_wait<1>();
        launch_kv_tiles_copy_tma<4, 9>(tma_gK(_, _, nxt_block1_index), sK1, tma_params.tma_K, barriers_K1, idx_in_warpgroup);
    }
    
    // Wait for rO1 += sP0 @ sV0R, launch TMA for the next V0R
    if constexpr (!IS_BLK0_LAST && !IS_BLK1_LAST) {
        cute::warpgroup_wait<0>();
        launch_kv_tiles_copy_tma<4, 9>(tma_gK(_, _, nxt_block0_index), sK0, tma_params.tma_K, barriers_K0, idx_in_warpgroup);
    }

    if constexpr (!IS_BLK0_LAST && !IS_BLK1_LAST && !IS_BLK2_LAST) {
        // Issue rP1 = sQ @ sK1, wait
        warpgroup_cooperative_qkt_gemm<T, 1>(sQ, sK1, rP1, rQ8, barriers_K1, cur_phase_K1, idx_in_warpgroup);
    }
    
    // Keep warpgroup_wait outside the branch so nvcc does not serialize WGMMA with extra barriers.
    cute::warpgroup_wait<0>();
}

// Stage one item's published rows in the row-major shared buffer (replaces FlashMLA's store_o staging).
// 520 float32 per row (FlashMLA's stride against bank conflicts). A root row stores bf16 pairs of O / L' at
// the head of its own 2080-byte row; a partial row stores its raw float32 O (never normalized).
template<
    typename T,
    typename Engine0, typename Layout0
>
__forceinline__ __device__ void stage_item_rows(
    Tensor<Engine0, Layout0> &rO,	// ((2, 2, 32), 1, 1)
    float rL[2],                    // pruned L' of the two rows held by this thread
    int rMode[2],                   // 0: not published by this item, 1: root, 2: a tree node
    char* sO_addr,
    int warpgroup_idx,
    int idx_in_warpgroup
) {
    using InputT = typename T::InputT;
    CUTLASS_PRAGMA_UNROLL
    for (int idx = 0; idx < size(rO); idx += 2) {
        int local_row = idx%4 >= 2 ? 1 : 0;
        int row = (idx_in_warpgroup/32)*16 + (idx_in_warpgroup%32/4) + (local_row ? 8 : 0);
        int col = warpgroup_idx*256 + (idx_in_warpgroup%4)*2 + idx/4*8;
        if (rMode[local_row] == 0) {
            continue;
        }
        if (rMode[local_row] == 1) {
            InputT b0 = (InputT)(rO(idx) / rL[local_row]);
            InputT b1 = (InputT)(rO(idx+1) / rL[local_row]);
            *(InputT*)(sO_addr + row*2080 + col*2) = b0;
            *(InputT*)(sO_addr + row*2080 + col*2 + 2) = b1;
        } else {
            *(float2*)(sO_addr + (row*520 + col)*4) = float2 { rO(idx), rO(idx+1) };
        }
    }
    cutlass::arch::fence_view_async_shared();
}

// Stage a full 64-row m_block of roots for one TMA store (FlashMLA's no-split staging): bf16 of O / L' in the
// K-major SW128 tile layout, stmatrix copies from the register fragments.
template<
    typename T,
    bool ROOT,
    typename Engine0, typename Layout0
>
__forceinline__ __device__ void stage_block_tma(
    Tensor<Engine0, Layout0> &rO,	// ((2, 2, 32), 1, 1)
    float rL[2],
    char* sO_addr,
    int warpgroup_idx,
    int idx_in_warpgroup
) {
    using ElemT = std::conditional_t<ROOT, typename T::InputT, float>;
    Tensor sOutputBuf = make_tensor(make_smem_ptr((ElemT*)sO_addr), tile_to_shape(
        GMMA::Layout_K_SW128_Atom<ElemT>{},
        Shape<Int<T::BLOCK_SIZE_M>, Int<T::HEAD_DIM_V>>{}
    ));
    Tensor sMyOutputBuf = local_tile(sOutputBuf, Shape<_64, _256>{}, make_coord(_0{}, warpgroup_idx));
    TiledCopy r2s_tiled_copy = make_tiled_copy_C(
        Copy_Atom<SM90_U32x4_STSM_N, ElemT>{},
        (typename T::TiledMMA_PV_LocalP){}
    );
    ThrCopy r2s_thr_copy = r2s_tiled_copy.get_slice(idx_in_warpgroup);
    Tensor rOb = make_tensor_like<ElemT>(rO);
    CUTLASS_PRAGMA_UNROLL
    for (int idx = 0; idx < size(rO); ++idx) {
        rOb(idx) = (ElemT)(rO(idx) / rL[idx%4 >= 2]);
    }
    Tensor r2s_thr_copy_rOb = r2s_thr_copy.retile_S(rOb);
    Tensor r2s_thr_copy_sMyOutputBuf = r2s_thr_copy.partition_D(sMyOutputBuf);
    cute::copy(r2s_tiled_copy, r2s_thr_copy_rOb, r2s_thr_copy_sMyOutputBuf);
    cutlass::arch::fence_view_async_shared();
}

// Thread 0: TMA store of the staged 64-row tile into out rows [row0, row0 + 64).
template<
    typename T,
    bool ROOT,
    typename TmaParams_
>
__forceinline__ __device__ void store_block_tma(
    TmaParams_ const &tma_params,
    char* sO_addr,
    int64_t row0
) {
    using ElemT = std::conditional_t<ROOT, typename T::InputT, float>;
    Tensor sOutputBuf = make_tensor(make_smem_ptr((ElemT*)sO_addr), tile_to_shape(
        GMMA::Layout_K_SW128_Atom<ElemT>{},
        Shape<Int<T::BLOCK_SIZE_M>, Int<T::HEAD_DIM_V>>{}
    ));
    Tensor tma_all = tma_params.tma_O.get_tma_tensor(tma_params.shape_O);
    auto layout_win = make_layout(Shape<Int<T::BLOCK_SIZE_M>, Int<T::HEAD_DIM_V>>{}, stride(tma_all.layout()));
    Tensor tma_win = make_tensor(tma_all.data() + tma_all.layout()(row0, 0), layout_win);
    auto thr_tma = tma_params.tma_O.get_slice(_0{});
    cute::copy(tma_params.tma_O, thr_tma.partition_S(sOutputBuf), thr_tma.partition_D(tma_win));
    cute::tma_store_arrive();
}

// Stage a full 64-row m_block of partial rows as 64 consecutive partial records (O[512] | L | M | pad): row-major,
// PARTIAL_WIDTH floats per row, so one bulk store publishes the whole tile (execution only: same record bits as
// the row-by-row publish).
template<
    typename T,
    typename Engine0, typename Layout0
>
__forceinline__ __device__ void stage_block_partial(
    Tensor<Engine0, Layout0> &rO,	// ((2, 2, 32), 1, 1)
    char* sO_addr,
    int warpgroup_idx,
    int idx_in_warpgroup
) {
    CUTLASS_PRAGMA_UNROLL
    for (int idx = 0; idx < size(rO); idx += 2) {
        int local_row = idx%4 >= 2 ? 1 : 0;
        int row = (idx_in_warpgroup/32)*16 + (idx_in_warpgroup%32/4) + (local_row ? 8 : 0);
        int col = warpgroup_idx*256 + (idx_in_warpgroup%4)*2 + idx/4*8;
        *(float2*)(sO_addr + (row*PARTIAL_WIDTH + col)*4) = float2 { rO(idx), rO(idx+1) };
    }
    cutlass::arch::fence_view_async_shared();
}

// Thread 0: TMA store of the staged 64-row tile into m_block `m_block` of request `req`, through a (rows per request,
// 512, 1, B) map (FlashMLA's no-split output map): rows past the request's last row are clipped by the map. Valid only
// when every request has the same number of rows (decode, uniform verify).
template<
    typename T,
    typename TmaParams_
>
__forceinline__ __device__ void store_block_uniform(
    TmaParams_ const &tma_params,
    char* sO_addr,
    int req,
    int m_block
) {
    using ElemT = typename T::InputT;
    Tensor sOutputBuf = make_tensor(make_smem_ptr((ElemT*)sO_addr), tile_to_shape(
        GMMA::Layout_K_SW128_Atom<ElemT>{},
        Shape<Int<T::BLOCK_SIZE_M>, Int<T::HEAD_DIM_V>>{}
    ));
    Tensor tma_gO = tma_params.tma_Ou.get_tma_tensor(tma_params.shape_Ou)(_, _, 0, req);
    Tensor my_tma_gO = flat_divide(tma_gO, Shape<Int<T::BLOCK_SIZE_M>, Int<T::HEAD_DIM_V>>{})(_, _, m_block, _0{});
    auto thr_tma = tma_params.tma_Ou.get_slice(_0{});
    cute::copy(tma_params.tma_Ou, thr_tma.partition_S(sOutputBuf), thr_tma.partition_D(my_tma_gO));
    cute::tma_store_arrive();
}

// Launch the TMA load of the 64-row Q window that starts at global row `row_base`. The tensor map covers exactly
// the q rows, so a window that runs past the last row is filled by TMA's out-of-bound rule; those rows are
// never published.
template<
    typename T,
    typename TmaParams_, typename Tensor0
>
__forceinline__ __device__ void launch_q_window_copy(
    TmaParams_ const &tma_params,
    const FwdParams &params,
    int row_base,
    Tensor0 &sQ,
    TMABarrier* barrier_Q
) {
    if (threadIdx.x == 0) {
        Tensor tma_gQ_all = tma_params.tma_Q.get_tma_tensor(tma_params.shape_Q);	// (total_rows, HEAD_DIM_K)
        auto layout_win = make_layout(Shape<Int<T::BLOCK_SIZE_M>, Int<T::HEAD_DIM_K>>{}, stride(tma_gQ_all.layout()));
        Tensor tma_gQ = make_tensor(tma_gQ_all.data() + tma_gQ_all.layout()(row_base, 0), layout_win);
        auto thr_tma = tma_params.tma_Q.get_slice(_0{});
        cute::copy(
            tma_params.tma_Q.with(reinterpret_cast<typename TMABarrier::ValueType &>(*barrier_Q), 0, cute::TMA::CacheHintSm90::EVICT_FIRST),
            thr_tma.partition_S(tma_gQ),
            thr_tma.partition_D(sQ)
        );
        barrier_Q->arrive_and_expect_tx(64*576*2);
    }
}

// Pending tree node on a worker's stack level (execution only): O as partial records (row*PARTIAL_WIDTH + column),
// then (L, M) of both local rows per thread, so a thread only ever loads what it stored.
static constexpr int NODE_LM_OFFSET = 64*PARTIAL_WIDTH;

// Fold (in place) the live leaf state of this thread's two rows (O in rO, L in rL, M in rM) as the right operand b of a
// tree merge whose left operand a is the pending node `node` (O[512] | L | M per row). The scalar graph is the combine
// kernel's merge: M = max(Ma, Mb); sa = exp2f(Ma - M); sb = exp2f(Mb - M); O = Oa*sa + Ob*sb; L = La*sa + Lb*sb (fmad
// off: products round separately and IEEE addition commutes, so which operand lives in registers is irrelevant). Every
// thread computes its rows' scales from the same operands. Rows past the m_block (never published) are skipped.
template<
    typename T,
    typename Engine0, typename Layout0
>
__forceinline__ __device__ void fold_node(
    Tensor<Engine0, Layout0> &rO,	// ((2, 2, 32), 1, 1)
    float rL[2],
    float rM[2],
    const float* node,
    int rows,
    int warpgroup_idx,
    int idx_in_warpgroup
) {
    CUTLASS_PRAGMA_UNROLL
    for (int local_row = 0; local_row < 2; ++local_row) {
        const int row = get_AorC_row_idx(local_row, idx_in_warpgroup);
        if (row < rows) {
            const float Ma = node[NODE_LM_OFFSET + threadIdx.x*4 + local_row*2 + 1];
            const float Mx = max(Ma, rM[local_row]);
            const float sa = exp2f(Ma - Mx);
            const float sb = exp2f(rM[local_row] - Mx);
            rL[local_row] = node[NODE_LM_OFFSET + threadIdx.x*4 + local_row*2]*sa + rL[local_row]*sb;
            rM[local_row] = Mx;
            const float* a = node + row*PARTIAL_WIDTH + warpgroup_idx*256 + (idx_in_warpgroup%4)*2;
            CUTLASS_PRAGMA_UNROLL
            for (int i = local_row ? 2 : 0; i < size(rO); i += 4) {
                rO(i) = rO(i)*sb + a[i/4*8]*sa;
                rO(i+1) = rO(i+1)*sb + a[i/4*8+1]*sa;
            }
        }
    }
}

// Push the live leaf state of this thread's rows as a pending node (O from every thread, L and M once per row), and
// reset the rows' running max for the next segment.
template<
    typename T,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1
>
__forceinline__ __device__ void push_node(
    Tensor<Engine0, Layout0> &rO,	// ((2, 2, 32), 1, 1)
    Tensor<Engine1, Layout1> &sM,
    float rL[2],
    float rM[2],
    float* node,
    int rows,
    int warpgroup_idx,
    int idx_in_warpgroup
) {
    CUTLASS_PRAGMA_UNROLL
    for (int idx = 0; idx < size(rO); idx += 2) {
        const int local_row = idx%4 >= 2 ? 1 : 0;
        const int row = get_AorC_row_idx(local_row, idx_in_warpgroup);
        const int col = warpgroup_idx*256 + (idx_in_warpgroup%4)*2 + idx/4*8;
        if (row >= rows) continue;
        *(float2*)(node + row*PARTIAL_WIDTH + col) = float2 { rO(idx), rO(idx+1) };
    }
    *(float4*)(node + NODE_LM_OFFSET + threadIdx.x*4) = float4 { rL[0], rM[0], rL[1], rM[1] };
    if (warpgroup_idx == 0 && idx_in_warpgroup%4 == 0) {
        sM(get_AorC_row_idx(0, idx_in_warpgroup)) = MAX_INIT_VAL_SM;
        sM(get_AorC_row_idx(1, idx_in_warpgroup)) = MAX_INIT_VAL_SM;
    }
}

// Fold the pending nodes of a run's post-order walk after its c-th segment (streaming binary counter: while the top
// node's size equals the live node's, i.e. c is even; at the run's last segment, every pending node). rL holds the
// segment's reduced L; rM receives M. Returns the number of nodes left pending.
template<
    typename T,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1
>
__forceinline__ __device__ int fold_pending(
    Tensor<Engine0, Layout0> &rO,
    Tensor<Engine1, Layout1> &sM,
    float rL[2],
    float rM[2],
    const float* stack,
    int c,
    bool last,
    int rows,
    int warpgroup_idx,
    int idx_in_warpgroup
) {
    __syncthreads();    // every thread sees the segment's final running max (last written by warpgroup 1's bunch)
    rM[0] = sM(get_AorC_row_idx(0, idx_in_warpgroup));
    rM[1] = sM(get_AorC_row_idx(1, idx_in_warpgroup));
    // pending nodes before the c-th segment: one per set bit of c - 1
    unsigned v = (unsigned)(c - 1);
    v = v - ((v >> 1) & 0x55555555u);
    v = (v & 0x33333333u) + ((v >> 2) & 0x33333333u);
    int depth = (int)((((v + (v >> 4)) & 0x0F0F0F0Fu)*0x01010101u) >> 24);
    #pragma unroll 1
    for (; depth > 0 && ((c & 1) == 0 || last); c >>= 1) {
        --depth;
        fold_node<T>(rO, rL, rM, stack + depth*(T::BLOCK_SIZE_M + 2)*PARTIAL_WIDTH, rows, warpgroup_idx, idx_in_warpgroup);
    }
    return depth;
}

template<typename T, typename TmaParams_>
__global__ void __launch_bounds__(T::NUM_THREADS, 1, 1)
mla_sm90_leaf_kernel(__grid_constant__ const FwdParams params, __grid_constant__ const TmaParams_ tma_params) {
    // Grid: [workers, 1, 1]. Items (runs of segments of one m_block, aligned to tree nodes) are sorted by descending cost;
    // worker w walks the item list the plan kernel built for it. It resets the accumulator at every segment and folds
    // the run's segments in place with the tree's merges, keeping pending nodes on its stack in global memory; which
    // worker holds an item, in which order, and how the trees were cut into runs never reaches the arithmetic.
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900)) || (defined(__CLION_IDE__) || defined(__VSCODE_IDE__))
    const int worker_idx = blockIdx.x;
    const int warpgroup_idx = threadIdx.x / 128;
    const int idx_in_warpgroup = threadIdx.x % 128;

    extern __shared__ char wksp_buf[];
    using SharedMemoryPlan = typename T::SharedMemoryPlan;
    SharedMemoryPlan &plan = *reinterpret_cast<SharedMemoryPlan*>(wksp_buf);
    Tensor sQ = make_tensor(make_smem_ptr(plan.smem_sQ.data()), (typename T::SmemLayoutQ){});
    Tensor sK0 = make_tensor(make_smem_ptr(plan.smem_sK0.data()), (typename T::SmemLayoutK){});
    Tensor sK1 = make_tensor(make_smem_ptr(plan.smem_sK1.data()), (typename T::SmemLayoutK){});
    Tensor sP0 = make_tensor(make_smem_ptr(plan.smem_sP0.data()), (typename T::SmemLayoutP0){});
    Tensor sP1 = flat_divide(sQ, Shape<Int<T::BLOCK_SIZE_M>, Int<T::PAGE_BLOCK_SIZE>>{})(_, _, _0{}, _8{}); // Overlap with sQ's 8-th tile
    Tensor sM = make_tensor(make_smem_ptr(plan.smem_sM.data()), make_shape(Int<T::BLOCK_SIZE_M>{}));
    Tensor sL_reduction_wksp = make_tensor(make_smem_ptr(plan.sL_reduction_wksp.data()), make_shape(Int<2*T::BLOCK_SIZE_M>{}));
    Tensor sScale0 = make_tensor(make_smem_ptr(plan.smem_sScale0.data()), make_shape(Int<T::BLOCK_SIZE_M>{}));
    Tensor sScale1 = make_tensor(make_smem_ptr(plan.smem_sScale1.data()), make_shape(Int<T::BLOCK_SIZE_M>{}));
    char* sO_addr = (char*)plan.smem_sK0.data();	// Overlap with sK0 and sK1

    // Prefetch TMA descriptors
    if (threadIdx.x == 0) {
        cute::prefetch_tma_descriptor(tma_params.tma_Q.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_K.get_tma_descriptor());
    }

    Tensor tma_gK = tma_params.tma_K.get_tma_tensor(tma_params.shape_K)(_, _, 0, _);
    TMABarrier* barriers_K0 = plan.barriers_K0;
    TMABarrier* barriers_K1 = plan.barriers_K1;
    TMABarrier* barrier_Q = &(plan.barrier_Q);

    // Initialize TMA barriers
    if (threadIdx.x == 0) {
        barrier_Q->init(1);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < 9; ++i) {
            barriers_K0[i].init(1);
            barriers_K1[i].init(1);
        }
        cutlass::arch::fence_view_async_shared();
    }
    __syncthreads();
    bool cur_phase_Q = 0, cur_phase_K0 = 0, cur_phase_K1 = 0;

    int item_pos = params.heads[worker_idx];
    if (item_pos < 0) {
        if (params.pdl) {
            cudaTriggerProgrammaticLaunchCompletion();
        }
        return;
    }
    const int h_q = params.h_q;
    const int seg_tokens = params.seg_tokens;

    // Copy the first Q window
    launch_q_window_copy<T>(tma_params, params, params.items[item_pos].row_base, sQ, barrier_Q);

    #pragma unroll 1
    while (item_pos >= 0) {
        constexpr int kBlockN = T::PAGE_BLOCK_SIZE;
        const Item item = params.items[item_pos];
        const int batch_idx = item.req;
        const int seqlen_k = __ldg(params.seqused_k + batch_idx);
        const int q_begin = __ldg(params.cu_seqlens_q + batch_idx);
        const int q_len = __ldg(params.cu_seqlens_q + batch_idx + 1) - q_begin;
        const int num_rows = q_len*h_q;
        const int first_row = item.m_block*T::BLOCK_SIZE_M;
        const int rows_here = min(num_rows - first_row, (int)T::BLOCK_SIZE_M);
        const int pos_first = seqlen_k - q_len + first_row / h_q;
        const int pos_last = seqlen_k - q_len + (first_row + rows_here - 1) / h_q;
        const int nmax = pos_last / seg_tokens + 1;
        const int k_last = item.k1 - 1;
        int* block_table_ptr = params.block_table + batch_idx*params.block_table_stride;	// (/) : (1)

        // The run's segments are walked as one FlashMLA block pipeline over [k0*SEG, border): SEG is a multiple of 128, so
        // segment ends are ends of block pairs, where a warpgroup pair iteration is complete. There the worker reduces the
        // segment's L, folds the tree nodes the segment completes, pushes the node, and resets the accumulator (rO, rL,
        // sM) as at an item start; the pipeline (next blocks' TMA and QK) runs on. Masking and the P of a segment's
        // first and last blocks are bitwise the same as in a walk that ends there (the border mask of a non-final
        // segment masks nothing; QK of any block uses Q tiles 0..8 in order).
        const int start_block_idx = item.k0*(seg_tokens/kBlockN);
        const int end_block_idx = (min(item.k1*seg_tokens, pos_last + 1) + kBlockN - 1) / kBlockN;

        // Row r of the request (token r/h_q) sits at logical position p = seqused_k - q_len + r/h_q and attends keys
        // [0, p]. Rows past the request's last row reuse the last row's border; they are never published. Within an
        // m_block borders differ by less than 64, so masking the last two KV blocks (FlashMLA's rule) covers every row.
        int rRightBorderForQSeq[2];
        CUTLASS_PRAGMA_UNROLL
        for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
            int row_idx = get_AorC_row_idx(local_row_idx, idx_in_warpgroup);
            int g = min(first_row + row_idx, num_rows - 1);
            int pos = seqlen_k - q_len + g / h_q;
            rRightBorderForQSeq[local_row_idx] = min(pos + 1, end_block_idx*kBlockN);
        }

        // Copy K0 and K1
        launch_kv_tiles_copy_tma<0, 9>(tma_gK(_, _, __ldg(block_table_ptr + start_block_idx)), sK0, tma_params.tma_K, barriers_K0, threadIdx.x);
        if (start_block_idx+1 < end_block_idx) {
            launch_kv_tiles_copy_tma<4, 9>(tma_gK(_, _, __ldg(block_table_ptr + start_block_idx+1)), sK1, tma_params.tma_K, barriers_K1, threadIdx.x);
            launch_kv_tiles_copy_tma<0, 4>(tma_gK(_, _, __ldg(block_table_ptr + start_block_idx+1)), sK1, tma_params.tma_K, barriers_K1, threadIdx.x);
        }

        Tensor rO = partition_fragment_C((typename T::TiledMMA_PV_LocalP){}, Shape<Int<T::BLOCK_SIZE_M>, Int<T::HEAD_DIM_V / 2>>{});	// ((2, 2, 32), 1, 1)
        float rL[2];
        rL[0] = rL[1] = 0.0f;
        float rM[2];

        // Reset at every item start
        cute::fill(rO, 0.);
        if (threadIdx.x < size(sM)) {
            sM[threadIdx.x] = MAX_INIT_VAL_SM;
        }

        // Wait for Q
        barrier_Q->wait(cur_phase_Q);
        cur_phase_Q ^= 1;

        using InputT = typename T::InputT;
        Tensor rQ8 = make_tensor<InputT>(Shape<Shape<_2, _2, _2>, _1, _4>{});
        retrieve_rP_from_sP<T>(rQ8, local_tile(sQ, Shape<_64, _64>{}, Coord<_0, _8>{}), idx_in_warpgroup);

        // Reduce this thread's rL[0..1] over the warp and across warpgroups (every thread ends with the rows' L)
        #define REDUCE_L() do { \
            rL[0] += __shfl_xor_sync(0xffffffff, rL[0], 1); \
            rL[0] += __shfl_xor_sync(0xffffffff, rL[0], 2); \
            rL[1] += __shfl_xor_sync(0xffffffff, rL[1], 1); \
            rL[1] += __shfl_xor_sync(0xffffffff, rL[1], 2); \
            int my_row_ = get_AorC_row_idx(0, idx_in_warpgroup); \
            if (idx_in_warpgroup%4 == 0) { \
                sL_reduction_wksp[my_row_ + warpgroup_idx*64] = rL[0]; \
                sL_reduction_wksp[my_row_ + 8 + warpgroup_idx*64] = rL[1]; \
            } \
            __syncthreads(); \
            if (warpgroup_idx == 0) { \
                rL[0] += sL_reduction_wksp[my_row_ + 64]; \
                rL[1] += sL_reduction_wksp[my_row_ + 8 + 64]; \
            } else { \
                if (idx_in_warpgroup%4 == 0) { \
                    sL_reduction_wksp[my_row_] += rL[0]; \
                    sL_reduction_wksp[my_row_ + 8] += rL[1]; \
                } \
                __syncwarp(); \
                rL[0] = sL_reduction_wksp[my_row_]; \
                rL[1] = sL_reduction_wksp[my_row_+8]; \
            } \
        } while (0)

        // Post-order walk of the run's subtree after segment k: fold the pending nodes whose size equals the live node's
        // (at the run's last segment: every pending node, the tree's right fold); push the live node unless the run ends.
        #define FOLD_SEGMENT(C, IS_LAST) \
            fold_pending<T>(rO, sM, rL, rM, \
                            params.stack + (int64_t)worker_idx*params.stack_depth*(T::BLOCK_SIZE_M + 2)*PARTIAL_WIDTH, \
                            (C), (IS_LAST), rows_here, warpgroup_idx, idx_in_warpgroup)

        // Segment boundary after block pair (BLOCK_IDX, BLOCK_IDX + 1): segment k = (BLOCK_IDX + 2)/(SEG/64) - 1 ends
        #define SEGMENT_BOUNDARY(BLOCK_IDX) do { \
            REDUCE_L(); \
            const int depth_ = FOLD_SEGMENT((BLOCK_IDX + 2)/(seg_tokens/kBlockN) - item.k0, false); \
            push_node<T>(rO, sM, rL, rM, (float*)params.stack + ((int64_t)worker_idx*params.stack_depth + depth_)*(T::BLOCK_SIZE_M + 2)*PARTIAL_WIDTH, rows_here, warpgroup_idx, idx_in_warpgroup); \
            rL[0] = rL[1] = 0.0f; \
            cute::fill(rO, 0.); \
            cutlass::arch::fence_view_async_shared(); \
            __syncthreads(); \
        } while (0)
        #define AT_BOUNDARY(BLOCK_IDX) ((BLOCK_IDX) + 2 < end_block_idx && ((BLOCK_IDX) + 2) % (seg_tokens/kBlockN) == 0)

        if (warpgroup_idx == 0) {
            // Warpgroup 0
            Tensor rP0 = make_tensor<float>((typename T::rP0Layout){});

            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < 9; ++i) {
                if (idx_in_warpgroup == 0)
                    barriers_K0[i].arrive_and_expect_tx(64*64*2);
                barriers_K0[i].wait(cur_phase_K0);
            }
            cur_phase_K0 ^= 1;

            if (start_block_idx-16777216 < end_block_idx) {     // NOTE We use this `if` to prevent register spilling
                warpgroup_cooperative_qkt_gemm_no_pipeline<T>(sQ, sK0, rP0, idx_in_warpgroup);
            }
            NamedBarrier::arrive_and_wait(128, NamedBarriers::sMInitialized);
            cute::warpgroup_wait<0>();

            #define LAUNCH_WG0_SUBROUTINE(IS_BLK0_LAST, IS_BLK1_LAST) \
                wg0_subroutine<T, IS_BLK0_LAST, IS_BLK1_LAST>( \
                    tma_gK, sQ, sK0, sK1, sP0, sP1, sM, sScale0, sScale1, \
                    rQ8, rP0, rO, rL, rRightBorderForQSeq, \
                    barriers_K0, barriers_K1, cur_phase_K0, \
                    tma_params, params, \
                    block_table_ptr, seqlen_k, block_idx, end_block_idx, idx_in_warpgroup \
                );

            int block_idx = start_block_idx;
            #pragma unroll 1
            for (; block_idx < end_block_idx-2; block_idx += 2) {
                LAUNCH_WG0_SUBROUTINE(false, false);
                if (AT_BOUNDARY(block_idx)) SEGMENT_BOUNDARY(block_idx);
            }

            if (block_idx+1 < end_block_idx) {
                LAUNCH_WG0_SUBROUTINE(false, true);
            } else if (block_idx < end_block_idx) {
                LAUNCH_WG0_SUBROUTINE(true, false);
            }

        } else {
            // Warpgroup 1
            Tensor rP1 = make_tensor<float>((typename T::rP0Layout){});

            if (start_block_idx+1 < end_block_idx) {
                warpgroup_cooperative_qkt_gemm<T, 1>(sQ, sK1, rP1, rQ8, barriers_K1, cur_phase_K1, idx_in_warpgroup);
                cute::warpgroup_wait<0>();
            }

            #define LAUNCH_WG1_SUBROUTINE(IS_BLK0_LAST, IS_BLK1_LAST, IS_BLK2_LAST) \
                wg1_subroutine<T, IS_BLK0_LAST, IS_BLK1_LAST, IS_BLK2_LAST>( \
                    tma_gK, sQ, sK0, sK1, sP0, sP1, sM, sScale0, sScale1, \
                    rQ8, rP1, rO, rL, rRightBorderForQSeq, \
                    barriers_K0, barriers_K1, cur_phase_K1, \
                    tma_params, params, \
                    block_table_ptr, seqlen_k, block_idx, end_block_idx, idx_in_warpgroup \
                );

            int block_idx = start_block_idx;
            #pragma unroll 1
            for (; block_idx < end_block_idx-3; block_idx += 2) {
                LAUNCH_WG1_SUBROUTINE(false, false, false);
                if (AT_BOUNDARY(block_idx)) SEGMENT_BOUNDARY(block_idx);
            }

            if (block_idx+2 < end_block_idx) {
                LAUNCH_WG1_SUBROUTINE(false, false, true);
                if (AT_BOUNDARY(block_idx)) {
                    // This variant leaves rO1 += rP1b @ sV1R to the next pair's wait; the boundary reads rO1 first
                    cute::warpgroup_wait<1>();
                    SEGMENT_BOUNDARY(block_idx);
                }
                block_idx += 2;
                LAUNCH_WG1_SUBROUTINE(true, false, false);
            } else if (block_idx+1 < end_block_idx) {
                LAUNCH_WG1_SUBROUTINE(false, true, false);
            } else if (block_idx < end_block_idx) {
                LAUNCH_WG1_SUBROUTINE(true, false, false);
            }
        }

        REDUCE_L();
        if (item.k1 - item.k0 > 1) {
            // The run's last segment: fold every pending node; the result is the run's node (a root when the run is
            // the whole tree), published through the leaf state's shared slots
            FOLD_SEGMENT(item.k1 - item.k0, true);
            if (warpgroup_idx == 0 && idx_in_warpgroup%4 == 0) {
                sL_reduction_wksp[get_AorC_row_idx(0, idx_in_warpgroup)] = rL[0];
                sL_reduction_wksp[get_AorC_row_idx(1, idx_in_warpgroup)] = rL[1];
                sM[get_AorC_row_idx(0, idx_in_warpgroup)] = rM[0];
                sM[get_AorC_row_idx(1, idx_in_warpgroup)] = rM[1];
            }
            __syncthreads();
        }
        #undef REDUCE_L
        #undef FOLD_SEGMENT
        #undef SEGMENT_BOUNDARY
        #undef AT_BOUNDARY

        // Root policy (FlashMLA's): L' = 1 when L is 0 or NaN. Used only by root rows.
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < 2; ++i)
            rL[i] = (rL[i] == 0.0f || rL[i] != rL[i]) ? 1.0f : rL[i];

        // Copy Q for the next item
        const int next_pos = item.next;
        if (next_pos >= 0) {
            launch_q_window_copy<T>(tma_params, params, params.items[next_pos].row_base, sQ, barrier_Q);
        } else if (params.pdl) {
            // Allow the next kernel (the combine kernel) to launch
            cudaTriggerProgrammaticLaunchCompletion();
        }

        // Rows published by this item: row with segments [0, n_row) at logical position pos has a non-empty last
        // segment here when pos + 1 > k_last*SEG; it is a root when the item covers its whole tree.
        #define ROW_PUBLISHED(pos) ((pos) >= 0 && (pos) + 1 > k_last*seg_tokens)
        #define ROW_ROOT(pos) (item.k0 == 0 && item.k1 >= (pos) / seg_tokens + 1)
        int rMode[2];
        CUTLASS_PRAGMA_UNROLL
        for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
            int row_idx = get_AorC_row_idx(local_row_idx, idx_in_warpgroup);
            int pos = seqlen_k - q_len + min(first_row + row_idx, num_rows - 1) / h_q;
            bool published = first_row + row_idx < num_rows && ROW_PUBLISHED(pos);
            rMode[local_row_idx] = !published ? 0 : (ROW_ROOT(pos) ? 1 : 2);
        }
        const int row_stride = slot_row_stride(rows_here, nmax);
        const int seg_stride = slot_seg_stride(rows_here);

        // Publish mode (execution only; every mode writes the same bits): an m_block of roots of a request whose rows
        // are all published, when every request has the same rows (decode), goes through one TMA store of a
        // (rows, 512, 1, B) map; a full m_block of published roots through one TMA store (FlashMLA's no-split store);
        // a full m_block of published nodes laid out as a 64-record tile through one bulk store; else row by row.
        const bool all_published = rows_here == T::BLOCK_SIZE_M && ROW_PUBLISHED(pos_first);
        const bool uniform_ok = params.uniform_rows > 0 && !params.publish_rows && num_rows == params.uniform_rows &&
            item.row_base == item.req*params.uniform_rows + first_row &&
            rows_here < T::BLOCK_SIZE_M && ROW_PUBLISHED(pos_first) && ROW_ROOT(pos_last);
        const int publish_mode = uniform_ok ? 3 : (!all_published || params.publish_rows) ? 0 : (ROW_ROOT(pos_last) ? 1 :
            (!ROW_ROOT(pos_first) ? 2 : 0));
        if (publish_mode == 1 || publish_mode == 3) {
            stage_block_tma<T, true>(rO, rL, sO_addr, warpgroup_idx, idx_in_warpgroup);
        } else if (publish_mode == 2) {
            stage_block_partial<T>(rO, sO_addr, warpgroup_idx, idx_in_warpgroup);
            if (threadIdx.x < T::BLOCK_SIZE_M) {
                float* rec = (float*)sO_addr + threadIdx.x*PARTIAL_WIDTH;
                rec[HEAD_DIM_V] = sL_reduction_wksp[threadIdx.x];
                rec[HEAD_DIM_V+1] = sM(threadIdx.x);
                rec[HEAD_DIM_V+2] = 0.0f;
                rec[HEAD_DIM_V+3] = 0.0f;
                cutlass::arch::fence_view_async_shared();
            }
        } else {
            stage_item_rows<T>(rO, rL, rMode, sO_addr, warpgroup_idx, idx_in_warpgroup);
        }
        __syncthreads();

        if (threadIdx.x == 0 && publish_mode == 3) {
            store_block_uniform<T>(tma_params, sO_addr, item.req, item.m_block);
        } else if (threadIdx.x == 0 && publish_mode == 1) {
            store_block_tma<T, true>(tma_params, sO_addr, item.row_base);
        } else if (threadIdx.x == 0 && publish_mode == 2) {
            float* slot = params.partials + ((int64_t)item.pslot_base + (int64_t)item.k0*seg_stride)*PARTIAL_WIDTH;
            SM90_BULK_COPY_S2G::copy(sO_addr, slot, T::BLOCK_SIZE_M*PARTIAL_WIDTH*sizeof(float));
            cute::tma_store_arrive();
        }

        // Publish row i from thread i
        const int i = threadIdx.x;
        const int g = first_row + i;
        if (publish_mode == 2) {
            // published by the bulk store above
        } else if (publish_mode == 3) {
            if (i < rows_here) {
                const float cur_L = sL_reduction_wksp[i];
                const float root_L = (cur_L == 0.0f || cur_L != cur_L) ? 1.0f : cur_L;
                params.lse[(int64_t)item.row_base + i] = logf(root_L) + sM(i) / (float)M_LOG2E;
            }
        } else if (publish_mode != 0 && i < T::BLOCK_SIZE_M) {
            const float cur_L = sL_reduction_wksp[i];
            const int64_t grow = (int64_t)item.row_base + i;
            const float root_L = (cur_L == 0.0f || cur_L != cur_L) ? 1.0f : cur_L;
            params.lse[grow] = logf(root_L) + sM(i) / (float)M_LOG2E;
        } else if (i < T::BLOCK_SIZE_M && g < num_rows) {
            const int pos = seqlen_k - q_len + g / h_q;
            if (ROW_PUBLISHED(pos)) {
                const float cur_L = sL_reduction_wksp[i];
                const int64_t grow = (int64_t)item.row_base + i;
                if (ROW_ROOT(pos)) {
                    const float root_L = (cur_L == 0.0f || cur_L != cur_L) ? 1.0f : cur_L;
                    SM90_BULK_COPY_S2G::copy(sO_addr + i*2080, (char*)params.out + grow*1024, 1024);
                    cute::tma_store_arrive();
                    params.lse[grow] = logf(root_L) + sM(i) / (float)M_LOG2E;
                } else {
                    float* slot = params.partials + ((int64_t)item.pslot_base + (int64_t)i*row_stride + (int64_t)item.k0*seg_stride)*PARTIAL_WIDTH;
                    SM90_BULK_COPY_S2G::copy(sO_addr + i*2080, slot, HEAD_DIM_V*sizeof(float));
                    cute::tma_store_arrive();
                    slot[HEAD_DIM_V] = cur_L;
                    slot[HEAD_DIM_V+1] = sM(i);
                }
            }
        }
        cute::tma_store_wait<0>();
        #undef ROW_PUBLISHED
        #undef ROW_ROOT

        item_pos = next_pos;
        if (item_pos >= 0)
            __syncthreads();
    }
#else
    if (cute::thread0()) {
        CUTE_INVALID_CONTROL_PATH("This kernel only supports sm90");
    }
#endif
}


template<typename InputT>
void run_mla_sm90_leaf_kernel(FwdParams &params, const LeafTensors &tensors) {
    using T = Traits<InputT>;
    using index_t = int64_t;
    auto shape_Q = make_shape((index_t)tensors.total_rows, (index_t)T::HEAD_DIM_K);
    auto tma_Q = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(make_gmem_ptr((InputT*)tensors.q_ptr), make_layout(shape_Q, make_stride((index_t)T::HEAD_DIM_K, _1{}))),
        tile_to_shape(
            GMMA::Layout_K_SW128_Atom<InputT>{},
            Shape<Int<T::BLOCK_SIZE_M>, Int<T::HEAD_DIM_K>>{}
        )
    );
    auto shape_K = make_shape(Int<T::PAGE_BLOCK_SIZE>{}, Int<T::HEAD_DIM_K>{}, (index_t)1, (index_t)tensors.num_blocks);
    auto tma_K = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(
            make_gmem_ptr((InputT*)tensors.k_ptr),
            make_layout(
                shape_K,
                make_stride(tensors.k_row_stride, _1{}, tensors.k_head_stride, tensors.k_batch_stride)
            )
        ),
        tile_to_shape(
            GMMA::Layout_K_SW128_Atom<InputT>{},
            Layout<
                Shape<Int<T::PAGE_BLOCK_SIZE>, Int<64>>,
                Stride<Int<T::HEAD_DIM_K>, _1>
            >{}
        )
    );
    auto shape_O = make_shape((index_t)tensors.total_rows, (index_t)T::HEAD_DIM_V);
    auto tma_O = cute::make_tma_copy(
        SM90_TMA_STORE{},
        make_tensor(make_gmem_ptr((InputT*)tensors.out_ptr), make_layout(shape_O, make_stride((index_t)T::HEAD_DIM_V, _1{}))),
        tile_to_shape(GMMA::Layout_K_SW128_Atom<InputT>{}, Shape<Int<T::BLOCK_SIZE_M>, Int<T::HEAD_DIM_V>>{})
    );
    const index_t urows = tensors.uniform_rows > 0 ? tensors.uniform_rows : 1;
    auto shape_Ou = make_shape((index_t)urows, (index_t)T::HEAD_DIM_V, (index_t)1, (index_t)std::max<index_t>(tensors.total_rows / urows, 1));
    auto tma_Ou = cute::make_tma_copy(
        SM90_TMA_STORE{},
        make_tensor(make_gmem_ptr((InputT*)tensors.out_ptr),
                    make_layout(shape_Ou, make_stride((index_t)T::HEAD_DIM_V, _1{}, urows*T::HEAD_DIM_V, urows*T::HEAD_DIM_V))),
        tile_to_shape(GMMA::Layout_K_SW128_Atom<InputT>{}, Shape<Int<T::BLOCK_SIZE_M>, Int<T::HEAD_DIM_V>>{})
    );
    TmaParams<decltype(shape_Q), decltype(tma_Q), decltype(shape_K), decltype(tma_K),
              decltype(shape_O), decltype(tma_O), decltype(shape_Ou), decltype(tma_Ou)> tma_params = {
        shape_Q, tma_Q,
        shape_K, tma_K,
        shape_O, tma_O,
        shape_Ou, tma_Ou,
    };
    auto kernel = &mla_sm90_leaf_kernel<T, decltype(tma_params)>;
    constexpr size_t smem_size = sizeof(typename T::SharedMemoryPlan);
    CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    cudaLaunchAttribute attributes[1];
    attributes[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attributes[0].val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t config = {
        dim3(params.workers, 1, 1),
        dim3(T::NUM_THREADS, 1, 1),
        smem_size,
        params.stream,
        params.pdl ? attributes : nullptr,
        params.pdl ? 1 : 0
    };
    cudaLaunchKernelEx(&config, kernel, params, tma_params);
    CHECK_CUDA_KERNEL_LAUNCH();
}

}
