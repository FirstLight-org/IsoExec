// SPDX-License-Identifier: MIT
#pragma once

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/arch/barrier.h>
#include <cute/arch/tmem_allocator_sm100.hpp>

#include "config.h"

namespace mla_sm100::fwd {

using namespace cute;
using TMABarrier = cutlass::arch::ClusterTransactionBarrier;

//: One TMEM data path, in 32-bit (float) addressing.  A TMEM address is (lane << 16) | column.
static constexpr uint32_t TMEM_DP = 1u << 16;

template<typename InputT_>
struct Traits {
    using InputT = InputT_;
    static_assert(std::is_same_v<InputT, cutlass::bfloat16_t> || std::is_same_v<InputT, cutlass::half_t>);

    static constexpr int BLOCK_SIZE_M = Config::BLOCK_SIZE_M;       // 64
    static constexpr int PAGE_BLOCK_SIZE = Config::PAGE_BLOCK_SIZE; // 64
    static constexpr int HEAD_DIM_K = Config::HEAD_DIM_K;           // 576
    static constexpr int HEAD_DIM_V = Config::HEAD_DIM_V;           // 512
    static constexpr int NUM_THREADS = Config::NUM_THREADS;
    static constexpr int KV_STAGES = Config::KV_STAGES;
    static constexpr int PV_N = Config::PV_N;                       // 256
    static constexpr int QK_CHUNK = Config::QK_CHUNK;               // 64
    static constexpr int QK_CHUNKS = HEAD_DIM_K / QK_CHUNK;         // 9 = 8 latent + 1 rope
    static constexpr int PV_HALVES = HEAD_DIM_V / PV_N;             // 2

    // Warp roles (the compute warps own disjoint 16-row groups of the row block, which is what the
    // M=64 TMEM accumulator layout dictates: row m lives on data path (m%16) + 32*(m/16)).
    static constexpr int NUM_COMPUTE_WARPS = 4;
    static constexpr int COMPUTE_THREADS = NUM_COMPUTE_WARPS * 32;  // 128
    static constexpr int WARP_MMA = 4;
    static constexpr int WARP_LOAD = 5;
    static constexpr int ROWS_PER_WARP = BLOCK_SIZE_M / NUM_COMPUTE_WARPS;   // 16

    // ---- SMEM tiles -------------------------------------------------------------------------
    using SmemLayoutKV = decltype(tile_to_shape(
        UMMA::Layout_K_SW128_Atom<InputT>{}, Shape<Int<PAGE_BLOCK_SIZE>, Int<HEAD_DIM_K>>{}));
    //: The same bytes, transposed: V(d, t) == KV(t, d).  MN-major B operand of the PV GEMM.
    using SmemLayoutV = decltype(composition(
        SmemLayoutKV{}, make_layout(Shape<Int<HEAD_DIM_V>, Int<PAGE_BLOCK_SIZE>>{}, GenRowMajor{})));
    //: The bf16 probabilities, A operand of the PV GEMM (one buffer; see leaf.cuh's issue order).
    using SmemLayoutP = decltype(tile_to_shape(
        UMMA::Layout_K_SW128_Atom<InputT>{}, Shape<Int<BLOCK_SIZE_M>, Int<PAGE_BLOCK_SIZE>>{}));

    // ---- UMMA atoms -------------------------------------------------------------------------
    //: S[64 rows, 64 keys] += Q[64, k] . K[64, k]^T, both operands K-major, 36 k-steps of 16.
    using MmaQK = decltype(make_tiled_mma(
        SM100_MMA_F16BF16_SS<InputT, InputT, float, BLOCK_SIZE_M, PAGE_BLOCK_SIZE,
                             UMMA::Major::K, UMMA::Major::K>{}));
    //: O[64 rows, 256 dims] += P[64, 64] . V[256, 64]^T, A K-major, B MN-major, 4 k-steps of 16.
    using MmaPV = decltype(make_tiled_mma(
        SM100_MMA_F16BF16_SS<InputT, InputT, float, BLOCK_SIZE_M, PV_N,
                             UMMA::Major::K, UMMA::Major::MN>{}));

    static constexpr uint32_t TMEM_COLS = 512;          // tcgen05.alloc needs a power of two
    // ROUND 5 TMEM map (Q in TMEM).  The A operand of an M = 64 TS MMA must live on the DP+0 half
    // (an A address at DP+16 faults with a misaligned-address error -- measured), and Q needs 288
    // columns of that half.  So the DP+0 half holds the two score buffers and Q, and BOTH halves of
    // the 512-wide O accumulator move to the DP+16 half, where a D accumulator already lived:
    //   DP+0 : S buffer 0 cols [0, 64), S buffer 1 cols [64, 128), Q cols [128, 416), 96 free
    //   DP+16: O[:, 0:256] cols [0, 256), O[:, 256:512] cols [256, 512)
    // An address change only: every accumulator is still one M = 64 fp32 accumulator of its own.
    static constexpr uint32_t TMEM_HALF_DP = 16;
    static constexpr uint32_t TMEM_S = 0;
    static constexpr uint32_t TMEM_Q = Config::S_BUFFERS * PAGE_BLOCK_SIZE;   // 128
    static_assert(Config::S_BUFFERS == 2, "TMEM_Q sits right after the score buffers");
    static constexpr int Q_COLS = HEAD_DIM_K / 2;              // 288: two bf16 per 32-bit column
    static_assert(TMEM_Q + Q_COLS <= TMEM_COLS);
    static_assert(2 * PV_N <= TMEM_COLS);

    //: TMEM address of O half `h` (0: dims 0..255, 1: dims 256..511) for compute warp `w`.
    __forceinline__ __device__ static uint32_t tmem_o(uint32_t base, int h, int w) {
        return base + (uint32_t)(h * PV_N) + (uint32_t)(TMEM_HALF_DP + w * 32) * TMEM_DP;
    }
    //: TMEM address of the whole O half `h` (row 0), as the UMMA accumulator address.
    __forceinline__ __device__ static uint32_t tmem_o_acc(uint32_t base, int h) {
        return base + (uint32_t)(h * PV_N) + (uint32_t)TMEM_HALF_DP * TMEM_DP;
    }
    //: TMEM address of score buffer `b` for compute warp `w`.
    __forceinline__ __device__ static uint32_t tmem_s(uint32_t base, int b, int w) {
        return base + TMEM_S + (uint32_t)(b * PAGE_BLOCK_SIZE) + (uint32_t)(w * 32) * TMEM_DP;
    }
    __forceinline__ __device__ static uint32_t tmem_s_acc(uint32_t base, int b) {
        return base + TMEM_S + (uint32_t)(b * PAGE_BLOCK_SIZE);
    }
    //: TMEM address of Q chunk `c` (dims 64c..64c+63), row 0, as the MMA's A operand.
    __forceinline__ __device__ static uint32_t tmem_q_acc(uint32_t base, int c) {
        return base + TMEM_Q + (uint32_t)(32 * c);
    }
    //: The same chunk, compute warp `w`'s 16 rows (its own subpartition).
    __forceinline__ __device__ static uint32_t tmem_q(uint32_t base, int c, int w) {
        return tmem_q_acc(base, c) + (uint32_t)(w * 32) * TMEM_DP;
    }

    // ---- TMEM access atoms ------------------------------------------------------------------
    // 16 data paths (one 16-row group of the M=64 layout) x 64 columns, 32 fp32 per lane.
    using T2R = SM100_TMEM_LOAD_16dp256b8x;
    using R2T = decltype(TMEM::tmem_load_to_store(T2R{}));
    using TmemTile = Layout<Shape<Int<ROWS_PER_WARP>, Int<64>>, Stride<Int<TMEM_DP>, _1>>;
    static constexpr int T2R_VALS = 32;
    // ROUND 5: a 16-row x 32-column (uint32) access, exactly one Q chunk (four K16 steps) of a warp.
    using T2RQ = SM100_TMEM_LOAD_16dp256b4x;
    using R2TQ = decltype(TMEM::tmem_load_to_store(T2RQ{}));
    using TmemTileQ = Layout<Shape<Int<ROWS_PER_WARP>, Int<32>>, Stride<Int<TMEM_DP>, _1>>;

    // ---- shared storage ---------------------------------------------------------------------
    struct SharedStorage {
        // ROUND 5: no Q tile.  Q lives in TMEM (see TMEM_Q) and its 72 KB is the third KV stage.
        cute::array_aligned<InputT, KV_STAGES * cosize_v<SmemLayoutKV>, 1024> smem_kv;
        cute::array_aligned<InputT, cosize_v<SmemLayoutP>, 1024> smem_p;
        TMABarrier bar_qg[3];                  // ROUND 5: Q chunks 0-2, 3-5, 6-8 in TMEM
        //: TMA arrival of ONE 8 KB head-dim chunk of a KV tile.  One barrier per
        //: (ring slot, chunk) rather than one per ring slot: the QK run only needs chunk c before
        //: its four k-steps on chunk c, and waiting for all nine puts eight chunk transfers on the
        //: critical path of every tile.  The "one mbarrier per (role, ring slot)" rule is kept and
        //: refined -- the slot is now (stage, chunk) -- and re-arming chunk c for tile g+2 still
        //: cannot pass the wait for tile g, because the load warp waits bar_pv[stage] first.
        TMABarrier bar_kv[KV_STAGES][QK_CHUNKS];
        TMABarrier bar_qk[Config::S_BUFFERS];  // tcgen05.commit after a QK run
        TMABarrier bar_pv[KV_STAGES];          // tcgen05.commit after a PV run (frees KV + P)
        TMABarrier bar_p;                      // compute warps: P stored and O corrected
        uint32_t tmem_ptr;
        uint32_t pad;
    };
    static constexpr int KV_TILE_BYTES = PAGE_BLOCK_SIZE * HEAD_DIM_K * (int)sizeof(InputT);
    static constexpr int KV_CHUNK_BYTES = PAGE_BLOCK_SIZE * QK_CHUNK * (int)sizeof(InputT);
};

// KV tiles are addressed by physical page id into a (64, 576, 1, num_blocks) map, as on sm90.  (Q had
// its own tensor map until round 5; since then the compute warps load it into TMEM themselves.)
template<typename ShapeK, typename TMA_K>
struct TmaParams {
    ShapeK shape_K;
    TMA_K tma_K;
};

}  // namespace mla_sm100::fwd
