// SPDX-License-Identifier: MIT
// IsoExec dsa_sm100: the sparse-attention leaf on Blackwell (recipe sm100_dsa_sparse_segtree_v1).
//
// Numerical program (README.md section 4; mla_sm100's leaf / merge / root, re-stated):
//   * one CTA = one query row t x 64 heads; the list idx[t, :] in ASCENDING list position, 64-slot
//     tiles; G groups of `seg` positions (G = 1: the unsplit chain).
//   * conversion point: each gathered token's 512 e4m3 values become bf16 in SMEM before any MMA:
//     cvt.rn.bf16x2.e4m3x2 (exact; one native F2FP on sm_100a), times the ue8m0 scale as bf16
//     (mul.rn.bf16x2: exact for every in-range finite value).  A pad slot (idx < 0, or past topk_length) is gathered from row
//     -1, which TMA zero-fills, so its K and V rows are exact zeros.
//   * QK: tcgen05.mma.cta_group::1.kind::f16, A = Q from TMEM, B = the bf16 tile, M = 64 heads, N = 64
//     slots, 32 k-steps of 16 over c = 0..511 ascending, first ScaleOut::Zero, fp32 TMEM accumulator.
//   * mask: a pad slot's score becomes MAX_INIT_VAL = -1e33 before the max.
//   * softmax per tile (mla_sm100 verbatim): new_max = max(sM, cur_max*scale_log2); scale_for_old =
//     exp2f(sM - new_max); P = exp2f(s*scale_log2 - new_max) -> bf16; the tile's full row sum as ascending
//     adjacent pairs per lane then shfl_xor 1, 2; L = L*scale_for_old + sum; O *= scale_for_old (skipped
//     only when exactly 1.0f for the warp's rows); PV with ScaleOut::Zero on a group's first tile.
//   * root: L' = (L == 0 || L != L) ? 1 : L; out = bf16(O / L'); lse = logf(L') + M / log2(e).
//
// Blackwell schedule (free): 12 warps.  Warps 0-3 softmax / correction / epilogue (warp w owns rows
// [16w, 16w+16)); warp 4 issues every UMMA; warps 5-7 translate the whole list through the block table
// into an SMEM row table (and check it: ascending, pads trailing, block-table entries >= 0 -- a refused
// list poisons its row with NaN), then warp 5 issues the TMA gather4 of the raw 528-byte token rows;
// warps 8-11 dequantise each stage IN PLACE (raw rows land at the stage's start; every raw word is in
// registers before the per-tile bar.sync 2).  Three 64 KB KV stages, P in SMEM, Q in TMEM.
// TMEM (cute's M = 64 fragments, printed on the host: an M = 64 A operand from TMEM occupies the
// HALF-0 data paths only, k-step k at column 8k -- a k-step at DP+16 faults as a misaligned address):
//   DP+16 half:  O[:, 0:256] cols [0, 256), O[:, 256:512] cols [256, 512)
//   DP+0  half:  Q cols [0, 256) (k-step k at col 8k), S buffer b cols [256 + 64b, 320 + 64b)
#pragma once

#include <math_constants.h>
#include <cute/tensor.hpp>
#include <cute/arch/copy_sm100.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/numeric_types.h>

#include "params.h"

namespace dsa_sm100::attn {

using namespace cute;
using Barrier = cutlass::arch::ClusterTransactionBarrier;
using bf16 = cutlass::bfloat16_t;

static constexpr float MAX_INIT_VAL_SM = -1e30f;
static constexpr float MAX_INIT_VAL = -1e33f;
static constexpr float RESCALE_THRESHOLD_V2 = 8.0f;   // recipe v2 only (log2 units)
static constexpr uint32_t TMEM_DP = 1u << 16;

template<int DQK_, bool BF16_ = false, int MAXK_ = 2048>
struct TLeaf {
    static constexpr int H = 64;                  // heads per CTA (UMMA M)
    static constexpr int TILE = 64;               // TILE_SLOTS (recipe)
    static constexpr int D = 512;                 // D_V (the output's width; V = the first 512 stored dims)
    static constexpr int DQK = DQK_;              // D_QK: 512 (V4.1, fp8 cache or bf16 kv) or 576 (GLM-5 / V3.2 training geometry, bf16 kv only)
    static constexpr int QCHUNKS = DQK / 64;      // 64-dim chunks of the QK chain (8 or 9)
    static constexpr bool BF16 = BF16_;           // the bf16-kv instantiation (no raw stages, no dequant)
    static constexpr bool HAS_RAW = !BF16;        // the fp8 raw stages exist only for the fp8 cache
    static constexpr int ROW_BYTES = 528;         // V41 fp8 token
    static constexpr int RAW_STRIDE = 576;        // SMEM row stride of a gathered raw token (144 words)
    // DSA_RAW_SEP 0: three bf16 stages, the raw gather lands IN its stage (dequant in place);
    //             1: two bf16 stages + two raw stages (the gather runs ahead of PV; raw freed after the dequant's loads)
    // bf16 kv: three 64 KB stages at 512 (the gather writes the stage itself; measured 2.76 -> 2.04 ms at
    // T = 4K vs two), two at 576 (3 x 72 KB does not fit)
    static constexpr int KV_STAGES = BF16_ ? (DQK_ == 512 ? 3 : 2) : (1 ? 2 : 3);
    static constexpr int RAW_STAGES = 2;
    // the SMEM translation table's capacity (list positions). 2048 for every dsa_sm100 recipe; 2112
    // (33 tiles) for glm_kpool_sm100's absorbed GLM recipe (2051-slot lists). Execution only.
    static constexpr int MAX_K = MAXK_;
    static constexpr bool SM_SPLIT = 1 && BF16_ && DQK_ == 512;   // fp8 decode +1%, bf16-512 prefill -2.5%, bf16-576 +5% (README "Round 3")
    static constexpr int NUM_DQ_WARPS = SM_SPLIT ? 4 : 8;   // dequant warps (4 or 8; README "Step 4" has both)
    // round 3: the softmax split by columns over two warps per TMEM lane quadrant (warps w and
    // WARP_SMB0 + w), the per-row fold order preserved exactly (README "Round 3")
    static constexpr int NUM_WARPS = 8 + NUM_DQ_WARPS + (SM_SPLIT ? 4 : 0);
    static constexpr int WARP_SMB0 = 8 + NUM_DQ_WARPS;   // the second softmax half (w % 4 = its quadrant)
    static constexpr int SCOLS = SM_SPLIT ? 32 : 64;     // S columns per softmax warp
    static constexpr int CHUNKS = 8;                // 64-dim chunks of a tile
    static constexpr int CGROUPS = 2;               // dequant arrivals per tile (chunks 0-3, 4-7)
    static constexpr int NUM_THREADS = NUM_WARPS * 32;
    static constexpr int WARP_MMA = 4, WARP_GATHER = 5, WARP_DQ0 = 8;
    static constexpr int PV_N = 256;
    static constexpr uint32_t TMEM_COLS = 512;
    static constexpr uint32_t TMEM_O = 0, TMEM_S = DQK / 2, TMEM_Q = 0;   // Q: DQK/2 columns of the DP+0 half, S after it
    static constexpr uint32_t O_DP = 16;              // O lives on the DP+16 half

    using SmemLayoutKV = decltype(tile_to_shape(UMMA::Layout_K_SW128_Atom<bf16>{}, Shape<Int<TILE>, Int<DQK>>{}));
    // V = the first 512 columns: chunk c (64 dims) sits at c x 8 KB in either width, so the 512-wide view
    // of the same base is V
    using SmemLayoutKV512 = decltype(tile_to_shape(UMMA::Layout_K_SW128_Atom<bf16>{}, Shape<Int<TILE>, Int<D>>{}));
    using SmemLayoutV = decltype(composition(SmemLayoutKV512{}, make_layout(Shape<Int<D>, Int<TILE>>{}, GenRowMajor{})));
    using SmemLayoutP = decltype(tile_to_shape(UMMA::Layout_K_SW128_Atom<bf16>{}, Shape<Int<H>, Int<TILE>>{}));
    using MmaQK = decltype(make_tiled_mma(SM100_MMA_F16BF16_SS<bf16, bf16, float, H, TILE, UMMA::Major::K, UMMA::Major::K>{}));
    using MmaPV = decltype(make_tiled_mma(SM100_MMA_F16BF16_SS<bf16, bf16, float, H, PV_N, UMMA::Major::K, UMMA::Major::MN>{}));
    using T2R = SM100_TMEM_LOAD_16dp256b8x;
    using T2R_S = std::conditional_t<SM_SPLIT, SM100_TMEM_LOAD_16dp256b4x, SM100_TMEM_LOAD_16dp256b8x>;
    using TmemTileS = Layout<Shape<_16, Int<SCOLS>>, Stride<Int<TMEM_DP>, _1>>;
    using R2T = decltype(TMEM::tmem_load_to_store(T2R{}));
    using TmemTile = Layout<Shape<_16, _64>, Stride<Int<TMEM_DP>, _1>>;

    struct SharedStorage {
        cute::array_aligned<bf16, KV_STAGES * cosize_v<SmemLayoutKV>, 1024> kv;   // 3 x 64 KB; raw rows land at its start
        cute::array_aligned<bf16, cosize_v<SmemLayoutP>, 1024> p;                  // 8 KB
        int rows[MAX_K];                     // physical token row of every list position (-1: pad)
        uint64_t vmask[MAX_K / TILE];        // bit j of word t: list position 64 t + j is a real key
        Barrier bar_q;                       // Q written to TMEM (compute warps)
        Barrier raw_full[3];
        Barrier raw_empty[RAW_STAGES];       // the dequant warps hold the raw words (one arrive per warp)
        cute::array_aligned<uint8_t, HAS_RAW ? RAW_STAGES * TILE * RAW_STRIDE : 128, 128> raw;
        Barrier kv_full[KV_STAGES][CGROUPS]; // chunk group of the stage dequantised (one arrive per dequant warp)
        Barrier bar_qk[2];                   // tcgen05.commit after a QK run
        Barrier bar_pv[KV_STAGES];           // tcgen05.commit after a PV run (frees the KV stage and P)
        Barrier bar_p;                       // compute warps: P stored and O corrected
        Barrier table_ready;                 // the translation table is complete (warp 5, after warps 5-7)
        Barrier tile_ready[MAX_K / TILE];    // DSA_TILE_READY: tile t's rows / validity word are published
        Barrier check_done;                  // DSA_TILE_READY: the cross-tile check has run (ss.bad final)
        float xmax[4][2][16];                // SM_SPLIT: per quadrant, per half, the tile's row maxima
        float xsum[4][32][2];                // SM_SPLIT: half 0's per-lane partial row sums, handed to half 1
        float xL[4][16];                     // SM_SPLIT: half 1's final L for half 0's epilogue
        int first_pos[MAX_K / TILE + 1], last_pos[MAX_K / TILE + 1], tail_valid[MAX_K / TILE + 1];   // per tile: first / last valid logical position
        int bad;                             // the list is not ascending with trailing pads (SPEC 4.1)
        uint32_t tmem_ptr;
    };
};
using T64 = TLeaf<512>;
using T64B = TLeaf<512, true>;
using T64B_K2112 = TLeaf<512, true, 2112>;   // glm_kpool_sm100: bf16 512-latent, lists up to 2112 slots
using T64_K2112 = TLeaf<512, false, 2112>;   // glm_kpool_sm100: fp8 V4.1 latent cache, lists up to 2112 slots
using T576 = TLeaf<576, true>;

__forceinline__ __device__ void tcgen05_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
__forceinline__ __device__ void tcgen05_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }

//: Four K16 steps of the QK run on one 64-dim chunk c, A (Q) from TMEM: k-step k = 4c + i lives at
//: column 8k = 32c + 8i of the DP+0 half.  B advances +2 (32 bytes) per step inside the SW128 chunk.
__forceinline__ __device__ void umma_qk_chunk_ts(uint32_t tmem_s, uint32_t tq, uint64_t desc_k,
                                                 uint32_t idesc, uint32_t accumulate) {
    asm volatile(
        "{\n\t.reg .b64 db;\n\t.reg .b32 ta;\n\t.reg .pred p,q,lead;\n\t"
        "elect.sync _|lead, -1;\n\t"
        "setp.ne.b32 p,%4,0;\n\t"
        "setp.ne.b32 q,1,0;\n\t"
        "mov.b64 db,%2;\n\t"
        "mov.b32 ta,%1;\n\t"
        "@lead tcgen05.mma.cta_group::1.kind::f16 [%0],[ta],db,%3,p;\n\t"
        "add.u64 db,db,2;\n\tadd.u32 ta,%1,8;\n\t"
        "@lead tcgen05.mma.cta_group::1.kind::f16 [%0],[ta],db,%3,q;\n\t"
        "add.u64 db,db,2;\n\tadd.u32 ta,%1,16;\n\t"
        "@lead tcgen05.mma.cta_group::1.kind::f16 [%0],[ta],db,%3,q;\n\t"
        "add.u64 db,db,2;\n\tadd.u32 ta,%1,24;\n\t"
        "@lead tcgen05.mma.cta_group::1.kind::f16 [%0],[ta],db,%3,q;\n\t"
        "}\n"
        :
        : "r"(tmem_s), "r"(tq), "l"(desc_k), "r"(idesc), "r"(accumulate)
        : "memory");
}

//: One PV half: four K16 steps of P.V into one 256-wide O accumulator (mla_sm100's chain).
__forceinline__ __device__ void umma_pv_half(uint32_t tmem_o, uint64_t desc_p, uint64_t desc_v,
                                             uint32_t idesc, uint32_t accumulate) {
    asm volatile(
        "{\n\t.reg .b64 da,db;\n\t.reg .pred p,q,lead;\n\t"
        "elect.sync _|lead, -1;\n\t"
        "setp.ne.b32 p,%4,0;\n\t"
        "setp.ne.b32 q,1,0;\n\t"
        "mov.b64 da,%1;\n\tmov.b64 db,%2;\n\t"
        "@lead tcgen05.mma.cta_group::1.kind::f16 [%0],da,db,%3,p;\n\t"
        "add.u64 da,da,2;\n\tadd.u64 db,db,128;\n\t"
        "@lead tcgen05.mma.cta_group::1.kind::f16 [%0],da,db,%3,q;\n\t"
        "add.u64 da,da,2;\n\tadd.u64 db,db,128;\n\t"
        "@lead tcgen05.mma.cta_group::1.kind::f16 [%0],da,db,%3,q;\n\t"
        "add.u64 da,da,2;\n\tadd.u64 db,db,128;\n\t"
        "@lead tcgen05.mma.cta_group::1.kind::f16 [%0],da,db,%3,q;\n\t"
        "}\n"
        :
        : "r"(tmem_o), "l"(desc_p), "l"(desc_v), "r"(idesc), "r"(accumulate)
        : "memory");
}

__forceinline__ __device__ void tma_gather4(const void *desc, Barrier &bar, void *smem, int col, int4 rows) {
    const uint32_t s = cute::cast_smem_ptr_to_uint(smem);
    const uint32_t b = cute::cast_smem_ptr_to_uint(&bar);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4, %5, %6}], [%7];\n"
        :: "r"(s), "l"(desc), "r"(col), "r"(rows.x), "r"(rows.y), "r"(rows.z), "r"(rows.w), "r"(b)
        : "memory");
}

// L2 prefetch of four token rows (no SMEM, no barrier): the gather of a later tile then hits L2.
__forceinline__ __device__ void tma_prefetch_gather4(const void *desc, int col, int4 rows) {
    asm volatile(
        "cp.async.bulk.prefetch.tensor.2d.L2.global.tile::gather4 [%0, {%1, %2, %3, %4, %5}];\n"
        :: "l"(desc), "r"(col), "r"(rows.x), "r"(rows.y), "r"(rows.z), "r"(rows.w) : "memory");
}

//: 2 x e4m3 (low byte = element 0) -> 2 x bf16 (exact: cvt.rn.bf16x2.e4m3x2 is ONE native F2FP on
//: sm_100a), times the ue8m0 scale `s2` (bf16x2 (2^e, 2^e)) with mul.rn.bf16x2 (exact in range).
//: The same values as e4m3 -> f16 -> f32 -> bf16 (FlashMLA's sequence, 5 instructions a pair; measured
//: 3.75K busy dequant cycles a tile with it).
__forceinline__ __device__ uint32_t dq2(uint16_t x, uint32_t s2) {
    uint32_t out;
    asm("{\n\t.reg .b32 t;\n\t"
        "cvt.rn.bf16x2.e4m3x2 t, %1;\n\t"
        "mul.rn.bf16x2 %0, t, %2;\n\t"
        "}\n"
        : "=r"(out) : "h"(x), "r"(s2));
    return out;
}
__forceinline__ __device__ uint32_t ue8m0_bf16x2(uint32_t e) {
    // bf16 2^(e-127) has the bit pattern e << 7 (e = 0 -> +0, the zero-filled pad row's scale)
    const uint32_t b = (e & 0xFFu) << 7;
    return b | (b << 16);
}

constexpr int kQfillUnroll = 1;


template<typename T>
__forceinline__ __device__ auto tmem_tile_s(uint32_t addr) {
    return make_tensor(make_tmem_ptr<float>(addr), typename T::TmemTileS{});
}
template<typename T>
__forceinline__ __device__ auto tmem_tile(uint32_t addr) {
    return make_tensor(make_tmem_ptr<float>(addr), typename T::TmemTile{});
}

//: bf16 kv: issuer k of NI issues the gather4s (g, c) of tile t with (8 g + c) % NI == k -- chunk c
//: (64 dims = 128 B) of rows 4g..4g+3 into the SW128 K-major tile (the TMA swizzle is the UMMA atom's).
template<typename T, typename SS>
__forceinline__ __device__ void bf16_issue_tile(SS &ss, const CUtensorMap *tm, int t, int k, int NI) {
    const int s = t % T::KV_STAGES;
    uint8_t *dst = reinterpret_cast<uint8_t *>(ss.kv.data() + s * (int)cosize_v<typename T::SmemLayoutKV>);
    #pragma unroll 1
    for (int i = k; i < (T::TILE / 4) * T::QCHUNKS; i += NI) {
        const int g = i / T::QCHUNKS, c = i % T::QCHUNKS;
        const int4 rr = *reinterpret_cast<const int4 *>(&ss.rows[t * T::TILE + 4 * g]);
        tma_gather4(tm, ss.raw_full[s], dst + c * (T::TILE * 128) + (4 * g) * 128, c * 64, rr);
    }
}

template<typename T, bool BF16KV, bool kLazy = false>
__global__ void __launch_bounds__(T::NUM_THREADS, 1)
dsa_sparse_leaf_kernel(const __grid_constant__ LeafParams p, const __grid_constant__ CUtensorMap tm_raw) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    extern __shared__ __align__(1024) char smem_raw[];
    // align by POINTER arithmetic on the shared array: an integer round trip (uintptr_t & ~1023) loses
    // the address space and turns every SMEM access of the kernel into a generic LD.E / ST.E (ncu:
    // the dequant's stores were ST.E.128 with a MEMBAR.ALL.CTA per fence)
    const uint32_t mis = static_cast<uint32_t>(__cvta_generic_to_shared(smem_raw)) & 1023u;
    typename T::SharedStorage &ss = *reinterpret_cast<typename T::SharedStorage *>(smem_raw + ((1024u - mis) & 1023u));
    constexpr int TILE = T::TILE;

    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int tok = blockIdx.x;
    const int head0 = blockIdx.y * T::H;
    const int K = p.topk;
    const int klen = p.topk_length ? min(__ldg(p.topk_length + tok), K) : K;
    // this CTA's group (SPEC 4.4): list positions [grp * seg, (grp + 1) * seg); G == 1 is the whole list
    const int grp = blockIdx.z;
    const int seg_tiles = p.seg / TILE;
    const int num_tiles = seg_tiles;
    const int pos0 = grp * p.seg;

    if (tid == 0) {
        ss.bad = 0;
        ss.bar_q.init(4);
        for (int i = 0; i < 3; ++i) ss.raw_full[i].init(1);
        for (int i = 0; i < T::RAW_STAGES; ++i) ss.raw_empty[i].init(T::NUM_DQ_WARPS);
        for (int i = 0; i < T::KV_STAGES; ++i) {
            for (int c = 0; c < T::CGROUPS; ++c) ss.kv_full[i][c].init(T::NUM_DQ_WARPS);
            ss.bar_pv[i].init(1);
        }
        for (int i = 0; i < 2; ++i) ss.bar_qk[i].init(1);
        ss.bar_p.init(T::SM_SPLIT ? 8 : 4);
        ss.table_ready.init(1);
        for (int i = 0; i < T::MAX_K / T::TILE; ++i) ss.tile_ready[i].init(1);
        ss.check_done.init(1);
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();
    cute::TMEM::Allocator1Sm tmem_allocator;
    if (warp == T::WARP_MMA) {
        tmem_allocator.allocate(T::TMEM_COLS, &ss.tmem_ptr);
        __syncwarp();
    }
    tcgen05_fence_before();
    __syncthreads();
    tcgen05_fence_after();
    const uint32_t tmem_base = ss.tmem_ptr;

    // ===== the list -> physical token rows: a translation pass over tiles tw, tw + NTW, ... =====
    // The translation is two DEPENDENT global loads per slot (list entry, then block-table entry);
    // every load of a translating warp is in flight at once.  A pad slot (idx < 0 or past topk_length)
    // maps to row -1: TMA zero-fills it, so its K and V rows are exact zeros.
    auto translate = [&](const int tw, auto ntw_c, auto tile_ready_c) {
        constexpr int NTW = decltype(ntw_c)::value;
        constexpr bool TILE_READY = decltype(tile_ready_c)::value;
        // ============ the list -> physical token rows (warps 5-7), then TMA gather4 (warp 5) ============
        // The translation is two DEPENDENT global loads per slot (list entry, then block-table entry);
        // prefetched one tile ahead they put both latencies on every tile (measured: 6.3K cycles of a
        // 10K-cycle tile).  So warps 5-7 translate the whole list up front, every load of a warp in
        // flight at once, into an SMEM table; warp 5 then streams the gathers from it.
        // A pad slot (idx < 0 or past topk_length) maps to row -1: TMA zero-fills it, so its K and V rows
        // are exact zeros.
        const int bt_row = p.bt_row ? __ldg(p.bt_row + tok) : tok;
        const int *bt = p.block_table + (int64_t)bt_row * p.bt_stride;
        const int *lst = p.idx + (int64_t)tok * p.idx_stride + pos0;
        // translator tw of NTW handles tiles tw, tw + NTW, ...
        constexpr int PER = (T::MAX_K / TILE + NTW - 1) / NTW;   // tiles per translating warp
        int2 ix[PER];
        #pragma unroll
        for (int i = 0; i < PER; ++i) {
            const int t = tw + NTW * i;
            const int pos = t * TILE + 2 * lane;
            ix[i] = (t < num_tiles) ? __ldg(reinterpret_cast<const int2 *>(lst + pos)) : make_int2(-1, -1);
        }
        int2 pg[PER];
        #pragma unroll
        for (int i = 0; i < PER; ++i) {
            const int pos = (tw + NTW * i) * TILE + 2 * lane;
            if constexpr (BF16KV) {
                if (p.block_table) {
                    // paged bf16 kv (execution mode, glm_kpool_sm100 engine): logical positions through the block
                    // table into 64-row pages, exactly as the fp8 cache; a page past the kv is refused below
                    const int bx = (ix[i].x >= 0 && pos0 + pos < klen) ? __ldg(bt + (ix[i].x >> 6)) : -1;
                    const int by = (ix[i].y >= 0 && pos0 + pos + 1 < klen) ? __ldg(bt + (ix[i].y >> 6)) : -1;
                    pg[i].x = (bx >= 0 && (int64_t)bx * 64 + 64 <= p.kv_rows) ? bx : -1;
                    pg[i].y = (by >= 0 && (int64_t)by * 64 + 64 <= p.kv_rows) ? by : -1;
                } else {
                    // the bf16 kv: a list entry is its row; an entry past the kv is refused below
                    pg[i].x = (ix[i].x >= 0 && pos0 + pos < klen) ? (ix[i].x < p.kv_rows ? 0 : -1) : -1;
                    pg[i].y = (ix[i].y >= 0 && pos0 + pos + 1 < klen) ? (ix[i].y < p.kv_rows ? 0 : -1) : -1;
                }
            } else {
                pg[i].x = (ix[i].x >= 0 && pos0 + pos < klen) ? __ldg(bt + (ix[i].x >> 6)) : -1;
                pg[i].y = (ix[i].y >= 0 && pos0 + pos + 1 < klen) ? __ldg(bt + (ix[i].y >> 6)) : -1;
            }
        }
        #pragma unroll
        for (int i = 0; i < PER; ++i) {
            const int t = tw + NTW * i;
            if (t < num_tiles) {
                const int pos = t * TILE + 2 * lane;
                const bool flat = BF16KV && !p.block_table;
                const int r0 = pg[i].x < 0 ? -1 : flat ? ix[i].x : pg[i].x * 64 + (ix[i].x & 63);
                const int r1 = pg[i].y < 0 ? -1 : flat ? ix[i].y : pg[i].y * 64 + (ix[i].y & 63);
                *reinterpret_cast<int2 *>(&ss.rows[pos]) = make_int2(r0, r1);
                // SPEC 4.1, checked here rather than by host kernels: valid entries strictly ascending,
                // pads (idx < 0 or past topk_length) only trailing.  A violation poisons the row (NaN out).
                {
                    const bool v0 = ix[i].x >= 0 && pos0 + pos < klen, v1 = ix[i].y >= 0 && pos0 + pos + 1 < klen;
                    const int nx = __shfl_down_sync(0xffffffffu, ix[i].x, 1);
                    const bool nv = __shfl_down_sync(0xffffffffu, (int)v0, 1) != 0;
                    // a selected position whose block-table entry is negative is refused, not masked (review 11)
                    bool bad = (v0 && pg[i].x < 0) || (v1 && pg[i].y < 0) || (v1 && !v0) || (v0 && v1 && ix[i].y <= ix[i].x) ||
                               (lane < 31 && nv && (!v1 || nx <= ix[i].y));
                    if (__any_sync(0xffffffffu, bad) && lane == 0) ss.bad = 1;
                    const uint32_t vb = __ballot_sync(0xffffffffu, v0);
                    const int last_lane = vb ? 31 - __clz(vb) : -1;
                    const int fx = __shfl_sync(0xffffffffu, ix[i].x, 0);
                    const int lx = __shfl_sync(0xffffffffu, v1 ? ix[i].y : ix[i].x, last_lane < 0 ? 0 : last_lane);
                    const bool tv = __shfl_sync(0xffffffffu, (int)v1, 31) != 0;
                    if (lane == 0) {
                        ss.first_pos[t] = (vb & 1u) ? fx : -1;
                        ss.last_pos[t] = last_lane >= 0 ? lx : -1;
                        ss.tail_valid[t] = tv;
                    }
                }
                // the tile's validity word: lane l owns bits 2l (r0) and 2l + 1 (r1)
                const uint32_t e = __ballot_sync(0xffffffffu, r0 >= 0), o = __ballot_sync(0xffffffffu, r1 >= 0);
                if (lane == 0) {
                    // interleave e (bits 2l) and o (bits 2l + 1) with the standard bit-spread (round 4:
                    // the 32-iteration loop delayed tile_ready[0] by ~1.5K cycles)
                    auto spread = [](uint64_t x) {
                        x = (x | (x << 16)) & 0x0000FFFF0000FFFFull;
                        x = (x | (x << 8)) & 0x00FF00FF00FF00FFull;
                        x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0Full;
                        x = (x | (x << 2)) & 0x3333333333333333ull;
                        x = (x | (x << 1)) & 0x5555555555555555ull;
                        return x;
                    };
                    ss.vmask[t] = spread(e) | (spread(o) << 1);
                }
                if constexpr (TILE_READY) {
                    __syncwarp();
                    if (lane == 0) ss.tile_ready[t].arrive();     // release: rows, vmask, first/last/tail, bad
                }
            }
        }
    };
    // the cross-tile half of SPEC 4.1's check (ascending across tiles and across this group's lower
    // boundary, pads trailing) -- one lane, over the per-tile summaries
    auto cross_check = [&]() {
        const int *lst = p.idx + (int64_t)tok * p.idx_stride + pos0;
        int prev = -1;
        bool seen_pad = false;
        if (pos0 > 0) {
            const int pp = __ldg(lst - 1);
            prev = (pp >= 0 && pos0 - 1 < klen) ? pp : -1;
            seen_pad = prev < 0;
        }
        for (int t = 0; t < num_tiles; ++t) {
            const int f = ss.first_pos[t], l = ss.last_pos[t];
            if (f >= 0) {
                if (seen_pad || (prev >= 0 && f <= prev)) ss.bad = 1;
                prev = l;
            }
            if (!ss.tail_valid[t]) seen_pad = true;
        }
        if (ss.bad && p.err) atomicOr(p.err, 1);           // an error flag: not a slot, not a merge
    };
    // Round 3: the table is published PER TILE (tile_ready[t]) by ten translating warps (6, 7 and the eight
    // dequant warps, tile t by translator t % 10), so warp 5 gathers tile 0 as soon as it exists instead of
    // after the whole list; warp 6 then runs the cross-tile check and publishes check_done for the epilogue.
    if (warp > T::WARP_GATHER && warp < T::WARP_DQ0 + T::NUM_DQ_WARPS) {
        const int tw = warp < T::WARP_DQ0 ? warp - T::WARP_GATHER - 1 : warp - T::WARP_DQ0 + 2;
        translate(tw, cute::Int<T::NUM_DQ_WARPS + 2>{}, cute::true_type{});
        if (warp == T::WARP_GATHER + 1 && lane == 0) {
            for (int t = 0; t < num_tiles; ++t) ss.tile_ready[t].wait(0);
            cross_check();
            ss.check_done.arrive();
        }
        __syncwarp();
    }
    if (warp >= T::WARP_GATHER && warp < T::WARP_DQ0) {
        if (warp == T::WARP_GATHER) {
            for (int t = 0; t < num_tiles; ++t) {
                const int s = t % T::KV_STAGES;
                ss.tile_ready[t].wait(0);                    // this tile's rows are in the table
                if (lane == 0) {
                    // the stage is free once PV(t - 3) has read it (QK(t - 3) read it before)
                    const int r = t % T::RAW_STAGES;
                    if constexpr (BF16KV) {
                        if (t >= T::KV_STAGES) (ss.bar_pv[s]).wait(((t / T::KV_STAGES) - 1) & 1);
                    } else {
                        if (t >= T::RAW_STAGES) (ss.raw_empty[r]).wait(((t / T::RAW_STAGES) - 1) & 1);
                    }
                    uint8_t *dst = BF16KV ? reinterpret_cast<uint8_t *>(ss.kv.data() + s * (int)cosize_v<typename T::SmemLayoutKV>)
                                          : ss.raw.data() + r * (TILE * T::RAW_STRIDE);
                    Barrier &rfull = ss.raw_full[BF16KV ? s : r];
                    if constexpr (BF16KV) {
                        // the tile's 128 gather4s are issued by the otherwise idle dequant warps (below);
                        // warp 5 only opens the stage's transaction
                        ss.raw_full[s].arrive_and_expect_tx(TILE * T::DQK * 2);
                        if (128 == 7) bf16_issue_tile<T>(ss, &tm_raw, t, 0, 7);
                    } else {
                        rfull.arrive_and_expect_tx(TILE * T::RAW_STRIDE);
                    }
                }
                if constexpr (!BF16KV) {
                    // lanes 0-15 issue the tile's 16 gather4s in one warp instruction (after lane 0's
                    // expect_tx: the stage is free for every lane once lane 0 passed bar_pv)
                    __syncwarp();
                    if (lane < TILE / 4) {
                        uint8_t *dst = ss.raw.data() + (t % T::RAW_STAGES) * (TILE * T::RAW_STRIDE);
                        Barrier &rfull = ss.raw_full[t % T::RAW_STAGES];
                        const int4 rr = *reinterpret_cast<const int4 *>(&ss.rows[t * TILE + 4 * lane]);
                        tma_gather4(&tm_raw, rfull, dst + (4 * lane) * T::RAW_STRIDE, 0, rr);
                    }
                }
                __syncwarp();
            }
        }
    } else if (warp >= T::WARP_DQ0 && warp < T::WARP_DQ0 + T::NUM_DQ_WARPS && BF16KV) {
        // ======== bf16 kv: warps 8-11 gather rows straight into the SW128 K-major tile (identity) ========
        // DSA_BF16_ISSUERS (bf16_issue_tile): 128 = every lane of warps 8-11 issues one gather4 a tile (shipped;
        // README Step 6 has the A/B: one lane of warp 5 took 2.4x as long).
        const int w = warp - T::WARP_DQ0;
        if (128 == 128) {   // (every lane of every dequant warp)
            // every lane of warps 8-11 issues one gather4 a tile: (g, c) = lane-issuer 32 w + lane
            for (int t = 0; t < num_tiles; ++t) {
                const int s = t % T::KV_STAGES;
                ss.tile_ready[t].wait(0);
                if (t >= T::KV_STAGES) ss.bar_pv[s].wait(((t / T::KV_STAGES) - 1) & 1);
                bf16_issue_tile<T>(ss, &tm_raw, t, 32 * w + lane, 32 * T::NUM_DQ_WARPS);
            }
        } else if (lane == 0) {
            ss.table_ready.wait(0);
            for (int t = 0; t < num_tiles; ++t) {
                const int s = t % T::KV_STAGES;
                if (t >= T::KV_STAGES) ss.bar_pv[s].wait(((t / T::KV_STAGES) - 1) & 1);
                if (128 == 7) bf16_issue_tile<T>(ss, &tm_raw, t, 3 + w, 7);
                else bf16_issue_tile<T>(ss, &tm_raw, t, w, 4);
            }
        }
        __syncwarp();
    } else if (warp >= T::WARP_DQ0 && warp < T::WARP_DQ0 + T::NUM_DQ_WARPS && !BF16KV) {
        // ================== dequantisation: raw e4m3 + ue8m0 -> bf16 SW128 K-major tile ==================
        // Thread i: rows {i/8 + RG j}, RG = 4 * NUM_DQ_WARPS, elements [64 c + 8 (i%8), +8) of chunk c.  The
        // chunks go in ascending order, each with its own arrival, so QK's chunk c can start as soon as
        // chunk c exists rather than after all eight.
        constexpr int RG = 4 * T::NUM_DQ_WARPS, RPT = TILE / RG;   // row groups, rows per thread
        const int wi = tid - T::WARP_DQ0 * 32;
        const int rg = wi / 8, sub = wi % 8;
        for (int t = 0; t < num_tiles; ++t) {
            const int s = t % T::KV_STAGES;
            char *dst = reinterpret_cast<char *>(ss.kv.data() + s * (int)cosize_v<typename T::SmemLayoutKV>);
            const int r = t % T::RAW_STAGES;
            (ss.raw_full[r]).wait((t / T::RAW_STAGES) & 1);
            const uint8_t *raw = ss.raw.data() + r * (TILE * T::RAW_STRIDE);
            // every raw word of this thread's rows first (one batch of independent shared loads), then
            // convert and store; a chunk group's arrival follows its stores (8 per-chunk fences and
            // arrivals made each chunk a dependent load -> convert -> store -> fence chain: 3.2K cycles)
            uint32_t scw[RPT][4];
            uint2 v[RPT][T::CHUNKS];
            #pragma unroll
            for (int j = 0; j < RPT; ++j) {
                const uint8_t *rr = raw + (rg + RG * j) * T::RAW_STRIDE;
                const uint4 sc = *reinterpret_cast<const uint4 *>(rr + 512);
                scw[j][0] = sc.x; scw[j][1] = sc.y; scw[j][2] = sc.z; scw[j][3] = sc.w;
                #pragma unroll
                for (int c = 0; c < T::CHUNKS; ++c) v[j][c] = *reinterpret_cast<const uint2 *>(rr + c * 64 + sub * 8);
            }
            // the raw stage is free once every dequant warp holds its words; the bf16 stage once PV(t - 2) read it
            __syncwarp();
            if (lane == 0) ss.raw_empty[r].arrive();
            if (t >= T::KV_STAGES) (ss.bar_pv[s]).wait(((t / T::KV_STAGES) - 1) & 1);
            #pragma unroll
            for (int cg = 0; cg < T::CGROUPS; ++cg) {
                #pragma unroll
                for (int cc = 0; cc < T::CHUNKS / T::CGROUPS; ++cc) {
                    const int c = cg * (T::CHUNKS / T::CGROUPS) + cc;
                    #pragma unroll
                    for (int j = 0; j < RPT; ++j) {
                        const int row = rg + RG * j;
                        const int blk = c * 2 + sub / 4;                  // the 32-element quant tile
                        const uint32_t s2 = ue8m0_bf16x2(scw[j][blk >> 2] >> ((blk & 3) * 8));
                        uint4 o;
                        o.x = dq2((uint16_t)(v[j][c].x & 0xFFFFu), s2);
                        o.y = dq2((uint16_t)(v[j][c].x >> 16), s2);
                        o.z = dq2((uint16_t)(v[j][c].y & 0xFFFFu), s2);
                        o.w = dq2((uint16_t)(v[j][c].y >> 16), s2);
                        // SW128 K-major: chunk c is an (64 rows x 128 B) block; 16-byte unit XOR row % 8
                        *reinterpret_cast<uint4 *>(dst + c * (TILE * 128) + row * 128 + ((sub ^ (row & 7)) * 16)) = o;
                    }
                }
                cutlass::arch::fence_view_async_shared();
                __syncwarp();
                if (lane == 0) ss.kv_full[s][cg].arrive();
            }
        }
    } else if (warp == T::WARP_MMA) {
        // ================================ the UMMA issuer ================================
        typename T::MmaQK mma_qk;
        typename T::MmaPV mma_pv;
        auto thr_qk = mma_qk.get_slice(0);
        auto thr_pv = mma_pv.get_slice(0);
        Tensor sP = make_tensor(make_smem_ptr(ss.p.data()), typename T::SmemLayoutP{});
        Tensor tOrP = thr_pv.partition_fragment_A(sP);
        const uint64_t desc_p = uint64_t(tOrP.data().desc_);
        const uint32_t kIdescQK = uint32_t(UMMA::make_instr_desc<bf16, bf16, float, T::H, TILE, UMMA::Major::K, UMMA::Major::K>());
        const uint32_t kIdescPV = uint32_t(UMMA::make_instr_desc<bf16, bf16, float, T::H, T::PV_N, UMMA::Major::K, UMMA::Major::MN>());
        const uint32_t tq = tmem_base + T::TMEM_Q;
        ss.bar_q.wait(0);
        tcgen05_fence_after();
        // the stages' UMMA descriptors, built once (review item 7): K of stage s, V's two 256-column halves
        uint64_t dK[T::KV_STAGES], dV0[T::KV_STAGES], dV1[T::KV_STAGES];
        #pragma unroll
        for (int s = 0; s < T::KV_STAGES; ++s) {
            Tensor sKV = make_tensor(make_smem_ptr(ss.kv.data() + s * (int)cosize_v<typename T::SmemLayoutKV>),
                                     typename T::SmemLayoutKV{});
            dK[s] = uint64_t(thr_qk.partition_fragment_B(sKV).data().desc_);
            Tensor sV = make_tensor(sKV.data(), typename T::SmemLayoutV{});
            dV0[s] = uint64_t(thr_pv.partition_fragment_B(flat_divide(sV, Shape<Int<T::PV_N>, Int<TILE>>{})(_, _, _0{}, _0{})).data().desc_);
            dV1[s] = uint64_t(thr_pv.partition_fragment_B(flat_divide(sV, Shape<Int<T::PV_N>, Int<TILE>>{})(_, _, _1{}, _0{})).data().desc_);
        }
        // (a select chain, not an indexed load: a runtime index puts the arrays on the stack)
        auto pick = [&](const uint64_t (&a)[T::KV_STAGES], int s) {
            uint64_t v = a[0];
            #pragma unroll
            for (int i = 1; i < T::KV_STAGES; ++i) v = s == i ? a[i] : v;
            return v;
        };
        auto qk = [&](int g) {
            const int s = g % T::KV_STAGES, b = g % 2;
            const uint64_t desc_k = pick(dK, s);
            const uint32_t tmem_s = tmem_base + T::TMEM_S + (uint32_t)(b * 64);
            #pragma unroll
            for (int c = 0; c < T::QCHUNKS; ++c) {
                if constexpr (BF16KV) {
                    if (c == 0) { (ss.raw_full[s]).wait((g / T::KV_STAGES) & 1); tcgen05_fence_after(); }
                } else if (c % (T::CHUNKS / T::CGROUPS) == 0) {
                    (ss.kv_full[s][c / (T::CHUNKS / T::CGROUPS)]).wait((g / T::KV_STAGES) & 1);
                    tcgen05_fence_after();
                }
                umma_qk_chunk_ts(tmem_s, tq + (uint32_t)(c * 32), desc_k + (uint64_t)(c * 512), kIdescQK,
                                 c == 0 ? 0u : 1u);
            }
            cutlass::arch::umma_arrive(reinterpret_cast<uint64_t *>(&ss.bar_qk[b]));
        };
        auto pv = [&](int g) {
            const int s = g % T::KV_STAGES;
            const uint32_t acc = (g % seg_tiles) == 0 ? 0u : 1u;      // a group's first PV overwrites O
            umma_pv_half(tmem_base + T::TMEM_O + T::O_DP * TMEM_DP, desc_p, pick(dV0, s), kIdescPV, acc);
            umma_pv_half(tmem_base + T::TMEM_O + T::O_DP * TMEM_DP + T::PV_N, desc_p, pick(dV1, s), kIdescPV, acc);
            cutlass::arch::umma_arrive(reinterpret_cast<uint64_t *>(&ss.bar_pv[s]));
        };
        // Issue order (Fable review item 1): QK(g+1) then PV(g), unless P(g) is already published, in
        // which case PV(g) goes first -- its retirement frees stage g % 3 for the gather of g + 3,
        // and QK(g+1) may still be waiting for its dequant.  The order is chosen per tile from the
        // barrier state; both orders write the same TMEM cells with the same instructions (S(g+1) and
        // O are disjoint; QK(g+1) still follows bar_p(g-1)), so the result is bit-identical.
        if (num_tiles > 0) qk(0);
        for (int g = 0; g < num_tiles; ++g) {
            if (g + 1 < num_tiles) qk(g + 1);
            (ss.bar_p).wait(g & 1);
            tcgen05_fence_after();
            pv(g);
        }
    } else if (warp < 4 || (T::SM_SPLIT && warp >= T::WARP_SMB0)) {
        const int qd = warp % 4;                                   // the TMEM lane quadrant = rows [16 qd, +16)
        // round 4 softmax diet (packed f32x2 + full-tile mask skip): bit-identical; shipped on the bf16-kv
        // instantiations only (prefill -1.4%; fp8 decode +1.6%: its softmax is not on the critical path)
        constexpr bool kDiet = 1 && T::BF16;
        const int half = (T::SM_SPLIT && warp >= T::WARP_SMB0) ? 1 : 0;   // S columns [SCOLS half, +SCOLS)
        auto pair_sync = [&]() { if constexpr (T::SM_SPLIT) asm volatile("bar.sync %0, 64;" :: "r"(3 + qd) : "memory"); };
        // ============================ softmax, correction, epilogue ============================
        // Q into TMEM first: lane l < 16 of warp w writes DP 32w + l = row 16w + l, all 32 k-steps,
        // k-step k at columns [8k, 8k + 8) (element pairs packed, the even element low).  Lanes 16..31
        // address the DP+16 half (O's columns): they write zeros there, before any PV, which the first
        // PV's ScaleOut::Zero overwrites anyway.
        if (half == 0) {
            const int m = qd * 16 + (lane % 16);
            const bool q_lane = lane < 16 && head0 + m < p.h_q;
            const bf16 *qrow = reinterpret_cast<const bf16 *>(p.q) + (int64_t)tok * p.q_tok_stride +
                               (int64_t)(head0 + m) * p.q_head_stride;
            // round 4: DSA_QFILL_BATCH k-step pairs' loads in flight before their STTMs (each STTM's asm
            // memory clobber otherwise serialises 16+ load -> store rounds on full global latency: the MMA
            // warp waited bar_q for 21% of its life, ncu round 4).  Same values, same TMEM cells.
            constexpr int QB = ((T::DQK / 32) % 4 == 0) ? 4 : 3;   // 576: 18 pairs -> 3
            static_assert((T::DQK / 32) % QB == 0, "whole batches of k-step pairs");
            for (int k0 = 0; k0 < T::DQK / 16; k0 += 2 * QB) {
                uint32_t r[QB][16];
                #pragma unroll
                for (int u = 0; u < QB; ++u) {
                    #pragma unroll
                    for (int jj = 0; jj < 2; ++jj) {
                        const int k = k0 + 2 * u + jj;
                        uint4 a = make_uint4(0, 0, 0, 0), b = make_uint4(0, 0, 0, 0);
                        if (q_lane) {
                            a = __ldg(reinterpret_cast<const uint4 *>(qrow + 16 * k));
                            b = __ldg(reinterpret_cast<const uint4 *>(qrow + 16 * k + 8));
                        }
                        r[u][8 * jj + 0] = a.x; r[u][8 * jj + 1] = a.y; r[u][8 * jj + 2] = a.z; r[u][8 * jj + 3] = a.w;
                        r[u][8 * jj + 4] = b.x; r[u][8 * jj + 5] = b.y; r[u][8 * jj + 6] = b.z; r[u][8 * jj + 7] = b.w;
                    }
                }
                #pragma unroll
                for (int u = 0; u < QB; ++u)
                    SM100_TMEM_STORE_32dp32b16x::copy(r[u][0], r[u][1], r[u][2], r[u][3], r[u][4], r[u][5], r[u][6], r[u][7],
                                                      r[u][8], r[u][9], r[u][10], r[u][11], r[u][12], r[u][13], r[u][14], r[u][15],
                                                      tmem_base + T::TMEM_Q + (uint32_t)(qd * 32) * TMEM_DP + (uint32_t)(8 * (k0 + 2 * u)));
            }
            cutlass::arch::fence_view_async_tmem_store();
            tcgen05_fence_before();
            __syncwarp();
            if (lane == 0) ss.bar_q.arrive();
        }
        constexpr int kRows = 16;
        const int row_a = qd * kRows + (lane / 4);
        const int row_b = row_a + 8;
        Tensor sP = make_tensor(make_smem_ptr(ss.p.data()), typename T::SmemLayoutP{});
        Tensor cTile = make_identity_tensor(Shape<Int<kRows>, Int<64>>{});
        auto tmem_probe = tmem_tile<T>(tmem_base);
        auto tiled_t2r = make_tmem_copy(typename T::T2R{}, tmem_probe);
        auto tiled_r2t = make_tmem_copy(typename T::R2T{}, tmem_probe);
        Tensor cFrag = tiled_t2r.get_slice(lane).partition_D(cTile);
        constexpr int kVals = decltype(size(cFrag))::value;
        static_assert(kVals == 32);
        auto tiled_t2r_s = make_tmem_copy(typename T::T2R_S{}, tmem_tile_s<T>(tmem_base));
        Tensor cFragS = tiled_t2r_s.get_slice(lane).partition_D(make_identity_tensor(Shape<Int<kRows>, Int<T::SCOLS>>{}));
        constexpr int kValsS = decltype(size(cFragS))::value;
        const float scale_log2 = p.scale_log2;
        float rM[2] = {MAX_INIT_VAL_SM, MAX_INIT_VAL_SM};
        float rL[2] = {0.0f, 0.0f};
        for (int g = 0; g < num_tiles; ++g) {
            const int buf = g % 2;
            const bool seg_start = (g % seg_tiles) == 0;
            (ss.tile_ready[g]).wait(0);                   // the tile's validity word
            (ss.bar_qk[buf]).wait((g / 2) & 1);
            tcgen05_fence_after();
            Tensor tS = tmem_tile_s<T>(tmem_base + T::TMEM_S + (uint32_t)(buf * 64 + half * T::SCOLS) + (uint32_t)(qd * 32) * TMEM_DP);
            Tensor tSp = tiled_t2r_s.get_slice(lane).partition_S(tS);
            Tensor rS = make_tensor<float>(shape(cFragS));
            copy(tiled_t2r_s, tSp, rS);
            cutlass::arch::fence_view_async_tmem_load();
            float cur_max[2] = {MAX_INIT_VAL, MAX_INIT_VAL};
            // the lane's columns are 8 r + 2 (lane % 4) + {0, 1}: shift the tile's validity word once, then
            // every bit test below is at a compile-time position
            const uint64_t vm = ss.vmask[g] >> (half * T::SCOLS + 2 * (lane % 4));
            // round 4 (diet): a tile whose 64 slots are all real keys needs no mask (warp-uniform: one word per tile)
            const bool full_tile = kDiet && ss.vmask[g] == ~0ull;
            if (!full_tile) {
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < kValsS; ++i) {
                    const int col8 = (i / 4) * 8 + (i % 2);                  // the column minus 2 (lane % 4)
                    rS(i) = ((vm >> col8) & 1u) ? rS(i) : MAX_INIT_VAL;      // mask before the max
                }
            }
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kValsS; ++i) {
                const int lr = get<0>(cFragS(i)) >= 8 ? 1 : 0;
                cur_max[lr] = fmaxf(cur_max[lr], rS(i));
            }
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < 2; ++i) {
                cur_max[i] = fmaxf(cur_max[i], __shfl_xor_sync(0xffffffffu, cur_max[i], 1));
                cur_max[i] = fmaxf(cur_max[i], __shfl_xor_sync(0xffffffffu, cur_max[i], 2));
            }
            if constexpr (T::SM_SPLIT) {
                // the other half's maxima of the same rows; fmaxf is exact, so the row max is v1's bits
                if (lane % 4 == 0) { ss.xmax[qd][half][lane / 4] = cur_max[0]; ss.xmax[qd][half][lane / 4 + 8] = cur_max[1]; }
                pair_sync();
                cur_max[0] = fmaxf(cur_max[0], ss.xmax[qd][half ^ 1][lane / 4]);
                cur_max[1] = fmaxf(cur_max[1], ss.xmax[qd][half ^ 1][lane / 4 + 8]);
            }
            if (seg_start) { rM[0] = rM[1] = MAX_INIT_VAL_SM; rL[0] = rL[1] = 0.0f; }
            float new_max[2], scale_old[2];
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < 2; ++i) {
                const float cand = fmaxf(rM[i], cur_max[i] * scale_log2);
                if constexpr (kLazy) {
                    // RECIPE v2 (sm100_dsa_sparse_v2): keep the running max while the tile's candidate is
                    // within RESCALE_THRESHOLD_V2 log2 units of it; kept rows have scale_old EXACTLY 1.0f
                    // (their O correction is a no-op; the warp skips it when all 16 rows keep).  A segment's
                    // first tile commits: rM = MAX_INIT_VAL_SM there.
                    const bool keep = cand - rM[i] <= RESCALE_THRESHOLD_V2;
                    new_max[i] = keep ? rM[i] : cand;
                    scale_old[i] = keep ? 1.0f : exp2f(rM[i] - cand);
                } else {
                    new_max[i] = cand;
                    scale_old[i] = exp2f(rM[i] - new_max[i]);
                }
                rM[i] = new_max[i];
            }
            float part[2] = {0.0f, 0.0f};
            uint32_t rP[kValsS / 2];
            float prs[kValsS / 2];                                   // the lane's pair sums p0 + p1, in order
if constexpr (kDiet) {
            // round 4: packed f32x2 (FMUL2.FTZ / FADD2.FTZ: the same per-element RN roundings as the scalar
            // FMUL.FTZ / FADD.FTZ, so the same bits): x = s * scale_log2 then + (-new_max) (a - b == a + (-b)
            // exactly); the pair sums of rows a and b (elements i, i+2 of a 4-group) in one FADD2
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kValsS; i += 4) {
                const float2 sl2 = make_float2(scale_log2, scale_log2);
                const float2 xa = __fadd2_rn(__fmul2_rn(make_float2(rS(i), rS(i + 1)), sl2), make_float2(-new_max[0], -new_max[0]));
                const float2 xb = __fadd2_rn(__fmul2_rn(make_float2(rS(i + 2), rS(i + 3)), sl2), make_float2(-new_max[1], -new_max[1]));
                const float p0 = exp2f(xa.x), p1 = exp2f(xa.y), p2 = exp2f(xb.x), p3 = exp2f(xb.y);
                const float2 ps = __fadd2_rn(make_float2(p0, p2), make_float2(p1, p3));
                prs[i / 2] = ps.x;
                prs[i / 2 + 1] = ps.y;
                bf16 pa[2] = {(bf16)p0, (bf16)p1}, pb[2] = {(bf16)p2, (bf16)p3};
                rP[i / 2] = *(const uint32_t *)pa;
                rP[i / 2 + 1] = *(const uint32_t *)pb;
            }
} else {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kValsS; i += 2) {
                const int lr = get<0>(cFragS(i)) >= 8 ? 1 : 0;
                const float p0 = exp2f(rS(i) * scale_log2 - new_max[lr]);
                const float p1 = exp2f(rS(i + 1) * scale_log2 - new_max[lr]);
                prs[i / 2] = p0 + p1;
                bf16 pair[2] = {(bf16)p0, (bf16)p1};
                rP[i / 2] = *(const uint32_t *)pair;
            }
}
            // SPEC 4.2's per-lane chain part = ((0 + pair_r0) + pair_r1) + ... over r ascending.  Split: half 0
            // runs r = 0..3 and hands its partials to half 1, which CONTINUES the same chain over r = 4..7 --
            // the identical sequence of fp32 adds -- then the shfl_xor 1, 2 combine and L, exactly v1's.
            if (T::SM_SPLIT && half == 1) {
                pair_sync();                                         // half 0's partials are in xsum
                part[0] = ss.xsum[qd][lane][0];
                part[1] = ss.xsum[qd][lane][1];
            }
if constexpr (kDiet) {
            {   // the a and b chains advanced together, one FADD2 per r
                float2 pp = make_float2(part[0], part[1]);
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < kValsS; i += 4) pp = __fadd2_rn(pp, make_float2(prs[i / 2], prs[i / 2 + 1]));
                part[0] = pp.x;
                part[1] = pp.y;
            }
} else {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kValsS; i += 2) {
                const int lr = get<0>(cFragS(i)) >= 8 ? 1 : 0;
                part[lr] += prs[i / 2];
            }
}
            if (T::SM_SPLIT && half == 0) {
                ss.xsum[qd][lane][0] = part[0];
                ss.xsum[qd][lane][1] = part[1];
                pair_sync();
            }
            if (!T::SM_SPLIT || half == 1) {
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < 2; ++i) {
                    part[i] += __shfl_xor_sync(0xffffffffu, part[i], 1);
                    part[i] += __shfl_xor_sync(0xffffffffu, part[i], 2);
                    rL[i] = rL[i] * scale_old[i] + part[i];
                }
            }
            // P(t) overwrites sP (PV(t-1)'s A operand) and O(t-1) is corrected: both need PV(t-1)
            if (g > 0) (ss.bar_pv[(g - 1) % T::KV_STAGES]).wait(((g - 1) / T::KV_STAGES) & 1);
            tcgen05_fence_after();
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kValsS; i += 2) {
                // an explicit st.shared (final review (b)(1): through cute's smem tensor these were generic ST.E)
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(
                                 &sP(qd * kRows + get<0>(cFragS(i)), half * T::SCOLS + get<1>(cFragS(i)))))), "r"(rP[i / 2]) : "memory");
            }
            const bool rescale = !seg_start && __any_sync(0xffffffffu, scale_old[0] != 1.0f || scale_old[1] != 1.0f);
            if (rescale) {
                auto tO_at = [&](int ch) {
                    const int h = ch / 4, cb = ch % 4;
                    return tiled_t2r.get_slice(lane).partition_S(tmem_tile<T>(
                        tmem_base + T::TMEM_O + (uint32_t)(T::O_DP + qd * 32) * TMEM_DP + (uint32_t)(h * T::PV_N + cb * 64)));
                };
                // SM_SPLIT: half h corrects O's columns [256 h, 256 h + 256) (chunks 4h .. 4h + 3)
                const int ch0 = T::SM_SPLIT ? 4 * half : 0, ch1 = T::SM_SPLIT ? ch0 + 4 : 8;
                CUTLASS_PRAGMA_UNROLL
                for (int ch = ch0; ch < ch1; ++ch) {
                    Tensor tOp = tO_at(ch);
                    Tensor rO = make_tensor<float>(shape(cFrag));
                    copy(tiled_t2r, tOp, rO);
                    cutlass::arch::fence_view_async_tmem_load();
if constexpr (kDiet) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int i = 0; i < kVals; i += 2) {                      // FMUL2.FTZ: per-element FMUL.FTZ
                        const float sc = scale_old[get<0>(cFrag(i)) >= 8 ? 1 : 0];
                        const float2 o2 = __fmul2_rn(make_float2(rO(i), rO(i + 1)), make_float2(sc, sc));
                        rO(i) = o2.x;
                        rO(i + 1) = o2.y;
                    }
} else {
                    CUTLASS_PRAGMA_UNROLL
                    for (int i = 0; i < kVals; ++i) rO(i) = rO(i) * scale_old[get<0>(cFrag(i)) >= 8 ? 1 : 0];
}
                    copy(tiled_r2t, rO, tOp);
                }
                cutlass::arch::fence_view_async_tmem_store();
            }
            cutlass::arch::fence_view_async_shared();
            tcgen05_fence_before();
            __syncwarp();
            if (lane == 0) ss.bar_p.arrive();
        }
        // ---- the group's end.  G == 1: the root, out = bf16(O / L'), lse = logf(L') + M / log2(e).
        // G > 1: the raw fp32 partial (O[512], L, M), never normalised (SPEC 4.2), for the combine.
        if (num_tiles > 0) {
            const int gl = num_tiles - 1;
            ss.bar_pv[gl % T::KV_STAGES].wait((gl / T::KV_STAGES) & 1);
            tcgen05_fence_after();
        }
        const bool publish_root = p.partials == nullptr;
        ss.check_done.wait(0);                           // the cross-tile check has decided ss.bad
        const bool poison = ss.bad != 0;                // a refused list: the row's outputs are NaN
        if constexpr (T::SM_SPLIT) {
            // half 1 holds L (it ran the row sums); half 0 normalises its 256 columns with the same L
            if (half == 1 && lane % 4 == 0) { ss.xL[qd][lane / 4] = rL[0]; ss.xL[qd][lane / 4 + 8] = rL[1]; }
            pair_sync();
            if (half == 0) { rL[0] = ss.xL[qd][lane / 4]; rL[1] = ss.xL[qd][lane / 4 + 8]; }
        }
        const bool rv[2] = {head0 + row_a < p.h_q, head0 + row_b < p.h_q};
        // sink-fused root (sm100_dsa_sparse_sink_v1 at G == 1): the expressions of sink.cu's
        // dsa_sink_combine_kernel on the raw (O, L, M) -- M' = max(M, sink log2e), sa = exp2(M - M'),
        // L' = L sa + exp2(sink log2e - M'), out = (O sa) / L', one bf16 rounding; an empty row is the sink alone.
        float Lp[2] = {(rL[0] == 0.0f || rL[0] != rL[0]) ? 1.0f : rL[0],
                       (rL[1] == 0.0f || rL[1] != rL[1]) ? 1.0f : rL[1]};
        float sa[2] = {1.0f, 1.0f}, lse_fused[2] = {0.0f, 0.0f};
        const bool fused_sink = publish_root && p.sink != nullptr;
        if (fused_sink) {
            CUTLASS_PRAGMA_UNROLL
            for (int lr = 0; lr < 2; ++lr) {
                const int hh = head0 + (lr ? row_b : row_a);
                const float sk = rv[lr] ? p.sink[hh] : 0.0f;
                const float sink_log2 = sk * (float)M_LOG2E;
                const bool empty = rL[lr] == 0.0f;
                const float Mf = empty ? sink_log2 : fmaxf(rM[lr], sink_log2);
                sa[lr] = empty ? 0.0f : exp2f(rM[lr] - Mf);
                Lp[lr] = rL[lr] * sa[lr] + exp2f(sink_log2 - Mf);
                lse_fused[lr] = empty ? sk : logf(Lp[lr]) + Mf / (float)M_LOG2E;
            }
        }
        const int G = gridDim.z;
        #pragma unroll 1
        for (int ch = (T::SM_SPLIT ? 4 * half : 0); ch < (T::SM_SPLIT ? 4 * half + 4 : 8); ++ch) {
            const int h = ch / 4, cb = ch % 4;
            const int col_base = h * T::PV_N + cb * 64;
            Tensor tO = tmem_tile<T>(tmem_base + T::TMEM_O + (uint32_t)(T::O_DP + qd * 32) * TMEM_DP + (uint32_t)(h * T::PV_N + cb * 64));
            Tensor tOp = tiled_t2r.get_slice(lane).partition_S(tO);
            Tensor rO = make_tensor<float>(shape(cFrag));
            if (num_tiles > 0) {
                copy(tiled_t2r, tOp, rO);
                cutlass::arch::fence_view_async_tmem_load();
            } else {
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < kVals; ++i) rO(i) = 0.0f;
            }
            CUTLASS_PRAGMA_UNROLL
            for (int lrx = 0; lrx < 2; ++lrx) {
                if (!rv[lrx] && !(0 && publish_root)) continue;   // (the quad transpose needs every lane)
                const int r = lrx ? row_b : row_a;
                if (publish_root) {
                    char *const op = (char *)p.out + (((int64_t)tok * p.h_q) + head0 + r) * (T::D * 2);
                    const float Lrow = Lp[lrx];
                    CUTLASS_PRAGMA_UNROLL
                    for (int i = 0; i < kVals; i += 2) {
                        if ((get<0>(cFrag(i)) >= 8 ? 1 : 0) != lrx) continue;
                        bf16 o2[2] = {(bf16)(rO(i) / Lrow), (bf16)(rO(i + 1) / Lrow)};
                        if (fused_sink) {
                            o2[0] = (bf16)((rO(i) * sa[lrx]) / Lrow);
                            o2[1] = (bf16)((rO(i + 1) * sa[lrx]) / Lrow);
                        }
                        if (poison) { o2[0] = (bf16)CUDART_NAN_F; o2[1] = (bf16)CUDART_NAN_F; }
                        *(uint32_t *)(op + (col_base + get<1>(cFrag(i))) * 2) = *(const uint32_t *)o2;
                    }
                } else {
                    float *const slot = p.partials + ((((int64_t)tok * G + grp) * p.h_q) + head0 + r) * 520;
                    CUTLASS_PRAGMA_UNROLL
                    for (int i = 0; i < kVals; i += 2) {
                        if ((get<0>(cFrag(i)) >= 8 ? 1 : 0) != lrx) continue;
                        *(float2 *)(slot + col_base + get<1>(cFrag(i))) =
                            poison ? make_float2(CUDART_NAN_F, CUDART_NAN_F) : make_float2(rO(i), rO(i + 1));
                    }
                }
            }
        }
        if (lane % 4 == 0 && (!T::SM_SPLIT || half == 1)) {
            #pragma unroll
            for (int lr = 0; lr < 2; ++lr) {
                if (!rv[lr]) continue;
                const int r = lr ? row_b : row_a;
                if (publish_root) {
                    p.lse[(int64_t)tok * p.h_q + head0 + r] = poison ? CUDART_NAN_F
                        : fused_sink ? lse_fused[lr] : logf(Lp[lr]) + rM[lr] / (float)M_LOG2E;
                } else {
                    float *const slot = p.partials + ((((int64_t)tok * G + grp) * p.h_q) + head0 + r) * 520;
                    *(float2 *)(slot + 512) = poison ? make_float2(CUDART_NAN_F, CUDART_NAN_F) : make_float2(rL[lr], rM[lr]);
                }
            }
        }
    }
    tcgen05_fence_before();
    __syncthreads();
    if (warp == T::WARP_MMA) {
        tmem_allocator.release_allocation_lock();
        tmem_allocator.free(tmem_base, T::TMEM_COLS);
    }
#endif
}

}  // namespace dsa_sm100::attn
