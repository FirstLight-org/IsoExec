// SPDX-License-Identifier: MIT
// The first PV step uses ScaleOut::Zero to overwrite D rather than add to zero.
#pragma once

#include <math_constants.h>
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/arch/barrier.h>

#include "utils.h"
#include "params.h"
#include "leaf.h"
#include "config.h"
#include "traits.h"

namespace mla_sm100::fwd {

using namespace cute;

// sM is seeded with MAX_INIT_VAL_SM and masked scores with MAX_INIT_VAL, so that
// MAX_INIT_VAL*scale_softmax_log2 < MAX_INIT_VAL_SM holds for every deployment scale.
static constexpr float MAX_INIT_VAL_SM = -1e30f;
static constexpr float MAX_INIT_VAL = -1e33f;
static constexpr int MAX_DEPTH = 12;   // tree levels a run may keep pending (2^12 segments)
//: RECIPE v2 (sm100_absorbed_segtree_v2) only: the lazy rescale threshold, in log2 units of the
//: scaled score -- FA4's value and gqa_sm100_v2's.  While a tile's max exceeds the row's running
//: max by at most this much, the running max is KEPT: no O rescale, scale_old == 1.0f exactly, and
//: the tile's probabilities are taken against the kept max, so the largest is at most 2^8 = 256,
//: which bf16 holds at the relative precision it holds 1.0 at.  The decision reads only the row's
//: own running max over its own fixed 64-key tiles in logical order inside one segment, reset at
//: every segment start: a pure function of logical position, as the exact max was.
static constexpr float RESCALE_THRESHOLD = 8.0f;

static constexpr int NODE_LM = 64 * 512;               // 32768: past the O region
__forceinline__ __device__ int node_o_off(int w, int ch, int lane) {
    return ((w * 8 + ch) * 8 * 32 + lane) * 4;         // + q * 128 for quad q
}

__forceinline__ __device__ void tcgen05_fence_before() {
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}
__forceinline__ __device__ void tcgen05_fence_after() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

// A bounded mbarrier wait, for locating a deadlock.  Off in the shipped kernel; a diagnostic build
// defines MLA_SM100_BOUNDED_WAIT, and then every wait site that gives up records its id in
// counts[7] and unwinds the CTA, so the host sees WHERE the kernel stopped instead of a hang.
#ifdef MLA_SM100_BOUNDED_WAIT
__forceinline__ __device__ bool mla_try_wait(TMABarrier &bar, uint32_t phase) {
    return bar.try_wait(phase);
}
__forceinline__ __device__ bool mla_wait_bounded(TMABarrier &bar, uint32_t phase, int id, int *counts) {
    for (int i = 0; i < (1 << 14); ++i) {
        if (mla_try_wait(bar, phase)) return true;
    }
    if ((threadIdx.x & 31) == 0) atomicMax(&counts[7], id);
    return false;
}
#define MLA_WAIT(BAR, PHASE, ID) \
    do { if (!mla_wait_bounded((BAR), (PHASE), (ID), params.counts)) return; } while (0)
#else
#define MLA_WAIT(BAR, PHASE, ID) (BAR).wait(PHASE)
#endif


//: The QK run's k-step descriptor ADVANCES, in 16-byte units: three K16 steps of +2 inside a
//: 64-dim chunk, then +506 to the first step of the next of the nine chunks (8 latent + 1 rotary),
//: i.e. absolute offsets 0,2,4,6, 512,514,516,518, ..., 4096,4098,4100,4102.  The chain is
//: INCREMENTAL on purpose: `add.u64 da,da,D` keeps exactly one descriptor live, so ptxas cannot
//: hoist thirty-six of them and run the register file out (the absolute form `da = base + OFF`
//: was tried first and took the leaf from 213 to 255 registers with 512 bytes of stack).
#define MLA_QK_STEPS(X)                                                                            \
    X(2) X(2) X(2)   X(506) X(2) X(2) X(2)   X(506) X(2) X(2) X(2)   X(506) X(2) X(2) X(2)         \
    X(506) X(2) X(2) X(2)   X(506) X(2) X(2) X(2)   X(506) X(2) X(2) X(2)   X(506) X(2) X(2) X(2)  \
    X(506) X(2) X(2) X(2)
//: The PV run's three k-step advances: A (P, K-major) +2, B (V, MN-major) +128.
#define MLA_PV_STEPS(X)  X(2, 128) X(2, 128) X(2, 128)

#define MLA_ASM_QK_STEP(D)                                                                         \
    "add.u64 da,da," #D ";\n\tadd.u64 db,db," #D ";\n\t"                                           \
    "@lead tcgen05.mma.cta_group::1.kind::f16 [%0],da,db,%3,q;\n\t"
#define MLA_INT_ONE(D) D,
#define MLA_INT_PAIR_A(DA, DB) DA,
#define MLA_INT_PAIR_B(DA, DB) DB,

//: The absolute k-step offsets the advances above add up to, for the compile-time check below.
template<int N>
struct DescOffsets { int v[N]; };
template<int N, int M>
constexpr DescOffsets<N> desc_offsets_from(const int (&advance)[M]) {
    static_assert(N == M + 1, "N k-steps have N-1 advances");
    DescOffsets<N> o{};
    for (int k = 1; k < N; ++k) o.v[k] = o.v[k - 1] + advance[k - 1];
    return o;
}

//: Whether a cute fragment layout's own k-step offsets are exactly the ones the asm chain walks.
//: A CUTLASS upgrade that moves a swizzle atom or a fragment layout fails the build here.
template<class L, int N>
constexpr bool desc_offsets_match(L, DescOffsets<N> const &expect) {
    if (int(cute::size<2>(L{})) != N) return false;
    for (int k = 0; k < N; ++k) {
        if (int(L{}(0, 0, k)) != expect.v[k]) return false;
    }
    return true;
}

inline constexpr int kQkAdvance[] = {MLA_QK_STEPS(MLA_INT_ONE)};
inline constexpr int kPvAdvanceA[] = {MLA_PV_STEPS(MLA_INT_PAIR_A)};
inline constexpr int kPvAdvanceB[] = {MLA_PV_STEPS(MLA_INT_PAIR_B)};
inline constexpr auto kQkDescOff = desc_offsets_from<36>(kQkAdvance);
inline constexpr auto kPvDescOffA = desc_offsets_from<4>(kPvAdvanceA);
inline constexpr auto kPvDescOffB = desc_offsets_from<4>(kPvAdvanceB);

//: One QK run: 36 k-steps of 16 over the ascending 0..575 chain, the first with ScaleOut::Zero.
//: One `elect.sync` for the whole run; the MMA is PREDICATED on it rather than branched around.
//: That matters: inside a branch only the elected lane is live, so ptxas puts the descriptor
//: advances back on the regular datapath (measured: 11.2 instructions per MMA with a branch
//: against 8.9 with the predicate, because the UIADD3s become IMAD + R2UR).
__forceinline__ __device__ void umma_qk_run(uint32_t tmem_s, uint64_t desc_q, uint64_t desc_kv,
                                            uint32_t idesc) {
    asm volatile(
        "{\n\t.reg .b64 da,db;\n\t.reg .pred p,q,lead;\n\t"
        "elect.sync _|lead, -1;\n\t"
        "setp.ne.b32 p,0,0;\n\t"
        "setp.ne.b32 q,1,0;\n\t"
        "mov.b64 da,%1;\n\tmov.b64 db,%2;\n\t"
        "@lead tcgen05.mma.cta_group::1.kind::f16 [%0],da,db,%3,p;\n\t"
        MLA_QK_STEPS(MLA_ASM_QK_STEP)
        "}\n"
        :
        : "r"(tmem_s), "l"(desc_q), "l"(desc_kv), "r"(idesc)
        : "memory");
}

//: One PV half: four K16 steps of P.V into one 256-wide O accumulator.  `accumulate` is 0 at a
//: segment's first tile (ScaleOut::Zero, D overwritten) and 1 otherwise -- the one runtime
//: predicate in either chain.  The two halves are issued as two runs in the order `cute::gemm`
//: issued them (dims 0..255, then 256..511), which is the order in-order retirement preserves.
#define MLA_ASM_PV_BODY                                                                       \
    "{\n\t.reg .b64 da,db;\n\t.reg .pred p,q,lead;\n\t"                                            \
    "elect.sync _|lead, -1;\n\t"                                                                   \
    "setp.ne.b32 p,%4,0;\n\t"                                                                      \
    "setp.ne.b32 q,1,0;\n\t"                                                                       \
    "mov.b64 da,%1;\n\tmov.b64 db,%2;\n\t"                                                         \
    "@lead tcgen05.mma.cta_group::1.kind::f16 [%0],da,db,%3,p;\n\t"                                \
    "add.u64 da,da,2;\n\tadd.u64 db,db,128;\n\t"                                                   \
    "@lead tcgen05.mma.cta_group::1.kind::f16 [%0],da,db,%3,q;\n\t"                                \
    "add.u64 da,da,2;\n\tadd.u64 db,db,128;\n\t"                                                   \
    "@lead tcgen05.mma.cta_group::1.kind::f16 [%0],da,db,%3,q;\n\t"                                \
    "add.u64 da,da,2;\n\tadd.u64 db,db,128;\n\t"                                                   \
    "@lead tcgen05.mma.cta_group::1.kind::f16 [%0],da,db,%3,q;\n\t"                                \
    "}\n"

//: Four K16 steps of a QK run on ONE 64-dim chunk.  Same instructions and same order as
//: `umma_qk_run`'s chunk of four; split out so a per-chunk arrival can be waited between chunks.
//: The caller advances both descriptors by 512 units per chunk, which is exactly what
//: `kQkDescOff` says the chunk stride is.
__forceinline__ __device__ void umma_qk_chunk(uint32_t tmem_s, uint64_t desc_q, uint64_t desc_kv,
                                              uint32_t idesc, uint32_t accumulate) {
    asm volatile(
        "{\n\t.reg .b64 da,db;\n\t.reg .pred p,q,lead;\n\t"
        "elect.sync _|lead, -1;\n\t"
        "setp.ne.b32 p,%4,0;\n\t"
        "setp.ne.b32 q,1,0;\n\t"
        "mov.b64 da,%1;\n\tmov.b64 db,%2;\n\t"
        "@lead tcgen05.mma.cta_group::1.kind::f16 [%0],da,db,%3,p;\n\t"
        "add.u64 da,da,2;\n\tadd.u64 db,db,2;\n\t"
        "@lead tcgen05.mma.cta_group::1.kind::f16 [%0],da,db,%3,q;\n\t"
        "add.u64 da,da,2;\n\tadd.u64 db,db,2;\n\t"
        "@lead tcgen05.mma.cta_group::1.kind::f16 [%0],da,db,%3,q;\n\t"
        "add.u64 da,da,2;\n\tadd.u64 db,db,2;\n\t"
        "@lead tcgen05.mma.cta_group::1.kind::f16 [%0],da,db,%3,q;\n\t"
        "}\n"
        :
        : "r"(tmem_s), "l"(desc_q), "l"(desc_kv), "r"(idesc), "r"(accumulate)
        : "memory");
}

//: ROUND 5: four K16 steps of a QK run on ONE 64-dim chunk with A (Q) FROM TMEM -- the
//: SM100_MMA_F16BF16_TS form.  Same k-steps in the same ascending order into the same accumulator
//: as the SS form above; only where A is read from moves, and whether that moves a bit is exactly
//: the question the A/B harness answers with torch.equal.  A advances 8 TMEM columns per K16 step
//: (16 bf16, two per 32-bit column), B (KV) the same +2 descriptor units as before.
__forceinline__ __device__ void umma_qk_chunk_ts(uint32_t tmem_s, uint32_t tmem_a, uint64_t desc_kv,
                                                 uint32_t idesc, uint32_t accumulate) {
    asm volatile(
        "{\n\t.reg .b64 db;\n\t.reg .b32 ta;\n\t.reg .pred p,q,lead;\n\t"
        "elect.sync _|lead, -1;\n\t"
        "setp.ne.b32 p,%4,0;\n\t"
        "setp.ne.b32 q,1,0;\n\t"
        "mov.b32 ta,%1;\n\tmov.b64 db,%2;\n\t"
        "@lead tcgen05.mma.cta_group::1.kind::f16 [%0],[ta],db,%3,p;\n\t"
        "add.u32 ta,ta,8;\n\tadd.u64 db,db,2;\n\t"
        "@lead tcgen05.mma.cta_group::1.kind::f16 [%0],[ta],db,%3,q;\n\t"
        "add.u32 ta,ta,8;\n\tadd.u64 db,db,2;\n\t"
        "@lead tcgen05.mma.cta_group::1.kind::f16 [%0],[ta],db,%3,q;\n\t"
        "add.u32 ta,ta,8;\n\tadd.u64 db,db,2;\n\t"
        "@lead tcgen05.mma.cta_group::1.kind::f16 [%0],[ta],db,%3,q;\n\t"
        "}\n"
        :
        : "r"(tmem_s), "r"(tmem_a), "l"(desc_kv), "r"(idesc), "r"(accumulate)
        : "memory");
}

__forceinline__ __device__ void umma_pv_half(uint32_t tmem_o, uint64_t desc_p, uint64_t desc_v,
                                             uint32_t idesc, uint32_t accumulate) {
    asm volatile(
        MLA_ASM_PV_BODY
        :
        : "r"(tmem_o), "l"(desc_p), "l"(desc_v), "r"(idesc), "r"(accumulate)
        : "memory");
}

//: TMEM tensor of one compute warp's 16-row group, 64 columns wide, at `addr`.
template<typename T>
__forceinline__ __device__ auto tmem_tile(uint32_t addr) {
    return make_tensor(make_tmem_ptr<float>(addr), typename T::TmemTile{});
}

//: 9 head-dim chunks of one KV page into `sKV`, EACH on its own transaction barrier, so the QK
//: run can start on chunk 0 without waiting for the other eight.
template<typename T, typename TmaK, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void load_kv_tile(Tensor<Engine0, Layout0> const &gKV,
                                             Tensor<Engine1, Layout1> &sKV,
                                             TmaK const &tma_K, TMABarrier *bar) {
    auto thr_tma = tma_K.get_slice(_0{});
    Tensor src = thr_tma.partition_S(gKV);
    Tensor dst = thr_tma.partition_D(sKV);
    CUTLASS_PRAGMA_UNROLL
    for (int c = 0; c < T::QK_CHUNKS; ++c) {
        bar[c].arrive_and_expect_tx(T::KV_CHUNK_BYTES);
        cute::copy(tma_K.with(reinterpret_cast<typename TMABarrier::ValueType &>(bar[c]), 0,
                              cute::TMA::CacheHintSm90::EVICT_FIRST),
                   src(_, _0{}, c), dst(_, _0{}, c));
    }
}

template<typename T, typename TmaParams_, bool kLazy>
__global__ void __launch_bounds__(T::NUM_THREADS, 1, 1)
mla_sm100_leaf_kernel(__grid_constant__ const FwdParams params,
                      __grid_constant__ const TmaParams_ tma_params) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    using InputT = typename T::InputT;
    constexpr int kBlockN = T::PAGE_BLOCK_SIZE;
    constexpr int kRows = T::ROWS_PER_WARP;
    constexpr int kOChunks = T::PV_HALVES * (T::PV_N / 64);   // 8 x (16 rows, 64 cols)
    constexpr int64_t NODE_STRIDE = (int64_t)T::BLOCK_SIZE_M * PARTIAL_WIDTH;
    static_assert((int)NODE_LM + 2 * T::BLOCK_SIZE_M <= T::BLOCK_SIZE_M * PARTIAL_WIDTH,
                  "the relaid-out node must fit the stack row it replaces");

    extern __shared__ char smem_raw[];
    typename T::SharedStorage &ss = *reinterpret_cast<typename T::SharedStorage *>(smem_raw);

    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int worker_idx = blockIdx.x;

    Tensor sP = make_tensor(make_smem_ptr(ss.smem_p.data()), typename T::SmemLayoutP{});

    if (tid == 0) {
        cute::prefetch_tma_descriptor(tma_params.tma_K.get_tma_descriptor());
        // ROUND 5: Q is written into TMEM by the four compute warps, one arrival each per group.
        for (int i = 0; i < 3; ++i) ss.bar_qg[i].init(T::NUM_COMPUTE_WARPS);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < T::KV_STAGES; ++i) {
            ss.bar_pv[i].init(1);
            CUTLASS_PRAGMA_UNROLL
            for (int c = 0; c < T::QK_CHUNKS; ++c) ss.bar_kv[i][c].init(1);
        }
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < Config::S_BUFFERS; ++i) ss.bar_qk[i].init(1);
        ss.bar_p.init(T::NUM_COMPUTE_WARPS);
    }
    cutlass::arch::fence_barrier_init();
    __syncthreads();

    // tcgen05.alloc is NON-BLOCKING and writes the base column through the tensor-memory proxy, so
    // a bare __syncthreads() does not order that write against a reader: the tcgen05 thread-sync
    // fence pair is what makes it visible.  Reading a garbage base means every later tcgen05.ld
    // addresses nothing, which hangs rather than faults.
    cute::TMEM::Allocator1Sm tmem_allocator;
    if (warp == T::WARP_MMA) {
        tmem_allocator.allocate(T::TMEM_COLS, &ss.tmem_ptr);
        __syncwarp();
    }
    tcgen05_fence_before();
    __syncthreads();
    tcgen05_fence_after();
    const uint32_t tmem_base = ss.tmem_ptr;

    int item_pos = params.heads[worker_idx];
    if (item_pos < 0 && params.pdl && tid == 0) {
        cudaTriggerProgrammaticLaunchCompletion();
    }
    const int h_q = params.h_q;
    const int seg_tokens = params.seg_tokens;
    const int seg_blocks = seg_tokens / kBlockN;
    uint32_t phase_q = 0;
    // mbarrier phases are CTA state, not per-item state: a worker may walk several items, so every
    // ring index and parity is derived from a tile counter that runs across the whole item list.
    // (Restarting the parity at each item is the classic one-bit-clock bug: a producer an even
    // number of generations ahead is indistinguishable from one that has not arrived.)
    int tile_base = 0;

    // Row ownership: warp `warp` owns rows [16*warp, 16*warp+16); the 16dp256b8x fragment gives
    // lane (lane%4, lane/4) the two rows lane/4 and lane/4 + 8 of that group.
    const int row_a = warp * kRows + (lane / 4);
    const int row_b = row_a + 8;
    // ROUND 5 item 3b: the WINDOW rows these two TMEM rows hold.  Window row j is placed on TMEM row
    // m = 16*(j%4) + j/4, so a block with ten valid rows puts 3, 3, 2, 2 of them on warps 0..3
    // instead of all ten on warp 0 -- whose epilogue was the whole compute chain at decode.

    // Static (row-in-group, column) coordinate of every register of a 16x64 TMEM fragment.
    Tensor cTile = make_identity_tensor(Shape<Int<kRows>, Int<64>>{});
    auto tmem_probe = tmem_tile<T>(tmem_base);
    auto tiled_t2r = make_tmem_copy(typename T::T2R{}, tmem_probe);
    auto tiled_r2t = make_tmem_copy(typename T::R2T{}, tmem_probe);
    Tensor cFrag = tiled_t2r.get_slice(lane).partition_D(cTile);
    constexpr int kVals = decltype(size(cFrag))::value;
    static_assert(kVals == T::T2R_VALS, "16dp256b8x must give 32 fp32 per lane");

    // A worker walks its own item list.  The bound is the plan's item count: a list that cycles is a
    // plan bug, and a kernel that spins on one is indistinguishable from a hung GPU, so it refuses
    // instead (the battery's capacity/overflow checks are what catch the plan).
    const int item_bound = params.counts[NUM_ITEMS];
    int items_walked = 0;
    #pragma unroll 1
    while (item_pos >= 0 && items_walked++ <= item_bound) {
        const Item item = params.items[item_pos];
        const int batch_idx = item.req;
        const int seqlen_k = __ldg(params.seqused_k + batch_idx);
        const int q_begin = __ldg(params.cu_seqlens_q + batch_idx);
        const int q_len = __ldg(params.cu_seqlens_q + batch_idx + 1) - q_begin;
        const int num_rows = q_len * h_q;
        const int first_row = item.m_block * T::BLOCK_SIZE_M;
        const int rows_here = min(num_rows - first_row, (int)T::BLOCK_SIZE_M);
        const int pos_first = seqlen_k - q_len + first_row / h_q;
        const int pos_last = seqlen_k - q_len + (first_row + rows_here - 1) / h_q;
        const int nmax = pos_last / seg_tokens + 1;
        const int k_last = item.k1 - 1;
        const int n_run = item.k1 - item.k0;
        const int *block_table_ptr = params.block_table + (int64_t)batch_idx * params.block_table_stride;

        const int start_block = item.k0 * seg_blocks;
        const int end_block = (min(item.k1 * seg_tokens, pos_last + 1) + kBlockN - 1) / kBlockN;
        const int num_tiles = max(end_block - start_block, 0);
        const int border_min = max(pos_first + 1, 0);   // borders increase with the row index
        // ROUND 5 item 3b, measured: spreading a block's rows over the four warps pays only when
        // the block fits ONE warp (rows_here <= 16: decode at h_q <= 16), where it puts 2-3 valid
        // rows on every warp instead of all of them on warp 0; at h_q = 20 it measured +8%, because
        // a lane holds two rows and the node push writes both, so 20 rows on four warps is 80
        // pushing lanes against 48.  Bit-identical either way (row-permutation equivariance of the
        // tensor core, measured), so the rule is free to depend on the item.
        const bool spread = rows_here <= kRows;
        const int win_a = spread ? 4 * (row_a % kRows) + row_a / kRows : row_a;
        const int win_b = spread ? 4 * (row_b % kRows) + row_b / kRows : row_b;

        Tensor tma_gK = tma_params.tma_K.get_tma_tensor(tma_params.shape_K)(_, _, 0, _);

        if (warp == T::WARP_LOAD) {
            if (lane == 0) {
                #pragma unroll 1
                for (int t = 0; t < num_tiles; ++t) {
                    const int g = tile_base + t;
                    const int stage = g % T::KV_STAGES;
                    if (g >= T::KV_STAGES) {
                        MLA_WAIT(ss.bar_pv[stage], ((g - T::KV_STAGES) / T::KV_STAGES) & 1, 10);
                    }
                    Tensor sKV = make_tensor(
                        make_smem_ptr(ss.smem_kv.data() + stage * (int)cosize_v<typename T::SmemLayoutKV>),
                        typename T::SmemLayoutKV{});
                    const int page = __ldg(block_table_ptr + start_block + t);
                    load_kv_tile<T>(tma_gK(_, _, page), sKV, tma_params.tma_K, ss.bar_kv[stage]);
                }
            }
        } else if (warp == T::WARP_MMA) {
            typename T::MmaQK mma_qk;
            typename T::MmaPV mma_pv;
            auto thr_qk = mma_qk.get_slice(0);
            auto thr_pv = mma_pv.get_slice(0);
            Tensor tOrP = thr_pv.partition_fragment_A(sP);

            // The instruction descriptor is a compile-time constant of the atom (it encodes only
            // the formats, M, N and the two majornesses), so it is a `mov` of an immediate onto the
            // uniform datapath rather than a value CUTLASS re-derives from `traits.idesc_`.
            // (`const`, not `constexpr`: cute's InstrDescriptor reads the inactive member of a
            // union, which C++ forbids in a constant expression.  The value is still folded to an
            // immediate -- round 1's SASS already carried it as `UMOV UR11, 0x4100490`.)
            const uint32_t kIdescQK = uint32_t(UMMA::make_instr_desc<
                InputT, InputT, float, T::BLOCK_SIZE_M, kBlockN,
                UMMA::Major::K, UMMA::Major::K>());
            const uint32_t kIdescPV = uint32_t(UMMA::make_instr_desc<
                InputT, InputT, float, T::BLOCK_SIZE_M, T::PV_N,
                UMMA::Major::K, UMMA::Major::MN>());

            // The asm chains' literal offsets ARE cute's own fragment offsets.  If a CUTLASS
            // upgrade changes a swizzle atom or a fragment layout, this fails to compile instead
            // of quietly issuing a different GEMM.
            static_assert(desc_offsets_match(decltype(tOrP.layout()){}, kPvDescOffA),
                          "PV A (P) k-step descriptor offsets moved");
            const uint64_t desc_p = uint64_t(tOrP.data().desc_);

            {
                // ROUND 5: Q readiness is waited per group inside the item's first QK.

                #define MLA_SM100_QK(GTILE, FIRST_)                                                             \
                    do {                                                                                \
                        const int stage_ = (GTILE) % T::KV_STAGES;                                       \
                        const int buf_ = (GTILE) % Config::S_BUFFERS;                                    \
                        const uint32_t phase_ = ((GTILE) / T::KV_STAGES) & 1;                             \
                        Tensor sKV_ = make_tensor(                                                        \
                            make_smem_ptr(ss.smem_kv.data() +                                             \
                                          stage_ * (int)cosize_v<typename T::SmemLayoutKV>),              \
                            typename T::SmemLayoutKV{});                                                  \
                        Tensor tSrKV_ = thr_qk.partition_fragment_B(sKV_);                                   \
                        static_assert(desc_offsets_match(decltype(tSrKV_.layout()){}, kQkDescOff),            \
                                      "QK B (KV) k-step descriptor offsets moved");                        \
                        const uint32_t tmem_s_ = T::tmem_s_acc(tmem_base, buf_);                           \
                        const uint64_t desc_kv_ = uint64_t(tSrKV_.data().desc_);                           \
                        _Pragma("unroll")                                                                  \
                        for (int c_ = 0; c_ < T::QK_CHUNKS; ++c_) {                                        \
                            if (FIRST_ && (c_ % 3) == 0) {                                                 \
                                MLA_WAIT(ss.bar_qg[c_ / 3], phase_q, 23);                                  \
                                tcgen05_fence_after();                                                     \
                            }                                                                              \
                            MLA_WAIT(ss.bar_kv[stage_][c_], phase_, 21);                                   \
                            umma_qk_chunk_ts(tmem_s_, T::tmem_q_acc(tmem_base, c_),                        \
                                          desc_kv_ + (uint64_t)(c_ * 512), kIdescQK,                       \
                                          c_ == 0 ? 0u : 1u);                                              \
                        }                                                                                  \
                        cutlass::arch::umma_arrive(reinterpret_cast<uint64_t *>(&ss.bar_qk[buf_]));         \
                    } while (0)

                if (num_tiles > 0) MLA_SM100_QK(tile_base, true);
                #pragma unroll 1
                for (int t = 0; t < num_tiles; ++t) {
                    const int g = tile_base + t;
                    if (t + 1 < num_tiles) MLA_SM100_QK(g + 1, false);
                    // bar_p is ONE barrier rotated over tiles, which the "one mbarrier per (role,
                    // ring slot)" rule would normally forbid.  It is safe here for a different
                    // reason: the compute warps cannot arrive for tile g+1 before bar_pv[g % KV_STAGES], which
                    // needs this wait to have passed, so the barrier is never armed two ahead.
                    MLA_WAIT(ss.bar_p, g & 1, 22);
                    tcgen05_fence_after();
                    const int stage = g % T::KV_STAGES;
                    Tensor sKV = make_tensor(
                        make_smem_ptr(ss.smem_kv.data() + stage * (int)cosize_v<typename T::SmemLayoutKV>),
                        typename T::SmemLayoutKV{});
                    Tensor sV = make_tensor(sKV.data(), typename T::SmemLayoutV{});
                    const bool seg_start = ((start_block + t) % seg_blocks) == 0;
                    Tensor tOrV0 = thr_pv.partition_fragment_B(
                        flat_divide(sV, Shape<Int<T::PV_N>, Int<kBlockN>>{})(_, _, _0{}, _0{}));
                    Tensor tOrV1 = thr_pv.partition_fragment_B(
                        flat_divide(sV, Shape<Int<T::PV_N>, Int<kBlockN>>{})(_, _, _1{}, _0{}));
                    static_assert(desc_offsets_match(decltype(tOrV0.layout()){}, kPvDescOffB),
                                  "PV B (V) k-step descriptor offsets moved");
                    static_assert(desc_offsets_match(decltype(tOrV1.layout()){}, kPvDescOffB),
                                  "PV B (V) k-step descriptor offsets moved");
                    const uint32_t pv_acc = seg_start ? 0u : 1u;
                    umma_pv_half(T::tmem_o_acc(tmem_base, 0), desc_p,
                                 uint64_t(tOrV0.data().desc_), kIdescPV, pv_acc);
                    umma_pv_half(T::tmem_o_acc(tmem_base, 1), desc_p,
                                 uint64_t(tOrV1.data().desc_), kIdescPV, pv_acc);
                    cutlass::arch::umma_arrive(reinterpret_cast<uint64_t *>(&ss.bar_pv[stage]));
                }
                #undef MLA_SM100_QK
            }
        } else if (warp < T::NUM_COMPUTE_WARPS) {
            {
                auto q_probe = make_tensor(make_tmem_ptr<uint32_t>(tmem_base), typename T::TmemTileQ{});
                auto q_t2r = make_tmem_copy(typename T::T2RQ{}, q_probe);
                auto q_r2t = make_tmem_copy(typename T::R2TQ{}, q_probe);
                Tensor cQ = q_t2r.get_slice(lane).partition_D(
                    make_identity_tensor(Shape<Int<kRows>, Int<32>>{}));
                const uint32_t *qg = reinterpret_cast<const uint32_t *>(params.q);
                #pragma unroll 1
                for (int grp = 0; grp < 3; ++grp) {
                CUTLASS_PRAGMA_UNROLL
                for (int cc = 0; cc < 3; ++cc) {
                    const int c = 3 * grp + cc;
                    Tensor tq = make_tensor(make_tmem_ptr<uint32_t>(T::tmem_q(tmem_base, c, warp)),
                                            typename T::TmemTileQ{});
                    Tensor tqp = q_t2r.get_slice(lane).partition_S(tq);
                    Tensor rq = make_tensor<uint32_t>(shape(cQ));
                    CUTLASS_PRAGMA_UNROLL
                    for (int i = 0; i < int(size(cQ)); ++i) {
                        const int m_ = warp * kRows + get<0>(cQ(i));
                        const int64_t grow = (int64_t)item.row_base +
                                             (spread ? 4 * (m_ % kRows) + m_ / kRows : m_);
                        rq(i) = grow < params.total_rows
                            ? __ldg(qg + grow * (T::HEAD_DIM_K / 2) + c * 32 + get<1>(cQ(i))) : 0u;
                    }
                    copy(q_r2t, rq, tqp);
                }
                cutlass::arch::fence_view_async_tmem_store();
                tcgen05_fence_before();
                __syncwarp();
                if (lane == 0) ss.bar_qg[grp].arrive();
                }
            }
            float rM[2] = {MAX_INIT_VAL_SM, MAX_INIT_VAL_SM};
            float rL[2] = {0.0f, 0.0f};
            int border[2];
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < 2; ++i) {
                const int g = min(first_row + (i ? win_b : win_a), num_rows - 1);
                border[i] = max(seqlen_k - q_len + g / h_q + 1, 0);
            }
            const float scale_log2 = params.scale_softmax_log2;
            float *const stack = params.stack +
                (int64_t)worker_idx * params.stack_depth * NODE_STRIDE;
            const int row_stride = slot_row_stride(rows_here, nmax);
            const int seg_stride = slot_seg_stride(rows_here);
            const bool row_valid[2] = {first_row + win_a < num_rows, first_row + win_b < num_rows};
            const int pos_r[2] = {seqlen_k - q_len + min(first_row + win_a, num_rows - 1) / h_q,
                                  seqlen_k - q_len + min(first_row + win_b, num_rows - 1) / h_q};
            const bool published[2] = {
                row_valid[0] && pos_r[0] >= 0 && (int64_t)pos_r[0] + 1 > (int64_t)k_last * seg_tokens,
                row_valid[1] && pos_r[1] >= 0 && (int64_t)pos_r[1] + 1 > (int64_t)k_last * seg_tokens};
            const bool is_root[2] = {item.k0 == 0 && item.k1 >= pos_r[0] / seg_tokens + 1,
                                     item.k0 == 0 && item.k1 >= pos_r[1] / seg_tokens + 1};

            // Fold the pending nodes of the run's post-order walk after its c-th segment, then
            // either push the result as a node (sink_push) or publish it (root / partial).
            // The scalar (L, M) fold runs first and hands the O fold its merge scales, exactly as
            // the combine kernel does, so both see one tree.
            auto drain = [&](int c, bool last, bool publish) -> int {
                int d = __popc((unsigned)(c - 1));
                int lvl[MAX_DEPTH];
                float sa[MAX_DEPTH][2], sb[MAX_DEPTH][2];
                int nf = 0;
                //: ROUND 5: levels 0 and 1 of this fold in registers (see the fold loop).
                int lv0 = 0, lv1 = 0;
                float sa0[2] = {1.0f, 1.0f}, sb0[2] = {1.0f, 1.0f};
                float sa1[2] = {1.0f, 1.0f}, sb1[2] = {1.0f, 1.0f};
                for (int cc = c; d > 0 && ((cc & 1) == 0 || last); cc >>= 1) {
                    --d;
                    lvl[nf] = d;
                    const float *node = stack + (int64_t)d * NODE_STRIDE;
                    CUTLASS_PRAGMA_UNROLL
                    for (int i = 0; i < 2; ++i) {
                        const int r = i ? row_b : row_a;
                        if (!row_valid[i]) { sa[nf][i] = 1.0f; sb[nf][i] = 1.0f; continue; }
                        const float2 lm = *(const float2 *)(node + NODE_LM + 2 * r);
                        const float La = lm.x;
                        const float Ma = lm.y;
                        const float Mx = fmaxf(Ma, rM[i]);
                        const float s_a = exp2f(Ma - Mx);
                        const float s_b = exp2f(rM[i] - Mx);
                        rL[i] = La * s_a + rL[i] * s_b;
                        rM[i] = Mx;
                        sa[nf][i] = s_a;
                        sb[nf][i] = s_b;
                        if (nf == 0) { sa0[i] = s_a; sb0[i] = s_b; }
                        if (nf == 1) { sa1[i] = s_a; sb1[i] = s_b; }
                    }
                    if (nf == 0) lv0 = d;
                    if (nf == 1) lv1 = d;
                    ++nf;
                }
                float *const sink = publish ? nullptr : stack + (int64_t)d * NODE_STRIDE;
                const float Lp[2] = {(rL[0] == 0.0f || rL[0] != rL[0]) ? 1.0f : rL[0],
                                     (rL[1] == 0.0f || rL[1] != rL[1]) ? 1.0f : rL[1]};
                #pragma unroll 1
                for (int ch = 0; ch < kOChunks; ++ch) {
                    const int h = ch / (T::PV_N / 64), cb = ch % (T::PV_N / 64);
                    const int col_base = h * T::PV_N + cb * 64;
                    Tensor tO = tmem_tile<T>(T::tmem_o(tmem_base, h, warp) + (uint32_t)(cb * 64));
                    Tensor tOp = tiled_t2r.get_slice(lane).partition_S(tO);
                    Tensor rO = make_tensor<float>(shape(cFrag));
                    copy(tiled_t2r, tOp, rO);
                    cutlass::arch::fence_view_async_tmem_load();
                    if (row_valid[0] || row_valid[1])
                    #pragma unroll 1
                    for (int j = 0; j < nf; ++j) {
                        int lvj;
                        float sj[2], tj[2];
                        if (j == 0) {
                            lvj = lv0; sj[0] = sa0[0]; sj[1] = sa0[1]; tj[0] = sb0[0]; tj[1] = sb0[1];
                        } else if (j == 1) {
                            lvj = lv1; sj[0] = sa1[0]; sj[1] = sa1[1]; tj[0] = sb1[0]; tj[1] = sb1[1];
                        } else {
                            lvj = lvl[j]; sj[0] = sa[j][0]; sj[1] = sa[j][1]; tj[0] = sb[j][0]; tj[1] = sb[j][1];
                        }
                        const float *node = stack + (int64_t)lvj * NODE_STRIDE +
                                            node_o_off(warp, ch, lane);
                        CUTLASS_PRAGMA_UNROLL
                        for (int q = 0; q < kVals / 4; ++q) {
                            const float4 a = *(const float4 *)(node + q * 128);
                            const float av[4] = {a.x, a.y, a.z, a.w};
                            CUTLASS_PRAGMA_UNROLL
                            for (int k = 0; k < 4; ++k) {
                                const int i = 4 * q + k;
                                const int lr = get<0>(cFrag(i)) >= 8 ? 1 : 0;
                                const float v = av[k] * sj[lr] + rO(i) * tj[lr];
                                rO(i) = row_valid[lr] ? v : rO(i);
                            }
                        }
                    }
                    if (!publish) {
                        if (row_valid[0] || row_valid[1]) {
                            float *snk = sink + node_o_off(warp, ch, lane);
                            CUTLASS_PRAGMA_UNROLL
                            for (int q = 0; q < kVals / 4; ++q) {
                                *(float4 *)(snk + q * 128) =
                                    float4{rO(4 * q), rO(4 * q + 1), rO(4 * q + 2), rO(4 * q + 3)};
                            }
                        }
                    }
                    if (publish) {
                        CUTLASS_PRAGMA_UNROLL
                        for (int lrx = 0; lrx < 2; ++lrx) {
                            if (!published[lrx]) continue;
                            const int r = lrx ? win_b : win_a;   // a GLOBAL row: the window row
                            if (is_root[lrx]) {
                                char *const op = (char *)params.out +
                                    ((int64_t)item.row_base + r) * (T::HEAD_DIM_V * 2);
                                //: `Lp[lrx]`, NOT its reciprocal.  Under `--prec-div=false`
                                //: `rO(i) / Lrow` is `div.full.f32`, whose denominator-only prefix
                                //: (an FSETP on |b|, a predicated rescale, MUFU.RCP) ptxas hoists
                                //: out of these eight iterations -- which is where round 4's
                                //: FSETP 24 -> 10, MUFU 310 -> 296 and FMUL 762 -> 748 come from.
                                //: Writing `1.f/Lp[lrx]` and multiplying instead would be
                                //: `rcp.approx` and WOULD move bits.  Do not "simplify" it.
                                const float Lrow = Lp[lrx];
                                CUTLASS_PRAGMA_UNROLL
                                for (int i = 0; i < kVals; i += 2) {
                                    if ((get<0>(cFrag(i)) >= 8 ? 1 : 0) != lrx) continue;
                                    InputT o2[2] = {(InputT)(rO(i) / Lrow),
                                                    (InputT)(rO(i + 1) / Lrow)};
                                    *(uint32_t *)(op + (col_base + get<1>(cFrag(i))) * 2) =
                                        *(const uint32_t *)o2;
                                }
                            } else {
                                float *const slot = params.partials +
                                    ((int64_t)item.pslot_base + (int64_t)r * row_stride +
                                     (int64_t)item.k0 * seg_stride) * PARTIAL_WIDTH + col_base;
                                CUTLASS_PRAGMA_UNROLL
                                for (int i = 0; i < kVals; i += 2) {
                                    if ((get<0>(cFrag(i)) >= 8 ? 1 : 0) != lrx) continue;
                                    *(float2 *)(slot + get<1>(cFrag(i))) =
                                        float2{rO(i), rO(i + 1)};
                                }
                            }
                        }
                    }
                }
                if (lane % 4 == 0) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int lr = 0; lr < 2; ++lr) {
                        const int r = lr ? row_b : row_a;     // TMEM row: the stack is leaf-private
                        const int rw = lr ? win_b : win_a;    // window row: what the combine and lse see
                        if (!publish) {
                            if (row_valid[lr]) {
                                *(float2 *)(sink + NODE_LM + 2 * r) = float2{rL[lr], rM[lr]};
                            }
                        } else if (published[lr]) {
                            if (is_root[lr]) {
                                params.lse[(int64_t)item.row_base + rw] =
                                    logf(Lp[lr]) + rM[lr] / (float)M_LOG2E;
                            } else {
                                float *slot = params.partials +
                                    ((int64_t)item.pslot_base + (int64_t)rw * row_stride +
                                     (int64_t)item.k0 * seg_stride) * PARTIAL_WIDTH;
                                slot[T::HEAD_DIM_V] = rL[lr];
                                slot[T::HEAD_DIM_V + 1] = rM[lr];
                            }
                        }
                    }
                }
                return d;
            };

            #pragma unroll 1
            for (int t = 0; t < num_tiles; ++t) {
                const int g = tile_base + t;
                const int buf = g % Config::S_BUFFERS;
                const int stage = g % T::KV_STAGES;
                const int tile_start = (start_block + t) * kBlockN;
                const bool seg_start = ((start_block + t) % seg_blocks) == 0;

                MLA_WAIT(ss.bar_qk[buf], (g / Config::S_BUFFERS) & 1, 30);
                tcgen05_fence_after();

                Tensor tS = tmem_tile<T>(T::tmem_s(tmem_base, buf, warp));
                Tensor tSp = tiled_t2r.get_slice(lane).partition_S(tS);
                Tensor rS = make_tensor<float>(shape(cFrag));
                copy(tiled_t2r, tSp, rS);
                cutlass::arch::fence_view_async_tmem_load();

                // Out-of-bound V of this tile must be finite: a masked probability is exactly 0 and
                // 0 * NaN is NaN.  Zero it before the PV GEMM reads it (QK has already read it, and
                // every key past seqused_k is masked).
                if (tile_start + kBlockN > seqlen_k) {
                    Tensor sKVz = make_tensor(
                        make_smem_ptr(ss.smem_kv.data() + stage * (int)cosize_v<typename T::SmemLayoutKV>),
                        typename T::SmemLayoutKV{});
                    const int valid = max(seqlen_k - tile_start, 0);
                    const int chunks = (kBlockN - valid) * (T::HEAD_DIM_V / 8);
                    for (int i = tid; i < chunks; i += T::COMPUTE_THREADS) {
                        const int r = valid + i / (T::HEAD_DIM_V / 8);
                        const int d8 = (i % (T::HEAD_DIM_V / 8)) * 8;
                        *(uint4 *)(&sKVz(r, d8)) = uint4{0u, 0u, 0u, 0u};
                    }
                }

                const bool need_mask = tile_start + kBlockN > border_min;
                float cur_max[2] = {MAX_INIT_VAL, MAX_INIT_VAL};
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < kVals; ++i) {
                    const int lr = get<0>(cFrag(i)) >= 8 ? 1 : 0;
                    rS(i) = (need_mask && tile_start + get<1>(cFrag(i)) >= border[lr]) ? MAX_INIT_VAL : rS(i);
                    cur_max[lr] = fmaxf(cur_max[lr], rS(i));
                }
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < 2; ++i) {
                    cur_max[i] = fmaxf(cur_max[i], __shfl_xor_sync(0xffffffffu, cur_max[i], 1));
                    cur_max[i] = fmaxf(cur_max[i], __shfl_xor_sync(0xffffffffu, cur_max[i], 2));
                }
                float new_max[2], scale_old[2];
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < 2; ++i) {
                    const float cand = fmaxf(rM[i], cur_max[i] * scale_log2);
                    if constexpr (kLazy) {
                        // RECIPE v2: keep the running max while the tile's max is within the
                        // threshold of it.  A segment's first tile always commits, because rM is
                        // MAX_INIT_VAL_SM there.  When kept, scale_old is EXACTLY 1.0f, so the O
                        // correction for this row is a no-op and the warp skips it when all its
                        // valid rows keep.
                        const bool keep = cand - rM[i] <= RESCALE_THRESHOLD;
                        new_max[i] = keep ? rM[i] : cand;
                        scale_old[i] = keep ? 1.0f : exp2f(rM[i] - cand);
                    } else {
                        new_max[i] = cand;
                        scale_old[i] = exp2f(rM[i] - new_max[i]);
                    }
                    rM[i] = new_max[i];
                }

                float part[2] = {0.0f, 0.0f};
                uint32_t rP[kVals / 2];
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < kVals; i += 2) {
                    const int lr = get<0>(cFrag(i)) >= 8 ? 1 : 0;
                    const float p0 = exp2f(rS(i) * scale_log2 - new_max[lr]);
                    const float p1 = exp2f(rS(i + 1) * scale_log2 - new_max[lr]);
                    part[lr] += p0 + p1;
                    InputT pair[2] = {(InputT)p0, (InputT)p1};
                    rP[i / 2] = *(const uint32_t *)pair;
                }
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < 2; ++i) {
                    part[i] += __shfl_xor_sync(0xffffffffu, part[i], 1);
                    part[i] += __shfl_xor_sync(0xffffffffu, part[i], 2);
                    rL[i] = rL[i] * scale_old[i] + part[i];
                }

                if (g > 0) MLA_WAIT(ss.bar_pv[(g - 1) % T::KV_STAGES], ((g - 1) / T::KV_STAGES) & 1, 31);
                tcgen05_fence_after();
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < kVals; i += 2) {
                    *(uint32_t *)(&sP(warp * kRows + get<0>(cFrag(i)), get<1>(cFrag(i)))) = rP[i / 2];
                }

                // O *= scale_for_old.  Exactly one is free (x*1.0f == x) and the segment's first
                // tile overwrites O, so neither needs the read-modify-write.
                //: ROUND 5: with the rows spread over the four warps, a warp holds 2-3 valid rows
                //: and 13-14 rows nobody reads, which at B >= 8 are the next requests' finite Q
                //: rows -- so ignore those when deciding whether this warp must rescale.
                const bool rescale = !seg_start &&
                    __any_sync(0xffffffffu, (row_valid[0] && scale_old[0] != 1.0f) ||
                                            (row_valid[1] && scale_old[1] != 1.0f));
                if (rescale) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int ch = 0; ch < kOChunks; ++ch) {
                        const int h = ch / (T::PV_N / 64), cb = ch % (T::PV_N / 64);
                        Tensor tO = tmem_tile<T>(T::tmem_o(tmem_base, h, warp) + (uint32_t)(cb * 64));
                        Tensor tOp = tiled_t2r.get_slice(lane).partition_S(tO);
                        Tensor rO = make_tensor<float>(shape(cFrag));
                        copy(tiled_t2r, tOp, rO);
                        cutlass::arch::fence_view_async_tmem_load();
                        CUTLASS_PRAGMA_UNROLL
                        for (int i = 0; i < kVals; ++i) {
                            rO(i) = rO(i) * scale_old[get<0>(cFrag(i)) >= 8 ? 1 : 0];
                        }
                        copy(tiled_r2t, rO, tOp);
                    }
                    cutlass::arch::fence_view_async_tmem_store();
                }
                cutlass::arch::fence_view_async_shared();
                tcgen05_fence_before();
                __syncwarp();
                if (lane == 0) ss.bar_p.arrive();

                // Segment boundary inside the run: fold, push the node, reset the accumulator.
                const int abs_next = start_block + t + 1;
                if (abs_next % seg_blocks == 0 && abs_next < end_block) {
                    MLA_WAIT(ss.bar_pv[stage], (g / T::KV_STAGES) & 1, 32);
                    tcgen05_fence_after();
                    drain(abs_next / seg_blocks - item.k0, false, false);
                    rL[0] = rL[1] = 0.0f;
                    rM[0] = rM[1] = MAX_INIT_VAL_SM;
                }
            }

            if (num_tiles > 0) {
                const int glast = tile_base + num_tiles - 1;
                MLA_WAIT(ss.bar_pv[glast % T::KV_STAGES], (glast / T::KV_STAGES) & 1, 33);
                tcgen05_fence_after();
                drain(n_run, true, true);
            }
        }

        const int next_pos = item.next;
        tile_base += num_tiles;
        __syncthreads();
        if (next_pos >= 0) {
            phase_q ^= 1;
        } else if (params.pdl && tid == 0) {
            cudaTriggerProgrammaticLaunchCompletion();
        }
        item_pos = next_pos;
    }

    tcgen05_fence_before();
    __syncthreads();
    if (warp == T::WARP_MMA) {
        tmem_allocator.release_allocation_lock();
        tmem_allocator.free(tmem_base, T::TMEM_COLS);
    }
#else
    if (cute::thread0()) {
        CUTE_INVALID_CONTROL_PATH("mla_sm100 leaf requires SM100a");
    }
#endif
}

template<typename InputT>
void run_mla_sm100_leaf_kernel(FwdParams &params, const LeafTensors &tensors) {
    using T = Traits<InputT>;
    using index_t = int64_t;
    auto shape_K = make_shape(Int<T::PAGE_BLOCK_SIZE>{}, Int<T::HEAD_DIM_K>{}, (index_t)1,
                              (index_t)tensors.num_blocks);
    auto tma_K = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(make_gmem_ptr((InputT *)tensors.k_ptr),
                    make_layout(shape_K, make_stride(tensors.k_row_stride, _1{},
                                                     tensors.k_head_stride, tensors.k_batch_stride))),
        tile_to_shape(UMMA::Layout_K_SW128_Atom<InputT>{},
                      Layout<Shape<Int<T::PAGE_BLOCK_SIZE>, Int<T::QK_CHUNK>>,
                             Stride<Int<T::HEAD_DIM_K>, _1>>{}));
    TmaParams<decltype(shape_K), decltype(tma_K)> tma_params = {shape_K, tma_K};
    params.q = tensors.q_ptr;   // ROUND 5: the compute warps read Q into TMEM themselves
    // Recipe v1 (exact running max) or v2 (lazy rescale): two instantiations of one leaf.
    auto kernel = (params.recipe == 2) ? &mla_sm100_leaf_kernel<T, decltype(tma_params), true>
                           : &mla_sm100_leaf_kernel<T, decltype(tma_params), false>;
    constexpr size_t smem_size = sizeof(typename T::SharedStorage);
    CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_size));

    cudaLaunchAttribute attributes[1];
    attributes[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attributes[0].val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t config = {
        dim3(params.workers, 1, 1), dim3(T::NUM_THREADS, 1, 1), smem_size, params.stream,
        params.pdl ? attributes : nullptr, params.pdl ? 1 : 0,
    };
    cudaLaunchKernelEx(&config, kernel, params, tma_params);
    CHECK_CUDA_KERNEL_LAUNCH();
}

}  // namespace mla_sm100::fwd
