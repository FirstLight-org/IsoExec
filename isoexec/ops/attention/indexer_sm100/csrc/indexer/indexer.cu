// SPDX-License-Identifier: MIT
// IsoExec sparse_sm100: the deterministic lightning indexer of DSA (recipe sm100_dsa_indexer_v1).
//
// Numerical program (README.md section 2):
//   1. d[t,h,j] = sum_c q[t,h,c] k[j,c] by tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale,
//      e4m3 x e4m3 with hardware ue8m0 block-32 scales, UMMA_K = 32, the four k-steps c = 0, 32, 64,
//      96 in ascending order, the first with ScaleOut::Zero, fp32 TMEM accumulator (spec.py
//      TC_DESCRIPTOR_INDEXER).
//   2. r = fmaxf(d, 0.0f).
//   3. s = ((0 + r0*w0) + r1*w1) + ... over the ASCENDING logical head index h = 0..63, fp32, multiply
//      then add (two roundings; --fmad=false), in ONE thread: the thread that owns key j's TMEM lane.
//   4. S[t,j] = bf16_rne(s), once.
//   5-7. Only keys in the row's window [ks, ke) (logical positions) are folded and stored; a key
//      outside it may be read by the MMA and is never folded.
//
// Blackwell schedule (free, execution only): a persistent CTA walks work items (a query block of QB
// tokens x a span of 128-key tiles); M = 128 keys (A = the K tile), N = 128 = QB tokens x 64 heads
// (B = Q; at QB = 1 rows 64..127 of the B tile are zero).  Warp 0 issues TMA, warp 1 transposes the
// scale factors and copies them to TMEM (tcgen05.cp), warp 2 issues the MMAs, warps 4..15 are three
// math warpgroups, warpgroup g owning TMEM accumulator stage g and folding every third tile.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdlib>

#include <cute/tensor.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/copy_sm100.hpp>
#include <cute/arch/mma_sm100_desc.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cutlass/arch/barrier.h>

#include "indexer.h"

namespace sparse_sm100::indexer {

using Barrier = cutlass::arch::ClusterTransactionBarrier;

static constexpr int D = 128;              // D_IDX (bytes per fp8 row)
static constexpr int H = 64;               // H_IDX
static constexpr int TILE = 128;           // keys per tile == UMMA_M
static constexpr int UMMA_N = 128;         // QB * 64 heads, padded
static constexpr int QS = 2;               // Q stages
static constexpr int KS = 8;               // K stages (4 measured: decode at 4.5 TB/s)
static constexpr int NWG = 3;    // math warpgroups == TMEM accumulator stages
static constexpr int NUM_THREADS = 128 + NWG * 128;
static constexpr int TMEM_COLS = 512;
static constexpr int TM_ACC = 0;                         // [NWG][128] accumulators
static constexpr int TM_SFQ = NWG * UMMA_N;              // [QS][4]
static constexpr int TM_SFK = TM_SFQ + QS * 4;           // [KS][4]
static_assert(TM_SFK + KS * 4 <= TMEM_COLS);

struct __align__(1024) Smem {
    uint8_t q[QS][UMMA_N * D];                 // 16 KB each, SW128 K-major (1024-aligned)
    uint8_t k[KS][TILE * D];                   // 16 KB each
    uint32_t q_sf_raw[QS][128];                // TMA destination (raw, one word per row)
    uint32_t q_sf[QS][128];                    // transposed for tcgen05.cp
    uint32_t k_sf_raw[KS][128];
    uint32_t k_sf[KS][128];
    __nv_bfloat16 wq[QS][2][H];                // the Q stage's bf16 weights (TMA, with Q)
    float2 w[NWG][H];                          // per-warpgroup fp32 weights (w of row 0, w of row 1)
    Barrier q_full[QS], q_sf_full[QS], q_ready[QS], q_empty[QS];
    Barrier k_full[KS], k_sf_full[KS], k_ready[KS], k_empty[KS];
    Barrier tm_full[NWG], tm_empty[NWG];
    uint32_t tmem_base;
    int live[NUM_THREADS / 32];                // per-warp maxima of the live key-span items (live_splits)
    int live_sum[NUM_THREADS / 32];            // per-warp sums of the live tiles (span_auto)
};

// ---- the work decomposition, evaluated identically by every role -------------------------------
struct Item {
    int row0;          // first query row of the block
    int nq;            // query rows in the block (1 or 2)
    int tile0, tile1;  // key tiles [tile0, tile1) (absolute: key = tile*128 + lane)
    int ks[2], ke[2];  // the rows' windows
};
//: The raw per-item inputs, loaded one item ahead so no role waits on them.
struct ItemRaw { int it; int ks[2], ke[2]; };

__device__ __forceinline__ ItemRaw item_load(const IndexerParams &p, int splits, int it) {
    ItemRaw r{it, {0, 0}, {0, 0}};
    const int qb = it / splits;
    if (it >= p.num_qblocks * splits) return r;
    const int row0 = qb * p.qb;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        if (i < p.qb && row0 + i < p.rows) {
            r.ks[i] = p.ks ? __ldg(p.ks + row0 + i) : 0;
            r.ke[i] = __ldg(p.ke + row0 + i);
        }
    }
    return r;
}

__device__ __forceinline__ Item item_of(const IndexerParams &p, int splits, int span, const ItemRaw &r) {
    Item out;
    const int qb = r.it / splits;
    const int sp = r.it % splits;
    out.row0 = qb * p.qb;
    out.nq = min(p.qb, p.rows - out.row0);
    int lo = 0x7fffffff, hi = 0;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        out.ks[i] = r.ks[i]; out.ke[i] = r.ke[i];
        if (i < out.nq && r.ke[i] > r.ks[i]) { lo = min(lo, r.ks[i]); hi = max(hi, r.ke[i]); }
    }
    out.tile0 = out.tile1 = 0;
    if (hi > lo) {
        const int t_lo = lo / TILE, t_hi = (hi + TILE - 1) / TILE;
        if (sp * span < t_hi - t_lo) {
            out.tile0 = t_lo + sp * span;
            out.tile1 = min(out.tile0 + span, t_hi);
        }
    }
    return out;
}

//: for (item : this CTA's items) fn(item), with the next item's inputs in flight.
//: ``splits`` is ``live_splits`` (<= p.splits): the items of key spans past every row's window are never
//: enumerated (they would load nothing and fold nothing), so the walk is bounded by the live context, not
//: by the captured width capacity. Item (qb, sp) still covers tiles [t_lo + sp*span, ...) of its block
//: and every live span keeps sp < splits: the same tiles, folded by the same per-tile program.
template<typename Fn>
__device__ __forceinline__ void for_each_item(const IndexerParams &p, int splits, int span, Fn &&fn) {
    const int num_items = p.num_qblocks * splits;
    ItemRaw nxt = item_load(p, splits, blockIdx.x);
    for (int it = blockIdx.x; it < num_items; it += gridDim.x) {
        const ItemRaw cur = nxt;
        nxt = item_load(p, splits, it + gridDim.x);
        fn(item_of(p, splits, span, cur));
    }
}

//: max over query blocks of the number of non-empty key-span items (item_of's own window arithmetic),
//: clamped to p.splits. Integer max over a fixed set (order-free; no atomics): each thread folds its
//: blocks, each warp reduces, one smem word per warp; every thread then reads all words after the
//: kernel's prologue __syncthreads (the caller), so every role sees the same value.
__device__ __forceinline__ void live_splits_partial(const IndexerParams &p, int span, int *live, int *live_sum) {
    int m = 0, sum = 0;
    if (p.splits > 1) {
        for (int qb = threadIdx.x; qb < p.num_qblocks; qb += NUM_THREADS) {
            const int row0 = qb * p.qb, nq = min(p.qb, p.rows - row0);
            int lo = 0x7fffffff, hi = 0;
            for (int i = 0; i < nq; ++i) {
                const int ks = p.ks ? __ldg(p.ks + row0 + i) : 0, ke = __ldg(p.ke + row0 + i);
                if (ke > ks) { lo = min(lo, ks); hi = max(hi, ke); }
            }
            if (hi > lo) {
                const int tiles = (hi + TILE - 1) / TILE - lo / TILE;
                m = max(m, (tiles + span - 1) / span);
                sum += tiles;
            }
        }
    } else {
        m = p.splits;
    }
    m = __reduce_max_sync(0xffffffffu, m);
    sum = __reduce_add_sync(0xffffffffu, sum);
    if ((threadIdx.x & 31) == 0) { live[threadIdx.x / 32] = m; live_sum[threadIdx.x / 32] = sum; }
}
__device__ __forceinline__ int live_splits(const IndexerParams &p, const int *live) {
    int m = 0;
    #pragma unroll
    for (int w = 0; w < NUM_THREADS / 32; ++w) m = max(m, live[w]);
    return min(m, p.splits);
}


//: IEEE binary32 round-to-nearest-even add with subnormal results and operands kept (no .ftz): the add
//: of the cross-rank tree completion (deepseek_v41_cache_sm100 leaf_tree, detreduce_sm100 tree), which
//: are built with --ftz=false.  Inline PTX, so this TU's --ftz=true cannot turn it into add.rn.ftz.f32.
__device__ __forceinline__ float ieee_add(float a, float b) {
    float r;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}
__device__ __forceinline__ float2 ieee_add2(float2 a, float2 b) {
    return make_float2(ieee_add(a.x, b.x), ieee_add(a.y, b.y));
}

__device__ __forceinline__ uint64_t smem_desc_sw128(const void *ptr) {
    cute::UMMA::SmemDescriptor d;
    d.desc_ = 0;
    d.version_ = 1;
    d.lbo_mode_ = 0;
    d.layout_type_ = static_cast<uint8_t>(cute::UMMA::LayoutType::SWIZZLE_128B);
    d.start_address_ = static_cast<uint16_t>(cute::cast_smem_ptr_to_uint(ptr) >> 4);
    d.base_offset_ = 0;
    d.stride_byte_offset_ = 1024 >> 4;           // 8 rows x 128 B per swizzle atom
    d.leading_byte_offset_ = 0;
    return d.desc_;
}
__device__ __forceinline__ uint64_t smem_desc_sf(const void *ptr) {
    cute::UMMA::SmemDescriptor d;
    d.desc_ = 0;
    d.version_ = 1;
    d.lbo_mode_ = 0;
    d.layout_type_ = static_cast<uint8_t>(cute::UMMA::LayoutType::SWIZZLE_NONE);
    d.start_address_ = static_cast<uint16_t>(cute::cast_smem_ptr_to_uint(ptr) >> 4);
    d.base_offset_ = 0;
    d.stride_byte_offset_ = (8 * 16) >> 4;
    d.leading_byte_offset_ = 0;
    return d.desc_;
}

__device__ __forceinline__ void fence_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void fence_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void umma_commit(Barrier &b) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                 :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&b))) : "memory");
}

//: One block-scaled k-step.  Issued by one elected thread of a converged warp.
__device__ __forceinline__ void mma_mxf8(uint32_t tmem_d, uint64_t da, uint64_t db, uint32_t idesc,
                                         uint32_t sfa, uint32_t sfb, uint32_t accumulate) {
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%5], [%6], p;\n\t"
        "}\n"
        :: "r"(tmem_d), "l"(da), "l"(db), "r"(idesc), "r"(accumulate), "r"(sfa), "r"(sfb) : "memory");
}

template<int N>
__device__ __forceinline__ void tmem_ld_32dp32b(uint32_t addr, float (&v)[N]);
template<>
__device__ __forceinline__ void tmem_ld_32dp32b<16>(uint32_t addr, float (&v)[16]) {
    uint32_t r[16];
    cute::SM100_TMEM_LOAD_32dp32b16x::copy(addr, r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9],
                                           r[10], r[11], r[12], r[13], r[14], r[15]);
    #pragma unroll
    for (int i = 0; i < 16; ++i) v[i] = __uint_as_float(r[i]);
}

__global__ void __launch_bounds__(NUM_THREADS, 1)
dsa_indexer_kernel(const __grid_constant__ IndexerParams p,
                   const __grid_constant__ CUtensorMap tm_q, const __grid_constant__ CUtensorMap tm_qsf,
                   const __grid_constant__ CUtensorMap tm_k, const __grid_constant__ CUtensorMap tm_ksf,
                   const __grid_constant__ CUtensorMap tm_w) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    extern __shared__ __align__(1024) unsigned char smem_raw[];
    // SW128 operand tiles need 1024-byte alignment; the dynamic window is only 16-byte aligned
    // (aligned by pointer arithmetic on the shared array: an integer round trip loses the address space
    // and makes every SMEM access a generic LD.E / ST.E)
    const uint32_t mis = static_cast<uint32_t>(__cvta_generic_to_shared(smem_raw)) & 1023u;
    Smem &ss = *reinterpret_cast<Smem *>(smem_raw + ((1024u - mis) & 1023u));
    const int tid = threadIdx.x;
    const int warp = tid / 32, lane = tid % 32;
    // PDL (launch structure only): nothing global is touched before the previous kernel in the stream has
    // completed; the next kernel may be scheduled from here (it waits for this grid in its own wait).
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

    if (tid == 0) {
        for (int i = 0; i < QS; ++i) {
            ss.q_full[i].init(1); ss.q_sf_full[i].init(1); ss.q_ready[i].init(1); ss.q_empty[i].init(1 + NWG);   // the MMA's commit + one arrive per math warpgroup
        }
        for (int i = 0; i < KS; ++i) {
            ss.k_full[i].init(1); ss.k_sf_full[i].init(1); ss.k_ready[i].init(1); ss.k_empty[i].init(1);
        }
        for (int i = 0; i < NWG; ++i) { ss.tm_full[i].init(1); ss.tm_empty[i].init(128); }
        cutlass::arch::fence_barrier_init();
    }
    // Q rows [64, 128) of a QB = 1 block are never written by TMA: zero them once; their scale
    // words are 0x7F (2^0).  Written before any TMA can touch the tile.
    for (int i = tid; i < QS * UMMA_N * D / 16; i += NUM_THREADS) {
        reinterpret_cast<uint4 *>(&ss.q[0][0])[i] = make_uint4(0, 0, 0, 0);
    }
    for (int i = tid; i < QS * 128; i += NUM_THREADS) (&ss.q_sf_raw[0][0])[i] = 0x7F7F7F7Fu;
    cute::TMEM::Allocator1Sm alloc;
    if (warp == 2) {
        alloc.allocate(TMEM_COLS, &ss.tmem_base);
        __syncwarp();
    }
    live_splits_partial(p, p.span, ss.live, ss.live_sum);
    cutlass::arch::fence_view_async_shared();
    fence_before();
    __syncthreads();
    fence_after();
    const uint32_t tmem = ss.tmem_base;
    // the key span of a work item: fixed (p.span), or (p.span_auto; the partial pass then ran with span 1, so
    // live[] holds the widest block's tiles) the smallest power of two <= span_auto giving at most one item per
    // CTA over the launch's live tiles, and splits = ceil(widest / span) (ceil is monotone). Schedule only:
    // every tile is folded by the same per-tile program whatever item holds it.
    int span = p.span;
    int splits = live_splits(p, ss.live);
    if (p.span_auto) {
        int total = 0;
        #pragma unroll
        for (int w = 0; w < NUM_THREADS / 32; ++w) total += ss.live_sum[w];
        const int per_cta = (total + (int)gridDim.x - 1) / (int)gridDim.x;
        span = 1;
        while (span < per_cta && span < p.span_auto) span <<= 1;
        splits = min((splits + span - 1) / span, p.splits);
    }

    if (warp == 0 || warp == 3) {
        // ================================ TMA producers ================================
        // Two producer warps: warp 0 issues the Q tiles, the weights and the K value tiles; warp 3 the
        // scale words.  One issuing thread was the bottleneck: with no fold and no MMA at all the kernel
        // ran at 4.7 TB/s with that thread never waiting (README "Step 3").
        const bool sf_role = warp == 3;
        // The whole warp walks the items: lane l fetches the page id of the item's l-th page (an item
        // spans at most 16 tiles = 32 pages), so no TMA waits on a dependent block-table load; lane 0
        // issues every TMA with the page id shuffled from its owner.
        int qi = 0, kt = 0;       // item and tile counters of this CTA
        for_each_item(p, splits, span, [&](const Item &item) {
            if (item.tile1 <= item.tile0) return;     // an empty item loads nothing
            int my_page = 0;
            if (p.paged) {
                const int ke = item.ke[0];
                const int lp = item.tile0 * 2 + lane;
                if (lane < (item.tile1 - item.tile0) * 2 && lp * 64 < ke) {
                    const int64_t btr = p.bt_rows ? __ldg(p.bt_rows + item.row0) : item.row0;
                    if (p.page_rows == 128)   // 128-entry pages: one block-table entry per two 64-key units
                        my_page = __ldg(p.block_table + btr * p.bt_stride + (lp >> 1));
                    else
                        my_page = __ldg(p.block_table + btr * p.bt_stride + lp);
                }
            }
            const int qs = qi % QS;
            if (lane == 0) {
                if (qi >= QS) (ss.q_empty[qs]).wait(((qi / QS) - 1) & 1);
                // a TMA box counts its full size, the OOB-filled rows of a short last block included
                if (!sf_role) {
                    ss.q_full[qs].arrive_and_expect_tx(p.qb * H * D + p.qb * H * 2);
                    cute::SM90_TMA_LOAD_2D::copy(&tm_w, reinterpret_cast<uint64_t *>(&ss.q_full[qs]),
                                                 0, &ss.wq[qs][0][0], 0, item.row0);
                    cute::SM90_TMA_LOAD_2D::copy(&tm_q, reinterpret_cast<uint64_t *>(&ss.q_full[qs]),
                                                 0, &ss.q[qs][0], 0, item.row0 * H);
                } else {
                    ss.q_sf_full[qs].arrive_and_expect_tx(p.qb * H * 4);
                    cute::SM90_TMA_LOAD_2D::copy(&tm_qsf, reinterpret_cast<uint64_t *>(&ss.q_sf_full[qs]),
                                                 0, &ss.q_sf_raw[qs][0], 0, item.row0);
                }
            }
            for (int t = item.tile0; t < item.tile1; ++t, ++kt) {
                const int s = kt % KS;
                int pg[2] = {0, 0};
                if (p.paged) {
                    pg[0] = __shfl_sync(0xffffffffu, my_page, (t - item.tile0) * 2);
                    pg[1] = __shfl_sync(0xffffffffu, my_page, (t - item.tile0) * 2 + 1);
                }
                if (lane == 0) {
                    if (kt >= KS) (ss.k_empty[s]).wait(((kt / KS) - 1) & 1);
                    if (!sf_role) {
                        ss.k_full[s].arrive_and_expect_tx(TILE * D);
                        if (p.paged) {
                            #pragma unroll
                            for (int h2 = 0; h2 < 2; ++h2) {
                                // 128-entry pages: key unit t * 2 + h2 is row half h2 of its page
                                const int prow = p.page_rows == 128 ? h2 * 64 : 0;
                                cute::SM90_TMA_LOAD_3D::copy(&tm_k, reinterpret_cast<uint64_t *>(&ss.k_full[s]), 0,
                                                             &ss.k[s][h2 * 64 * D], 0, prow, pg[h2]);
                            }
                        } else {
                            cute::SM90_TMA_LOAD_2D::copy(&tm_k, reinterpret_cast<uint64_t *>(&ss.k_full[s]), 0,
                                                         &ss.k[s][0], 0, t * TILE);
                        }
                    } else {
                        ss.k_sf_full[s].arrive_and_expect_tx(TILE * 4);
                        if (p.paged) {
                            #pragma unroll
                            for (int h2 = 0; h2 < 2; ++h2) {
                                const int prow = p.page_rows == 128 ? h2 * 64 : 0;
                                cute::SM90_TMA_LOAD_2D::copy(&tm_ksf, reinterpret_cast<uint64_t *>(&ss.k_sf_full[s]), 0,
                                                             &ss.k_sf_raw[s][h2 * 64], prow, pg[h2]);
                            }
                        } else {
                            cute::SM90_TMA_LOAD_1D::copy(&tm_ksf, reinterpret_cast<uint64_t *>(&ss.k_sf_full[s]), 0,
                                                         &ss.k_sf_raw[s][0], t * TILE);
                        }
                    }
                }
                __syncwarp();
            }
            ++qi;
        });
    } else if (warp == 1) {
        // ============== scale factors: transpose for tcgen05.cp, copy into TMEM ==============
        // tcgen05.cp.32x128b.warpx4 wants, for lane l, the 4 words of rows l, l+32, l+64, l+96.
        int qi = 0, kt = 0;
        auto transpose_copy = [&](const uint32_t *raw, uint32_t *dst, uint32_t col) {
            uint32_t v[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) v[i] = raw[i * 32 + lane];
            *reinterpret_cast<uint4 *>(dst + lane * 4) = make_uint4(v[0], v[1], v[2], v[3]);
            cutlass::arch::fence_view_async_shared();
            __syncwarp();
            if (cute::elect_one_sync()) cute::SM100_UTCCP_4x32dp128bit_1cta::copy(smem_desc_sf(dst), tmem + col);
            __syncwarp();
        };
        for_each_item(p, splits, span, [&](const Item &item) {
            if (item.tile1 <= item.tile0) return;
            const int qs = qi % QS;
            (ss.q_sf_full[qs]).wait((qi / QS) & 1);
            fence_after();                  // the previous MMAs on these TMEM columns are done (q_empty)
            transpose_copy(ss.q_sf_raw[qs], ss.q_sf[qs], TM_SFQ + qs * 4);
            if (cute::elect_one_sync()) { fence_before(); ss.q_ready[qs].arrive(); }
            __syncwarp();
            for (int t = item.tile0; t < item.tile1; ++t, ++kt) {
                const int s = kt % KS;
                (ss.k_sf_full[s]).wait((kt / KS) & 1);
                fence_after();
                transpose_copy(ss.k_sf_raw[s], ss.k_sf[s], TM_SFK + s * 4);
                if (cute::elect_one_sync()) { fence_before(); ss.k_ready[s].arrive(); }
                __syncwarp();
            }
            ++qi;
        });
    } else if (warp == 2) {
        // ================================ MMA issuer ================================
        const uint32_t idesc_base = uint32_t(cute::UMMA::make_instr_desc_block_scaled<
            cutlass::float_e4m3_t, cutlass::float_e4m3_t, float, cutlass::float_ue8m0_t,
            TILE, UMMA_N, cute::UMMA::Major::K, cute::UMMA::Major::K>());
        int qi = 0, kt = 0;
        for_each_item(p, splits, span, [&](const Item &item) {
            if (item.tile1 <= item.tile0) return;
            const int qs = qi % QS;
            (ss.q_full[qs]).wait((qi / QS) & 1);
            (ss.q_ready[qs]).wait((qi / QS) & 1);
            fence_after();
            const uint64_t db0 = smem_desc_sw128(&ss.q[qs][0]);
            for (int t = item.tile0; t < item.tile1; ++t, ++kt) {
                const int s = kt % KS;
                const int g = kt % NWG;
                (ss.k_full[s]).wait((kt / KS) & 1);
                (ss.k_ready[s]).wait((kt / KS) & 1);
                if (kt >= NWG) (ss.tm_empty[g]).wait(((kt / NWG) - 1) & 1);
                fence_after();
                const uint64_t da0 = smem_desc_sw128(&ss.k[s][0]);
                if (cute::elect_one_sync()) {
                    #pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        cute::UMMA::InstrDescriptorBlockScaled id;
                        id.desc_ = idesc_base;
                        id.a_sf_id_ = k;
                        id.b_sf_id_ = k;
                        // k-step k covers bytes [32k, 32k+32) of each 128-byte row: +2 in 16-byte units
                        mma_mxf8(tmem + TM_ACC + g * UMMA_N, da0 + 2 * k, db0 + 2 * k, id.desc_,
                                 tmem + TM_SFK + s * 4, tmem + TM_SFQ + qs * 4, k == 0 ? 0u : 1u);
                    }
                    umma_commit(ss.k_empty[s]);
                }
                __syncwarp();
            }
            if (cute::elect_one_sync()) umma_commit(ss.q_empty[qs]);
            __syncwarp();
            ++qi;
        });
    } else if (warp >= 4) {
        // ================================ the fold ================================
        const int g = (warp - 4) / 4;                 // warpgroup == TMEM stage
        const int sub = warp % 4;                     // TMEM subpartition: lanes [32 sub, 32 sub + 32)
        const int wtid = tid - 128 - g * 128;
        int kt = 0, qi = 0;
        for_each_item(p, splits, span, [&](const Item &item) {
            if (item.tile1 <= item.tile0) return;
            const int qs = qi % QS;
            // this warpgroup folds the item's tiles with kt % NWG == g (tiles go round-robin by kt)
            const int ntiles = item.tile1 - item.tile0;
            const int first = (g - kt % NWG + NWG) % NWG;          // offset of this warpgroup's first tile
            // Every math warpgroup must observe this query generation before returning its
            // q_empty credit, including groups with no tile in a short tail item. Otherwise
            // an inactive group can run QS items ahead and credit a future generation into
            // the current barrier phase, letting TMA overwrite Q while its MMA still reads it.
            (ss.q_full[qs]).wait((qi / QS) & 1);
            // ... and EVERY THREAD of the group, not only the crediting thread: the credit below is
            // one arrive (wtid 0) for all 128 threads, and the group's four warps are not otherwise
            // synchronized in an item where it has no tile. Without this barrier a fast warp of the
            // group credits item qi, the producer reloads the stage for item qi + QS, and a slow
            // warp of the same group that has not yet polled q_full for item qi sees the barrier two
            // phases ahead -- its parity wait never succeeds and the CTA deadlocks (production TP4
            // "lamport word timeout"). The same bar.sync also orders the previous
            // item's reads of ss.w[g] before the copy below. Synchronization only: no bits change.
            asm volatile("bar.sync %0, 128;" :: "r"(1 + g));
            if (first < ntiles) {
                // the item's weights, fp32 once (SPEC 2 item 3), from the Q stage into this warpgroup's copy
                for (int h = wtid; h < H; h += 128) {
                    const float w0 = __bfloat162float(ss.wq[qs][0][h]);
                    const float w1 = item.nq > 1 ? __bfloat162float(ss.wq[qs][1][h]) : 0.0f;
                    ss.w[g][h] = make_float2(w0, w1);
                }
                asm volatile("bar.sync %0, 128;" :: "r"(1 + g));
            }
            if (wtid == 0) ss.q_empty[qs].arrive();                     // done with this Q stage
            float wreg[H];                                              // one row: weights in registers
            if (item.nq == 1 && first < ntiles && p.tree_leaves == 0) {
                #pragma unroll
                for (int h = 0; h < H; ++h) wreg[h] = ss.w[g][h].x;
            }
            for (int u = first; u < ntiles; u += NWG) {
                const int t = item.tile0 + u;
                const int kk = kt + u;                                  // this tile's global counter
                // ONE commit per tile (k_empty[s]): it both frees the K stage and publishes the
                // accumulator.  Unambiguous: MMA(kk + KS) needs tm_empty from fold(kk + KS - NWG) > kk.
                (ss.k_empty[kk % KS]).wait((kk / KS) & 1);
                fence_after();
                const int j = t * TILE + sub * 32 + lane;
                const uint32_t lane_addr = tmem + ((uint32_t)(sub * 32) << 16) + TM_ACC + g * UMMA_N;
                float s0 = 0.0f, s1 = 0.0f;
                if (p.tree_leaves > 0) {
                    // sm100_dsa_indexer_tree_v1: leaf l = heads [4l, 4l + 4), ((0 + r w) + r w) + ...,
                    // then the balanced adjacent-pair tree over the leaves (power-of-two count).  A rank
                    // holding an aligned block of leaves computes exactly that block's subtree.  Heads
                    // past the leaves (zero padding) are never loaded.  Two rows: packed float2 lanes.
                    const int nheads = 4 * p.tree_leaves;
                    float2 acc[8];
                    #pragma unroll
                    for (int l = 0; l < 8; ++l) acc[l] = make_float2(0.0f, 0.0f);
                    if (item.nq == 2) {
                        const float4 *w4 = reinterpret_cast<const float4 *>(&ss.w[g][0]);
                        #pragma unroll
                        for (int c = 0; c < H / 16; ++c) {
                            if (c < 2 && c * 16 < nheads) {  // <= 8 leaves = 32 heads; fully unrolled: c is a constant per copy
                                float v0[16], v1[16];
                                tmem_ld_32dp32b<16>(lane_addr + c * 16, v0);
                                tmem_ld_32dp32b<16>(lane_addr + H + c * 16, v1);
                                cutlass::arch::fence_view_async_tmem_load();
                                #pragma unroll
                                for (int i = 0; i < 16; i += 2) {
                                    const float4 ww = w4[(c * 16 + i) / 2];
                                    const int l = (c * 16 + i) / 4;
                                    const float2 ra = make_float2(fmaxf(v0[i], 0.0f), fmaxf(v1[i], 0.0f));
                                    acc[l] = __fadd2_rn(acc[l], __fmul2_rn(ra, make_float2(ww.x, ww.y)));
                                    const float2 rb = make_float2(fmaxf(v0[i + 1], 0.0f), fmaxf(v1[i + 1], 0.0f));
                                    acc[l] = __fadd2_rn(acc[l], __fmul2_rn(rb, make_float2(ww.z, ww.w)));
                                }
                            }
                        }
                    } else {
                        // one row: each leaf's four heads chain in a register, written once
                        #pragma unroll
                        for (int c = 0; c < H / 16; ++c) {
                            if (c < 2 && c * 16 < nheads) {
                                float v0[16];
                                tmem_ld_32dp32b<16>(lane_addr + c * 16, v0);
                                cutlass::arch::fence_view_async_tmem_load();
                                #pragma unroll
                                for (int i = 0; i < 16; i += 4) {
                                    float a = 0.0f;
                                    #pragma unroll
                                    for (int k = 0; k < 4; ++k) {
                                        const float r = fmaxf(v0[i + k], 0.0f);
                                        const float prod = r * ss.w[g][c * 16 + i + k].x;
                                        a = a + prod;
                                    }
                                    acc[(c * 16 + i) / 4].x = a;
                                }
                            }
                        }
                    }
                    // Levels whose pair count is <= 2^(tree_ieee_levels - 1) (the top tree_ieee_levels
                    // levels) add with IEEE semantics (ieee_add2: the cross-rank leaf_tree's add.rn.f32);
                    // the lower ones with this TU's FTZ packed add.  tree_ieee_levels == 0: all FTZ.
                    int L = p.tree_leaves;
                    const int ieee_from = (1 << p.tree_ieee_levels) >> 1;     // largest IEEE pair count
                    #pragma unroll
                    for (int lvl = 0; lvl < 3; ++lvl) {
                        const bool ieee = L / 2 <= ieee_from;
                        #pragma unroll
                        for (int k = 0; k < 4; ++k)
                            if (2 * k + 1 < L)
                                acc[k] = ieee ? ieee_add2(acc[2 * k], acc[2 * k + 1])
                                              : __fadd2_rn(acc[2 * k], acc[2 * k + 1]);
                        L = L > 1 ? L / 2 : 1;
                    }
                    s0 = acc[0].x;
                    s1 = acc[0].y;
                } else
                if (item.nq == 2) {
                    // two independent chains (row 0 and row 1 of the block) packed per head: the same
                    // per-element fmaxf, RN multiply and RN add as the scalar fold.  The weights come
                    // two heads per 16-byte shared load (one per head was 64 LDS per thread per tile
                    // and starved the UMMA's operand reads: measured, README "Step 3").
                    float2 s = make_float2(0.0f, 0.0f);
                    const float4 *w4 = reinterpret_cast<const float4 *>(&ss.w[g][0]);
                    #pragma unroll
                    for (int c = 0; c < H / 16; ++c) {
                        float v0[16], v1[16];
                        tmem_ld_32dp32b<16>(lane_addr + c * 16, v0);
                        tmem_ld_32dp32b<16>(lane_addr + H + c * 16, v1);
                        cutlass::arch::fence_view_async_tmem_load();
                        #pragma unroll
                        for (int i = 0; i < 16; i += 2) {
                            const float4 ww = w4[(c * 16 + i) / 2];
                            const float2 ra = make_float2(fmaxf(v0[i], 0.0f), fmaxf(v1[i], 0.0f));
                            s = __fadd2_rn(s, __fmul2_rn(ra, make_float2(ww.x, ww.y)));
                            const float2 rb = make_float2(fmaxf(v0[i + 1], 0.0f), fmaxf(v1[i + 1], 0.0f));
                            s = __fadd2_rn(s, __fmul2_rn(rb, make_float2(ww.z, ww.w)));
                        }
                    }
                    s0 = s.x; s1 = s.y;
                } else
                {
                    float s = 0.0f;
                    #pragma unroll
                    for (int c = 0; c < H / 16; ++c) {
                        float v[16];
                        tmem_ld_32dp32b<16>(lane_addr + c * 16, v);
                        cutlass::arch::fence_view_async_tmem_load();
                        #pragma unroll
                        for (int i = 0; i < 16; ++i) {
                            const float r = fmaxf(v[i], 0.0f);
                            const float prod = r * wreg[c * 16 + i];
                            s = s + prod;
                        }
                    }
                    s0 = s;
                }
                fence_before();
                ss.tm_empty[g].arrive();
                if (p.out_f32) {
                    float *of = p.out_f32 + (int64_t)item.row0 * p.out_stride + j;
                    if (j >= item.ks[0] && j < item.ke[0]) of[0] = s0;
                    if (item.nq == 2 && j >= item.ks[1] && j < item.ke[1]) of[p.out_stride] = s1;
                } else {
                    __nv_bfloat16 *o = p.out + (int64_t)item.row0 * p.out_stride + j;
                    if (j >= item.ks[0] && j < item.ke[0]) o[0] = __float2bfloat16_rn(s0);
                    if (item.nq == 2 && j >= item.ks[1] && j < item.ke[1]) o[p.out_stride] = __float2bfloat16_rn(s1);
                }
            }
            kt += ntiles;
            ++qi;
        });
    }
    fence_before();
    __syncthreads();
    if (warp == 2) {
        alloc.release_allocation_lock();
        alloc.free(tmem, TMEM_COLS);
    }
#endif
}

// ---------------------------------------------------------------------------------------------
// host
// ---------------------------------------------------------------------------------------------

static CUtensorMap make_map(CUtensorMapDataType dt, int rank, void *base, const uint64_t *dims,
                            const uint64_t *strides_bytes, const uint32_t *box, CUtensorMapSwizzle swz) {
    CUtensorMap m;
    uint32_t estr[5] = {1, 1, 1, 1, 1};
    CUresult r = cuTensorMapEncodeTiled(&m, dt, rank, base, dims, strides_bytes, box, estr,
                                        CU_TENSOR_MAP_INTERLEAVE_NONE, swz, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS) {
        const char *msg = nullptr;
        cuGetErrorString(r, &msg);
        fprintf(stderr, "cuTensorMapEncodeTiled failed: %s\n", msg ? msg : "?");
        abort();
    }
    return m;
}

size_t smem_bytes() { return sizeof(Smem) + 1024; }

void launch(const IndexerParams &p, const IndexerTensors &t, int grid, cudaStream_t stream) {
    // Q: [rows * 64, 128] bytes, box (128 B, qb * 64 rows), SW128
    const uint64_t qd[2] = {(uint64_t)D, (uint64_t)p.rows * H};
    const uint64_t qs[1] = {(uint64_t)D};
    const uint32_t qb[2] = {(uint32_t)D, (uint32_t)(p.qb * H)};
    CUtensorMap tm_q = make_map(CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, t.q, qd, qs, qb, CU_TENSOR_MAP_SWIZZLE_128B);
    // Q scale words: [rows, 64] int32, box (64, qb)
    const uint64_t sd[2] = {(uint64_t)H, (uint64_t)p.rows};
    const uint64_t sst[1] = {(uint64_t)H * 4};
    const uint32_t sb[2] = {(uint32_t)H, (uint32_t)p.qb};
    CUtensorMap tm_qsf = make_map(CU_TENSOR_MAP_DATA_TYPE_INT32, 2, t.q_sf, sd, sst, sb, CU_TENSOR_MAP_SWIZZLE_NONE);
    CUtensorMap tm_k, tm_ksf;
    if (p.paged) {
        // values: [pages][64 rows][128 B] with a page stride of 64 * 132 bytes
        // (128-entry pages: [pages][128 rows][128 B], a 64-row box at row 0 or 64; scales [pages][128])
        const uint64_t page_rows = p.page_rows == 128 ? 128 : 64;
        const uint64_t kd[3] = {(uint64_t)D, page_rows, (uint64_t)t.num_pages};
        const uint64_t kst[2] = {(uint64_t)D, (uint64_t)t.page_stride};
        const uint32_t kb[3] = {(uint32_t)D, 64, 1};
        tm_k = make_map(CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, t.k, kd, kst, kb, CU_TENSOR_MAP_SWIZZLE_128B);
        const uint64_t fd[2] = {page_rows, (uint64_t)t.num_pages};
        const uint64_t fst[1] = {(uint64_t)t.page_stride};
        const uint32_t fb[2] = {64, 1};
        tm_ksf = make_map(CU_TENSOR_MAP_DATA_TYPE_INT32, 2, t.k_sf, fd, fst, fb, CU_TENSOR_MAP_SWIZZLE_NONE);
    } else {
        const uint64_t kd[2] = {(uint64_t)D, (uint64_t)t.num_keys};
        const uint64_t kst[1] = {(uint64_t)D};
        const uint32_t kb[2] = {(uint32_t)D, (uint32_t)TILE};
        tm_k = make_map(CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, t.k, kd, kst, kb, CU_TENSOR_MAP_SWIZZLE_128B);
        const uint64_t fd[1] = {(uint64_t)t.num_keys};
        const uint64_t fst[1] = {4};
        const uint32_t fb[1] = {(uint32_t)TILE};
        tm_ksf = make_map(CU_TENSOR_MAP_DATA_TYPE_INT32, 1, t.k_sf, fd, fst, fb, CU_TENSOR_MAP_SWIZZLE_NONE);
    }
    // weights: [rows, 64] bf16 (row stride w_stride elements), box (64, qb)
    const uint64_t wd[2] = {(uint64_t)H, (uint64_t)p.rows};
    const uint64_t wst[1] = {(uint64_t)p.w_stride * 2};
    const uint32_t wb[2] = {(uint32_t)H, (uint32_t)p.qb};
    CUtensorMap tm_w = make_map(CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void *)p.weights, wd, wst, wb,
                                CU_TENSOR_MAP_SWIZZLE_NONE);
    const size_t smem = smem_bytes();
    cudaFuncSetAttribute(dsa_indexer_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(grid);
    cfg.blockDim = dim3(NUM_THREADS);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    const char *pdl_env = std::getenv("ISOEXEC_SMALL_PDL");
    cfg.attrs = attr;
    cfg.numAttrs = (pdl_env && pdl_env[0] == '0') ? 0 : 1;
    cudaLaunchKernelEx(&cfg, dsa_indexer_kernel, p, tm_q, tm_qsf, tm_k, tm_ksf, tm_w);
}

}  // namespace sparse_sm100::indexer
