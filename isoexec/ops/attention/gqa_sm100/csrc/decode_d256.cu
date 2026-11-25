// SPDX-License-Identifier: BSD-3-Clause
// The HD256 DECODE leaf: the recipe's leaf program (../README.md section 1.4) for work items whose
// packed query tile holds at most 16 rows (decode, and speculative decode up to 16 / ratio tokens),
// executed SWAP-AB on ONE CTA: the KV tile is the MMA's M operand and the query rows its N operand.
//
// WHY. The general D256 leaf runs a CTA PAIR sized for 256 packed query rows per item. A decode item
// (one token x `ratio` heads of one KV head) fills 4 of them, so the tensor core, the 192 KiB of
// operands per CTA and the one-item-per-pair grid all serve 1/64 of a tile while the call is bound by
// KV bandwidth. Here a CTA streams the item's keys and values through a ring of 16 KiB TMA stages,
// several CTAs share an SM, and the MMAs are M = 128 x N = 16.
//
// WHAT DOES NOT CHANGE -- every bit of the numerical program:
//   * the segment (`seg`, anchored at key 0), the 128-key tile, the per-tile order;
//   * QK^T: every score is the fp32 contraction of one query row with one key over the 256 features
//     in SIXTEEN K16 `tcgen05.mma.kind::f16` steps in ascending feature order, the first overwriting.
//     Swap-AB computes S^T[key, row] with the SAME K16 instruction over the SAME 16 products per step
//     in the SAME order; the product a*b is exact in fp32 for bf16 inputs and which operand is "A" does
//     not enter the contraction. The relation `decode == general leaf` gates this bitwise;
//   * PV: every output element accumulates the tile's 128 keys in EIGHT K16 steps in ascending key
//     order into the rescaled fp32 accumulator, the segment's first step overwriting;
//   * the softmax: numerics.cuh's instructions in the general leaf's order -- masked scores to -inf
//     before the max, the row max seeded with the running max, the lazy rescale at
//     RESCALE_THRESHOLD, the affine + two exp2 implementations by column pair, P to bf16, the two
//     packed row-sum chains seeded `L * alpha` in ascending column order. One WARP owns a row here
//     (the general leaf: one thread); the row sums still run as the two sequential chains (the
//     lanes hand each other the next pair), and the row max is a butterfly of `fmaxf`, which equals
//     the four seeded chains for any result but +-0 and NaN -- there the chains run verbatim;
//   * the rescale `O *= alpha` exactly when `alpha != 1` on a tile after the first, `mul.rn.ftz`;
//   * the published (O, L, M), the root, the lse, and the canonical-tree combine.
//
// Execution only, and free: the swap, the TMEM placement (S^T and O^T, lanes = keys / features), the
// stage ring, the warp roles, the grid (one CTA per (request, KV head) walks every `split`-th
// segment), the partial-slot layout and the combine's grid.
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <cstdint>
#include <algorithm>
#include <tuple>

#include "params.h"
#include "numerics.cuh"
#include "tmem.cuh"

namespace gqa_sm100 {
namespace decode256 {

static constexpr int DIM = 256;
static constexpr int ROWS = DECODE_ROWS;        // packed query rows per item: the MMA's N
static constexpr int STAGE_BYTES = 16384;       // one ring stage
static constexpr int STAGES = 5;                // ring depth: two CTAs per SM
static constexpr int THREADS = 192;             // producer warp, MMA warp, four math warps
static constexpr int MATH_THREAD = 64;
static constexpr int TMEM_COLUMNS = 64;         // S^T slots at 0 and 16, O^T halves at 32 and 48
static constexpr int O_COLUMN = 32;
static constexpr int SROW = 132;                // padded fp32 row of the score staging buffer
static constexpr int MAX_SLOT_CACHE = 128;      // KV tiles whose cache slot is precomputed

//: M = 128, N = 16, bf16 x bf16 -> fp32. QK: A = K (K-major), B = Q (K-major). PV: A = V^T
//: (MN-major, the transpose-A bit 15), B = P (K-major).
static constexpr uint32_t QK_IDESC = (1u << 4) | (1u << 7) | (1u << 10) | ((16u / 8) << 17)
                                   | ((128u / 16) << 24);
static constexpr uint32_t PV_IDESC = QK_IDESC | (1u << 15);

struct DecodeParams {
    int h_q;
    int h_kv;
    int seg;
    int page;                 // paged: page size in key slots; 0 for a packed cache
    int max_n;                // segments the grid covers: ceil(max_seqlen_k / seg)
    int rows;                 // partial-slot rows per (request, KV head, segment): ratio * max_q
    int64_t block_table_stride;
    float scale_log2;
    const int* __restrict__ cu_seqlens_q;
    const int* __restrict__ seqused_k;
    const int* __restrict__ block_table;
    const int* __restrict__ cu_seqlens_k;
    void* __restrict__ out;               // bf16 [T, h_q, 256]
    float* __restrict__ lse;              // [T, h_q]
    float* __restrict__ partial_o;        // [B * h_kv * max_n * rows, 256]
    float* __restrict__ partial_l;
    float* __restrict__ partial_m;
};

struct alignas(1024) DecodeShared {
    uint8_t ring[STAGES][STAGE_BYTES];
    uint8_t q[4][ROWS * 128];             // Q, four 64-feature chunks of 16 rows x 128 B
    uint8_t p[2][2][ROWS * 128];          // P, [slot][64-key chunk] of 16 rows x 128 B
    float s[ROWS * SROW];                 // S staged row-major for the row owners
    float alpha[ROWS];
    float row_l[ROWS];
    float row_m[ROWS];
    int key_slots[MAX_SLOT_CACHE];
    uint64_t full[STAGES], empty[STAGES];
    uint64_t q_full, o_free, s_full[2], p_full[2], pv_done[2];
    uint32_t tmem;
};

__device__ __forceinline__ uint64_t kmajor_desc(const void* p) {
    return uint64_t(shared_address(p) >> 4) | (uint64_t(1) << 16) | (uint64_t(64) << 32) |
           (uint64_t(1) << 46) | (uint64_t(2) << 61);
}

//: MN-major 128 B swizzle: LBO = the stride between 64-element feature atoms, SBO = 1024 B (8 keys).
__device__ __forceinline__ uint64_t mnmajor_desc(const void* p, uint32_t lbo_bytes) {
    return uint64_t(shared_address(p) >> 4) | (uint64_t(lbo_bytes >> 4) << 16) |
           (uint64_t(64) << 32) | (uint64_t(1) << 46) | (uint64_t(2) << 61);
}

__device__ __forceinline__ void mma(uint32_t d, uint64_t a, uint64_t b, uint32_t idesc, bool acc) {
    asm volatile("{ .reg .pred p; setp.ne.b32 p,%4,0;\n"
                 "tcgen05.mma.cta_group::1.kind::f16 [%0],%1,%2,%3,p; }\n"
                 :: "r"(d), "l"(a), "l"(b), "r"(idesc), "r"(int(acc)) : "memory");
}

__device__ __forceinline__ void expect_tx(uint64_t* barrier, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"
                 :: "r"(shared_address(barrier)), "r"(bytes) : "memory");
}

__device__ __forceinline__ void math_sync() {
    asm volatile("bar.sync 1,128;" ::: "memory");
}

//: `mul.rn.ftz.f32x2` with a per-lane multiplier: the rescale's instruction, on two query rows.
__device__ __forceinline__ FloatPair multiply_lanes(FloatPair a, FloatPair b) {
    FloatPair c;
    asm("{ .reg .b64 x,y,z; mov.b64 x,{%2,%3}; mov.b64 y,{%4,%5}; "
        "mul.rn.ftz.f32x2 z,x,y; mov.b64 {%0,%1},z; }"
        : "=f"(c.lo), "=f"(c.hi) : "f"(a.lo), "f"(a.hi), "f"(b.lo), "f"(b.hi));
    return c;
}

//: One CTA serves the segments `seg_index, seg_index + Y, seg_index + 2Y, ...` of one (request, KV
//: head), `Y = gridDim.y`, in ascending order. Every segment is its own leaf: the accumulator, the
//: running max and the row sum RESET at its first tile, and its (O, L, M) is published (or, for a
//: single-segment row, rooted) before the next segment's first PV overwrites the accumulator. Which
//: CTA walks which segments is execution: `Y = ceil(max_seqlen_k / seg)` is one segment per CTA.
template <int RATIO>
__global__ __launch_bounds__(THREADS, 2) void gqa_decode_d256(
        const __grid_constant__ CUtensorMap mapq,
        const __grid_constant__ CUtensorMap mapk,
        const __grid_constant__ CUtensorMap mapv,
        const __grid_constant__ DecodeParams p) {
    const int unit = int(blockIdx.x);  // request * h_kv + kv_head
    const int seg_first = int(blockIdx.y);
    const int seg_step = int(gridDim.y);
    const int req = unit / p.h_kv;
    const int kv_head = unit - req * p.h_kv;
    const int q_begin = p.cu_seqlens_q[req];
    const int q_end = p.cu_seqlens_q[req + 1];
    const int used = p.seqused_k[req];
    const int q_len = q_end - q_begin;
    //: The item's widest row is its last token, whose border is `used` (host: q_len <= 16 / ratio).
    //: EARLY EXIT before any barrier: every thread reads the same three numbers.
    if (q_len <= 0 || used <= seg_first * p.seg) return;
    const int n_used = segment_count(used, p.seg);
    const int nsegs = (n_used - seg_first + seg_step - 1) / seg_step;   // >= 1
    const int seg_tiles = p.seg / KEY_TILE;                              // tiles of a full segment
    //: Every segment of this CTA but its last is full, so a CTA tile ordinal g sits in the CTA's
    //: segment g / seg_tiles at tile g % seg_tiles.
    const int last_begin = (seg_first + (nsegs - 1) * seg_step) * p.seg;
    const int total = (nsegs - 1) * seg_tiles + (min(last_begin + p.seg, used) - last_begin + KEY_TILE - 1) / KEY_TILE;
    auto key_base_of = [&](int g) {
        const int i = g / seg_tiles;
        return (seg_first + i * seg_step) * p.seg + (g - i * seg_tiles) * KEY_TILE;
    };

    //: Aligned by OFFSETTING the shared array itself, so every access stays in the shared state
    //: space (an integer round trip turned them into generic loads).
    extern __shared__ uint8_t raw_storage[];
    auto& s = *reinterpret_cast<DecodeShared*>(
        raw_storage + ((1024u - (shared_address(raw_storage) & 1023u)) & 1023u));
    const int thread = int(threadIdx.x);
    const int warp = thread >> 5, lane = thread & 31;

    if (thread == 0) {
        #pragma unroll
        for (int i = 0; i < STAGES; ++i) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" :: "r"(shared_address(&s.full[i])));
            asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" :: "r"(shared_address(&s.empty[i])));
        }
        asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" :: "r"(shared_address(&s.q_full)));
        asm volatile("mbarrier.init.shared::cta.b64 [%0],4;" :: "r"(shared_address(&s.o_free)));
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" :: "r"(shared_address(&s.s_full[i])));
            asm volatile("mbarrier.init.shared::cta.b64 [%0],4;" :: "r"(shared_address(&s.p_full[i])));
            asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" :: "r"(shared_address(&s.pv_done[i])));
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    //: The KV tiles' cache slots, so the single TMA thread never waits on a block-table load. The
    //: host sizes the grid so a CTA holds at most MAX_SLOT_CACHE tiles.
    if (warp == 0) {
        for (int g = lane; g < total; g += 32) {
            const int key = key_base_of(g);
            s.key_slots[g] = p.page > 0
                ? p.block_table[int64_t(req) * p.block_table_stride + key / p.page] * p.page + key % p.page
                : p.cu_seqlens_k[req] + key;
        }
    }
    if (warp == 1) {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0],%1;"
                     :: "r"(shared_address(&s.tmem)), "r"(TMEM_COLUMNS) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
    }
    synchronize_cta();
    const uint32_t tmem = s.tmem;

    if (warp == 0) {
        // ---------------- producer: Q, then K(0), K(1), V(0), K(2), V(1), ..., V(last) ----------
        if (lane == 0) {
            //: The Q box is (ROWS / RATIO) tokens x RATIO heads: all ROWS rows unless RATIO does not
            //: divide them (ratio 6: 12 rows), when the tile's last rows stay unloaded -- they are
            //: past every item's q_len * RATIO rows, so the row mask already excludes them.
            expect_tx(&s.q_full, 4 * (ROWS / RATIO) * RATIO * 128);
            #pragma unroll
            for (int c = 0; c < 4; ++c)
                load_tile(mapq, reinterpret_cast<uint16_t*>(s.q[c]), &s.q_full, c, kv_head * RATIO, q_begin);
            int index = 0;
            auto acquire = [&]() -> int {
                const int st = index % STAGES;
                if (index >= STAGES) bar_wait(&s.empty[st], ((index / STAGES) - 1) & 1);
                expect_tx(&s.full[st], STAGE_BYTES);
                ++index;
                return st;
            };
            auto keys = [&](int g) {
                const int ks = s.key_slots[g];
                #pragma unroll 1
                for (int c = 0; c < 4; ++c) {
                    const int st = acquire();
                    load_tile(mapk, reinterpret_cast<uint16_t*>(s.ring[st]), &s.full[st], c, kv_head, ks);
                }
            };
            auto values = [&](int g) {
                const int ks = s.key_slots[g];
                #pragma unroll 1
                for (int j = 0; j < 4; ++j) {
                    const int st = acquire();
                    #pragma unroll
                    for (int c = 0; c < 4; ++c)
                        load_tile(mapv, reinterpret_cast<uint16_t*>(s.ring[st] + c * 4096), &s.full[st],
                                  c, kv_head, ks + 32 * j);
                }
            };
            keys(0);
            #pragma unroll 1
            for (int g = 0; g < total; ++g) {
                if (g + 1 < total) keys(g + 1);
                values(g);
            }
        }
        __syncwarp();
    } else if (warp == 1) {
        // ---------------- MMA issue ----------------------------------------------------------------
        int index = 0;
        auto take = [&]() -> int {
            const int st = index % STAGES;
            bar_wait(&s.full[st], (index / STAGES) & 1);
            ++index;
            return st;
        };
        bar_wait(&s.q_full, 0);
        auto qk = [&](int u) {
            const uint32_t dst = tmem + uint32_t((u & 1) * ROWS);
            #pragma unroll 1
            for (int c = 0; c < 4; ++c) {
                const int st = take();
                asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                if (lane == 0) {
                    const uint64_t a = kmajor_desc(s.ring[st]);
                    const uint64_t b = kmajor_desc(s.q[c]);
                    #pragma unroll
                    for (int k = 0; k < 4; ++k)
                        mma(dst, a + 2 * k, b + 2 * k, QK_IDESC, (c | k) != 0);
                    commit_one(&s.empty[st]);
                }
                __syncwarp();
            }
            if (lane == 0) commit_one(&s.s_full[u & 1]);
            __syncwarp();
        };
        auto pv = [&](int g) {
            const int slot = g & 1;
            const int seg_i = g / seg_tiles;
            const bool first = g == seg_i * seg_tiles;   // the segment's first tile: overwrite
            const int key_base = key_base_of(g);
            const int valid = min(KEY_TILE, max(0, used - key_base));
            #pragma unroll 1
            for (int j = 0; j < 4; ++j) {
                const int st = take();
                //: Value rows at or past `seqused_k` are unwritten cache memory; their
                //: probabilities are exactly +0.0 but 0 * NaN is NaN, so they are zeroed here
                //: before the MMA reads them (the general leaf's publisher does the same).
                if (valid < 32 * (j + 1)) {
                    const int lo = max(0, valid - 32 * j);       // first invalid key of the stage
                    uint4* base = reinterpret_cast<uint4*>(s.ring[st]);
                    const int units = (32 - lo) * 8;           // per feature chunk, 16 B units
                    for (int i = lane; i < 4 * units; i += 32) {
                        const int c = i / units, rest = i - c * units;
                        base[c * 256 + lo * 8 + rest] = make_uint4(0, 0, 0, 0);
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                }
                if (j == 0) {
                    //: The previous segment's accumulator has been read out before this one's
                    //: first PV overwrites it.
                    if (first && seg_i > 0) bar_wait(&s.o_free, (seg_i - 1) & 1);
                    bar_wait(&s.p_full[slot], (g >> 1) & 1);
                }
                asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                if (lane == 0) {
                    #pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const uint64_t a = mnmajor_desc(s.ring[st] + h * 2 * 4096, 4096);
                        #pragma unroll
                        for (int k = 0; k < 2; ++k) {
                            const int step = 2 * j + k;   // K16 step over the tile's 128 keys
                            const uint64_t b = kmajor_desc(s.p[slot][step >> 2]) + 2 * (step & 3);
                            mma(tmem + O_COLUMN + h * ROWS, a + 128 * k, b, PV_IDESC,
                                !first || step != 0);
                        }
                    }
                    commit_one(&s.empty[st]);
                }
                __syncwarp();
            }
            if (lane == 0) commit_one(&s.pv_done[slot]);
            __syncwarp();
        };
        qk(0);
        #pragma unroll 1
        for (int g = 0; g < total; ++g) {
            if (g + 1 < total) qk(g + 1);
            pv(g);
        }
    } else {
        // ---------------- math warps: S staging, the row softmax, the rescale, the epilogues -----
        const int quadrant = warp & 3;               // the TMEM lane quadrant this warp may access
        const int lane_index = quadrant * 32 + lane; // key (S^T) or feature within a half (O^T)
        const int mw = warp - 2;                     // 0..3
        const float scale_log2 = p.scale_log2;
        //: ONE WARP PER ROW: warp mw owns rows mw, mw + 4, mw + 8, mw + 12, and lane l holds the
        //: row's columns 4l .. 4l+3 (column pairs 2l and 2l+1). Every lane keeps the row's running
        //: (M, L) -- warp-uniform. The elementwise steps (mask, affine, exp2 by column pair, bf16) are
        //: the recipe's instructions per element; the two ORDER-SENSITIVE steps are kept exact:
        //:   * the row sums run the recipe's two packed chains over the columns in ascending order
        //:     (lane order), each lane fetching the chain's next pair by shuffle;
        //:   * the row max is a butterfly of `fmaxf`, which equals the recipe's four seeded running
        //:     maxima whenever the result is neither zero nor NaN (the maximum of a NaN-free set is a
        //:     unique value, and a nonzero value has one encoding); on +-0 or NaN, where an operand
        //:     order could pick a different encoding, the four sequential chains run instead.
        float maximum[4], denominator[4];
        #pragma unroll
        for (int k = 0; k < 4; ++k) { maximum[k] = -CUDART_INF_F; denominator[k] = 0.0f; }

        #pragma unroll 1
        for (int g = 0; g < total; ++g) {
            const int slot = g & 1;
            const int seg_i = g / seg_tiles;
            const int u = g - seg_i * seg_tiles;
            const int seg_index = seg_first + seg_i * seg_step;
            const int seg_begin = seg_index * p.seg;
            if (u == 0) {
                //: RESET at every segment start.
                #pragma unroll
                for (int k = 0; k < 4; ++k) { maximum[k] = -CUDART_INF_F; denominator[k] = 0.0f; }
            }
            bar_wait(&s.s_full[slot], (g >> 1) & 1);
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
            {
                float v[16];
                load16(tmem + uint32_t(slot * ROWS), v);
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                #pragma unroll
                for (int i = 0; i < ROWS; ++i) s.s[i * SROW + lane_index] = v[i];
            }
            math_sync();
            //: P slot reuse: PV(g - 2) read it.
            if (g >= 2) bar_wait(&s.pv_done[slot], ((g - 2) >> 1) & 1);
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int row = mw + 4 * k;
                const int tok = q_begin + row / RATIO;
                const int bord = used - q_end + tok + 1;
                //: A row this segment does not reach publishes nothing here; its P column and its O
                //: column are never read, so it is skipped (identity alpha keeps it off the rescale).
                if (!(row / RATIO < q_len && bord > seg_begin)) {
                    if (lane == 0) s.alpha[row] = 1.0f;
                    continue;
                }
                const int local_limit = bord - 1 - seg_begin - u * KEY_TILE;
                const float* srow = s.s + row * SROW;
                const float4 raw = *reinterpret_cast<const float4*>(srow + 4 * lane);
                float x[4] = {raw.x, raw.y, raw.z, raw.w};
                #pragma unroll
                for (int i = 0; i < 4; ++i)
                    if (4 * lane + i > local_limit) x[i] = -CUDART_INF_F;
                const float running = maximum[k];
                float m = fmaxf(fmaxf(x[0], x[1]), fmaxf(x[2], x[3]));
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
                float next_maximum = fmaxf(running, m);
                if (next_maximum == 0.0f || next_maximum != next_maximum) {
                    // The recipe's own four seeded chains, verbatim (warp-uniform).
                    float max0 = running, max1 = running, max2 = running, max3 = running;
                    for (int jj = 0; jj < 32; ++jj) {
                        float a0 = srow[jj], a1 = srow[32 + jj], a2 = srow[64 + jj], a3 = srow[96 + jj];
                        if (jj > local_limit) a0 = -CUDART_INF_F;
                        if (32 + jj > local_limit) a1 = -CUDART_INF_F;
                        if (64 + jj > local_limit) a2 = -CUDART_INF_F;
                        if (96 + jj > local_limit) a3 = -CUDART_INF_F;
                        max0 = fmaxf(max0, a0);
                        max1 = fmaxf(max1, a1);
                        max2 = fmaxf(max2, a2);
                        max3 = fmaxf(max3, a3);
                    }
                    next_maximum = fmaxf(fmaxf(max0, max1), fmaxf(max2, max3));
                }
                const bool empty_row = next_maximum == -CUDART_INF_F;
                float safe_maximum = empty_row ? 0.0f : next_maximum;
                const float shift = __fmul_rn(running - safe_maximum, scale_log2);
                const bool keep = shift >= -RESCALE_THRESHOLD;
                if (keep) {
                    next_maximum = running;
                    safe_maximum = running;
                }
                const float alpha = keep ? 1.0f : exp2_native(shift);
                if (lane == 0) s.alpha[row] = empty_row ? 1.0f : alpha;
                const float bias = __fmul_rn(-safe_maximum, scale_log2);
                float e[4];
                uint32_t packed[2];
                #pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const FloatPair shifted = affine_pair(x[2 * h], x[2 * h + 1], scale_log2, bias);
                    const FloatPair y = emulated_exp2(2 * lane + h)
                        ? polynomial_exp2(shifted.lo, shifted.hi)
                        : FloatPair{exp2_native(shifted.lo), exp2_native(shifted.hi)};
                    e[2 * h] = y.lo;
                    e[2 * h + 1] = y.hi;
                    packed[h] = pack_bf16_pair(y.lo, y.hi);
                }
                // Keys 4l .. 4l+3: chunk l/16, sixteen-byte unit (l%16)/2, half l%2; 128 B swizzle.
                {
                    const int unit16 = ((lane & 15) >> 1) ^ (row & 7);
                    *reinterpret_cast<uint2*>(s.p[slot][lane >> 4] + row * 128 + unit16 * 16 + (lane & 1) * 8) =
                        make_uint2(packed[0], packed[1]);
                }
                FloatPair sum01 = {__fmul_rn(denominator[k], alpha), 0.0f}, sum23 = {0.0f, 0.0f};
                #pragma unroll
                for (int src = 0; src < 32; ++src) {
                    const float a = __shfl_sync(0xffffffffu, e[0], src);
                    const float b = __shfl_sync(0xffffffffu, e[1], src);
                    const float c = __shfl_sync(0xffffffffu, e[2], src);
                    const float d = __shfl_sync(0xffffffffu, e[3], src);
                    sum01 = add_pair(sum01, {a, b});
                    sum23 = add_pair(sum23, {c, d});
                }
                denominator[k] = (sum01.lo + sum01.hi) + (sum23.lo + sum23.hi);
                maximum[k] = next_maximum;
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            math_sync();
            //: The rescale, `O *= alpha` exactly where the general leaf's correction multiplies: on
            //: a tile after the segment's first, for a row whose alpha is not 1.
            if (u != 0) {
                bool any = false;
                #pragma unroll
                for (int i = 0; i < ROWS; ++i) any |= s.alpha[i] != 1.0f;
                if (any) {
                    bar_wait(&s.pv_done[(g - 1) & 1], ((g - 1) >> 1) & 1);
                    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                    #pragma unroll 1
                    for (int h = 0; h < 2; ++h) {
                        float o[16];
                        const uint32_t addr = tmem + uint32_t(O_COLUMN + h * ROWS);
                        load16(addr, o);
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        #pragma unroll
                        for (int i = 0; i < ROWS; i += 2) {
                            const float a0 = s.alpha[i], a1 = s.alpha[i + 1];
                            const FloatPair m = multiply_lanes({o[i], o[i + 1]}, {a0, a1});
                            if (a0 != 1.0f) o[i] = m.lo;
                            if (a1 != 1.0f) o[i + 1] = m.hi;
                        }
                        store16(addr, o);
                    }
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                }
            }
            asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
            __syncwarp();
            //: A RELEASE arrival: P was written through the generic proxy (each writer fenced it to
            //: the async proxy before the named barrier) and the tensor core reads it.
            if (lane == 0)
                asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _,[%0];"
                             :: "r"(shared_address(&s.p_full[slot])) : "memory");

            const bool seg_end = g + 1 == total || u + 1 == seg_tiles;
            if (!seg_end) continue;
            // ---------------- the segment's epilogue --------------------------------------------
            bar_wait(&s.pv_done[slot], (g >> 1) & 1);
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
            if (lane == 0) {
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    s.row_l[mw + 4 * k] = denominator[k];
                    s.row_m[mw + 4 * k] = maximum[k] == -CUDART_INF_F ? M_EMPTY : maximum[k];
                }
            }
            math_sync();
            #pragma unroll 1
            for (int h = 0; h < 2; ++h) {
                float o[16];
                load16(tmem + uint32_t(O_COLUMN + h * ROWS), o);
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                const int feature = h * 128 + lane_index;
                #pragma unroll
                for (int row = 0; row < ROWS; ++row) {
                    // the right neighbour's feature, so even lanes hold a (feature, feature + 1) pair
                    const float right = __shfl_down_sync(0xffffffffu, o[row], 1);
                    const int tr = row / RATIO;
                    if (tr >= q_len) continue;
                    const int tok = q_begin + tr;
                    const int bord = used - q_end + tok + 1;
                    if (bord <= seg_begin) continue;
                    const int head = kv_head * RATIO + (row % RATIO);
                    const float leaf_l = s.row_l[row];
                    if (segment_count(bord, p.seg) == 1) {
                        // The whole tree is this leaf: the one rounding happens here.
                        if ((lane & 1) == 0) {
                            float lo = o[row], hi = right;
                            multiply_pair(lo, hi, root_inverse(leaf_l));
                            auto* out = static_cast<__nv_bfloat16*>(p.out);
                            *reinterpret_cast<uint32_t*>(out + (int64_t(tok) * p.h_q + head) * DIM + feature) =
                                pack_bf16_pair(lo, hi);
                        }
                        if (h == 0 && thread == MATH_THREAD)
                            p.lse[int64_t(tok) * p.h_q + head] = root_lse(leaf_l, s.row_m[row], scale_log2);
                    } else {
                        const int64_t pslot = (int64_t(unit) * p.max_n + seg_index) * p.rows + row;
                        p.partial_o[pslot * DIM + feature] = o[row];
                        if (h == 0 && thread == MATH_THREAD) {
                            p.partial_l[pslot] = leaf_l;
                            p.partial_m[pslot] = s.row_m[row];
                        }
                    }
                }
            }
            asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
            //: Every math thread has read the row scalars and the accumulator.
            math_sync();
            if (lane == 0)
                asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _,[%0];"
                             :: "r"(shared_address(&s.o_free)) : "memory");
        }
    }

    synchronize_cta();
    if (warp == 1)
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0,%1;"
                     :: "r"(tmem), "r"(TMEM_COLUMNS) : "memory");
}
//: The canonical-tree combine over a decode row's published segments: one warp per (request, KV
//: head, row), the streaming binary counter of combine.cu (every node is one segment), the same
//: `merge_scales` / `merge_o_pair` / `merge_l` with the lower segment as the left operand, the same
//: root. Rows with a single segment were finished by the leaf.
__global__ __launch_bounds__(128) void gqa_decode_combine(const __grid_constant__ DecodeParams p,
                                                         int ratio, int batch) {
    constexpr int EPT = 8;
    constexpr int DEPTH = MAX_COMBINE_DEPTH;
    const int lane = threadIdx.x & 31;
    const int64_t warps = int64_t(batch) * p.h_kv * p.rows;
    for (int64_t w = int64_t(blockIdx.x) * 4 + (threadIdx.x >> 5); w < warps;
         w += int64_t(gridDim.x) * 4) {
        const int unit = int(w / p.rows), row = int(w - int64_t(unit) * p.rows);
        const int req = unit / p.h_kv, kv_head = unit - req * p.h_kv;
        const int q_begin = p.cu_seqlens_q[req], q_end = p.cu_seqlens_q[req + 1];
        const int tr = row / ratio;
        if (tr >= q_end - q_begin) continue;
        const int token = q_begin + tr;
        const int border = p.seqused_k[req] - q_end + token + 1;
        const int n = segment_count(border, p.seg);
        if (n <= 1) continue;
        const int head = kv_head * ratio + row % ratio;
        const int64_t slot0 = int64_t(unit) * p.max_n * p.rows + row;

        float o[DEPTH][EPT], l[DEPTH], m[DEPTH];
        int z[DEPTH];
        int depth = 0;
        #pragma unroll 1
        for (int k = 0; k < n; ++k) {
            const int64_t slot = slot0 + int64_t(k) * p.rows;
            const float* src = p.partial_o + slot * DIM + lane * EPT;
            const float4 a = *reinterpret_cast<const float4*>(src);
            const float4 b = *reinterpret_cast<const float4*>(src + 4);
            o[depth][0] = a.x; o[depth][1] = a.y; o[depth][2] = a.z; o[depth][3] = a.w;
            o[depth][4] = b.x; o[depth][5] = b.y; o[depth][6] = b.z; o[depth][7] = b.w;
            l[depth] = p.partial_l[slot];
            m[depth] = p.partial_m[slot];
            z[depth] = 1;
            ++depth;
            #pragma unroll 1
            while (depth > 1 && (z[depth - 2] == z[depth - 1] || k + 1 == n)) {
                const int a_ = depth - 2, b_ = depth - 1;
                const MergeScales ms = merge_scales(m[a_], m[b_], p.scale_log2);
                #pragma unroll
                for (int e = 0; e < EPT; e += 2)
                    merge_o_pair(o[a_][e], o[a_][e + 1], o[b_][e], o[b_][e + 1], ms.sa, ms.sb);
                l[a_] = merge_l(l[a_], l[b_], ms.sa, ms.sb);
                m[a_] = ms.m;
                z[a_] += z[b_];
                --depth;
            }
        }
        const float inv = root_inverse(l[0]);
        uint32_t words[4];
        #pragma unroll
        for (int jj = 0; jj < 4; ++jj) {
            float lo = o[0][2 * jj], hi = o[0][2 * jj + 1];
            multiply_pair(lo, hi, inv);
            words[jj] = pack_bf16_pair(lo, hi);
        }
        auto* out = static_cast<__nv_bfloat16*>(p.out);
        *reinterpret_cast<uint4*>(out + (int64_t(token) * p.h_q + head) * DIM + lane * EPT) =
            make_uint4(words[0], words[1], words[2], words[3]);
        if (lane == 0) p.lse[int64_t(token) * p.h_q + head] = root_lse(l[0], m[0], p.scale_log2);
    }
}

template <int RATIO>
cudaError_t launch_leaf(const DecodeParams& p, const CUtensorMap& mq, const CUtensorMap& mk,
                        const CUtensorMap& mv, int units, int split, cudaStream_t stream) {
    const int smem = int(sizeof(DecodeShared)) + 1024;
    const cudaError_t st = cudaFuncSetAttribute(gqa_decode_d256<RATIO>,
                                                cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (st != cudaSuccess) return st;
    gqa_decode_d256<RATIO><<<dim3(units, split, 1), THREADS, smem, stream>>>(mq, mk, mv, p);
    return cudaGetLastError();
}

cudaError_t launch_ratio(const DecodeParams& p, const CUtensorMap& mq, const CUtensorMap& mk,
                         const CUtensorMap& mv, int ratio, int units, int split, cudaStream_t stream) {
    switch (ratio) {
        case 1: return launch_leaf<1>(p, mq, mk, mv, units, split, stream);
        case 2: return launch_leaf<2>(p, mq, mk, mv, units, split, stream);
        case 3: return launch_leaf<3>(p, mq, mk, mv, units, split, stream);
        case 4: return launch_leaf<4>(p, mq, mk, mv, units, split, stream);
        case 6: return launch_leaf<6>(p, mq, mk, mv, units, split, stream);
        case 8: return launch_leaf<8>(p, mq, mk, mv, units, split, stream);
        case 16: return launch_leaf<16>(p, mq, mk, mv, units, split, stream);
        default: return cudaErrorInvalidValue;
    }
}

//: 5D bf16 tensor map over [tokens, heads, 256] with a (64, 1, head_box, rows) box; KV maps take
//: 256 B L2 promotion (execution only).
CUresult encode(void* storage, const void* input, int tokens, int heads, int rows, int head_box,
                int64_t token_stride, int64_t head_stride, bool promote) {
    const uint64_t head_bytes = uint64_t(head_stride) * 2;
    const uint64_t token_bytes = uint64_t(token_stride) * 2;
    uint64_t dims[5] = {64, uint64_t(DIM / 64), uint64_t(heads), uint64_t(tokens), 1};
    uint64_t strides[4] = {128, head_bytes, token_bytes, uint64_t(tokens) * token_bytes};
    uint32_t box[5] = {64, 1, uint32_t(head_box), uint32_t(rows), 1};
    uint32_t steps[5] = {1, 1, 1, 1, 1};
    return cuTensorMapEncodeTiled(static_cast<CUtensorMap*>(storage), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 5,
        const_cast<void*>(input), dims, strides, box, steps, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        promote ? CU_TENSOR_MAP_L2_PROMOTION_L2_256B : CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

struct MapStore {
    alignas(64) unsigned char bytes[128];
    CUtensorMap* map() { return reinterpret_cast<CUtensorMap*>(bytes); }
};

}  // namespace decode256

//: The decode leaf's half of `forward` (host.cu), which has checked the arguments, chosen this leaf
//: and allocated `out` and `lse`: the tensor maps, the partial slots, the leaf and the combine.
void decode_d256_forward(const at::Tensor& q, const at::Tensor& k_cache, const at::Tensor& v_cache,
                         const at::Tensor& cu_seqlens_q, const at::Tensor& seqused_k,
                         const c10::optional<at::Tensor>& block_table,
                         const c10::optional<at::Tensor>& cu_seqlens_k, double softmax_scale, int seg,
                         int max_seqlen_q, int max_seqlen_k, int h_kv, int page, int kv_tokens,
                         const at::Tensor& out, const at::Tensor& lse, int split_override) {
    using namespace decode256;
    const int T = int(q.size(0)), h_q = int(q.size(1));
    const int B = int(seqused_k.numel());
    const int ratio = h_q / h_kv;
    TORCH_CHECK(int(q.size(2)) == DIM && max_seqlen_q >= 1 && max_seqlen_q * ratio <= ROWS,
                "gqa_sm100: the decode leaf serves head dim 256 with max_seqlen_q * ratio <= 16");
    if (B == 0 || T == 0) return;

    const auto opts_f = at::TensorOptions().dtype(at::kFloat).device(q.device());
    const int max_n = std::max(1, (max_seqlen_k + seg - 1) / seg);
    const int rows = max_seqlen_q * ratio;
    const int64_t slots = max_n > 1 ? int64_t(B) * h_kv * max_n * rows : 0;
    auto partial_o = at::empty({slots, DIM}, opts_f);
    auto partial_l = at::empty({slots}, opts_f);
    auto partial_m = at::empty({slots}, opts_f);

    MapStore mq, mk, mv;
    const int tok_dim = block_table.has_value() ? 1 : 0;
    CUresult r = encode(mq.map(), q.data_ptr(), T, h_q, ROWS / ratio, ratio, int64_t(h_q) * DIM, DIM, false);
    TORCH_CHECK(r == CUDA_SUCCESS, "gqa_sm100: encoding the decode Q tensor map failed");
    r = encode(mk.map(), k_cache.data_ptr(), kv_tokens, h_kv, KEY_TILE, 1, k_cache.stride(tok_dim),
               k_cache.stride(-2), true);
    TORCH_CHECK(r == CUDA_SUCCESS, "gqa_sm100: encoding the decode K tensor map failed (token and head "
                                   "strides must be multiples of 8 elements)");
    r = encode(mv.map(), v_cache.data_ptr(), kv_tokens, h_kv, 32, 1, v_cache.stride(tok_dim),
               v_cache.stride(-2), true);
    TORCH_CHECK(r == CUDA_SUCCESS, "gqa_sm100: encoding the decode V tensor map failed");

    DecodeParams dp{};
    dp.h_q = h_q;
    dp.h_kv = h_kv;
    dp.seg = seg;
    dp.page = page;
    dp.max_n = max_n;
    dp.rows = rows;
    dp.block_table_stride = block_table.has_value() ? block_table->stride(0) : 0;
    dp.scale_log2 = float(softmax_scale) * 1.4426950408889634f;
    dp.cu_seqlens_q = cu_seqlens_q.data_ptr<int>();
    dp.seqused_k = seqused_k.data_ptr<int>();
    dp.block_table = block_table.has_value() ? block_table->data_ptr<int>() : nullptr;
    dp.cu_seqlens_k = cu_seqlens_k.has_value() ? cu_seqlens_k->data_ptr<int>() : nullptr;
    dp.out = out.data_ptr();
    dp.lse = lse.data_ptr<float>();
    dp.partial_o = partial_o.data_ptr<float>();
    dp.partial_l = partial_l.data_ptr<float>();
    dp.partial_m = partial_m.data_ptr<float>();

    auto stream = at::cuda::getCurrentCUDAStream();
    const int units = B * h_kv;
    //: How many CTAs share a (request, KV head)'s segments: CTA y walks segments y, y + split, ...
    //: Execution only. One segment per CTA (`split = max_n`) launches `units * max_n` CTAs, and in a
    //: decode graph captured at `max_seqlen_k = max_model_len` most of them find no segment and exit
    //: -- each still holding a CTA slot's shared memory while it reads `seqused_k`. So the split is
    //: the fewest CTAs per unit that still gives the machine about one CTA per resident slot (two
    //: per SM), and at least enough that no CTA walks more than MAX_SLOT_CACHE tiles.
    const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    const int seg_tiles = seg / KEY_TILE;
    TORCH_CHECK(seg_tiles <= MAX_SLOT_CACHE, "gqa_sm100: seg exceeds the decode leaf's tile cache");
    const int min_split = std::max(1, (max_n * seg_tiles + MAX_SLOT_CACHE - 1) / MAX_SLOT_CACHE);
    int split = split_override > 0 ? split_override : std::max(1, (2 * sms + units - 1) / units);
    split = std::min(max_n, std::max(split, min_split));
    C10_CUDA_CHECK(launch_ratio(dp, *mq.map(), *mk.map(), *mv.map(), ratio, units, split, stream));
    if (max_n > 1) {
        const int64_t warps = int64_t(units) * rows;
        const int blocks = int(std::max<int64_t>(1, std::min<int64_t>((warps + 3) / 4, int64_t(sms) * 16)));
        gqa_decode_combine<<<blocks, 128, 0, stream>>>(dp, ratio, B);
        C10_CUDA_CHECK(cudaGetLastError());
    }
}

}  // namespace gqa_sm100
