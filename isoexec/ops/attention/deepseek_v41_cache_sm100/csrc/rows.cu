// SPDX-License-Identifier: MIT
// IsoExec deepseek_v41_cache_sm100: forward row kernels of the V4.1 attention composition.
// Ownership: one warp per row, lane-contiguous elements (numerics.cuh).  No reduction ever crosses a
// row, so every output row is a pure function of its input row (batch invariance by construction).
// No atomics of any kind.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cmath>

#include "numerics.cuh"
#include "pdl.cuh"

namespace v41 {

using bf16 = __nv_bfloat16;

struct RowParams {
    const bf16 *x; int64_t x_stride;          // input rows [rows, D] (elements)
    // a 3-D input [tokens, x_group, D] (e.g. a column block of a packed projection): row r at
    // (r / x_group) * x_group_stride + (r % x_group) * x_stride; x_group == 0: row r at r * x_stride
    int64_t x_group, x_group_stride;
    int64_t rows;
    const float *w; float eps;                // RMSNorm weight (fp32 copy of the parameter)
    const float *cosv, *sinv; int64_t table_rows;   // [P, 32] rotary table (rope_dim 64)
    const int *pos; int64_t pos_div, pos_mod, pos_mul, pos_add;
    // pos(r) = (pos ? pos[r / pos_div] : (r / pos_div) % pos_mod) * pos_mul + pos_add
    bf16 *out; int64_t out_stride;            // bf16 output rows (nullable)
    uint8_t *codes; int32_t *sf;              // e4m3 codes [.., D] + the exponent as four equal ue8m0 bytes [..] (FP8_B128)
    // destination row: group_major ? (h/hpg)*(R*hpg) + t*hpg + h%hpg with t = r/heads, h = r%heads
    //                               : (r / map_a) * map_b + map_c + r % map_a
    int64_t map_a, map_b, map_c, group_major_tokens, gm_heads, gm_hpg;
    const int *row_flag;                      // nullable: skip rows whose flag is 0
    int64_t pad_heads;                        // CODES: > 0 -> also write the zero heads h + k*pad_heads < 64
    // CODES into a paged FP8 index-key pool instead of codes / sf (index_k_store): record row pdest[r]
    // (page pdest / prows, row pdest % prows; prows x 128 code bytes then prows x 4 scale bytes per page;
    // prows = 64, or 128 for vLLM's ratio-1 index-key pages), < 0 skipped: the bytes paged_index_write
    // would copy from codes / sf.
    const int64_t *pdest; uint8_t *ppool; int64_t page_bytes, prows;
};

__device__ __forceinline__ int64_t dst_row(const RowParams &p, int64_t r) {
    if (p.group_major_tokens) {
        const int64_t t = r / p.gm_heads, h = r % p.gm_heads;
        return (h / p.gm_hpg) * (p.group_major_tokens * p.gm_hpg) + t * p.gm_hpg + (h % p.gm_hpg);
    }
    return (r / p.map_a) * p.map_b + p.map_c + (r % p.map_a);
}

template <int E>
__device__ __forceinline__ void load_row(const bf16 *src, float (&v)[E]) {
    if constexpr (E % 8 == 0) {
#pragma unroll
        for (int i = 0; i < E / 8; ++i) {
            const uint4 u = reinterpret_cast<const uint4 *>(src)[i];
            const bf16 *h = reinterpret_cast<const bf16 *>(&u);
#pragma unroll
            for (int k = 0; k < 8; ++k) v[i * 8 + k] = __bfloat162float(h[k]);
        }
    } else {
        static_assert(E == 4, "lane width");
        const uint2 u = *reinterpret_cast<const uint2 *>(src);
        const bf16 *h = reinterpret_cast<const bf16 *>(&u);
#pragma unroll
        for (int k = 0; k < 4; ++k) v[k] = __bfloat162float(h[k]);
    }
}

template <int E>
__device__ __forceinline__ void store_row(bf16 *dst, const float (&v)[E]) {
    if constexpr (E % 8 == 0) {
#pragma unroll
        for (int i = 0; i < E / 8; ++i) {
            uint4 u;
            bf16 *h = reinterpret_cast<bf16 *>(&u);
#pragma unroll
            for (int k = 0; k < 8; ++k) h[k] = __float2bfloat16_rn(v[i * 8 + k]);
            reinterpret_cast<uint4 *>(dst)[i] = u;
        }
    } else {
        uint2 u;
        bf16 *h = reinterpret_cast<bf16 *>(&u);
#pragma unroll
        for (int k = 0; k < 4; ++k) h[k] = __float2bfloat16_rn(v[k]);
        *reinterpret_cast<uint2 *>(dst) = u;
    }
}

// ROPE: 0 none, 1 forward, 2 inverse.  Q: QuantKind.  CODES: emit e4m3 codes + the packed exponent word (FP8_B128).
// The RoPE output is rounded to BF16 except for Q_FP8_B32_UE8M0_F32 (the compressed-KV record quantizes the FP32
// rotation, CACHE-SPEC section 2).
//: The row program of row r (one warp, lane-contiguous elements); row_kernel and the paired index_kq launch run it.
template <int D, bool NORM, int ROPE, int Q, bool CODES>
__device__ __forceinline__ void row_body(const RowParams &p, int64_t r, int lane) {
    constexpr int E = D / 32;
    if (r >= p.rows) return;
    if (p.row_flag && !p.row_flag[r]) return;
    float v[E];
    const int64_t xoff = p.x_group ? (r / p.x_group) * p.x_group_stride + (r % p.x_group) * p.x_stride : r * p.x_stride;
    load_row<E>(p.x + xoff + lane * E, v);
    if constexpr (NORM) {
        const float inv = rms_inv<D>(row_sum_squares<E>(v), p.eps);
#pragma unroll
        for (int i = 0; i < E; ++i) v[i] = bf16r(__fmul_rn(__fmul_rn(v[i], inv), p.w[lane * E + i]));
    }
    if constexpr (ROPE != 0) {
        constexpr int TAIL = D - 64;
        if (lane * E >= TAIL) {
            const int64_t pr = p.pos ? int64_t(p.pos[r / p.pos_div]) : (r / p.pos_div) % p.pos_mod;
            const int64_t position = pr * p.pos_mul + p.pos_add;
            const float *c = p.cosv + position * 32, *s = p.sinv + position * 32;
#pragma unroll
            for (int i = 0; i < E; i += 2) {
                const int j = (lane * E + i - TAIL) / 2;
                rope_pair(v[i], v[i + 1], c[j], s[j], ROPE == 2);
                if constexpr (!QuantTraits<Q>::F32_IN) {
                    v[i] = bf16r(v[i]);
                    v[i + 1] = bf16r(v[i + 1]);
                }
            }
        }
    }
    const int64_t d = dst_row(p, r);
    if constexpr (Q != Q_NONE) {
        using T = QuantTraits<Q>;
        static_assert(T::G % E == 0, "group/lane layout");
        constexpr int GL = T::G / E;  // lanes per group
        float amax = 0.f;
#pragma unroll
        for (int i = 0; i < E; ++i) amax = fmaxf(amax, fabsf(v[i]));
#pragma unroll
        for (int m = 1; m < GL; m <<= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, m));
        uint8_t byte;
        const float scale = quant_scale<Q>(amax, byte);
        if constexpr (CODES) {
            static_assert(Q == Q_FP8_B128_UE8M0 && D == 128 && E == 4, "index code emission");
            uint32_t packed = 0;
#pragma unroll
            for (int i = 0; i < E; ++i) packed |= uint32_t(quant_code(v[i], scale)) << (8 * i);
            // one exponent per row, as vLLM's SM100 index cache stores it: four equal ue8m0 bytes
            const int32_t sfw = int32_t(uint32_t(byte) * 0x01010101u);
            if (p.pdest) {
                const int64_t pd = p.pdest[r];
                if (pd >= 0) {
                    uint8_t *pbase = p.ppool + (pd / p.prows) * p.page_bytes;
                    reinterpret_cast<uint32_t *>(pbase + (pd % p.prows) * 128)[lane] = packed;
                    if (lane == 0) *reinterpret_cast<int32_t *>(pbase + p.prows * 128 + (pd % p.prows) * 4) = sfw;
                }
            } else {
                reinterpret_cast<uint32_t *>(p.codes + d * D)[lane] = packed;
                if (lane == 0) p.sf[d] = sfw;
            }
            if (p.pad_heads) {   // zero-weight padding heads: zero codes, unit scales (exactly 0 dots)
                const int64_t h = r % p.pad_heads;
                for (int64_t hh = h + p.pad_heads; hh < 64; hh += p.pad_heads) {
                    const int64_t d2 = d + (hh - h);
                    reinterpret_cast<uint32_t *>(p.codes + d2 * D)[lane] = 0u;
                    if (lane == 0) p.sf[d2] = 0x7F7F7F7F;
                }
            }
        }
#pragma unroll
        for (int i = 0; i < E; ++i) v[i] = bf16r(__fmul_rn(quant_decode(v[i], scale), scale));
    }
    if (p.out) store_row<E>(p.out + d * p.out_stride + lane * E, v);
}

template <int D, bool NORM, int ROPE, int Q, bool CODES>
__global__ void __launch_bounds__(256) row_kernel(RowParams p) {
    pdl::begin();
    row_body<D, NORM, ROPE, Q, CODES>(p, int64_t(blockIdx.x) * 8 + (threadIdx.x >> 5), threadIdx.x & 31);
}

// The index keys of a KV-source layer (ROW_INDEX_K into the paged pool, blocks [0, kblocks)) and its index queries
// (ROW_INDEX_Q, the blocks after) in ONE launch: two independent row sets, each row by its own program.
__global__ void __launch_bounds__(256) index_kq_kernel(RowParams pk, RowParams pq, unsigned kblocks) {
    pdl::begin();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (blockIdx.x < kblocks)
        row_body<128, true, 1, Q_FP8_B128_UE8M0, true>(pk, int64_t(blockIdx.x) * 8 + warp, lane);
    else
        row_body<128, false, 1, Q_FP8_B128_UE8M0, true>(pq, int64_t(blockIdx.x - kblocks) * 8 + warp, lane);
}

// ---- compressor: complete-group pooling + bf16 + RMSNorm ------------------------------------------
// proj fp32 rows [*, 1024] (kv | score).  Row r of the output = group g of batch b, tokens
// base(b) + g*ratio + [0, ratio).  Decode mode: state [B, ratio, 1024] fp32 is updated at slot
// pos % ratio first and the group is pooled from the state when (pos + 1) % ratio == 0.
struct PoolParams {
    const float *proj; int64_t proj_stride;
    int64_t rows, groups, tokens;   // prefill: rows = B * groups; token base = b*tokens + g*ratio
    int ratio;
    const float *w; float eps;
    bf16 *out;                      // [rows, 512]
    float *state;                   // decode: [B, ratio, 1024]
    const int *pos;                 // decode: [B]
    int *flag;                      // decode: [B] 1 if a group completed
};

template <bool DECODE>
__global__ void __launch_bounds__(256) pool_norm_kernel(PoolParams p) {
    pdl::begin();
    constexpr int E = 16;
    const int lane = threadIdx.x & 31;
    const int64_t r = int64_t(blockIdx.x) * 8 + (threadIdx.x >> 5);
    if (r >= p.rows) return;
    const float *kv, *sc;
    int64_t stride;
    if constexpr (DECODE) {
        const int position = p.pos[r];
        const int slot = position % p.ratio;
        float *st = p.state + r * int64_t(p.ratio) * 1024;
        const float *src = p.proj + r * p.proj_stride;
#pragma unroll
        for (int i = 0; i < E; ++i) {
            st[slot * 1024 + lane * E + i] = src[lane * E + i];
            st[slot * 1024 + 512 + lane * E + i] = src[512 + lane * E + i];
        }
        const bool done = (position + 1) % p.ratio == 0;
        if (lane == 0) p.flag[r] = done ? 1 : 0;
        if (!done) return;
        __syncwarp();
        kv = st; sc = st + 512; stride = 1024;
    } else {
        const int64_t b = r / p.groups, g = r % p.groups;
        const float *base = p.proj + (b * p.tokens + g * p.ratio) * p.proj_stride;
        kv = base; sc = base + 512; stride = p.proj_stride;
    }
    float v[E];
#pragma unroll
    for (int i = 0; i < E; ++i) {
        const int c = lane * E + i;
        v[i] = bf16r(pool_channel(kv + c, sc + c, stride, p.ratio));
    }
    const float inv = rms_inv<512>(row_sum_squares<E>(v), p.eps);
#pragma unroll
    for (int i = 0; i < E; ++i) v[i] = bf16r(__fmul_rn(__fmul_rn(v[i], inv), p.w[lane * E + i]));
    store_row<E>(p.out + r * 512 + lane * E, v);
}

// ---- candidate blocks ------------------------------------------------------------------------------
// bm[r, j] = max over positions [8j, min(8j+8, ke)) of scores[r, :], j < nb = ceil(ke/8); the newest
// block (nb-1) is pinned to +inf.  Exact (max is order-free; finite scores).
__global__ void block_max_kernel(const bf16 *scores, int64_t s_stride, const int *ke, int64_t rows,
                                 int block, bf16 *bm, int64_t bm_stride, int *nb_out) {
    pdl::begin();
    const int64_t r = blockIdx.y;
    if (r >= rows) return;
    const int k = ke[r];
    const int nb = (k + block - 1) / block;
    if (blockIdx.x == 0 && threadIdx.x == 0) nb_out[r] = nb;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < nb; j += gridDim.x * blockDim.x) {
        float m = -INFINITY;
        if (j == nb - 1) {
            m = INFINITY;
        } else if (block == 8 && (s_stride & 7) == 0 && (reinterpret_cast<uintptr_t>(scores) & 15) == 0) {
            // the block's 8 scores as one 16-byte load (max is exact and order free: the same value)
            const uint4 u = *reinterpret_cast<const uint4 *>(scores + r * s_stride + j * 8);
            const bf16 *h = reinterpret_cast<const bf16 *>(&u);
#pragma unroll
            for (int q = 0; q < 8; ++q) m = fmaxf(m, __bfloat162float(h[q]));
        } else {
            for (int q = 0; q < block; ++q) m = fmaxf(m, __bfloat162float(scores[r * s_stride + j * block + q]));
        }
        bm[r * bm_stride + j] = __float2bfloat16_rn(m);
    }
}

// Positions of a row whose block is not among the selected blocks become -inf (inside [0, ke)).
// sel: ascending block ids, -1 padded (dsa_topk's output).  When nb <= nsel every block is selected and nothing changes.
// Grid (rows, chunks): every CTA of a row builds the row's bitmap (the same bits) and masks its own
// interleaved 8-position slices of [0, ke) -- bounded by the row's live length, not the score width.
// Each position's decision and written value are those of the one-CTA form; only ownership changed.
__global__ void candidate_mask_kernel(bf16 *scores, int64_t s_stride, const int *ke, const int *sel,
                                      int64_t sel_stride, int nsel, int block, int64_t rows, bool vec) {
    pdl::begin();
    extern __shared__ unsigned int keep[];
    const int64_t r = blockIdx.x;
    if (r >= rows) return;
    const int k = ke[r];
    const int nb = (k + block - 1) / block;
    if (nb <= nsel) return;
    const int64_t slice0 = int64_t(blockIdx.y) * blockDim.x * 8;
    if (slice0 >= k) return;                              // this CTA owns no live position of the row
    // bitmap word w = the selected ids in [32 w, 32 w + 32): sel is ascending with -1 pads at its tail (as
    // unsigned, the pads sort last), so each word is one lower_bound + a forward walk by its own thread
    // (no atomics; the same bits as OR-ing every selected id).
    const unsigned *srow = reinterpret_cast<const unsigned *>(sel + r * sel_stride);
    for (int wj = threadIdx.x; wj < (nb + 31) / 32; wj += blockDim.x) {
        const unsigned lo = unsigned(wj) * 32u;
        int a = 0, z = nsel;
        while (a < z) {
            const int m = (a + z) >> 1;
            if (srow[m] < lo) a = m + 1; else z = m;
        }
        unsigned word = 0;
        for (int i = a; i < nsel; ++i) {
            const unsigned b = srow[i];
            if (b >= lo + 32u || b >= unsigned(nb)) break;
            word |= 1u << (b - lo);
        }
        keep[wj] = word;
    }
    __syncthreads();
    const bf16 ninf = __float2bfloat16_rn(-INFINITY);
    bf16 *row = scores + r * s_stride;
    const int64_t step = int64_t(gridDim.y) * blockDim.x * 8;
    for (int64_t q0 = slice0 + int64_t(threadIdx.x) * 8; q0 < k; q0 += step) {
        if (vec && block == 8 && q0 + 8 <= k) {
            // one whole candidate block (q0 is a multiple of 8): kept -> untouched, else eight -inf; no read
            const int b = int(q0 / 8);
            if (!(keep[b >> 5] & (1u << (b & 31)))) {
                const uint32_t w = uint32_t(__bfloat16_as_ushort(ninf)) * 0x10001u;
                *reinterpret_cast<uint4 *>(row + q0) = make_uint4(w, w, w, w);
            }
        } else if (vec && q0 + 8 <= k) {
            uint4 v = *reinterpret_cast<const uint4 *>(row + q0);
            bf16 *e = reinterpret_cast<bf16 *>(&v);
            bool any = false;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int b = int(q0 + i) / block;
                if (!(keep[b >> 5] & (1u << (b & 31)))) { e[i] = ninf; any = true; }
            }
            if (any) *reinterpret_cast<uint4 *>(row + q0) = v;
        } else {
            for (int64_t q = q0; q < q0 + 8 && q < k; ++q) {
                const int b = int(q) / block;
                if (!(keep[b >> 5] & (1u << (b & 31)))) row[q] = ninf;
            }
        }
    }
}

// ---- attention lists --------------------------------------------------------------------------------
// Stable removal of internal negative pads for the backward reverse index.
// One warp owns a row; valid IDs already ascend across the window/compressed
// regions. No atomics, sorting, or changes to the forward's fixed-slot layout.
__global__ void compact_lists_kernel(const int *input, int *output, int64_t rows, int K) {
    pdl::begin();
    const int64_t row = int64_t(blockIdx.x) * (blockDim.x / 32) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= rows) return;
    const int *src = input + row * K;
    int *dst = output + row * K;
    for (int i = lane; i < K; i += 32) dst[i] = -1;
    __syncwarp();
    int count = 0;
    for (int base = 0; base < K; base += 32) {
        const int p = base + lane;
        const int id = p < K ? src[p] : -1;
        const unsigned mask = __ballot_sync(0xffffffffu, id >= 0);
        if (id >= 0) dst[count + __popc(mask & ((1u << lane) - 1))] = id;
        count += __popc(mask);
    }
}

// Row r: [window ids ascending | compressed ids ascending | -1 ...], width K.
//   prefill (ring == 0): b = r / T, t = r % T (or pos[r]); window = b*S + [max(f, t-W+1), t];
//                        compressed = b*S + comp_off + c
//   decode  (ring == 1): window = b*S + win_off + ((p-n+1) % W) + i (doubled ring), n = min(p+1, W)
//                        compressed = b*S + comp_off + c
// Optional per-row int32 operands (prefill only):
//   floor[r]: absolute position f >= 0 below which the window holds no key (the CED decoder segment start): n =
//             min(p + 1 - f, W), the window positions [p - n + 1, p] = [max(f, p - W + 1), p]; absent: f = 0, the
//             same integers as before the operand existed;
//   wend[r]:  the row of position p's own window key (the window rows are [wend - n + 1, wend]: positions
//             [p - n + 1, p] at consecutive rows); absent: b*S + win_off + p;
//   cbase[r]: the row of compressed entry 0 of the row's sequence; absent: b*S + comp_off.
__global__ void build_lists_kernel(const int *topk, int64_t topk_stride, int ktopk, const int *pos,
                                   int64_t rows, int64_t T, int64_t S, int64_t win_off, int64_t comp_off,
                                   int W, int mode, int *lists, int K, const int *floor, const int *wend,
                                   const int *cbase) {
    pdl::begin();
    // mode bit 0: doubled ring window; bit 1: fixed slots (window in [0, W) with its pads in place, the
    // compressed ids from slot W) -- the layout of the dual-rail recipe's main + extra lists
    const int64_t r = int64_t(blockIdx.x) * (blockDim.x / 32) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (r >= rows) return;
    const int64_t b = r / T;
    const int64_t p = pos ? pos[r] : r % T;
    const int64_t f = floor && floor[r] > 0 ? floor[r] : 0;
    const int64_t span = p + 1 - f > 0 ? p + 1 - f : 0;
    const int n = int(span < W ? span : W);
    int64_t wbase;
    const bool ring = mode & 1, fixed = mode & 2;
    if (ring) wbase = b * S + win_off + ((p - n + 1) % W);
    else if (wend) wbase = int64_t(wend[r]) - n + 1;
    else wbase = b * S + win_off + (p - n + 1);
    const int64_t cb = cbase ? int64_t(cbase[r]) : b * S + comp_off;
    int *out = lists + r * K;
    for (int i = lane; i < n; i += 32) out[i] = int(wbase + i);
    const int c0 = fixed ? W : n;
    if (fixed)
        for (int i = n + lane; i < W; i += 32) out[i] = -1;
    // compressed: valid prefix of the ascending, -1 padded top-k row
    int count = 0;
    for (int i0 = 0; i0 < ktopk; i0 += 32) {
        const int i = i0 + lane;
        const int id = i < ktopk ? topk[r * topk_stride + i] : -1;
        const unsigned m = __ballot_sync(0xffffffffu, id >= 0);
        if (id >= 0) out[c0 + count + __popc(m & ((1u << lane) - 1))] = int(cb + id);
        count += __popc(m);
    }
    for (int i = c0 + count + lane; i < K; i += 32) out[i] = -1;
}

// ---- cache writes: row r of src (n_words int32) goes to dst + idx[j][r] * dst_stride_words for each
// index set j with idx >= 0.  Destinations are unique by construction (positions), plain stores.
__global__ void scatter_rows_kernel(const int32_t *src, int64_t rows, int64_t n_words, int32_t *dst,
                                    int64_t dst_stride_words, const int64_t *idx, int sets, const int *flag) {
    pdl::begin();
    const int64_t r = blockIdx.x;
    if (r >= rows) return;
    if (flag && !flag[r]) return;
    for (int j = 0; j < sets; ++j) {
        const int64_t d = idx[j * rows + r];
        if (d < 0) continue;
        for (int64_t w = threadIdx.x; w < n_words; w += blockDim.x) dst[d * dst_stride_words + w] = src[r * n_words + w];
    }
}

// ---- decode index preparation (one launch per layer step; replaces torch index glue) --------------
// out64 [5, B]: ring row, ring row + W, compressed row, index-key code row (128 B units), scale word
// (4 B units) of the paged FP8 index-key cache; out32 [2, B]: group-start position, compressed length ke.
__global__ void decode_prep_kernel(const int *pos, int64_t B, int64_t A, int64_t off, int W, int64_t comp_off,
                                   int ratio, int64_t pages, int64_t *out64, int *out32) {
    pdl::begin();
    const int64_t b = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (b >= B) return;
    const int64_t p = pos[b];
    const int64_t ring = b * A + off + p % W;
    out64[b] = ring;
    out64[B + b] = ring + W;
    const int64_t c = ratio ? p / ratio : 0;
    out64[2 * B + b] = b * A + comp_off + c;
    const int64_t page = b * pages + c / 64;
    out64[3 * B + b] = page * (64 * 132 / 128) + c % 64;
    out64[4 * B + b] = page * (64 * 132 / 4) + (64 * 128) / 4 + c % 64;
    out32[b] = ratio ? int(p - (ratio - 1) > 0 ? p - (ratio - 1) : 0) : 0;
    out32[B + b] = ratio ? int((p + 1) / ratio) : 0;
}

// ---- RoPE on the 64-wide tail in place (inference): 8 threads per row, 4 pairs (16 B) per thread,
// the same scalar graph as the row kernel's tail -------------------------------------------------------
__global__ void rope_tail_inplace_kernel(__nv_bfloat16 *x, int64_t rows, int64_t row_stride, int D, const float *cosv,
                                         const float *sinv, const int *pos, int64_t pos_div, int64_t pos_mod,
                                         int64_t pos_mul, int inverse) {
    pdl::begin();
    const int64_t t = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t r = t / 8;
    const int q = int(t % 8);                         // pairs 4q .. 4q+3
    if (r >= rows) return;
    const int64_t pr = pos ? int64_t(pos[r / pos_div]) : (r / pos_div) % pos_mod;
    const int64_t position = pr * pos_mul;
    uint4 *ptr = reinterpret_cast<uint4 *>(x + r * row_stride + (D - 64) + 8 * q);
    uint4 u = *ptr;
    __nv_bfloat162 *h = reinterpret_cast<__nv_bfloat162 *>(&u);
    const float4 c = *reinterpret_cast<const float4 *>(cosv + position * 32 + 4 * q);
    const float4 s = *reinterpret_cast<const float4 *>(sinv + position * 32 + 4 * q);
    const float cc[4] = {c.x, c.y, c.z, c.w}, ss[4] = {s.x, s.y, s.z, s.w};
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        float a = __bfloat162float(h[k].x), b = __bfloat162float(h[k].y);
        rope_pair(a, b, cc[k], ss[k], inverse != 0);
        h[k] = __floats2bfloat162_rn(a, b);
    }
    *ptr = u;
}

// ---- row-block permutation: dst row (m, g) <- src row (g, m), rows of n_vec 16-byte vectors ---------
__global__ void permute_rows_kernel(const uint4 *src, uint4 *dst, int64_t G, int64_t M, int64_t n_vec, int to_token_major) {
    pdl::begin();
    const int64_t row = blockIdx.x;                 // source row index (g, m) or (m, g)
    int64_t g, m;
    if (to_token_major) { g = row / M; m = row % M; } else { m = row / G; g = row % G; }
    const int64_t s = to_token_major ? (g * M + m) : (m * G + g);
    const int64_t d = to_token_major ? (m * G + g) : (g * M + m);
    for (int64_t v = threadIdx.x; v < n_vec; v += blockDim.x) dst[d * n_vec + v] = src[s * n_vec + v];
}

// ---- TP-invariant leaf tree: root of the balanced adjacent-pair tree over L (power of two) FP32 leaves
// (level 0 pairs (0,1), (2,3), ...); out = FP32 root, or its one BF16 rounding.  A rank holding an
// aligned block of leaves evaluates that block's subtree; the ranks' subtree roots then complete the
// same tree, so every TP degree dividing L computes the same expression. ---------------------------
__global__ void leaf_tree_kernel(const float *x, int L, int64_t n, float *out_f32, __nv_bfloat16 *out_bf16) {
    pdl::begin();
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) {
        float acc[16];
#pragma unroll
        for (int l = 0; l < 16; ++l) acc[l] = l < L ? x[int64_t(l) * n + i] : 0.f;
        int m = L;
#pragma unroll
        for (int lvl = 0; lvl < 4; ++lvl) {
#pragma unroll
            for (int k = 0; k < 8; ++k)
                if (2 * k + 1 < m) acc[k] = __fadd_rn(acc[2 * k], acc[2 * k + 1]);
            m = m > 1 ? m / 2 : 1;
        }
        if (out_f32) out_f32[i] = acc[0];
        else out_bf16[i] = __float2bfloat16_rn(acc[0]);
    }
}

}  // namespace v41

// ================================ host side =========================================================
using namespace v41;

namespace {

const float *opt_f(const c10::optional<torch::Tensor> &t) { return t ? t->data_ptr<float>() : nullptr; }

template <int D, bool NORM, int ROPE, int Q, bool CODES>
void launch_row(const RowParams &p) {
    if (!p.rows) return;
    const int64_t blocks = (p.rows + 7) / 8;
    TORCH_CHECK(blocks < (int64_t(1) << 31));
    pdl::launch(row_kernel<D, NORM, ROPE, Q, CODES>, dim3((unsigned)blocks), dim3(256), size_t(0), at::cuda::getCurrentCUDAStream(), p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

// The row input of a launch: bf16 [rows, D] with 16-byte aligned rows, or [tokens, G, D] (unit last stride, 16-byte aligned
// head and token strides) read as tokens * G rows in place. Addresses only: every row keeps its program.
static void set_rows_input(RowParams &p, const torch::Tensor &x, int64_t D, const char *name) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && (x.dim() == 2 || x.dim() == 3) && x.size(-1) == D &&
                x.stride(-1) == 1 && (reinterpret_cast<uintptr_t>(x.data_ptr()) & 15) == 0, name,
                ": bf16 [rows, D] or [tokens, heads, D], unit last stride, 16-byte aligned");
    p.x = reinterpret_cast<const bf16 *>(x.data_ptr());
    if (x.dim() == 2) {
        TORCH_CHECK(x.stride(0) % 8 == 0, name, ": 16-byte aligned rows");
        p.x_stride = x.stride(0);
        p.rows = x.size(0);
    } else {
        TORCH_CHECK(x.stride(0) % 8 == 0 && x.stride(1) % 8 == 0, name, ": 16-byte aligned token / head strides");
        p.x_stride = x.stride(1);
        p.x_group = x.size(1);
        p.x_group_stride = x.stride(0);
        p.rows = x.size(0) * x.size(1);
    }
}

// kind selects one declared instantiation (ops.py ROW_KINDS).
void row_op(int64_t kind, const torch::Tensor &x, int64_t D,
            const c10::optional<torch::Tensor> &w, double eps,
            const c10::optional<torch::Tensor> &cosv, const c10::optional<torch::Tensor> &sinv,
            const c10::optional<torch::Tensor> &pos, int64_t pos_div, int64_t pos_mod, int64_t pos_mul,
            int64_t pos_add,
            const c10::optional<torch::Tensor> &out, const c10::optional<torch::Tensor> &codes,
            const c10::optional<torch::Tensor> &sf,
            int64_t map_a, int64_t map_b, int64_t map_c, int64_t group_major_tokens, int64_t gm_heads, int64_t gm_hpg,
            const c10::optional<torch::Tensor> &row_flag, int64_t pad_heads) {
    const c10::cuda::CUDAGuard guard(x.device());
    RowParams p{};
    set_rows_input(p, x, D, "x");
    p.w = opt_f(w);
    p.eps = float(eps);
    p.cosv = opt_f(cosv);
    p.sinv = opt_f(sinv);
    p.pos = pos ? pos->data_ptr<int>() : nullptr;
    p.pos_div = pos_div; p.pos_mod = pos_mod; p.pos_mul = pos_mul; p.pos_add = pos_add;
    p.out = out ? reinterpret_cast<bf16 *>(out->data_ptr()) : nullptr;
    p.out_stride = out ? out->stride(0) : 0;
    if (out) TORCH_CHECK(out->scalar_type() == at::kBFloat16 && out->dim() == 2 && out->size(1) == D &&
                         out->stride(1) == 1 && out->stride(0) % 8 == 0 && out->is_cuda());
    p.codes = codes ? reinterpret_cast<uint8_t *>(codes->data_ptr()) : nullptr;
    p.sf = sf ? sf->data_ptr<int32_t>() : nullptr;
    p.map_a = map_a; p.map_b = map_b; p.map_c = map_c; p.group_major_tokens = group_major_tokens; p.gm_heads = gm_heads; p.gm_hpg = gm_hpg;
    TORCH_CHECK(!group_major_tokens || (gm_heads > 0 && gm_hpg > 0 && gm_heads % gm_hpg == 0));
    p.row_flag = row_flag ? row_flag->data_ptr<int>() : nullptr;
    p.pad_heads = pad_heads;
    TORCH_CHECK(pad_heads == 0 || (kind == 7 && map_a == pad_heads && map_b == 64), "padding is for index queries");
    TORCH_CHECK(map_a > 0 && pos_div > 0 && pos_mod > 0);
    switch (kind) {
        case 0: TORCH_CHECK(D == 1280 && p.w); launch_row<1280, true, 0, Q_NONE, false>(p); break;          // q_norm
        case 1: TORCH_CHECK(D == 512 && p.w && p.cosv); launch_row<512, true, 1, Q_FP8_B32_UE8M0, false>(p); break;  // window kv
        case 2: TORCH_CHECK(D == 512 && p.cosv); launch_row<512, false, 1, Q_NONE, false>(p); break;       // q rope
        case 3: TORCH_CHECK(D == 512 && p.cosv); launch_row<512, false, 2, Q_NONE, false>(p); break;       // inverse output rope
        case 4: TORCH_CHECK(D == 512 && p.w); launch_row<512, true, 0, Q_NONE, false>(p); break;           // ratio-1 latent norm
        case 5: TORCH_CHECK(D == 512 && p.cosv); launch_row<512, false, 1, Q_FP8_B32_UE8M0_F32, false>(p); break;  // compressed kv
        case 6: TORCH_CHECK(D == 128 && p.w && p.cosv && p.codes && p.sf);
                launch_row<128, true, 1, Q_FP8_B128_UE8M0, true>(p); break;                                  // index k
        case 7: TORCH_CHECK(D == 128 && p.cosv && p.codes && p.sf);
                launch_row<128, false, 1, Q_FP8_B128_UE8M0, true>(p); break;                                 // index q
        case 8: TORCH_CHECK(D == 512 && p.cosv); launch_row<512, true, 1, Q_NONE, false>(p); break;        // norm+rope (no quant; qualification)
        case 9: TORCH_CHECK(D == 128 && p.w); launch_row<128, true, 0, Q_NONE, false>(p); break;           // k_norm alone (qualification)
        default: TORCH_CHECK(false, "unknown row kind");
    }
}

//: ROW_INDEX_K (kind 6: RMSNorm(128) -> RoPE at pos -> FP8 E4M3, one UE8M0 exponent per key) written straight
//: into the paged index-key pool at record rows ``dest`` (-1: skipped): one launch for row_op kind 6 +
//: paged_index_write, the same bytes.
void index_k_store(const torch::Tensor &x, const torch::Tensor &w, double eps, const torch::Tensor &cosv,
                   const torch::Tensor &sinv, const torch::Tensor &pos, const torch::Tensor &dest, torch::Tensor &pool) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.size(1) == 128 &&
                x.stride(1) == 1 && x.stride(0) % 8 == 0 && (reinterpret_cast<uintptr_t>(x.data_ptr()) & 15) == 0);
    TORCH_CHECK(w.scalar_type() == at::kFloat && w.numel() == 128);
    TORCH_CHECK(pos.scalar_type() == at::kInt && pos.is_contiguous() && pos.numel() == x.size(0));
    TORCH_CHECK(dest.scalar_type() == at::kLong && dest.is_contiguous() && dest.numel() == x.size(0));
    // [pages, 64, 1, 132], or [pages, 128, 1, 132] (vLLM's ratio-1 index-key pages, dest = page * 128 + row):
    // entries per page and the page stride are address parameters only
    TORCH_CHECK(pool.scalar_type() == at::kByte && pool.dim() == 4 && (pool.size(1) == 64 || pool.size(1) == 128) &&
                pool.size(3) == 132 && pool.stride(1) == 132 && pool.stride(3) == 1 &&
                pool.stride(0) >= pool.size(1) * 132 && pool.stride(0) % 16 == 0,
                "paged index keys: uint8 [pages, 64 or 128, 1, 132]");
    const c10::cuda::CUDAGuard guard(x.device());
    RowParams p{};
    p.x = reinterpret_cast<const bf16 *>(x.data_ptr());
    p.x_stride = x.stride(0);
    p.rows = x.size(0);
    p.w = w.data_ptr<float>();
    p.eps = float(eps);
    p.cosv = cosv.data_ptr<float>();
    p.sinv = sinv.data_ptr<float>();
    p.pos = pos.data_ptr<int>();
    p.pos_div = 1; p.pos_mod = int64_t(1) << 40; p.pos_mul = 1; p.pos_add = 0;
    p.map_a = 1; p.map_b = 1; p.map_c = 0;
    p.pdest = dest.data_ptr<int64_t>();
    p.ppool = pool.data_ptr<uint8_t>();
    p.page_bytes = pool.stride(0);
    p.prows = pool.size(1);
    launch_row<128, true, 1, Q_FP8_B128_UE8M0, true>(p);
}

//: index_k_store (the layer's index keys into the paged pool) and row_op kind 7 (ROW_INDEX_Q: the index queries'
//: RoPE + FP8 b128 codes, rows of the index heads or padded to 64) in ONE launch. The two row sets are independent and each row keeps its
//: own program, so every byte is the separate launches'.
void index_kq_store(const torch::Tensor &k, const torch::Tensor &w, double eps, const torch::Tensor &cosv,
                    const torch::Tensor &sinv, const torch::Tensor &kpos, const torch::Tensor &dest, torch::Tensor &pool,
                    const torch::Tensor &q, const torch::Tensor &qpos, int64_t qdiv, torch::Tensor &codes,
                    torch::Tensor &sf, int64_t heads, int64_t pad_heads) {
    TORCH_CHECK(k.is_cuda() && k.scalar_type() == at::kBFloat16 && k.dim() == 2 && k.size(1) == 128 &&
                k.stride(1) == 1 && k.stride(0) % 8 == 0 && (reinterpret_cast<uintptr_t>(k.data_ptr()) & 15) == 0);
    TORCH_CHECK(w.scalar_type() == at::kFloat && w.numel() == 128);
    TORCH_CHECK(kpos.scalar_type() == at::kInt && kpos.is_contiguous() && kpos.numel() == k.size(0));
    TORCH_CHECK(dest.scalar_type() == at::kLong && dest.is_contiguous() && dest.numel() == k.size(0));
    // [pages, 64, 1, 132], or [pages, 128, 1, 132] (vLLM's ratio-1 index-key pages, dest = page * 128 + row):
    // entries per page and the page stride are address parameters only
    TORCH_CHECK(pool.scalar_type() == at::kByte && pool.dim() == 4 && (pool.size(1) == 64 || pool.size(1) == 128) &&
                pool.size(3) == 132 && pool.stride(1) == 132 && pool.stride(3) == 1 &&
                pool.stride(0) >= pool.size(1) * 132 && pool.stride(0) % 16 == 0,
                "paged index keys: uint8 [pages, 64 or 128, 1, 132]");
    TORCH_CHECK(qpos.scalar_type() == at::kInt && qpos.is_contiguous() && qdiv > 0);
    RowParams pq{};                     // row_op(kind 7, q, 128, table, Pos(pos=qpos, div=qdiv), codes, sf, dst=(heads, qheads, 0))
    set_rows_input(pq, q, 128, "q");    // [rows, 128] or [tokens, heads, 128] (a column block of a packed projection)
    TORCH_CHECK(heads > 0 && heads <= 64 && pq.rows % heads == 0);
    // codes / sf rows of qheads heads: the index heads themselves (qheads == heads) or padded to 64 (the FP32-tree
    // fold, spec.index_q_heads), the padding written as zero heads only then
    const int64_t qrows = pq.rows / heads;
    const int64_t qheads = qrows ? codes.numel() / (qrows * 128) : heads;
    TORCH_CHECK(qheads == heads || qheads == 64, "codes: [rows, heads or 64, 128]");
    TORCH_CHECK(pad_heads == 0 || (pad_heads == heads && qheads == 64), "padding is for index queries padded to 64");
    TORCH_CHECK(codes.is_contiguous() && codes.element_size() == 1 && codes.numel() == qrows * qheads * 128);
    TORCH_CHECK(sf.scalar_type() == at::kInt && sf.is_contiguous() && sf.numel() == qrows * qheads);
    const c10::cuda::CUDAGuard guard(k.device());
    RowParams pk{};
    pk.x = reinterpret_cast<const bf16 *>(k.data_ptr());
    pk.x_stride = k.stride(0);
    pk.rows = k.size(0);
    pk.w = w.data_ptr<float>();
    pk.eps = float(eps);
    pk.cosv = cosv.data_ptr<float>();
    pk.sinv = sinv.data_ptr<float>();
    pk.pos = kpos.data_ptr<int>();
    pk.pos_div = 1; pk.pos_mod = int64_t(1) << 40; pk.pos_mul = 1; pk.pos_add = 0;
    pk.map_a = 1; pk.map_b = 1; pk.map_c = 0;
    pk.pdest = dest.data_ptr<int64_t>();
    pk.ppool = pool.data_ptr<uint8_t>();
    pk.page_bytes = pool.stride(0);
    pk.prows = pool.size(1);
    pq.cosv = cosv.data_ptr<float>();
    pq.sinv = sinv.data_ptr<float>();
    pq.pos = qpos.data_ptr<int>();
    pq.pos_div = qdiv; pq.pos_mod = int64_t(1) << 40; pq.pos_mul = 1; pq.pos_add = 0;
    pq.codes = reinterpret_cast<uint8_t *>(codes.data_ptr());
    pq.sf = sf.data_ptr<int32_t>();
    pq.map_a = heads; pq.map_b = qheads; pq.map_c = 0;
    pq.pad_heads = pad_heads;
    const unsigned kblocks = unsigned((pk.rows + 7) / 8), qblocks = unsigned((pq.rows + 7) / 8);
    if (!kblocks && !qblocks) return;
    pdl::launch(index_kq_kernel, dim3(kblocks + qblocks), dim3(256), size_t(0), at::cuda::getCurrentCUDAStream(), pk, pq,
                kblocks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void pool_norm(const torch::Tensor &proj, int64_t ratio, const torch::Tensor &w, double eps,
               torch::Tensor &out, int64_t groups, int64_t tokens,
               const c10::optional<torch::Tensor> &state, const c10::optional<torch::Tensor> &pos,
               const c10::optional<torch::Tensor> &flag) {
    TORCH_CHECK(proj.is_cuda() && proj.scalar_type() == at::kFloat && proj.dim() == 2 && proj.size(1) == 1024 &&
                proj.stride(1) == 1, "proj: fp32 [rows, 1024]");
    TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.is_contiguous() && out.size(-1) == 512);
    TORCH_CHECK(w.scalar_type() == at::kFloat && w.numel() == 512 && ratio >= 1 && ratio <= 64);
    const c10::cuda::CUDAGuard guard(proj.device());
    PoolParams p{};
    p.proj = proj.data_ptr<float>();
    p.proj_stride = proj.stride(0);
    p.ratio = int(ratio);
    p.w = w.data_ptr<float>();
    p.eps = float(eps);
    p.out = reinterpret_cast<bf16 *>(out.data_ptr());
    const bool decode = state.has_value();
    if (decode) {
        TORCH_CHECK(pos && flag && state->scalar_type() == at::kFloat && state->is_contiguous() &&
                    state->size(0) == proj.size(0) && state->size(1) == ratio && state->size(2) == 1024);
        p.rows = proj.size(0);
        p.state = state->data_ptr<float>();
        p.pos = pos->data_ptr<int>();
        p.flag = flag->data_ptr<int>();
    } else {
        p.groups = groups;
        p.tokens = tokens;
        p.rows = out.numel() / 512;
        TORCH_CHECK(groups > 0 || p.rows == 0);
    }
    if (!p.rows) return;
    const unsigned blocks = unsigned((p.rows + 7) / 8);
    if (decode) pdl::launch(pool_norm_kernel<true>, dim3(blocks), dim3(256), size_t(0), at::cuda::getCurrentCUDAStream(), p);
    else pdl::launch(pool_norm_kernel<false>, dim3(blocks), dim3(256), size_t(0), at::cuda::getCurrentCUDAStream(), p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void block_max(const torch::Tensor &scores, const torch::Tensor &ke, int64_t block, torch::Tensor &bm,
               torch::Tensor &nb) {
    TORCH_CHECK(scores.scalar_type() == at::kBFloat16 && scores.stride(1) == 1 && bm.scalar_type() == at::kBFloat16 &&
                bm.stride(1) == 1 && ke.scalar_type() == at::kInt && nb.scalar_type() == at::kInt);
    const int64_t rows = scores.size(0);
    if (!rows) return;
    const c10::cuda::CUDAGuard guard(scores.device());
    const int64_t width = bm.size(1);
    // CTAs per row: enough to cover small batches (up to 64), but about 4 waves of CTAs in total at large batches --
    // each thread strides over the row's blocks, so the split only moves work between CTAs (same max per block)
    const int64_t per_row = std::max<int64_t>(1, std::min<int64_t>(std::min<int64_t>((width + 255) / 256, 64),
                                                                   (148 * 4 + rows - 1) / rows));
    const dim3 grid{unsigned(per_row), unsigned(rows), 1u};
    pdl::launch(block_max_kernel, dim3(grid), dim3(256), size_t(0), at::cuda::getCurrentCUDAStream(), reinterpret_cast<const bf16 *>(scores.data_ptr()), scores.stride(0), ke.data_ptr<int>(), rows, int(block),
        reinterpret_cast<bf16 *>(bm.data_ptr()), bm.stride(0), nb.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void candidate_mask(torch::Tensor &scores, const torch::Tensor &ke, const torch::Tensor &sel, int64_t block,
                    int64_t max_blocks) {
    TORCH_CHECK(scores.scalar_type() == at::kBFloat16 && scores.stride(1) == 1 && sel.scalar_type() == at::kInt &&
                sel.stride(1) == 1 && ke.scalar_type() == at::kInt);
    const int64_t rows = scores.size(0);
    if (!rows) return;
    TORCH_CHECK(block > 0 && max_blocks >= (scores.size(1) + block - 1) / block,
                "candidate bitmap must cover score width");
    const int64_t bitmap_bytes = ((max_blocks + 31) / 32) * sizeof(unsigned int);
    TORCH_CHECK(bitmap_bytes <= 48 * 1024, "candidate bitmap exceeds shared memory");
    const c10::cuda::CUDAGuard guard(scores.device());
    // chunks per row: enough CTAs for small batches, capped by the width (an idle chunk exits at once)
    const int64_t per_cta = 256 * 8;
    const int64_t chunks = std::max<int64_t>(1, std::min<int64_t>((scores.size(1) + per_cta - 1) / per_cta,
                                                                  std::max<int64_t>(1, 256 / rows)));
    const bool vec = scores.stride(0) % 8 == 0 && reinterpret_cast<uintptr_t>(scores.data_ptr()) % 16 == 0;
    pdl::launch(candidate_mask_kernel, dim3(dim3(unsigned(rows), unsigned(chunks))), dim3(256), size_t(size_t(std::max<int64_t>(bitmap_bytes, sizeof(unsigned int)))), at::cuda::getCurrentCUDAStream(), reinterpret_cast<bf16 *>(scores.data_ptr()), scores.stride(0), ke.data_ptr<int>(), sel.data_ptr<int>(),
        sel.stride(0), int(sel.size(1)), int(block), rows, vec);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static const int *row_operand(const c10::optional<torch::Tensor> &t, int64_t rows, const char *name) {
    if (!t) return nullptr;
    TORCH_CHECK(t->scalar_type() == at::kInt && t->is_contiguous() && t->numel() == rows, name,
                ": int32 contiguous, one entry per row");
    return t->data_ptr<int>();
}

void build_lists(const c10::optional<torch::Tensor> &topk, const c10::optional<torch::Tensor> &pos,
                 int64_t rows, int64_t T, int64_t S, int64_t win_off, int64_t comp_off, int64_t W, int64_t mode,
                 torch::Tensor &lists, const c10::optional<torch::Tensor> &floor,
                 const c10::optional<torch::Tensor> &wend, const c10::optional<torch::Tensor> &cbase) {
    TORCH_CHECK(lists.scalar_type() == at::kInt && lists.is_contiguous() && lists.size(0) == rows);
    TORCH_CHECK(!((mode & 1) && (floor || wend)), "build_lists: the doubled-ring window takes no floor / window rows");
    if (!rows) return;
    const int *fp = row_operand(floor, rows, "floor"), *wp = row_operand(wend, rows, "wend"),
              *cp = row_operand(cbase, rows, "cbase");
    if (pos) TORCH_CHECK(pos->scalar_type() == at::kInt && pos->is_contiguous() && pos->numel() == rows);
    const int K = int(lists.size(1));
    const int *tk = nullptr;
    int64_t tks = 0;
    int ktopk = 0;
    if (topk) {
        TORCH_CHECK(topk->scalar_type() == at::kInt && topk->stride(1) == 1 && topk->size(0) == rows);
        tk = topk->data_ptr<int>();
        tks = topk->stride(0);
        ktopk = int(topk->size(1));
    }
    TORCH_CHECK(W + ktopk <= K, "list width");
    const c10::cuda::CUDAGuard guard(lists.device());
    pdl::launch(build_lists_kernel, dim3(unsigned((rows + 7) / 8)), dim3(256), size_t(0), at::cuda::getCurrentCUDAStream(), tk, tks, ktopk, pos ? pos->data_ptr<int>() : nullptr, rows, T, S, win_off, comp_off, int(W), int(mode),
        lists.data_ptr<int>(), K, fp, wp, cp);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor compact_lists(const torch::Tensor &lists) {
    TORCH_CHECK(lists.is_cuda() && lists.scalar_type() == at::kInt && lists.dim() == 2 && lists.is_contiguous());
    auto output = torch::empty_like(lists);
    const int64_t rows = lists.size(0);
    if (!rows || !lists.size(1)) return output;
    const c10::cuda::CUDAGuard guard(lists.device());
    pdl::launch(compact_lists_kernel, dim3(unsigned((rows + 7) / 8)), dim3(256), size_t(0), at::cuda::getCurrentCUDAStream(), lists.data_ptr<int>(), output.data_ptr<int>(), rows, int(lists.size(1)));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

void scatter_rows(const torch::Tensor &src, torch::Tensor &dst, int64_t dst_stride_bytes, const torch::Tensor &idx,
                  const c10::optional<torch::Tensor> &flag) {
    TORCH_CHECK(src.is_contiguous() && src.dim() == 2 && (src.size(1) * src.element_size()) % 4 == 0);
    TORCH_CHECK(dst.is_contiguous() && dst_stride_bytes % 4 == 0 && idx.scalar_type() == at::kLong &&
                idx.is_contiguous() && idx.dim() == 2 && idx.size(1) == src.size(0));
    const int64_t rows = src.size(0);
    if (!rows) return;
    const c10::cuda::CUDAGuard guard(src.device());
    pdl::launch(scatter_rows_kernel, dim3(unsigned(rows)), dim3(128), size_t(0), at::cuda::getCurrentCUDAStream(), reinterpret_cast<const int32_t *>(src.data_ptr()), rows, src.size(1) * src.element_size() / 4,
        reinterpret_cast<int32_t *>(dst.data_ptr()), dst_stride_bytes / 4, idx.data_ptr<int64_t>(), int(idx.size(0)),
        flag ? flag->data_ptr<int>() : nullptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void decode_prep(const torch::Tensor &pos, int64_t A, int64_t off, int64_t W, int64_t comp_off, int64_t ratio,
                 int64_t pages, torch::Tensor &out64, torch::Tensor &out32) {
    TORCH_CHECK(pos.scalar_type() == at::kInt && pos.is_contiguous() && out64.scalar_type() == at::kLong &&
                out64.is_contiguous() && out64.numel() == 5 * pos.numel() && out32.scalar_type() == at::kInt &&
                out32.is_contiguous() && out32.numel() == 2 * pos.numel());
    const int64_t B = pos.numel();
    if (!B) return;
    const c10::cuda::CUDAGuard guard(pos.device());
    pdl::launch(decode_prep_kernel, dim3(unsigned((B + 127) / 128)), dim3(128), size_t(0), at::cuda::getCurrentCUDAStream(), pos.data_ptr<int>(), B, A, off, int(W), comp_off, int(ratio), pages, out64.data_ptr<int64_t>(), out32.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void rope_tail_inplace(torch::Tensor &x, int64_t D, const torch::Tensor &cosv, const torch::Tensor &sinv,
                       const c10::optional<torch::Tensor> &pos, int64_t pos_div, int64_t pos_mod, int64_t pos_mul,
                       bool inverse) {
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.size(1) == D && x.stride(1) == 1 &&
                x.stride(0) % 8 == 0 && D % 8 == 0 && D >= 64 && (reinterpret_cast<uintptr_t>(x.data_ptr()) & 15) == 0,
                "x: bf16 [rows, D] with 16-byte aligned rows");
    TORCH_CHECK(cosv.is_contiguous() && sinv.is_contiguous() && cosv.size(-1) == 32);
    const int64_t rows = x.size(0);
    if (!rows) return;
    const c10::cuda::CUDAGuard guard(x.device());
    const int64_t threads = rows * 8;
    pdl::launch(rope_tail_inplace_kernel, dim3(unsigned((threads + 255) / 256)), dim3(256), size_t(0), at::cuda::getCurrentCUDAStream(), reinterpret_cast<__nv_bfloat16 *>(x.data_ptr()), rows, x.stride(0), int(D), cosv.data_ptr<float>(),
        sinv.data_ptr<float>(), pos ? pos->data_ptr<int>() : nullptr, pos_div, pos_mod, pos_mul, inverse ? 1 : 0);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void permute_rows(const torch::Tensor &src, torch::Tensor &dst, int64_t G, int64_t M, bool to_token_major) {
    TORCH_CHECK(src.is_contiguous() && dst.is_contiguous() && src.numel() == dst.numel() &&
                src.scalar_type() == dst.scalar_type());
    const int64_t bytes = src.numel() * src.element_size();
    TORCH_CHECK(G * M > 0 && bytes % (G * M * 16) == 0, "rows must be whole 16-byte vectors");
    const int64_t n_vec = bytes / (G * M * 16);
    const c10::cuda::CUDAGuard guard(src.device());
    pdl::launch(permute_rows_kernel, dim3(unsigned(G * M)), dim3(128), size_t(0), at::cuda::getCurrentCUDAStream(), reinterpret_cast<const uint4 *>(src.data_ptr()), reinterpret_cast<uint4 *>(dst.data_ptr()), G, M, n_vec,
        to_token_major ? 1 : 0);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void leaf_tree(const torch::Tensor &stacked, torch::Tensor &out) {
    TORCH_CHECK(stacked.scalar_type() == at::kFloat && stacked.is_contiguous() && out.is_contiguous() &&
                (out.scalar_type() == at::kFloat || out.scalar_type() == at::kBFloat16) &&
                stacked.numel() == stacked.size(0) * out.numel());
    const int L = int(stacked.size(0));
    TORCH_CHECK(L >= 1 && L <= 16 && (L & (L - 1)) == 0, "leaf count: a power of two <= 16");
    const int64_t n = out.numel();
    if (!n) return;
    const c10::cuda::CUDAGuard guard(out.device());
    const unsigned blocks = unsigned(std::min<int64_t>((n + 255) / 256, 1 << 16));
    pdl::launch(leaf_tree_kernel, dim3(blocks), dim3(256), size_t(0), at::cuda::getCurrentCUDAStream(), stacked.data_ptr<float>(), L, n, out.scalar_type() == at::kFloat ? out.data_ptr<float>() : nullptr,
        out.scalar_type() == at::kFloat ? nullptr : reinterpret_cast<__nv_bfloat16 *>(out.data_ptr()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// backward kernels live in rows_bwd.cu
void row_bwd(int64_t kind, const torch::Tensor &x, const torch::Tensor &dy, int64_t D,
             const c10::optional<torch::Tensor> &w, double eps,
             const c10::optional<torch::Tensor> &cosv, const c10::optional<torch::Tensor> &sinv,
             const c10::optional<torch::Tensor> &pos, int64_t pos_div, int64_t pos_mod, int64_t pos_mul,
             int64_t pos_add, torch::Tensor &dx, const c10::optional<torch::Tensor> &dw_rows,
             int64_t map_a, int64_t map_b, int64_t map_c, int64_t group_major_tokens, int64_t gm_heads, int64_t gm_hpg);
void segment_sum(const torch::Tensor &rows, int64_t segment, torch::Tensor &out);
void pool_norm_bwd(const torch::Tensor &proj, int64_t ratio, const torch::Tensor &w, double eps,
                   const torch::Tensor &dy, int64_t groups, int64_t tokens, torch::Tensor &dproj,
                   torch::Tensor &dw_rows);

void paged_quant_write(int64_t kind, const torch::Tensor &x, const c10::optional<torch::Tensor> &w, double eps,
                       const torch::Tensor &cosv, const torch::Tensor &sinv, const torch::Tensor &pos,
                       const torch::Tensor &dest, torch::Tensor &pool);
void pool_varlen(const torch::Tensor &proj, const torch::Tensor &cu_q, const torch::Tensor &p0, const torch::Tensor &state,
                 const torch::Tensor &slot, const torch::Tensor &g_seq, const torch::Tensor &g_k, int64_t ratio,
                 const torch::Tensor &w, double eps, torch::Tensor &out, torch::Tensor &valid, torch::Tensor &cidx, bool wide);
void comp_post(const torch::Tensor &proj, const torch::Tensor &cu_q, const torch::Tensor &p0, torch::Tensor &state,
               const torch::Tensor &slot, const torch::Tensor &g_seq, const torch::Tensor &g_k, int64_t ratio,
               const torch::Tensor &w, double eps, torch::Tensor &out, torch::Tensor &valid, torch::Tensor &cidx,
               torch::Tensor &gpos, const torch::Tensor &cbt, int64_t cper, torch::Tensor &dest,
               const c10::optional<torch::Tensor> &ibt, const c10::optional<torch::Tensor> &idest,
               const torch::Tensor &cosv, const torch::Tensor &sinv, torch::Tensor &pool);
void state_update(const torch::Tensor &proj, const torch::Tensor &cu_q, const torch::Tensor &p0, torch::Tensor &state,
                  const torch::Tensor &slot, int64_t ratio);
void paged_lists(const torch::Tensor &tok_seq, const torch::Tensor &tok_pos, const torch::Tensor &win_bt, int64_t W,
                 const c10::optional<torch::Tensor> &topk, const c10::optional<torch::Tensor> &comp_bt,
                 torch::Tensor &main_list, const c10::optional<torch::Tensor> &extra, int64_t win_per,
                 int64_t comp_per);

void paged_rows(const torch::Tensor &bt, const torch::Tensor &seq, const torch::Tensor &pos,
                const c10::optional<torch::Tensor> &mask, torch::Tensor &out, int64_t per);
void paged_index_write(const torch::Tensor &codes, const torch::Tensor &sf, const torch::Tensor &dest,
                       torch::Tensor &pool);
void paged_lists_masked(const torch::Tensor &tok_seq, const torch::Tensor &tok_pos, const torch::Tensor &win_bt,
                        int64_t W, const c10::optional<torch::Tensor> &topk, const c10::optional<torch::Tensor> &comp_bt,
                        torch::Tensor &main_list, const c10::optional<torch::Tensor> &extra,
                        const c10::optional<torch::Tensor> &valid, int64_t win_per, int64_t comp_per,
                        const c10::optional<torch::Tensor> &floor);
void permute_quant_g32(const torch::Tensor &y, torch::Tensor &codes, torch::Tensor &scales, int64_t G, int64_t M);
void qkv_post(const torch::Tensor &qkv, const torch::Tensor &qw, const torch::Tensor &kw, double eps,
              const torch::Tensor &cosv, const torch::Tensor &sinv, const torch::Tensor &pos,
              const torch::Tensor &dest, torch::Tensor &pool, torch::Tensor &codes, torch::Tensor &scales,
              const c10::optional<torch::Tensor> &q_out);
void qkv_post_packed(const torch::Tensor &qkv, const torch::Tensor &qw, const torch::Tensor &kw, double eps,
                     const torch::Tensor &cosv, const torch::Tensor &sinv, const torch::Tensor &pos,
                     const torch::Tensor &dest, torch::Tensor &pool, torch::Tensor &codes, torch::Tensor &scales,
                     const c10::optional<torch::Tensor> &q_out, torch::Tensor &packed);
void q_kv_post(const torch::Tensor &q, const torch::Tensor &kv, const torch::Tensor &qw, const torch::Tensor &kw,
               double eps, const torch::Tensor &cosv, const torch::Tensor &sinv, const torch::Tensor &pos,
               const torch::Tensor &dest, torch::Tensor &pool, torch::Tensor &q_out);
void q_kv_post_rows(const torch::Tensor &q, const torch::Tensor &kv, const torch::Tensor &qw, const torch::Tensor &kw,
                    double eps, const torch::Tensor &cosv, const torch::Tensor &sinv, const torch::Tensor &pos,
                    torch::Tensor &dest, torch::Tensor &pool, torch::Tensor &q_out, const torch::Tensor &win_bt,
                    const torch::Tensor &tok_seq, const c10::optional<torch::Tensor> &valid, int64_t per,
                    const c10::optional<torch::Tensor> &main_list);
void rms_quant_5120(const torch::Tensor &x, const c10::optional<torch::Tensor> &pre, const torch::Tensor &w,
                    double eps, torch::Tensor &out, const c10::optional<torch::Tensor> &codes,
                    const c10::optional<torch::Tensor> &scales);
void index_bounds(const torch::Tensor &pos, const c10::optional<torch::Tensor> &valid, int64_t ratio, torch::Tensor &ke);
void window_rows_lists(const std::vector<torch::Tensor> &slots, const std::vector<torch::Tensor> &dec,
                       const std::vector<c10::optional<torch::Tensor>> &pre, const std::vector<int64_t> &per,
                       const std::vector<int64_t> &offset, const torch::Tensor &valid, int64_t nd, int64_t npf,
                       int64_t block, torch::Tensor &rows_out, torch::Tensor &lists_out);
void decoder_window_lists(const torch::Tensor &rows_in, const torch::Tensor &lists_in, const std::vector<int64_t> &src,
                          const std::vector<int64_t> &floor, const torch::Tensor &sel, const torch::Tensor &pos,
                          const c10::optional<torch::Tensor> &fl, torch::Tensor &rows_out, torch::Tensor &lists_out);
void page_tables(const std::vector<torch::Tensor> &bts, const std::vector<int64_t> &pages,
                 const std::vector<int64_t> &mult, const std::vector<int64_t> &step, const std::vector<int64_t> &offset,
                 std::vector<torch::Tensor> &outs);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("page_tables", &page_tables);
    m.def("window_rows_lists", &window_rows_lists);
    m.def("decoder_window_lists", &decoder_window_lists);
    m.def("index_bounds", &index_bounds);
    m.def("rms_quant_5120", &rms_quant_5120);
    m.def("qkv_post", &qkv_post);
    m.def("qkv_post_packed", &qkv_post_packed);
    m.def("q_kv_post", &q_kv_post);
    m.def("q_kv_post_rows", &q_kv_post_rows, pybind11::arg("q"), pybind11::arg("kv"), pybind11::arg("qw"), pybind11::arg("kw"),
          pybind11::arg("eps"), pybind11::arg("cosv"), pybind11::arg("sinv"), pybind11::arg("pos"), pybind11::arg("dest"),
          pybind11::arg("pool"), pybind11::arg("q_out"), pybind11::arg("win_bt"), pybind11::arg("tok_seq"),
          pybind11::arg("valid"), pybind11::arg("per"), pybind11::arg("main_list") = pybind11::none());
    m.def("permute_quant_g32", &permute_quant_g32);
    m.def("paged_rows", &paged_rows, pybind11::arg("bt"), pybind11::arg("seq"), pybind11::arg("pos"), pybind11::arg("mask"), pybind11::arg("out"), pybind11::arg("per") = 64);
    m.def("paged_index_write", &paged_index_write);
    m.def("paged_lists_masked", &paged_lists_masked, pybind11::arg("tok_seq"), pybind11::arg("tok_pos"), pybind11::arg("win_bt"), pybind11::arg("W"), pybind11::arg("topk"),
          pybind11::arg("comp_bt"), pybind11::arg("main_list"), pybind11::arg("extra"), pybind11::arg("valid"), pybind11::arg("win_per") = 64, pybind11::arg("comp_per") = 64,
          pybind11::arg("floor") = pybind11::none());
    m.def("paged_quant_write", &paged_quant_write);
    m.def("pool_varlen", &pool_varlen, pybind11::arg("proj"), pybind11::arg("cu_q"), pybind11::arg("p0"), pybind11::arg("state"), pybind11::arg("slot"), pybind11::arg("g_seq"), pybind11::arg("g_k"), pybind11::arg("ratio"), pybind11::arg("w"), pybind11::arg("eps"), pybind11::arg("out"), pybind11::arg("valid"), pybind11::arg("cidx"), pybind11::arg("wide") = false);
    m.def("state_update", &state_update);
    m.def("comp_post", &comp_post);
    m.def("index_k_store", &index_k_store);
    m.def("index_kq_store", &index_kq_store);
    m.def("paged_lists", &paged_lists, pybind11::arg("tok_seq"), pybind11::arg("tok_pos"), pybind11::arg("win_bt"), pybind11::arg("W"), pybind11::arg("topk"), pybind11::arg("comp_bt"),
          pybind11::arg("main_list"), pybind11::arg("extra"), pybind11::arg("win_per") = 64, pybind11::arg("comp_per") = 64);
    m.def("row_op", &row_op);
    m.def("pool_norm", &pool_norm);
    m.def("block_max", &block_max);
    m.def("candidate_mask", &candidate_mask);
    m.def("build_lists", &build_lists, pybind11::arg("topk"), pybind11::arg("pos"), pybind11::arg("rows"), pybind11::arg("T"),
          pybind11::arg("S"), pybind11::arg("win_off"), pybind11::arg("comp_off"), pybind11::arg("W"), pybind11::arg("mode"),
          pybind11::arg("lists"), pybind11::arg("floor") = pybind11::none(), pybind11::arg("wend") = pybind11::none(),
          pybind11::arg("cbase") = pybind11::none());
    m.def("compact_lists", &compact_lists);
    m.def("scatter_rows", &scatter_rows);
    m.def("decode_prep", &decode_prep);
    m.def("rope_tail_inplace", &rope_tail_inplace);
    m.def("permute_rows", &permute_rows);
    m.def("leaf_tree", &leaf_tree);
    m.def("row_bwd", &row_bwd);
    m.def("segment_sum", &segment_sum);
    m.def("pool_norm_bwd", &pool_norm_bwd);
}
