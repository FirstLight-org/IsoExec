// SPDX-License-Identifier: MIT
// IsoExec deepseek_v41_cache_sm100: paged-cache kernels (engine hosting: vLLM / SGLang block tables).
//
// * paged_quant_write: the window-KV / compressed-KV row programs of rows.cu (same scalar graph,
//   numerics.cuh), emitting the V4.1 cache record the dual-rail sink leaf reads for both pools: FP8 528 B
//   tokens (512 e4m3 codes, 16 UE8M0 bytes, block 32; port/CACHE-SPEC.md), at a physical row given per token.
//   The window record quantizes the BF16-rounded RoPE output, the compressed record the FP32 one.
// * pool_varlen: complete compression groups of a packed varlen batch (prefill chunks and decode rows
//   mixed), each group from [FP32 partial-group state | new FP32 projections] in ascending token
//   order with the pool.cu scalar graph, then BF16 and RMSNorm; and state_update (the new tail).
// * paged_lists: the dual-rail main list (W window slots, -1 pads in place) and extra list (top-k
//   compressed ids), as physical cache rows through the block tables.
// One warp per row; no atomics; every output row is a function of its own inputs.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cmath>

#include "norm_segments.cuh"
#include "numerics.cuh"
#include "pdl.cuh"
#include <cstdlib>

namespace v41 {

using bf16 = __nv_bfloat16;

template <bool NORM, int Q>
__device__ __forceinline__ void paged_quant_write_row(
        const bf16 *x, int64_t x_stride, int64_t r, const float *w, float eps, const float *cosv,
        const float *sinv, const int *pos, const int64_t *dest, uint8_t *pool, int lane);

template <bool NORM, int Q>
__global__ void __launch_bounds__(256) paged_quant_write_kernel(
        const bf16 *x, int64_t x_stride, int64_t rows, const float *w, float eps, const float *cosv,
        const float *sinv, const int *pos, const int64_t *dest, uint8_t *pool) {
    pdl::begin();
    const int lane = threadIdx.x & 31;
    const int64_t r = int64_t(blockIdx.x) * 8 + (threadIdx.x >> 5);
    if (r >= rows) return;
    paged_quant_write_row<NORM, Q>(x, x_stride, r, w, eps, cosv, sinv, pos, dest, pool, lane);
}

// one row of paged_quant_write (shared by the fused Q/KV post-projection kernel below)
template <bool NORM, int Q>
__device__ __forceinline__ void paged_quant_write_row(
        const bf16 *x, int64_t x_stride, int64_t r, const float *w, float eps, const float *cosv,
        const float *sinv, const int *pos, const int64_t *dest, uint8_t *pool, int lane) {
    constexpr int E = 16;                             // D = 512
    const int64_t d = dest[r];
    if (d < 0) return;
    float v[E];
    {
        const uint4 *src = reinterpret_cast<const uint4 *>(x + r * x_stride + lane * E);
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const uint4 u = src[i];
            const bf16 *hh = reinterpret_cast<const bf16 *>(&u);
#pragma unroll
            for (int k = 0; k < 8; ++k) v[i * 8 + k] = __bfloat162float(hh[k]);
        }
    }
    if constexpr (NORM) {
        const float inv = rms_inv<512>(row_sum_squares<E>(v), eps);
#pragma unroll
        for (int i = 0; i < E; ++i) v[i] = bf16r(__fmul_rn(__fmul_rn(v[i], inv), w[lane * E + i]));
    }
    if (lane * E >= 448) {
        const int64_t position = pos[r];
        const float *c = cosv + position * 32, *s = sinv + position * 32;
#pragma unroll
        for (int i = 0; i < E; i += 2) {
            const int j = (lane * E + i - 448) / 2;
            rope_pair(v[i], v[i + 1], c[j], s[j], false);
            if constexpr (!QuantTraits<Q>::F32_IN) {
                v[i] = bf16r(v[i]);
                v[i + 1] = bf16r(v[i + 1]);
            }
        }
    }
    static_assert(Q == Q_FP8_B32_UE8M0 || Q == Q_FP8_B32_UE8M0_F32, "paged KV records");
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < E; ++i) amax = fmaxf(amax, fabsf(v[i]));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));     // group 32 = lanes (2g, 2g+1)
    uint8_t byte;
    const float scale = quant_scale<Q>(amax, byte);
    uint8_t *rec = pool + d * 528;
    uint32_t words[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        uint32_t wv = 0;
#pragma unroll
        for (int b = 0; b < 4; ++b) wv |= uint32_t(quant_code(v[4 * k + b], scale)) << (8 * b);
        words[k] = wv;
    }
    *reinterpret_cast<uint4 *>(rec + lane * 16) = make_uint4(words[0], words[1], words[2], words[3]);
    if (!(lane & 1)) rec[512 + lane / 2] = byte;
}

// group slot g of sequence s = g_seq[g], k = g_k[g]: tokens base + k*ratio + [0, ratio) with
// base = p0 - p0 % ratio; a token before p0 is state row (q - base), else new row cu_q[s] + q - p0.
__global__ void __launch_bounds__(256) pool_varlen_kernel(
        const float *proj, const int *cu_q, const int *p0s, const float *state, const int *slot, const int *g_seq,
        const int *g_k, int64_t groups, int ratio, const float *w, float eps, bf16 *out, int *valid, int *cidx,
        int64_t slot_stride) {
    pdl::begin();
    constexpr int E = 16;
    const int lane = threadIdx.x & 31;
    const int64_t g = int64_t(blockIdx.x) * 8 + (threadIdx.x >> 5);
    if (g >= groups) return;
    const int s = g_seq[g], k = g_k[g];
    const int p0 = p0s[s], n = cu_q[s + 1] - cu_q[s];
    const int rem = p0 % ratio, base = p0 - rem;
    const int G = (rem + n) / ratio;
    if (lane == 0) {
        valid[g] = k < G ? 1 : 0;
        cidx[g] = base / ratio + k;
    }
    if (k >= G) return;
    const float *rowp[64];
    for (int j = 0; j < ratio; ++j) {
        const int q = base + k * ratio + j;
        rowp[j] = q < p0 ? state + int64_t(slot[s]) * slot_stride + int64_t(q - base) * 1024
                         : proj + int64_t(cu_q[s] + (q - p0)) * 1024;
    }
    float v[E];
#pragma unroll
    for (int i = 0; i < E; ++i) {
        const int c = lane * E + i;
        // pool.cu scalar graph, tokens ascending
        float maximum = -INFINITY;
        for (int j = 0; j < ratio; ++j) maximum = fmaxf(maximum, rowp[j][512 + c]);
        float denominator = 0.f;
        for (int j = 0; j < ratio; ++j) denominator += expf(rowp[j][512 + c] - maximum);
        float result = 0.f;
        for (int j = 0; j < ratio; ++j) {
            const float pj = expf(rowp[j][512 + c] - maximum) / denominator;
            result += pj * rowp[j][c];
        }
        v[i] = bf16r(result);
    }
    const float inv = rms_inv<512>(row_sum_squares<E>(v), eps);
    bf16 *o = out + g * 512 + lane * E;
#pragma unroll
    for (int i = 0; i < E; ++i) o[i] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(v[i], inv), w[lane * E + i]));
}

// After the pool: the incomplete tail [base + G*ratio, p0 + n) becomes state rows [0, tail).  Rows still
// before p0 only move when a group completed (then the tail lies entirely in the new tokens).
__global__ void state_update_kernel(const float *proj, const int *cu_q, const int *p0s, float *state, const int *slot,
                                    int64_t seqs, int ratio, int64_t slot_stride) {
    pdl::begin();
    const int64_t s = blockIdx.x;
    if (s >= seqs) return;
    const int p0 = p0s[s], n = cu_q[s + 1] - cu_q[s];
    const int rem = p0 % ratio, base = p0 - rem;
    const int G = (rem + n) / ratio;
    const int nb = base + G * ratio;
    float *st = state + int64_t(slot[s]) * slot_stride;
    for (int q = (nb > p0 ? nb : p0); q < p0 + n; ++q) {
        const float *src = proj + int64_t(cu_q[s] + (q - p0)) * 1024;
        float *dst = st + int64_t(q - nb) * 1024;
        for (int c = threadIdx.x; c < 1024; c += blockDim.x) dst[c] = src[c];
    }
}

// main [R, W]: window positions [p-n+1, p] ascending (n = min(p+1, W)) as physical rows of the window
// pool through win_bt, then -1; extra [R, Kx]: valid top-k ids through comp_bt, then -1.
__global__ void paged_lists_kernel(const int *tok_seq, const int *tok_pos, int64_t rows, const int *win_bt,
                                   int64_t win_bt_stride, int W, const int *topk, int64_t topk_stride, int ktopk,
                                   const int *comp_bt, int64_t comp_bt_stride, int *main_list, int *extra, int kx,
                                   int win_per, int comp_per) {
    pdl::begin();
    const int64_t r = int64_t(blockIdx.x) * (blockDim.x / 32) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (r >= rows) return;
    const int s = tok_seq[r], p = tok_pos[r];
    const int n = p + 1 < W ? p + 1 : W;
    int *m = main_list + r * W;
    for (int i = lane; i < W; i += 32) {
        if (i < n) {
            const int q = p - n + 1 + i;
            m[i] = win_bt[int64_t(s) * win_bt_stride + q / 64] * win_per + q % 64;
        } else {
            m[i] = -1;
        }
    }
    if (!extra) return;
    int *x = extra + r * kx;
    int count = 0;
    for (int i0 = 0; i0 < ktopk; i0 += 32) {
        const int i = i0 + lane;
        const int id = i < ktopk ? topk[r * topk_stride + i] : -1;
        const unsigned bal = __ballot_sync(0xffffffffu, id >= 0);
        if (id >= 0)
            x[count + __popc(bal & ((1u << lane) - 1))] = comp_bt[int64_t(s) * comp_bt_stride + id / 64] * comp_per + id % 64;
        count += __popc(bal);
    }
    for (int i = count + lane; i < kx; i += 32) x[i] = -1;
}

// rows[r] = bt[seq[r], pos[r] / 64] * 64 + pos[r] % 64 as int64 (physical record row), or -1 where the
// optional row mask is zero (padded graph rows / incomplete compression groups never write persistent KV).
// mask: bool (1 byte) or int32 per row. One thread per row; integer index math only.
template <typename M>
__global__ void paged_rows_kernel(const int *bt, int64_t bt_stride, const int *seq, const int *pos, const M *mask,
                                  int64_t rows, int64_t *out, int per) {
    pdl::begin();
    const int64_t r = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    if (mask && !mask[r]) { out[r] = -1; return; }
    const int p = pos[r];
    out[r] = int64_t(bt[int64_t(seq[r]) * bt_stride + p / 64]) * per + p % 64;
}

// The paged FP8 index-key record of row dest[r] (page = dest / 64, row = dest % 64; records
// [pages, 64, 1, 132]: 64 x 128 code bytes then 64 x 4 scale bytes per page); dest < 0 writes nothing.
// Same bytes as the two scatter_rows calls it replaces. One warp per row.
__global__ void paged_index_write_kernel(const uint8_t *codes, const int *sf, const int64_t *dest, int64_t rows,
                                         uint8_t *pool, int64_t page_bytes) {
    pdl::begin();
    const int64_t r = int64_t(blockIdx.x) * 8 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (r >= rows) return;
    const int64_t d = dest[r];
    if (d < 0) return;
    const int64_t page = d / 64, row = d % 64;
    uint8_t *base = pool + page * page_bytes;
    reinterpret_cast<uint32_t *>(base + row * 128)[lane] = reinterpret_cast<const uint32_t *>(codes + r * 128)[lane];
    if (lane == 0) *reinterpret_cast<int *>(base + 64 * 128 + row * 4) = sf[r];
}

// paged_index_write_kernel on 128-entry pages (vLLM's ratio-1 index-key pages, [pages, 128, 1, 132]:
// 128 x 128 code bytes then 128 x 4 scale bytes per page): page = dest / 128, row = dest % 128.
// Address math only; the 64-entry kernel above is the default.
__global__ void paged_index_write128_kernel(const uint8_t *codes, const int *sf, const int64_t *dest, int64_t rows,
                                            uint8_t *pool, int64_t page_bytes) {
    pdl::begin();
    const int64_t r = int64_t(blockIdx.x) * 8 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (r >= rows) return;
    const int64_t d = dest[r];
    if (d < 0) return;
    const int64_t page = d / 128, row = d % 128;
    uint8_t *base = pool + page * page_bytes;
    reinterpret_cast<uint32_t *>(base + row * 128)[lane] = reinterpret_cast<const uint32_t *>(codes + r * 128)[lane];
    if (lane == 0) *reinterpret_cast<int *>(base + 128 * 128 + row * 4) = sf[r];
}

// paged_lists with an optional row mask: rows whose mask is zero get all -1 lists (the separate
// masked_fill of main and extra, fused). Optional floor[r] (int32, absolute position f; the CED decoder segment
// start): the main list holds the window positions [max(f, p - W + 1), p] (n = min(p + 1 - f, W) slots, then -1);
// absent: f = 0, the same integers as without the operand.
__global__ void paged_lists_masked_kernel(const int *tok_seq, const int *tok_pos, int64_t rows, const int *win_bt,
                                          int64_t win_bt_stride, int W, const int *topk, int64_t topk_stride,
                                          int ktopk, const int *comp_bt, int64_t comp_bt_stride, int *main_list,
                                          int *extra, int kx, const bool *valid, int win_per, int comp_per,
                                          const int *floor) {
    pdl::begin();
    const int64_t r = int64_t(blockIdx.x) * (blockDim.x / 32) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (r >= rows) return;
    if (valid && !valid[r]) {
        for (int i = lane; i < W; i += 32) main_list[r * W + i] = -1;
        if (extra)
            for (int i = lane; i < kx; i += 32) extra[r * kx + i] = -1;
        return;
    }
    const int s = tok_seq[r], p = tok_pos[r];
    const int f = floor && floor[r] > 0 ? floor[r] : 0;
    const int span = p + 1 - f > 0 ? p + 1 - f : 0;
    const int n = span < W ? span : W;
    int *m = main_list + r * W;
    for (int i = lane; i < W; i += 32) {
        if (i < n) {
            const int q = p - n + 1 + i;
            m[i] = win_bt[int64_t(s) * win_bt_stride + q / 64] * win_per + q % 64;
        } else {
            m[i] = -1;
        }
    }
    if (!extra) return;
    int *x = extra + r * kx;
    int count = 0;
    if (ktopk <= 32 * 16) {
        // all ids, then all block-table entries, in flight at once (same compaction, same order)
        int ids[16], rows_[16];
#pragma unroll
        for (int t = 0; t < 16; ++t) {
            const int i = t * 32 + lane;
            ids[t] = i < ktopk ? topk[r * topk_stride + i] : -1;
        }
#pragma unroll
        for (int t = 0; t < 16; ++t)
            rows_[t] = ids[t] >= 0 ? comp_bt[int64_t(s) * comp_bt_stride + ids[t] / 64] * comp_per + ids[t] % 64 : 0;
#pragma unroll
        for (int t = 0; t < 16; ++t) {
            const unsigned bal = __ballot_sync(0xffffffffu, ids[t] >= 0);
            if (ids[t] >= 0) x[count + __popc(bal & ((1u << lane) - 1))] = rows_[t];
            count += __popc(bal);
        }
    } else {
        for (int i0 = 0; i0 < ktopk; i0 += 32) {
            const int i = i0 + lane;
            const int id = i < ktopk ? topk[r * topk_stride + i] : -1;
            const unsigned bal = __ballot_sync(0xffffffffu, id >= 0);
            if (id >= 0)
                x[count + __popc(bal & ((1u << lane) - 1))] = comp_bt[int64_t(s) * comp_bt_stride + id / 64] * comp_per + id % 64;
            count += __popc(bal);
        }
    }
    for (int i = count + lane; i < kx; i += 32) x[i] = -1;
}

// wo_a output (group-major rows g*M + m of [G*M, R] BF16) -> wo_b's checkpoint FP8 activation encoding,
// token-major: codes [M, G*R] E4M3 and UE8M0 exponents [M, G*R/32]. The quantizer body is
// quant/checkpoint_sm100's quant_g32_vec4 verbatim (same thread mapping, same scalar graph), so the
// bytes equal permute_rows followed by quantize(kind="fp8", block_size=32, scale_format="ue8m0").
__global__ void permute_quant_g32_kernel(const __nv_bfloat16 *y, uint8_t *out, uint8_t *scales, int64_t G,
                                         int64_t M, int64_t R, int64_t groups) {
    pdl::begin();
    const int lane = threadIdx.x & 7;
    // every lane of a warp runs the shuffles (full-warp mask); a group past the end computes on group 0's
    // values and stores nothing
    const int64_t group_raw = int64_t(blockIdx.x) * 16 + threadIdx.x / 8;
    const bool live = group_raw < groups;
    const int64_t group = live ? group_raw : 0;
    const int64_t per_row = R / 32;
    const int64_t c = group % per_row, mg = group / per_row;      // token-major (m, g)
    const int64_t m = mg / G, g = mg % G;
    const __nv_bfloat16 *x = y + (g * M + m) * R + c * 32;
    const uint2 packed = *reinterpret_cast<const uint2 *>(x + lane * 4);
    const uint32_t words[2] = {packed.x, packed.y};
    float values[4];
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const unsigned short bits = (words[i / 2] >> (16 * (i & 1))) & 0xffff;
        values[i] = __bfloat162float(__ushort_as_bfloat16(bits));
        amax = fmaxf(amax, fabsf(values[i]));
    }
#pragma unroll
    for (int delta = 4; delta; delta >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, delta, 8));
    if (!live) return;
    amax = fmaxf(amax, 1.0e-4f);
    const float raw_scale = __fmul_rn(amax, 1.0f / 448.0f);
    const uint32_t bits = __float_as_uint(raw_scale);
    const uint32_t exponent = ((bits >> 23) & 255) + ((bits & 0x7fffff) != 0);
    const float scale = __uint_as_float(exponent << 23);
    if (lane == 0) scales[group] = exponent;
    uint32_t codes = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float value = fminf(448.0f, fmaxf(-448.0f, __fdiv_rn(values[i], scale)));
        codes |= uint32_t(__nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3)) << (8 * i);
    }
    *reinterpret_cast<uint32_t *>(out + group * 32 + lane * 4) = codes;
}

//: Window record rows computed in the kernel (q_kv_post): dest[r] = win_bt[tok_seq[r], pos / 64] * per + pos % 64,
//: -1 where valid[r] == 0 -- paged_rows' integer formula, written to dest (the same int64 buffer _rows fills).
//: main != nullptr: also the row's dual-rail main list (W slots: window positions [p - n + 1, p] ascending as cache
//: rows, n = min(p + 1, W), then -1; all -1 for a masked row) -- paged_lists_masked's main list, by the whole warp.
struct WinRows {
    const int *bt; int64_t bt_stride; const int *seq; const bool *valid; int per;
    int *main; int W;
};

__device__ __forceinline__ void win_row(const WinRows &wr, const int *pos, int64_t *dest, int64_t r) {
    if (!wr.bt) return;
    if (wr.valid && !wr.valid[r]) { dest[r] = -1; return; }
    const int p = pos[r];
    dest[r] = int64_t(wr.bt[int64_t(wr.seq[r]) * wr.bt_stride + p / 64]) * wr.per + p % 64;
}

__device__ __forceinline__ void win_main_list(const WinRows &wr, const int *pos, int64_t r, int lane) {
    if (!wr.main) return;
    int *m = wr.main + r * wr.W;
    if (wr.valid && !wr.valid[r]) {
        for (int i = lane; i < wr.W; i += 32) m[i] = -1;
        return;
    }
    const int s = wr.seq[r], p = pos[r];
    const int n = p + 1 < wr.W ? p + 1 : wr.W;
    for (int i = lane; i < wr.W; i += 32) {
        if (i < n) {
            const int q = p - n + 1 + i;
            m[i] = wr.bt[int64_t(s) * wr.bt_stride + q / 64] * wr.per + q % 64;
        } else {
            m[i] = -1;
        }
    }
}

// One 8-element BF16 load of a wq_a row, widened to FP32.
__device__ __forceinline__ void load_q8(const bf16 *qkv, int64_t off, float (&v)[8]) {
    const uint4 u = *reinterpret_cast<const uint4 *>(qkv + off);
    const bf16 *h = reinterpret_cast<const bf16 *>(&u);
#pragma unroll
    for (int k = 0; k < 8; ++k) v[k] = __bfloat162float(h[k]);
}

// Fused post-projection of the stacked [wq_a | wkv] output (one launch for what were three):
//  blockIdx.y == 0: q_norm (row_kernel<1280, NORM> scalar graph: same lane layout E = 40, same
//    row_sum_squares / rms_inv, same bf16 roundings), then the checkpoint FP8 activation encoding of the
//    normalized BF16 row (quant/checkpoint_sm100 quant_g32_vec4's per-group arithmetic: amax floor 1e-4,
//    power-of-two UE8M0 scale, E4M3 satfinite; max is order free) -> codes [N, 1280], ue8m0 [N, 40];
//    optional BF16 qr (q_out) as row_kernel stores it.
//  blockIdx.y == 1: the window-KV record of the wkv columns (paged_quant_write_row<NORM, FP8>).
// One warp per row; no atomics.
__global__ void __launch_bounds__(256) qkv_post_kernel(
        const bf16 *qkv, int64_t stride, const bf16 *kvp, int64_t kv_stride, int64_t rows, const float *qw,
        const float *kw, float eps,
        const float *cosv, const float *sinv, const int *pos, const int64_t *dest, uint8_t *pool,
        uint8_t *codes, uint8_t *scales, bf16 *q_out, int32_t *packed, int64_t pad, WinRows wr) {
    pdl::begin();
    constexpr int D = 1280, E = D / 32;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int64_t r = int64_t(blockIdx.x) * 8 + warp;
    if (r >= rows) {
        // the packed scale words of the TMA padding rows [rows, pad) are zero (layout.pack_activation_scales)
        if (packed && blockIdx.y == 0 && r < pad && lane < D / 128) packed[lane * pad + r] = 0;
        return;
    }
    if (blockIdx.y == 1) {
        if (lane == 0) win_row(wr, pos, const_cast<int64_t *>(dest), r);
        win_main_list(wr, pos, r, lane);
        __syncwarp();
        paged_quant_write_row<true, Q_FP8_B32_UE8M0>(kvp, kv_stride, r, kw, eps, cosv, sinv, pos, dest, pool, lane);
        return;
    }
    __shared__ float buf[8][D];
    float v[E];
#pragma unroll
    for (int i = 0; i < E / 8; ++i) {
        float t[8];
        load_q8(qkv, r * stride + lane * E + i * 8, t);
#pragma unroll
        for (int k = 0; k < 8; ++k) v[i * 8 + k] = t[k];
    }
    const float inv = rms_inv<D>(row_sum_squares<E>(v), eps);
#pragma unroll
    for (int i = 0; i < E; ++i) {
        v[i] = bf16r(__fmul_rn(__fmul_rn(v[i], inv), qw[lane * E + i]));
        buf[warp][lane * E + i] = v[i];
    }
    if (q_out) {
        bf16 *dst = q_out + r * D + lane * E;
#pragma unroll
        for (int i = 0; i < E / 8; ++i) {
            uint4 u;
            bf16 *h = reinterpret_cast<bf16 *>(&u);
#pragma unroll
            for (int k = 0; k < 8; ++k) h[k] = __float2bfloat16_rn(v[i * 8 + k]);
            reinterpret_cast<uint4 *>(dst)[i] = u;
        }
    }
    if (!codes) return;                                // BF16 qr only (q_kv_post)
    __syncwarp();
    // 40 groups of 32: 4 groups per pass (8 lanes x 4 values each, the quant_g32_vec4 mapping)
    const int sub = lane & 7;
    for (int g0 = 0; g0 < D / 32; g0 += 4) {
        const int g = g0 + lane / 8;
        float values[4];
        float amax = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            values[i] = buf[warp][g * 32 + sub * 4 + i];
            amax = fmaxf(amax, fabsf(values[i]));
        }
#pragma unroll
        for (int delta = 4; delta; delta >>= 1)       // the whole warp is here: full mask, 8-lane segments
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, delta, 8));
        amax = fmaxf(amax, 1.0e-4f);
        const float raw_scale = __fmul_rn(amax, 1.0f / 448.0f);
        const uint32_t bits = __float_as_uint(raw_scale);
        const uint32_t exponent = ((bits >> 23) & 255) + ((bits & 0x7fffff) != 0);
        const float scale = __uint_as_float(exponent << 23);
        if (sub == 0) scales[r * (D / 32) + g] = exponent;
        // the same byte in DeepGEMM's MN-major packed layout (layout.pack_activation_scales: word
        // [g / 4][r] holds the four adjacent K-group bytes of row r, byte g % 4): a copy, no arithmetic
        if (packed && sub == 0) reinterpret_cast<uint8_t *>(packed + (g / 4) * pad + r)[g % 4] = uint8_t(exponent);
        uint32_t cw = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float value = fminf(448.0f, fmaxf(-448.0f, __fdiv_rn(values[i], scale)));
            cw |= uint32_t(__nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3)) << (8 * i);
        }
        *reinterpret_cast<uint32_t *>(codes + r * D + g * 32 + sub * 4) = cw;
    }
}

// qkv_post for small row counts, split over independent columns. Grid (rows, QS + 1), one warp per CTA:
//  blockIdx.y < QS: the q part of row r for the 8 FP8 groups [8 y, 8 y + 8) -- the row's RMS factor is evaluated
//    by every part exactly as qkv_post_kernel does (lane l owns elements [40 l, 40 l + 40), the same
//    row_sum_squares chain and butterfly, the same rms_inv), then each element of the part gets the same
//    bf16r(x * inv * w) and each group the same quant_g32_vec4 encoding (amax is order free); only which warp
//    owns a group changed.
//  blockIdx.y == QS: the window-KV record of row r (paged_quant_write_row, unchanged).
// So every output byte is computed by the same per-element program as qkv_post_kernel; rows x 6 CTAs instead of
// ceil(rows / 8) x 2.
template <int QS>
__global__ void __launch_bounds__(32) qkv_post_split_kernel(
        const bf16 *qkv, int64_t stride, const bf16 *kvp, int64_t kv_stride, int64_t rows, const float *qw,
        const float *kw, float eps,
        const float *cosv, const float *sinv, const int *pos, const int64_t *dest, uint8_t *pool,
        uint8_t *codes, uint8_t *scales, bf16 *q_out, int32_t *packed, int64_t pad, WinRows wr) {
    pdl::begin();
    constexpr int D = 1280, E = D / 32, GPP = D / 32 / QS, EPL = GPP * 32 / 32;   // groups per part, elems per lane
    static_assert(D / 32 % QS == 0 && EPL == 8, "8 elements per lane");
    const int lane = threadIdx.x;
    const int64_t r = blockIdx.x;
    const int part = blockIdx.y;
    if (part == QS) {
        if (lane == 0) win_row(wr, pos, const_cast<int64_t *>(dest), r);
        win_main_list(wr, pos, r, lane);
        __syncwarp();
        paged_quant_write_row<true, Q_FP8_B32_UE8M0>(kvp, kv_stride, r, kw, eps, cosv, sinv, pos, dest, pool, lane);
        return;
    }
    if (packed && part == 0 && r == rows - 1)
        for (int64_t q = rows; q < pad; ++q)
            if (lane < D / 128) packed[lane * pad + q] = 0;      // TMA padding rows (pack_activation_scales)
    float v[E];
#pragma unroll
    for (int i = 0; i < E / 8; ++i) {
        float t[8];
        load_q8(qkv, r * stride + lane * E + i * 8, t);
#pragma unroll
        for (int k = 0; k < 8; ++k) v[i * 8 + k] = t[k];
    }
    const float inv = rms_inv<D>(row_sum_squares<E>(v), eps);
    const int e0 = part * GPP * 32 + lane * EPL;                   // this lane's 8 elements; group = e0 / 32
    const int g = e0 / 32;
    float y[EPL];
    {
        float t[8];
        load_q8(qkv, r * stride + e0, t);
#pragma unroll
        for (int k = 0; k < EPL; ++k) y[k] = bf16r(__fmul_rn(__fmul_rn(t[k], inv), qw[e0 + k]));
    }
    if (q_out) {
        uint4 u;
        bf16 *h = reinterpret_cast<bf16 *>(&u);
#pragma unroll
        for (int k = 0; k < EPL; ++k) h[k] = __float2bfloat16_rn(y[k]);
        *reinterpret_cast<uint4 *>(q_out + r * D + e0) = u;
    }
    if (!codes) return;                                // BF16 qr only (q_kv_post)
    float amax = 0.0f;
#pragma unroll
    for (int k = 0; k < EPL; ++k) amax = fmaxf(amax, fabsf(y[k]));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));    // the group's 4 lanes (max is order free)
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    amax = fmaxf(amax, 1.0e-4f);
    const float raw_scale = __fmul_rn(amax, 1.0f / 448.0f);
    const uint32_t bits = __float_as_uint(raw_scale);
    const uint32_t exponent = ((bits >> 23) & 255) + ((bits & 0x7fffff) != 0);
    const float scale = __uint_as_float(exponent << 23);
    if ((lane & 3) == 0) {
        scales[r * (D / 32) + g] = exponent;
        if (packed) reinterpret_cast<uint8_t *>(packed + (g / 4) * pad + r)[g % 4] = uint8_t(exponent);
    }
    uint32_t lo = 0, hi = 0;
#pragma unroll
    for (int k = 0; k < EPL; ++k) {
        const float value = fminf(448.0f, fmaxf(-448.0f, __fdiv_rn(y[k], scale)));
        const uint32_t code = __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
        if (k < 4) lo |= code << (8 * k); else hi |= code << (8 * (k - 4));
    }
    *reinterpret_cast<uint2 *>(codes + r * D + e0) = make_uint2(lo, hi);
}

// RMSNorm of a hidden row (N = 5120) with its BF16 output AND the checkpoint FP8 encoding of that
// output in one launch (the attention input feeds an FP8 checkpoint GEMM). The norm is norms_sm100's
// schedule at N = 5120 (norm_segments.cuh: segments of S consecutive columns, chains from +0, adjacent tree; S
// from schedule_policy through norms_sm100/schedule.py), realised with K = 5 loads per lane and L = 128 lanes,
// one row per CTA; rstd = frsqrt_rn(add.rn(div.rn(root, N), eps)), y = rn_bf16(mul.rn(mul.rn(v,
// rstd), g)) with a BF16 weight); the encoding is quant_g32_vec4's per-group program
// (a 32-group = 4 consecutive lanes of one chunk: amax over 32 values is order free).
// COLLAPSE: the row is first collapsed from the four mHC streams x [T, 4, N] with the (shifted) pre
// [T, 4] by mhc_sm100's collapse program (((p0*x0 + p1*x1) + p2*x2) + p3*x3, every product and sum
// rounded, one BF16 rounding), i.e. the norm reads exactly the BF16 row collapse_kernel would have stored.
// codes == nullptr: BF16 output only.
template <bool COLLAPSE>
__global__ void __launch_bounds__(128) rms_quant_5120_kernel(
        const bf16 *x, int64_t sx, const float *pre, const bf16 *w, float eps, bf16 *out, uint8_t *codes,
        uint8_t *scales) {
    pdl::begin();
    constexpr int K = 5, L = 128, C = 8 * K, N = 8 * L * K;
    const int64_t row = blockIdx.x;
    const int lane = threadIdx.x;
    float v[C];
    if constexpr (COLLAPSE) {
        const float4 pr = *reinterpret_cast<const float4 *>(pre + row * 4);
        const float pv[4] = {pr.x, pr.y, pr.z, pr.w};
#pragma unroll
        for (int k = 0; k < K; ++k) {
            float o[8];
#pragma unroll
            for (int st = 0; st < 4; ++st) {
                const uint4 u = reinterpret_cast<const uint4 *>(x + (row * 4 + st) * N + k * 8 * L + 8 * lane)[0];
                const bf16 *h = reinterpret_cast<const bf16 *>(&u);
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const float prod = __fmul_rn(pv[st], __bfloat162float(h[j]));
                    o[j] = st == 0 ? prod : __fadd_rn(o[j], prod);
                }
            }
#pragma unroll
            for (int j = 0; j < 8; ++j) v[8 * k + j] = bf16r(o[j]);
        }
    } else {
        const bf16 *xr = x + row * sx;
#pragma unroll
        for (int k = 0; k < K; ++k) {
            const uint4 u = reinterpret_cast<const uint4 *>(xr + k * 8 * L + 8 * lane)[0];
            const bf16 *h = reinterpret_cast<const bf16 *>(&u);
#pragma unroll
            for (int j = 0; j < 8; ++j) v[8 * k + j] = __bfloat162float(h[j]);
        }
    }
    __shared__ float sh[K * L];     // sh[c]: the partial of segment c = k L + lane (columns 8c .. 8c + 7)
#pragma unroll
    for (int k = 0; k < K; ++k) sh[k * L + lane] = norm_segments::segment_sq(v + 8 * k);
    __syncthreads();
    const float acc = norm_segments::segment_tree_warp<K * L>(sh);
    const float rs = __frsqrt_rn(__fadd_rn(__fdiv_rn(acc, float(N)), eps));
    bf16 *orow = out + row * N;
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const uint4 wu = reinterpret_cast<const uint4 *>(w + k * 8 * L + 8 * lane)[0];
        const bf16 *wh = reinterpret_cast<const bf16 *>(&wu);
        float y[8];
        uint4 packed;
        bf16 *ph = reinterpret_cast<bf16 *>(&packed);
        float amax = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            ph[j] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(v[8 * k + j], rs), __bfloat162float(wh[j])));
            y[j] = __bfloat162float(ph[j]);
            amax = fmaxf(amax, fabsf(y[j]));
        }
        reinterpret_cast<uint4 *>(orow + k * 8 * L + 8 * lane)[0] = packed;
        if (!codes) continue;
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
        amax = fmaxf(amax, 1.0e-4f);
        const float raw_scale = __fmul_rn(amax, 1.0f / 448.0f);
        const uint32_t bits = __float_as_uint(raw_scale);
        const uint32_t exponent = ((bits >> 23) & 255) + ((bits & 0x7fffff) != 0);
        const float scale = __uint_as_float(exponent << 23);
        const int64_t e0 = int64_t(k) * 8 * L + 8 * lane;            // first element of this lane
        if ((lane & 3) == 0) scales[row * (N / 32) + e0 / 32] = exponent;
        uint32_t lo = 0, hi = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float value = fminf(448.0f, fmaxf(-448.0f, __fdiv_rn(y[j], scale)));
            const uint32_t code = __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
            if (j < 4) lo |= code << (8 * j); else hi |= code << (8 * (j - 4));
        }
        *reinterpret_cast<uint2 *>(codes + row * N + e0) = make_uint2(lo, hi);
    }
}

// pool_varlen with one CTA per group and one thread per channel (512): the per-channel pool.cu scalar
// graph is evaluated by its own thread (same ops, same order, same expf), the pooled BF16 row is staged
// in shared memory and warp 0 forms the RMS sum with pool_varlen_kernel's exact lane layout (lane l owns
// channels 16l..16l+15, row_sum_squares<16>), so every output bit equals pool_varlen_kernel's. The
// warp-per-group form is latency bound at decode (3 x ratio dependent global loads per element, 16
// elements per lane in sequence); here the loads of all channels are in flight at once.
__global__ void __launch_bounds__(512) pool_varlen_wide_kernel(
        const float *proj, const int *cu_q, const int *p0s, const float *state, const int *slot, const int *g_seq,
        const int *g_k, int64_t groups, int ratio, const float *w, float eps, bf16 *out, int *valid, int *cidx,
        int64_t slot_stride) {
    pdl::begin();
    constexpr int E = 16;
    const int64_t g = blockIdx.x;
    const int c = threadIdx.x;
    const int s = g_seq[g], k = g_k[g];
    const int p0 = p0s[s], n = cu_q[s + 1] - cu_q[s];
    const int rem = p0 % ratio, base = p0 - rem;
    const int G = (rem + n) / ratio;
    if (c == 0) {
        valid[g] = k < G ? 1 : 0;
        cidx[g] = base / ratio + k;
    }
    if (k >= G) return;
    auto row = [&](int j) -> const float * {
        const int q = base + k * ratio + j;
        return q < p0 ? state + int64_t(slot[s]) * slot_stride + int64_t(q - base) * 1024
                      : proj + int64_t(cu_q[s] + (q - p0)) * 1024;
    };
    float maximum = -INFINITY;
    for (int j = 0; j < ratio; ++j) maximum = fmaxf(maximum, row(j)[512 + c]);
    float denominator = 0.f;
    for (int j = 0; j < ratio; ++j) denominator += expf(row(j)[512 + c] - maximum);
    float result = 0.f;
    for (int j = 0; j < ratio; ++j) {
        const float pj = expf(row(j)[512 + c] - maximum) / denominator;
        result += pj * row(j)[c];
    }
    __shared__ float pooled[512];
    __shared__ float inv_sh;
    pooled[c] = bf16r(result);
    __syncthreads();
    if (c < 32) {
        float v[E];
#pragma unroll
        for (int i = 0; i < E; ++i) v[i] = pooled[c * E + i];
        const float inv = rms_inv<512>(row_sum_squares<E>(v), eps);
        if (c == 0) inv_sh = inv;
    }
    __syncthreads();
    out[g * 512 + c] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(pooled[c], inv_sh), w[c]));
}

// The ratio > 1 compressor step of a paged batch in ONE launch (comp_post): per group slot g, exactly
//  * pool_varlen_wide_kernel (valid, cidx, the pooled + RMSNormed BF16 latent row; an invalid slot's latent
//    row is zero, as the torch.zeros buffer the separate form pools into);
//  * gpos[g] = cidx * ratio and the record rows dest[g] / idest[g] of paged_rows (block table entry
//    bt[seq, cidx / 64] * per + cidx % 64, -1 when invalid);
//  * the compressed-KV record of the latent row (paged_quant_write kind 1: paged_quant_write_row<false, FP8_F32>
//    by warp 0 on the BF16 row as stored, position gpos);
//  * state_update of sequence s by the CTA of its slot k == 0, after every read of that slot's state (only slot
//    k == 0 reads state rows: q = base + k * ratio + j < p0 needs k == 0), so the tail rows it writes are read
//    by no other CTA of this launch.
// Every output byte is computed by the same per-element program as the separate kernels.
__global__ void __launch_bounds__(512) comp_post_kernel(
        const float *proj, const int *cu_q, const int *p0s, float *state, const int *slot, const int *g_seq,
        const int *g_k, int64_t groups, int ratio, const float *w, float eps, bf16 *out, int *valid, int *cidx,
        int64_t slot_stride, int *gpos, const int *cbt, int64_t cbt_stride, int cper, int64_t *dest,
        const int *ibt, int64_t ibt_stride, int64_t *idest, const float *cosv, const float *sinv, uint8_t *pool) {
    pdl::begin();
    constexpr int E = 16;
    const int64_t g = blockIdx.x;
    const int c = threadIdx.x;
    const int s = g_seq[g], k = g_k[g];
    const int p0 = p0s[s], n = cu_q[s + 1] - cu_q[s];
    const int rem = p0 % ratio, base = p0 - rem;
    const int G = (rem + n) / ratio;
    const bool ok = k < G;
    const int ci = base / ratio + k;
    if (c == 0) {
        valid[g] = ok ? 1 : 0;
        cidx[g] = ci;
        gpos[g] = ci * ratio;
        dest[g] = ok ? int64_t(cbt[int64_t(s) * cbt_stride + ci / 64]) * cper + ci % 64 : int64_t(-1);
        if (idest) idest[g] = ok ? int64_t(ibt[int64_t(s) * ibt_stride + ci / 64]) * 64 + ci % 64 : int64_t(-1);
    }
    __shared__ float pooled[512];
    __shared__ float inv_sh;
    __shared__ __align__(16) bf16 row_bf[512];
    if (ok) {
        auto row = [&](int j) -> const float * {
            const int q = base + k * ratio + j;
            return q < p0 ? state + int64_t(slot[s]) * slot_stride + int64_t(q - base) * 1024
                          : proj + int64_t(cu_q[s] + (q - p0)) * 1024;
        };
        float maximum = -INFINITY;
        for (int j = 0; j < ratio; ++j) maximum = fmaxf(maximum, row(j)[512 + c]);
        float denominator = 0.f;
        for (int j = 0; j < ratio; ++j) denominator += expf(row(j)[512 + c] - maximum);
        float result = 0.f;
        for (int j = 0; j < ratio; ++j) {
            const float pj = expf(row(j)[512 + c] - maximum) / denominator;
            result += pj * row(j)[c];
        }
        pooled[c] = bf16r(result);
        __syncthreads();
        if (c < 32) {
            float v[E];
#pragma unroll
            for (int i = 0; i < E; ++i) v[i] = pooled[c * E + i];
            const float inv = rms_inv<512>(row_sum_squares<E>(v), eps);
            if (c == 0) inv_sh = inv;
        }
        __syncthreads();
        const bf16 y = __float2bfloat16_rn(__fmul_rn(__fmul_rn(pooled[c], inv_sh), w[c]));
        out[g * 512 + c] = y;
        row_bf[c] = y;
        __syncthreads();
        if (c < 32)
            paged_quant_write_row<false, Q_FP8_B32_UE8M0_F32>(row_bf, 0, 0, nullptr, 0.f, cosv, sinv, gpos + g, dest + g,
                                                        pool, c);
    } else {
        out[g * 512 + c] = __float2bfloat16_rn(0.f);
    }
    if (k != 0) return;
    // state_update of sequence s (state_update_kernel's copy, 512 threads)
    __syncthreads();
    const int nb = base + G * ratio;
    float *st = state + int64_t(slot[s]) * slot_stride;
    for (int q = (nb > p0 ? nb : p0); q < p0 + n; ++q) {
        const float *src = proj + int64_t(cu_q[s] + (q - p0)) * 1024;
        float *dst = st + int64_t(q - nb) * 1024;
        for (int cc = c; cc < 1024; cc += blockDim.x) dst[cc] = src[cc];
    }
}

// index bound of each row: ke = (pos + 1) / ratio (pos >= 0), 0 for a masked row (one launch for the
// torch add / floor-divide / where it replaces; integer math only).
__global__ void index_bounds_kernel(const int *pos, const bool *valid, int ratio, int64_t rows, int *ke) {
    pdl::begin();
    const int64_t r = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    ke[r] = (valid && !valid[r]) ? 0 : (pos[r] + 1) / ratio;
}

}  // namespace v41

using namespace v41;

void paged_quant_write(int64_t kind, const torch::Tensor &x, const c10::optional<torch::Tensor> &w, double eps,
                       const torch::Tensor &cosv, const torch::Tensor &sinv, const torch::Tensor &pos,
                       const torch::Tensor &dest, torch::Tensor &pool) {
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.size(1) == 512 && x.stride(1) == 1 &&
                x.stride(0) % 8 == 0 && (reinterpret_cast<uintptr_t>(x.data_ptr()) & 15) == 0);
    TORCH_CHECK(pos.scalar_type() == at::kInt && pos.is_contiguous() && pos.numel() == x.size(0));
    TORCH_CHECK(dest.scalar_type() == at::kLong && dest.is_contiguous() && dest.numel() == x.size(0));
    TORCH_CHECK(pool.scalar_type() == at::kByte && pool.is_contiguous());
    const int64_t rows = x.size(0);
    if (!rows) return;
    const c10::cuda::CUDAGuard guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    const unsigned blocks = unsigned((rows + 7) / 8);
    const bf16 *xp = reinterpret_cast<const bf16 *>(x.data_ptr());
    if (kind == 0) {          // window KV: RMSNorm -> RoPE -> FP8 528 B record
        TORCH_CHECK(w && w->scalar_type() == at::kFloat && w->numel() == 512 && pool.numel() % 528 == 0);
        pdl::launch(paged_quant_write_kernel<true, Q_FP8_B32_UE8M0>, dim3(blocks), dim3(256), size_t(0), stream, xp, x.stride(0), rows, w->data_ptr<float>(), float(eps), cosv.data_ptr<float>(), sinv.data_ptr<float>(),
            pos.data_ptr<int>(), dest.data_ptr<int64_t>(), pool.data_ptr<uint8_t>());
    } else {                  // compressed KV: RoPE (FP32) -> FP8 528 B record
        TORCH_CHECK(kind == 1 && pool.numel() % 528 == 0);
        pdl::launch(paged_quant_write_kernel<false, Q_FP8_B32_UE8M0_F32>, dim3(blocks), dim3(256), size_t(0), stream, xp, x.stride(0), rows, nullptr, 0.f, cosv.data_ptr<float>(), sinv.data_ptr<float>(), pos.data_ptr<int>(),
            dest.data_ptr<int64_t>(), pool.data_ptr<uint8_t>());
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void pool_varlen(const torch::Tensor &proj, const torch::Tensor &cu_q, const torch::Tensor &p0, const torch::Tensor &state,
                 const torch::Tensor &slot, const torch::Tensor &g_seq, const torch::Tensor &g_k, int64_t ratio,
                 const torch::Tensor &w, double eps, torch::Tensor &out, torch::Tensor &valid, torch::Tensor &cidx,
                 bool wide) {
    TORCH_CHECK(proj.scalar_type() == at::kFloat && proj.is_contiguous() && proj.size(-1) == 1024);
    // slots may be strided (block-chunk KV placement); each slot's [ratio, 1024] rows are dense
    TORCH_CHECK(state.scalar_type() == at::kFloat && state.dim() == 3 && state.size(1) == ratio && state.size(2) == 1024 &&
                state.stride(2) == 1 && state.stride(1) == 1024 && state.stride(0) >= ratio * 1024);
    TORCH_CHECK(ratio >= 2 && ratio <= 64 && w.scalar_type() == at::kFloat && w.numel() == 512);
    const int64_t groups = g_seq.numel();
    TORCH_CHECK(out.numel() == groups * 512 && valid.numel() == groups && cidx.numel() == groups);
    if (!groups) return;
    const c10::cuda::CUDAGuard guard(proj.device());
    if (wide) {
        pdl::launch(pool_varlen_wide_kernel, dim3(unsigned(groups)), dim3(512), size_t(0), at::cuda::getCurrentCUDAStream(), proj.data_ptr<float>(), cu_q.data_ptr<int>(), p0.data_ptr<int>(), state.data_ptr<float>(), slot.data_ptr<int>(),
            g_seq.data_ptr<int>(), g_k.data_ptr<int>(), groups, int(ratio), w.data_ptr<float>(), float(eps),
            reinterpret_cast<bf16 *>(out.data_ptr()), valid.data_ptr<int>(), cidx.data_ptr<int>(), state.stride(0));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return;
    }
    pdl::launch(pool_varlen_kernel, dim3(unsigned((groups + 7) / 8)), dim3(256), size_t(0), at::cuda::getCurrentCUDAStream(), proj.data_ptr<float>(), cu_q.data_ptr<int>(), p0.data_ptr<int>(), state.data_ptr<float>(), slot.data_ptr<int>(),
        g_seq.data_ptr<int>(), g_k.data_ptr<int>(), groups, int(ratio), w.data_ptr<float>(), float(eps),
        reinterpret_cast<bf16 *>(out.data_ptr()), valid.data_ptr<int>(), cidx.data_ptr<int>(), state.stride(0));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void comp_post(const torch::Tensor &proj, const torch::Tensor &cu_q, const torch::Tensor &p0, torch::Tensor &state,
               const torch::Tensor &slot, const torch::Tensor &g_seq, const torch::Tensor &g_k, int64_t ratio,
               const torch::Tensor &w, double eps, torch::Tensor &out, torch::Tensor &valid, torch::Tensor &cidx,
               torch::Tensor &gpos, const torch::Tensor &cbt, int64_t cper, torch::Tensor &dest,
               const c10::optional<torch::Tensor> &ibt, const c10::optional<torch::Tensor> &idest,
               const torch::Tensor &cosv, const torch::Tensor &sinv, torch::Tensor &pool) {
    TORCH_CHECK(proj.scalar_type() == at::kFloat && proj.is_contiguous() && proj.size(-1) == 1024);
    TORCH_CHECK(state.scalar_type() == at::kFloat && state.dim() == 3 && state.size(1) == ratio && state.size(2) == 1024 &&
                state.stride(2) == 1 && state.stride(1) == 1024 && state.stride(0) >= ratio * 1024);
    TORCH_CHECK(ratio >= 2 && ratio <= 64 && w.scalar_type() == at::kFloat && w.numel() == 512);
    const int64_t groups = g_seq.numel();
    TORCH_CHECK(g_seq.scalar_type() == at::kInt && g_k.scalar_type() == at::kInt && g_seq.is_contiguous() &&
                g_k.is_contiguous() && g_k.numel() == groups);
    TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.is_contiguous() && out.numel() == groups * 512);
    TORCH_CHECK(valid.scalar_type() == at::kInt && valid.numel() == groups && cidx.scalar_type() == at::kInt &&
                cidx.numel() == groups && gpos.scalar_type() == at::kInt && gpos.is_contiguous() && gpos.numel() == groups);
    TORCH_CHECK(cper >= 64 || cper == 1, "block-table entries are 64-record pages or record rows");
    TORCH_CHECK(cbt.scalar_type() == at::kInt && cbt.dim() == 2 && cbt.stride(1) == 1);
    TORCH_CHECK(dest.scalar_type() == at::kLong && dest.is_contiguous() && dest.numel() == groups);
    TORCH_CHECK(ibt.has_value() == idest.has_value(), "index table and index rows go together");
    const int *ip = nullptr;
    int64_t *idp = nullptr;
    int64_t istride = 0;
    if (ibt) {
        TORCH_CHECK(ibt->scalar_type() == at::kInt && ibt->dim() == 2 && ibt->stride(1) == 1);
        TORCH_CHECK(idest->scalar_type() == at::kLong && idest->is_contiguous() && idest->numel() == groups);
        ip = ibt->data_ptr<int>();
        istride = ibt->stride(0);
        idp = idest->data_ptr<int64_t>();
    }
    TORCH_CHECK(pool.scalar_type() == at::kByte && pool.is_contiguous() && pool.numel() % 528 == 0);
    if (!groups) return;
    const c10::cuda::CUDAGuard guard(proj.device());
    pdl::launch(comp_post_kernel, dim3(unsigned(groups)), dim3(512), size_t(0), at::cuda::getCurrentCUDAStream(),
        proj.data_ptr<float>(), cu_q.data_ptr<int>(), p0.data_ptr<int>(), state.data_ptr<float>(), slot.data_ptr<int>(),
        g_seq.data_ptr<int>(), g_k.data_ptr<int>(), groups, int(ratio), w.data_ptr<float>(), float(eps),
        reinterpret_cast<bf16 *>(out.data_ptr()), valid.data_ptr<int>(), cidx.data_ptr<int>(), state.stride(0),
        gpos.data_ptr<int>(), cbt.data_ptr<int>(), cbt.stride(0), int(cper), dest.data_ptr<int64_t>(), ip, istride, idp,
        cosv.data_ptr<float>(), sinv.data_ptr<float>(), pool.data_ptr<uint8_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void state_update(const torch::Tensor &proj, const torch::Tensor &cu_q, const torch::Tensor &p0, torch::Tensor &state,
                  const torch::Tensor &slot, int64_t ratio) {
    TORCH_CHECK(state.scalar_type() == at::kFloat && state.dim() == 3 && state.size(1) == ratio && state.size(2) == 1024 &&
                state.stride(2) == 1 && state.stride(1) == 1024 && state.stride(0) >= ratio * 1024);
    const int64_t seqs = p0.numel();
    if (!seqs) return;
    const c10::cuda::CUDAGuard guard(proj.device());
    pdl::launch(state_update_kernel, dim3(unsigned(seqs)), dim3(256), size_t(0), at::cuda::getCurrentCUDAStream(), proj.data_ptr<float>(), cu_q.data_ptr<int>(), p0.data_ptr<int>(), state.data_ptr<float>(), slot.data_ptr<int>(),
        seqs, int(ratio), state.stride(0));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void paged_lists(const torch::Tensor &tok_seq, const torch::Tensor &tok_pos, const torch::Tensor &win_bt, int64_t W,
                 const c10::optional<torch::Tensor> &topk, const c10::optional<torch::Tensor> &comp_bt,
                 torch::Tensor &main_list, const c10::optional<torch::Tensor> &extra, int64_t win_per,
                 int64_t comp_per) {
    const int64_t rows = tok_seq.numel();
    TORCH_CHECK(win_per >= 64 || win_per == 1); TORCH_CHECK(comp_per >= 64 || comp_per == 1);
    TORCH_CHECK(main_list.scalar_type() == at::kInt && main_list.is_contiguous() && main_list.size(0) == rows &&
                main_list.size(1) == W);
    TORCH_CHECK(win_bt.scalar_type() == at::kInt && win_bt.stride(1) == 1);
    if (!rows) return;
    const int *tk = nullptr, *cbt = nullptr;
    int64_t tks = 0, cbs = 0;
    int ktopk = 0, kx = 0;
    int *ex = nullptr;
    if (extra) {
        TORCH_CHECK(topk && comp_bt && topk->stride(1) == 1 && comp_bt->stride(1) == 1 && extra->is_contiguous());
        tk = topk->data_ptr<int>(); tks = topk->stride(0); ktopk = int(topk->size(1));
        cbt = comp_bt->data_ptr<int>(); cbs = comp_bt->stride(0);
        ex = extra->data_ptr<int>(); kx = int(extra->size(1));
        TORCH_CHECK(ktopk <= kx);
    }
    const c10::cuda::CUDAGuard guard(tok_seq.device());
    pdl::launch(paged_lists_kernel, dim3(unsigned((rows + 7) / 8)), dim3(256), size_t(0), at::cuda::getCurrentCUDAStream(), tok_seq.data_ptr<int>(), tok_pos.data_ptr<int>(), rows, win_bt.data_ptr<int>(), win_bt.stride(0), int(W),
        tk, tks, ktopk, cbt, cbs, main_list.data_ptr<int>(), ex, kx, int(win_per), int(comp_per));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void paged_rows(const torch::Tensor &bt, const torch::Tensor &seq, const torch::Tensor &pos,
                const c10::optional<torch::Tensor> &mask, torch::Tensor &out, int64_t per) {
    const int64_t rows = seq.numel();
    TORCH_CHECK(per >= 64 || per == 1, "block-table entries are 64-record pages or record rows");
    TORCH_CHECK(bt.scalar_type() == at::kInt && bt.dim() == 2 && bt.stride(1) == 1);
    TORCH_CHECK(seq.scalar_type() == at::kInt && seq.is_contiguous() && pos.scalar_type() == at::kInt &&
                pos.is_contiguous() && pos.numel() == rows);
    TORCH_CHECK(out.scalar_type() == at::kLong && out.is_contiguous() && out.numel() == rows);
    if (!rows) return;
    const c10::cuda::CUDAGuard guard(seq.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    const unsigned blocks = unsigned((rows + 127) / 128);
    if (mask && mask->scalar_type() == at::kInt) {
        TORCH_CHECK(mask->is_contiguous() && mask->numel() == rows);
        pdl::launch(paged_rows_kernel<int>, dim3(blocks), dim3(128), size_t(0), stream, bt.data_ptr<int>(), bt.stride(0), seq.data_ptr<int>(),
            pos.data_ptr<int>(), mask->data_ptr<int>(), rows, out.data_ptr<int64_t>(), int(per));
    } else {
        const bool *mp = nullptr;
        if (mask) {
            TORCH_CHECK(mask->scalar_type() == at::kBool && mask->is_contiguous() && mask->numel() == rows);
            mp = mask->data_ptr<bool>();
        }
        pdl::launch(paged_rows_kernel<bool>, dim3(blocks), dim3(128), size_t(0), stream, bt.data_ptr<int>(), bt.stride(0), seq.data_ptr<int>(),
            pos.data_ptr<int>(), mp, rows, out.data_ptr<int64_t>(), int(per));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void paged_index_write(const torch::Tensor &codes, const torch::Tensor &sf, const torch::Tensor &dest,
                       torch::Tensor &pool) {
    const int64_t rows = dest.numel();
    TORCH_CHECK(codes.scalar_type() == at::kByte && codes.is_contiguous() && codes.numel() == rows * 128);
    TORCH_CHECK(sf.scalar_type() == at::kInt && sf.is_contiguous() && sf.numel() == rows);
    TORCH_CHECK(dest.scalar_type() == at::kLong && dest.is_contiguous());
    // [pages, 64, 1, 132] with dense pages; the page stride may exceed a page (block-chunk KV placement)
    // [pages, 128, 1, 132]: 128-entry pages (vLLM's ratio-1 index-key cache), dest = page * 128 + row
    if (pool.dim() == 4 && pool.size(1) == 128) {
        TORCH_CHECK(pool.scalar_type() == at::kByte && pool.size(2) == 1 && pool.size(3) == 132 &&
                    pool.stride(3) == 1 && pool.stride(2) == 132 && pool.stride(1) == 132 &&
                    pool.stride(0) >= 128 * 132 && pool.stride(0) % 16 == 0);
        if (!rows) return;
        const c10::cuda::CUDAGuard guard(dest.device());
        pdl::launch(paged_index_write128_kernel, dim3(unsigned((rows + 7) / 8)), dim3(256), size_t(0), at::cuda::getCurrentCUDAStream(), codes.data_ptr<uint8_t>(), sf.data_ptr<int>(), dest.data_ptr<int64_t>(), rows, pool.data_ptr<uint8_t>(), pool.stride(0));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return;
    }
    TORCH_CHECK(pool.scalar_type() == at::kByte && pool.dim() == 4 && pool.size(1) == 64 && pool.size(2) == 1 &&
                pool.size(3) == 132 && pool.stride(3) == 1 && pool.stride(2) == 132 && pool.stride(1) == 132 &&
                pool.stride(0) >= 64 * 132 && pool.stride(0) % 16 == 0);
    if (!rows) return;
    const c10::cuda::CUDAGuard guard(dest.device());
    pdl::launch(paged_index_write_kernel, dim3(unsigned((rows + 7) / 8)), dim3(256), size_t(0), at::cuda::getCurrentCUDAStream(), codes.data_ptr<uint8_t>(), sf.data_ptr<int>(), dest.data_ptr<int64_t>(), rows, pool.data_ptr<uint8_t>(), pool.stride(0));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void paged_lists_masked(const torch::Tensor &tok_seq, const torch::Tensor &tok_pos, const torch::Tensor &win_bt,
                        int64_t W, const c10::optional<torch::Tensor> &topk, const c10::optional<torch::Tensor> &comp_bt,
                        torch::Tensor &main_list, const c10::optional<torch::Tensor> &extra,
                        const c10::optional<torch::Tensor> &valid, int64_t win_per, int64_t comp_per,
                        const c10::optional<torch::Tensor> &floor) {
    const int64_t rows = tok_seq.numel();
    TORCH_CHECK(win_per >= 64 || win_per == 1); TORCH_CHECK(comp_per >= 64 || comp_per == 1);
    TORCH_CHECK(main_list.scalar_type() == at::kInt && main_list.is_contiguous() && main_list.size(0) == rows &&
                main_list.size(1) == W);
    TORCH_CHECK(win_bt.scalar_type() == at::kInt && win_bt.stride(1) == 1);
    if (!rows) return;
    const int *tk = nullptr, *cbt = nullptr;
    int64_t tks = 0, cbs = 0;
    int ktopk = 0, kx = 0;
    int *ex = nullptr;
    if (extra) {
        TORCH_CHECK(topk && comp_bt && topk->stride(1) == 1 && comp_bt->stride(1) == 1 && extra->is_contiguous());
        tk = topk->data_ptr<int>(); tks = topk->stride(0); ktopk = int(topk->size(1));
        cbt = comp_bt->data_ptr<int>(); cbs = comp_bt->stride(0);
        ex = extra->data_ptr<int>(); kx = int(extra->size(1));
        TORCH_CHECK(ktopk <= kx);
    }
    const bool *vp = nullptr;
    if (valid) {
        TORCH_CHECK(valid->scalar_type() == at::kBool && valid->is_contiguous() && valid->numel() == rows);
        vp = valid->data_ptr<bool>();
    }
    const int *fl = nullptr;
    if (floor) {
        TORCH_CHECK(floor->scalar_type() == at::kInt && floor->is_contiguous() && floor->numel() == rows,
                    "floor: int32 contiguous, one entry per row");
        fl = floor->data_ptr<int>();
    }
    const c10::cuda::CUDAGuard guard(tok_seq.device());
    pdl::launch(paged_lists_masked_kernel, dim3(unsigned((rows + 7) / 8)), dim3(256), size_t(0), at::cuda::getCurrentCUDAStream(), tok_seq.data_ptr<int>(), tok_pos.data_ptr<int>(), rows, win_bt.data_ptr<int>(), win_bt.stride(0), int(W),
        tk, tks, ktopk, cbt, cbs, main_list.data_ptr<int>(), ex, kx, vp, int(win_per), int(comp_per), fl);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void permute_quant_g32(const torch::Tensor &y, torch::Tensor &codes, torch::Tensor &scales, int64_t G, int64_t M) {
    TORCH_CHECK(y.scalar_type() == at::kBFloat16 && y.is_contiguous() && y.dim() == 2 && y.size(0) == G * M &&
                y.size(1) % 32 == 0);
    const int64_t R = y.size(1);
    TORCH_CHECK(codes.is_contiguous() && codes.element_size() == 1 && codes.numel() == G * M * R);
    TORCH_CHECK(scales.is_contiguous() && scales.element_size() == 1 && scales.numel() == G * M * R / 32);
    const int64_t groups = G * M * R / 32;
    if (!groups) return;
    const c10::cuda::CUDAGuard guard(y.device());
    pdl::launch(permute_quant_g32_kernel, dim3(unsigned((groups + 15) / 16)), dim3(128), size_t(0), at::cuda::getCurrentCUDAStream(), reinterpret_cast<const __nv_bfloat16 *>(y.data_ptr()), reinterpret_cast<uint8_t *>(codes.data_ptr()),
        reinterpret_cast<uint8_t *>(scales.data_ptr()), G, M, R, groups);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// q: [rows, 1280] BF16 rows (stride free), kv: [rows, 512] BF16 rows; codes / scales: the checkpoint FP8
// encoding of qr (both or neither; neither = BF16 qr only, q_out required).
static void q_kv_post_impl(const torch::Tensor &q, const torch::Tensor &kv, const torch::Tensor &qw,
                           const torch::Tensor &kw, double eps, const torch::Tensor &cosv, const torch::Tensor &sinv,
                           const torch::Tensor &pos, const torch::Tensor &dest, torch::Tensor &pool,
                           const c10::optional<torch::Tensor> &codes, const c10::optional<torch::Tensor> &scales,
                           const c10::optional<torch::Tensor> &q_out, const c10::optional<torch::Tensor> &packed,
                           WinRows wr = WinRows{nullptr, 0, nullptr, nullptr, 64, nullptr, 0}) {
    TORCH_CHECK(q.scalar_type() == at::kBFloat16 && q.dim() == 2 && q.size(1) == 1280 &&
                q.stride(1) == 1 && q.stride(0) % 8 == 0 && (reinterpret_cast<uintptr_t>(q.data_ptr()) & 15) == 0);
    const int64_t rows = q.size(0);
    TORCH_CHECK(kv.scalar_type() == at::kBFloat16 && kv.dim() == 2 && kv.size(0) == rows && kv.size(1) == 512 &&
                kv.stride(1) == 1 && kv.stride(0) % 8 == 0 && (reinterpret_cast<uintptr_t>(kv.data_ptr()) & 15) == 0);
    TORCH_CHECK(qw.scalar_type() == at::kFloat && qw.numel() == 1280 && kw.scalar_type() == at::kFloat && kw.numel() == 512);
    TORCH_CHECK(pos.scalar_type() == at::kInt && pos.is_contiguous() && pos.numel() == rows);
    TORCH_CHECK(dest.scalar_type() == at::kLong && dest.is_contiguous() && dest.numel() == rows);
    TORCH_CHECK(pool.scalar_type() == at::kByte && pool.is_contiguous() && pool.numel() % 528 == 0);
    TORCH_CHECK(codes.has_value() == scales.has_value(), "codes and scales go together");
    uint8_t *cp = nullptr, *sp = nullptr;
    if (codes) {
        TORCH_CHECK(codes->is_contiguous() && codes->element_size() == 1 && codes->numel() == rows * 1280);
        TORCH_CHECK(scales->is_contiguous() && scales->element_size() == 1 && scales->numel() == rows * 40);
        cp = reinterpret_cast<uint8_t *>(codes->data_ptr());
        sp = reinterpret_cast<uint8_t *>(scales->data_ptr());
    }
    bf16 *qo = nullptr;
    if (q_out) {
        TORCH_CHECK(q_out->scalar_type() == at::kBFloat16 && q_out->is_contiguous() && q_out->numel() == rows * 1280);
        qo = reinterpret_cast<bf16 *>(q_out->data_ptr());
    }
    TORCH_CHECK(cp || qo, "nothing to write for qr");
    const int64_t pad = (rows + 3) / 4 * 4;
    int32_t *pk = nullptr;
    if (packed) {
        // [1280 / 128, pad] int32, contiguous (pack_activation_scales' buffer before its transposed view)
        TORCH_CHECK(cp, "packed scales need the encoding");
        TORCH_CHECK(packed->scalar_type() == at::kInt && packed->is_contiguous() && packed->dim() == 2 &&
                    packed->size(0) == 1280 / 128 && packed->size(1) == pad, "packed scales [10, pad4(rows)] int32");
        pk = packed->data_ptr<int32_t>();
    }
    if (!rows) return;
    const c10::cuda::CUDAGuard guard(q.device());
    const bf16 *qp = reinterpret_cast<const bf16 *>(q.data_ptr());
    const bf16 *kp = reinterpret_cast<const bf16 *>(kv.data_ptr());
    const char *split_env = std::getenv("ISOEXEC_QKV_POST_SPLIT_ROWS");   // launch shape only (default 256)
    const int64_t split_rows = split_env ? std::atoll(split_env) : int64_t(256);
    if (rows <= split_rows) {
        pdl::launch(qkv_post_split_kernel<5>, dim3(unsigned(rows), 6), dim3(32), size_t(0), at::cuda::getCurrentCUDAStream(),
            qp, q.stride(0), kp, kv.stride(0), rows, qw.data_ptr<float>(), kw.data_ptr<float>(),
            float(eps), cosv.data_ptr<float>(), sinv.data_ptr<float>(), pos.data_ptr<int>(), dest.data_ptr<int64_t>(),
            pool.data_ptr<uint8_t>(), cp, sp, qo, pk, pad, wr);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return;
    }
    const dim3 grid(unsigned((rows + 7) / 8), 2);
    pdl::launch(qkv_post_kernel, dim3(grid), dim3(256), size_t(0), at::cuda::getCurrentCUDAStream(),
        qp, q.stride(0), kp, kv.stride(0), rows, qw.data_ptr<float>(), kw.data_ptr<float>(),
        float(eps), cosv.data_ptr<float>(), sinv.data_ptr<float>(), pos.data_ptr<int>(), dest.data_ptr<int64_t>(),
        pool.data_ptr<uint8_t>(), cp, sp, qo, pk, pad, wr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static void qkv_post_impl(const torch::Tensor &qkv, const torch::Tensor &qw, const torch::Tensor &kw, double eps,
                          const torch::Tensor &cosv, const torch::Tensor &sinv, const torch::Tensor &pos,
                          const torch::Tensor &dest, torch::Tensor &pool, torch::Tensor &codes, torch::Tensor &scales,
                          const c10::optional<torch::Tensor> &q_out, const c10::optional<torch::Tensor> &packed) {
    TORCH_CHECK(qkv.scalar_type() == at::kBFloat16 && qkv.dim() == 2 && qkv.size(1) == 1280 + 512 &&
                qkv.stride(1) == 1 && qkv.stride(0) % 8 == 0 && (reinterpret_cast<uintptr_t>(qkv.data_ptr()) & 15) == 0);
    q_kv_post_impl(qkv.narrow(1, 0, 1280), qkv.narrow(1, 1280, 512), qw, kw, eps, cosv, sinv, pos, dest, pool,
                   codes, scales, q_out, packed);
}

void qkv_post(const torch::Tensor &qkv, const torch::Tensor &qw, const torch::Tensor &kw, double eps,
              const torch::Tensor &cosv, const torch::Tensor &sinv, const torch::Tensor &pos,
              const torch::Tensor &dest, torch::Tensor &pool, torch::Tensor &codes, torch::Tensor &scales,
              const c10::optional<torch::Tensor> &q_out) {
    qkv_post_impl(qkv, qw, kw, eps, cosv, sinv, pos, dest, pool, codes, scales, q_out, c10::nullopt);
}

//: qkv_post plus the activation scales of qr in DeepGEMM's packed MN-major layout (``packed``: the
//: [10, pad4(rows)] int32 buffer of layout.pack_activation_scales, byte for byte), so the wq_b /
//: indexer wq_b DeepGEMM projections need no separate packing launch.
void qkv_post_packed(const torch::Tensor &qkv, const torch::Tensor &qw, const torch::Tensor &kw, double eps,
                     const torch::Tensor &cosv, const torch::Tensor &sinv, const torch::Tensor &pos,
                     const torch::Tensor &dest, torch::Tensor &pool, torch::Tensor &codes, torch::Tensor &scales,
                     const c10::optional<torch::Tensor> &q_out, torch::Tensor &packed) {
    qkv_post_impl(qkv, qw, kw, eps, cosv, sinv, pos, dest, pool, codes, scales, q_out, packed);
}

//: The BF16-master form of qkv_post: q_norm of the wq_a rows ``q`` into ``q_out`` (row_kernel ROW_Q_NORM's
//: bytes) and the window-KV record of the wkv rows ``kv`` (paged_quant_write kind 0's bytes) in ONE launch,
//: from two separate projection outputs; no FP8 encoding of qr.
void q_kv_post(const torch::Tensor &q, const torch::Tensor &kv, const torch::Tensor &qw, const torch::Tensor &kw,
               double eps, const torch::Tensor &cosv, const torch::Tensor &sinv, const torch::Tensor &pos,
               const torch::Tensor &dest, torch::Tensor &pool, torch::Tensor &q_out) {
    q_kv_post_impl(q, kv, qw, kw, eps, cosv, sinv, pos, dest, pool, c10::nullopt, c10::nullopt, q_out, c10::nullopt);
}

//: q_kv_post that also computes the window record rows (paged_rows of win_bt / tok_seq / pos / valid, per) into
//: ``dest`` itself: one launch for the rows kernel + q_norm + the window-KV record.
void q_kv_post_rows(const torch::Tensor &q, const torch::Tensor &kv, const torch::Tensor &qw, const torch::Tensor &kw,
                    double eps, const torch::Tensor &cosv, const torch::Tensor &sinv, const torch::Tensor &pos,
                    torch::Tensor &dest, torch::Tensor &pool, torch::Tensor &q_out, const torch::Tensor &win_bt,
                    const torch::Tensor &tok_seq, const c10::optional<torch::Tensor> &valid, int64_t per,
                    const c10::optional<torch::Tensor> &main_list) {
    TORCH_CHECK(per >= 64 || per == 1, "block-table entries are 64-record pages or record rows");
    int *mp = nullptr;
    int W = 0;
    if (main_list) {
        TORCH_CHECK(main_list->scalar_type() == at::kInt && main_list->is_contiguous() && main_list->dim() == 2 &&
                    main_list->size(0) == q.size(0), "main_list: int32 [rows, W]");
        mp = main_list->data_ptr<int>();
        W = int(main_list->size(1));
    }
    TORCH_CHECK(win_bt.scalar_type() == at::kInt && win_bt.dim() == 2 && win_bt.stride(1) == 1);
    TORCH_CHECK(tok_seq.scalar_type() == at::kInt && tok_seq.is_contiguous() && tok_seq.numel() == q.size(0));
    const bool *vp = nullptr;
    if (valid) {
        TORCH_CHECK(valid->scalar_type() == at::kBool && valid->is_contiguous() && valid->numel() == q.size(0));
        vp = valid->data_ptr<bool>();
    }
    WinRows wr{win_bt.data_ptr<int>(), win_bt.stride(0), tok_seq.data_ptr<int>(), vp, int(per), mp, W};
    q_kv_post_impl(q, kv, qw, kw, eps, cosv, sinv, pos, dest, pool, c10::nullopt, c10::nullopt, q_out, c10::nullopt, wr);
}

// x: [T, 5120] rows, or with pre given the mHC streams [T, 4, 5120] collapsed first. codes/scales optional.
void rms_quant_5120(const torch::Tensor &x, const c10::optional<torch::Tensor> &pre, const torch::Tensor &w,
                    double eps, torch::Tensor &out, const c10::optional<torch::Tensor> &codes,
                    const c10::optional<torch::Tensor> &scales) {
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.size(-1) == 5120 && x.stride(-1) == 1 &&
                (reinterpret_cast<uintptr_t>(x.data_ptr()) & 15) == 0);
    TORCH_CHECK(w.scalar_type() == at::kBFloat16 && w.is_contiguous() && w.numel() == 5120);
    int64_t rows;
    if (pre) {
        TORCH_CHECK(x.dim() == 3 && x.size(1) == 4 && x.is_contiguous(), "streams [T, 4, 5120] contiguous");
        rows = x.size(0);
        TORCH_CHECK(pre->scalar_type() == at::kFloat && pre->is_contiguous() && pre->numel() == rows * 4);
    } else {
        TORCH_CHECK(x.dim() == 2 && x.stride(0) % 8 == 0);
        rows = x.size(0);
    }
    TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.is_contiguous() && out.numel() == rows * 5120);
    TORCH_CHECK(codes.has_value() == scales.has_value());
    if (codes) {
        TORCH_CHECK(codes->is_contiguous() && codes->element_size() == 1 && codes->numel() == rows * 5120);
        TORCH_CHECK(scales->is_contiguous() && scales->element_size() == 1 && scales->numel() == rows * 160);
    }
    if (!rows) return;
    const c10::cuda::CUDAGuard guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    uint8_t *cp = codes ? reinterpret_cast<uint8_t *>(codes->data_ptr()) : nullptr;
    uint8_t *sp = scales ? reinterpret_cast<uint8_t *>(scales->data_ptr()) : nullptr;
    const bf16 *xp = reinterpret_cast<const bf16 *>(x.data_ptr());
    const bf16 *wp = reinterpret_cast<const bf16 *>(w.data_ptr());
    bf16 *op = reinterpret_cast<bf16 *>(out.data_ptr());
    if (pre)
        pdl::launch(rms_quant_5120_kernel<true>, dim3(unsigned(rows)), dim3(128), size_t(0), stream, xp, 0, pre->data_ptr<float>(), wp, float(eps),
                                                                        op, cp, sp);
    else
        pdl::launch(rms_quant_5120_kernel<false>, dim3(unsigned(rows)), dim3(128), size_t(0), stream, xp, x.stride(0), nullptr, wp, float(eps), op,
                                                                         cp, sp);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void index_bounds(const torch::Tensor &pos, const c10::optional<torch::Tensor> &valid, int64_t ratio, torch::Tensor &ke) {
    const int64_t rows = pos.numel();
    TORCH_CHECK(pos.scalar_type() == at::kInt && pos.is_contiguous() && ke.scalar_type() == at::kInt &&
                ke.is_contiguous() && ke.numel() == rows && ratio >= 1);
    const bool *vp = nullptr;
    if (valid) {
        TORCH_CHECK(valid->scalar_type() == at::kBool && valid->is_contiguous() && valid->numel() == rows);
        vp = valid->data_ptr<bool>();
    }
    if (!rows) return;
    const c10::cuda::CUDAGuard guard(pos.device());
    pdl::launch(index_bounds_kernel, dim3(unsigned((rows + 127) / 128)), dim3(128), size_t(0), at::cuda::getCurrentCUDAStream(), pos.data_ptr<int>(), vp, int(ratio), rows, ke.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------------------------- engine window glue
// window_rows_lists: the engine's per-layer window write rows and main lists for EVERY window layer of a step in ONE
// launch (the per-layer torch index glue of engine.EngineStep.bind_window: 18 launches per layer). Per layer l:
//   rows[l, r]     = slot_rows(slots_l[r])                                   (int64)
//   lists[l, r, w] = valid[r] ? int32(slot_rows(src_l[r, w])) : -1
// with slot_rows(s) = s >= 0 ? (s / block) * per_l + s % block + offset_l : -1 (s >= 0: C and floor division agree)
// and src_l[r] = dec_l row r (decode-only step: every row reads the decode buffer, as the as_strided view did), or
// dec_l row r (r < nd) / pre_l row r - nd (r < nd + npf) / -1 (mixed step: torch.cat of the two plus a -1 tail).
// The int32 list value is the int64 value truncated to 32 bits: exactly torch's int32 arithmetic (wrapping) on an int32
// source and its .to(int32) on an int64 one. Integer index math only; no atomics; one thread per (layer, row).
constexpr int kWinMaxLayers = 64;
struct WinLayers {
    const void *slots[kWinMaxLayers];
    const void *dec[kWinMaxLayers];
    const void *pre[kWinMaxLayers];
    int64_t dec_stride[kWinMaxLayers];
    int64_t pre_stride[kWinMaxLayers];
    int64_t per[kWinMaxLayers];
    int64_t offset[kWinMaxLayers];
};

template <typename S, typename L>
__global__ void window_rows_lists_kernel(const __grid_constant__ WinLayers a, const bool *valid, int64_t rows, int W,
                                         int64_t nd, int64_t npf, int64_t block, int64_t *rows_out, int *lists_out) {
    pdl::begin();
    const int l = blockIdx.y;
    const int64_t r = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const int64_t per = a.per[l], off = a.offset[l];
    const int64_t s = int64_t(static_cast<const S *>(a.slots[l])[r]);
    rows_out[int64_t(l) * rows + r] = s >= 0 ? (s / block) * per + s % block + off : -1;
    int *out = lists_out + (int64_t(l) * rows + r) * W;
    const L *src = nullptr;
    if (npf == 0) src = static_cast<const L *>(a.dec[l]) + r * a.dec_stride[l];
    else if (r < nd) src = static_cast<const L *>(a.dec[l]) + r * a.dec_stride[l];
    else if (r < nd + npf) src = static_cast<const L *>(a.pre[l]) + (r - nd) * a.pre_stride[l];
    const bool ok = valid[r];
    for (int w = 0; w < W; ++w) {
        int64_t v = -1;
        if (ok) {
            const int64_t x = src ? int64_t(src[w]) : -1;
            v = x >= 0 ? (x / block) * per + x % block + off : -1;
        }
        out[w] = int(uint32_t(uint64_t(v)));
    }
}

void window_rows_lists(const std::vector<torch::Tensor> &slots, const std::vector<torch::Tensor> &dec,
                       const std::vector<c10::optional<torch::Tensor>> &pre, const std::vector<int64_t> &per,
                       const std::vector<int64_t> &offset, const torch::Tensor &valid, int64_t nd, int64_t npf,
                       int64_t block, torch::Tensor &rows_out, torch::Tensor &lists_out) {
    const int64_t L = int64_t(slots.size());
    TORCH_CHECK(L > 0 && L <= kWinMaxLayers, "window_rows_lists: 1..", kWinMaxLayers, " layers");
    TORCH_CHECK(int64_t(dec.size()) == L && int64_t(pre.size()) == L && int64_t(per.size()) == L &&
                int64_t(offset.size()) == L);
    TORCH_CHECK(block > 0 && nd >= 0 && npf >= 0);
    TORCH_CHECK(valid.scalar_type() == at::kBool && valid.is_contiguous() && valid.dim() == 1);
    const int64_t rows = valid.numel();
    TORCH_CHECK(rows_out.scalar_type() == at::kLong && rows_out.is_contiguous() && rows_out.dim() == 2 &&
                rows_out.size(0) == L && rows_out.size(1) == rows);
    TORCH_CHECK(lists_out.scalar_type() == at::kInt && lists_out.is_contiguous() && lists_out.dim() == 3 &&
                lists_out.size(0) == L && lists_out.size(1) == rows);
    const int W = int(lists_out.size(2));
    const auto st = slots[0].scalar_type(), lt = dec[0].scalar_type();
    TORCH_CHECK(st == at::kLong || st == at::kInt, "slot ids are int32 or int64");
    TORCH_CHECK(lt == at::kLong || lt == at::kInt, "window lists are int32 or int64");
    WinLayers a = {};
    for (int64_t l = 0; l < L; ++l) {
        TORCH_CHECK(slots[l].scalar_type() == st && slots[l].is_contiguous() && slots[l].numel() >= rows);
        TORCH_CHECK(dec[l].scalar_type() == lt && dec[l].dim() == 2 && dec[l].stride(1) == 1 && dec[l].size(1) == W);
        a.slots[l] = slots[l].data_ptr();
        a.dec[l] = dec[l].data_ptr();
        a.dec_stride[l] = dec[l].stride(0);
        if (npf == 0) {
            // every row reads the decode buffer (the old as_strided view over its storage): check the rows exist
            TORCH_CHECK(dec[l].storage_offset() + (rows - 1) * dec[l].stride(0) + W <=
                        int64_t(dec[l].storage().nbytes() / dec[l].element_size()),
                        "window_rows_lists: the decode list buffer has fewer rows than the step");
        } else {
            TORCH_CHECK(dec[l].size(0) >= nd && nd + npf <= rows);
            TORCH_CHECK(pre[l].has_value() && pre[l]->scalar_type() == lt && pre[l]->dim() == 2 &&
                        pre[l]->stride(1) == 1 && pre[l]->size(1) == W && pre[l]->size(0) >= npf);
            a.pre[l] = pre[l]->data_ptr();
            a.pre_stride[l] = pre[l]->stride(0);
        }
        a.per[l] = per[l];
        a.offset[l] = offset[l];
    }
    if (!rows) return;
    const c10::cuda::CUDAGuard guard(valid.device());
    const dim3 grid(unsigned((rows + 127) / 128), unsigned(L));
    auto stream = at::cuda::getCurrentCUDAStream();
#define V41_WRL(S, LT) pdl::launch(window_rows_lists_kernel<S, LT>, grid, dim3(128), size_t(0), stream, a, \
        valid.data_ptr<bool>(), rows, W, nd, npf, block, rows_out.data_ptr<int64_t>(), lists_out.data_ptr<int>())
    if (st == at::kLong && lt == at::kInt) V41_WRL(int64_t, int);
    else if (st == at::kLong) V41_WRL(int64_t, int64_t);
    else if (lt == at::kInt) V41_WRL(int, int);
    else V41_WRL(int, int64_t);
#undef V41_WRL
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// decoder_window_lists: the decoder-row batch's window rows and main lists for every decoder layer of a step in ONE
// launch (ced._engine_rows' two index_select per layer and paged.floor_window_lists' 16 launches per layer). Per
// decoder layer d (source layer src[d] of the step tensors rows_in [Ls, N] / lists_in [Ls, N, W]) and decoder row i:
//   rows_out[d, i]  = rows_in[src[d], sel[i]]
//   lists_out[d, i] = floor[d] ? floor_window_lists(lists_in[src[d], sel[i]], pos[i], fl[i]) : lists_in[src[d], sel[i]]
// floor_window_lists (paged.py): n = #{w : x[w] >= 0}; drop = max(0, f - (p - n + 1)) (int64); out[w] = w + drop < n ?
// x[min(w + drop, W - 1)] : -1. Integer index math only; one thread per (decoder layer, row).
struct DecLayers {
    int src[kWinMaxLayers];
    int floor[kWinMaxLayers];
};

__global__ void decoder_window_lists_kernel(const __grid_constant__ DecLayers a, const int64_t *rows_in,
                                            const int *lists_in, int64_t n_tokens, const int64_t *sel,
                                            const int *pos, const int *fl, int64_t rows, int W, int64_t *rows_out,
                                            int *lists_out) {
    pdl::begin();
    const int d = blockIdx.y;
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= rows) return;
    const int64_t t = sel[i];
    const int64_t s = a.src[d];
    rows_out[int64_t(d) * rows + i] = rows_in[s * n_tokens + t];
    const int *x = lists_in + (s * n_tokens + t) * W;
    int *out = lists_out + (int64_t(d) * rows + i) * W;
    if (!a.floor[d] || fl == nullptr) {
        for (int w = 0; w < W; ++w) out[w] = x[w];
        return;
    }
    int64_t n = 0;
    for (int w = 0; w < W; ++w) n += x[w] >= 0;
    int64_t drop = int64_t(fl[i]) - (int64_t(pos[i]) - n + 1);
    if (drop < 0) drop = 0;
    for (int w = 0; w < W; ++w) {
        const int64_t j = int64_t(w) + drop;
        out[w] = j < n ? x[j < W - 1 ? j : W - 1] : -1;
    }
}

void decoder_window_lists(const torch::Tensor &rows_in, const torch::Tensor &lists_in, const std::vector<int64_t> &src,
                          const std::vector<int64_t> &floor, const torch::Tensor &sel, const torch::Tensor &pos,
                          const c10::optional<torch::Tensor> &fl, torch::Tensor &rows_out, torch::Tensor &lists_out) {
    const int64_t D = int64_t(src.size());
    TORCH_CHECK(D > 0 && D <= kWinMaxLayers && int64_t(floor.size()) == D);
    TORCH_CHECK(rows_in.scalar_type() == at::kLong && rows_in.is_contiguous() && rows_in.dim() == 2);
    TORCH_CHECK(lists_in.scalar_type() == at::kInt && lists_in.is_contiguous() && lists_in.dim() == 3 &&
                lists_in.size(0) == rows_in.size(0) && lists_in.size(1) == rows_in.size(1));
    const int64_t N = rows_in.size(1);
    const int W = int(lists_in.size(2));
    TORCH_CHECK(sel.scalar_type() == at::kLong && sel.is_contiguous() && sel.dim() == 1);
    const int64_t R = sel.numel();
    TORCH_CHECK(pos.scalar_type() == at::kInt && pos.is_contiguous() && pos.numel() == R);
    const int *fp = nullptr;
    if (fl) {
        TORCH_CHECK(fl->scalar_type() == at::kInt && fl->is_contiguous() && fl->numel() == R);
        fp = fl->data_ptr<int>();
    }
    TORCH_CHECK(rows_out.scalar_type() == at::kLong && rows_out.is_contiguous() && rows_out.dim() == 2 &&
                rows_out.size(0) == D && rows_out.size(1) == R);
    TORCH_CHECK(lists_out.scalar_type() == at::kInt && lists_out.is_contiguous() && lists_out.dim() == 3 &&
                lists_out.size(0) == D && lists_out.size(1) == R && lists_out.size(2) == W);
    DecLayers a = {};
    for (int64_t d = 0; d < D; ++d) {
        TORCH_CHECK(src[d] >= 0 && src[d] < rows_in.size(0));
        a.src[d] = int(src[d]);
        a.floor[d] = floor[d] ? 1 : 0;
    }
    if (!R) return;
    const c10::cuda::CUDAGuard guard(sel.device());
    pdl::launch(decoder_window_lists_kernel, dim3(unsigned((R + 127) / 128), unsigned(D)), dim3(128), size_t(0),
                at::cuda::getCurrentCUDAStream(), a, rows_in.data_ptr<int64_t>(), lists_in.data_ptr<int>(), N,
                sel.data_ptr<int64_t>(), pos.data_ptr<int>(), fp, R, W, rows_out.data_ptr<int64_t>(),
                lists_out.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// page_tables: the step's per-source page tables (engine.expand_pages / split_pages) in ONE launch. Table t:
//   out_t[b, j * pages_t + h] = int32(bt_t[b, j] * mult_t + h * step_t + offset_t)   (b < B, j < J_t, h < pages_t)
// expand_pages: mult = rows per block, step = 64, offset = the row view's offset; split_pages: mult = units, step = 1,
// offset = 0. torch evaluates both in int32 (int32 tables and aranges): the int64 value truncated to 32 bits is that
// wrapping arithmetic exactly. Integer index math only; one thread per output entry.
constexpr int kMaxTables = 16;
struct PageTables {
    const int *bt[kMaxTables];
    int64_t bt_stride[kMaxTables];
    int64_t cols[kMaxTables];
    int64_t pages[kMaxTables];
    int64_t mult[kMaxTables];
    int64_t step[kMaxTables];
    int64_t offset[kMaxTables];
    int *out[kMaxTables];
};

__global__ void page_tables_kernel(const __grid_constant__ PageTables a, int64_t rows) {
    pdl::begin();
    const int t = blockIdx.y;
    const int64_t width = a.cols[t] * a.pages[t];
    const int64_t e = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= rows * width) return;
    const int64_t b = e / width, k = e % width, j = k / a.pages[t], h = k % a.pages[t];
    const int64_t v = int64_t(a.bt[t][b * a.bt_stride[t] + j]) * a.mult[t] + h * a.step[t] + a.offset[t];
    a.out[t][e] = int(uint32_t(uint64_t(v)));
}

void page_tables(const std::vector<torch::Tensor> &bts, const std::vector<int64_t> &pages,
                 const std::vector<int64_t> &mult, const std::vector<int64_t> &step, const std::vector<int64_t> &offset,
                 std::vector<torch::Tensor> &outs) {
    const int64_t T = int64_t(bts.size());
    TORCH_CHECK(T > 0 && T <= kMaxTables && int64_t(pages.size()) == T && int64_t(mult.size()) == T &&
                int64_t(step.size()) == T && int64_t(offset.size()) == T && int64_t(outs.size()) == T);
    const int64_t rows = bts[0].size(0);
    PageTables a = {};
    int64_t widest = 0;
    for (int64_t t = 0; t < T; ++t) {
        const auto &bt = bts[t];
        TORCH_CHECK(bt.scalar_type() == at::kInt && bt.dim() == 2 && bt.stride(1) == 1 && bt.size(0) == rows);
        TORCH_CHECK(pages[t] > 0);
        auto &o = outs[t];
        TORCH_CHECK(o.scalar_type() == at::kInt && o.is_contiguous() && o.dim() == 2 && o.size(0) == rows &&
                    o.size(1) == bt.size(1) * pages[t]);
        a.bt[t] = bt.data_ptr<int>();
        a.bt_stride[t] = bt.stride(0);
        a.cols[t] = bt.size(1);
        a.pages[t] = pages[t];
        a.mult[t] = mult[t];
        a.step[t] = step[t];
        a.offset[t] = offset[t];
        a.out[t] = o.data_ptr<int>();
        widest = std::max(widest, rows * bt.size(1) * pages[t]);
    }
    if (!widest) return;
    const c10::cuda::CUDAGuard guard(bts[0].device());
    pdl::launch(page_tables_kernel, dim3(unsigned((widest + 127) / 128), unsigned(T)), dim3(128), size_t(0),
                at::cuda::getCurrentCUDAStream(), a, rows);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
