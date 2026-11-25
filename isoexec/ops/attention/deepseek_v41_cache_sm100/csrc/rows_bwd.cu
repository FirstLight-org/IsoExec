// SPDX-License-Identifier: MIT
// IsoExec deepseek_v41_cache_sm100: backward of the row kernels (BF16 actor training recipe).
// Quantizers are identity straight-through, BF16 roundings are identity (the reference training
// recipe's convention).  Every reduction is owned: one warp per row for row reductions, and the
// RMSNorm weight gradient is a two-level fixed-segment sum over rows (segment 256, ascending), so
// there are no atomics of any kind and the result is run-to-run deterministic.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include "numerics.cuh"

namespace v41 {

using bf16 = __nv_bfloat16;

struct RowBwdParams {
    const bf16 *x; int64_t x_stride;
    const bf16 *dy; int64_t dy_stride;
    int64_t rows;
    const float *w; float eps;
    const float *cosv, *sinv;
    const int *pos; int64_t pos_div, pos_mod, pos_mul, pos_add;
    bf16 *dx; int64_t dx_stride;
    float *dw_rows;                          // [rows, D] per-row weight-gradient terms (nullable)
    int64_t map_a, map_b, map_c, group_major_tokens, gm_heads, gm_hpg;
};

__device__ __forceinline__ int64_t dst_row_b(const RowBwdParams &p, int64_t r) {
    if (p.group_major_tokens) {
        const int64_t t = r / p.gm_heads, h = r % p.gm_heads;
        return (h / p.gm_hpg) * (p.group_major_tokens * p.gm_hpg) + t * p.gm_hpg + (h % p.gm_hpg);
    }
    return (r / p.map_a) * p.map_b + p.map_c + (r % p.map_a);
}

template <int E>
__device__ __forceinline__ void load_e(const bf16 *src, float (&v)[E]) {
#pragma unroll
    for (int k = 0; k < E; ++k) v[k] = __bfloat162float(src[k]);
}

template <int E>
__device__ __forceinline__ float lane_tree_sum(const float (&v)[E]) {
    float s = v[0];
#pragma unroll
    for (int i = 1; i < E; ++i) s = __fadd_rn(s, v[i]);
#pragma unroll
    for (int d = 16; d; d >>= 1) s = __fadd_rn(s, __shfl_xor_sync(0xffffffffu, s, d));
    return s;
}

// ROPE: 0 none, 1 forward rope was applied (backward rotates by the conjugate), 2 inverse was applied.
template <int D, bool NORM, int ROPE>
__global__ void __launch_bounds__(256) row_bwd_kernel(RowBwdParams p) {
    constexpr int E = D / 32;
    const int lane = threadIdx.x & 31;
    const int64_t r = int64_t(blockIdx.x) * 8 + (threadIdx.x >> 5);
    if (r >= p.rows) return;
    float g[E];
    load_e<E>(p.dy + dst_row_b(p, r) * p.dy_stride + lane * E, g);
    if constexpr (ROPE != 0) {
        constexpr int TAIL = D - 64;
        if (lane * E >= TAIL) {
            const int64_t pr = p.pos ? int64_t(p.pos[r / p.pos_div]) : (r / p.pos_div) % p.pos_mod;
            const int64_t position = pr * p.pos_mul + p.pos_add;
            const float *c = p.cosv + position * 32, *s = p.sinv + position * 32;
#pragma unroll
            for (int i = 0; i < E; i += 2) {
                const int j = (lane * E + i - TAIL) / 2;
                rope_pair(g[i], g[i + 1], c[j], s[j], ROPE == 1);   // transpose of a rotation
            }
        }
    }
    if constexpr (NORM) {
        float x[E];
        load_e<E>(p.x + r * p.x_stride + lane * E, x);
        const float inv = rms_inv<D>(row_sum_squares<E>(x), p.eps);
        float t[E];
#pragma unroll
        for (int i = 0; i < E; ++i) t[i] = __fmul_rn(__fmul_rn(g[i], p.w[lane * E + i]), x[i]);
        const float dot = lane_tree_sum<E>(t);                       // sum_j w_j g_j x_j
        const float k = __fdiv_rn(__fmul_rn(__fmul_rn(__fmul_rn(inv, inv), inv), dot), float(D));
        if (p.dw_rows) {
#pragma unroll
            for (int i = 0; i < E; ++i) p.dw_rows[r * D + lane * E + i] = __fmul_rn(g[i], __fmul_rn(x[i], inv));
        }
#pragma unroll
        for (int i = 0; i < E; ++i)
            g[i] = __fsub_rn(__fmul_rn(__fmul_rn(g[i], p.w[lane * E + i]), inv), __fmul_rn(x[i], k));
    }
    bf16 *dst = p.dx + r * p.dx_stride + lane * E;
#pragma unroll
    for (int i = 0; i < E; ++i) dst[i] = __float2bfloat16_rn(g[i]);
}

// out[d] = sum over rows in segments of 256 (ascending inside a segment), segments folded ascending.
__global__ void segment_partial_kernel(const float *rows, int64_t R, int64_t D, int64_t seg, float *part) {
    const int64_t d = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t s = blockIdx.y;
    if (d >= D) return;
    const int64_t r0 = s * seg, r1 = min(R, r0 + seg);
    float acc = 0.f;
    for (int64_t r = r0; r < r1; ++r) acc = __fadd_rn(acc, rows[r * D + d]);
    part[s * D + d] = acc;
}

__global__ void segment_fold_kernel(const float *part, int64_t S, int64_t D, float *out) {
    const int64_t d = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (d >= D) return;
    float acc = 0.f;
    for (int64_t s = 0; s < S; ++s) acc = __fadd_rn(acc, part[s * D + d]);
    out[d] = acc;
}

// Compressor backward: rms backward of the pooled latent, then the pool backward of pool.cu
// (dv_j = p_j dy, ds_j = (p_j (v_j - y)) dy with y the fp32 pool output), per (group, channel).
__global__ void __launch_bounds__(256) pool_norm_bwd_kernel(const float *proj, int64_t proj_stride, int64_t rows,
                                                            int64_t groups, int64_t tokens, int ratio,
                                                            const float *w, float eps, const bf16 *dy,
                                                            float *dproj, float *dw_rows) {
    constexpr int E = 16;
    const int lane = threadIdx.x & 31;
    const int64_t r = int64_t(blockIdx.x) * 8 + (threadIdx.x >> 5);
    if (r >= rows) return;
    const int64_t b = r / groups, gi = r % groups;
    const int64_t t0 = b * tokens + gi * ratio;
    const float *base = proj + t0 * proj_stride;
    float y[E], v[E], g[E];
#pragma unroll
    for (int i = 0; i < E; ++i) {
        const int c = lane * E + i;
        y[i] = pool_channel(base + c, base + 512 + c, proj_stride, ratio);
        v[i] = bf16r(y[i]);
        g[i] = __bfloat162float(dy[r * 512 + c]);
    }
    const float inv = rms_inv<512>(row_sum_squares<E>(v), eps);
    float t[E];
#pragma unroll
    for (int i = 0; i < E; ++i) t[i] = __fmul_rn(__fmul_rn(g[i], w[lane * E + i]), v[i]);
    const float dot = lane_tree_sum<E>(t);
    const float k = __fdiv_rn(__fmul_rn(__fmul_rn(__fmul_rn(inv, inv), inv), dot), 512.0f);
#pragma unroll
    for (int i = 0; i < E; ++i) {
        dw_rows[r * 512 + lane * E + i] = __fmul_rn(g[i], __fmul_rn(v[i], inv));
        g[i] = __fsub_rn(__fmul_rn(__fmul_rn(g[i], w[lane * E + i]), inv), __fmul_rn(v[i], k));
    }
#pragma unroll
    for (int i = 0; i < E; ++i) {
        const int c = lane * E + i;
        const float *s = base + 512 + c, *vv = base + c;
        float maximum = -INFINITY;
        for (int j = 0; j < ratio; ++j) maximum = fmaxf(maximum, s[j * proj_stride]);
        float denominator = 0.f;
        for (int j = 0; j < ratio; ++j) denominator += expf(s[j * proj_stride] - maximum);
        for (int j = 0; j < ratio; ++j) {
            const float pj = expf(s[j * proj_stride] - maximum) / denominator;
            float *row = dproj + (t0 + j) * 1024;
            row[c] = pj * g[i];
            row[512 + c] = (pj * (vv[j * proj_stride] - y[i])) * g[i];
        }
    }
}

}  // namespace v41

using namespace v41;

void row_bwd(int64_t kind, const torch::Tensor &x, const torch::Tensor &dy, int64_t D,
             const c10::optional<torch::Tensor> &w, double eps,
             const c10::optional<torch::Tensor> &cosv, const c10::optional<torch::Tensor> &sinv,
             const c10::optional<torch::Tensor> &pos, int64_t pos_div, int64_t pos_mod, int64_t pos_mul,
             int64_t pos_add, torch::Tensor &dx, const c10::optional<torch::Tensor> &dw_rows,
             int64_t map_a, int64_t map_b, int64_t map_c, int64_t group_major_tokens, int64_t gm_heads, int64_t gm_hpg) {
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.size(1) == D && x.stride(1) == 1);
    TORCH_CHECK(dy.scalar_type() == at::kBFloat16 && dy.dim() == 2 && dy.size(1) == D && dy.stride(1) == 1);
    TORCH_CHECK(dx.scalar_type() == at::kBFloat16 && dx.dim() == 2 && dx.size(1) == D && dx.stride(1) == 1 &&
                dx.size(0) == x.size(0));
    const c10::cuda::CUDAGuard guard(x.device());
    RowBwdParams p{};
    p.x = reinterpret_cast<const bf16 *>(x.data_ptr()); p.x_stride = x.stride(0);
    p.dy = reinterpret_cast<const bf16 *>(dy.data_ptr()); p.dy_stride = dy.stride(0);
    p.rows = x.size(0);
    p.w = w ? w->data_ptr<float>() : nullptr;
    p.eps = float(eps);
    p.cosv = cosv ? cosv->data_ptr<float>() : nullptr;
    p.sinv = sinv ? sinv->data_ptr<float>() : nullptr;
    p.pos = pos ? pos->data_ptr<int>() : nullptr;
    p.pos_div = pos_div; p.pos_mod = pos_mod; p.pos_mul = pos_mul; p.pos_add = pos_add;
    p.dx = reinterpret_cast<bf16 *>(dx.data_ptr()); p.dx_stride = dx.stride(0);
    p.dw_rows = dw_rows ? dw_rows->data_ptr<float>() : nullptr;
    if (dw_rows) TORCH_CHECK(dw_rows->is_contiguous() && dw_rows->numel() == p.rows * D);
    p.map_a = map_a; p.map_b = map_b; p.map_c = map_c; p.group_major_tokens = group_major_tokens; p.gm_heads = gm_heads; p.gm_hpg = gm_hpg;
    TORCH_CHECK(!group_major_tokens || (gm_heads > 0 && gm_hpg > 0 && gm_heads % gm_hpg == 0));
    if (!p.rows) return;
    const unsigned blocks = unsigned((p.rows + 7) / 8);
    auto stream = at::cuda::getCurrentCUDAStream();
    switch (kind) {
        case 0: row_bwd_kernel<1280, true, 0><<<blocks, 256, 0, stream>>>(p); break;
        case 1: row_bwd_kernel<512, true, 1><<<blocks, 256, 0, stream>>>(p); break;
        case 2: row_bwd_kernel<512, false, 1><<<blocks, 256, 0, stream>>>(p); break;
        case 3: row_bwd_kernel<512, false, 2><<<blocks, 256, 0, stream>>>(p); break;
        case 4: row_bwd_kernel<512, true, 0><<<blocks, 256, 0, stream>>>(p); break;
        case 5: row_bwd_kernel<512, false, 1><<<blocks, 256, 0, stream>>>(p); break;
        case 8: row_bwd_kernel<512, true, 1><<<blocks, 256, 0, stream>>>(p); break;
        default: TORCH_CHECK(false, "row kind has no backward");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void segment_sum(const torch::Tensor &rows, int64_t segment, torch::Tensor &out) {
    TORCH_CHECK(rows.scalar_type() == at::kFloat && rows.is_contiguous() && rows.dim() == 2);
    TORCH_CHECK(out.scalar_type() == at::kFloat && out.is_contiguous() && out.numel() == rows.size(1));
    const c10::cuda::CUDAGuard guard(rows.device());
    const int64_t R = rows.size(0), D = rows.size(1);
    const int64_t S = std::max<int64_t>(1, (R + segment - 1) / segment);
    auto part = torch::empty({S, D}, rows.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid(unsigned((D + 127) / 128), unsigned(S));
    if (R) segment_partial_kernel<<<grid, 128, 0, stream>>>(rows.data_ptr<float>(), R, D, segment, part.data_ptr<float>());
    else part.zero_();
    segment_fold_kernel<<<unsigned((D + 127) / 128), 128, 0, stream>>>(part.data_ptr<float>(), S, D, out.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void pool_norm_bwd(const torch::Tensor &proj, int64_t ratio, const torch::Tensor &w, double eps,
                   const torch::Tensor &dy, int64_t groups, int64_t tokens, torch::Tensor &dproj,
                   torch::Tensor &dw_rows) {
    TORCH_CHECK(proj.scalar_type() == at::kFloat && proj.dim() == 2 && proj.size(1) == 1024 && proj.stride(1) == 1);
    TORCH_CHECK(dproj.scalar_type() == at::kFloat && dproj.is_contiguous() && dproj.size(0) == proj.size(0) &&
                dproj.size(1) == 1024);
    TORCH_CHECK(dy.scalar_type() == at::kBFloat16 && dy.is_contiguous() && dy.size(-1) == 512);
    const int64_t rows = dy.numel() / 512;
    TORCH_CHECK(dw_rows.scalar_type() == at::kFloat && dw_rows.is_contiguous() && dw_rows.numel() == rows * 512);
    if (!rows) return;
    const c10::cuda::CUDAGuard guard(proj.device());
    pool_norm_bwd_kernel<<<unsigned((rows + 7) / 8), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        proj.data_ptr<float>(), proj.stride(0), rows, groups, tokens, int(ratio), w.data_ptr<float>(), float(eps),
        reinterpret_cast<const bf16 *>(dy.data_ptr()), dproj.data_ptr<float>(), dw_rows.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
