// SPDX-License-Identifier: MIT
// Explicit sm100_dsa_sparse_sink_v1 recipe; frozen no-sink root is unchanged.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cmath>

namespace dsa_sm100::attn {

static constexpr int MAXG = 8;

namespace {
struct Node { float o[16]; float L, M; };

__device__ __forceinline__ void merge(Node &a, const Node &b) {
    if (b.L == 0.0f) return;                       // b empty: a verbatim
    if (a.L == 0.0f) { a = b; return; }            // a empty: b verbatim
    const float M = fmaxf(a.M, b.M);
    const float sa = exp2f(a.M - M), sb = exp2f(b.M - M);
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        const float x = a.o[i] * sa;
        const float y = b.o[i] * sb;
        a.o[i] = x + y;
    }
    const float la = a.L * sa, lb = b.L * sb;
    a.L = la + lb;
    a.M = M;
}

__global__ void __launch_bounds__(128) dsa_sink_combine_kernel(const float *partials, int G, int h_q, int rows,
                                                          __nv_bfloat16 *out, float *lse, const float *sink) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int64_t rowid = (int64_t)blockIdx.x * 4 + warp;            // (token, head)
    if (rowid >= rows) return;
    const int64_t tok = rowid / h_q, head = rowid % h_q;
    Node n[MAXG];
    #pragma unroll
    for (int g = 0; g < MAXG; ++g) {
        if (g >= G) break;
        const float *src = partials + (((tok * G + g) * h_q) + head) * 520;
        #pragma unroll
        for (int i = 0; i < 16; i += 4) {
            const float4 v = __ldg(reinterpret_cast<const float4 *>(src + 16 * lane + i));
            n[g].o[i] = v.x; n[g].o[i + 1] = v.y; n[g].o[i + 2] = v.z; n[g].o[i + 3] = v.w;
        }
        const float2 lm = __ldg(reinterpret_cast<const float2 *>(src + 512));
        n[g].L = lm.x;
        n[g].M = lm.y;
    }
    // the canonical adjacent-pair tree: level by level, node i <- merge(node 2i, node 2i+1), an odd last
    // node carried up verbatim
    int cnt = G;
    #pragma unroll
    for (int lvl = 0; lvl < 3; ++lvl) {
        if (cnt <= 1) break;
        #pragma unroll
        for (int i = 0; i < MAXG / 2; ++i) {
            if (2 * i + 1 < cnt) { Node a = n[2 * i]; merge(a, n[2 * i + 1]); n[i] = a; }
            else if (2 * i < cnt) { n[i] = n[2 * i]; }
        }
        cnt = (cnt + 1) / 2;
    }
    // Add a zero-value sink AFTER the unchanged group tree, BEFORE output rounding.
    // Treat an empty attention row as the sink alone; never evaluate -inf - -inf.
    const float sink_log2 = sink[head] * (float)M_LOG2E;
    const bool empty = n[0].L == 0.0f;
    const float M = empty ? sink_log2 : fmaxf(n[0].M, sink_log2);
    const float sa = empty ? 0.0f : exp2f(n[0].M - M);
    const float L = n[0].L * sa + exp2f(sink_log2 - M);
    __nv_bfloat16 *o = out + rowid * 512 + 16 * lane;
    #pragma unroll
    for (int i = 0; i < 16; i += 2) {
        *reinterpret_cast<__nv_bfloat162 *>(o + i) =
            __floats2bfloat162_rn((n[0].o[i] * sa) / L, (n[0].o[i + 1] * sa) / L);
    }
    if (lane == 0) lse[rowid] = empty ? sink[head] : logf(L) + M / (float)M_LOG2E;
}

// d_sink[h] = sum over rows t, ascending, of v[t, h] = -exp(sink[h] - lse[t, h]) * Delta[t, h], with
// Delta[t, h] = out . dO in each lane's fma order and the xor tree.  The per-row terms are independent: one warp
// per (t, h) computes v (sink_term_kernel); the sum over t is the same ascending fp32 chain per head
// (sink_sum_kernel, one thread per head), so the bits equal the former one-warp-per-head loop.  No atomics.
__global__ void sink_term_kernel(const __nv_bfloat16 *out, const __nv_bfloat16 *dout, const float *lse,
                                 const float *sink, float *v, int64_t TH, int H) {
    const int64_t row = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x % 32;
    if (row >= TH) return;
    const int h = (int)(row % H);
    float delta = 0.0f;
    for (int c = lane * 8; c < 512; c += 256) {
        #pragma unroll
        for (int i = 0; i < 8; ++i)
            delta = fmaf(__bfloat162float(out[row * 512 + c + i]), __bfloat162float(dout[row * 512 + c + i]), delta);
    }
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        delta += __shfl_xor_sync(0xffffffffu, delta, mask);
    if (lane == 0) v[row] = -expf(sink[h] - lse[row]) * delta;
}

__global__ void sink_sum_kernel(const float *v, float *grad, int T, int H) {
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= H) return;
    float sum = 0.0f;
    #pragma unroll 8
    for (int t = 0; t < T; ++t) sum += v[(int64_t)t * H + h];
    grad[h] = sum;
}
}  // namespace

void launch_sink_combine(const float *partials, int G, int h_q, int rows, void *out, float *lse,
                         const float *sink, cudaStream_t s) {
    dsa_sink_combine_kernel<<<(rows + 3) / 4, 128, 0, s>>>(partials, G, h_q, rows, (__nv_bfloat16 *)out, lse, sink);
}

void sink_grad(const torch::Tensor &out, const torch::Tensor &dout, const torch::Tensor &lse,
               const torch::Tensor &sink, torch::Tensor &grad) {
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kBFloat16 && out.is_contiguous() &&
                out.dim() == 3 && out.size(2) == 512, "out: contiguous CUDA bf16 [T,H,512]");
    TORCH_CHECK(dout.device() == out.device() && dout.scalar_type() == at::kBFloat16 &&
                dout.is_contiguous() && dout.sizes() == out.sizes(), "dout must match out");
    TORCH_CHECK(lse.device() == out.device() && lse.scalar_type() == at::kFloat && lse.is_contiguous() &&
                lse.numel() == out.size(0) * out.size(1), "lse: fp32 [T,H]");
    TORCH_CHECK(sink.device() == out.device() && sink.scalar_type() == at::kFloat && sink.is_contiguous() &&
                sink.dim() == 1 && sink.size(0) == out.size(1), "sink: fp32 [H]");
    TORCH_CHECK(grad.device() == out.device() && grad.scalar_type() == at::kFloat && grad.is_contiguous() &&
                grad.sizes() == sink.sizes(), "grad must match sink");
    if (out.size(1) == 0) return;
    const int T = (int)out.size(0), H = (int)out.size(1);
    if (T == 0) { grad.zero_(); return; }
    auto v = torch::empty({(int64_t)T * H}, lse.options());
    auto stream = c10::cuda::getCurrentCUDAStream();
    const int64_t TH = (int64_t)T * H;
    sink_term_kernel<<<(unsigned)((TH * 32 + 255) / 256), 256, 0, stream>>>(
        (const __nv_bfloat16 *)out.data_ptr(), (const __nv_bfloat16 *)dout.data_ptr(), lse.data_ptr<float>(),
        sink.data_ptr<float>(), v.data_ptr<float>(), TH, H);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    sink_sum_kernel<<<(H + 63) / 64, 64, 0, stream>>>(v.data_ptr<float>(), grad.data_ptr<float>(), T, H);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace dsa_sm100::attn
