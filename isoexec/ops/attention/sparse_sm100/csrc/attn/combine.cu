// SPDX-License-Identifier: MIT
// IsoExec sparse_sm100: the group-tree combine of the sparse attention (README.md 4.3-4.4).
//
// One warp per (token, head): the G fp32 partials (O[512], L, M) of that row are folded by the
// canonical adjacent-pair tree in ascending group index, unpaired nodes carried up verbatim -- the same
// object reference.tree_fold evaluates and the in-place one-worker fold walks.  merge(a, b), a the lower
// index: an EMPTY operand (L == 0, the exact emptiness test) is bypassed verbatim; otherwise M = max,
// sa = exp2f(a.M - M), sb = exp2f(b.M - M), O = a.O*sa + b.O*sb, L = a.L*sa + b.L*sb (two products,
// one add; --fmad=false).  Root: L' = (L == 0 || L != L) ? 1 : L, out = bf16(O / L'),
// lse = logf(L') + M / log2(e).  Lane l owns O elements [16 l, 16 l + 16).
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cmath>

namespace sparse_sm100::attn {

static constexpr int MAXG = 8;

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

__global__ void __launch_bounds__(128) sparse_combine_kernel(const float *partials, int G, int h_q, int rows,
                                                          __nv_bfloat16 *out, float *lse) {
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
    const float Lp = (n[0].L == 0.0f || n[0].L != n[0].L) ? 1.0f : n[0].L;
    __nv_bfloat16 *o = out + rowid * 512 + 16 * lane;
    #pragma unroll
    for (int i = 0; i < 16; i += 2) {
        *reinterpret_cast<__nv_bfloat162 *>(o + i) = __floats2bfloat162_rn(n[0].o[i] / Lp, n[0].o[i + 1] / Lp);
    }
    if (lane == 0) lse[rowid] = logf(Lp) + n[0].M / (float)M_LOG2E;
}

void launch_combine(const float *partials, int G, int h_q, int rows, void *out, float *lse, cudaStream_t s) {
    sparse_combine_kernel<<<(rows + 3) / 4, 128, 0, s>>>(partials, G, h_q, rows, (__nv_bfloat16 *)out, lse);
}

}  // namespace sparse_sm100::attn
