// SPDX-License-Identifier: MIT
// IsoExec dsa_sm100: the sparse-attention backward's own kernels (recipe sm100_dsa_sparse_bwd_v1, SPEC 6.1
// under the backward rule: no atomics anywhere, not bitwise-required, fp64-accurate).
//
// The backward runs over chunks of query rows.  Per chunk the five GEMM phases (S = q K^T, dP = dO V^T,
// dQ = dS K, dKV_rows = [dS | P]^T [q | dO]) are cuBLAS batched GEMMs with fp32 outputs (cuBLAS's default
// atomics mode is CUBLAS_ATOMICS_NOT_ALLOWED; README "Step 10" has the SASS scan of every kernel a call
// launches).  This file holds the rest, each a plain owner-computes kernel:
//   * delta:  Delta[t, h] = sum_c out[t, h, c] dO[t, h, c]       (one warp per (t, h), fixed tree)
//   * grad:   P = exp2(S scale log2e - lse log2e) (pads 0), dS = scale P (dP - Delta); writes the
//             concatenated A operand [dS ; P] (bf16) of the key-side GEMM and dS for dQ
//   * reduce: dKV[j] += rows[t, p] for the chunk's (t, p) with idx[t, p] == j, one fp32 chain per key in
//             ascending (t, p), carried across chunks through acc (chunk- and batch-invariant) -- a CTA OWNS
//             a block of KB keys; the slot range of its keys in row t is found by binary search (the lists
//             are ascending, pads trailing).  No atomics.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAException.h>

#include "ggemm.cuh"

namespace {

__global__ void delta_kernel(const __nv_bfloat16 *__restrict__ out, const __nv_bfloat16 *__restrict__ dO,
                             float *__restrict__ delta, int rows, int dv) {
    const int w = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
    if (w >= rows) return;
    const __nv_bfloat16 *o = out + (int64_t)w * dv, *d = dO + (int64_t)w * dv;
    float acc = 0.0f;
    for (int c = lane * 8; c < dv; c += 256) {
        const uint4 a = *reinterpret_cast<const uint4 *>(o + c), b = *reinterpret_cast<const uint4 *>(d + c);
        const __nv_bfloat162 *a2 = reinterpret_cast<const __nv_bfloat162 *>(&a), *b2 = reinterpret_cast<const __nv_bfloat162 *>(&b);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 x = __bfloat1622float2(a2[i]), y = __bfloat1622float2(b2[i]);
            acc = fmaf(x.x, y.x, acc);
            acc = fmaf(x.y, y.y, acc);
        }
    }
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, m);
    if (lane == 0) delta[w] = acc;
}

// Sink training needs the unrounded softmax expectation. Reusing the BF16
// output can create a spurious dQ when one key has probability almost one.
// One warp owns a query/head; no atomics or cross-row accumulation.
__global__ void score_delta_kernel(const float* S, const float* dP, const float* lse,
                                   const int* idx, int64_t idx_stride, float* delta,
                                   int C, int H, int K, float scale_log2, int64_t sb) {
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x % 32;
    if (row >= C * H) return;
    float acc = 0.0f;
    const int tok = row / H;
    const int *irow = idx + int64_t(tok) * idx_stride;
    const int64_t base = int64_t(tok) * sb + int64_t(row - tok * H) * K;   // S / dP row of (token, head)
    const float l2 = lse[row] * 1.4426950408889634f;
    for (int p = lane; p < K; p += 32) {
        if (irow[p] >= 0) {
            const int64_t i = base + p;
            const float probability = exp2f(S[i] * scale_log2 - l2);
            acc += probability * dP[i];
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_xor_sync(0xffffffffu, acc, offset);
    if (lane == 0) delta[row] = acc;
}

// S, dP: fp32 [C, H, K]; lse, delta: fp32 [C, H] (lse natural log); idx [C, K] (row stride idx_stride).
// A: bf16 [C, 2H, K]: rows [0, H) = scale * dS, rows [H, 2H) = P.
__global__ void grad_kernel(const float *__restrict__ S, const float *__restrict__ dP, const float *__restrict__ lse,
                            const float *__restrict__ delta, const int *__restrict__ idx, int64_t idx_stride,
                            const int *__restrict__ topk_length, int t0,
                            __nv_bfloat16 *__restrict__ A, int C, int H, int K, float scale_log2, float scale, int64_t sb) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;   // over C * H * K
    if (i >= (int64_t)C * H * K) return;
    const int p = (int)(i % K);
    const int64_t th = i / K;
    const int h = (int)(th % H), c = (int)(th / H);
    const int ix = idx[(int64_t)c * idx_stride + p];
    const int klen = topk_length ? topk_length[t0 + c] : K;
    const bool valid = ix >= 0 && p < klen;
    const float l2 = lse[th] * 1.4426950408889634f;
    const int64_t si = (int64_t)c * sb + (int64_t)h * K + p;           // S / dP: token stride sb
    const float pr = valid ? exp2f(S[si] * scale_log2 - l2) : 0.0f;
    const float pb = __bfloat162float(__float2bfloat16(pr));            // P as the forward rounded it
    const float ds = valid ? scale * pb * (dP[si] - delta[th]) : 0.0f;
    A[((int64_t)c * 2 * H + h) * K + p] = __float2bfloat16(ds);
    A[((int64_t)c * 2 * H + H + h) * K + p] = __float2bfloat16(pb);
}

__device__ __forceinline__ int lower_bound_valid(const int *row, int n, int key) {
    // first slot p in [0, n) with row[p] >= key (n = the valid prefix length)
    int lo = 0, hi = n;
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (row[mid] < key) lo = mid + 1; else hi = mid;
    }
    return lo;
}

// ranges[c, kb] = the slot range [a, b) of row c whose keys fall in key block kb (rows ascending, pads
// trailing): one thread per (row, key block), a binary search each -- all in flight at once.
__global__ void ranges_kernel(const int *__restrict__ idx, int64_t idx_stride, const int *__restrict__ topk_length,
                              int t0, int C, int K, int nkb, int KB, int2 *__restrict__ ranges) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)C * nkb) return;
    const int c = (int)(i / nkb), kb = (int)(i % nkb);
    const int *row = idx + (int64_t)c * idx_stride;
    const int klen = topk_length ? min(topk_length[t0 + c], K) : K;
    int lo = 0, hi = klen;
    while (lo < hi) { const int mid = (lo + hi) >> 1; if (row[mid] >= 0) lo = mid + 1; else hi = mid; }
    const int n = lo;
    ranges[i] = make_int2(lower_bound_valid(row, n, kb * KB), lower_bound_valid(row, n, (kb + 1) * KB));
}

// rows: fp32 [C, K, D]; acc: fp32 [S, D] (read-modify-write, owned per key block).  ONE chain per key:
// a thread owns one float4 column of the block's KB keys, starts from acc (the previous chunks' value) and
// adds the chunk's sources in ascending (row, slot) order, then stores acc.  Over the call this is
// dKV[j] = ((0 + c_1) + c_2) + ... over j's sources in ascending query row (slot order inside a row), so the
// bits do not depend on the chunking or on other sequences' rows in the call (they never select j).
// The CTA stages RB rows' slot ranges in SMEM cooperatively so the walk is not latency bound.  No atomics.
constexpr int RB = 256;
template<int KB>
__global__ void reduce_kernel(const float *__restrict__ rows, const int *__restrict__ idx, int64_t idx_stride,
                              const int2 *__restrict__ ranges, float *__restrict__ acc, int C, int K, int D, int S, int nkb) {
    extern __shared__ float sacc[];                       // [KB][D]
    __shared__ int2 srng[RB];
    const int kb = blockIdx.x, j0 = kb * KB;
    const int nd4 = D / 4, d4 = threadIdx.x;
    float4 *col = reinterpret_cast<float4 *>(sacc);
    for (int e = threadIdx.x; e < KB * nd4; e += blockDim.x) {
        const int jj = e / nd4, dd = e % nd4;
        col[e] = j0 + jj < S ? reinterpret_cast<const float4 *>(acc + (int64_t)(j0 + jj) * D)[dd]
                             : make_float4(0.f, 0.f, 0.f, 0.f);
    }
    for (int c0 = 0; c0 < C; c0 += RB) {
        const int nr = min(RB, C - c0);
        __syncthreads();
        for (int u = threadIdx.x; u < nr; u += blockDim.x) srng[u] = ranges[(int64_t)(c0 + u) * nkb + kb];
        __syncthreads();
        if (d4 >= nd4) continue;
        for (int u = 0; u < nr; ++u) {
            const int2 r = srng[u];
            if (r.x >= r.y) continue;
            const int c = c0 + u;
            const int *row = idx + (int64_t)c * idx_stride;
            for (int p = r.x; p < r.y; p += 4) {
                int jj[4];
                float4 x[4];
                #pragma unroll
                for (int v = 0; v < 4; ++v) {
                    const bool ok = p + v < r.y;
                    jj[v] = ok ? __ldg(row + p + v) - j0 : -1;
                    x[v] = ok ? __ldg(reinterpret_cast<const float4 *>(rows + ((int64_t)c * K + p + v) * D) + d4)
                              : make_float4(0.f, 0.f, 0.f, 0.f);
                }
                #pragma unroll
                for (int v = 0; v < 4; ++v) {
                    if ((unsigned)jj[v] >= (unsigned)KB) continue;   // pads, and any key outside the block
                    float4 y = col[jj[v] * nd4 + d4];
                    y.x += x[v].x; y.y += x[v].y; y.z += x[v].z; y.w += x[v].w;
                    col[jj[v] * nd4 + d4] = y;
                }
            }
        }
    }
    __syncthreads();
    for (int e = threadIdx.x; e < KB * nd4; e += blockDim.x) {
        const int jj = e / nd4, dd = e % nd4;
        if (j0 + jj < S) reinterpret_cast<float4 *>(acc + (int64_t)(j0 + jj) * D)[dd] = col[e];
    }
}

void delta(const torch::Tensor &out, const torch::Tensor &dO, torch::Tensor &d) {
    TORCH_CHECK(out.is_contiguous() && dO.is_contiguous() && out.scalar_type() == at::kBFloat16 && out.size(-1) % 256 == 0);
    const int rows = (int)(out.numel() / out.size(-1));
    delta_kernel<<<(rows * 32 + 255) / 256, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16 *>(out.data_ptr()), reinterpret_cast<const __nv_bfloat16 *>(dO.data_ptr()),
        d.data_ptr<float>(), rows, (int)out.size(-1));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void score_delta(const torch::Tensor &S, const torch::Tensor &dP, const torch::Tensor &lse,
                 const torch::Tensor &idx, torch::Tensor &d, double scale) {
    TORCH_CHECK(S.stride(2) == 1 && S.stride(1) == S.size(2) && dP.strides() == S.strides() && lse.is_contiguous() && d.is_contiguous());
    TORCH_CHECK(S.scalar_type() == at::kFloat && dP.scalar_type() == at::kFloat &&
                lse.scalar_type() == at::kFloat && d.scalar_type() == at::kFloat);
    TORCH_CHECK(S.dim() == 3 && S.sizes() == dP.sizes() && idx.scalar_type() == at::kInt);
    const int C = int(S.size(0)), H = int(S.size(1)), K = int(S.size(2));
    TORCH_CHECK(idx.dim() == 2 && idx.size(0) == C && idx.size(1) == K && idx.stride(1) == 1);
    TORCH_CHECK(lse.numel() == int64_t(C) * H && d.numel() == int64_t(C) * H);
    score_delta_kernel<<<(C * H * 32 + 255) / 256, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
        S.data_ptr<float>(), dP.data_ptr<float>(), lse.data_ptr<float>(), idx.data_ptr<int>(),
        idx.stride(0), d.data_ptr<float>(), C, H, K, float(scale * 1.4426950408889634), S.stride(0));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void grad(const torch::Tensor &S, const torch::Tensor &dP, const torch::Tensor &lse, const torch::Tensor &dlt,
          const torch::Tensor &idx, const c10::optional<torch::Tensor> &topk_length, int64_t t0, torch::Tensor &A,
          double scale) {
    const int C = (int)S.size(0), H = (int)S.size(1), K = (int)S.size(2);
    TORCH_CHECK(S.stride(2) == 1 && S.stride(1) == S.size(2) && dP.strides() == S.strides() && lse.is_contiguous() &&
                dlt.is_contiguous() && A.is_contiguous(), "S / dP: [C, H, K] rows (any token stride)");
    TORCH_CHECK(idx.stride(1) == 1 && idx.size(0) == C && idx.size(1) == K);
    const int64_t n = (int64_t)C * H * K;
    grad_kernel<<<(unsigned)((n + 255) / 256), 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
        S.data_ptr<float>(), dP.data_ptr<float>(), lse.data_ptr<float>(), dlt.data_ptr<float>(), idx.data_ptr<int>(),
        idx.stride(0), topk_length ? topk_length->data_ptr<int>() : nullptr, (int)t0,
        reinterpret_cast<__nv_bfloat16 *>(A.data_ptr()), C, H, K, (float)(scale * 1.4426950408889634), (float)scale, S.stride(0));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void reduce(const torch::Tensor &rows, const torch::Tensor &idx, const c10::optional<torch::Tensor> &topk_length,
            int64_t t0, torch::Tensor &acc) {
    const int C = (int)rows.size(0), K = (int)rows.size(1), D = (int)rows.size(2), S = (int)acc.size(0);
    TORCH_CHECK(rows.is_contiguous() && acc.is_contiguous() && D % 4 == 0 && acc.size(1) == D && D / 4 <= 256);
    constexpr int KB = 8;
    const int nkb = (S + KB - 1) / KB;
    auto ranges = torch::empty({(int64_t)C * nkb * 2}, idx.options());
    auto stream = c10::cuda::getCurrentCUDAStream();
    const int64_t n = (int64_t)C * nkb;
    ranges_kernel<<<(unsigned)((n + 255) / 256), 256, 0, stream>>>(idx.data_ptr<int>(), idx.stride(0),
        topk_length ? topk_length->data_ptr<int>() : nullptr, (int)t0, C, K, nkb, KB,
        reinterpret_cast<int2 *>(ranges.data_ptr<int>()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const int GT = ((D / 4 + 31) / 32) * 32;              // one thread per float4 column
    const size_t smem = (size_t)KB * D * sizeof(float);
    cudaFuncSetAttribute(reduce_kernel<KB>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    reduce_kernel<KB><<<nkb, GT, smem, stream>>>(
        rows.data_ptr<float>(), idx.data_ptr<int>(), idx.stride(0), reinterpret_cast<const int2 *>(ranges.data_ptr<int>()),
        acc.data_ptr<float>(), C, K, D, S, nkb);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace


// ---------------------------------------------------------------------------------- gather-GEMMs (ggemm.cuh)
static CUtensorMap map2d(const void *ptr, CUtensorMapDataType dt, int esize, cuuint64_t d0, cuuint64_t d1, cuuint64_t pitch,
                         cuuint32_t b0, cuuint32_t b1, CUtensorMapSwizzle sw) {
    CUtensorMap m;
    const cuuint64_t dims[2] = {d0, d1};
    const cuuint64_t strides[1] = {pitch * esize};
    const cuuint32_t box[2] = {b0, b1}, estr[2] = {1, 1};
    const CUresult r = cuTensorMapEncodeTiled(&m, dt, 2, const_cast<void *>(ptr), dims, strides, box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                              sw, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", int(r));
    return m;
}

template <int MODE, int NC>
static void launch_ggemm(const dsa_sm100::ggemm::Maps &maps, dsa_sm100::ggemm::Params p) {
    using namespace dsa_sm100::ggemm;
    using Cf = Cfg<NC>;
    const int D = p.dqk;
    p.nnt = MODE == SD ? p.K / NC : D / NC;
    p.nmt = MODE == SD ? (2 * p.h) / MR : MODE == DQ ? (p.h + MR - 1) / MR : (p.K + MR - 1) / MR;
    p.nk = MODE == SD ? D / KC : MODE == DQ ? p.K / KC : (2 * p.h) / KC;
    p.ntiles = p.T * p.nmt * p.nnt;
    int sms = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, c10::cuda::current_device()));
    auto kern = ggemm_kernel<MODE, NC>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, Cf::SMEM_BYTES));
    if (p.ntiles == 0) return;
    kern<<<std::min(p.ntiles, sms), NT, Cf::SMEM_BYTES, c10::cuda::getCurrentCUDAStream()>>>(maps, p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static int pick_nc(int n) { return n % 256 == 0 ? 256 : (n % 128 == 0 ? 128 : 64); }

// [S ; dP] fp32 [C, 2h, K] = [q ; dO] . kv[lists]^T per token (no gathered copy of the keys)
void ggemm_sd(const torch::Tensor &q, const torch::Tensor &dO, const torch::Tensor &kv, const torch::Tensor &lists, torch::Tensor &sd) {
    const int C = q.size(0), h = q.size(1), D = q.size(2), K = lists.size(1), dv = dO.size(2);
    TORCH_CHECK(q.is_contiguous() && dO.is_contiguous() && kv.is_contiguous() && sd.is_contiguous() && lists.stride(1) == 1);
    TORCH_CHECK(h % 64 == 0 && D % 64 == 0 && K % 64 == 0 && kv.size(1) == D && sd.size(0) == C && sd.size(1) == 2 * h && sd.size(2) == K);
    dsa_sm100::ggemm::Maps maps{};
    maps.a0 = map2d(q.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, D, (cuuint64_t)C * h, D, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B);
    maps.a1 = map2d(dO.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, dv, (cuuint64_t)C * h, dv, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B);
    maps.kv = map2d(kv.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, D, kv.size(0), D, 64, 1, CU_TENSOR_MAP_SWIZZLE_128B);
    dsa_sm100::ggemm::Params p{};
    p.mode = 0; p.h = h; p.dqk = D; p.K = K; p.T = C;
    p.lists = lists.data_ptr<int>(); p.lists_stride = lists.stride(0); p.sd = sd.data_ptr<float>();
    const int nc = pick_nc(K);
    if (nc == 256) launch_ggemm<0, 256>(maps, p); else if (nc == 128) launch_ggemm<0, 128>(maps, p); else launch_ggemm<0, 64>(maps, p);
}

// dq bf16 [C, h, D] = dS . kv[lists] per token, dS = rows [0, h) of the grad kernel's [C, 2h, K]
void ggemm_dq(const torch::Tensor &arows, const torch::Tensor &kv, const torch::Tensor &lists, torch::Tensor &dq) {
    const int C = dq.size(0), h = dq.size(1), D = dq.size(2), K = lists.size(1);
    TORCH_CHECK(arows.is_contiguous() && kv.is_contiguous() && dq.stride(2) == 1 && dq.stride(1) == D && lists.stride(1) == 1);
    TORCH_CHECK(arows.size(0) == C && arows.size(1) == 2 * h && arows.size(2) == K && K % 64 == 0 && D % 64 == 0);
    dsa_sm100::ggemm::Maps maps{};
    maps.a0 = map2d(arows.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, K, (cuuint64_t)C * 2 * h, K, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B);
    maps.kv = map2d(kv.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, D, kv.size(0), D, 64, 1, CU_TENSOR_MAP_SWIZZLE_128B);
    {
        const cuuint64_t dims[3] = {(cuuint64_t)D, (cuuint64_t)h, (cuuint64_t)C};
        const cuuint64_t strides[2] = {(cuuint64_t)dq.stride(1) * 2, (cuuint64_t)dq.stride(0) * 2};
        const cuuint32_t box[3] = {32, 32, 1}, estr[3] = {1, 1, 1};
        const CUresult r = cuTensorMapEncodeTiled(&maps.y, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, dq.data_ptr(), dims, strides, box, estr,
                                                  CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_64B,
                                                  CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled (dq) failed: ", int(r));
    }
    dsa_sm100::ggemm::Params p{};
    p.mode = 1; p.h = h; p.dqk = D; p.K = K; p.T = C;
    p.lists = lists.data_ptr<int>(); p.lists_stride = lists.stride(0);
    const int nc = pick_nc(D);
    if (nc == 256) launch_ggemm<1, 256>(maps, p); else if (nc == 128) launch_ggemm<1, 128>(maps, p); else launch_ggemm<1, 64>(maps, p);
}

// rows fp32 [C, K, D] = [dS ; P]^T [q ; dO|0] per token (the dK + dV contribution of every slot; no stacked copy)
void ggemm_rows(const torch::Tensor &arows, const torch::Tensor &q, const torch::Tensor &dO, torch::Tensor &rows) {
    const int C = q.size(0), h = q.size(1), D = q.size(2), K = arows.size(2), dv = dO.size(2);
    TORCH_CHECK(arows.is_contiguous() && q.is_contiguous() && dO.is_contiguous() && rows.is_contiguous());
    TORCH_CHECK(arows.size(0) == C && arows.size(1) == 2 * h && h % 64 == 0 && D % 64 == 0 && K % 64 == 0 &&
                rows.size(0) == C && rows.size(1) == K && rows.size(2) == D);
    dsa_sm100::ggemm::Maps maps{};
    maps.a0 = map2d(arows.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, K, (cuuint64_t)C * 2 * h, K, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B);
    maps.bq = map2d(q.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, D, (cuuint64_t)C * h, D, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B);
    maps.bdo = map2d(dO.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, dv, (cuuint64_t)C * h, dv, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B);
    dsa_sm100::ggemm::Params p{};
    p.mode = 2; p.h = h; p.dqk = D; p.K = K; p.T = C; p.sd = rows.data_ptr<float>();
    const int nc = pick_nc(D);
    if (nc == 256) launch_ggemm<2, 256>(maps, p); else if (nc == 128) launch_ggemm<2, 128>(maps, p); else launch_ggemm<2, 64>(maps, p);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("delta", &delta);
    m.def("score_delta", &score_delta);
    m.def("grad", &grad);
    m.def("reduce", &reduce);
    m.def("ggemm_sd", &ggemm_sd);
    m.def("ggemm_dq", &ggemm_dq);
    m.def("ggemm_rows", &ggemm_rows);
}
