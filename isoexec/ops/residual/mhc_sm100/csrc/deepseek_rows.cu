// SPDX-License-Identifier: Apache-2.0
// DeepSeek-V4.1 mHC boundary, ROW part (inference): [expand] + shifted collapse + RMSNorm [+ FP8 encoding], ONE launch.
//
// Replaces, bit for bit, the row-local kernels of one mHC boundary:
//   merge_fwd_kernel<CAST = false>   (deepseek_mhc_sm100.expand: x_new = bf16(post*out + sum_s comb[s]*x_s), all fp32)
//   v41::rms_quant_5120_kernel<true> (attention/deepseek_v41_cache_sm100 paged.cu: collapse of x_new with the SHIFTED pre,
//                                     norms_sm100 RMSNorm at N = 5120, optional quant_g32 FP8 1x32 UE8M0 encoding)
// The mHC coefficients of the new streams (proj_kernel + tail_kernel, split_fwd.cu) are NOT computed here: the collapse
// uses the previous sublayer's (shifted) pre, so nothing here waits for them and the caller may run them concurrently.
//
// NUMERICAL PROGRAM (each value is produced by exactly the operations, in exactly the order, of the kernels replaced):
//  * expand, element (d, h): t1 = mul.rn(post[d], out[h]); acc = mul.rn(comb[0][d], x0[h]);
//    acc = add.rn(acc, mul.rn(comb[s][d], x_s[h])) for s = 1, 2, 3; x_new = rn_bf16(add.rn(t1, acc)).
//  * collapse, column h: o = mul.rn(p0, xn0); o = add.rn(o, mul.rn(p_s, xn_s)) for s = 1, 2, 3; v = rn_bf16(o)
//    (xn = the bf16 x_new just stored, or the input streams without the expand).
//  * RMSNorm (norms_sm100's schedule, N = 5120; norm_segments.cuh): the row is 640 segments of 8 consecutive columns,
//    segment c = 8c..8c+7, each a chain acc = add.rn(acc, mul.rn(v, v)) from +0, combined by the adjacent tree over
//    c (an unpaired last partial moves up). Norm lane l (L = 128) forms the partials of segments l + 128k (k = 0..4);
//    warp 0 runs the tree. The segment length is schedule_policy's, through norms_sm100/schedule.py.
//    rs = frsqrt_rn(add.rn(div.rn(acc, N), eps)); y = rn_bf16(mul.rn(mul.rn(v, rs), w)).
//  * encoding: per 32-group (4 consecutive chunks) amax = fmaxf over |y| (order free), floor 1e-4, scale = 2^ceil(log2(
//    amax / 448)) as UE8M0, code = e4m3_satfinite(clamp(div.rn(y, scale), +-448)).
// Ownership only (never a value): CPT chunks per thread. CPT = 1 (640 threads, chunk c = thread c; the squares are staged
// in shared memory for the 128 norm lanes) serves small batches (latency); CPT = 5 (128 threads = the norm lanes, chunk
// l + 128k in registers, rms_quant_5120's own layout) serves large batches (one CTA per row either way).
// No atomics; every output row is a function of its own input row.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "pdl.cuh"

#include <cooperative_groups.h>
#include <cuda_fp8.h>

#include "common.cuh"
#include "norm_segments.cuh"

namespace mhc {
namespace dsrows {
namespace cg = cooperative_groups;

constexpr int N = 5120, L = 128, KC = 5, CH = N / 8;   // 640 chunks of 8; chunk c = k * L + l
using bf16 = __nv_bfloat16;

__device__ __forceinline__ float bf16r(float a) { return __bfloat162float(__float2bfloat16_rn(a)); }

template <int CPT, bool MERGE, bool ENCODE>
__global__ void __launch_bounds__(CH / CPT) rows_kernel(const bf16 *__restrict__ x, const bf16 *__restrict__ out,
                                                        const float *__restrict__ post, const float *__restrict__ comb,
                                                        const float *__restrict__ pre, const bf16 *__restrict__ w,
                                                        float eps, bf16 *__restrict__ xnew, bf16 *__restrict__ h,
                                                        uint8_t *__restrict__ codes, uint8_t *__restrict__ scales,
                                                        int32_t *__restrict__ packed, int64_t rows, int64_t pad) {
  pdl::begin();   // launch structure only (pdl.cuh): nothing is read before the predecessor completes
  constexpr int NT = CH / CPT;
  static_assert(CPT == 1 || CPT == 5, "CPT = 1 (smem-staged squares) or 5 (norm-lane layout)");
  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;
  __shared__ float s_fold[KC * L];   // one partial per segment of the row
  __shared__ float s_rs;
  __shared__ __align__(16) float s_sq[CPT == 1 ? CH * 8 : 1];

  if (ENCODE && packed && row == rows - 1)
    for (int64_t i = tid; i < (pad - rows) * (N / 128); i += NT)
      packed[(i / (pad - rows)) * pad + rows + i % (pad - rows)] = 0;   // TMA padding rows: zero words
  float pv[4];
  {
    const float4 q = *reinterpret_cast<const float4 *>(pre + row * 4);
    pv[0] = q.x; pv[1] = q.y; pv[2] = q.z; pv[3] = q.w;
  }
  float pb[4], cb[16];
  if constexpr (MERGE) {
    const float4 q = *reinterpret_cast<const float4 *>(post + row * 4);
    pb[0] = q.x; pb[1] = q.y; pb[2] = q.z; pb[3] = q.w;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const float4 c = *reinterpret_cast<const float4 *>(comb + row * 16 + 4 * k);
      cb[4 * k] = c.x; cb[4 * k + 1] = c.y; cb[4 * k + 2] = c.z; cb[4 * k + 3] = c.w;
    }
  }
  float v[CPT][8];
#pragma unroll
  for (int i = 0; i < CPT; ++i) {
    const int c = tid + NT * i;
    const int64_t col = (int64_t)c * 8;
    float xs[4][8];
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const uint4 q = *reinterpret_cast<const uint4 *>(x + (row * 4 + s) * N + col);
      const uint32_t wv[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
      for (int j = 0; j < 4; ++j) { xs[s][2 * j] = bf_lo(wv[j]); xs[s][2 * j + 1] = bf_hi(wv[j]); }
    }
    if constexpr (MERGE) {
      float ov[8];
      {
        const uint4 q = *reinterpret_cast<const uint4 *>(out + row * N + col);
        const uint32_t wv[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) { ov[2 * j] = bf_lo(wv[j]); ov[2 * j + 1] = bf_hi(wv[j]); }
      }
      float r[4][8];
#pragma unroll
      for (int d = 0; d < 4; ++d)
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const float t1 = __fmul_rn(pb[d], ov[e]);
          float acc = __fmul_rn(cb[0 * 4 + d], xs[0][e]);
          acc = __fadd_rn(acc, __fmul_rn(cb[1 * 4 + d], xs[1][e]));
          acc = __fadd_rn(acc, __fmul_rn(cb[2 * 4 + d], xs[2][e]));
          acc = __fadd_rn(acc, __fmul_rn(cb[3 * 4 + d], xs[3][e]));
          r[d][e] = __fadd_rn(t1, acc);
        }
#pragma unroll
      for (int d = 0; d < 4; ++d) {
        uint4 o;
        o.x = pack_f2_bf16(r[d][0], r[d][1]); o.y = pack_f2_bf16(r[d][2], r[d][3]);
        o.z = pack_f2_bf16(r[d][4], r[d][5]); o.w = pack_f2_bf16(r[d][6], r[d][7]);
        *reinterpret_cast<uint4 *>(xnew + (row * 4 + d) * N + col) = o;
#pragma unroll
        for (int e = 0; e < 8; ++e) xs[d][e] = bf16r(r[d][e]);   // the stored bf16 streams feed the collapse
      }
    }
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      float o = __fmul_rn(pv[0], xs[0][e]);
      o = __fadd_rn(o, __fmul_rn(pv[1], xs[1][e]));
      o = __fadd_rn(o, __fmul_rn(pv[2], xs[2][e]));
      o = __fadd_rn(o, __fmul_rn(pv[3], xs[3][e]));
      v[i][e] = bf16r(o);
    }
    if constexpr (CPT == 1) {
      float4 *dst = reinterpret_cast<float4 *>(s_sq + (size_t)c * 8);
      dst[0] = make_float4(__fmul_rn(v[i][0], v[i][0]), __fmul_rn(v[i][1], v[i][1]), __fmul_rn(v[i][2], v[i][2]),
                           __fmul_rn(v[i][3], v[i][3]));
      dst[1] = make_float4(__fmul_rn(v[i][4], v[i][4]), __fmul_rn(v[i][5], v[i][5]), __fmul_rn(v[i][6], v[i][6]),
                           __fmul_rn(v[i][7], v[i][7]));
    }
  }
  // norm lane l < 128: acc = add.rn(acc, v*v) over chunks l + 128k (k ascending), columns j ascending
  if constexpr (CPT == 1) __syncthreads();
  if (tid < L) {
#pragma unroll
    for (int k = 0; k < KC; ++k) {
      if constexpr (CPT == 1)
        s_fold[k * L + tid] = norm_segments::segment_of_squares(s_sq + (size_t)(k * L + tid) * 8);
      else
        s_fold[k * L + tid] = norm_segments::segment_sq(v[k]);
    }
  }
  if constexpr (CPT == 1) {
    asm volatile("bar.sync 1, 128;" ::: "memory");   // the four norm warps
  } else {
    __syncthreads();
  }
  if (tid < 32) {   // the adjacent tree over the row's KC * L segments, by warp 0
    const float acc = norm_segments::segment_tree_warp<KC * L>(s_fold);
    if (tid == 0) s_rs = __frsqrt_rn(__fadd_rn(__fdiv_rn(acc, float(N)), eps));
  }
  __syncthreads();
  const float rs = s_rs;
#pragma unroll
  for (int i = 0; i < CPT; ++i) {
    const int c = tid + NT * i;
    const int64_t e0 = (int64_t)c * 8;
    const uint4 wu = *reinterpret_cast<const uint4 *>(w + e0);
    const uint32_t ww[4] = {wu.x, wu.y, wu.z, wu.w};
    float y[8];
    uint32_t pk[4];
    float amax = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const bf16 lo = __float2bfloat16_rn(__fmul_rn(__fmul_rn(v[i][2 * j], rs), bf_lo(ww[j])));
      const bf16 hi = __float2bfloat16_rn(__fmul_rn(__fmul_rn(v[i][2 * j + 1], rs), bf_hi(ww[j])));
      pk[j] = pack_bf16(lo, hi);
      y[2 * j] = __bfloat162float(lo);
      y[2 * j + 1] = __bfloat162float(hi);
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) amax = fmaxf(amax, fabsf(y[j]));
    *reinterpret_cast<uint4 *>(h + row * N + e0) = make_uint4(pk[0], pk[1], pk[2], pk[3]);
    if constexpr (ENCODE) {
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
      amax = fmaxf(amax, 1.0e-4f);
      const float raw_scale = __fmul_rn(amax, 1.0f / 448.0f);
      const uint32_t bits = __float_as_uint(raw_scale);
      const uint32_t exponent = ((bits >> 23) & 255) + ((bits & 0x7fffff) != 0);
      const float scale = __uint_as_float(exponent << 23);
      if ((c & 3) == 0) scales[row * (N / 32) + e0 / 32] = (uint8_t)exponent;
      // the same byte in DeepGEMM's packed MN-major activation-scale layout (deepgemm_sm100
      // layout.pack_activation_scales: word [g / 4][row], byte g % 4), when the caller asks for it
      if (packed && (c & 3) == 0) {
        const int64_t g = e0 / 32;
        reinterpret_cast<uint8_t *>(packed + (g / 4) * pad + row)[g & 3] = (uint8_t)exponent;
      }
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
}

// Cluster form (small batches): ONE row per cluster of KC = 5 CTAs; CTA k (cluster rank) owns the norm layout's chunk
// column k (chunks k * 128 + l, l = thread), so the per-row memory traffic is spread over five SMs. Every CTA stages its
// 128 x 8 squares in its shared memory; after one cluster barrier EACH CTA evaluates norm lane l's chain over the five
// CTAs' squares (k ascending, j ascending: rms_quant's chain, read through distributed shared memory) and the same
// wide_fold, so all five compute the identical rs; a second barrier phase (arrive after the remote reads, wait before
// exit) keeps every CTA's squares alive until its peers have read them. Per element the operations are rows_kernel's.
template <bool MERGE, bool ENCODE>
__global__ void __cluster_dims__(KC, 1, 1) __launch_bounds__(L)
    rows_cluster_kernel(const bf16 *__restrict__ x, const bf16 *__restrict__ out, const float *__restrict__ post,
                        const float *__restrict__ comb, const float *__restrict__ pre, const bf16 *__restrict__ w,
                        float eps, bf16 *__restrict__ xnew, bf16 *__restrict__ h, uint8_t *__restrict__ codes,
                        uint8_t *__restrict__ scales, int32_t *__restrict__ packed, int64_t rows, int64_t pad) {
  pdl::begin();   // launch structure only (pdl.cuh): nothing is read before the predecessor completes
  const int64_t row = blockIdx.x / KC;
  const int kq = (int)(blockIdx.x % KC);            // == cluster rank (cluster dims (KC, 1, 1), grid = T * KC)
  const int tid = threadIdx.x;
  const int c = kq * L + tid;
  const int64_t col = (int64_t)c * 8;
  __shared__ __align__(16) float s_sq[L * 8];
  __shared__ float s_fold[KC * L];   // one partial per segment of the row
  __shared__ float s_rs;

  if (ENCODE && packed && row == rows - 1 && kq == 0)
    for (int64_t i = tid; i < (pad - rows) * (N / 128); i += L)
      packed[(i / (pad - rows)) * pad + rows + i % (pad - rows)] = 0;   // TMA padding rows: zero words (as rows_kernel)
  float pv[4];
  {
    const float4 q = *reinterpret_cast<const float4 *>(pre + row * 4);
    pv[0] = q.x; pv[1] = q.y; pv[2] = q.z; pv[3] = q.w;
  }
  float xs[4][8];
#pragma unroll
  for (int s = 0; s < 4; ++s) {
    const uint4 q = *reinterpret_cast<const uint4 *>(x + (row * 4 + s) * N + col);
    const uint32_t wv[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) { xs[s][2 * j] = bf_lo(wv[j]); xs[s][2 * j + 1] = bf_hi(wv[j]); }
  }
  if constexpr (MERGE) {
    float pb[4], cb[16];
    {
      const float4 q = *reinterpret_cast<const float4 *>(post + row * 4);
      pb[0] = q.x; pb[1] = q.y; pb[2] = q.z; pb[3] = q.w;
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const float4 cc = *reinterpret_cast<const float4 *>(comb + row * 16 + 4 * k);
        cb[4 * k] = cc.x; cb[4 * k + 1] = cc.y; cb[4 * k + 2] = cc.z; cb[4 * k + 3] = cc.w;
      }
    }
    float ov[8];
    {
      const uint4 q = *reinterpret_cast<const uint4 *>(out + row * N + col);
      const uint32_t wv[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
      for (int j = 0; j < 4; ++j) { ov[2 * j] = bf_lo(wv[j]); ov[2 * j + 1] = bf_hi(wv[j]); }
    }
    float r[4][8];
#pragma unroll
    for (int d = 0; d < 4; ++d)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const float t1 = __fmul_rn(pb[d], ov[e]);
        float acc = __fmul_rn(cb[0 * 4 + d], xs[0][e]);
        acc = __fadd_rn(acc, __fmul_rn(cb[1 * 4 + d], xs[1][e]));
        acc = __fadd_rn(acc, __fmul_rn(cb[2 * 4 + d], xs[2][e]));
        acc = __fadd_rn(acc, __fmul_rn(cb[3 * 4 + d], xs[3][e]));
        r[d][e] = __fadd_rn(t1, acc);
      }
#pragma unroll
    for (int d = 0; d < 4; ++d) {
      uint4 o;
      o.x = pack_f2_bf16(r[d][0], r[d][1]); o.y = pack_f2_bf16(r[d][2], r[d][3]);
      o.z = pack_f2_bf16(r[d][4], r[d][5]); o.w = pack_f2_bf16(r[d][6], r[d][7]);
      *reinterpret_cast<uint4 *>(xnew + (row * 4 + d) * N + col) = o;
#pragma unroll
      for (int e = 0; e < 8; ++e) xs[d][e] = bf16r(r[d][e]);
    }
  }
  float v[8];
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    float o = __fmul_rn(pv[0], xs[0][e]);
    o = __fadd_rn(o, __fmul_rn(pv[1], xs[1][e]));
    o = __fadd_rn(o, __fmul_rn(pv[2], xs[2][e]));
    o = __fadd_rn(o, __fmul_rn(pv[3], xs[3][e]));
    v[e] = bf16r(o);
  }
  {
    float4 *dst = reinterpret_cast<float4 *>(s_sq + (size_t)tid * 8);
    dst[0] = make_float4(__fmul_rn(v[0], v[0]), __fmul_rn(v[1], v[1]), __fmul_rn(v[2], v[2]), __fmul_rn(v[3], v[3]));
    dst[1] = make_float4(__fmul_rn(v[4], v[4]), __fmul_rn(v[5], v[5]), __fmul_rn(v[6], v[6]), __fmul_rn(v[7], v[7]));
  }
  asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
  {
    cg::cluster_group cluster = cg::this_cluster();
#pragma unroll
    for (int k = 0; k < KC; ++k)   // segment k L + tid lives in cluster rank k's staged squares
      s_fold[k * L + tid] = norm_segments::segment_of_squares(cluster.map_shared_rank(s_sq, k) + (size_t)tid * 8);
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");   // done reading the peers' squares
  }
  __syncthreads();
  if (tid < 32) {   // the adjacent tree over the row's KC * L segments, by warp 0
    const float acc = norm_segments::segment_tree_warp<KC * L>(s_fold);
    if (tid == 0) s_rs = __frsqrt_rn(__fadd_rn(__fdiv_rn(acc, float(N)), eps));
  }
  __syncthreads();
  const float rs = s_rs;
  {
    const int64_t e0 = col;
    const uint4 wu = *reinterpret_cast<const uint4 *>(w + e0);
    const uint32_t ww[4] = {wu.x, wu.y, wu.z, wu.w};
    float y[8];
    uint32_t pk[4];
    float amax = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const bf16 lo = __float2bfloat16_rn(__fmul_rn(__fmul_rn(v[2 * j], rs), bf_lo(ww[j])));
      const bf16 hi = __float2bfloat16_rn(__fmul_rn(__fmul_rn(v[2 * j + 1], rs), bf_hi(ww[j])));
      pk[j] = pack_bf16(lo, hi);
      y[2 * j] = __bfloat162float(lo);
      y[2 * j + 1] = __bfloat162float(hi);
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) amax = fmaxf(amax, fabsf(y[j]));
    *reinterpret_cast<uint4 *>(h + row * N + e0) = make_uint4(pk[0], pk[1], pk[2], pk[3]);
    if constexpr (ENCODE) {
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
      amax = fmaxf(amax, 1.0e-4f);
      const float raw_scale = __fmul_rn(amax, 1.0f / 448.0f);
      const uint32_t bits = __float_as_uint(raw_scale);
      const uint32_t exponent = ((bits >> 23) & 255) + ((bits & 0x7fffff) != 0);
      const float scale = __uint_as_float(exponent << 23);
      if ((c & 3) == 0) scales[row * (N / 32) + e0 / 32] = (uint8_t)exponent;
      // the packed DeepGEMM scale layout, exactly as rows_kernel writes it (word [g / 4][row], byte g % 4)
      if (packed && (c & 3) == 0) {
        const int64_t g = e0 / 32;
        reinterpret_cast<uint8_t *>(packed + (g / 4) * pad + row)[g & 3] = (uint8_t)exponent;
      }
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
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");      // peers finished reading our squares
}

}  // namespace dsrows
}  // namespace mhc

namespace mhc_host {

using namespace mhc;

// rows per call up to which the cluster form (5 CTAs per row) is chosen automatically (ownership only).
constexpr int64_t CLUSTER_MAX_ROWS = 16;   // measured: cluster 5.0 / 5.8 us vs 5.6 / 6.9 at T = 1 / 8; 7.1 vs 7.0 at T = 32

// x: streams [T, 4, 5120] bf16 (the PREVIOUS streams when out/post/comb are given: then x_new = expand(out, x, post, comb)
// is stored to xnew); pre: the shifted pre [T, 4] fp32; w: bf16 [5120]; h: bf16 [T, 5120]; codes/scales: optional FP8
// encoding of h ([T, 5120] e4m3, [T, 160] UE8M0). cpt: 0 = automatic, 1 or 5 (ownership only).
void deepseek_rows_cuda(const at::Tensor &x, const c10::optional<at::Tensor> &out, const c10::optional<at::Tensor> &post,
                        const c10::optional<at::Tensor> &comb, const at::Tensor &pre, const at::Tensor &w, double eps,
                        const c10::optional<at::Tensor> &xnew, at::Tensor &h, const c10::optional<at::Tensor> &codes,
                        const c10::optional<at::Tensor> &scales, int64_t cpt, const c10::optional<at::Tensor> &packed) {
  using dsrows::N;
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() == 3 && x.size(1) == 4 && x.size(2) == N &&
                  x.scalar_type() == at::kBFloat16, "deepseek_rows: x must be contiguous CUDA bf16 [T, 4, 5120]");
  const int64_t T = x.size(0);
  const auto dev = x.device();
  auto on_dev = [&](const at::Tensor &t) { return t.is_cuda() && t.device() == dev && t.is_contiguous(); };
  TORCH_CHECK(on_dev(pre) && pre.scalar_type() == at::kFloat && pre.numel() == T * 4, "deepseek_rows: pre fp32 [T, 4]");
  TORCH_CHECK(on_dev(w) && w.scalar_type() == at::kBFloat16 && w.numel() == N, "deepseek_rows: bf16 norm weight [5120]");
  TORCH_CHECK(on_dev(h) && h.scalar_type() == at::kBFloat16 && h.numel() == T * N, "deepseek_rows: h bf16 [T, 5120]");
  const bool merge = out.has_value();
  TORCH_CHECK(merge == post.has_value() && merge == comb.has_value() && merge == xnew.has_value(),
              "deepseek_rows: the expand needs out, post, comb and xnew together");
  if (merge) {
    TORCH_CHECK(on_dev(*out) && out->scalar_type() == at::kBFloat16 && out->numel() == T * N, "deepseek_rows: out bf16 [T, 5120]");
    TORCH_CHECK(on_dev(*post) && post->scalar_type() == at::kFloat && post->numel() == T * 4, "deepseek_rows: post fp32 [T, 4]");
    TORCH_CHECK(on_dev(*comb) && comb->scalar_type() == at::kFloat && comb->numel() == T * 16, "deepseek_rows: comb fp32 [T, 16]");
    TORCH_CHECK(on_dev(*xnew) && xnew->scalar_type() == at::kBFloat16 && xnew->numel() == T * 4 * N,
                "deepseek_rows: xnew bf16 [T, 4, 5120]");
  }
  const bool enc = codes.has_value();
  TORCH_CHECK(enc == scales.has_value(), "deepseek_rows: codes and scales together");
  if (enc) {
    TORCH_CHECK(on_dev(*codes) && codes->element_size() == 1 && codes->numel() == T * N, "deepseek_rows: codes [T, 5120]");
    TORCH_CHECK(on_dev(*scales) && scales->element_size() == 1 && scales->numel() == T * (N / 32),
                "deepseek_rows: scales [T, 160]");
  }
  const int64_t pad = (T + 3) / 4 * 4;
  int32_t *pk = nullptr;
  if (packed) {
    TORCH_CHECK(enc && on_dev(*packed) && packed->scalar_type() == at::kInt && packed->dim() == 2 &&
                    packed->size(0) == N / 128 && packed->size(1) == pad,
                "deepseek_rows: packed scales int32 [40, pad4(T)] (with codes/scales)");
    pk = packed->data_ptr<int32_t>();
  }
  TORCH_CHECK(cpt == 0 || cpt == 1 || cpt == 2 || cpt == 5, "deepseek_rows: cpt in {0, 1, 2 (cluster), 5}");
  if (T == 0) return;
  if (cpt == 0) cpt = T <= CLUSTER_MAX_ROWS ? 2 : T <= 256 ? 1 : 5;
  const c10::cuda::CUDAGuard guard(dev);
  auto st = at::cuda::getCurrentCUDAStream().stream();
  using bf = __nv_bfloat16;
  const bf *xp = reinterpret_cast<const bf *>(x.data_ptr());
  const bf *op = merge ? reinterpret_cast<const bf *>(out->data_ptr()) : nullptr;
  const float *pp = merge ? post->data_ptr<float>() : nullptr;
  const float *cp = merge ? comb->data_ptr<float>() : nullptr;
  bf *xn = merge ? reinterpret_cast<bf *>(xnew->data_ptr()) : nullptr;
  uint8_t *cd = enc ? reinterpret_cast<uint8_t *>(codes->data_ptr()) : nullptr;
  uint8_t *sc = enc ? reinterpret_cast<uint8_t *>(scales->data_ptr()) : nullptr;
  const float *prp = pre.data_ptr<float>();
  const bf *wp = reinterpret_cast<const bf *>(w.data_ptr());
  bf *hp = reinterpret_cast<bf *>(h.data_ptr());
#define DS_ROWS(CPTV, MG, EN)                                                                                       \
  pdl::launch(dsrows::rows_kernel<CPTV, MG, EN>, dim3((unsigned)T), dim3(dsrows::CH / CPTV), size_t(0), st, xp, op, pp, cp, prp, wp, (float)eps, \
                                                                              xn, hp, cd, sc, pk, T, pad)
#define DS_ROWS_CL(MG, EN)                                                                                         \
  pdl::launch(dsrows::rows_cluster_kernel<MG, EN>, dim3((unsigned)(T * dsrows::KC)), dim3(dsrows::L), size_t(0), st, xp, op, pp, cp, prp, wp,     \
                                                                                  (float)eps, xn, hp, cd, sc, pk, T, pad)
  if (cpt == 2) {
    if (merge) { if (enc) DS_ROWS_CL(true, true); else DS_ROWS_CL(true, false); }
    else       { if (enc) DS_ROWS_CL(false, true); else DS_ROWS_CL(false, false); }
  } else if (cpt == 1) {
    if (merge) { if (enc) DS_ROWS(1, true, true); else DS_ROWS(1, true, false); }
    else       { if (enc) DS_ROWS(1, false, true); else DS_ROWS(1, false, false); }
  } else {
    if (merge) { if (enc) DS_ROWS(5, true, true); else DS_ROWS(5, true, false); }
    else       { if (enc) DS_ROWS(5, false, true); else DS_ROWS(5, false, false); }
  }
#undef DS_ROWS
#undef DS_ROWS_CL
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void deepseek_rows_cuda(const at::Tensor &x, const c10::optional<at::Tensor> &out, const c10::optional<at::Tensor> &post,
                        const c10::optional<at::Tensor> &comb, const at::Tensor &pre, const at::Tensor &w, double eps,
                        const c10::optional<at::Tensor> &xnew, at::Tensor &h, const c10::optional<at::Tensor> &codes,
                        const c10::optional<at::Tensor> &scales, int64_t cpt) {
  deepseek_rows_cuda(x, out, post, comb, pre, w, eps, xnew, h, codes, scales, cpt, c10::nullopt);
}

}  // namespace mhc_host
