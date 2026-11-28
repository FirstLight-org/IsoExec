// SPDX-License-Identifier: Apache-2.0
// mHC merge (GLM recipe): res[i][h] = post[i] * out[h] + sum_s comb[s][i] * x[s][h]  in the STREAM dtype.
//
// NUMERICAL PROGRAM (cast boundaries are part of the contract, README "Numerical program"):
//   post_b = bf16(post), comb_b = bf16(comb)                       (cast at the merge)
//   t1     = bf16(post_b[i] * out[h])                              (one rounded bf16 product)
//   t2     = bf16( ((comb_b[0][i]*x0 + comb_b[1][i]*x1) + comb_b[2][i]*x2) + comb_b[3][i]*x3 )   (fp32, rounded each op)
//   res    = bf16(t1 + t2)
// Backward (g = d res, bf16):
//   d out[h]    = bf16( ((g0*post_b[0] + g1*post_b[1]) + g2*post_b[2]) + g3*post_b[3] )
//   d x_s[h]    = bf16( ((comb_b[s][0]*g0 + comb_b[s][1]*g1) + comb_b[s][2]*g2) + comb_b[s][3]*g3 )
//   d post[i]   = fp32(bf16( sum_h g_i[h] * out[h] ))               (cast backward: the bf16 post receives a bf16 gradient)
//   d comb[s][i]= fp32(bf16( sum_h g_i[h] * x_s[h] ))
// The two H reductions are fixed trees: a thread owns chunks j, j+128, ... of 8 h (ascending, sequential fp32), then a
// xor butterfly over the warp (16, 8, 4, 2, 1) and (w0 + w1) + (w2 + w3) over the four warps. A function of H only.
// No atomics.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <tuple>

#include "common.cuh"

#ifndef MERGE_VEC
#define MERGE_VEC 8
#endif
namespace mhc {

__device__ __forceinline__ float bfr(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
// CAST (GLM): round to bf16 and back at this boundary; otherwise (DeepSeek expand) keep fp32.
template <bool CAST> __device__ __forceinline__ float cvt(float v) { return CAST ? bfr(v) : v; }
// acc + a*b. When both factors are bf16 values (a, b with <= 8 significant bits) the product is exact in fp32, so the fused
// form rounds exactly once, like the separate multiply and add: bitwise identical, one instruction fewer.
template <bool EXACT> __device__ __forceinline__ float mac(float a, float b, float acc) {
  return EXACT ? __fmaf_rn(a, b, acc) : __fadd_rn(acc, __fmul_rn(a, b));
}

// DeepSeek collapse backward: dx_s[h] = bf16(pre_s * dcol[h]).
__global__ void __launch_bounds__(256) collapse_bwd_dx_kernel(const __nv_bfloat16 *__restrict__ dcol,
                                                              const float *__restrict__ pre,
                                                              __nv_bfloat16 *__restrict__ dx, int T, int H) {
  const int hv = H / 8;
  const int64_t total = (int64_t)T * hv;
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (int64_t)gridDim.x * blockDim.x) {
    const int64_t row = i / hv;
    const int h = (int)(i - row * hv) * 8;
    const float4 pr = *reinterpret_cast<const float4 *>(pre + row * 4);
    const float pv[4] = {pr.x, pr.y, pr.z, pr.w};
    const uint4 q = *reinterpret_cast<const uint4 *>(dcol + row * H + h);
    const uint32_t w[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      uint4 o;
      o.x = pack_f2_bf16(__fmul_rn(pv[s], bf_lo(w[0])), __fmul_rn(pv[s], bf_hi(w[0])));
      o.y = pack_f2_bf16(__fmul_rn(pv[s], bf_lo(w[1])), __fmul_rn(pv[s], bf_hi(w[1])));
      o.z = pack_f2_bf16(__fmul_rn(pv[s], bf_lo(w[2])), __fmul_rn(pv[s], bf_hi(w[2])));
      o.w = pack_f2_bf16(__fmul_rn(pv[s], bf_lo(w[3])), __fmul_rn(pv[s], bf_hi(w[3])));
      *reinterpret_cast<uint4 *>(dx + (row * 4 + s) * H + h) = o;
    }
  }
}

template <int V>
struct VecIO;   // V bf16 elements per thread: V = 8 -> 16-byte accesses, V = 4 -> 8-byte accesses
template <> struct VecIO<8> {
  static __device__ __forceinline__ void load(uint32_t (&w)[4], const __nv_bfloat16 *p) {
    const uint4 q = *reinterpret_cast<const uint4 *>(p);
    w[0] = q.x; w[1] = q.y; w[2] = q.z; w[3] = q.w;
  }
  static __device__ __forceinline__ void store(__nv_bfloat16 *p, const float (&r)[8]) {
    uint4 o;
    o.x = pack_f2_bf16(r[0], r[1]); o.y = pack_f2_bf16(r[2], r[3]); o.z = pack_f2_bf16(r[4], r[5]); o.w = pack_f2_bf16(r[6], r[7]);
    *reinterpret_cast<uint4 *>(p) = o;
  }
};
template <> struct VecIO<4> {
  static __device__ __forceinline__ void load(uint32_t (&w)[2], const __nv_bfloat16 *p) {
    const uint2 q = *reinterpret_cast<const uint2 *>(p);
    w[0] = q.x; w[1] = q.y;
  }
  static __device__ __forceinline__ void store(__nv_bfloat16 *p, const float (&r)[4]) {
    uint2 o;
    o.x = pack_f2_bf16(r[0], r[1]); o.y = pack_f2_bf16(r[2], r[3]);
    *reinterpret_cast<uint2 *>(p) = o;
  }
};

template <bool CAST, int V>
__global__ void __launch_bounds__(256) merge_fwd_kernel(const __nv_bfloat16 *__restrict__ x,
                                                        const __nv_bfloat16 *__restrict__ out,
                                                        const float *__restrict__ post, const float *__restrict__ comb,
                                                        __nv_bfloat16 *__restrict__ res, int T, int H) {
  const int hv = H / V;
  const int64_t total = (int64_t)T * hv;
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (int64_t)gridDim.x * blockDim.x) {
    const int64_t row = i / hv;
    const int h = (int)(i - row * hv) * V;
    float pb[4], cb[16];
    {
      const float4 q = *reinterpret_cast<const float4 *>(post + row * 4);
      pb[0] = cvt<CAST>(q.x); pb[1] = cvt<CAST>(q.y); pb[2] = cvt<CAST>(q.z); pb[3] = cvt<CAST>(q.w);
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const float4 c = *reinterpret_cast<const float4 *>(comb + row * 16 + 4 * k);
        cb[4 * k] = cvt<CAST>(c.x); cb[4 * k + 1] = cvt<CAST>(c.y); cb[4 * k + 2] = cvt<CAST>(c.z); cb[4 * k + 3] = cvt<CAST>(c.w);
      }
    }
    float xv[4][V], ov[V];
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      uint32_t w[V / 2];
      VecIO<V>::load(w, x + (row * 4 + s) * H + h);
#pragma unroll
      for (int j = 0; j < V / 2; ++j) { xv[s][2 * j] = bf_lo(w[j]); xv[s][2 * j + 1] = bf_hi(w[j]); }
    }
    {
      uint32_t w[V / 2];
      VecIO<V>::load(w, out + row * H + h);
#pragma unroll
      for (int j = 0; j < V / 2; ++j) { ov[2 * j] = bf_lo(w[j]); ov[2 * j + 1] = bf_hi(w[j]); }
    }
#pragma unroll
    for (int d = 0; d < 4; ++d) {   // destination stream
      float r[V];
#pragma unroll
      for (int e = 0; e < V; ++e) {
        const float t1 = cvt<CAST>(__fmul_rn(pb[d], ov[e]));
        float acc = __fmul_rn(cb[0 * 4 + d], xv[0][e]);
        acc = mac<CAST>(cb[1 * 4 + d], xv[1][e], acc);        // CAST: comb_b and x are bf16 -> exact products
        acc = mac<CAST>(cb[2 * 4 + d], xv[2][e], acc);
        acc = mac<CAST>(cb[3 * 4 + d], xv[3][e], acc);
        r[e] = __fadd_rn(t1, cvt<CAST>(acc));
      }
      VecIO<V>::store(res + (row * 4 + d) * H + h, r);
    }
  }
}

// One row per 128-thread block.
template <bool CAST>
__global__ void __launch_bounds__(128) merge_bwd_kernel(
    const __nv_bfloat16 *__restrict__ g, const __nv_bfloat16 *__restrict__ x, const __nv_bfloat16 *__restrict__ out,
    const float *__restrict__ post, const float *__restrict__ comb, __nv_bfloat16 *__restrict__ dx,
    __nv_bfloat16 *__restrict__ dout, float *__restrict__ dpost, float *__restrict__ dcomb, int T, int H) {
  __shared__ float red[4][20];
  const int row = blockIdx.x;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  float pb[4], cb[16];
  {
    const float4 q = *reinterpret_cast<const float4 *>(post + (size_t)row * 4);
    pb[0] = cvt<CAST>(q.x); pb[1] = cvt<CAST>(q.y); pb[2] = cvt<CAST>(q.z); pb[3] = cvt<CAST>(q.w);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const float4 c = *reinterpret_cast<const float4 *>(comb + (size_t)row * 16 + 4 * k);
      cb[4 * k] = cvt<CAST>(c.x); cb[4 * k + 1] = cvt<CAST>(c.y); cb[4 * k + 2] = cvt<CAST>(c.z); cb[4 * k + 3] = cvt<CAST>(c.w);
    }
  }
  float accp[4] = {0.f, 0.f, 0.f, 0.f}, accc[16];   // accc[s*4 + i]
#pragma unroll
  for (int k = 0; k < 16; ++k) accc[k] = 0.f;
  const int hv = H / 8;
  for (int c = tid; c < hv; c += 128) {
    const int h = c * 8;
    float gv[4][8], xv[4][8], ov[8];
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      uint4 q = *reinterpret_cast<const uint4 *>(g + ((size_t)row * 4 + s) * H + h);
      uint32_t w[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
      for (int j = 0; j < 4; ++j) { gv[s][2 * j] = bf_lo(w[j]); gv[s][2 * j + 1] = bf_hi(w[j]); }
      q = *reinterpret_cast<const uint4 *>(x + ((size_t)row * 4 + s) * H + h);
      w[0] = q.x; w[1] = q.y; w[2] = q.z; w[3] = q.w;
#pragma unroll
      for (int j = 0; j < 4; ++j) { xv[s][2 * j] = bf_lo(w[j]); xv[s][2 * j + 1] = bf_hi(w[j]); }
    }
    {
      const uint4 q = *reinterpret_cast<const uint4 *>(out + (size_t)row * H + h);
      const uint32_t w[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
      for (int j = 0; j < 4; ++j) { ov[2 * j] = bf_lo(w[j]); ov[2 * j + 1] = bf_hi(w[j]); }
    }
    // reductions (ascending h within the chunk)
#pragma unroll
    for (int e = 0; e < 8; ++e) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        accp[i] = mac<true>(gv[i][e], ov[e], accp[i]);        // bf16 x bf16: exact products in both recipes
#pragma unroll
        for (int s = 0; s < 4; ++s) accc[s * 4 + i] = mac<true>(gv[i][e], xv[s][e], accc[s * 4 + i]);
      }
    }
    // elementwise gradients
    float dov[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      float a = __fmul_rn(gv[0][e], pb[0]);
      a = mac<CAST>(gv[1][e], pb[1], a);
      a = mac<CAST>(gv[2][e], pb[2], a);
      dov[e] = mac<CAST>(gv[3][e], pb[3], a);
    }
    uint4 o4;
    o4.x = pack_f2_bf16(dov[0], dov[1]); o4.y = pack_f2_bf16(dov[2], dov[3]);
    o4.z = pack_f2_bf16(dov[4], dov[5]); o4.w = pack_f2_bf16(dov[6], dov[7]);
    *reinterpret_cast<uint4 *>(dout + (size_t)row * H + h) = o4;
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      float d[8];
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        float a = __fmul_rn(cb[s * 4 + 0], gv[0][e]);
        a = mac<CAST>(cb[s * 4 + 1], gv[1][e], a);
        a = mac<CAST>(cb[s * 4 + 2], gv[2][e], a);
        d[e] = mac<CAST>(cb[s * 4 + 3], gv[3][e], a);
      }
      uint4 d4;
      d4.x = pack_f2_bf16(d[0], d[1]); d4.y = pack_f2_bf16(d[2], d[3]);
      d4.z = pack_f2_bf16(d[4], d[5]); d4.w = pack_f2_bf16(d[6], d[7]);
      *reinterpret_cast<uint4 *>(dx + ((size_t)row * 4 + s) * H + h) = d4;
    }
  }
  // reduction tree: butterfly over the warp, then (w0 + w1) + (w2 + w3)
  float v[20];
#pragma unroll
  for (int i = 0; i < 4; ++i) v[i] = accp[i];
#pragma unroll
  for (int k = 0; k < 16; ++k) v[4 + k] = accc[k];
#pragma unroll
  for (int k = 0; k < 20; ++k) {
#pragma unroll
    for (int m = 16; m >= 1; m >>= 1) v[k] = __fadd_rn(v[k], __shfl_xor_sync(0xffffffffu, v[k], m));
  }
  if (lane == 0) {
#pragma unroll
    for (int k = 0; k < 20; ++k) red[warp][k] = v[k];
  }
  __syncthreads();
  if (tid < 20) {
    const float tot = __fadd_rn(__fadd_rn(red[0][tid], red[1][tid]), __fadd_rn(red[2][tid], red[3][tid]));
    const float q = cvt<CAST>(tot);   // CAST: gradient of the bf16 cast; else the fp32 gradient
    if (tid < 4) dpost[(size_t)row * 4 + tid] = q;
    else {
      const int k = tid - 4;   // k = s*4 + i
      dcomb[(size_t)row * 16 + k] = q;
    }
  }
}

}  // namespace mhc

namespace mhc_host {

using namespace mhc;

static void check_stream_args(const at::Tensor &x, const at::Tensor &out, const at::Tensor &post,
                              const at::Tensor &comb) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() == 3 && x.size(1) == 4 && x.scalar_type() == at::kBFloat16,
              "streams must be contiguous CUDA bf16 [T, 4, H]");
  const int64_t T = x.size(0), H = x.size(2);
  TORCH_CHECK(H % 8 == 0, "H must be a multiple of 8");
  TORCH_CHECK(out.is_contiguous() && out.scalar_type() == at::kBFloat16 && out.sizes() == at::IntArrayRef({T, H}),
              "output must be contiguous bf16 [T, H]");
  TORCH_CHECK(post.is_contiguous() && post.scalar_type() == at::kFloat && post.numel() == T * 4, "post must be fp32 [T, 4]");
  TORCH_CHECK(comb.is_contiguous() && comb.scalar_type() == at::kFloat && comb.numel() == T * 16, "comb must be fp32 [T, 4, 4]");
}

at::Tensor merge_forward_cuda(const at::Tensor &x, const at::Tensor &out, const at::Tensor &post,
                              const at::Tensor &comb, bool cast) {
  check_stream_args(x, out, post, comb);
  const c10::cuda::CUDAGuard guard(x.device());
  const int64_t T = x.size(0), H = x.size(2);
  at::Tensor res = at::empty({T, 4, H}, x.options());
  if (T == 0) return res;
  auto st = at::cuda::getCurrentCUDAStream().stream();
  constexpr int V = MERGE_VEC;
  const unsigned grid = (unsigned)std::min<int64_t>((T * (H / V) + 255) / 256, 1 << 20);
  auto args = std::make_tuple(reinterpret_cast<const __nv_bfloat16 *>(x.data_ptr()),
                              reinterpret_cast<const __nv_bfloat16 *>(out.data_ptr()), post.data_ptr<float>(),
                              comb.data_ptr<float>(), reinterpret_cast<__nv_bfloat16 *>(res.data_ptr()), (int)T, (int)H);
  if (cast) std::apply([&](auto... a) { merge_fwd_kernel<true, V><<<grid, 256, 0, st>>>(a...); }, args);
  else std::apply([&](auto... a) { merge_fwd_kernel<false, V><<<grid, 256, 0, st>>>(a...); }, args);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return res;
}

// returns (dx [T,4,H] bf16, dout [T,H] bf16, dpost [T,4] fp32, dcomb [T,4,4] fp32)
std::vector<at::Tensor> merge_backward_cuda(const at::Tensor &g, const at::Tensor &x, const at::Tensor &out,
                                            const at::Tensor &post, const at::Tensor &comb, bool cast) {
  check_stream_args(x, out, post, comb);
  TORCH_CHECK(g.is_contiguous() && g.sizes() == x.sizes() && g.scalar_type() == at::kBFloat16, "g must match streams");
  const c10::cuda::CUDAGuard guard(x.device());
  const int64_t T = x.size(0), H = x.size(2);
  at::Tensor dx = at::empty_like(x), dout = at::empty_like(out);
  at::Tensor dpost = at::empty({T, 4}, post.options()), dcomb = at::empty({T, 4, 4}, comb.options());
  if (T == 0) return {dx, dout, dpost, dcomb};
  auto st = at::cuda::getCurrentCUDAStream().stream();
  auto args = std::make_tuple(reinterpret_cast<const __nv_bfloat16 *>(g.data_ptr()),
                              reinterpret_cast<const __nv_bfloat16 *>(x.data_ptr()),
                              reinterpret_cast<const __nv_bfloat16 *>(out.data_ptr()), post.data_ptr<float>(),
                              comb.data_ptr<float>(), reinterpret_cast<__nv_bfloat16 *>(dx.data_ptr()),
                              reinterpret_cast<__nv_bfloat16 *>(dout.data_ptr()), dpost.data_ptr<float>(),
                              dcomb.data_ptr<float>(), (int)T, (int)H);
  if (cast) std::apply([&](auto... a) { merge_bwd_kernel<true><<<(unsigned)T, 128, 0, st>>>(a...); }, args);
  else std::apply([&](auto... a) { merge_bwd_kernel<false><<<(unsigned)T, 128, 0, st>>>(a...); }, args);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dx, dout, dpost, dcomb};
}

at::Tensor collapse_backward_dx_cuda(const at::Tensor &dcol, const at::Tensor &pre) {
  TORCH_CHECK(dcol.is_cuda() && dcol.is_contiguous() && dcol.dim() == 2 && dcol.scalar_type() == at::kBFloat16, "bad dcol");
  const int64_t T = dcol.size(0), H = dcol.size(1);
  TORCH_CHECK(H % 8 == 0 && pre.is_contiguous() && pre.scalar_type() == at::kFloat && pre.numel() == T * 4, "bad pre");
  const c10::cuda::CUDAGuard guard(dcol.device());
  at::Tensor dx = at::empty({T, 4, H}, dcol.options());
  if (T == 0) return dx;
  auto st = at::cuda::getCurrentCUDAStream().stream();
  collapse_bwd_dx_kernel<<<(unsigned)std::min<int64_t>((T * (H / 8) + 255) / 256, 1 << 20), 256, 0, st>>>(
      reinterpret_cast<const __nv_bfloat16 *>(dcol.data_ptr()), pre.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16 *>(dx.data_ptr()), (int)T, (int)H);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return dx;
}

}  // namespace mhc_host
