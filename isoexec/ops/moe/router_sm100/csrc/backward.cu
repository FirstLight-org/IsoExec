// SPDX-License-Identifier: Apache-2.0
//
// The router backward (DESIGN.md "Backward"): no library GEMM, no atomics on a value, every
// reduction a declared order.
//
//   bwd_select   one warp per token: the FORWARD's selection core recomputed from the logits (so the
//                selection, the rank order and D are the forward's bits), then the scalar backward of
//                the weights in slot order -> dl [T, k] (the logit gradient at the selected experts;
//                every other expert and the bias get exactly zero, nothing flows through selection).
//   dx           dX[t, h] = chain over the token's k slots in ascending expert id of
//                RN(dl[t, j] * W[e_j, h]), from +0.0, rounded once to bf16. Rows independent.
//   dw_partial   dW over the token axis: segments of SEG_T canonical rows; per (segment, e, h) a chain
//                from +0.0 over the segment's tokens that selected e, ascending token, of
//                RN(dl * x[t, h]); plus an integer count (empty-segment flag).
//   dw_tree      the balanced adjacent-pair tree over segment index, empty segments skipped by the
//                integer count, an unpaired node carried verbatim; fp32 root (= the fp32 main grad).
#include "blackwell.cuh"
#include "select.cuh"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cub/block/block_scan.cuh>

namespace rt100 {
namespace {

template <int MODE> __device__ __forceinline__ float dscore(float x, float s) {
  if constexpr (MODE == kSigmoid) {
    // d sigmoid / dx = s * (1 - s), two roundings
    return __fmul_rn(s, __fsub_rn(1.f, s));
  } else {
    // d sqrt(softplus(x)) / dx = sigmoid(x) / (2 s) on the log1p branch, 1 / (2 s) on x > 20;
    // returned as the numerator factor; the caller divides by 2s (the s == 0 guard lives there).
    return x > 20.f ? 1.f : 1.f / __fadd_rn(1.f, expf(-x));
  }
}

template <int MODE, int K, int EPL>
__global__ __launch_bounds__(256) void bwd_select_kernel(const float *__restrict__ logits,
                                                         int64_t stride, int T, SelectArgs a,
                                                         const float *__restrict__ dp,
                                                         float *__restrict__ dl,
                                                         int *__restrict__ oidx) {
  const int64_t t = int64_t(blockIdx.x) * 8 + int(threadIdx.x) / 32;
  const int lane = int(threadIdx.x & 31u);
  const bool valid = t < T;
  const int e0 = lane * EPL;
  float v[1][EPL];
  load_lane<EPL>(valid ? logits + t * stride + e0 : logits, valid ? max(0, min(EPL, a.E - e0)) : 0,
                 v[0]);
  const bool vd[1] = {valid};
  uint32_t win[1][K];
  float sw[1][K];
  select_core<MODE, K, EPL, 1>(v, vd, a, win, sw);
  if (!valid)
    return;
  const int KS = K + a.S;
  const bool active = a.mask ? (a.mask[t] != 0) : true;
  // slot order (ascending expert id): sq[q], dq[q], lq[q] for q = 0..K-1
  float sq[K], dq[K], lq[K];
#pragma unroll
  for (int q = 0; q < K; ++q) {
    sq[q] = 0.f;
    lq[q] = 0.f;
  }
#pragma unroll
  for (int j = 0; j < K; ++j) {
    int slot = 0;
#pragma unroll
    for (int l = 0; l < K; ++l)
      slot += win[0][l] < win[0][j];
#pragma unroll
    for (int q = 0; q < K; ++q)
      if (q == slot)
        sq[q] = sw[0][j];
  }
  // the selected logits in slot order, fetched from global (one lane's row is the token's row)
#pragma unroll
  for (int q = 0; q < K; ++q) {
    uint32_t id = 0u;
#pragma unroll
    for (int j = 0; j < K; ++j) {
      int slot = 0;
#pragma unroll
      for (int l = 0; l < K; ++l)
        slot += win[0][l] < win[0][j];
      id = slot == q ? win[0][j] : id;
    }
    lq[q] = logits[t * stride + id];
    dq[q] = dp[t * K + q];
  }
  // D: the forward's renormaliser, bit for bit (the ascending-expert-id chain from +0, + 1e-20)
  const float D = __fadd_rn(selected_sum<K>(sw[0], win[0]), 1e-20f);
  // A = sum_q RN(dp_q * s_q), an ascending chain in slot order from +0.0
  float A = 0.f;
#pragma unroll
  for (int q = 0; q < K; ++q)
    A = __fadd_rn(A, __fmul_rn(dq[q], sq[q]));
  const float DD = __fmul_rn(D, D);
  if (lane < K) {
    float s = 0.f, d = 0.f, l = 0.f;
#pragma unroll
    for (int q = 0; q < K; ++q) {
      s = lane == q ? sq[q] : s;
      d = lane == q ? dq[q] : d;
      l = lane == q ? lq[q] : l;
    }
    // ds = c * ((dp * D - A) / (D * D))
    const float ds = __fmul_rn(a.scale, __fsub_rn(__fmul_rn(d, D), A) / DD);
    float g;
    if constexpr (MODE == kSigmoid) {
      g = __fmul_rn(ds, dscore<MODE>(l, s));
    } else {
      // dl = (ds * sigmoid(l)) / (2 s); s == 0 -> 0 (the limit; declared)
      g = s == 0.f ? 0.f : __fmul_rn(ds, dscore<MODE>(l, s)) / __fmul_rn(2.f, s);
    }
    dl[t * K + lane] = active ? g : 0.f;
    // the forward table's ids, slot q = lane (ascending expert id)
    uint32_t id = 0u;
#pragma unroll
    for (int j = 0; j < K; ++j) {
      int slot = 0;
#pragma unroll
      for (int l = 0; l < K; ++l)
        slot += win[0][l] < win[0][j];
      id = slot == lane ? win[0][j] : id;
    }
    oidx[t * KS + lane] = active ? int(id) : -1;
  } else if (lane < KS) {
    oidx[t * KS + lane] = active ? a.E + (lane - K) : -1;
  }
}

//: dX: one CTA per token; a thread owns VW = 4 consecutive columns of every 1024-column stripe, so
//: all of a token's k * H/4 weight loads are independent and only the k-term chain per column is
//: serial. Chain from +0.0 in slot order (ascending expert id), RN(dl * W) then RN(+), one bf16
//: rounding at the end.
template <int K>
__global__ __launch_bounds__(256) void dx_kernel(const int *__restrict__ idx, int KS,
                                                 const float *__restrict__ dl,
                                                 const __nv_bfloat16 *__restrict__ W, int64_t wstride,
                                                 __nv_bfloat16 *__restrict__ dx, int64_t xstride,
                                                 int T, int H, int E) {
  const int64_t t = blockIdx.x;
  int e[K];
  float g[K];
#pragma unroll
  for (int j = 0; j < K; ++j) {
    e[j] = idx[t * KS + j];
    g[j] = dl[t * K + j];
  }
  for (int h = int(threadIdx.x) * 8; h < H; h += 2048) {
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    uint4 wv[K];
#pragma unroll
    for (int j = 0; j < K; ++j)
      wv[j] = (e[j] >= 0 && e[j] < E) ? *reinterpret_cast<const uint4 *>(W + int64_t(e[j]) * wstride + h)
                                      : make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
    for (int j = 0; j < K; ++j) {
      if (e[j] < 0 || e[j] >= E)
        continue;                    // a masked token has no slot: skipped, never multiplied by 0
      const __nv_bfloat16 *wb = reinterpret_cast<const __nv_bfloat16 *>(&wv[j]);
#pragma unroll
      for (int u = 0; u < 8; ++u)
        acc[u] = __fadd_rn(acc[u], __fmul_rn(g[j], __bfloat162float(wb[u])));
    }
    __nv_bfloat16 o[8];
#pragma unroll
    for (int u = 0; u < 8; ++u)
      o[u] = __float2bfloat16_rn(acc[u]);
    *reinterpret_cast<uint4 *>(dx + t * xstride + h) = *reinterpret_cast<uint4 *>(o);
  }
}

//: dW partial of one (segment, expert). The CTA first compacts the segment's routes that name
//: expert e into a list in ASCENDING TOKEN order -- a ballot and an exclusive prefix over the
//: warps, an exact integer position, no atomic -- then a thread owns 8-column groups
//: h = 8 * (threadIdx.x + 256 i) and runs the chain over the list from +0.0. Grid x = expert fastest,
//: so a segment's x rows stay L2-resident while all its experts read them.
template <int K>
__global__ __launch_bounds__(256) void dw_partial_kernel(const int *__restrict__ idx, int KS,
                                                         const float *__restrict__ dl,
                                                         const __nv_bfloat16 *__restrict__ X,
                                                         int64_t xstride, int T, int H, int E,
                                                         int seg_t, float *__restrict__ part,
                                                         int *__restrict__ cnt) {
  const int seg = blockIdx.y, e = blockIdx.x;
  const int t0 = seg * seg_t, t1 = min(T, t0 + seg_t);
  __shared__ int stok[1024];
  __shared__ float sdl[1024];
  __shared__ int wsum[8];
  __shared__ int total;
  const int warp = int(threadIdx.x) / 32, lane = int(threadIdx.x) % 32;
  if (threadIdx.x == 0)
    total = 0;
  __syncthreads();
  for (int c0 = t0; c0 < t1; c0 += 256) {
    const int t = c0 + int(threadIdx.x);
    float g = 0.f;
    bool hit = false;
    if (t < t1) {
#pragma unroll
      for (int j = 0; j < K; ++j)
        if (idx[int64_t(t) * KS + j] == e) {
          hit = true;
          g = dl[int64_t(t) * K + j];
        }
    }
    const unsigned m = __ballot_sync(0xffffffffu, hit);
    if (lane == 0)
      wsum[warp] = __popc(m);
    __syncthreads();
    int before = total;
    for (int w = 0; w < warp; ++w)
      before += wsum[w];
    if (hit) {
      const int pos = before + __popc(m & ((1u << lane) - 1u));
      stok[pos] = t;
      sdl[pos] = g;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      int add = 0;
      for (int w = 0; w < 8; ++w)
        add += wsum[w];
      total += add;
    }
    __syncthreads();
  }
  const int n = total;
  if (threadIdx.x == 0)
    cnt[seg * E + e] = n;
  float *dst = part + (int64_t(seg) * E + e) * H;
  constexpr int kGroups = 4;         // 8-column groups a thread carries: H <= 4 * 2048 = 8192
  constexpr int kAhead = 8;          // list entries whose loads are in flight before their adds
  const int G = (H / 8 + 255) / 256;  // groups actually used
  float acc[kGroups][8];
#pragma unroll
  for (int i = 0; i < kGroups; ++i)
#pragma unroll
    for (int u = 0; u < 8; ++u)
      acc[i][u] = 0.f;
  for (int r0 = 0; r0 < n; r0 += kAhead) {
    uint4 xv[kAhead][kGroups];
#pragma unroll
    for (int q = 0; q < kAhead; ++q) {
      const bool ok = r0 + q < n;
      const __nv_bfloat16 *xr = X + int64_t(ok ? stok[r0 + q] : 0) * xstride;
#pragma unroll
      for (int i = 0; i < kGroups; ++i) {
        const int h = 8 * (int(threadIdx.x) + 256 * i);
        xv[q][i] = (ok && i < G && h < H) ? *reinterpret_cast<const uint4 *>(xr + h)
                                          : make_uint4(0u, 0u, 0u, 0u);
      }
    }
#pragma unroll
    for (int q = 0; q < kAhead; ++q) {
      if (r0 + q >= n)
        break;                       // past the list: skipped, never added
      const float g = sdl[r0 + q];
#pragma unroll
      for (int i = 0; i < kGroups; ++i) {
        if (i >= G)
          break;
        const __nv_bfloat16 *xb = reinterpret_cast<const __nv_bfloat16 *>(&xv[q][i]);
#pragma unroll
        for (int u = 0; u < 8; ++u)
          acc[i][u] = __fadd_rn(acc[i][u], __fmul_rn(g, __bfloat162float(xb[u])));
      }
    }
  }
#pragma unroll
  for (int i = 0; i < kGroups; ++i) {
    const int h = 8 * (int(threadIdx.x) + 256 * i);
    if (i < G && h < H) {
      *reinterpret_cast<float4 *>(dst + h) = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
      *reinterpret_cast<float4 *>(dst + h + 4) = make_float4(acc[i][4], acc[i][5], acc[i][6], acc[i][7]);
    }
  }
}

//: the adjacent-pair tree over segments; an empty node (count 0) is skipped, never added. NS is the
//: segment count rounded up to a power of two (the padded slots are empty).
template <int NS>
__global__ __launch_bounds__(256) void dw_tree_kernel(const float *__restrict__ part,
                                                      const int *__restrict__ cnt, int nseg, int E,
                                                      int H, float *__restrict__ dw) {
  const int64_t i = int64_t(blockIdx.x) * 256 + threadIdx.x;
  if (i >= int64_t(E) * H)
    return;
  const int e = int(i / H);
  float v[NS];
  bool full[NS];
#pragma unroll
  for (int s = 0; s < NS; ++s) {
    full[s] = s < nseg && cnt[s * E + e] > 0;
    v[s] = full[s] ? part[int64_t(s) * E * H + i] : 0.f;
  }
  int n = nseg;
#pragma unroll
  for (int width = NS; width > 1; width /= 2) {
    const int m = (n + 1) / 2;
#pragma unroll
    for (int s = 0; s < width / 2; ++s) {
      const int l = 2 * s, r = 2 * s + 1;
      if (s < m) {
        if (r >= n) {                // the unpaired tail is carried verbatim
          v[s] = v[l];
          full[s] = full[l];
        } else if (full[l] && full[r]) {
          v[s] = __fadd_rn(v[l], v[r]);
          full[s] = true;
        } else {
          v[s] = full[l] ? v[l] : v[r];
          full[s] = full[l] || full[r];
        }
      }
    }
    n = m;
  }
  dw[i] = full[0] ? v[0] : 0.f;
}

} // namespace

//: the selection geometry was validated by the forward call that produced `logits` (ops.py runs
//: `route_select_only` on the same arguments first, which refuses what the program cannot serve).
void bwd_select_cuda(const at::Tensor &logits, const c10::optional<at::Tensor> &bias,
                     const c10::optional<at::Tensor> &mask, int64_t mode, int64_t K, int64_t S,
                     int64_t n_group, int64_t topk_group, double scale, const at::Tensor &dp,
                     at::Tensor dl, at::Tensor oidx) {
  TORCH_CHECK(logits.is_cuda() && logits.scalar_type() == at::kFloat && logits.stride(1) == 1 &&
                  dp.is_contiguous() && dl.is_contiguous(),
              "router_sm100 bwd: fp32 logits, contiguous dp and dl");
  const c10::cuda::CUDAGuard guard(logits.device());
  const int T = int(logits.size(0)), E = int(logits.size(1));
  SelectArgs a{bias ? bias->data_ptr<float>() : nullptr,
               mask ? reinterpret_cast<const uint8_t *>(mask->data_ptr<bool>()) : nullptr,
               E, int(n_group), int(topk_group), int(S), float(scale), nullptr, nullptr};
  if (!T)
    return;
  const int blocks = (T + 7) / 8;
  auto stream = at::cuda::getCurrentCUDAStream();
  const int EPL = E <= 64 ? 2 : E <= 128 ? 4 : E <= 256 ? 8 : E <= 384 ? 12 : E <= 512 ? 16 : 32;
  bool done = false;
#define BS(MODE_, K_, EPL_)                                                                  \
  if (!done && mode == MODE_ && K == K_ && EPL == EPL_) {                                    \
    done = true;                                                                             \
    bwd_select_kernel<MODE_, K_, EPL_><<<blocks, 256, 0, stream>>>(                          \
        logits.data_ptr<float>(), logits.stride(0), T, a, dp.data_ptr<float>(),              \
        dl.data_ptr<float>(), oidx.data_ptr<int>());                                         \
  }
  BS(kSigmoid, 4, 2)
  BS(kSigmoid, 8, 8)
  BS(kSigmoid, 8, 12)                       // GLM-5.3-Flash, E = 288
  BS(kSqrtSoftplus, 6, 8)
  BS(kSqrtSoftplus, 6, 12)
  BS(kSqrtSoftplus, 6, 16)
  BS(kSqrtSoftplus, 8, 8)
#undef BS
  TORCH_CHECK(done, "router_sm100 bwd: (routing, E, top_k) not served");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dx_cuda(const at::Tensor &idx, const at::Tensor &dl, const at::Tensor &W, at::Tensor dx) {
  const c10::cuda::CUDAGuard guard(W.device());
  const int T = int(dx.size(0)), H = int(dx.size(1)), E = int(W.size(0)), K = int(dl.size(1));
  TORCH_CHECK(H % 8 == 0 && W.stride(1) == 1 && dx.is_contiguous() && W.stride(0) % 8 == 0,
              "router_sm100 dx: H % 8, unit inner strides");
  if (!T)
    return;
  auto stream = at::cuda::getCurrentCUDAStream();
#define DXK(KK)                                                                              \
  dx_kernel<KK><<<T, 256, 0, stream>>>(idx.data_ptr<int>(), int(idx.size(1)),               \
                                       dl.data_ptr<float>(),                                 \
                                       reinterpret_cast<const __nv_bfloat16 *>(W.data_ptr()), \
                                       W.stride(0),                                          \
                                       reinterpret_cast<__nv_bfloat16 *>(dx.data_ptr()),     \
                                       dx.stride(0), T, H, E)
  switch (K) {
  case 4: DXK(4); break;
  case 6: DXK(6); break;
  case 8: DXK(8); break;
  default: TORCH_CHECK(false, "router_sm100 dx: top_k in {4, 6, 8}");
  }
#undef DXK
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dw_cuda(const at::Tensor &idx, const at::Tensor &dl, const at::Tensor &X, int64_t seg_t,
             at::Tensor part, at::Tensor cnt, at::Tensor dw) {
  const c10::cuda::CUDAGuard guard(X.device());
  const int T = int(X.size(0)), H = int(X.size(1)), E = int(dw.size(0)), K = int(dl.size(1));
  const int nseg = int(part.size(0));
  TORCH_CHECK(nseg <= 128 && E <= 512, "router_sm100 dw: at most 128 token segments, E <= 512");
  TORCH_CHECK(H <= 8192 && H % 8 == 0 && X.stride(1) == 1 && X.stride(0) % 8 == 0,
              "router_sm100 dw: H <= 8192, H % 8, unit inner stride");
  auto stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(seg_t <= 1024, "router_sm100 dw: SEG_T <= 1024");
  if (!T) {
    // No partial/count producer runs for an empty batch. Its weight gradient is +0.
    C10_CUDA_CHECK(cudaMemsetAsync(dw.data_ptr<float>(), 0, dw.numel() * sizeof(float), stream));
    return;
  }
  if (T) {
#define DWP(KK)                                                                              \
  dw_partial_kernel<KK><<<dim3(E, nseg), 256, 0, stream>>>(                                  \
      idx.data_ptr<int>(), int(idx.size(1)), dl.data_ptr<float>(),                           \
      reinterpret_cast<const __nv_bfloat16 *>(X.data_ptr()), X.stride(0), T, H, E, int(seg_t), \
      part.data_ptr<float>(), cnt.data_ptr<int>())
    switch (K) {
    case 4: DWP(4); break;
    case 6: DWP(6); break;
    case 8: DWP(8); break;
    default: TORCH_CHECK(false, "router_sm100 dw: top_k in {4, 6, 8}");
    }
#undef DWP
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  const int64_t n = int64_t(E) * H;
  const int blocks = int((n + 255) / 256);
#define DWT(NS)                                                                              \
  dw_tree_kernel<NS><<<blocks, 256, 0, stream>>>(part.data_ptr<float>(), cnt.data_ptr<int>(), \
                                                 nseg, E, H, dw.data_ptr<float>())
  if (nseg <= 1) DWT(1);
  else if (nseg <= 2) DWT(2);
  else if (nseg <= 4) DWT(4);
  else if (nseg <= 8) DWT(8);
  else if (nseg <= 16) DWT(16);
  else if (nseg <= 32) DWT(32);
  else if (nseg <= 64) DWT(64);
  else DWT(128);
#undef DWT
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace rt100
