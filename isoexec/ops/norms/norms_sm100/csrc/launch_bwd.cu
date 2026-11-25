// SPDX-License-Identifier: Apache-2.0
#include "bwd.cuh"
#include "launch.h"
#include "plan.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

namespace norms_sm100 {
namespace {

template <int CPT, int UNROLL>
__global__ __launch_bounds__(256, 1) void
fold_dw_bands(const float *__restrict__ partials, float *__restrict__ dw, int64_t bands,
              int64_t width) {
  int64_t col = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) * CPT;
  if (col >= width)
    return;
  float sum[CPT];
#pragma unroll
  for (int c = 0; c < CPT; ++c)
    sum[c] = 0.f;
  int64_t band = 0;
  for (; band + UNROLL <= bands; band += UNROLL) {
    float t[UNROLL][CPT];
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      const float *p = partials + (band + u) * width + col;
      if constexpr (CPT == 4) {
        float4 q = *reinterpret_cast<const float4 *>(p);
        t[u][0] = q.x; t[u][1] = q.y; t[u][2] = q.z; t[u][3] = q.w;
      } else if constexpr (CPT == 2) {
        float2 q = *reinterpret_cast<const float2 *>(p);
        t[u][0] = q.x; t[u][1] = q.y;
      } else {
#pragma unroll
        for (int c = 0; c < CPT; ++c)
          t[u][c] = p[c];
      }
    }
#pragma unroll
    for (int u = 0; u < UNROLL; ++u)
#pragma unroll
      for (int c = 0; c < CPT; ++c)
        sum[c] = __fadd_rn(sum[c], t[u][c]);
  }
  for (; band < bands; ++band) {
    const float *p = partials + band * width + col;
#pragma unroll
    for (int c = 0; c < CPT; ++c)
      sum[c] = __fadd_rn(sum[c], p[c]);
  }
#pragma unroll
  for (int c = 0; c < CPT; ++c)
    if (col + c < width)
      dw[col + c] = sum[c];
}

template <int TPB, int STAGE>
__global__ __launch_bounds__(TPB, 1) void
fold_dw_staged(const float *__restrict__ partials, float *__restrict__ dw, int64_t bands,
               int64_t width) {
  constexpr int COLS = 32, PER = STAGE * COLS / TPB;
  __shared__ float st[STAGE * COLS];
  int64_t col0 = int64_t(blockIdx.x) * COLS;
  int tid = threadIdx.x;
  int live = tid < COLS && col0 + tid < width;
  float sum = 0.f;
  for (int64_t b0 = 0; b0 < bands; b0 += STAGE) {
    int n = int(bands - b0 < STAGE ? bands - b0 : STAGE);
    float v[PER];
#pragma unroll
    for (int u = 0; u < PER; ++u) {
      int idx = u * TPB + tid, sb = idx / COLS, c = idx % COLS;
      v[u] = (idx < n * COLS && col0 + c < width) ? partials[(b0 + sb) * width + col0 + c] : 0.f;
    }
    __syncthreads(); // the previous stage's reads are done before this stage overwrites the buffer
#pragma unroll
    for (int u = 0; u < PER; ++u)
      st[u * TPB + tid] = v[u];
    __syncthreads();
    if (live) {
#pragma unroll 8
      for (int sb = 0; sb < n; ++sb)
        sum = __fadd_rn(sum, st[sb * COLS + tid]);
    }
  }
  if (live)
    dw[col0 + tid] = sum;
}

struct RmsBwdArgs {
  const __nv_bfloat16 *dy, *x, *dres;
  int64_t sdy, sx, sdres;
  const float *rstd;
  const void *w;
  __nv_bfloat16 *dx;
  float *dwp;
  int64_t M, bands;
  int iters;
};

template <int K, int L, int MODE, bool WF, bool RES>
void run_rms_bwd(const RmsBwdArgs &a) {
  auto stream = at::cuda::getCurrentCUDAStream();
  const int var = bwd_var({L, K}, a.M);
  if constexpr (8 * K >= 16) {
    if (var == BWD_STREAM) {
      rms_bwd<K, L, MODE, WF, RES, BWD_STREAM><<<dim3(a.bands), 128, 0, stream>>>(
          a.dy, a.sdy, a.x, a.sx, a.rstd, a.w, a.dres, a.sdres, a.dx, a.dwp, a.M, a.iters);
      return;
    }
  }
  if constexpr (L == 128 && K <= 5) {
    if (var == BWD_RING) {
      rms_bwd<K, L, MODE, WF, RES, BWD_RING><<<dim3(a.bands), 128, 0, stream>>>(
          a.dy, a.sdy, a.x, a.sx, a.rstd, a.w, a.dres, a.sdres, a.dx, a.dwp, a.M, a.iters);
      return;
    }
  }
  rms_bwd<K, L, MODE, WF, RES, BWD_KEEP><<<dim3(a.bands), 128, 0, stream>>>(
      a.dy, a.sdy, a.x, a.sx, a.rstd, a.w, a.dres, a.sdres, a.dx, a.dwp, a.M, a.iters);
}

struct GatedBwdArgs {
  const __nv_bfloat16 *dy, *x, *z;
  int64_t sdy, sx, szt, szh;
  const float *rstd;
  const void *w;
  __nv_bfloat16 *dx, *dz;
  float *dwp;
  int64_t M, bands;
  int hv, iters;
  bool zaligned;
};

template <int K, int L, int MODE, bool WF>
void run_gated_bwd(const GatedBwdArgs &a) {
  auto stream = at::cuda::getCurrentCUDAStream();
  if (a.zaligned)
    gated_bwd<K, L, MODE, WF, true><<<dim3(a.bands), 128, 0, stream>>>(
        a.dy, a.sdy, a.x, a.sx, a.z, a.szt, a.szh, a.rstd, a.w, a.dx, a.dz, a.dwp, a.M, a.hv, a.iters);
  else
    gated_bwd<K, L, MODE, WF, false><<<dim3(a.bands), 128, 0, stream>>>(
        a.dy, a.sdy, a.x, a.sx, a.z, a.szt, a.szh, a.rstd, a.w, a.dx, a.dz, a.dwp, a.M, a.hv, a.iters);
}

#define BWD_WIDTHS(X)                                                                              \
  X(1, 128) X(2, 128) X(3, 128) X(4, 128) X(5, 128) X(6, 128) X(8, 128) X(1, 32) X(2, 32)          \
  X(3, 32) X(5, 32) X(6, 32) X(7, 32) X(1, 16) X(3, 16) X(7, 128)
#define GATED_BWD_WIDTHS(X) X(1, 16) X(3, 16) X(2, 32)

} // namespace

void launch_dw_fold(const at::Tensor &partials, const at::Tensor &dw) {
  int64_t width = dw.numel(), bands = partials.size(0);
  FoldPlan p = fold_plan(bands, width);
  auto stream = at::cuda::getCurrentCUDAStream();
  const float *pp = partials.data_ptr<float>();
  float *dp = dw.data_ptr<float>();
  if (p.mode == FOLD_SHARED) {
    int64_t sblocks = (width + 31) / 32; // 32 columns per CTA, whatever its thread count
#define STAGED(TPB, STAGE)                                                                         \
  fold_dw_staged<TPB, STAGE><<<dim3(sblocks), TPB, 0, stream>>>(pp, dp, bands, width);             \
  C10_CUDA_KERNEL_LAUNCH_CHECK();                                                                  \
  return;
    if (p.threads >= 256) {
      if (p.unroll >= 256) { STAGED(256, 256) }
      if (p.unroll >= 128) { STAGED(256, 128) }
      STAGED(256, 64)
    }
    if (p.unroll >= 256) { STAGED(128, 256) }
    if (p.unroll >= 128) { STAGED(128, 128) }
    STAGED(128, 64)
#undef STAGED
  }
  int64_t lanes = (width + p.cpt - 1) / p.cpt;
  int64_t blocks = (lanes + p.threads - 1) / p.threads;
#define FOLD(CPT, UNROLL)                                                                          \
  fold_dw_bands<CPT, UNROLL><<<dim3(blocks), p.threads, 0, stream>>>(pp, dp, bands, width);        \
  C10_CUDA_KERNEL_LAUNCH_CHECK();                                                                  \
  return;
#define FOLD_U(CPT)                                                                                \
  if (p.unroll >= 64 && CPT == 1) { FOLD(CPT, 64) }                                                \
  if (p.unroll >= 32) { FOLD(CPT, 32) }                                                            \
  if (p.unroll >= 16) { FOLD(CPT, 16) }                                                            \
  if (p.unroll >= 8) { FOLD(CPT, 8) }                                                              \
  if (p.unroll >= 4) { FOLD(CPT, 4) }                                                              \
  FOLD(CPT, 1)
  if (p.cpt == 4 && width % 4 == 0) { FOLD_U(4) }
  if (p.cpt == 2 && width % 2 == 0) { FOLD_U(2) }
  FOLD_U(1)
#undef FOLD_U
#undef FOLD
}

void launch_rms_bwd(const at::Tensor &dy, const at::Tensor &x, const at::Tensor &rstd,
                    const at::Tensor &w, const at::Tensor &dres, const at::Tensor &dx,
                    const at::Tensor &dwp, int mode, int64_t bands, int iters) {
  Lanes l = bwd_lanes(x.size(1));
  RmsBwdArgs a{bf16_ptr(dy),
               bf16_ptr(x),
               dres.defined() ? bf16_ptr(dres) : nullptr,
               dy.stride(0),
               x.stride(0),
               dres.defined() ? dres.stride(0) : 0,
               rstd.data_ptr<float>(),
               w.defined() ? w.data_ptr() : nullptr,
               bf16_mut(dx),
               dwp.defined() ? dwp.data_ptr<float>() : nullptr,
               x.size(0),
               bands,
               iters};
  bool wf = w.defined() && w.scalar_type() == at::kFloat, res = dres.defined();
  TORCH_CHECK(mode == 1 || mode == 2, "norms_sm100: rms mode must be 1 (plain) or 2 (1+w)");
#define X(KK, LL)                                                                                  \
  if (l.K == KK && l.L == LL) {                                                                    \
    if (mode == 1) {                                                                               \
      if (wf) { if (res) run_rms_bwd<KK, LL, 1, true, true>(a); else run_rms_bwd<KK, LL, 1, true, false>(a); } \
      else { if (res) run_rms_bwd<KK, LL, 1, false, true>(a); else run_rms_bwd<KK, LL, 1, false, false>(a); }  \
    } else {                                                                                       \
      if (wf) { if (res) run_rms_bwd<KK, LL, 2, true, true>(a); else run_rms_bwd<KK, LL, 2, true, false>(a); } \
      else { if (res) run_rms_bwd<KK, LL, 2, false, true>(a); else run_rms_bwd<KK, LL, 2, false, false>(a); }  \
    }                                                                                              \
    C10_CUDA_KERNEL_LAUNCH_CHECK();                                                                \
    return;                                                                                        \
  }
  BWD_WIDTHS(X)
#undef X
  TORCH_CHECK(false, "norms_sm100: width ", x.size(1), " has no instantiated backward kernel");
}

void launch_gated_bwd(const at::Tensor &dy, const at::Tensor &x, const at::Tensor &z,
                      const at::Tensor &rstd, const at::Tensor &w, const at::Tensor &dx,
                      const at::Tensor &dz, const at::Tensor &dwp, int mode, int64_t bands,
                      int iters) {
  Lanes l = bwd_lanes(x.size(1));
  GatedBwdArgs a{bf16_ptr(dy),
                 bf16_ptr(x),
                 bf16_ptr(z),
                 dy.stride(0),
                 x.stride(0),
                 z.stride(0),
                 z.stride(1),
                 rstd.data_ptr<float>(),
                 w.defined() ? w.data_ptr() : nullptr,
                 bf16_mut(dx),
                 bf16_mut(dz),
                 dwp.defined() ? dwp.data_ptr<float>() : nullptr,
                 x.size(0),
                 bands,
                 int(z.size(1)),
                 iters,
                 aligned16(z)};
  bool wf = w.defined() && w.scalar_type() == at::kFloat;
#define X(KK, LL)                                                                                  \
  if (l.K == KK && l.L == LL) {                                                                    \
    if (mode == 0) run_gated_bwd<KK, LL, 0, false>(a);                                             \
    else if (mode == 1) { if (wf) run_gated_bwd<KK, LL, 1, true>(a); else run_gated_bwd<KK, LL, 1, false>(a); } \
    else { if (wf) run_gated_bwd<KK, LL, 2, true>(a); else run_gated_bwd<KK, LL, 2, false>(a); }   \
    C10_CUDA_KERNEL_LAUNCH_CHECK();                                                                \
    return;                                                                                        \
  }
  GATED_BWD_WIDTHS(X)
#undef X
  TORCH_CHECK(false, "norms_sm100: width ", x.size(1), " has no instantiated gated backward kernel");
}

} // namespace norms_sm100
