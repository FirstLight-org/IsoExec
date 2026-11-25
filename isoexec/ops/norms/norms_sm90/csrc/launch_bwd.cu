// SPDX-License-Identifier: Apache-2.0
// Backward launchers.
#include "bwd.cuh"
#include "launch.h"
#include "plan.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

namespace norms_sm90 {
namespace {

__global__ void fold_dw_bands(const float *__restrict__ partials, float *__restrict__ dw,
                              int64_t bands, int64_t width) {
  int64_t col = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (col >= width)
    return;
  float sum = 0.f;
  for (int64_t band = 0; band < bands; ++band)
    sum = __fadd_rn(sum, partials[band * width + col]);
  dw[col] = sum;
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

template <int K, int L, int MODE, bool WF, int RES>
void run_rms_bwd(const RmsBwdArgs &a) {
  rms_bwd<K, L, MODE, WF, RES><<<dim3(a.bands), 128, 0, at::cuda::getCurrentCUDAStream()>>>(
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

// L=128: N in {1024,2048,3072,4096,5120,6144,8192}; L=32: {256,512,768,1280,1536,1792};
// L=16: {128,384}.
#define BWD_WIDTHS(X)                                                                              \
  X(1, 128) X(2, 128) X(3, 128) X(4, 128) X(5, 128) X(6, 128) X(8, 128) X(1, 32) X(2, 32)          \
  X(3, 32) X(5, 32) X(6, 32) X(7, 32) X(1, 16) X(3, 16)
#define GATED_BWD_WIDTHS(X) X(1, 16) X(3, 16) X(2, 32)

} // namespace

void launch_dw_fold(const at::Tensor &partials, const at::Tensor &dw) {
  int64_t width = dw.numel();
  fold_dw_bands<<<(width + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      partials.data_ptr<float>(), dw.data_ptr<float>(), partials.size(0), width);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_rms_bwd(const at::Tensor &dy, const at::Tensor &x, const at::Tensor &rstd,
                    const at::Tensor &w, const at::Tensor &dres, const at::Tensor &dx,
                    const at::Tensor &dwp, int mode, int64_t bands, int iters, bool dres_after_round) {
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
  bool wf = w.defined() && w.scalar_type() == at::kFloat;
  int res = !dres.defined() ? 0 : dres_after_round ? 2 : 1;
  TORCH_CHECK(mode == 1 || mode == 2, "norms_sm90: rms mode must be 1 (plain) or 2 (1+w)");
#define RUN_RES(KK, LL, MM, WW)                                                                    \
  if (res == 0) run_rms_bwd<KK, LL, MM, WW, 0>(a);                                                 \
  else if (res == 1) run_rms_bwd<KK, LL, MM, WW, 1>(a);                                            \
  else run_rms_bwd<KK, LL, MM, WW, 2>(a);
#define X(KK, LL)                                                                                  \
  if (l.K == KK && l.L == LL) {                                                                    \
    if (mode == 1) {                                                                               \
      if (wf) { RUN_RES(KK, LL, 1, true) } else { RUN_RES(KK, LL, 1, false) }                      \
    } else {                                                                                       \
      if (wf) { RUN_RES(KK, LL, 2, true) } else { RUN_RES(KK, LL, 2, false) }                      \
    }                                                                                              \
    C10_CUDA_KERNEL_LAUNCH_CHECK();                                                                \
    return;                                                                                        \
  }
  BWD_WIDTHS(X)
#undef X
#undef RUN_RES
  TORCH_CHECK(false, "norms_sm90: width ", x.size(1), " has no instantiated backward kernel");
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
  TORCH_CHECK(false, "norms_sm90: width ", x.size(1), " has no instantiated gated backward kernel");
}

} // namespace norms_sm90
