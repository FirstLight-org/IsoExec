// SPDX-License-Identifier: Apache-2.0
// Forward launchers: validated tensors in, one CTA launch out.
#include "fwd.cuh"
#include "plan.h"
#include "launch.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

namespace norms_sm90 {
namespace {

struct FwdArgs {
  const __nv_bfloat16 *x, *r;
  int64_t sx, sr;
  const void *w;
  __nv_bfloat16 *hout, *out;
  float *rstd;
  int64_t M;
  float eps;
};

template <int K, int L, int MODE, bool WF, bool RES>
void run_rms(const FwdArgs &a) {
  int th = fwd_threads({L, K}, a.M), rpb = th / L;
  int64_t blocks = (a.M + rpb - 1) / rpb;
  if constexpr (RES)
    add_rms_fwd<K, L, MODE, WF, true><<<dim3(blocks), th, 0, at::cuda::getCurrentCUDAStream()>>>(
        a.x, a.sx, a.r, a.sr, a.w, a.hout, a.out, a.rstd, a.M, rpb, a.eps);
  else
    rms_fwd<K, L, MODE, WF, RES, true><<<dim3(blocks), th, 0, at::cuda::getCurrentCUDAStream()>>>(
        a.x, a.sx, a.r, a.sr, a.w, a.hout, a.out, a.rstd, a.M, rpb, a.eps);
}

struct GatedArgs {
  const __nv_bfloat16 *x, *z;
  int64_t sx, szt, szh;
  const void *w;
  __nv_bfloat16 *out;
  float *rstd;
  int64_t M;
  int hv;
  float eps;
  bool zaligned;
};

template <int K, int L, int MODE, bool WF>
void run_gated(const GatedArgs &a) {
  int th = fwd_threads({L, K}, a.M), rpb = th / L;
  int64_t blocks = (a.M + rpb - 1) / rpb;
  auto stream = at::cuda::getCurrentCUDAStream();
  if (a.zaligned)
    gated_fwd<K, L, MODE, WF, true, true><<<dim3(blocks), th, 0, stream>>>(
        a.x, a.sx, a.z, a.szt, a.szh, a.w, a.out, a.rstd, a.M, a.hv, rpb, a.eps);
  else
    gated_fwd<K, L, MODE, WF, true, false><<<dim3(blocks), th, 0, stream>>>(
        a.x, a.sx, a.z, a.szt, a.szh, a.w, a.out, a.rstd, a.M, a.hv, rpb, a.eps);
}

// Instantiated widths: L=32 for N in {256,512,768,1024,2048,4096,5120,6144,8192}; L=16 for {128,384}.
#define FWD_WIDTHS(X)                                                                              \
  X(1, 32) X(2, 32) X(3, 32) X(4, 32) X(8, 32) X(16, 32) X(20, 32) X(24, 32) X(32, 32) X(1, 16)    \
  X(3, 16)
// The GDN value head width is 128; 384 and 512 exist for the test battery.
#define GATED_WIDTHS(X) X(1, 16) X(3, 16) X(2, 32)

} // namespace

void launch_rms_fwd(const at::Tensor &x, const at::Tensor &r, const at::Tensor &w,
                    const at::Tensor &hout, const at::Tensor &out, const at::Tensor &rstd,
                    double eps, int mode) {
  Lanes l = fwd_lanes(x.size(1));
  FwdArgs a{bf16_ptr(x),
            r.defined() ? bf16_ptr(r) : nullptr,
            x.stride(0),
            r.defined() ? r.stride(0) : 0,
            w.defined() ? w.data_ptr() : nullptr,
            hout.defined() ? bf16_mut(hout) : nullptr,
            bf16_mut(out),
            rstd.data_ptr<float>(),
            x.size(0),
            float(eps)};
  bool wf = w.defined() && w.scalar_type() == at::kFloat, res = r.defined();
  TORCH_CHECK(mode == 1 || mode == 2, "norms_sm90: rms mode must be 1 (plain) or 2 (1+w)");
#define X(KK, LL)                                                                                    \
  if (l.K == KK && l.L == LL) {                                                                      \
    if (mode == 1) {                                                                               \
      if (wf) { if (res) run_rms<KK, LL, 1, true, true>(a); else run_rms<KK, LL, 1, true, false>(a); } \
      else { if (res) run_rms<KK, LL, 1, false, true>(a); else run_rms<KK, LL, 1, false, false>(a); }  \
    } else {                                                                                       \
      if (wf) { if (res) run_rms<KK, LL, 2, true, true>(a); else run_rms<KK, LL, 2, true, false>(a); } \
      else { if (res) run_rms<KK, LL, 2, false, true>(a); else run_rms<KK, LL, 2, false, false>(a); }  \
    }                                                                                              \
    C10_CUDA_KERNEL_LAUNCH_CHECK();                                                                \
    return;                                                                                        \
  }
  FWD_WIDTHS(X)
#undef X
  TORCH_CHECK(false, "norms_sm90: width ", x.size(1), " has no instantiated forward kernel");
}

void launch_gated_fwd(const at::Tensor &x, const at::Tensor &z, const at::Tensor &w,
                      const at::Tensor &out, const at::Tensor &rstd, double eps, int mode) {
  // x [M, N] (row stride free), z [T, hv, N] (token and head strides free).
  Lanes l = fwd_lanes(x.size(1));
  GatedArgs a{bf16_ptr(x), bf16_ptr(z),          x.stride(0), z.stride(0), z.stride(1),
              w.defined() ? w.data_ptr() : nullptr, bf16_mut(out), rstd.data_ptr<float>(),
              x.size(0),   int(z.size(1)),       float(eps),
              aligned16(z)};
  bool wf = w.defined() && w.scalar_type() == at::kFloat;
#define X(KK, LL)                                                                                    \
  if (l.K == KK && l.L == LL) {                                                                      \
    if (mode == 0) run_gated<KK, LL, 0, false>(a);                                                   \
    else if (mode == 1) { if (wf) run_gated<KK, LL, 1, true>(a); else run_gated<KK, LL, 1, false>(a); } \
    else { if (wf) run_gated<KK, LL, 2, true>(a); else run_gated<KK, LL, 2, false>(a); }               \
    C10_CUDA_KERNEL_LAUNCH_CHECK();                                                                \
    return;                                                                                        \
  }
  GATED_WIDTHS(X)
#undef X
  TORCH_CHECK(false, "norms_sm90: width ", x.size(1), " has no instantiated gated kernel");
}

} // namespace norms_sm90
