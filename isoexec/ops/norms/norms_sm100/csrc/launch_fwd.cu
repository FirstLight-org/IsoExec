// SPDX-License-Identifier: Apache-2.0
#include "fwd.cuh"
#include "plan.h"
#include "launch.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

namespace norms_sm100 {
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
  FwdPlan p = fwd_plan({L, K}, a.M);
  int rpb = p.threads / L;
  int64_t blocks = (a.M + rpb - 1) / rpb;
  auto stream = at::cuda::getCurrentCUDAStream();
#define LAUNCH(V)                                                                                  \
  if constexpr (K == 7 && L == 128 && V == VAR_KEEP)                                               \
    rms_fwd_occ<K, L, MODE, WF, RES, true, V><<<dim3(blocks), p.threads, 0, stream>>>(              \
        a.x, a.sx, a.r, a.sr, a.w, a.hout, a.out, a.rstd, a.M, rpb, a.eps);                        \
  else                                                                                             \
    rms_fwd<K, L, MODE, WF, RES, true, V><<<dim3(blocks), p.threads, 0, stream>>>(                  \
        a.x, a.sx, a.r, a.sr, a.w, a.hout, a.out, a.rstd, a.M, rpb, a.eps);                        \
  return;
  if constexpr (L > 32) {
    constexpr bool OCC = RES && K == 7 && L == 128;
    if (p.var == VAR_PREFETCH && !OCC) { LAUNCH(VAR_PREFETCH) }
    LAUNCH(VAR_KEEP)
  }
  if constexpr (L <= 32 && 8 * K >= 16) {
    if constexpr (K <= 8) {
      if (p.var == VAR_PREFETCH) { LAUNCH(VAR_PREFETCH) }
    }
    if constexpr (!RES) {
      if constexpr (K <= 20) {
        if (p.var == VAR_STREAM_PF) { LAUNCH(VAR_STREAM_PF) }
      }
      if constexpr (K >= 8) {
        if (p.var == VAR_STREAM_L2) { LAUNCH(VAR_STREAM_L2) }
        if constexpr (K <= 20) {
          if (p.var == VAR_KEEP_L2) { LAUNCH(VAR_KEEP_L2) }
        }
      }
      if (p.var != VAR_KEEP) { LAUNCH(VAR_STREAM) }
    }
  }
  LAUNCH(VAR_KEEP)
#undef LAUNCH
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

// (K, L) of every width fwd_lanes returns: N = 256 K at L = 32, 128 and 384 at L = 16, and the four timed widths.
#define FWD_WIDTHS(X)                                                                              \
  X(1, 32) X(2, 32) X(3, 32) X(4, 32) X(8, 32) X(24, 32) X(32, 32) X(1, 16) X(3, 16)               \
  X(3, 64) X(4, 128) X(5, 128) X(7, 128)
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
  TORCH_CHECK(mode >= 1 && mode <= 3, "norms_sm100: rms mode must be 1 (plain), 2 (1+w) or 3 (round-then-scale)");
  TORCH_CHECK(mode != 3 || (!wf && !res), "norms_sm100: round-then-scale takes a bf16 weight and no residual");
#define X(KK, LL)                                                                                    \
  if (l.K == KK && l.L == LL) {                                                                      \
    if (mode == 3) {                                                                               \
      run_rms<KK, LL, 3, false, false>(a);                                                         \
    } else if (mode == 1) {                                                                               \
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
  TORCH_CHECK(false, "norms_sm100: width ", x.size(1), " (L=", l.L, ", K=", l.K,
              ") has no instantiated forward kernel");
}

void launch_gated_fwd(const at::Tensor &x, const at::Tensor &z, const at::Tensor &w,
                      const at::Tensor &out, const at::Tensor &rstd, double eps, int mode) {
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
  TORCH_CHECK(false, "norms_sm100: width ", x.size(1), " has no instantiated gated kernel");
}

int hidden_segment() { return HIDDEN_SEG; }

} // namespace norms_sm100
