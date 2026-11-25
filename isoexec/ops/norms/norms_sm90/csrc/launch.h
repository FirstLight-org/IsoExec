// SPDX-License-Identifier: Apache-2.0
// Launcher declarations shared by the host translation units.
#pragma once
#include <ATen/ATen.h>

struct __nv_bfloat16;

namespace norms_sm90 {

inline const __nv_bfloat16 *bf16_ptr(const at::Tensor &t) {
  return static_cast<const __nv_bfloat16 *>(t.data_ptr());
}
inline __nv_bfloat16 *bf16_mut(const at::Tensor &t) {
  return static_cast<__nv_bfloat16 *>(t.data_ptr());
}

// Forward. x/r are [M, N] with unit column stride; w is [N] (bf16 or fp32) or undefined.
// hout (fused add only) and out are contiguous [M, N]; rstd is contiguous fp32 [M].
void launch_rms_fwd(const at::Tensor &x, const at::Tensor &r, const at::Tensor &w,
                    const at::Tensor &hout, const at::Tensor &out, const at::Tensor &rstd,
                    double eps, int mode);
// x [M, N]; z [T, hv, N] with unit feature stride; out [M, N]; rstd [M].
void launch_gated_fwd(const at::Tensor &x, const at::Tensor &z, const at::Tensor &w,
                      const at::Tensor &out, const at::Tensor &rstd, double eps, int mode);

// Backward. dy/x/dres [M, N] unit column stride; dx contiguous [M, N]; dwp is fully written
// contiguous fp32 [bands, N] weight-gradient partials (unused for mode 0). dres_after_round: add
// dres to the bf16-rounded dx (the unfused bf16 sum) instead of to the fp32 dx.
void launch_rms_bwd(const at::Tensor &dy, const at::Tensor &x, const at::Tensor &rstd,
                    const at::Tensor &w, const at::Tensor &dres, const at::Tensor &dx,
                    const at::Tensor &dwp, int mode, int64_t bands, int iters,
                    bool dres_after_round = false);
void launch_gated_bwd(const at::Tensor &dy, const at::Tensor &x, const at::Tensor &z,
                      const at::Tensor &rstd, const at::Tensor &w, const at::Tensor &dx,
                      const at::Tensor &dz, const at::Tensor &dwp, int mode, int64_t bands,
                      int iters);

// Fold band partials in ascending band order, one independent FP32 accumulator per column.
void launch_dw_fold(const at::Tensor &partials, const at::Tensor &dw);

} // namespace norms_sm90
