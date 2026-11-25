// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <ATen/ATen.h>

struct __nv_bfloat16;

namespace norms_sm100 {

inline const __nv_bfloat16 *bf16_ptr(const at::Tensor &t) {
  return static_cast<const __nv_bfloat16 *>(t.data_ptr());
}
inline __nv_bfloat16 *bf16_mut(const at::Tensor &t) {
  return static_cast<__nv_bfloat16 *>(t.data_ptr());
}

void launch_rms_fwd(const at::Tensor &x, const at::Tensor &r, const at::Tensor &w,
                    const at::Tensor &hout, const at::Tensor &out, const at::Tensor &rstd,
                    double eps, int mode);
void launch_gated_fwd(const at::Tensor &x, const at::Tensor &z, const at::Tensor &w,
                      const at::Tensor &out, const at::Tensor &rstd, double eps, int mode);

void launch_rms_bwd(const at::Tensor &dy, const at::Tensor &x, const at::Tensor &rstd,
                    const at::Tensor &w, const at::Tensor &dres, const at::Tensor &dx,
                    const at::Tensor &dwp, int mode, int rpg);
void launch_gated_bwd(const at::Tensor &dy, const at::Tensor &x, const at::Tensor &z,
                      const at::Tensor &rstd, const at::Tensor &w, const at::Tensor &dx,
                      const at::Tensor &dz, const at::Tensor &dwp, int mode, int rpg);

void launch_dw_fold(const at::Tensor &partials, const at::Tensor &dw, int rpg);

// The segment lengths the kernels were compiled with (schedule.py passes them from schedule_policy).
int hidden_segment();
int row_segment();
// Execution plan of the weight gradient: rows per thread group (1 or the segment length) and the partial count.
int row_group(int64_t rows, int requested);
int64_t row_units(int64_t rows, int rpg);

} // namespace norms_sm100
