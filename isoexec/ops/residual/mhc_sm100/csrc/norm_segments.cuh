// SPDX-License-Identifier: Apache-2.0
#pragma once
// The RMSNorm sum of squares as isoexec/ops/norms/norms_sm100 schedules it, for a kernel that fuses the norm with
// its neighbours: segments of S consecutive columns, each a chain from +0 in ascending order, combined by the
// adjacent tree (pair (0,1), (2,3), ...; an unpaired last partial moves up). S is not written here: it is
// norms_sm100/schedule.py's value (isoexec/ops/schedule_policy.py, rule 3, scalar step), passed by this package's
// _extension.py. The fused kernel must give norms_sm100's bits, so it must not carry a schedule of its own.
// The same file is in ops/residual/mhc_sm100/csrc and ops/attention/deepseek_v41_cache_sm100/csrc (a build directory
// is keyed on its own csrc, so a header shared across packages would not rebuild its users).
#include <cuda_runtime.h>

#ifndef NORMS_SM100_HIDDEN_SEGMENT
#error "build through _extension.py: norms_sm100/schedule.py supplies the segment length from schedule_policy"
#endif

namespace norm_segments {

constexpr int SEG = NORMS_SM100_HIDDEN_SEGMENT;
static_assert(SEG == 8, "the fused norm kernels form one segment per chunk of eight columns");

constexpr int pow2ceil(int k) {
  int m = 1;
  while (m < k)
    m *= 2;
  return m;
}

// One segment from its SEG values: acc = +0; acc = add.rn(acc, mul.rn(v, v)), columns ascending.
__device__ __forceinline__ float segment_sq(const float *v) {
  float s = 0.f;
#pragma unroll
  for (int j = 0; j < SEG; ++j)
    s = __fadd_rn(s, __fmul_rn(v[j], v[j]));
  return s;
}

// The same segment from squares that were rounded and stored already (mul.rn(v, v) as fp32).
__device__ __forceinline__ float segment_of_squares(const float *sq) {
  float s = 0.f;
#pragma unroll
  for (int j = 0; j < SEG; ++j)
    s = __fadd_rn(s, sq[j]);
  return s;
}

// Adjacent tree over NLEAF segment partials in shared memory, leaf[c] = the partial of columns SEG c .. SEG c + SEG - 1,
// run by ONE warp after the partials are visible: thread t forms the subtree of the B consecutive partials
// [B t, B t + B) (B a power of two, at most 32 subtrees), shuffle levels pair neighbouring subtrees with the lower one
// on the left, and a subtree with no right neighbour moves up unchanged. All 32 threads return the root.
template <int NLEAF> __device__ __forceinline__ float segment_tree_warp(const float *leaf) {
  constexpr int B = pow2ceil((NLEAF + 31) / 32), NODES = NLEAF / B;
  static_assert(NLEAF % B == 0, "the row must be whole subtrees");
  const int t = threadIdx.x & 31;
  const int first = t < NODES ? B * t : 0; // threads past the last subtree read a valid slot; unused
  float u[B];
#pragma unroll
  for (int i = 0; i < B; ++i)
    u[i] = leaf[first + i];
#pragma unroll
  for (int s = 1; s < B; s <<= 1)
#pragma unroll
    for (int q = 0; q < B; q += 2 * s)
      u[q] = __fadd_rn(u[q], u[q + s]);
  float nv = u[0];
#pragma unroll
  for (int s = 1; s < 32; s <<= 1) {
    float o = __shfl_xor_sync(0xffffffffu, nv, s);
    if (t + s < NODES) // read at threads that are multiples of 2s: the right neighbour exists
      nv = __fadd_rn(nv, o);
  }
  return __shfl_sync(0xffffffffu, nv, 0);
}

} // namespace norm_segments
