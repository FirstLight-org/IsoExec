// SPDX-License-Identifier: Apache-2.0
#include "bwd.cuh"
#include "launch.h"
#include "plan.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

namespace norms_sm100 {
namespace {

// Adjacent tree over up to N values at stride `step` of p (N = 4 or 8), `count` of them present: a node with no
// right neighbour moves up unchanged.
template <int N> __device__ __forceinline__ float tree_present(const float *p, int step, int count) {
  float t[N];
#pragma unroll
  for (int i = 0; i < N; ++i)
    t[i] = p[i * step];
#pragma unroll
  for (int s = 1; s < N; s <<= 1)
#pragma unroll
    for (int q = 0; q < N; q += 2 * s)
      if (q + s < count)
        t[q] = __fadd_rn(t[q], t[q + s]);
  return t[0];
}

// The weight gradient: one level of the adjacent tree over the row segments' partials. CTA (x, y) owns 32 columns
// and the aligned block of 256 consecutive partials [256 y, 256 y + 256) (fewer in the last block): a subtree. All
// threads stage the block in shared memory; its eight levels then run as three rounds (subtrees of 8, of 64, then
// 4 of those), every (column, subtree) on its own thread. The root goes to row y of `out`; the host applies the
// kernel again to the roots until one row is left, which is the same tree: node j of level k covers partials
// [j 2^k, (j + 1) 2^k), and a node with no right neighbour moves up unchanged at every level.
// ROWS: the staged values are single rows' terms (the backward's rpg == 1 plan). The first round is then each
// segment's chain from +0 over its ROW_SEG rows (the last segment short), and a block is 256 / ROW_SEG segments.
template <bool ROWS>
__global__ __launch_bounds__(256, 1) void
fold_dw_tree(const float *__restrict__ partials, float *__restrict__ out, int64_t count, int64_t width) {
  constexpr int COLS = 32, BLOCK = 256, TPB = 256, PER = BLOCK * COLS / TPB;
  __shared__ float st[BLOCK * COLS];
  const int64_t col0 = int64_t(blockIdx.x) * COLS, b0 = int64_t(blockIdx.y) * BLOCK;
  const int tid = threadIdx.x, col = tid % COLS;
  const int n = int(count - b0 < BLOCK ? count - b0 : BLOCK);
  for (int u = 0; u * TPB + tid < n * COLS; ++u) { // only the staged values that exist are read
    int idx = u * TPB + tid, sb = idx / COLS, c = idx % COLS;
    st[idx] = col0 + c < width ? partials[(b0 + sb) * width + col0 + c] : 0.f;
  }
  __syncthreads();
#pragma unroll
  for (int u = 0; u < 4; ++u) { // 32 groups of 8 staged values per column
    int g = tid / COLS + 8 * u;
    if (8 * g < n) {
      if constexpr (ROWS) { // one segment: its rows ascending, from +0
        static_assert(ROW_SEG == 8, "the per-row plan stages one segment per group of 8 rows");
        float acc = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i)
          if (8 * g + i < n)
            acc = __fadd_rn(acc, st[(8 * g + i) * COLS + col]);
        st[8 * g * COLS + col] = acc;
      } else { // a subtree of 8 partials
        st[8 * g * COLS + col] = tree_present<8>(st + 8 * g * COLS + col, COLS, n - 8 * g);
      }
    }
  }
  __syncthreads();
  const int n8 = (n + 7) / 8;
  if (tid < 4 * COLS) { // subtrees of 8 of those: 4 per column
    int h = tid / COLS;
    if (8 * h < n8)
      st[64 * h * COLS + col] = tree_present<8>(st + 64 * h * COLS + col, 8 * COLS, n8 - 8 * h);
  }
  __syncthreads();
  if (tid < COLS && col0 + col < width)
    out[int64_t(blockIdx.y) * width + col0 + col] = tree_present<4>(st + col, 64 * COLS, (n + 63) / 64);
}

constexpr int64_t kPerRowMax = 256; // rows; measured (policy_validation/port/seq/logs/rms_release_bwd_plans.log)

struct RmsBwdArgs {
  const __nv_bfloat16 *dy, *x, *dres;
  int64_t sdy, sx, sdres;
  const float *rstd;
  const void *w;
  __nv_bfloat16 *dx;
  float *dwp;
  int64_t M, nseg;
  int rpg;
};

// CTAs: one per 128 / L thread groups (L <= 64), else one per group.
template <int L> int64_t bwd_blocks(int64_t nseg) {
  constexpr int RPI = L <= 64 ? 128 / L : 1;
  return (nseg + RPI - 1) / RPI;
}

template <int K, int L, int MODE, bool WF, bool RES>
void run_rms_bwd(const RmsBwdArgs &a) {
  auto stream = at::cuda::getCurrentCUDAStream();
  const int var = bwd_var({L, K}, a.M);
  const int64_t blocks = bwd_blocks<L>(a.nseg);
  if constexpr (8 * K >= 16) {
    if (var == BWD_STREAM) {
      rms_bwd<K, L, MODE, WF, RES, BWD_STREAM><<<dim3(blocks), 128, 0, stream>>>(
          a.dy, a.sdy, a.x, a.sx, a.rstd, a.w, a.dres, a.sdres, a.dx, a.dwp, a.M, a.nseg, a.rpg);
      return;
    }
  }
  if constexpr (L == 128 && K <= 5) {
    if (var == BWD_RING) {
      rms_bwd<K, L, MODE, WF, RES, BWD_RING><<<dim3(blocks), 128, 0, stream>>>(
          a.dy, a.sdy, a.x, a.sx, a.rstd, a.w, a.dres, a.sdres, a.dx, a.dwp, a.M, a.nseg, a.rpg);
      return;
    }
  }
  rms_bwd<K, L, MODE, WF, RES, BWD_KEEP><<<dim3(blocks), 128, 0, stream>>>(
      a.dy, a.sdy, a.x, a.sx, a.rstd, a.w, a.dres, a.sdres, a.dx, a.dwp, a.M, a.nseg, a.rpg);
}

struct GatedBwdArgs {
  const __nv_bfloat16 *dy, *x, *z;
  int64_t sdy, sx, szt, szh;
  const float *rstd;
  const void *w;
  __nv_bfloat16 *dx, *dz;
  float *dwp;
  int64_t M, nseg;
  int hv, rpg;
  bool zaligned;
};

template <int K, int L, int MODE, bool WF>
void run_gated_bwd(const GatedBwdArgs &a) {
  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t blocks = bwd_blocks<L>(a.nseg);
  if (a.zaligned)
    gated_bwd<K, L, MODE, WF, true><<<dim3(blocks), 128, 0, stream>>>(
        a.dy, a.sdy, a.x, a.sx, a.z, a.szt, a.szh, a.rstd, a.w, a.dx, a.dz, a.dwp, a.M, a.hv, a.nseg, a.rpg);
  else
    gated_bwd<K, L, MODE, WF, false><<<dim3(blocks), 128, 0, stream>>>(
        a.dy, a.sdy, a.x, a.sx, a.z, a.szt, a.szh, a.rstd, a.w, a.dx, a.dz, a.dwp, a.M, a.hv, a.nseg, a.rpg);
}

#define BWD_WIDTHS(X)                                                                              \
  X(1, 128) X(2, 128) X(3, 128) X(4, 128) X(5, 128) X(6, 128) X(8, 128) X(1, 32) X(2, 32)          \
  X(3, 32) X(5, 32) X(6, 32) X(7, 32) X(1, 16) X(3, 16) X(7, 128)
#define GATED_BWD_WIDTHS(X) X(1, 16) X(3, 16) X(2, 32)

} // namespace

int row_segment() { return ROW_SEG; }

// Rows per thread group (execution only; see bwd.cuh): single rows while the row count is small enough that one
// row per group keeps more CTAs in flight than a segment per group, else whole segments.
int row_group(int64_t rows, int requested) {
  if (requested == 1 || requested == ROW_SEG)
    return requested;
  TORCH_CHECK(requested == 0, "norms_sm100: row_group must be 0 (plan), 1 or ", ROW_SEG);
  return rows <= kPerRowMax ? 1 : ROW_SEG;
}

int64_t row_units(int64_t rows, int rpg) { return (rows + rpg - 1) / rpg; }

void launch_dw_fold(const at::Tensor &partials, const at::Tensor &dw, int rpg) {
  int64_t width = dw.numel(), count = partials.size(0);
  auto stream = at::cuda::getCurrentCUDAStream();
  at::Tensor in = partials;
  bool rows = rpg == 1;
  while (true) { // one launch per eight tree levels
    int64_t roots = (count + 255) / 256;
    at::Tensor out = roots == 1 ? dw : at::empty({roots, width}, partials.options());
    if (rows)
      fold_dw_tree<true><<<dim3((width + 31) / 32, roots), 256, 0, stream>>>(in.data_ptr<float>(), out.data_ptr<float>(),
                                                                            count, width);
    else
      fold_dw_tree<false><<<dim3((width + 31) / 32, roots), 256, 0, stream>>>(in.data_ptr<float>(), out.data_ptr<float>(),
                                                                             count, width);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (roots == 1)
      return;
    in = out;
    count = roots;
    rows = false; // the roots of 256 / ROW_SEG segments are subtrees: the remaining levels are the tree's
  }
}

void launch_rms_bwd(const at::Tensor &dy, const at::Tensor &x, const at::Tensor &rstd,
                    const at::Tensor &w, const at::Tensor &dres, const at::Tensor &dx,
                    const at::Tensor &dwp, int mode, int rpg) {
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
               row_units(x.size(0), rpg),
               rpg};
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
                      const at::Tensor &dz, const at::Tensor &dwp, int mode, int rpg) {
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
                 row_units(x.size(0), rpg),
                 int(z.size(1)),
                 rpg,
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
