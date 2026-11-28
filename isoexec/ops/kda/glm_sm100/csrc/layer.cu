// SPDX-License-Identifier: Apache-2.0
// The GLM KDA layer's element-wise stages around the recurrence, forward and backward:
//   gate   log_decay = lb (*) sigmoid(e(A_log_h) (*) (bf(f) (+) dt_bias_c))           (fp32 out)
//   beta   beta = rn(sigmoid(bf(b)))                                                  (bf16 out, as torch's
//          bf16 sigmoid: fp32 evaluation, one rounding)
//   gnorm  y = rn(((x (*) r) (*) w) (*) sigmoid(z)),  r = rsqrt_rn(FOLD32(x^2) (/) 128 (+) eps)
// sigmoid(y) = 1 (/) (1 (+) expf(-y)). Weight gradients (dt_bias, A_log, the norm weight) reduce over tokens
// in FIXED segments (SEG rows per partial, ascending), folded by a second kernel in ascending segment
// order: no atomics, run-to-run deterministic.
#include <cuda_runtime.h>

#include "common.cuh"

namespace kda_sm100 {

namespace {
constexpr int SEG = 256;   // rows (tokens or token-heads) per weight-gradient partial
__device__ __forceinline__ float sigm(float y) { return __fdiv_rn(1.f, fadd(1.f, expf(-y))); }
__device__ __forceinline__ float ld_any(const void *p, int64_t i, int bf16) {
  return bf16 ? bf(static_cast<const uint16_t *>(p)[i]) : static_cast<const float *>(p)[i];
}
__device__ __forceinline__ void st_any(void *p, int64_t i, int bf16, float x) {
  if (bf16) static_cast<uint16_t *>(p)[i] = rn(x);
  else static_cast<float *>(p)[i] = x;
}
}  // namespace

// ---------------------------------------------------------------------------- forget gate
__global__ void gate_fwd_kernel(const void *f, int f_bf16, const float *dt_bias, const float *A_log, float lb,
                                float *out, int64_t n, int HK) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int c = int(i % HK), h = c / D;
  const float x = fadd(ld_any(f, i, f_bf16), dt_bias[c]);
  out[i] = fmul(lb, sigm(fmul(expf(A_log[h]), x)));
}

// grid (nseg, HK / 128): thread = channel; rows [seg * SEG, ..) ascending. Writes df and the partials
// pdt[seg][c] = sum dx, pA[seg][c] = sum dy (*) x (*) a.
__global__ void gate_bwd_kernel(const float *dld, const void *f, int f_bf16, void *df, int df_bf16,
                                const float *dt_bias, const float *A_log, float lb, float *pdt, float *pA,
                                int64_t T, int HK) {
  const int seg = blockIdx.x, c = blockIdx.y * blockDim.x + threadIdx.x;
  if (c >= HK) return;
  const float a = expf(A_log[c / D]), dtb = dt_bias[c];
  float sdt = 0.f, sA = 0.f;
  const int64_t t1 = min(T, int64_t(seg + 1) * SEG);
  for (int64_t t = int64_t(seg) * SEG; t < t1; ++t) {
    const int64_t i = t * HK + c;
    const float x = fadd(ld_any(f, i, f_bf16), dtb);
    const float s = sigm(fmul(a, x));
    const float dy = fmul(fmul(dld[i], lb), fmul(s, fsub(1.f, s)));
    const float dx = fmul(dy, a);
    st_any(df, i, df_bf16, dx);
    sdt = fadd(sdt, dx);
    sA = ffma(fmul(dy, x), a, sA);
  }
  pdt[int64_t(seg) * HK + c] = sdt;
  pA[int64_t(seg) * HK + c] = sA;
}

// One thread per channel: ddt_bias_c = sum_seg pdt (ascending); then one thread per head:
// dA_log_h = sum_c (sum_seg pA) (ascending c).
__global__ void gate_fold_kernel(const float *pdt, const float *pA, int nseg, int HK, float *ddt, float *dA,
                                 float *colA) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < HK) {
    float s1 = 0.f, s2 = 0.f;
    for (int s = 0; s < nseg; ++s) {
      s1 = fadd(s1, pdt[int64_t(s) * HK + c]);
      s2 = fadd(s2, pA[int64_t(s) * HK + c]);
    }
    ddt[c] = s1;
    colA[c] = s2;
  }
}
__global__ void gate_fold_heads_kernel(const float *colA, int H, float *dA) {
  const int h = blockIdx.x * blockDim.x + threadIdx.x;
  if (h >= H) return;
  float s = 0.f;
  for (int c = 0; c < D; ++c) s = fadd(s, colA[h * D + c]);
  dA[h] = s;
}

// ---------------------------------------------------------------------------- beta
__global__ void beta_fwd_kernel(const uint16_t *b, uint16_t *out, int64_t n) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = rn(sigm(bf(b[i])));
}
// db = rn(dbeta (*) ((1 (-) y) (*) y)), y the bf16 forward output (torch's sigmoid backward).
__global__ void beta_bwd_kernel(const void *dbeta, int dbeta_bf16, const uint16_t *y, uint16_t *db, int64_t n) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float yv = bf(y[i]);
  db[i] = rn(fmul(ld_any(dbeta, i, dbeta_bf16), fmul(fsub(1.f, yv), yv)));
}

// ---------------------------------------------------------------------------- sigmoid-gated RMS norm
// One warp per row of 128; lane l owns elements 4l..4l+3.
__global__ void gnorm_fwd_kernel(const uint16_t *x, const uint16_t *z, const void *w, int w_bf16, float eps,
                                 uint16_t *y, float *rstd, int64_t rows) {
  const int64_t row = int64_t(blockIdx.x) * (blockDim.x >> 5) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (row >= rows) return;
  const uint2 xw = *reinterpret_cast<const uint2 *>(x + row * D + 4 * lane);
  const uint2 zw = *reinterpret_cast<const uint2 *>(z + row * D + 4 * lane);
  const float xv[4] = {bf(lo16(xw.x)), bf(hi16(xw.x)), bf(lo16(xw.y)), bf(hi16(xw.y))};
  const float zv[4] = {bf(lo16(zw.x)), bf(hi16(zw.x)), bf(lo16(zw.y)), bf(hi16(zw.y))};
  float ss = 0.f;
  #pragma unroll
  for (int e = 0; e < 4; ++e) ss = ffma(xv[e], xv[e], ss);
  #pragma unroll
  for (int off = 1; off < 32; off *= 2) ss = fadd(ss, __shfl_xor_sync(0xffffffffu, ss, off));
  const float r = __frsqrt_rn(fadd(__fdiv_rn(ss, float(D)), eps));
  if (lane == 0) rstd[row] = r;
  float o[4];
  #pragma unroll
  for (int e = 0; e < 4; ++e) o[e] = fmul(fmul(fmul(xv[e], r), ld_any(w, 4 * lane + e, w_bf16)), sigm(zv[e]));
  uint2 ow;
  ow.x = rn2(o[0], o[1]);
  ow.y = rn2(o[2], o[3]);
  *reinterpret_cast<uint2 *>(y + row * D + 4 * lane) = ow;
}

// grid over segments of SEG rows, 128 threads = 4 warps each walking rows of the segment (warp w: rows
// w, w + 4, ..). dx, dz per row; per-warp column partials of dw kept per lane then combined in a fixed
// order (warp 0..3) into pw[seg][128].
__global__ void gnorm_bwd_kernel(const uint16_t *dy, const uint16_t *x, const uint16_t *z, const void *w, int w_bf16,
                                 const float *rstd, uint16_t *dx, uint16_t *dz, float *pw, int64_t rows) {
  __shared__ float part[4][D];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int64_t r0 = int64_t(blockIdx.x) * SEG, r1 = min(rows, r0 + SEG);
  float wv[4], acc[4] = {0.f, 0.f, 0.f, 0.f};
  #pragma unroll
  for (int e = 0; e < 4; ++e) wv[e] = ld_any(w, 4 * lane + e, w_bf16);
  for (int64_t row = r0 + warp; row < r1; row += 4) {
    const uint2 xw = *reinterpret_cast<const uint2 *>(x + row * D + 4 * lane);
    const uint2 zw = *reinterpret_cast<const uint2 *>(z + row * D + 4 * lane);
    const uint2 gw = *reinterpret_cast<const uint2 *>(dy + row * D + 4 * lane);
    const float xv[4] = {bf(lo16(xw.x)), bf(hi16(xw.x)), bf(lo16(xw.y)), bf(hi16(xw.y))};
    const float zv[4] = {bf(lo16(zw.x)), bf(hi16(zw.x)), bf(lo16(zw.y)), bf(hi16(zw.y))};
    const float gv[4] = {bf(lo16(gw.x)), bf(hi16(gw.x)), bf(lo16(gw.y)), bf(hi16(gw.y))};
    const float r = rstd[row];
    float n[4], dn[4], dzv[4], dot = 0.f;
    #pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float s = sigm(zv[e]);
      n[e] = fmul(xv[e], r);
      dn[e] = fmul(fmul(gv[e], wv[e]), s);
      dzv[e] = fmul(fmul(fmul(gv[e], n[e]), wv[e]), fmul(s, fsub(1.f, s)));
      acc[e] = ffma(fmul(gv[e], n[e]), s, acc[e]);
      dot = ffma(dn[e], n[e], dot);
    }
    #pragma unroll
    for (int off = 1; off < 32; off *= 2) dot = fadd(dot, __shfl_xor_sync(0xffffffffu, dot, off));
    const float mdot = __fdiv_rn(dot, float(D));
    float dxv[4];
    #pragma unroll
    for (int e = 0; e < 4; ++e) dxv[e] = fmul(r, fsub(dn[e], fmul(n[e], mdot)));
    uint2 o1, o2;
    o1.x = rn2(dxv[0], dxv[1]);
    o1.y = rn2(dxv[2], dxv[3]);
    o2.x = rn2(dzv[0], dzv[1]);
    o2.y = rn2(dzv[2], dzv[3]);
    *reinterpret_cast<uint2 *>(dx + row * D + 4 * lane) = o1;
    *reinterpret_cast<uint2 *>(dz + row * D + 4 * lane) = o2;
  }
  #pragma unroll
  for (int e = 0; e < 4; ++e) part[warp][4 * lane + e] = acc[e];
  __syncthreads();
  const int c = threadIdx.x;
  pw[int64_t(blockIdx.x) * D + c] = fadd(fadd(part[0][c], part[1][c]), fadd(part[2][c], part[3][c]));
}

__global__ void colfold_kernel(const float *p, int nseg, int width, void *out, int out_bf16) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= width) return;
  float s = 0.f;
  for (int i = 0; i < nseg; ++i) s = fadd(s, p[int64_t(i) * width + c]);
  st_any(out, c, out_bf16, s);
}

// ---------------------------------------------------------------------------- launches
void gate_fwd_launch(const void *f, int f_bf16, const float *dt_bias, const float *A_log, float lb, float *out,
                     int64_t n, int HK, cudaStream_t s) {
  if (n) gate_fwd_kernel<<<unsigned((n + 255) / 256), 256, 0, s>>>(f, f_bf16, dt_bias, A_log, lb, out, n, HK);
}
void gate_bwd_launch(const float *dld, const void *f, int f_bf16, void *df, int df_bf16, const float *dt_bias,
                     const float *A_log, float lb, float *pdt, float *pA, float *colA, int64_t T, int HK, int H,
                     float *ddt, float *dA, cudaStream_t s) {
  const int nseg = int((T + SEG - 1) / SEG);
  if (nseg) gate_bwd_kernel<<<dim3(nseg, (HK + 127) / 128), 128, 0, s>>>(dld, f, f_bf16, df, df_bf16, dt_bias, A_log,
                                                                         lb, pdt, pA, T, HK);
  gate_fold_kernel<<<(HK + 127) / 128, 128, 0, s>>>(pdt, pA, nseg, HK, ddt, dA, colA);
  gate_fold_heads_kernel<<<(H + 63) / 64, 64, 0, s>>>(colA, H, dA);
}
void beta_fwd_launch(const uint16_t *b, uint16_t *out, int64_t n, cudaStream_t s) {
  if (n) beta_fwd_kernel<<<unsigned((n + 255) / 256), 256, 0, s>>>(b, out, n);
}
void beta_bwd_launch(const void *dbeta, int dbeta_bf16, const uint16_t *y, uint16_t *db, int64_t n, cudaStream_t s) {
  if (n) beta_bwd_kernel<<<unsigned((n + 255) / 256), 256, 0, s>>>(dbeta, dbeta_bf16, y, db, n);
}
void gnorm_fwd_launch(const uint16_t *x, const uint16_t *z, const void *w, int w_bf16, float eps, uint16_t *y,
                      float *rstd, int64_t rows, cudaStream_t s) {
  if (rows) gnorm_fwd_kernel<<<unsigned((rows + 7) / 8), 256, 0, s>>>(x, z, w, w_bf16, eps, y, rstd, rows);
}
void gnorm_bwd_launch(const uint16_t *dy, const uint16_t *x, const uint16_t *z, const void *w, int w_bf16,
                      const float *rstd, uint16_t *dx, uint16_t *dz, float *pw, void *dw, int dw_bf16, int64_t rows,
                      cudaStream_t s) {
  const int nseg = int((rows + SEG - 1) / SEG);
  if (nseg) gnorm_bwd_kernel<<<nseg, 128, 0, s>>>(dy, x, z, w, w_bf16, rstd, dx, dz, pw, rows);
  colfold_kernel<<<1, D, 0, s>>>(pw, nseg, D, dw, dw_bf16);
}
int seg_rows() { return SEG; }

// ---- the o_proj leaf tree (tensor parallel; TP-degree invariant): the root of the balanced adjacent-pair
// FP32 tree over L (a power of two <= 8) leaves [L][n] -- level 0 pairs (0,1), (2,3), ... -- written as the
// FP32 root, or rounded once to bf16. A rank holding an aligned block of leaves evaluates that block's
// subtree; the ranks' subtree roots, in rank order, complete the same tree (pik's ROOT / PUBLISH scheme).
__global__ void leaf_tree_kernel(const float *x, int L, int64_t n, float *out_f32, uint16_t *out_bf16) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) {
    float a[8];
    #pragma unroll
    for (int l = 0; l < 8; ++l) a[l] = l < L ? x[int64_t(l) * n + i] : 0.f;
    for (int m = L; m > 1; m >>= 1)   // one level: m nodes -> m / 2 (L is a power of two)
      #pragma unroll
      for (int k = 0; k < 4; ++k)
        if (k < m / 2) a[k] = fadd(a[2 * k], a[2 * k + 1]);
    if (out_f32) out_f32[i] = a[0];
    else out_bf16[i] = rn(a[0]);
  }
}
void leaf_tree_launch(const float *x, int L, int64_t n, float *out_f32, uint16_t *out_bf16, cudaStream_t s) {
  if (n) leaf_tree_kernel<<<unsigned((n + 255) / 256 < 65536 ? (n + 255) / 256 : 65536), 256, 0, s>>>(x, L, n, out_f32, out_bf16);
}

}  // namespace kda_sm100
