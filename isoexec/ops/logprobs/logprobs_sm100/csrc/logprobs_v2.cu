// SPDX-License-Identifier: Apache-2.0
// Row-invariant fp32 log-softmax / sampled logprob on SM100 (arithmetic version 2; recipe in ops.py).
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <type_traits>

namespace logprobs_sm100_v2 {
namespace {

using bf16 = __nv_bfloat16;
constexpr int STRIP = 8;    // columns per strip (recipe)
constexpr int LANES = 256;  // lanes per leaf (recipe); strip q of a leaf belongs to lane q % LANES
constexpr int G = 8;        // leaves per full vocabulary row (recipe)
constexpr float NINF = -__builtin_huge_valf();

template <class T> __device__ __forceinline__ float widen(T v) {
  if constexpr (std::is_same_v<T, bf16>)
    return __bfloat162float(v);
  else
    return v;
}

template <class T> struct __align__(16) pack8 {
  T v[8];
};

// The input value of one column: x, or x / t (__fdiv_rn) when a temperature is applied.
template <class T, bool TEMP> __device__ __forceinline__ float value(T raw, float t) {
  if constexpr (TEMP)
    return __fdiv_rn(widen(raw), t);
  else
    return widen(raw);
}

// Leaf state (M_g, S_g): two passes compute the exact max, then sum exp(v - F) in lane order.
// F = 0 for an all -inf leaf, keeping its sum at +0; otherwise F = M_g.
// Omitting v1's per-column rescale changes rounding, so this is a separate recipe.
// ENT also sums, per lane in the same pass, a = exp(v - F) * (v - F) (+0 for -inf columns), chained per
// leaf in lane order into A_g (relative to M_g = F); M_g and S_g are the same statements either way.
template <class T, bool TEMP, bool PACKED, bool ENT = false>
__device__ __forceinline__ void leaf_state_body(const T *__restrict__ x, int64_t stride, int L,
                                                float *__restrict__ mout, float *__restrict__ sout,
                                                int gl, const float *__restrict__ temp,
                                                float *__restrict__ aout = nullptr) {
  int row = blockIdx.x, g = blockIdx.y;
  const auto *xs = reinterpret_cast<const pack8<T> *>(x + int64_t(row) * stride + int64_t(g) * L);
  int strips = L / STRIP;
  __shared__ float lmax[LANES];
  __shared__ float lsum[LANES];
  __shared__ float smax[2];  // [0] = M_g, [1] = the frame F the sum is taken in
  float t = 1.f;
  if constexpr (TEMP)
    t = *temp;
  float m = NINF;
  for (int q = threadIdx.x; q < strips; q += LANES) {
    pack8<T> p = xs[q];
    float a[STRIP];
#pragma unroll
    for (int j = 0; j < STRIP; ++j)
      a[j] = value<T, TEMP>(p.v[j], t);
#pragma unroll
    for (int w = STRIP / 2; w > 0; w >>= 1)
#pragma unroll
      for (int j = 0; j < w; ++j)
        a[j] = fmaxf(a[j], a[j + w]);
    m = fmaxf(m, a[0]);
  }
  lmax[threadIdx.x] = m;
  __syncthreads();
  if (threadIdx.x == 0) {
    float mm = NINF;
    for (int l = 0; l < LANES; ++l)
      mm = lmax[l] > mm ? lmax[l] : mm;
    smax[0] = mm;
    smax[1] = mm == NINF ? 0.f : mm;
  }
  __syncthreads();
  float M = smax[0], F = smax[1];
  float s = 0.f, a = 0.f;
  for (int q = threadIdx.x; q < strips; q += LANES) {
    pack8<T> p = xs[q];
#pragma unroll
    for (int j = 0; j < STRIP; ++j) {
      float v = value<T, TEMP>(p.v[j], t);
      float d = __fsub_rn(v, F);
      float e = expf(d);
      s = __fadd_rn(s, e);
      // A -inf column has e = +0; the clamp keeps 0 * d at -0 instead of NaN.
      if constexpr (ENT)
        a = __fadd_rn(a, __fmul_rn(e, fmaxf(d, -3.402823466e38f)));
    }
  }
  lsum[threadIdx.x] = s;
  __shared__ float lent[ENT ? LANES : 1];
  if constexpr (ENT)
    lent[threadIdx.x] = a;
  __syncthreads();
  if (threadIdx.x == 0) {
    float S = lsum[0];
    for (int l = 1; l < LANES; ++l)
      S = __fadd_rn(S, lsum[l]);
    if constexpr (ENT) {
      float A = lent[0];
      for (int l = 1; l < LANES; ++l)
        A = __fadd_rn(A, lent[l]);
      aout[int64_t(row) * gl + g] = A;
    }
    if constexpr (PACKED) {
      mout[int64_t(row) * 2 * G + g] = M;
      mout[int64_t(row) * 2 * G + G + g] = S;
    } else {
      mout[int64_t(row) * gl + g] = M;
      sout[int64_t(row) * gl + g] = S;
    }
  }
}

template <class T, bool TEMP>
__global__ __launch_bounds__(LANES) void leaf_state(const T *__restrict__ x, int64_t stride, int L,
                                                    float *__restrict__ mout, float *__restrict__ sout,
                                                    int gl, const float *__restrict__ temp) {
  leaf_state_body<T, TEMP, false>(x, stride, L, mout, sout, gl, temp);
}

template <class T, bool TEMP>
__global__ __launch_bounds__(LANES) void leaf_state_ent(const T *__restrict__ x, int64_t stride, int L,
                                                        float *__restrict__ mout, float *__restrict__ sout,
                                                        float *__restrict__ aout, int gl,
                                                        const float *__restrict__ temp) {
  leaf_state_body<T, TEMP, false, true>(x, stride, L, mout, sout, gl, temp, aout);
}

template <class T, bool TEMP>
__global__ __launch_bounds__(LANES) void leaf_state_split(const T *__restrict__ x, int64_t stride, int L,
                                                          float *__restrict__ leaves,
                                                          const float *__restrict__ temp) {
  leaf_state_body<T, TEMP, true>(x, stride, L, leaves, nullptr, G, temp);
}

// The row merge (shared by merge_rows and the fused full-row write so both evaluate one expression).
__device__ __forceinline__ void merge_leaves(const float *mg, const float *sg, float &Mout, double &lsout) {
  float M = NINF;
#pragma unroll
  for (int g = 0; g < G; ++g)
    M = mg[g] > M ? mg[g] : M;
  double w[G];
#pragma unroll
  for (int g = 0; g < G; ++g) {
    if (mg[g] == NINF)
      w[g] = sg[g] == 0.f ? 0.0 : double(__builtin_nan(""));
    else
      w[g] = double(sg[g]) * exp(double(mg[g]) - double(M));
  }
#pragma unroll
  for (int step = 1; step < G; step *= 2)
#pragma unroll
    for (int g = 0; g + step < G; g += 2 * step)
      w[g] = w[g] + w[g + step];
  Mout = M;
  lsout = M == NINF ? double(__builtin_nan("")) : log(w[0]);
}

// Row merge over the G leaves: w_g = double(S_g) * exp(double(M_g) - double(M)) (absent leaf, i.e.
// M_g == -inf and S_g == +0: +0; M_g == -inf with NaN S_g: NaN), balanced tree with lower leaves on the
// left, logS = log(w) in double (NaN when M == -inf). Published per row: M, L32 = RN32(logS) and, for
// the backward, lse = M + logS (double). Sampled logprob = (v_t - M) - L32 in fp32: the expression the
// full-row write evaluates at that column.
template <class T, bool TEMP, int SRC>  // SRC: 0 none, 1 gather from x, 2 caller-supplied raw values
__global__ __launch_bounds__(32) void merge_rows(
    const float *__restrict__ mleaf, const float *__restrict__ sleaf, float *__restrict__ rM,
    float *__restrict__ rL, double *__restrict__ lse, float *__restrict__ sampled_out,
    const T *__restrict__ x, int64_t stride, const int64_t *__restrict__ target,
    const float *__restrict__ sampled_raw, int64_t vocab, int rows, const float *__restrict__ temp) {
  int row = blockIdx.x * 32 + threadIdx.x;
  if (row >= rows)
    return;
  float M;
  double ls;
  merge_leaves(mleaf + int64_t(row) * G, sleaf + int64_t(row) * G, M, ls);
  float L32 = float(ls);
  rM[row] = M;
  rL[row] = L32;
  if (lse)
    lse[row] = double(M) + ls;
  if constexpr (SRC != 0) {
    int64_t tg = target[row];
    bool valid = tg >= 0 && tg < vocab;
    float v = 0.f;
    if constexpr (SRC == 1) {
      float t = 1.f;
      if constexpr (TEMP)
        t = *temp;
      if (valid)
        v = value<T, TEMP>(x[int64_t(row) * stride + tg], t);
    } else {
      v = sampled_raw[row];
    }
    sampled_out[row] = valid ? __fsub_rn(__fsub_rn(v, M), L32) : 0.f;
  }
}

// Row entropy over the G leaves (a metric, not part of the scoring contract; deterministic and TP
// invariant because the leaves are fixed vocabulary slices merged in a fixed order). In double:
// e_g = exp(M_g - M), S = tree(S_g * e_g), A = tree(e_g * (A_g + S_g * (M_g - M))), H = log(S) - A / S,
// with the balanced lower-left tree merge_leaves uses. Absent leaves (M_g == -inf, S_g == +0) add 0.
__global__ __launch_bounds__(32) void entropy_rows(const float *__restrict__ mleaf,
                                                   const float *__restrict__ sleaf,
                                                   const float *__restrict__ aleaf,
                                                   float *__restrict__ ent, int rows) {
  int row = blockIdx.x * 32 + threadIdx.x;
  if (row >= rows)
    return;
  const float *mg = mleaf + int64_t(row) * G, *sg = sleaf + int64_t(row) * G, *ag = aleaf + int64_t(row) * G;
  float M = NINF;
#pragma unroll
  for (int g = 0; g < G; ++g)
    M = mg[g] > M ? mg[g] : M;
  double w[G], u[G];
#pragma unroll
  for (int g = 0; g < G; ++g) {
    if (mg[g] == NINF) {
      w[g] = sg[g] == 0.f ? 0.0 : double(__builtin_nan(""));
      u[g] = w[g];
    } else {
      double d = double(mg[g]) - double(M);
      double e = exp(d);
      w[g] = double(sg[g]) * e;
      u[g] = e * (double(ag[g]) + double(sg[g]) * d);
    }
  }
#pragma unroll
  for (int step = 1; step < G; step *= 2)
#pragma unroll
    for (int g = 0; g + step < G; g += 2 * step) {
      w[g] = w[g] + w[g + step];
      u[g] = u[g] + u[g + step];
    }
  ent[row] = M == NINF ? __builtin_nanf("") : float(log(w[0]) - u[0] / w[0]);
}

// Full row write: out = (v - M) - L32 (fp32). Elementwise, so the thread layout is free.
template <class T, bool TEMP>
__global__ __launch_bounds__(256) void write_rows(float *__restrict__ out, int64_t ostride,
                                                  const T *__restrict__ x, int64_t stride, int L,
                                                  const float *__restrict__ rM, const float *__restrict__ rL,
                                                  const float *__restrict__ temp) {
  int row = blockIdx.x, g = blockIdx.y;
  const auto *xs = reinterpret_cast<const pack8<T> *>(x + int64_t(row) * stride + int64_t(g) * L);
  auto *os = reinterpret_cast<pack8<float> *>(out + int64_t(row) * ostride + int64_t(g) * L);
  float M = rM[row], L32 = rL[row];
  float t = 1.f;
  if constexpr (TEMP)
    t = *temp;
  int strips = L / STRIP;
  for (int q = threadIdx.x; q < strips; q += 256) {
    pack8<T> p = xs[q];
    pack8<float> o;
#pragma unroll
    for (int j = 0; j < STRIP; ++j)
      o.v[j] = __fsub_rn(__fsub_rn(value<T, TEMP>(p.v[j], t), M), L32);
    os[q] = o;
  }
}

// Engine full row, merge fused into the write: thread 0 of every (row, leaf) CTA evaluates merge_leaves
// (the identical expression merge_rows uses) and shares (M, L32); then out = (v - M) - L32.
template <class T, bool TEMP>
__global__ __launch_bounds__(256) void merge_write_rows(float *__restrict__ out, int64_t ostride,
                                                        const T *__restrict__ x, int64_t stride, int L,
                                                        const float *__restrict__ leaves,
                                                        const float *__restrict__ temp) {
  int row = blockIdx.x, g = blockIdx.y;
  __shared__ float ml[2];
  if (threadIdx.x == 0) {
    float M;
    double ls;
    merge_leaves(leaves + int64_t(row) * 2 * G, leaves + int64_t(row) * 2 * G + G, M, ls);
    ml[0] = M;
    ml[1] = float(ls);
  }
  __syncthreads();
  const auto *xs = reinterpret_cast<const pack8<T> *>(x + int64_t(row) * stride + int64_t(g) * L);
  auto *os = reinterpret_cast<pack8<float> *>(out + int64_t(row) * ostride + int64_t(g) * L);
  float M = ml[0], L32 = ml[1];
  float t = 1.f;
  if constexpr (TEMP)
    t = *temp;
  int strips = L / STRIP;
  for (int q = threadIdx.x; q < strips; q += 256) {
    pack8<T> p = xs[q];
    pack8<float> o;
#pragma unroll
    for (int j = 0; j < STRIP; ++j)
      o.v[j] = __fsub_rn(__fsub_rn(value<T, TEMP>(p.v[j], t), M), L32);
    os[q] = o;
  }
}

// VJP of the sampled logprob: dx = g * (onehot(target) - exp(x - lse)); one read and one write per
// column. No bitwise contract (backward).
template <class T, class TO>
__global__ __launch_bounds__(256) void grad_rows(TO *__restrict__ dx, int64_t dstride,
                                                 const T *__restrict__ x, int64_t stride, int L,
                                                 const double *__restrict__ lse,
                                                 const float *__restrict__ gout,
                                                 const int64_t *__restrict__ ltarget) {
  int row = blockIdx.x, g = blockIdx.y;
  const auto *xs = reinterpret_cast<const pack8<T> *>(x + int64_t(row) * stride + int64_t(g) * L);
  auto *os = reinterpret_cast<pack8<TO> *>(dx + int64_t(row) * dstride + int64_t(g) * L);
  float gr = gout[row];
  double ls = lse[row];
  int64_t tg = ltarget[row] - int64_t(g) * L;  // target column relative to this leaf
  int strips = L / STRIP;
  for (int q = threadIdx.x; q < strips; q += 256) {
    pack8<T> p = xs[q];
    pack8<TO> o;
#pragma unroll
    for (int j = 0; j < STRIP; ++j) {
      float pr = expf(float(double(widen(p.v[j])) - ls));
      float v = -gr * pr;
      if (int64_t(q) * STRIP + j == tg)
        v = v + gr;
      if constexpr (std::is_same_v<TO, bf16>)
        o.v[j] = __float2bfloat16_rn(v);
      else
        o.v[j] = v;
    }
    os[q] = o;
  }
}

}  // namespace
}  // namespace logprobs_sm100_v2

using namespace logprobs_sm100_v2;

namespace {

void check_rows(const at::Tensor &x) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2, "logprobs_sm100_v2: expected CUDA [rows, V]");
  TORCH_CHECK(x.dtype() == at::kFloat || x.dtype() == at::kBFloat16, "logprobs_sm100_v2: fp32/bf16 rows");
  TORCH_CHECK(x.size(0) <= 1 || x.size(1) == 0 || x.stride(1) == 1, "logprobs_sm100_v2: row stride(1) must be 1");
  TORCH_CHECK(uintptr_t(x.data_ptr()) % 16 == 0, "logprobs_sm100_v2: rows must be 16-byte aligned");
}

const float *temp_ptr(const c10::optional<at::Tensor> &temperature) {
  if (!temperature.has_value())
    return nullptr;
  TORCH_CHECK(temperature->dtype() == at::kFloat && temperature->is_cuda() && temperature->numel() == 1,
              "logprobs_sm100_v2: temperature must be a CUDA fp32 scalar tensor");
  return temperature->data_ptr<float>();
}

}  // namespace

// Leaf states of the local rows: x [rows, V_local] holds `leaves_local` whole leaves.
void leaf_states_cuda(const at::Tensor &x, int64_t leaves_local, at::Tensor mleaf, at::Tensor sleaf,
                      const c10::optional<at::Tensor> &temperature, const c10::optional<at::Tensor> &aleaf) {
  check_rows(x);
  const at::cuda::CUDAGuard guard(x.device());
  int rows = x.size(0), V = x.size(1);
  if (rows == 0)
    return;
  TORCH_CHECK(leaves_local > 0 && V % (leaves_local * STRIP) == 0, "logprobs_sm100_v2: leaf geometry");
  int L = V / leaves_local;
  const float *tp = temp_ptr(temperature);
  dim3 grid(rows, leaves_local);
  auto stream = at::cuda::getCurrentCUDAStream();
  bool xbf = x.dtype() == at::kBFloat16;
  if (aleaf.has_value()) {
    TORCH_CHECK(aleaf->dtype() == at::kFloat && aleaf->sizes() == mleaf.sizes() && aleaf->is_contiguous(),
                "logprobs_sm90: aleaf must be fp32 like mleaf");
    float *ap = aleaf->data_ptr<float>();
#define LEAFE(T, TEMP)                                                                             \
  return leaf_state_ent<T, TEMP><<<grid, LANES, 0, stream>>>(                                      \
      reinterpret_cast<const T *>(x.data_ptr()), x.stride(0), L, mleaf.data_ptr<float>(),          \
      sleaf.data_ptr<float>(), ap, int(leaves_local), tp);
    if (xbf && tp) LEAFE(bf16, true)
    if (xbf && !tp) LEAFE(bf16, false)
    if (!xbf && tp) LEAFE(float, true)
    LEAFE(float, false)
#undef LEAFE
  }
#define LEAF(T, TEMP)                                                                              \
  return leaf_state<T, TEMP><<<grid, LANES, 0, stream>>>(                                          \
      reinterpret_cast<const T *>(x.data_ptr()), x.stride(0), L, mleaf.data_ptr<float>(),          \
      sleaf.data_ptr<float>(), int(leaves_local), tp);
  if (xbf && tp) LEAF(bf16, true)
  if (xbf && !tp) LEAF(bf16, false)
  if (!xbf && tp) LEAF(float, true)
  LEAF(float, false)
#undef LEAF
}

// Row merge. x + target given -> gather the sampled column; sampled_raw + target -> use it; no
// `sampled` -> no sampled output. Writes rM, rL, lse (optional) and sampled (optional).
void merge_rows_cuda(const at::Tensor &mleaf, const at::Tensor &sleaf, at::Tensor rM, at::Tensor rL,
                     const c10::optional<at::Tensor> &lse, const c10::optional<at::Tensor> &sampled,
                     const c10::optional<at::Tensor> &x, const c10::optional<at::Tensor> &target,
                     const c10::optional<at::Tensor> &sampled_raw, int64_t vocab,
                     const c10::optional<at::Tensor> &temperature) {
  const at::cuda::CUDAGuard guard(mleaf.device());
  int rows = mleaf.size(0);
  if (rows == 0)
    return;
  TORCH_CHECK(mleaf.size(1) == G && sleaf.size(1) == G, "logprobs_sm100_v2: merge needs all G leaves");
  const float *tp = temp_ptr(temperature);
  int src = sampled.has_value() ? (x.has_value() ? 1 : 2) : 0;
  bool xbf = x.has_value() && x->dtype() == at::kBFloat16;
  auto stream = at::cuda::getCurrentCUDAStream();
#define MERGE(T, TEMP, SRC)                                                                          \
  return merge_rows<T, TEMP, SRC><<<(rows + 31) / 32, 32, 0, stream>>>(                              \
      mleaf.data_ptr<float>(), sleaf.data_ptr<float>(), rM.data_ptr<float>(), rL.data_ptr<float>(), \
      lse.has_value() ? lse->data_ptr<double>() : nullptr,                                           \
      sampled.has_value() ? sampled->data_ptr<float>() : nullptr,                                    \
      x.has_value() ? reinterpret_cast<const T *>(x->data_ptr()) : nullptr,                          \
      x.has_value() ? x->stride(0) : 0, target.has_value() ? target->data_ptr<int64_t>() : nullptr, \
      sampled_raw.has_value() ? sampled_raw->data_ptr<float>() : nullptr, vocab, rows, tp);
  if (src == 0) MERGE(float, false, 0)
  if (src == 2) MERGE(float, false, 2)
  if (xbf && tp) MERGE(bf16, true, 1)
  if (xbf && !tp) MERGE(bf16, false, 1)
  if (!xbf && tp) MERGE(float, true, 1)
  MERGE(float, false, 1)
#undef MERGE
}

void write_rows_cuda(at::Tensor out, const at::Tensor &x, const at::Tensor &rM, const at::Tensor &rL,
                     const c10::optional<at::Tensor> &temperature) {
  check_rows(x);
  const at::cuda::CUDAGuard guard(x.device());
  int rows = x.size(0), V = x.size(1);
  if (rows == 0)
    return;
  TORCH_CHECK(out.dtype() == at::kFloat && (rows <= 1 || out.stride(1) == 1) &&
                  uintptr_t(out.data_ptr()) % 16 == 0,
              "logprobs_sm100_v2: out must be aligned fp32 rows");
  int L = V / G;
  const float *tp = temp_ptr(temperature);
  bool xbf = x.dtype() == at::kBFloat16;
  auto stream = at::cuda::getCurrentCUDAStream();
#define WRITE(T, TEMP)                                                                               \
  return write_rows<T, TEMP><<<dim3(rows, G), 256, 0, stream>>>(                                     \
      out.data_ptr<float>(), out.stride(0), reinterpret_cast<const T *>(x.data_ptr()), x.stride(0), \
      L, rM.data_ptr<float>(), rL.data_ptr<float>(), tp);
  if (xbf && tp) WRITE(bf16, true)
  if (xbf && !tp) WRITE(bf16, false)
  if (!xbf && tp) WRITE(float, true)
  WRITE(float, false)
#undef WRITE
}

// Engine steady state in one call: two launches (leaf states; fused merge + write), one scratch buffer.
at::Tensor full_rows_cuda(const at::Tensor &x, const c10::optional<at::Tensor> &temperature) {
  check_rows(x);
  const at::cuda::CUDAGuard guard(x.device());
  int rows = x.size(0), V = x.size(1);
  auto fopt = x.options().dtype(at::kFloat);
  auto out = at::empty({rows, V}, fopt);
  if (rows == 0)
    return out;
  TORCH_CHECK(V % (G * STRIP) == 0, "logprobs_sm100_v2: vocab must be a multiple of 64");
  int L = V / G;
  auto leaves = at::empty({rows, 2 * G}, fopt);  // [M_0..M_7, S_0..S_7] per row
  const float *tp = temp_ptr(temperature);
  bool xbf = x.dtype() == at::kBFloat16;
  auto stream = at::cuda::getCurrentCUDAStream();
  float *lp = leaves.data_ptr<float>();
#define FULL(T, TEMP)                                                                                \
  {                                                                                                  \
    leaf_state_split<T, TEMP><<<dim3(rows, G), LANES, 0, stream>>>(                                  \
        reinterpret_cast<const T *>(x.data_ptr()), x.stride(0), L, lp, tp);                          \
    merge_write_rows<T, TEMP><<<dim3(rows, G), 256, 0, stream>>>(                                    \
        out.data_ptr<float>(), out.stride(0), reinterpret_cast<const T *>(x.data_ptr()), x.stride(0),\
        L, lp, tp);                                                                                  \
    return out;                                                                                      \
  }
  if (xbf && tp) FULL(bf16, true)
  if (xbf && !tp) FULL(bf16, false)
  if (!xbf && tp) FULL(float, true)
  FULL(float, false)
#undef FULL
}

// dx [rows, V_local] (fp32 or bf16); local_target = target - vocab_start, anything outside
// [0, V_local) contributes no one-hot term.
void grad_rows_cuda(at::Tensor dx, const at::Tensor &x, const at::Tensor &lse, const at::Tensor &gout,
                    const at::Tensor &local_target, int64_t leaves_local) {
  check_rows(x);
  const at::cuda::CUDAGuard guard(x.device());
  int rows = x.size(0), V = x.size(1);
  if (rows == 0)
    return;
  TORCH_CHECK((rows <= 1 || dx.stride(1) == 1) && uintptr_t(dx.data_ptr()) % 16 == 0,
              "logprobs_sm100_v2: dx layout");
  int L = V / leaves_local;
  bool xbf = x.dtype() == at::kBFloat16, obf = dx.dtype() == at::kBFloat16;
  auto stream = at::cuda::getCurrentCUDAStream();
#define GRAD(T, TO)                                                                                  \
  return grad_rows<T, TO><<<dim3(rows, leaves_local), 256, 0, stream>>>(                             \
      reinterpret_cast<TO *>(dx.data_ptr()), dx.stride(0), reinterpret_cast<const T *>(x.data_ptr()), \
      x.stride(0), L, lse.data_ptr<double>(), gout.data_ptr<float>(), local_target.data_ptr<int64_t>());
  if (xbf && obf) GRAD(bf16, bf16)
  if (xbf && !obf) GRAD(bf16, float)
  if (!xbf && obf) GRAD(float, bf16)
  GRAD(float, float)
#undef GRAD
}

// Row entropy from all G leaf states (M, S, A) of every row.
void entropy_rows_cuda(const at::Tensor &mleaf, const at::Tensor &sleaf, const at::Tensor &aleaf, at::Tensor ent) {
  const at::cuda::CUDAGuard guard(mleaf.device());
  int rows = mleaf.size(0);
  if (rows == 0)
    return;
  TORCH_CHECK(mleaf.size(1) == G && sleaf.size(1) == G && aleaf.size(1) == G, "logprobs_sm90: entropy needs all G leaves");
  TORCH_CHECK(mleaf.is_contiguous() && sleaf.is_contiguous() && aleaf.is_contiguous() && ent.is_contiguous(),
              "logprobs_sm90: entropy operands must be contiguous");
  entropy_rows<<<(rows + 31) / 32, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
      mleaf.data_ptr<float>(), sleaf.data_ptr<float>(), aleaf.data_ptr<float>(), ent.data_ptr<float>(), rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
