// SPDX-License-Identifier: Apache-2.0
// Row-invariant fp32 log-softmax / sampled logprob on SM100 (arithmetic version 3: the schedule of
// isoexec/ops/schedule_policy.py; stated in schedule.py and reference.py).
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <type_traits>

// ---- The numerical schedule of the log-sum-exp ---------------------------------------------------
// The vocabulary is G = d_max consecutive shards of L = V / G columns (a TP rank holds G / world whole
// shards). Inside a shard: segments of S consecutive columns, the last one short, combined by the
// adjacent tree (pair (0,1), (2,3), ...; an unpaired last partial moves up); then the adjacent tree over
// the G shards. S and G are not written here: logprobs_sm100/schedule.py takes them from
// isoexec/ops/schedule_policy.py (rule 3, scalar step) and the declared d_max and passes them to the
// compiler.
#if !defined(LOGPROBS_SM100_SEGMENT) || !defined(LOGPROBS_SM100_SHARDS)
#error "logprobs_sm100: build through _extension.py; schedule.py supplies the segment length and d_max"
#endif

namespace logprobs_sm100 {
namespace {

using bf16 = __nv_bfloat16;
constexpr int SEG = LOGPROBS_SM100_SEGMENT;  // S: columns of a segment (schedule)
constexpr int G = LOGPROBS_SM100_SHARDS;     // d_max: vocabulary shards of a full row (declaration)
constexpr int STRIP = 8;                     // columns per load (execution; a shard is whole strips)
constexpr int LANES = 256;                   // threads of a (row, shard) CTA (execution)
constexpr int MAXB = 128;                    // segments one thread may own (execution; stack depth 8)
constexpr float NINF = -__builtin_huge_valf();
static_assert(SEG % STRIP == 0 && SEG >= STRIP && SEG <= 64, "logprobs_sm100: a segment is 1-8 whole strips");
static_assert(G >= 1 && G <= 32 && (G & (G - 1)) == 0, "logprobs_sm100: d_max must be a power of two <= 32");

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

// A partial (m, s[, a]) of a set of columns: m = their maximum (-inf for none; NaN never wins),
// s = sum exp(v - F), a = sum exp(v - F) * (v - F) (entropy only; a -inf column adds +0), with
// F = m, or +0 when m == -inf so that an all -inf set keeps s == +0.
struct Part {
  float m, s, a;
};

// One segment: columns ascending, fp32. m = fmaxf chain from -inf; s (and a) = add.rn chains from +0.
template <class T, bool TEMP, bool ENT>
__device__ __forceinline__ Part segment(const pack8<T> *__restrict__ xs, int nstrip, float t) {
  float v[SEG];
  float m = NINF;
#pragma unroll
  for (int k = 0; k < SEG / STRIP; ++k) {
    if (k < nstrip) {
      pack8<T> p = xs[k];
#pragma unroll
      for (int j = 0; j < STRIP; ++j) {
        v[k * STRIP + j] = value<T, TEMP>(p.v[j], t);
        m = fmaxf(m, v[k * STRIP + j]);
      }
    }
  }
  float F = m == NINF ? 0.f : m;
  float s = 0.f, a = 0.f;
#pragma unroll
  for (int k = 0; k < SEG / STRIP; ++k) {
    if (k < nstrip) {
#pragma unroll
      for (int j = 0; j < STRIP; ++j) {
        float d = __fsub_rn(v[k * STRIP + j], F);
        float e = expf(d);
        s = __fadd_rn(s, e);
        if constexpr (ENT)
          a = __fadd_rn(a, v[k * STRIP + j] == NINF ? 0.f : __fmul_rn(e, d));
      }
    }
  }
  return {m, s, a};
}

// The reducer's merge, left operand first: both partials are rescaled to the larger maximum and added.
// A side with m == -inf holds s == +0 (or NaN from a NaN column, which propagates) and adds s * 0.
template <bool ENT> __device__ __forceinline__ void merge(Part &l, const Part &r) {
  float M = fmaxf(l.m, r.m);
  float F = M == NINF ? 0.f : M;
  float dl = __fsub_rn(l.m, F), dr = __fsub_rn(r.m, F);
  float el = expf(dl), er = expf(dr);
  if constexpr (ENT) {
    float al = l.m == NINF ? __fmul_rn(l.s, 0.f) : __fmul_rn(el, __fadd_rn(l.a, __fmul_rn(l.s, dl)));
    float ar = r.m == NINF ? __fmul_rn(r.s, 0.f) : __fmul_rn(er, __fadd_rn(r.a, __fmul_rn(r.s, dr)));
    l.a = __fadd_rn(al, ar);
  }
  l.s = __fadd_rn(__fmul_rn(l.s, el), __fmul_rn(r.s, er));
  l.m = M;
}

// Shard state: the partial of shard g of the row, i.e. the adjacent tree over its ceil(L / S) segments.
// Realisation: thread i owns the aligned block of B consecutive segments [i B, i B + B) (B a power of
// two chosen by the host so that at most LANES blocks exist) and merges them in tree order with a
// binary-counter stack; the blocks' partials then meet in shared memory, lower block on the left, a
// block with no right neighbour moving up unchanged. B and LANES decide who computes what, not what.
template <class T, bool TEMP, bool PACKED, bool ENT = false>
__device__ __forceinline__ void leaf_state_body(const T *__restrict__ x, int64_t stride, int L,
                                                float *__restrict__ mout, float *__restrict__ sout,
                                                int gl, const float *__restrict__ temp, int B,
                                                float *__restrict__ aout = nullptr) {
  int row = blockIdx.x, g = blockIdx.y, i = threadIdx.x;
  const auto *xs = reinterpret_cast<const pack8<T> *>(x + int64_t(row) * stride + int64_t(g) * L);
  const int strips = L / STRIP, per = SEG / STRIP;
  const int nseg = (strips + per - 1) / per, nblk = (nseg + B - 1) / B;
  __shared__ float pm[LANES];
  __shared__ float ps[LANES];
  __shared__ float pa[ENT ? LANES : 1];
  float t = 1.f;
  if constexpr (TEMP)
    t = *temp;
  Part p{NINF, 0.f, 0.f};
  if (i < nblk) {
    const int l0 = i * B, nl = min(B, nseg - l0);
    Part st[8];
    int top = 0;
    for (int l = 0; l < nl; ++l) {
      const int q = (l0 + l) * per;
      Part cur = segment<T, TEMP, ENT>(xs + q, min(per, strips - q), t);
      for (int k = l; k & 1; k >>= 1) { // the left sibling is on the stack: merge(left, this)
        --top;
        merge<ENT>(st[top], cur);
        cur = st[top];
      }
      st[top++] = cur;
    }
    p = st[top - 1];
    for (int k = top - 2; k >= 0; --k) { // a short block: unpaired nodes move up to their left neighbour
      merge<ENT>(st[k], p);
      p = st[k];
    }
    pm[i] = p.m;
    ps[i] = p.s;
    if constexpr (ENT)
      pa[i] = p.a;
  }
  for (int step = 1; step < LANES; step *= 2) {
    __syncthreads();
    if (i % (2 * step) == 0 && i + step < nblk) {
      merge<ENT>(p, Part{pm[i + step], ps[i + step], ENT ? pa[i + step] : 0.f});
      pm[i] = p.m;
      ps[i] = p.s;
      if constexpr (ENT)
        pa[i] = p.a;
    }
  }
  if (i == 0) {
    if constexpr (ENT)
      aout[int64_t(row) * gl + g] = p.a;
    if constexpr (PACKED) {
      mout[int64_t(row) * 2 * G + g] = p.m;
      mout[int64_t(row) * 2 * G + G + g] = p.s;
    } else {
      mout[int64_t(row) * gl + g] = p.m;
      sout[int64_t(row) * gl + g] = p.s;
    }
  }
}

template <class T, bool TEMP>
__global__ __launch_bounds__(LANES) void leaf_state(const T *__restrict__ x, int64_t stride, int L,
                                                    float *__restrict__ mout, float *__restrict__ sout,
                                                    int gl, const float *__restrict__ temp, int B) {
  leaf_state_body<T, TEMP, false>(x, stride, L, mout, sout, gl, temp, B);
}

template <class T, bool TEMP>
__global__ __launch_bounds__(LANES) void leaf_state_ent(const T *__restrict__ x, int64_t stride, int L,
                                                        float *__restrict__ mout, float *__restrict__ sout,
                                                        float *__restrict__ aout, int gl,
                                                        const float *__restrict__ temp, int B) {
  leaf_state_body<T, TEMP, false, true>(x, stride, L, mout, sout, gl, temp, B, aout);
}

template <class T, bool TEMP>
__global__ __launch_bounds__(LANES) void leaf_state_split(const T *__restrict__ x, int64_t stride, int L,
                                                          float *__restrict__ leaves,
                                                          const float *__restrict__ temp, int B) {
  leaf_state_body<T, TEMP, true>(x, stride, L, leaves, nullptr, G, temp, B);
}

// The row's partial from its G shard partials: the adjacent tree with the same merge, lower shard on
// the left. (Shared by merge_rows, entropy_rows and the fused full-row write: one expression.)
template <bool ENT>
__device__ __forceinline__ Part row_root(const float *mg, const float *sg, const float *ag) {
  Part w[G];
#pragma unroll
  for (int g = 0; g < G; ++g)
    w[g] = Part{mg[g], sg[g], ENT ? ag[g] : 0.f};
#pragma unroll
  for (int step = 1; step < G; step *= 2)
#pragma unroll
    for (int g = 0; g + step < G; g += 2 * step)
      merge<ENT>(w[g], w[g + step]);
  return w[0];
}

// (M, log S): logS = log(double(S)) (NaN when M == -inf: no finite column).
__device__ __forceinline__ void merge_leaves(const float *mg, const float *sg, float &Mout, double &lsout) {
  Part r = row_root<false>(mg, sg, nullptr);
  Mout = r.m;
  lsout = r.m == NINF ? double(__builtin_nan("")) : log(double(r.s));
}

// Row merge over the G shard partials. Published per row: M, L32 = RN32(logS) and, for the backward,
// lse = M + logS (double). Sampled logprob = (v_t - M) - L32 in fp32: the expression the full-row write
// evaluates at that column.
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

// Row entropy (a metric, not part of the scoring contract; deterministic and TP invariant because it
// is the same tree over the same shards): H = log(S) - A / S in double from the row's (M, S, A).
__global__ __launch_bounds__(32) void entropy_rows(const float *__restrict__ mleaf,
                                                   const float *__restrict__ sleaf,
                                                   const float *__restrict__ aleaf,
                                                   float *__restrict__ ent, int rows) {
  int row = blockIdx.x * 32 + threadIdx.x;
  if (row >= rows)
    return;
  Part r = row_root<true>(mleaf + int64_t(row) * G, sleaf + int64_t(row) * G, aleaf + int64_t(row) * G);
  ent[row] = r.m == NINF ? __builtin_nanf("") : float(log(double(r.s)) - double(r.a) / double(r.s));
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
}  // namespace logprobs_sm100

using namespace logprobs_sm100;

// (segment length S, vocabulary shards d_max) this extension was compiled with.
std::tuple<int, int> schedule_cuda() { return {SEG, G}; }

namespace {

void check_rows(const at::Tensor &x) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2, "logprobs_sm100: expected CUDA [rows, V]");
  TORCH_CHECK(x.dtype() == at::kFloat || x.dtype() == at::kBFloat16, "logprobs_sm100: fp32/bf16 rows");
  TORCH_CHECK(x.size(0) <= 1 || x.size(1) == 0 || x.stride(1) == 1, "logprobs_sm100: row stride(1) must be 1");
  TORCH_CHECK(uintptr_t(x.data_ptr()) % 16 == 0, "logprobs_sm100: rows must be 16-byte aligned");
}

// Segments per thread for a shard of L columns: the smallest power of two >= 4 (128 columns a thread, the
// measured choice) that leaves at most LANES blocks. Execution only.
int block_segments(int L) {
  int nseg = (L + SEG - 1) / SEG, B = 4;
  while ((nseg + B - 1) / B > LANES)
    B *= 2;
  TORCH_CHECK(B <= MAXB, "logprobs_sm100: a shard of ", L, " columns is larger than one CTA reduces");
  return B;
}

const float *temp_ptr(const c10::optional<at::Tensor> &temperature) {
  if (!temperature.has_value())
    return nullptr;
  TORCH_CHECK(temperature->dtype() == at::kFloat && temperature->is_cuda() && temperature->numel() == 1,
              "logprobs_sm100: temperature must be a CUDA fp32 scalar tensor");
  return temperature->data_ptr<float>();
}

}  // namespace

// Shard states of the local rows: x [rows, V_local] holds `leaves_local` whole shards ("leaves" in the API).
void leaf_states_cuda(const at::Tensor &x, int64_t leaves_local, at::Tensor mleaf, at::Tensor sleaf,
                      const c10::optional<at::Tensor> &temperature, const c10::optional<at::Tensor> &aleaf) {
  check_rows(x);
  const at::cuda::CUDAGuard guard(x.device());
  int rows = x.size(0), V = x.size(1);
  if (rows == 0)
    return;
  TORCH_CHECK(leaves_local > 0 && V % (leaves_local * STRIP) == 0, "logprobs_sm100: leaf geometry");
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
      sleaf.data_ptr<float>(), ap, int(leaves_local), tp, block_segments(L));
    if (xbf && tp) LEAFE(bf16, true)
    if (xbf && !tp) LEAFE(bf16, false)
    if (!xbf && tp) LEAFE(float, true)
    LEAFE(float, false)
#undef LEAFE
  }
#define LEAF(T, TEMP)                                                                              \
  return leaf_state<T, TEMP><<<grid, LANES, 0, stream>>>(                                          \
      reinterpret_cast<const T *>(x.data_ptr()), x.stride(0), L, mleaf.data_ptr<float>(),          \
      sleaf.data_ptr<float>(), int(leaves_local), tp, block_segments(L));
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
  TORCH_CHECK(mleaf.size(1) == G && sleaf.size(1) == G, "logprobs_sm100: merge needs all G shard partials");
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
              "logprobs_sm100: out must be aligned fp32 rows");
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
  TORCH_CHECK(V % (G * STRIP) == 0, "logprobs_sm100: vocab must be whole 8-column strips per shard");
  int L = V / G;
  auto leaves = at::empty({rows, 2 * G}, fopt);  // [M_0..M_7, S_0..S_7] per row
  const float *tp = temp_ptr(temperature);
  bool xbf = x.dtype() == at::kBFloat16;
  auto stream = at::cuda::getCurrentCUDAStream();
  float *lp = leaves.data_ptr<float>();
#define FULL(T, TEMP)                                                                                \
  {                                                                                                  \
    leaf_state_split<T, TEMP><<<dim3(rows, G), LANES, 0, stream>>>(                                  \
        reinterpret_cast<const T *>(x.data_ptr()), x.stride(0), L, lp, tp, block_segments(L));       \
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
              "logprobs_sm100: dx layout");
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
