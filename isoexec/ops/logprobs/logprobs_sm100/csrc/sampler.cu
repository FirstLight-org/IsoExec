// SPDX-License-Identifier: Apache-2.0
// Fused decode sampler (SM100): Gumbel-max token, rank and top-1 under the rowinv recipe.
//
// Sampling: Gumbel-max over the sampling logits, token = argmax_j (x_j / t + g_j), greedy rows
// (t < eps) take argmax_j x_j. The noise g_j = -log(-log(u_j)) is a pure function of
// (seed, offset, row, column) via Philox4x32-10, so a row's token never depends on the other rows,
// the CTA split or the launch geometry. The argmax order is total -- larger score first, then the
// smaller column -- and NaN scores never win, so every partial combine is exact and order-free.
//
// Logprobs: the rowinv recipe's (M, L32) per row come from the logprobs extension's own leaf_states
// + merge_rows (the trainer's kernels). This file only re-evaluates the full-row write expression
// lp_j = (v_j - M) - L32 (fp32, round-to-nearest, v_j = x_j or x_j / T with __fdiv_rn) to count the
// sampled token's rank and pick the top-1 logprob, with torch.topk's order (NaN first, then larger,
// then the smaller column). No atomics: every cross-CTA result goes through a partial buffer and a
// fixed-order finalize.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <type_traits>

namespace isoexec_fused_sampler {
namespace {

using bf16 = __nv_bfloat16;
constexpr int STRIP = 8;
constexpr int THREADS = 256;
constexpr int NONE = 0x7fffffff;
constexpr float NINF = -__builtin_huge_valf();
constexpr float SAMPLING_EPS = 1e-5f;  // vLLM's _SAMPLING_EPS: below it a row is greedy

template <class T> __device__ __forceinline__ float widen(T v) {
  if constexpr (std::is_same_v<T, bf16>)
    return __bfloat162float(v);
  else
    return v;
}

template <class T> struct __align__(16) pack8 {
  T v[8];
};

// ---- Philox4x32-10 -------------------------------------------------------------------------------
struct u4 {
  uint32_t x, y, z, w;
};

__device__ __forceinline__ u4 philox(u4 c, uint32_t k0, uint32_t k1) {
#pragma unroll
  for (int r = 0; r < 10; ++r) {
    uint32_t hi0 = __umulhi(0xD2511F53u, c.x), lo0 = 0xD2511F53u * c.x;
    uint32_t hi1 = __umulhi(0xCD9E8D57u, c.z), lo1 = 0xCD9E8D57u * c.z;
    c = u4{hi1 ^ c.y ^ k0, lo1, hi0 ^ c.w ^ k1, lo0};
    k0 += 0x9E3779B9u;
    k1 += 0xBB67AE85u;
  }
  return c;
}

// Gumbel noise from 23 random bits: u = (k + 0.5) * 2^-23 is exact in fp32 and lies in
// [2^-24, 1 - 2^-24]. E = -log(u) > 0 uses the full-precision logf, so the u -> 1 tail (tiny E,
// the largest noise, the draws that win) keeps its relative accuracy; g = -log(E) uses the fast
// __logf, whose error is relative except at E ~ 1, where it is an absolute ~2^-21 on g ~ 0.
// g lies in [-2.82, 16.64].
__device__ __forceinline__ float gumbel(uint32_t r) {
  float u = __fmul_rn(__uint2float_rn(r >> 9) + 0.5f, 1.1920928955078125e-7f);  // 2^-23
  float e = -logf(u);
  return -__logf(e);
}

// An upper bound on every draw: the largest, at u = 1 - 2^-24, is -log(-log(u)) = 16.64 (both logs
// are within a few ulp there), so 17 bounds them all without relying on either log being monotone.
__device__ __forceinline__ float gmax() { return 17.0f; }

// (score, column) order: larger score first, then the smaller column; NaN never wins.
__device__ __forceinline__ bool beats(float s, int i, float bs, int bi) {
  return s > bs || (s == bs && i < bi);
}

// torch.topk order for the top-1 logprob: NaN first (smaller column among NaNs), then larger, then
// the smaller column.
__device__ __forceinline__ bool top_beats(float v, int i, float bv, int bi) {
  bool vn = v != v, bn = bv != bv;
  if (vn || bn)
    return vn && (!bn || i < bi);
  return v > bv || (v == bv && i < bi);
}

template <bool TOP>
__device__ __forceinline__ void block_argmax(float &v, int &i, float *sv, int *si) {
  // Fixed tree: warp butterfly, then warp 0 over the 8 warp winners. The order is total, so the
  // winner is the same for any tree; the tree is fixed anyway.
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    float ov = __shfl_xor_sync(0xffffffffu, v, o);
    int oi = __shfl_xor_sync(0xffffffffu, i, o);
    bool b = TOP ? top_beats(ov, oi, v, i) : beats(ov, oi, v, i);
    if (b) {
      v = ov;
      i = oi;
    }
  }
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (lane == 0) {
    sv[warp] = v;
    si[warp] = i;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int w = 1; w < THREADS / 32; ++w) {
      bool b = TOP ? top_beats(sv[w], si[w], v, i) : beats(sv[w], si[w], v, i);
      if (b) {
        v = sv[w];
        i = si[w];
      }
    }
  }
}

// A lower bound on every draw: the smallest, at u = 2^-24, is -log(-log(u)) = -2.81.
__device__ __forceinline__ float gmin() { return -3.0f; }

// grid (rows, split): partial row maxima of the sampling logits (fmaxf: exact, NaN never wins).
template <class T>
__global__ __launch_bounds__(THREADS) void row_max_partials(const T *__restrict__ x, int64_t stride, int L,
                                                            float *__restrict__ pmax) {
  int row = blockIdx.x, g = blockIdx.y, split = gridDim.y;
  const auto *xs = reinterpret_cast<const pack8<T> *>(x + int64_t(row) * stride + int64_t(g) * L);
  float m = NINF;
  int strips = L / STRIP;
  for (int q = threadIdx.x; q < strips; q += THREADS) {
    pack8<T> p = xs[q];
#pragma unroll
    for (int j = 0; j < STRIP; ++j)
      m = fmaxf(m, widen(p.v[j]));
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1)
    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  __shared__ float sm[THREADS / 32];
  if ((threadIdx.x & 31) == 0)
    sm[threadIdx.x >> 5] = m;
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int w = 1; w < THREADS / 32; ++w)
      m = fmaxf(m, sm[w]);
    pmax[int64_t(row) * split + g] = m;
  }
}

// grid (rows, split); CTA g covers columns [g * L, (g + 1) * L), L a multiple of STRIP.
//
// Exact prune (when pmax is given): the column holding the row max M scores at least
// LB = (M / t) + gmin(), and a column with (x / t) + gmax() < LB scores strictly below LB, so it can
// neither win nor tie; its noise is never drawn (the same holds against the thread's running best).
// fdiv_rn / fadd_rn are monotone, so the bound survives rounding. The token equals the unpruned one.
template <class T>
__global__ __launch_bounds__(THREADS) void gumbel_partials(
    const T *__restrict__ x, int64_t stride, int L, const float *__restrict__ temp, int greedy_all,
    uint32_t seed_lo, uint32_t seed_hi, uint32_t off_lo, uint32_t off_hi, int prune, const float *__restrict__ pmax,
    int pmax_n, float *__restrict__ pval, int *__restrict__ pidx) {
  int row = blockIdx.x, g = blockIdx.y, split = gridDim.y;
  int64_t c0 = int64_t(g) * L;
  const auto *xs = reinterpret_cast<const pack8<T> *>(x + int64_t(row) * stride + c0);
  float t = greedy_all ? 0.f : temp[row];
  bool greedy = !(t >= SAMPLING_EPS);
  bool unit = t == 1.f;  // x / 1 == x exactly: skip the divide
  float lb = NINF;
  if (pmax != nullptr && !greedy) {
    float m = NINF;
    for (int k = 0; k < pmax_n; ++k)
      m = fmaxf(m, pmax[int64_t(row) * pmax_n + k]);
    lb = __fadd_rn(unit ? m : __fdiv_rn(m, t), gmin());
  }
  float best = NINF;
  int bi = NONE;
  int strips = L / STRIP;
  for (int q = threadIdx.x; q < strips; q += THREADS) {
    pack8<T> p = xs[q];
    int col = int(c0) + q * STRIP;
    float s[STRIP];
    if (greedy) {
#pragma unroll
      for (int j = 0; j < STRIP; ++j)
        s[j] = widen(p.v[j]);
    } else {
      // prune == 0 (tests): a NaN threshold fails every `<`, so every column draws its noise.
      float thr = prune ? fmaxf(lb, best) : __int_as_float(0x7fffffff);
      float xt[STRIP];
      bool any = false;
#pragma unroll
      for (int j = 0; j < STRIP; ++j) {
        xt[j] = unit ? widen(p.v[j]) : __fdiv_rn(widen(p.v[j]), t);
        any |= !(__fadd_rn(xt[j], gmax()) < thr);
      }
      if (!any)
        continue;
      u4 r0 = philox(u4{uint32_t(col >> 2), uint32_t(row), off_lo, off_hi}, seed_lo, seed_hi);
      u4 r1 = philox(u4{uint32_t((col >> 2) + 1), uint32_t(row), off_lo, off_hi}, seed_lo, seed_hi);
      uint32_t r[STRIP] = {r0.x, r0.y, r0.z, r0.w, r1.x, r1.y, r1.z, r1.w};
#pragma unroll
      for (int j = 0; j < STRIP; ++j)
        s[j] = __fadd_rn(xt[j], gumbel(r[j]));
    }
#pragma unroll
    for (int j = 0; j < STRIP; ++j)
      if (beats(s[j], col + j, best, bi)) {
        best = s[j];
        bi = col + j;
      }
  }
  __shared__ float sv[THREADS / 32];
  __shared__ int si[THREADS / 32];
  block_argmax<false>(best, bi, sv, si);
  if (threadIdx.x == 0) {
    pval[int64_t(row) * split + g] = best;
    pidx[int64_t(row) * split + g] = bi;
  }
}

// One warp per row (split <= 32): lane g holds partial g; the (score, column) order is total, so the
// butterfly's winner is the unique maximum whatever the tree.
__global__ void gumbel_finalize(const float *__restrict__ pval, const int *__restrict__ pidx, int split,
                                int rows, int64_t *__restrict__ token) {
  int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (row >= rows)
    return;
  float best = NINF;
  int bi = NONE;
  if (lane < split) {
    best = pval[int64_t(row) * split + lane];
    bi = pidx[int64_t(row) * split + lane];
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    float ov = __shfl_xor_sync(0xffffffffu, best, o);
    int oi = __shfl_xor_sync(0xffffffffu, bi, o);
    if (beats(ov, oi, best, bi)) {
      best = ov;
      bi = oi;
    }
  }
  if (lane == 0)
    token[row] = bi == NONE ? 0 : bi;  // an all-NaN row (no finite order) samples column 0
}

// The rowinv input value, exactly as logprobs.cu's value<T, TEMP>.
template <class T, bool TEMP> __device__ __forceinline__ float value(T raw, float t) {
  if constexpr (TEMP)
    return __fdiv_rn(widen(raw), t);
  else
    return widen(raw);
}

// grid (rows, split): rank count (lp_j >= lp_tok) and top-1 partials of lp_j = (v_j - M) - L32.
template <class T, bool TEMP>
__global__ __launch_bounds__(THREADS) void rank_partials(
    const T *__restrict__ x, int64_t stride, int L, const float *__restrict__ rM, const float *__restrict__ rL,
    const float *__restrict__ lp_tok, const float *__restrict__ temp, int *__restrict__ pcnt,
    float *__restrict__ ptop, int *__restrict__ ptopi) {
  int row = blockIdx.x, g = blockIdx.y, split = gridDim.y;
  int64_t c0 = int64_t(g) * L;
  const auto *xs = reinterpret_cast<const pack8<T> *>(x + int64_t(row) * stride + c0);
  float M = rM[row], L32 = rL[row], ref = lp_tok[row];
  float t = 1.f;
  if constexpr (TEMP)
    t = *temp;
  int cnt = 0;
  float top = NINF;
  int ti = NONE;
  int strips = L / STRIP;
  for (int q = threadIdx.x; q < strips; q += THREADS) {
    pack8<T> p = xs[q];
    int col = int(c0) + q * STRIP;
#pragma unroll
    for (int j = 0; j < STRIP; ++j) {
      float lp = __fsub_rn(__fsub_rn(value<T, TEMP>(p.v[j], t), M), L32);
      cnt += lp >= ref;
      if (top_beats(lp, col + j, top, ti)) {
        top = lp;
        ti = col + j;
      }
    }
  }
  __shared__ float sv[THREADS / 32];
  __shared__ int si[THREADS / 32];
  __shared__ int sc[THREADS / 32];
  // Integer count: exact in any order.
#pragma unroll
  for (int o = 16; o > 0; o >>= 1)
    cnt += __shfl_xor_sync(0xffffffffu, cnt, o);
  if ((threadIdx.x & 31) == 0)
    sc[threadIdx.x >> 5] = cnt;
  block_argmax<true>(top, ti, sv, si);  // contains the __syncthreads that publishes sc
  if (threadIdx.x == 0) {
    int total = 0;
    for (int w = 0; w < THREADS / 32; ++w)
      total += sc[w];
    pcnt[int64_t(row) * split + g] = total;
    ptop[int64_t(row) * split + g] = top;
    ptopi[int64_t(row) * split + g] = ti;
  }
}

// One warp per row: rank = sum of the split counts (integer, exact); top-1 under the total
// torch.topk order. Outputs in vLLM's gather_logprobs layout (column 0 the sampled token, then
// the top-1 when k == 1).
__global__ void rank_finalize(const int *__restrict__ pcnt, const float *__restrict__ ptop,
                              const int *__restrict__ ptopi, int split, int rows, int k,
                              const int64_t *__restrict__ token, const float *__restrict__ lp_tok,
                              int32_t *__restrict__ out_idx, float *__restrict__ out_lp,
                              int64_t *__restrict__ out_rank) {
  int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (row >= rows)
    return;
  int64_t total = 0;
  float top = NINF;
  int ti = NONE;
  if (lane < split) {
    int64_t o = int64_t(row) * split + lane;
    total = pcnt[o];
    top = ptop[o];
    ti = ptopi[o];
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    total += __shfl_xor_sync(0xffffffffu, total, o);
    float ov = __shfl_xor_sync(0xffffffffu, top, o);
    int oi = __shfl_xor_sync(0xffffffffu, ti, o);
    if (top_beats(ov, oi, top, ti)) {
      top = ov;
      ti = oi;
    }
  }
  if (lane != 0)
    return;
  out_rank[row] = total;
  out_idx[int64_t(row) * (1 + k)] = int32_t(token[row]);
  out_lp[int64_t(row) * (1 + k)] = lp_tok[row];
  if (k == 1) {
    out_idx[int64_t(row) * 2 + 1] = ti == NONE ? 0 : ti;
    out_lp[int64_t(row) * 2 + 1] = top;
  }
}

void check_rows(const at::Tensor &x, int split) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2, "fused_sampler: expected CUDA [rows, V]");
  TORCH_CHECK(x.dtype() == at::kFloat || x.dtype() == at::kBFloat16, "fused_sampler: fp32/bf16 rows");
  TORCH_CHECK(x.size(0) <= 1 || x.stride(1) == 1, "fused_sampler: row stride(1) must be 1");
  TORCH_CHECK(uintptr_t(x.data_ptr()) % 16 == 0 && (x.size(0) <= 1 || (x.stride(0) * x.element_size()) % 16 == 0),
              "fused_sampler: rows must be 16-byte aligned");
  TORCH_CHECK(split > 0 && split <= 32 && x.size(1) % (int64_t(split) * STRIP) == 0, "fused_sampler: vocab / split geometry");
}

}  // namespace
}  // namespace isoexec_fused_sampler

using namespace isoexec_fused_sampler;

// token [rows] int64 <- Gumbel-max over x (the sampling logits). temperature [rows] fp32 or None
// (None: every row greedy). seed / offset select the Philox stream.
void sample_cuda(const at::Tensor &x, const c10::optional<at::Tensor> &temperature, int64_t seed,
                 int64_t offset, int64_t split, bool prune, const c10::optional<at::Tensor> &row_max,
                 at::Tensor pmax, at::Tensor pval, at::Tensor pidx, at::Tensor token) {
  check_rows(x, int(split));
  const at::cuda::CUDAGuard guard(x.device());
  int rows = x.size(0), V = x.size(1);
  if (rows == 0)
    return;
  const float *tp = nullptr;
  if (temperature.has_value()) {
    TORCH_CHECK(temperature->dtype() == at::kFloat && temperature->is_cuda() && temperature->numel() == rows &&
                    temperature->is_contiguous(),
                "fused_sampler: temperature must be a contiguous CUDA fp32 [rows] tensor");
    tp = temperature->data_ptr<float>();
  }
  int L = V / int(split);
  auto stream = at::cuda::getCurrentCUDAStream();
  uint64_t s = uint64_t(seed), o = uint64_t(offset);
  // The prune bound needs the row max; a greedy-only batch draws no noise and skips it. `row_max`
  // [rows, n] holds partial maxima the caller already has (the rowinv leaf maxima of the same x).
  const float *pm = nullptr;
  int pm_n = int(split);
  bool scan = false;
  if (prune && tp != nullptr) {
    if (row_max.has_value()) {
      TORCH_CHECK(row_max->dtype() == at::kFloat && row_max->dim() == 2 && row_max->size(0) == rows &&
                      row_max->is_contiguous(),
                  "fused_sampler: row_max must be a contiguous fp32 [rows, n] tensor");
      pm = row_max->data_ptr<float>();
      pm_n = int(row_max->size(1));
    } else {
      pm = pmax.data_ptr<float>();
      scan = true;
    }
  }
#define SAMPLE(T)                                                                                    \
  if (scan)                                                                                          \
    row_max_partials<T><<<dim3(rows, int(split)), THREADS, 0, stream>>>(                            \
        reinterpret_cast<const T *>(x.data_ptr()), x.stride(0), L, pmax.data_ptr<float>());       \
  gumbel_partials<T><<<dim3(rows, int(split)), THREADS, 0, stream>>>(                               \
      reinterpret_cast<const T *>(x.data_ptr()), x.stride(0), L, tp, tp == nullptr, uint32_t(s),     \
      uint32_t(s >> 32), uint32_t(o), uint32_t(o >> 32), int(prune), pm, pm_n, pval.data_ptr<float>(),   \
      pidx.data_ptr<int>());
  if (x.dtype() == at::kBFloat16) {
    SAMPLE(bf16)
  } else {
    SAMPLE(float)
  }
#undef SAMPLE
  gumbel_finalize<<<(rows + 3) / 4, 128, 0, stream>>>(pval.data_ptr<float>(), pidx.data_ptr<int>(),
                                                          int(split), rows, token.data_ptr<int64_t>());
}

// Rank of the sampled token and the top-1 logprob under the rowinv row (rM, rL), written in vLLM's
// gather_logprobs layout: out_idx / out_lp [rows, 1 + k] (k in {0, 1}), out_rank [rows] int64.
void rank_cuda(const at::Tensor &x, const at::Tensor &rM, const at::Tensor &rL, const at::Tensor &lp_tok,
               const at::Tensor &token, const c10::optional<at::Tensor> &temperature, int64_t split,
               int64_t k, at::Tensor pcnt, at::Tensor ptop, at::Tensor ptopi, at::Tensor out_idx,
               at::Tensor out_lp, at::Tensor out_rank) {
  check_rows(x, int(split));
  TORCH_CHECK(k == 0 || k == 1, "fused_sampler: k must be 0 or 1");
  const at::cuda::CUDAGuard guard(x.device());
  int rows = x.size(0), V = x.size(1);
  if (rows == 0)
    return;
  const float *tp = nullptr;
  if (temperature.has_value()) {
    TORCH_CHECK(temperature->dtype() == at::kFloat && temperature->is_cuda() && temperature->numel() == 1,
                "fused_sampler: score temperature must be a CUDA fp32 scalar tensor");
    tp = temperature->data_ptr<float>();
  }
  int L = V / int(split);
  auto stream = at::cuda::getCurrentCUDAStream();
  bool xbf = x.dtype() == at::kBFloat16;
#define RANK(T, TEMP)                                                                                  \
  rank_partials<T, TEMP><<<dim3(rows, int(split)), THREADS, 0, stream>>>(                             \
      reinterpret_cast<const T *>(x.data_ptr()), x.stride(0), L, rM.data_ptr<float>(), rL.data_ptr<float>(), \
      lp_tok.data_ptr<float>(), tp, pcnt.data_ptr<int>(), ptop.data_ptr<float>(), ptopi.data_ptr<int>());
  if (xbf && tp)
    RANK(bf16, true)
  else if (xbf)
    RANK(bf16, false)
  else if (tp)
    RANK(float, true)
  else
    RANK(float, false)
#undef RANK
  rank_finalize<<<(rows + 3) / 4, 128, 0, stream>>>(
      pcnt.data_ptr<int>(), ptop.data_ptr<float>(), ptopi.data_ptr<int>(), int(split), rows, int(k),
      token.data_ptr<int64_t>(), lp_tok.data_ptr<float>(), out_idx.data_ptr<int32_t>(), out_lp.data_ptr<float>(),
      out_rank.data_ptr<int64_t>());
}
