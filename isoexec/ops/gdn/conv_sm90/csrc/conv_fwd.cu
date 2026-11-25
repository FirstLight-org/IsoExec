// SPDX-License-Identifier: Apache-2.0
// Forward kernels: packed varlen (trainer / prefill, optional carried state) and decode update.
#include "common.cuh"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace conv_sm90 {
namespace {

constexpr int CPT = 8;  // channels per thread (one packed 16-byte bf16 load)

struct StateMap {
  const void *base;  // conv state [lines, D, W-1] (any strides)
  int64_t s0, s1, s2;
  const int *idx;       // per-sequence cache line, or nullptr
  const bool *has;      // per-sequence has-initial-state, or nullptr
  int null_block;       // lines < 0 or == null_block are padding: never read or written
};

template <class T, class TW, int TT, int BT, bool BIAS, int SILU, bool STATE>
__global__ __launch_bounds__(BT) void varlen_fwd(
    T *__restrict__ out, const T *__restrict__ x, const TW *__restrict__ w,
    const TW *__restrict__ bias, const int *__restrict__ cu, int nseq, int total, int D, int64_t xrow,
    const __grid_constant__ StateMap sm) {
  // xrow: x's row stride in 8-channel packs (D / CPT when packed; x may be a column window of a wider row,
  // e.g. the q|k|v columns of the in_proj GEMM output, read in place). out is packed.
  int a = blockIdx.x * TT;
  int e = min(a + TT, total);
  // Metadata for tokens [a, min(e + 2, total)): hist = valid history taps (0..3), flags bit0 = first token
  // of its sequence, bit1 = last token, seqof = sequence id. With a carried state, the tile holding a
  // sequence's first token owns every state access of that sequence: it also computes the (at most two)
  // following tokens that spill into the next tile, and writes the final window after its token loop. No
  // other tile reads or writes that cache line, so one launch is race-free.
  __shared__ uint8_t hist[TT + 2];
  __shared__ uint8_t flags[TT + 2];
  __shared__ int32_t seqof[TT + 2];
  __shared__ int32_t range[2];
  __shared__ int32_t starts[TT + 1];
  if (threadIdx.x == 0) {
    int lo = 0, hi = nseq - 1;  // largest s with cu[s] <= a
    while (lo < hi) {
      int mid = (lo + hi + 1) >> 1;
      if (cu[mid] <= a)
        lo = mid;
      else
        hi = mid - 1;
    }
    int s = lo, start = cu[s], end = cu[s + 1];
    int stop = min(e + 2, total), nst = 0;
    int a_eff = a, e_eff = e;
    for (int t = a; t < stop; ++t) {
      while (t >= end) {
        ++s;
        start = end;
        end = cu[s + 1];
      }
      int k = t - start;
      hist[t - a] = k < 3 ? k : 3;
      flags[t - a] = (k == 0 ? 1 : 0) | (t == end - 1 ? 2 : 0);
      seqof[t - a] = s;
      if constexpr (STATE) {
        if (t < e && k == 0)
          starts[nst++] = s;
        if (t == a_eff && t < e && k < 3 && start < a)
          ++a_eff;  // owned by the previous tile (spill of a sequence that started there)
        if (t == e_eff && t >= e && k < 3 && start < e)
          ++e_eff;  // spill of a sequence that started here
      }
    }
    if constexpr (STATE) {
      range[0] = a_eff;
      range[1] = e_eff;
      starts[TT] = nst;
    }
  }
  __syncthreads();
  int c0 = (blockIdx.y * BT + threadIdx.x) * CPT;
  if (c0 >= D)
    return;
  int t0 = a, t1 = e;
  if constexpr (STATE) {
    t0 = range[0];
    t1 = range[1];
  }

  float w0[CPT], w1[CPT], w2[CPT], w3[CPT], b[CPT];
  load_weights<TW, CPT>(w, c0, w0, w1, w2, w3);
  if constexpr (BIAS) {
#pragma unroll
    for (int j = 0; j < CPT; ++j)
      b[j] = widen(bias[c0 + j]);
  }
  float h1[CPT], h2[CPT], h3[CPT], cur[CPT];
  const auto *xp = reinterpret_cast<const pack8<T> *>(x);
  auto *op = reinterpret_cast<pack8<T> *>(out);
  int64_t row = D / CPT;
  for (int t = t0; t < t1; ++t) {
    int k = hist[t - a], f = flags[t - a];
    if (t == t0 || (f & 1)) {
      // (Re)initialize the window: in-sequence inputs, else the carried state, else +0.
      int line = -1;
      if constexpr (STATE) {
        int s = seqof[t - a];
        line = sm.idx[s];
        if (line < 0 || line == sm.null_block || !sm.has[s])
          line = -1;
      }
      // m = 1..3: the input m tokens back. In-sequence (m <= k) from x, else state slot 3-(m-k), else +0.
      auto fetch = [&](int m, float (&v)[CPT]) {
        if (m <= k) {
          pack8<T> p = xp[int64_t(t - m) * xrow + c0 / CPT];
#pragma unroll
          for (int j = 0; j < CPT; ++j)
            v[j] = widen(p.v[j]);
          return;
        }
        if constexpr (STATE) {
          if (line >= 0) {
            const T *base = static_cast<const T *>(sm.base) + int64_t(line) * sm.s0;
            int64_t kk = 3 - (m - k);
#pragma unroll
            for (int j = 0; j < CPT; ++j)
              v[j] = widen(base[int64_t(c0 + j) * sm.s1 + kk * sm.s2]);
            return;
          }
        }
#pragma unroll
        for (int j = 0; j < CPT; ++j)
          v[j] = 0.f;
      };
      fetch(1, h1);
      fetch(2, h2);
      fetch(3, h3);
    } else {
#pragma unroll
      for (int j = 0; j < CPT; ++j) {
        h3[j] = h2[j];
        h2[j] = h1[j];
        h1[j] = cur[j];
      }
    }
    pack8<T> p = xp[int64_t(t) * xrow + c0 / CPT];
    pack8<T> o;
#pragma unroll
    for (int j = 0; j < CPT; ++j) {
      cur[j] = widen(p.v[j]);
      float acc = preact(h3[j], h2[j], h1[j], cur[j], w0[j], w1[j], w2[j], w3[j]);
      if constexpr (BIAS)
        acc = __fadd_rn(acc, b[j]);
      if constexpr (SILU == 1)
        acc = silu(acc);
      o.v[j] = narrow<T>(acc);
    }
    op[int64_t(t) * row + c0 / CPT] = o;
  }
  if constexpr (STATE) {
    // Final windows of the sequences that start in this tile: new[k] = x[end-3+k] when that token is in
    // the sequence, else the old window shifted left by len (old[k+len]), or +0 without an initial state.
    // Ascending k never reads a slot it already wrote.
    int nst = starts[TT];
    for (int i = 0; i < nst; ++i) {
      int sq = starts[i];
      int line = sm.idx[sq];
      if (line < 0 || line == sm.null_block)
        continue;
      int start = cu[sq], end = cu[sq + 1], len = end - start;
      bool has = sm.has[sq];
      T *base = static_cast<T *>(const_cast<void *>(sm.base)) + int64_t(line) * sm.s0;
#pragma unroll
      for (int k = 0; k < 3; ++k) {
        int t = end - 3 + k;
        T v[CPT];
        if (t >= start) {
          pack8<T> p = xp[int64_t(t) * xrow + c0 / CPT];
#pragma unroll
          for (int j = 0; j < CPT; ++j)
            v[j] = p.v[j];
        } else {
#pragma unroll
          for (int j = 0; j < CPT; ++j)
            v[j] = has ? base[int64_t(c0 + j) * sm.s1 + (k + len) * sm.s2] : narrow<T>(0.f);
        }
#pragma unroll
        for (int j = 0; j < CPT; ++j)
          base[int64_t(c0 + j) * sm.s1 + k * sm.s2] = v[j];
      }
    }
  }
}


// Decode: one token per lane, window slides in place in the carried state.
template <class T, class TW, int BT, bool BIAS, int SILU, int LAYOUT>
__global__ __launch_bounds__(BT) void update_fwd(
    T *__restrict__ out, const T *__restrict__ x, const TW *__restrict__ w,
    const TW *__restrict__ bias, int D, int64_t sx, int64_t so, const __grid_constant__ StateMap sm) {
  // sx / so: row strides (elements) of x and out; both multiples of CPT with 16-byte aligned rows. x may
  // be a column window of a wider row (the in_proj GEMM output); out == x is the in-place form.
  int lane = blockIdx.x;
  int c0 = (blockIdx.y * BT + threadIdx.x) * CPT;
  if (c0 >= D)
    return;
  int line = sm.idx[lane];
  if (line < 0 || line == sm.null_block) {
    // Padding lane: its output row keeps the input, as the in-place form leaves it (byte copy).
    if (out != x)
      reinterpret_cast<pack8<T> *>(out + int64_t(lane) * so)[c0 / CPT] =
          reinterpret_cast<const pack8<T> *>(x + int64_t(lane) * sx)[c0 / CPT];
    return;
  }
  float w0[CPT], w1[CPT], w2[CPT], w3[CPT], b[CPT];
  load_weights<TW, CPT>(w, c0, w0, w1, w2, w3);
  if constexpr (BIAS) {
#pragma unroll
    for (int j = 0; j < CPT; ++j)
      b[j] = widen(bias[c0 + j]);
  }
  T *base = static_cast<T *>(const_cast<void *>(sm.base)) + int64_t(line) * sm.s0;
  pack8<T> p = reinterpret_cast<const pack8<T> *>(x + int64_t(lane) * sx)[c0 / CPT];
  T s0[CPT], s1[CPT], s2[CPT];
  if constexpr (LAYOUT == 1) {  // [line, D, 3] contiguous: c0's taps start the 24-element run
    auto *pb = reinterpret_cast<const pack8<T> *>(base) + c0 * 3 / CPT;
    pack8<T> a0 = pb[0], a1 = pb[1], a2 = pb[2];
#pragma unroll
    for (int j = 0; j < CPT; ++j) {
      int q = j * 3;
      s0[j] = (q < 8 ? a0.v[q] : q < 16 ? a1.v[q - 8] : a2.v[q - 16]);
      s1[j] = (q + 1 < 8 ? a0.v[q + 1] : q + 1 < 16 ? a1.v[q - 7] : a2.v[q - 15]);
      s2[j] = (q + 2 < 8 ? a0.v[q + 2] : q + 2 < 16 ? a1.v[q - 6] : a2.v[q - 14]);
    }
  } else if constexpr (LAYOUT == 2) {  // [line, 3, D] contiguous viewed as [line, D, 3]
    auto *pb = reinterpret_cast<const pack8<T> *>(base) + c0 / CPT;
    pack8<T> a0 = pb[0], a1 = pb[D / CPT], a2 = pb[2 * (D / CPT)];
#pragma unroll
    for (int j = 0; j < CPT; ++j) {
      s0[j] = a0.v[j];
      s1[j] = a1.v[j];
      s2[j] = a2.v[j];
    }
  } else {
#pragma unroll
    for (int j = 0; j < CPT; ++j) {
      int64_t o = int64_t(c0 + j) * sm.s1;
      s0[j] = base[o];
      s1[j] = base[o + sm.s2];
      s2[j] = base[o + 2 * sm.s2];
    }
  }
  pack8<T> o;
#pragma unroll
  for (int j = 0; j < CPT; ++j) {
    float acc = preact(widen(s0[j]), widen(s1[j]), widen(s2[j]), widen(p.v[j]), w0[j], w1[j], w2[j],
                       w3[j]);
    if constexpr (BIAS)
      acc = __fadd_rn(acc, b[j]);
    if constexpr (SILU == 1)
      acc = silu(acc);
    o.v[j] = narrow<T>(acc);
  }
  if constexpr (LAYOUT == 1) {
    auto *pb = reinterpret_cast<pack8<T> *>(base) + c0 * 3 / CPT;
    pack8<T> a0, a1, a2;
#pragma unroll
    for (int j = 0; j < CPT; ++j) {
      int q = j * 3;
      T v[3] = {s1[j], s2[j], p.v[j]};
#pragma unroll
      for (int k = 0; k < 3; ++k) {
        int r = q + k;
        if (r < 8)
          a0.v[r] = v[k];
        else if (r < 16)
          a1.v[r - 8] = v[k];
        else
          a2.v[r - 16] = v[k];
      }
    }
    pb[0] = a0;
    pb[1] = a1;
    pb[2] = a2;
  } else if constexpr (LAYOUT == 2) {
    auto *pb = reinterpret_cast<pack8<T> *>(base) + c0 / CPT;
    pack8<T> a0, a1;
#pragma unroll
    for (int j = 0; j < CPT; ++j) {
      a0.v[j] = s1[j];
      a1.v[j] = s2[j];
    }
    pb[0] = a0;
    pb[D / CPT] = a1;
    pb[2 * (D / CPT)] = p;
  } else {
#pragma unroll
    for (int j = 0; j < CPT; ++j) {
      int64_t q = int64_t(c0 + j) * sm.s1;
      base[q] = s1[j];
      base[q + sm.s2] = s2[j];
      base[q + 2 * sm.s2] = p.v[j];
    }
  }
  reinterpret_cast<pack8<T> *>(out + int64_t(lane) * so)[c0 / CPT] = o;
}

}  // namespace
}  // namespace conv_sm90

using namespace conv_sm90;

namespace {
template <class T, class TW, int TT, int BT, bool BIAS, int SILU, bool STATE>
void launch_varlen(at::Tensor &out, const at::Tensor &x, const at::Tensor &w,
                   const c10::optional<at::Tensor> &bias, const at::Tensor &cu, int nseq,
                   int total, int D, const StateMap &sm) {
  int groups = D / CPT;
  const int64_t xrow = x.stride(0) / CPT;
  dim3 grid((total + TT - 1) / TT, (groups + BT - 1) / BT);
  varlen_fwd<T, TW, TT, BT, BIAS, SILU, STATE><<<grid, BT, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<T *>(out.data_ptr()), reinterpret_cast<const T *>(x.data_ptr()),
      reinterpret_cast<const TW *>(w.data_ptr()),
      BIAS ? reinterpret_cast<const TW *>(bias->data_ptr()) : nullptr, cu.data_ptr<int>(), nseq,
      total, D, xrow, sm);
}

template <class T, class TW, int TT, int BT>
void dispatch_flags(bool b, int s, bool st, at::Tensor &out, const at::Tensor &x,
                    const at::Tensor &w, const c10::optional<at::Tensor> &bias,
                    const at::Tensor &cu, int nseq, int total, int D, const StateMap &sm) {
#define L3(B, S, ST) if (b == B && s == S && st == ST) return launch_varlen<T, TW, TT, BT, B, S, ST>(out, x, w, bias, cu, nseq, total, D, sm);
  {
    L3(true, 0, true) L3(true, 0, false) L3(false, 0, true) L3(false, 0, false)
    L3(true, 1, true) L3(true, 1, false) L3(false, 1, true) L3(false, 1, false)
  }
#undef L3
}
}  // namespace

void varlen_fwd_cuda(at::Tensor out, const at::Tensor &x, const at::Tensor &w,
                     const c10::optional<at::Tensor> &bias, const at::Tensor &cu,
                     const c10::optional<at::Tensor> &state, const c10::optional<at::Tensor> &idx,
                     const c10::optional<at::Tensor> &has, int64_t null_block, int64_t silu_act,
                     int64_t tile, int64_t threads) {
  const at::cuda::CUDAGuard guard(x.device());
  int total = x.size(0), D = x.size(1), nseq = cu.size(0) - 1;
  if (total == 0 || nseq <= 0)
    return;
  TORCH_CHECK(x.stride(1) == 1 && x.stride(0) % CPT == 0 && x.stride(0) >= D && (uintptr_t(x.data_ptr()) % 16) == 0 &&
                  out.is_contiguous() && (uintptr_t(out.data_ptr()) % 16) == 0,
              "conv_sm90.varlen_fwd: x must have unit column stride and 16-byte aligned rows, out must be packed");
  bool st = state.has_value();
  StateMap sm{st ? state->data_ptr() : nullptr,
              st ? state->stride(0) : 0,
              st ? state->stride(1) : 0,
              st ? state->stride(2) : 0,
              st ? idx->data_ptr<int>() : nullptr,
              st ? has->data_ptr<bool>() : nullptr,
              int(null_block)};
  bool xbf = x.dtype() == at::kBFloat16, wbf = w.dtype() == at::kBFloat16;
#define CFG(TT, BT)                                                                                  \
  if (tile == TT && threads == BT) {                                                                \
    if (xbf && wbf) return dispatch_flags<bf16, bf16, TT, BT>(bias.has_value(), int(silu_act), st, out, x, w, bias, cu, nseq, total, D, sm); \
    if (xbf && !wbf) return dispatch_flags<bf16, float, TT, BT>(bias.has_value(), silu_act, st, out, x, w, bias, cu, nseq, total, D, sm); \
    if (!xbf && wbf) return dispatch_flags<float, bf16, TT, BT>(bias.has_value(), silu_act, st, out, x, w, bias, cu, nseq, total, D, sm); \
    return dispatch_flags<float, float, TT, BT>(bias.has_value(), silu_act, st, out, x, w, bias, cu, nseq, total, D, sm); \
  }
  CFG(8, 64)
  CFG(8, 128)
  CFG(16, 128)
  CFG(32, 256)



#undef CFG
  TORCH_CHECK(false, "unsupported tile/threads");
}

namespace {
template <class T, class TW, int BT, bool BIAS, int SILU, int LAYOUT>
void launch_update(at::Tensor &out, const at::Tensor &x, const at::Tensor &w,
                   const c10::optional<at::Tensor> &bias, int N, int D, const StateMap &sm) {
  dim3 grid(N, ((D / CPT) + BT - 1) / BT);
  update_fwd<T, TW, BT, BIAS, SILU, LAYOUT><<<grid, BT, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<T *>(out.data_ptr()), reinterpret_cast<const T *>(x.data_ptr()),
      reinterpret_cast<const TW *>(w.data_ptr()),
      BIAS ? reinterpret_cast<const TW *>(bias->data_ptr()) : nullptr, D, x.stride(0), out.stride(0), sm);
}
}  // namespace

void update_fwd_cuda(at::Tensor out, const at::Tensor &x, const at::Tensor &w,
                     const c10::optional<at::Tensor> &bias, at::Tensor state, const at::Tensor &idx,
                     int64_t null_block, int64_t silu_act, int64_t threads) {
  const at::cuda::CUDAGuard guard(x.device());
  int N = x.size(0), D = x.size(1);
  if (N == 0)
    return;
  auto rows_ok = [&](const at::Tensor &t) {
    return t.dim() == 2 && t.size(0) == N && t.size(1) == D && (t.stride(1) == 1 || D == 1) &&
           (uintptr_t(t.data_ptr()) % 16) == 0 && ((t.stride(0) * t.element_size()) % 16 == 0 || N == 1);
  };
  TORCH_CHECK(rows_ok(x) && rows_ok(out) && out.scalar_type() == x.scalar_type(),
              "conv_sm90.update_fwd: x/out must be [N, D] with unit column stride and 16-byte aligned rows");
  StateMap sm{state.data_ptr(), state.stride(0), state.stride(1), state.stride(2),
              idx.data_ptr<int>(), nullptr, int(null_block)};
  bool xbf = x.dtype() == at::kBFloat16, wbf = w.dtype() == at::kBFloat16, b = bias.has_value();
  int act = int(silu_act);
  bool aligned = (uintptr_t(state.data_ptr()) % 16) == 0 && (state.stride(0) * state.element_size()) % 16 == 0;
  int lay = !aligned || !xbf ? 0
            : (state.stride(1) == 3 && state.stride(2) == 1) ? 1
            : (state.stride(1) == 1 && state.stride(2) == D) ? 2 : 0;
#define U2(T, TW, BT, LAY)                                                                         \
  if (b && act) return launch_update<T, TW, BT, true, 1, LAY>(out, x, w, bias, N, D, sm);         \
  if (b && !act) return launch_update<T, TW, BT, true, 0, LAY>(out, x, w, bias, N, D, sm);        \
  if (!b && act) return launch_update<T, TW, BT, false, 1, LAY>(out, x, w, bias, N, D, sm);       \
  return launch_update<T, TW, BT, false, 0, LAY>(out, x, w, bias, N, D, sm);
#define U(T, TW, BT)                                                                               \
  if (lay == 1) { U2(T, TW, BT, 1) }                                                               \
  if (lay == 2) { U2(T, TW, BT, 2) }                                                               \
  U2(T, TW, BT, 0)
#define CFG(BT)                                                                                    \
  if (threads == BT) {                                                                             \
    if (xbf && wbf) { U(bf16, bf16, BT) }                                                          \
    if (xbf && !wbf) { U(bf16, float, BT) }                                                        \
    if (!xbf && wbf) { U(float, bf16, BT) }                                                        \
    U(float, float, BT)                                                                            \
  }
  CFG(32)
  CFG(128)
#undef CFG
#undef U
#undef U2
  TORCH_CHECK(false, "unsupported threads");
}
