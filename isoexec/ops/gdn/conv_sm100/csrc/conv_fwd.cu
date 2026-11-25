// SPDX-License-Identifier: Apache-2.0
#include "common.cuh"
#include "recipe_v2.cuh"
#include "vec.cuh"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace conv_sm100 {
namespace {

// Channels per thread is a template parameter (see the header note); 8 is one packed 16-byte bf16
// access, 4 and 2 trade that width for registers, hence for resident warps.

struct StateMap {
  const void *base;  // conv state [lines, D, W-1] (any strides)
  int64_t s0, s1, s2;
  const int *idx;       // per-sequence cache line, or nullptr
  const bool *has;      // per-sequence has-initial-state, or nullptr
  int null_block;       // lines < 0 or == null_block are padding: never read or written
};

/// Largest ``s`` in ``[0, nseq-1]`` with ``cu[s] <= t``: the sequence token ``t`` belongs to.
__device__ __forceinline__ int seq_of(const int *__restrict__ cu, int nseq, int t) {
  int lo = 0, hi = nseq - 1;
  while (lo < hi) {
    int mid = (lo + hi + 1) >> 1;
    if (cu[mid] <= t)
      lo = mid;
    else
      hi = mid - 1;
  }
  return lo;
}

/// The recipe for one token: tap accumulation in declared order, bias, SiLU, one RNE cast.
/// ``RECIPE`` selects the declared rounding schedule: 1 is common.cuh's (the default, and the only
/// one the engine and trainer reach), 2 is recipe_v2.cuh's explicit-``fma.rn`` contraction, which is
/// a separately declared impl (``sm100_conv_fma_v1``) and never the default. Nothing else differs.
template <class T, int CPTv, bool BIAS, int SILU, int RECIPE>
__device__ __forceinline__ void emit(packv<T, CPTv> *dst, const float (&h3)[CPTv], const float (&h2)[CPTv],
                                     const float (&h1)[CPTv], const float (&x0)[CPTv],
                                     const float (&w0)[CPTv], const float (&w1)[CPTv],
                                     const float (&w2)[CPTv], const float (&w3)[CPTv],
                                     const float (&b)[CPTv]) {
  packv<T, CPTv> o;
  if constexpr (RECIPE == 1 && CPTv % 2 == 0) {
    // Recipe v1 on channel PAIRS with packed fp32 (`mul/add.rn.f32x2`, vec.cuh): two independent
    // IEEE RN operations per instruction, so every rounding point of common.cuh's graph is the same
    // and only the issue count halves. `expf` (libdevice) and `__fdividef` stay scalar per lane.
#pragma unroll
    for (int j = 0; j < CPTv; j += 2) {
      float2 acc = mul2(make_float2(w0[j], w0[j + 1]), make_float2(h3[j], h3[j + 1]));
      acc = add2(acc, mul2(make_float2(w1[j], w1[j + 1]), make_float2(h2[j], h2[j + 1])));
      acc = add2(acc, mul2(make_float2(w2[j], w2[j + 1]), make_float2(h1[j], h1[j + 1])));
      acc = add2(acc, mul2(make_float2(w3[j], w3[j + 1]), make_float2(x0[j], x0[j + 1])));
      if constexpr (BIAS)
        acc = add2(acc, make_float2(b[j], b[j + 1]));
      if constexpr (SILU == 1) {
        const float2 den = add2(make_float2(1.f, 1.f), make_float2(expf(-acc.x), expf(-acc.y)));
        acc = make_float2(__fdividef(acc.x, den.x), __fdividef(acc.y, den.y));
      }
      o.v[j] = narrow<T>(acc.x);
      o.v[j + 1] = narrow<T>(acc.y);
    }
    *dst = o;
    return;
  }
#pragma unroll
  for (int j = 0; j < CPTv; ++j) {
    float acc = preact_v<RECIPE>(h3[j], h2[j], h1[j], x0[j], w0[j], w1[j], w2[j], w3[j]);
    if constexpr (BIAS)
      acc = __fadd_rn(acc, b[j]);
    if constexpr (SILU == 1)
      acc = silu(acc);
    o.v[j] = narrow<T>(acc);
  }
  *dst = o;
}

template <class T, class TW, int BT, int CPT, bool BIAS, int SILU, bool STATE, int RECIPE>
__global__ __launch_bounds__(BT) void varlen_fwd(
    T *__restrict__ out, const T *__restrict__ x, const TW *__restrict__ w,
    const TW *__restrict__ bias, const int *__restrict__ cu, int nseq, int total, int D, int tile,
    const __grid_constant__ StateMap sm) {
  const int a = blockIdx.x * tile;
  if (a >= total)
    return;
  const int e = min(a + tile, total);
  const int c0 = (blockIdx.y * BT + threadIdx.x) * CPT;
  if (c0 >= D)
    return;

  float w0[CPT], w1[CPT], w2[CPT], w3[CPT], b[CPT];
  load_weights<TW, CPT>(w, c0, w0, w1, w2, w3);
#pragma unroll
  for (int j = 0; j < CPT; ++j)
    b[j] = 0.f;
  if constexpr (BIAS) {
#pragma unroll
    for (int j = 0; j < CPT; ++j)
      b[j] = widen(bias[c0 + j]);
  }

  using vec = packv<T, CPT>;
  const int64_t row = D / CPT;  // vector rows per token
  const auto *xp = reinterpret_cast<const vec *>(x) + c0 / CPT;
  auto *op = reinterpret_cast<vec *>(out) + c0 / CPT;
  auto ld = [&](int t, float (&dst)[CPT]) { widen_vec<T, CPT>(xp[int64_t(t) * row], dst); };

  int s = seq_of(cu, nseq, a);
  int start = cu[s], end = cu[s + 1];

  if (a - start >= 3 && e <= end) {
    // Interior tile: one sequence, three valid history taps everywhere, no state read or write
    // (the line's owner is the tile holding this sequence's first token, which is not this one) and
    // no spill (`k >= 3` at both ends). Four-deep rotating window, no metadata, no barrier.
    float v[4][CPT];
    ld(a - 3, v[0]);
    ld(a - 2, v[1]);
    ld(a - 1, v[2]);
    int t = a;
#define STEP(i)                                                                                    \
  ld(t + (i), v[((i) + 3) & 3]);                                                                   \
  emit<T, CPT, BIAS, SILU, RECIPE>(op + int64_t(t + (i)) * row, v[(i) & 3], v[((i) + 1) & 3],       \
                           v[((i) + 2) & 3], v[((i) + 3) & 3], w0, w1, w2, w3, b);
    for (; t + 4 <= e; t += 4) {
      STEP(0)
      STEP(1)
      STEP(2)
      STEP(3)
    }
#pragma unroll
    for (int i = 0; i < 3; ++i) {
      if (t + i < e) {
        STEP(i)
      }
    }
#undef STEP
    return;
  }

  // --- general path: sequence boundaries, missing history, carried state --------------------------
  int t0 = a, t1 = e;
  if constexpr (STATE) {
    // A sequence's cache line is owned entirely by the tile holding that sequence's first token: it
    // reads the initial window and writes the final one. So that tile also computes the (at most
    // two) following tokens whose history still reaches into the state, and the next tile skips
    // them -- one launch, no races, no atomics. Identical rule to the shared-memory version; only
    // the walk moved into registers.
    int cs = s, cstart = start, cend = end;
    int ae = a;
    while (ae < e) {
      while (ae >= cend && cs + 1 < nseq) {
        ++cs;
        cstart = cend;
        cend = cu[cs + 1];
      }
      if (ae - cstart < 3 && cstart < a)
        ++ae;
      else
        break;
    }
    int ee = e, stop = min(e + 2, total);
    for (int t = e; t < stop; ++t) {
      while (t >= cend && cs + 1 < nseq) {
        ++cs;
        cstart = cend;
        cend = cu[cs + 1];
      }
      if (t == ee && t - cstart < 3 && cstart < e)
        ++ee;
      else
        break;
    }
    t0 = ae;
    t1 = ee;
  }

  if (t0 != a) {
    s = seq_of(cu, nseq, t0);
    start = cu[s];
    end = cu[s + 1];
  }
  float h1[CPT], h2[CPT], h3[CPT], cur[CPT];
  for (int t = t0; t < t1; ++t) {
    while (t >= end && s + 1 < nseq) {  // `s + 1 < nseq`: a short cu[] is caller error, not an OOB read
      ++s;
      start = end;
      end = cu[s + 1];
    }
    const int k = min(t - start, 3);
    if (t == t0 || t == start) {
      // (Re)initialize the window: in-sequence inputs, else the carried state, else +0.
      int line = -1;
      if constexpr (STATE) {
        line = sm.idx[s];
        if (line < 0 || line == sm.null_block || !sm.has[s])
          line = -1;
      }
      // m = 1..3: the input m tokens back. In-sequence (m <= k) from x, else state slot 3-(m-k),
      // else +0.
      auto fetch = [&](int m, float (&v)[CPT]) {
        if (m <= k) {
          ld(t - m, v);
          return;
        }
        if constexpr (STATE) {
          if (line >= 0) {
            const T *base = static_cast<const T *>(sm.base) + int64_t(line) * sm.s0;
            const int64_t kk = 3 - (m - k);
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
    ld(t, cur);
    emit<T, CPT, BIAS, SILU, RECIPE>(op + int64_t(t) * row, h3, h2, h1, cur, w0, w1, w2, w3, b);
  }

  if constexpr (STATE) {
    // Final windows of the sequences that start in this tile: new[k] = x[end-3+k] when that token is
    // in the sequence, else the old window shifted left by len (old[k+len]), or +0 without an
    // initial state. Ascending k never reads a slot it already wrote. Empty sequences own no token
    // and are skipped, exactly as the `hist/flags` walk skipped them.
    int sq = seq_of(cu, nseq, a);
    if (cu[sq] < a)
      ++sq;
    for (; sq < nseq && cu[sq] < e; ++sq) {
      const int st = cu[sq], en = cu[sq + 1];
      if (en == st)
        continue;
      const int line = sm.idx[sq];
      if (line < 0 || line == sm.null_block)
        continue;
      const int len = en - st;
      const bool has = sm.has[sq];
      T *base = static_cast<T *>(const_cast<void *>(sm.base)) + int64_t(line) * sm.s0;
#pragma unroll
      for (int k = 0; k < 3; ++k) {
        const int t = en - 3 + k;
        T v[CPT];
        if (t >= st) {
          vec p = xp[int64_t(t) * row];
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

// Decode: one token per lane, window slides in place in the carried state. CPTd channels per thread
// is a template parameter: a decode launch is latency bound (one token, three taps, D channels), so
// at small batch the useful knob is how many *threads* the channel axis yields, and a narrower
// per-thread width buys 2x or 4x of them. Padding lanes copy x through to out instead of returning,
// which is what lets ops.py drop its pre-copy.
template <class T, class TW, int BT, int CPTd, bool BIAS, int SILU, int LAYOUT, int RECIPE>
__global__ __launch_bounds__(BT) void update_fwd(
    T *__restrict__ out, const T *__restrict__ x, const TW *__restrict__ w,
    const TW *__restrict__ bias, int D, int64_t sx, int64_t so, const __grid_constant__ StateMap sm) {
  // sx / so: row strides (elements) of x and out, multiples of CPTd with 16-byte aligned rows. x may be
  // the q|k|v column window of the wider in_proj GEMM output row; out == x is the in-place form.
  const int lane = blockIdx.x;
  const int c0 = (blockIdx.y * BT + threadIdx.x) * CPTd;
  if (c0 >= D)
    return;
  const int64_t offx = int64_t(lane) * (sx / CPTd) + c0 / CPTd;
  const int64_t offo = int64_t(lane) * (so / CPTd) + c0 / CPTd;
  const int line = sm.idx[lane];
  if (line < 0 || line == sm.null_block) {
    // Padding lane: output keeps the input (the in-place form's behaviour), state untouched.
    if (out != x)
      reinterpret_cast<packv<T, CPTd> *>(out)[offo] = reinterpret_cast<const packv<T, CPTd> *>(x)[offx];
    return;
  }
  float w0[CPTd], w1[CPTd], w2[CPTd], w3[CPTd], b[CPTd];
  load_weights<TW, CPTd>(w, c0, w0, w1, w2, w3);
  if constexpr (BIAS) {
#pragma unroll
    for (int j = 0; j < CPTd; ++j)
      b[j] = widen(bias[c0 + j]);
  }
  T *base = static_cast<T *>(const_cast<void *>(sm.base)) + int64_t(line) * sm.s0;
  packv<T, CPTd> p = reinterpret_cast<const packv<T, CPTd> *>(x)[offx];
  T s0[CPTd], s1[CPTd], s2[CPTd];
  if constexpr (LAYOUT == 1 && CPTd == 8) {
    // [line, D, 3] contiguous: c0's 24 taps are three packed 16-byte runs (unchanged from the
    // previous plan -- at CPTd == 8 the 3-tap-per-channel run still vectorizes).
    auto *pb = reinterpret_cast<const pack8<T> *>(base) + c0 * 3 / 8;
    pack8<T> a0 = pb[0], a1 = pb[1], a2 = pb[2];
#pragma unroll
    for (int j = 0; j < CPTd; ++j) {
      int q = j * 3;
      s0[j] = (q < 8 ? a0.v[q] : q < 16 ? a1.v[q - 8] : a2.v[q - 16]);
      s1[j] = (q + 1 < 8 ? a0.v[q + 1] : q + 1 < 16 ? a1.v[q - 7] : a2.v[q - 15]);
      s2[j] = (q + 2 < 8 ? a0.v[q + 2] : q + 2 < 16 ? a1.v[q - 6] : a2.v[q - 14]);
    }
  } else if constexpr (LAYOUT == 1) {  // narrower thread: 3 scalar taps, still warp-coalesced
    const T *pb = base + int64_t(c0) * 3;
#pragma unroll
    for (int j = 0; j < CPTd; ++j) {
      s0[j] = pb[3 * j];
      s1[j] = pb[3 * j + 1];
      s2[j] = pb[3 * j + 2];
    }
  } else if constexpr (LAYOUT == 2) {  // [line, 3, D] contiguous viewed as [line, D, 3]
    const auto *pb = reinterpret_cast<const packv<T, CPTd> *>(base) + c0 / CPTd;
    packv<T, CPTd> a0 = pb[0], a1 = pb[D / CPTd], a2 = pb[2 * (D / CPTd)];
#pragma unroll
    for (int j = 0; j < CPTd; ++j) {
      s0[j] = a0.v[j];
      s1[j] = a1.v[j];
      s2[j] = a2.v[j];
    }
  } else {
#pragma unroll
    for (int j = 0; j < CPTd; ++j) {
      int64_t o = int64_t(c0 + j) * sm.s1;
      s0[j] = base[o];
      s1[j] = base[o + sm.s2];
      s2[j] = base[o + 2 * sm.s2];
    }
  }
  packv<T, CPTd> o;
#pragma unroll
  for (int j = 0; j < CPTd; ++j) {
    float acc = preact_v<RECIPE>(widen(s0[j]), widen(s1[j]), widen(s2[j]), widen(p.v[j]), w0[j],
                                 w1[j], w2[j], w3[j]);
    if constexpr (BIAS)
      acc = __fadd_rn(acc, b[j]);
    if constexpr (SILU == 1)
      acc = silu(acc);
    o.v[j] = narrow<T>(acc);
  }
  if constexpr (LAYOUT == 1 && CPTd == 8) {
    auto *pb = reinterpret_cast<pack8<T> *>(base) + c0 * 3 / 8;
    pack8<T> a0, a1, a2;
#pragma unroll
    for (int j = 0; j < CPTd; ++j) {
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
  } else if constexpr (LAYOUT == 1) {
    T *pb = base + int64_t(c0) * 3;
#pragma unroll
    for (int j = 0; j < CPTd; ++j) {
      pb[3 * j] = s1[j];
      pb[3 * j + 1] = s2[j];
      pb[3 * j + 2] = p.v[j];
    }
  } else if constexpr (LAYOUT == 2) {
    auto *pb = reinterpret_cast<packv<T, CPTd> *>(base) + c0 / CPTd;
    packv<T, CPTd> a0, a1;
#pragma unroll
    for (int j = 0; j < CPTd; ++j) {
      a0.v[j] = s1[j];
      a1.v[j] = s2[j];
    }
    pb[0] = a0;
    pb[D / CPTd] = a1;
    pb[2 * (D / CPTd)] = p;
  } else {
#pragma unroll
    for (int j = 0; j < CPTd; ++j) {
      int64_t q = int64_t(c0 + j) * sm.s1;
      base[q] = s1[j];
      base[q + sm.s2] = s2[j];
      base[q + 2 * sm.s2] = p.v[j];
    }
  }
  reinterpret_cast<packv<T, CPTd> *>(out)[offo] = o;
}

}  // namespace
}  // namespace conv_sm100

using namespace conv_sm100;

namespace {
template <class T, class TW, int BT, int CPT, bool BIAS, int SILU, bool STATE, int RECIPE>
void launch_varlen(at::Tensor &out, const at::Tensor &x, const at::Tensor &w,
                   const c10::optional<at::Tensor> &bias, const at::Tensor &cu, int nseq,
                   int total, int D, int tile, const StateMap &sm) {
  int groups = D / CPT;
  dim3 grid((total + tile - 1) / tile, (groups + BT - 1) / BT);
  varlen_fwd<T, TW, BT, CPT, BIAS, SILU, STATE, RECIPE><<<grid, BT, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<T *>(out.data_ptr()), reinterpret_cast<const T *>(x.data_ptr()),
      reinterpret_cast<const TW *>(w.data_ptr()),
      BIAS ? reinterpret_cast<const TW *>(bias->data_ptr()) : nullptr, cu.data_ptr<int>(), nseq,
      total, D, tile, sm);
}

template <class T, class TW, int BT, int CPT, int RECIPE>
void dispatch_flags(bool b, int s, bool st, at::Tensor &out, const at::Tensor &x,
                    const at::Tensor &w, const c10::optional<at::Tensor> &bias,
                    const at::Tensor &cu, int nseq, int total, int D, int tile, const StateMap &sm) {
#define L3(B, S, ST) if (b == B && s == S && st == ST) return launch_varlen<T, TW, BT, CPT, B, S, ST, RECIPE>(out, x, w, bias, cu, nseq, total, D, tile, sm);
  {
    L3(true, 0, true) L3(true, 0, false) L3(false, 0, true) L3(false, 0, false)
    L3(true, 1, true) L3(true, 1, false) L3(false, 1, true) L3(false, 1, false)
  }
#undef L3
}
}  // namespace

// Launch configs: (threads per block, channels per thread). The token tile is a runtime argument, so
// ops.py wave-aligns the grid at any token count instead of snapping to a compiled constant, and only
// the two knobs that change the generated code -- block width and vector width -- cost instantiations
// (this package's real compile-time budget; see README's rejected variants).
void varlen_fwd_cuda(at::Tensor out, const at::Tensor &x, const at::Tensor &w,
                     const c10::optional<at::Tensor> &bias, const at::Tensor &cu,
                     const c10::optional<at::Tensor> &state, const c10::optional<at::Tensor> &idx,
                     const c10::optional<at::Tensor> &has, int64_t null_block, int64_t silu_act,
                     int64_t tile, int64_t threads, int64_t cpt, int64_t recipe) {
  const at::cuda::CUDAGuard guard(x.device());
  int total = x.size(0), D = x.size(1), nseq = cu.size(0) - 1;
  if (total == 0 || nseq <= 0)
    return;
  TORCH_CHECK(tile >= 1 && tile <= (1 << 20), "conv_sm100: tile out of range");
  bool st = state.has_value();
  StateMap sm{st ? state->data_ptr() : nullptr,
              st ? state->stride(0) : 0,
              st ? state->stride(1) : 0,
              st ? state->stride(2) : 0,
              st ? idx->data_ptr<int>() : nullptr,
              st ? has->data_ptr<bool>() : nullptr,
              int(null_block)};
  bool xbf = x.dtype() == at::kBFloat16, wbf = w.dtype() == at::kBFloat16;
  int T = int(tile);
  TORCH_CHECK(recipe == 1 || recipe == 2, "conv_sm100: recipe must be 1 or 2");
#define CFGR(BT, CP, R)                                                                              \
  if (threads == BT && cpt == CP) {                                                                 \
    if (xbf && wbf) return dispatch_flags<bf16, bf16, BT, CP, R>(bias.has_value(), int(silu_act), st, out, x, w, bias, cu, nseq, total, D, T, sm); \
    if (xbf && !wbf) return dispatch_flags<bf16, float, BT, CP, R>(bias.has_value(), silu_act, st, out, x, w, bias, cu, nseq, total, D, T, sm); \
    if (!xbf && wbf) return dispatch_flags<float, bf16, BT, CP, R>(bias.has_value(), silu_act, st, out, x, w, bias, cu, nseq, total, D, T, sm); \
    return dispatch_flags<float, float, BT, CP, R>(bias.has_value(), silu_act, st, out, x, w, bias, cu, nseq, total, D, T, sm); \
  }
  // Recipe v2 (recipe_v2.cuh: explicit fma.rn contraction, a separately declared impl id, never the
  // default) is instantiated for four of the six launch configs, not all of them: every
  // instantiation of this kernel is paid for in compile time, v2's plan rule is v1's -- `fwd_plan`
  // picks 128 threads and cpt 4 at every measured shape -- and its battery needs one block width
  // plus all three vector widths to make the plan-invariance claim. A config outside that set is a
  // refusal, not a silent fallback to another schedule.
  if (recipe == 2) {
    CFGR(128, 8, 2)
    CFGR(128, 4, 2)
    CFGR(128, 2, 2)
    CFGR(64, 8, 2)
    TORCH_CHECK(false, "conv_sm100: recipe 2 is instantiated for threads/cpt in "
                       "{(128,8), (128,4), (128,2), (64,8)}");
  }
#define CFG(BT, CP) CFGR(BT, CP, 1)
  CFG(64, 8)
  CFG(128, 8)
  CFG(256, 8)
  CFG(128, 4)
  CFG(256, 4)
  CFG(128, 2)

#undef CFG
#undef CFGR
  TORCH_CHECK(false, "unsupported threads/cpt");
}

namespace {
template <class T, class TW, int BT, int CPTd, bool BIAS, int SILU, int LAYOUT, int RECIPE>
void launch_update(at::Tensor &out, const at::Tensor &x, const at::Tensor &w,
                   const c10::optional<at::Tensor> &bias, int N, int D, const StateMap &sm) {
  dim3 grid(N, ((D / CPTd) + BT - 1) / BT);
  update_fwd<T, TW, BT, CPTd, BIAS, SILU, LAYOUT, RECIPE><<<grid, BT, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<T *>(out.data_ptr()), reinterpret_cast<const T *>(x.data_ptr()),
      reinterpret_cast<const TW *>(w.data_ptr()),
      BIAS ? reinterpret_cast<const TW *>(bias->data_ptr()) : nullptr, D,
      N > 1 ? x.stride(0) : int64_t(D), N > 1 ? out.stride(0) : int64_t(D), sm);
}
}  // namespace

void update_fwd_cuda(at::Tensor out, const at::Tensor &x, const at::Tensor &w,
                     const c10::optional<at::Tensor> &bias, at::Tensor state, const at::Tensor &idx,
                     int64_t null_block, int64_t silu_act, int64_t threads, int64_t cpt,
                     int64_t recipe) {
  const at::cuda::CUDAGuard guard(x.device());
  int N = x.size(0), D = x.size(1);
  if (N == 0)
    return;
  // Rows of x / out: unit column stride, 16-byte aligned base, and a row stride that is a whole number
  // of CPT-wide packs (a single row has no meaningful stride; it is read as packed).
  auto rows_ok = [&](const at::Tensor &t) {
    return t.dim() == 2 && t.size(0) == N && t.size(1) == D && t.stride(1) == 1 &&
           (uintptr_t(t.data_ptr()) % 16) == 0 &&
           (N == 1 || ((t.stride(0) % cpt) == 0 && (t.stride(0) * t.element_size()) % 16 == 0));
  };
  TORCH_CHECK(rows_ok(x) && rows_ok(out) && out.scalar_type() == x.scalar_type(),
              "conv_sm100.update_fwd: x/out must be [N, D] with unit column stride and 16-byte aligned rows");
  StateMap sm{state.data_ptr(), state.stride(0), state.stride(1), state.stride(2),
              idx.data_ptr<int>(), nullptr, int(null_block)};
  bool xbf = x.dtype() == at::kBFloat16, wbf = w.dtype() == at::kBFloat16, b = bias.has_value();
  int act = int(silu_act);
  // Unchanged from the previous plan: the two packed state layouts are bf16-only and need a
  // 16-byte-aligned base, otherwise the generic strided path runs.
  bool aligned = (uintptr_t(state.data_ptr()) % 16) == 0 && (state.stride(0) * state.element_size()) % 16 == 0;
  int lay = !aligned || !xbf ? 0
            : (state.stride(1) == 3 && state.stride(2) == 1) ? 1
            : (state.stride(1) == 1 && state.stride(2) == D) ? 2 : 0;
  TORCH_CHECK(recipe == 1 || recipe == 2, "conv_sm100: recipe must be 1 or 2");
#define U2(T, TW, BT, CP, LAY, R)                                                                  \
  if (b && act) return launch_update<T, TW, BT, CP, true, 1, LAY, R>(out, x, w, bias, N, D, sm);   \
  if (b && !act) return launch_update<T, TW, BT, CP, true, 0, LAY, R>(out, x, w, bias, N, D, sm);  \
  if (!b && act) return launch_update<T, TW, BT, CP, false, 1, LAY, R>(out, x, w, bias, N, D, sm); \
  return launch_update<T, TW, BT, CP, false, 0, LAY, R>(out, x, w, bias, N, D, sm);
#define U(T, TW, BT, CP, R)                                                                        \
  if (lay == 1) { U2(T, TW, BT, CP, 1, R) }                                                        \
  if (lay == 2) { U2(T, TW, BT, CP, 2, R) }                                                        \
  U2(T, TW, BT, CP, 0, R)
#define CFGR(BT, CP, R)                                                                            \
  if (threads == BT && cpt == CP) {                                                                \
    if (xbf && wbf) { U(bf16, bf16, BT, CP, R) }                                                   \
    if (xbf && !wbf) { U(bf16, float, BT, CP, R) }                                                 \
    if (!xbf && wbf) { U(float, bf16, BT, CP, R) }                                                 \
    U(float, float, BT, CP, R)                                                                     \
  }
  // Recipe v2's decode exists so its battery can make the same decode == chunked prefill == trainer
  // forward claim v1's does; the three configs are the two `decode_plan` returns plus the (32, 2)
  // the split/state round-trip walks.
  if (recipe == 2) {
    CFGR(32, 8, 2)
    CFGR(128, 2, 2)
    CFGR(32, 2, 2)
    TORCH_CHECK(false, "conv_sm100: recipe 2 decode is instantiated for threads/cpt in "
                       "{(32,8), (128,2), (32,2)}");
  }
#define CFG(BT, CP) CFGR(BT, CP, 1)
  // (threads, channels per thread). The 8-wide entries keep the previous plan's 128-bit decode
  // access; the 2-wide ones exist because a one-row decode at D=1024 has only 128 channel groups to
  // spread over 148 SMs at CPT=8 and 512 at CPT=2.
  CFG(32, 8)
  CFG(128, 8)
  CFG(256, 8)
  CFG(32, 2)
  CFG(128, 2)
#undef CFG
#undef CFGR
#undef U
#undef U2
  TORCH_CHECK(false, "unsupported threads/cpt");
}
