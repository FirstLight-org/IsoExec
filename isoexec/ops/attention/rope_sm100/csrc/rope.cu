// SPDX-License-Identifier: Apache-2.0
#include "common.cuh"
#include "vecio.cuh"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <optional>
#include <vector>

using namespace rope_sm100;

namespace {

// Blackwell execution-plan knobs (free; README.md "Execution plan" holds the B200 sweep that set
// every literal below). Hopper's rope_sm90 kernel targets 128 threads/CTA and 2 pairs/thread
// (rpb = min(hq+hk, 128/units, 1024/units)); neither literal is inherited here
// (docs/sm100-bringup.md: "do not copy the sm90 schedule"). Both are runtime-settable from python
// (`set_rope_plan`) so a sweep never rebuilds the binary under test.

struct RopeChoice {
  int64_t threads;
  int pairs;
};

RopeChoice rope_default(int64_t elems, int64_t half, int64_t heads) {
  int64_t threads;
  int pairs;
  if (elems < (int64_t(1) << 18)) {
    threads = 256;
    pairs = 2;
  } else if (elems < (int64_t(1) << 21)) {
    threads = 1024;
    pairs = 8;
  } else {
    threads = 256;
    pairs = 8;
    if (half % 16 == 0 && half / 16 >= 4) {
      int64_t u = 1; // threads per row at E = 16, i.e. next_pow2(half / 16)
      while (u < half / 16)
        u <<= 1;
      const int64_t rows = heads < threads / u ? heads : threads / u;
      if ((rows * u) % 32 == 0)
        pairs = 16;
    }
  }
  while (pairs > 2 && half / pairs < 4)
    pairs >>= 1;
  return RopeChoice{threads, pairs};
}

__global__ void table_kernel(const double *inv, float *out, int64_t first,
                             int half) {
  int64_t row = blockIdx.x;
  double p = double(first + row);
  for (int j = threadIdx.x; j < half; j += blockDim.x) {
    double a = p * inv[j];
    out[row * 2 * half + j] = float(cos(a));
    out[row * 2 * half + half + j] = float(sin(a));
  }
}

struct Operand {
  int64_t s0, s1, o0, o1; // input and output token/head strides
  int heads;
};

// E = rotation pairs per thread, WIDE = whether one access may span the whole E-element block.
// E = 2 with WIDE = false is rope_sm90's ownership and access width verbatim; E = 4/8/16 (and E = 2 with
// WIDE) give one thread 4/8 adjacent pairs fetched by 8/16-byte neox accesses and 16/32-byte
// interleaved ones. WIDE is false whenever the operand's base offset or row stride cannot carry a
// wide access, which is a fall back to more instructions for the same bits, never to other bits.
// The rotation itself, the table lookup per column and every rounding point are E-independent -- rope
// has no cross-element reduction, so pair ownership and access width are pure plan.
template <class T, bool IN_I, bool OUT_I, bool MROPE, int E, bool WIDE>
__global__ void rope_kernel(const T *q, const T *k, T *qo, T *ko,
                            const __grid_constant__ Operand oq,
                            const __grid_constant__ Operand ok,
                            const __grid_constant__ Positions P, int dim,
                            int rot, int logu, int rpb, bool transpose,
                            bool copy_pass, bool tabvec) {
  const int t = blockIdx.x;
  const int h = blockIdx.y * rpb + (threadIdx.x >> logu);
  const int v = threadIdx.x & ((1 << logu) - 1);
  if (h >= oq.heads + ok.heads)
    return;
  const bool isq = h < oq.heads;
  const int hh = isq ? h : h - oq.heads;
  const T *src = isq ? q + t * oq.s0 + hh * oq.s1 : k + t * ok.s0 + hh * ok.s1;
  T *dst = isq ? qo + t * oq.o0 + hh * oq.o1 : ko + t * ok.o0 + hh * ok.o1;
  const int half = rot / 2;
  const int j = E * v;
  if (j < half) {
    float a[E], b[E];
    if constexpr (IN_I) {
      float w[2 * E];
      rdv<T, 2 * E, WIDE>(src + 2 * j, w);
#pragma unroll
      for (int i = 0; i < E; ++i) {
        a[i] = w[2 * i];
        b[i] = w[2 * i + 1];
      }
    } else {
      rdv<T, E, WIDE>(src + j, a);
      rdv<T, E, WIDE>(src + j + half, b);
    }
    // This thread's E table columns are CONTIGUOUS, so without MRoPE the cos and sin windows come in
    // as two vector loads instead of 2E scalar ones (`tabvec`; the host clears it if the table's base
    // is not 16-byte aligned, and MRoPE clears it because there each column may read a different table
    // ROW). Same fp32 values, a quarter of the load instructions.
    float cs[E], sn[E];
    if (!MROPE && tabvec) {
      const float *r = table_row<false>(P, t, 0, rot);
      rdv<float, E>(r + j, cs);
      rdv<float, E>(r + half + j, sn);
    } else {
#pragma unroll
      for (int i = 0; i < E; ++i) {
        const float *r = table_row<MROPE>(P, t, j + i, rot);
        cs[i] = r[j + i];
        sn[i] = r[half + j + i];
      }
    }
    float y[E], z[E];
#pragma unroll
    for (int i = 0; i < E; ++i) {
      float sv = sn[i];
      if (transpose)
        sv = -sv;
      rotate(a[i], b[i], cs[i], sv, y[i], z[i]);
    }
    if constexpr (OUT_I) {
      float w[2 * E];
#pragma unroll
      for (int i = 0; i < E; ++i) {
        w[2 * i] = y[i];
        w[2 * i + 1] = z[i];
      }
      wrv<T, 2 * E, WIDE>(dst + 2 * j, w);
    } else {
      wrv<T, E, WIDE>(dst + j, y);
      wrv<T, E, WIDE>(dst + j + half, z);
    }
  }
  // Pass-through lanes: two elements per thread, exactly as rope_sm90 -- unchanged by E, since the
  // partial-rotary geometries that reach it are not the ones E is tuned for.
  if (copy_pass && rot + 2 * v < dim)
    cp2(dst + rot + 2 * v, src + rot + 2 * v);
}

template <class T> T *ptr(const at::Tensor &t) {
  return t.defined() && t.numel() ? static_cast<T *>(t.data_ptr()) : nullptr;
}

} // namespace

void table_cuda(const at::Tensor &inv, at::Tensor out, int64_t first) {
  TORCH_CHECK(inv.is_cuda() && inv.scalar_type() == at::kDouble &&
              inv.is_contiguous());
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kFloat &&
              out.is_contiguous() && out.dim() == 2 &&
              out.size(1) == 2 * inv.numel());
  const c10::cuda::CUDAGuard guard(out.device());
  if (!out.size(0) || !inv.numel())
    return;
  int half = inv.numel();
  table_kernel<<<out.size(0), std::min(half, 256), 0,
                 at::cuda::getCurrentCUDAStream()>>>(
      inv.data_ptr<double>(), out.data_ptr<float>(), first, half);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

namespace {
int64_t next_pow2(int64_t n) {
  int64_t p = 1;
  while (p < n)
    p <<= 1;
  return p;
}

// A wide-access plan needs the operand rows aligned for it and their token/head strides whole
// multiples of the access granularity. Otherwise E halves until it does (E = 2 always works, because
// `pair_operand` already guarantees the sm90 4-byte contract) -- the output bits never move.
bool vec_ok(const at::Tensor &x, int64_t gr) {
  if (!x.defined() || !x.numel())
    return true;
  const int64_t nb = gr * x.element_size();
  const int64_t chunk = (nb % 16 == 0) ? 16 : ((nb % 8 == 0) ? 8 : 4);
  return x.stride(-1) == 1 && x.stride(0) % gr == 0 && x.stride(1) % gr == 0 &&
         (uintptr_t(x.data_ptr()) % uint64_t(chunk)) == 0;
}
} // namespace


// The kernels load/store two 16-bit elements at once from even element
// offsets of 4-byte aligned rows.
bool pair_aligned(const at::Tensor &t) {
  return t.stride(-1) == 1 && (uintptr_t(t.data_ptr()) & 3) == 0 &&
         t.stride(0) % 2 == 0 && t.stride(1) % 2 == 0;
}

// A fresh allocation is aligned; contiguous() alone would keep an offset view.
at::Tensor pair_operand(const at::Tensor &t) {
  return pair_aligned(t) ? t : t.clone(at::MemoryFormat::Contiguous);
}

bool positions_ok(const at::Tensor &p, const at::Tensor &x) {
  return p.is_cuda() && p.device() == x.device() &&
         p.scalar_type() == at::kLong &&
         ((p.dim() == 1 && p.size(0) == x.size(0)) ||
          (p.dim() == 2 && p.size(0) == 3 && p.size(1) == x.size(0)));
}

bool table_ok(const at::Tensor &t, const at::Tensor &x) {
  return t.is_cuda() && t.device() == x.device() &&
         t.scalar_type() == at::kFloat && t.dim() == 2 && t.is_contiguous();
}

// Returns {qo, ko} (ko undefined without k), or {} when the kernel does not
// schedule the request (the caller then evaluates the torch recipe).
std::vector<at::Tensor> rope_apply(const at::Tensor &q_in,
                                   const std::optional<at::Tensor> &k_in,
                                   const at::Tensor &positions,
                                   const at::Tensor &table, int64_t m1,
                                   int64_t m2, bool in_interleaved,
                                   bool out_interleaved, bool transpose,
                                   bool inplace) {
  const bool has_k = k_in.has_value();
  const int64_t rot = table.dim() == 2 ? table.size(1) : 0;
  auto ok = [&](const at::Tensor &x) {
    return x.is_cuda() && x.dim() == 3 && x.size(0) == q_in.size(0) &&
           x.size(2) == q_in.size(2) && x.scalar_type() == q_in.scalar_type() &&
           x.device() == q_in.device() &&
           (x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kHalf ||
            x.scalar_type() == at::kFloat);
  };
  if (!ok(q_in) || (has_k && !ok(*k_in)) || !table_ok(table, q_in) ||
      !positions_ok(positions, q_in) || rot <= 0 || rot % 4 ||
      rot > q_in.size(2) || q_in.size(0) >= (int64_t(1) << 31) ||
      !mrope_mask_extent_ok(rot, m1, m2, positions.dim()))
    return {};
  if (inplace && (in_interleaved != out_interleaved || !pair_aligned(q_in) ||
                  (has_k && !pair_aligned(*k_in))))
    return {};
  const c10::cuda::CUDAGuard guard(q_in.device());
  at::Tensor q = pair_operand(q_in);
  at::Tensor k = has_k ? pair_operand(*k_in) : at::Tensor();
  // Lanes >= rot are bit copies. Out-of-place partial rotary either copies them
  // inside the kernel (threads own two pass-through elements) or, for large
  // operands, clones the operand in bulk and rotates the clone in place. Plan
  // only: measured crossover in README.md.
  const bool partial = rot < q.size(2) && !inplace;
  const int64_t elems = q.numel() + (has_k ? k.numel() : 0);
  const bool bulk = partial && elems >= (int64_t(4) << 20);
  auto fresh = [&](const at::Tensor &x) {
    return bulk ? x.clone(at::MemoryFormat::Contiguous) : at::empty(x.sizes(), x.options());
  };
  at::Tensor qo = inplace ? q : fresh(q);
  at::Tensor ko = !has_k ? at::Tensor() : inplace ? k : fresh(k);
  const int64_t T = q.size(0);
  if (!T)
    return {qo, ko};
  const int hq = q.size(1), hk = has_k ? k.size(1) : 0, dim = q.size(2);
  const bool copy = partial && !bulk;
  // Pairs per thread (access width) and threads per CTA: both runtime-selectable plan knobs.
  const int64_t half64 = rot / 2;
  const RopeChoice want = rope_default(elems, half64, int64_t(hq + hk));
  const int64_t threads_target = want.threads;
  int E = want.pairs;
  if (E != 2 && E != 4 && E != 8 && E != 16)
    E = 2;
  bool wide = false;
  for (;;) {
    const int64_t gr = (in_interleaved || out_interleaved) ? int64_t(2 * E) : int64_t(E);
    wide = half64 % E == 0 && vec_ok(q, gr) && (!has_k || vec_ok(k, gr)) && vec_ok(qo, gr) &&
           (!has_k || vec_ok(ko, gr));
    if (wide || E == 2)
      break; // E == 2 without a wide access is rope_sm90's plan exactly, and always legal
    E >>= 1;
  }
  const int64_t units =
      next_pow2(std::max<int64_t>({half64 / E, copy ? (dim - rot) / 2 : 0, 1}));
  const int logu = 63 - __builtin_clzll(uint64_t(units));
  const int rpb = int(std::max<int64_t>(
      1, std::min<int64_t>({int64_t(hq + hk), (threads_target + units - 1) / units, 1024 / units})));
  if (bulk) {
    q = qo;
    k = ko;
  }
  Operand oq{q.stride(0), q.stride(1), qo.stride(0), qo.stride(1), hq};
  Operand okk{has_k ? k.stride(0) : 0, has_k ? k.stride(1) : 0,
              has_k ? ko.stride(0) : 0, has_k ? ko.stride(1) : 0, hk};
  const bool one_d = positions.dim() == 1;
  Positions P{positions.data_ptr<int64_t>(), table.data_ptr<float>(),
              one_d ? 0 : positions.stride(0), positions.stride(-1),
              table.size(0), uint64_t(m1), uint64_t(m2), 0};
  const bool mrope = (m1 | m2) != 0 && !one_d;
  // The cos/sin vector load needs the table's base 16-byte aligned; a row offset is p*rot floats with
  // rot % 4 == 0, hence a multiple of 16 bytes on its own.
  const bool tabvec = (uintptr_t(table.data_ptr()) % 16) == 0;
  dim3 grid(T, (hq + hk + rpb - 1) / rpb);
  const int threads = rpb << logu;
  auto stream = at::cuda::getCurrentCUDAStream();
#define LAUNCH_EW(TT, I, O, MR, EE, WD)                                        \
  rope_kernel<TT, I, O, MR, EE, WD><<<grid, threads, 0, stream>>>(             \
      ptr<const TT>(q), ptr<const TT>(k), ptr<TT>(qo), ptr<TT>(ko), oq, okk,   \
      P, dim, int(rot), logu, rpb, transpose, copy, tabvec)
#define LAUNCH_E(TT, I, O, MR, EE)                                             \
  do {                                                                         \
    if (wide)                                                                  \
      LAUNCH_EW(TT, I, O, MR, EE, true);                                       \
    else                                                                       \
      LAUNCH_EW(TT, I, O, MR, EE, false);                                      \
  } while (0)
#define LAUNCH1(TT, I, O, MR)                                                  \
  do {                                                                         \
    if (E == 16)                                                               \
      LAUNCH_E(TT, I, O, MR, 16);                                              \
    else if (E == 8)                                                           \
      LAUNCH_E(TT, I, O, MR, 8);                                               \
    else if (E == 4)                                                           \
      LAUNCH_E(TT, I, O, MR, 4);                                               \
    else                                                                       \
      LAUNCH_E(TT, I, O, MR, 2);                                               \
  } while (0)
#define LAUNCH(TT, I, O)                                                       \
  do {                                                                         \
    if (mrope)                                                                 \
      LAUNCH1(TT, I, O, true);                                                 \
    else                                                                       \
      LAUNCH1(TT, I, O, false);                                                \
  } while (0)
#define LAYOUTS(TT)                                                            \
  do {                                                                         \
    if (in_interleaved) {                                                      \
      if (out_interleaved)                                                     \
        LAUNCH(TT, true, true);                                                \
      else                                                                     \
        LAUNCH(TT, true, false);                                               \
    } else {                                                                   \
      if (out_interleaved)                                                     \
        LAUNCH(TT, false, true);                                               \
      else                                                                     \
        LAUNCH(TT, false, false);                                              \
    }                                                                          \
  } while (0)
  switch (q.scalar_type()) {
  case at::kBFloat16:
    LAYOUTS(__nv_bfloat16);
    break;
  case at::kHalf:
    LAYOUTS(__half);
    break;
  default:
    LAYOUTS(float);
  }
#undef LAYOUTS
#undef LAUNCH
#undef LAUNCH1
#undef LAUNCH_E
#undef LAUNCH_EW
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {qo, ko};
}
