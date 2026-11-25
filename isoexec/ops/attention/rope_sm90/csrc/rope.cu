// SPDX-License-Identifier: Apache-2.0
// RoPE on q and k in one launch, plus the cos/sin table builder.
//
// Numerical contract (rope_sm90 v1, see README.md):
//   table:   T[p, j] = fp32_rn(cos(double(p) * inv_freq[j])),
//            T[p, h + j] = fp32_rn(sin(...)); device libm double cos/sin,
//            inv_freq[j] supplied by the host in double.
//   rotate:  y1 = rn(x1*c) -rn rn(x2*s) ; y2 = rn(x2*c) +rn rn(x1*s), fp32,
//            then one RN narrowing to the operand dtype. x widens exactly.
//   pairs:   neox (j, j+h) or interleaved (2j, 2j+1); output layout is
//            independent of the input layout. Lanes >= rot are bit copies.
// Execution plan (free): grid (token, head group), a thread owns the two
// pairs (2v, 2v+1) and/or the two pass-through elements rot+2v, rot+2v+1.
#include "common.cuh"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <optional>
#include <vector>

using namespace rope_sm90;

namespace {

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

template <class T, bool IN_I, bool OUT_I, bool MROPE>
__global__ void rope_kernel(const T *q, const T *k, T *qo, T *ko,
                            const __grid_constant__ Operand oq,
                            const __grid_constant__ Operand ok,
                            const __grid_constant__ Positions P, int dim,
                            int rot, int logu, int rpb, bool transpose,
                            bool copy_pass) {
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
  if (2 * v < half) {
    const int j = 2 * v;
    float a0, a1, b0, b1;
    if constexpr (IN_I) {
      rd2(src + 2 * j, a0, b0);
      rd2(src + 2 * j + 2, a1, b1);
    } else {
      rd2(src + j, a0, a1);
      rd2(src + j + half, b0, b1);
    }
    const float *r0 = table_row<MROPE>(P, t, j, rot);
    const float *r1 = MROPE ? table_row<MROPE>(P, t, j + 1, rot) : r0;
    float s0 = r0[half + j], s1 = r1[half + j + 1];
    if (transpose) {
      s0 = -s0;
      s1 = -s1;
    }
    float y0, z0, y1, z1;
    rotate(a0, b0, r0[j], s0, y0, z0);
    rotate(a1, b1, r1[j + 1], s1, y1, z1);
    if constexpr (OUT_I) {
      wr2(dst + 2 * j, y0, z0);
      wr2(dst + 2 * j + 2, y1, z1);
    } else {
      wr2(dst + j, y0, y1);
      wr2(dst + j + half, z0, z1);
    }
  }
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
  const int64_t units = next_pow2(std::max<int64_t>({rot / 4, copy ? (dim - rot) / 2 : 0, 1}));
  const int logu = 63 - __builtin_clzll(uint64_t(units));
  const int rpb = int(std::max<int64_t>(
      1, std::min<int64_t>({int64_t(hq + hk), (128 + units - 1) / units, 1024 / units})));
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
  dim3 grid(T, (hq + hk + rpb - 1) / rpb);
  const int threads = rpb << logu;
  auto stream = at::cuda::getCurrentCUDAStream();
#define LAUNCH1(TT, I, O, MR)                                                  \
  rope_kernel<TT, I, O, MR><<<grid, threads, 0, stream>>>(                     \
      ptr<const TT>(q), ptr<const TT>(k), ptr<TT>(qo), ptr<TT>(ko), oq, okk,   \
      P, dim, int(rot), logu, rpb, transpose, copy)
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
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {qo, ko};
}
