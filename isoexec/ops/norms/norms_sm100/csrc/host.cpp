// SPDX-License-Identifier: Apache-2.0
#include "launch.h"
#include "plan.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <optional>
#include <torch/extension.h>

namespace norms_sm100 {
namespace {

using at::Tensor;
using torch::autograd::AutogradContext;
using torch::autograd::variable_list;

void activate(const Tensor &t) {
  CUcontext context;
  TORCH_CHECK(cuCtxGetCurrent(&context) == CUDA_SUCCESS);
  if (!context)
    C10_CUDA_CHECK(cudaSetDevice(t.get_device()));
}

bool aligned(const Tensor &t, int64_t row_stride) {
  return uintptr_t(t.data_ptr()) % 16 == 0 && (row_stride * t.element_size()) % 16 == 0;
}

Tensor rows(const Tensor &t, int64_t n, const char *what) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kBFloat16, "norms_sm100: ", what,
              " must be a CUDA bfloat16 tensor, got ", t.dtype(), " on ", t.device());
  TORCH_CHECK(t.dim() >= 1 && t.size(-1) == n, "norms_sm100: ", what, " last dim must be ", n);
  Tensor r = t.dim() == 2 ? t : t.reshape({-1, n});
  if (r.size(0) > 1 || r.dim() != 2) {
    if (r.stride(1) != 1 || !aligned(r, r.stride(0)))
      r = r.contiguous();
  } else if (r.stride(1) != 1 || uintptr_t(r.data_ptr()) % 16) {
    r = r.contiguous();
  }
  if (!aligned(r, r.size(0) > 1 ? r.stride(0) : n))
    r = r.clone(at::MemoryFormat::Contiguous);
  return r;
}

Tensor weight(const std::optional<Tensor> &w, int64_t n) {
  if (!w)
    return Tensor();
  TORCH_CHECK(w->is_cuda() && (w->scalar_type() == at::kBFloat16 || w->scalar_type() == at::kFloat),
              "norms_sm100: weight must be CUDA bfloat16 or float32");
  TORCH_CHECK(w->numel() == n, "norms_sm100: weight must have ", n, " elements");
  Tensor r = w->reshape({n});
  if (!r.is_contiguous() || uintptr_t(r.data_ptr()) % 16)
    r = r.clone(at::MemoryFormat::Contiguous);
  return r;
}

Tensor gate(const Tensor &z, int64_t n) {
  TORCH_CHECK(z.is_cuda() && z.scalar_type() == at::kBFloat16 && z.dim() == 3 && z.size(2) == n,
              "norms_sm100: gate must be CUDA bfloat16 [T, hv, N]");
  if (z.stride(2) != 1 || uintptr_t(z.data_ptr()) % 4 || (z.stride(0) * 2) % 4 || (z.stride(1) * 2) % 4)
    return z.contiguous();
  return z;
}

struct FwdOut {
  Tensor out, h, rstd;
};

FwdOut rms_forward(const Tensor &x, const std::optional<Tensor> &r, const std::optional<Tensor> &w,
                   double eps, int mode) {
  int64_t n = x.size(-1);
  const c10::cuda::CUDAGuard guard(x.device());
  Tensor x2 = rows(x, n, "x"), r2, w2 = weight(w, n);
  if (r) {
    TORCH_CHECK(r->sizes() == x.sizes(), "norms_sm100: residual shape must match x");
    r2 = rows(*r, n, "residual");
  }
  auto opts = x2.options();
  int64_t m = x2.size(0);
  FwdOut o{at::empty({m, n}, opts), r ? at::empty({m, n}, opts) : Tensor(),
           at::empty({m}, opts.dtype(at::kFloat))};
  if (m)
    launch_rms_fwd(x2, r2, w2, o.h, o.out, o.rstd, eps, mode);
  o.out = o.out.view(x.sizes());
  if (r)
    o.h = o.h.view(x.sizes());
  return o;
}

std::pair<Tensor, Tensor> rms_backward(const Tensor &dy, const Tensor &x, const Tensor &rstd,
                                       const Tensor &w, const Tensor &dres, int mode) {
  int64_t n = x.size(-1);
  const c10::cuda::CUDAGuard guard(x.device());
  activate(x);
  Tensor dyc = dy.scalar_type() == at::kBFloat16 ? dy : dy.to(at::kBFloat16);
  Tensor x2 = rows(x, n, "x"), dy2 = rows(dyc, n, "grad"), dr2;
  if (dres.defined())
    dr2 = rows(dres.scalar_type() == at::kBFloat16 ? dres : dres.to(at::kBFloat16), n, "dres");
  int64_t m = x2.size(0);
  Tensor dx = at::empty({m, n}, x2.options());
  Tensor dw;
  if (!m)
    return {dx.view(x.sizes()), w.defined() ? at::zeros_like(w) : Tensor()};
  Tensor dwp = at::empty({row_segments(m), n}, x2.options().dtype(at::kFloat));
  launch_rms_bwd(dy2, x2, rstd.reshape({m}), w, dr2, dx, dwp, mode);
  if (w.defined()) {
    Tensor folded = at::empty({n}, dwp.options());
    launch_dw_fold(dwp, folded);
    dw = w.scalar_type() == at::kFloat ? folded : folded.to(w.scalar_type());
  }
  return {dx.view(x.sizes()), dw};
}

std::pair<Tensor, Tensor> gated_forward(const Tensor &x, const Tensor &z,
                                        const std::optional<Tensor> &w, double eps, int mode) {
  TORCH_CHECK(x.dim() == 3 && z.sizes() == x.sizes(),
              "norms_sm100: gated norm takes x and z of shape [T, hv, N]");
  int64_t n = x.size(2);
  const c10::cuda::CUDAGuard guard(x.device());
  Tensor x2 = rows(x, n, "x"), z3 = gate(z, n), w2 = weight(w, n);
  int64_t m = x2.size(0);
  Tensor out = at::empty({m, n}, x2.options());
  Tensor rstd = at::empty({m}, x2.options().dtype(at::kFloat));
  if (m)
    launch_gated_fwd(x2, z3, w2, out, rstd, eps, mode);
  return {out.view({x.size(0), x.size(1) * n}), rstd};
}

struct GatedGrads {
  Tensor dx, dz, dw;
};

GatedGrads gated_backward(const Tensor &dy, const Tensor &x, const Tensor &z, const Tensor &rstd,
                          const Tensor &w, int mode) {
  int64_t n = x.size(2);
  const c10::cuda::CUDAGuard guard(x.device());
  activate(x);
  Tensor x2 = rows(x, n, "x"), z3 = gate(z, n);
  Tensor dyc = dy.scalar_type() == at::kBFloat16 ? dy : dy.to(at::kBFloat16);
  Tensor dy2 = rows(dyc.reshape({-1, n}), n, "grad");
  int64_t m = x2.size(0);
  GatedGrads g{at::empty({m, n}, x2.options()), at::empty({m, n}, x2.options()), Tensor()};
  if (m) {
    Tensor dwp = w.defined() ? at::empty({row_segments(m), n}, x2.options().dtype(at::kFloat)) : Tensor();
    launch_gated_bwd(dy2, x2, z3, rstd.reshape({m}), w, g.dx, g.dz, dwp, mode);
    if (w.defined()) {
      Tensor folded = at::empty({n}, dwp.options());
      launch_dw_fold(dwp, folded);
      g.dw = w.scalar_type() == at::kFloat ? folded : folded.to(w.scalar_type());
    }
  } else if (w.defined()) {
    g.dw = at::zeros_like(w);
  }
  g.dx = g.dx.view(x.sizes());
  g.dz = g.dz.view(x.sizes());
  return g;
}

struct RMSNormFn : public torch::autograd::Function<RMSNormFn> {
  static variable_list forward(AutogradContext *ctx, Tensor x, Tensor w, double eps, int64_t mode) {
    FwdOut o = rms_forward(x, std::nullopt, w, eps, int(mode));
    ctx->save_for_backward({x, w, o.rstd});
    ctx->saved_data["mode"] = mode;
    return {o.out};
  }
  static variable_list backward(AutogradContext *ctx, variable_list grads) {
    auto s = ctx->get_saved_variables();
    if (!grads[0].defined())
      return {Tensor(), Tensor(), Tensor(), Tensor()};
    auto [dx, dw] =
        rms_backward(grads[0], s[0], s[2], s[1], Tensor(), int(ctx->saved_data["mode"].toInt()));
    return {dx, dw, Tensor(), Tensor()};
  }
};

struct AddRMSNormFn : public torch::autograd::Function<AddRMSNormFn> {
  static variable_list forward(AutogradContext *ctx, Tensor x, Tensor r, Tensor w, double eps,
                               int64_t mode) {
    FwdOut o = rms_forward(x, r, w, eps, int(mode));
    ctx->save_for_backward({o.h, w, o.rstd});
    ctx->saved_data["mode"] = mode;
    return {o.out, o.h};
  }
  static variable_list backward(AutogradContext *ctx, variable_list grads) {
    auto s = ctx->get_saved_variables();
    Tensor dy = grads[0], dh = grads[1], dtot, dw;
    if (dy.defined()) {
      std::tie(dtot, dw) =
          rms_backward(dy, s[0], s[2], s[1], dh, int(ctx->saved_data["mode"].toInt()));
    } else {
      dtot = dh;
    }
    return {dtot, dtot, dw, Tensor(), Tensor()};
  }
};

struct GatedNormFn : public torch::autograd::Function<GatedNormFn> {
  static variable_list forward(AutogradContext *ctx, Tensor x, Tensor z, Tensor w, double eps,
                               int64_t mode) {
    auto [out, rstd] = gated_forward(x, z, w.defined() ? std::optional<Tensor>(w) : std::nullopt, eps, int(mode));
    ctx->save_for_backward({x, z, w, rstd});
    ctx->saved_data["mode"] = mode;
    return {out};
  }
  static variable_list backward(AutogradContext *ctx, variable_list grads) {
    auto s = ctx->get_saved_variables();
    if (!grads[0].defined())
      return {Tensor(), Tensor(), Tensor(), Tensor(), Tensor()};
    GatedGrads g = gated_backward(grads[0], s[0], s[1], s[3], s[2], int(ctx->saved_data["mode"].toInt()));
    return {g.dx, g.dz, g.dw, Tensor(), Tensor()};
  }
};

bool wants_grad(std::initializer_list<const std::optional<Tensor> *> ts) {
  if (!torch::autograd::GradMode::is_enabled())
    return false;
  for (auto *t : ts)
    if (*t && (*t)->requires_grad())
      return true;
  return false;
}

Tensor opt(const std::optional<Tensor> &t) { return t ? *t : Tensor(); }

} // namespace


Tensor rms_norm(const Tensor &x, const std::optional<Tensor> &w, double eps, bool zero_centered) {
  TORCH_CHECK(w, "norms_sm100.rms_norm requires a weight");
  int64_t mode = zero_centered ? 2 : 1;
  std::optional<Tensor> xo = x;
  if (wants_grad({&xo, &w}))
    return RMSNormFn::apply(x, *w, eps, mode)[0];
  return rms_forward(x, std::nullopt, w, eps, int(mode)).out;
}

Tensor rms_norm_round_scale(const Tensor &x, const Tensor &w, double eps) {
  TORCH_CHECK(w.scalar_type() == at::kBFloat16, "norms_sm100.rms_norm_round_scale: bf16 weight");
  TORCH_CHECK(!(at::GradMode::is_enabled() && (x.requires_grad() || w.requires_grad())),
              "norms_sm100.rms_norm_round_scale is forward-only");
  return rms_forward(x, std::nullopt, std::optional<Tensor>(w), eps, 3).out;
}

std::tuple<Tensor, Tensor> add_rms_norm(const Tensor &x, const Tensor &residual,
                                        const std::optional<Tensor> &w, double eps,
                                        bool zero_centered) {
  TORCH_CHECK(w, "norms_sm100.add_rms_norm requires a weight");
  int64_t mode = zero_centered ? 2 : 1;
  std::optional<Tensor> xo = x, ro = residual;
  if (wants_grad({&xo, &ro, &w})) {
    auto o = AddRMSNormFn::apply(x, residual, *w, eps, mode);
    return {o[0], o[1]};
  }
  FwdOut o = rms_forward(x, residual, w, eps, int(mode));
  return {o.out, o.h};
}

Tensor gated_rms_norm(const Tensor &x, const Tensor &z, const std::optional<Tensor> &w, double eps,
                      bool zero_centered) {
  int64_t mode = !w ? 0 : zero_centered ? 2 : 1;
  std::optional<Tensor> xo = x, zo = z;
  if (wants_grad({&xo, &zo, &w}))
    return GatedNormFn::apply(x, z, opt(w), eps, mode)[0];
  return gated_forward(x, z, w, eps, int(mode)).first;
}


std::tuple<Tensor, Tensor, Tensor> raw_rms_fwd(const Tensor &x, const std::optional<Tensor> &r,
                                               const std::optional<Tensor> &w, double eps,
                                               bool zero_centered) {
  FwdOut o = rms_forward(x, r, w, eps, zero_centered ? 2 : 1);
  return {o.out, o.h, o.rstd};
}

std::tuple<Tensor, Tensor> raw_rms_bwd(const Tensor &dy, const Tensor &x, const Tensor &rstd,
                                       const std::optional<Tensor> &w,
                                       const std::optional<Tensor> &dres, bool zero_centered) {
  auto [dx, dw] = rms_backward(dy, x, rstd, weight(w, x.size(-1)), opt(dres), zero_centered ? 2 : 1);
  return {dx, dw};
}

std::tuple<Tensor, Tensor> raw_gated_fwd(const Tensor &x, const Tensor &z,
                                         const std::optional<Tensor> &w, double eps,
                                         bool zero_centered) {
  auto [out, rstd] = gated_forward(x, z, w, eps, !w ? 0 : zero_centered ? 2 : 1);
  return {out, rstd};
}

std::tuple<Tensor, Tensor, Tensor> raw_gated_bwd(const Tensor &dy, const Tensor &x, const Tensor &z,
                                                 const Tensor &rstd, const std::optional<Tensor> &w,
                                                 bool zero_centered) {
  GatedGrads g = gated_backward(dy, x, z, rstd, weight(w, x.size(-1)), !w ? 0 : zero_centered ? 2 : 1);
  return {g.dx, g.dz, g.dw};
}

// (hidden-axis segment length, row segment length) the kernels were compiled with.
std::tuple<int, int> schedule() { return {hidden_segment(), row_segment()}; }

// Execution geometry, none of it numerical: (forward threads per row, forward loads per thread, backward threads per
// row, backward loads per thread, forward CTA threads, forward register plan, backward load plan) at `rows` rows.
std::tuple<int, int, int, int, int, int, int> plan(int64_t n, int64_t rows) {
  Lanes fl = fwd_lanes(n), bl = bwd_lanes(n);
  FwdPlan f = fwd_plan(fl, rows);
  return {fl.L, fl.K, bl.L, bl.K, f.threads, f.var, bwd_var(bl, rows)};
}

} // namespace norms_sm100

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  using namespace norms_sm100;
  m.def("rms_norm", &rms_norm, py::arg("x"), py::arg("weight"), py::arg("eps"), py::arg("zero_centered"));
  m.def("rms_norm_round_scale", &rms_norm_round_scale, py::arg("x"), py::arg("weight"), py::arg("eps"));
  m.def("add_rms_norm", &add_rms_norm, py::arg("x"), py::arg("residual"), py::arg("weight"), py::arg("eps"),
        py::arg("zero_centered"));
  m.def("gated_rms_norm", &gated_rms_norm, py::arg("x"), py::arg("z"), py::arg("weight"), py::arg("eps"),
        py::arg("zero_centered"));
  m.def("raw_rms_fwd", &raw_rms_fwd);
  m.def("raw_rms_bwd", &raw_rms_bwd);
  m.def("raw_gated_fwd", &raw_gated_fwd);
  m.def("raw_gated_bwd", &raw_gated_bwd);
  m.def("schedule", &schedule);
  m.def("fwd_supported", &fwd_supported);
  m.def("plan", &plan, py::arg("n"), py::arg("rows"));
}
