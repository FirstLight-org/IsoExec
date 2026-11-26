// SPDX-License-Identifier: Apache-2.0
// Torch binding of the recurrent KDA kernel. Shape and dtype policy lives in ops.py.
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>


namespace kda_rec {
struct Params {
  const uint16_t *q, *k, *v;
  int64_t q_ts, k_ts, v_ts;
  const float *g;
  int64_t g_ts;
  const uint16_t *f;
  int64_t f_ts;
  const float *dt_bias, *A_log;
  float lb;
  const void *beta;
  int64_t beta_ts;
  int beta_mode;
  const int *cu;
  const int *slot_in, *slot_out;
  int null_slot;
  float *state;
  int64_t slot_stride;
  uint16_t *o;
  int64_t o_ts;
  int N, H;
  float *ckpt;
  const int *cp_base;
};
bool rec_launch(const Params &p, bool fused, bool packed, bool ws, int lanes, int nc, int vb, cudaStream_t st);

template <class T>
static T *ptr(const at::Tensor &t) { return reinterpret_cast<T *>(t.data_ptr()); }

void rec(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, const at::Tensor &gate, const at::Tensor &beta,
         const c10::optional<at::Tensor> &dt_bias, const c10::optional<at::Tensor> &A_log, double lb, bool fused,
         int64_t beta_mode, const c10::optional<at::Tensor> &cu, const at::Tensor &slot_in, const at::Tensor &slot_out,
         int64_t null_slot, at::Tensor &state, at::Tensor &o, int64_t N, int64_t lanes, int64_t nc, int64_t vb, bool packed, bool ws, const c10::optional<at::Tensor> &ckpt,
         const c10::optional<at::Tensor> &cp_base) {
  const int64_t H = q.size(1);
  for (const at::Tensor *t : std::initializer_list<const at::Tensor *>{&q, &k, &v, &o})
    TORCH_CHECK(t->scalar_type() == at::kBFloat16 && t->dim() == 3 && t->size(1) == H && t->size(2) == 128 &&
                    t->stride(2) == 1 && t->stride(1) == 128 && t->stride(0) % 4 == 0 &&
                    reinterpret_cast<uintptr_t>(t->data_ptr()) % 8 == 0,
                "kda_rec: q, k, v, o bf16 [T, H, 128], 128-contiguous heads, token stride % 4 == 0");
  TORCH_CHECK(state.scalar_type() == at::kFloat && state.dim() == 4 && state.size(1) == H && state.size(2) == 128 &&
                  state.size(3) == 128 && state.stride(3) == 1 && state.stride(2) == 128 && state.stride(1) == 128 * 128 &&
                  state.stride(0) % 4 == 0 && reinterpret_cast<uintptr_t>(state.data_ptr()) % 16 == 0,
              "kda_rec: state fp32 [slots, H, 128 (v), 128 (k)] contiguous per slot, 16B aligned");
  TORCH_CHECK(slot_in.scalar_type() == at::kInt && slot_out.scalar_type() == at::kInt && slot_in.numel() >= N &&
                  slot_out.numel() >= N, "kda_rec: int32 slot_in / slot_out [N]");
  Params p{};
  p.q = ptr<const uint16_t>(q); p.k = ptr<const uint16_t>(k); p.v = ptr<const uint16_t>(v);
  p.q_ts = q.stride(0); p.k_ts = k.stride(0); p.v_ts = v.stride(0);
  if (fused) {
    TORCH_CHECK(gate.scalar_type() == at::kBFloat16 && gate.dim() == 2 && gate.size(1) == H * 128 && gate.stride(1) == 1 &&
                    gate.stride(0) % 4 == 0 && dt_bias && A_log, "kda_rec: fused gate needs bf16 f [T, H*128], dt_bias, A_log");
    p.f = ptr<const uint16_t>(gate); p.f_ts = gate.stride(0);
    p.dt_bias = ptr<const float>(*dt_bias); p.A_log = ptr<const float>(*A_log); p.lb = float(lb);
  } else {
    TORCH_CHECK(gate.scalar_type() == at::kFloat && gate.dim() == 3 && gate.size(1) == H && gate.size(2) == 128 &&
                    gate.stride(2) == 1 && gate.stride(1) == 128 && gate.stride(0) % 4 == 0 &&
                    reinterpret_cast<uintptr_t>(gate.data_ptr()) % 16 == 0, "kda_rec: g fp32 [T, H, 128]");
    p.g = ptr<const float>(gate); p.g_ts = gate.stride(0);
  }
  TORCH_CHECK(beta.dim() == 2 && beta.size(1) == H && beta.stride(1) == 1, "kda_rec: beta [T, H]");
  TORCH_CHECK((beta_mode == 1) == (beta.scalar_type() == at::kFloat) && (beta_mode == 1 || beta.scalar_type() == at::kBFloat16),
              "kda_rec: beta dtype vs mode");
  p.beta = beta.data_ptr(); p.beta_ts = beta.stride(0); p.beta_mode = int(beta_mode);
  p.cu = cu ? ptr<const int>(*cu) : nullptr;
  p.slot_in = ptr<const int>(slot_in); p.slot_out = ptr<const int>(slot_out); p.null_slot = int(null_slot);
  p.state = ptr<float>(state); p.slot_stride = state.stride(0);
  p.o = ptr<uint16_t>(o); p.o_ts = o.stride(0);
  p.N = int(N); p.H = int(H);
  if (ckpt) {
    TORCH_CHECK(cp_base && ckpt->scalar_type() == at::kFloat && ckpt->is_contiguous() && cp_base->scalar_type() == at::kInt,
                "kda_rec: ckpt fp32 contiguous + int32 cp_base");
    p.ckpt = ptr<float>(*ckpt); p.cp_base = ptr<const int>(*cp_base);
  }
  TORCH_CHECK(rec_launch(p, fused, packed, ws, int(lanes), int(nc), int(vb), at::cuda::getCurrentCUDAStream()),
              "kda_rec: unsupported launch shape (lanes, nc, vb) = (", lanes, ", ", nc, ", ", vb, ")");
}

struct BwdParams {
  const uint16_t *q, *k, *v;
  int64_t q_ts, k_ts, v_ts;
  const float *g;
  int64_t g_ts;
  const void *beta;
  int64_t beta_ts;
  int beta_mode;
  const int *cu;
  const float *ckpt;
  const int *cp_base;
  const uint16_t *dO;
  int64_t dO_ts;
  const float *dht;
  uint16_t *dq, *dk, *dv;
  float *dg;
  float *dbeta;
  float *dh0;
  float *scratch;
  int N, H;
};
void bwd_launch(const BwdParams &p, int grid, cudaStream_t st);

void bwd(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, const at::Tensor &g, const at::Tensor &beta,
         int64_t beta_mode, const at::Tensor &cu, const at::Tensor &ckpt, const at::Tensor &cp_base, const at::Tensor &dO,
         const c10::optional<at::Tensor> &dht, at::Tensor &dq, at::Tensor &dk, at::Tensor &dv, at::Tensor &dg,
         at::Tensor &dbeta, const c10::optional<at::Tensor> &dh0, at::Tensor &scratch, int64_t grid) {
  const int64_t H = q.size(1);
  for (const at::Tensor *t : std::initializer_list<const at::Tensor *>{&q, &k, &v, &dO, &dq, &dk, &dv})
    TORCH_CHECK(t->scalar_type() == at::kBFloat16 && t->dim() == 3 && t->size(1) == H && t->size(2) == 128 &&
                    t->stride(2) == 1 && t->stride(1) == 128, "kda_rec bwd: bf16 [T, H, 128] tensors");
  TORCH_CHECK(dq.is_contiguous() && dk.is_contiguous() && dv.is_contiguous() && dg.is_contiguous() && dbeta.is_contiguous(),
              "kda_rec bwd: contiguous gradient outputs");
  TORCH_CHECK(g.scalar_type() == at::kFloat && g.stride(2) == 1 && g.stride(1) == 128 && dg.scalar_type() == at::kFloat &&
                  dbeta.scalar_type() == at::kFloat, "kda_rec bwd: fp32 g, dg, dbeta");
  TORCH_CHECK(scratch.scalar_type() == at::kFloat && scratch.numel() >= grid * 64 * 128 * 128, "kda_rec bwd: scratch");
  BwdParams p{};
  p.q = ptr<const uint16_t>(q); p.k = ptr<const uint16_t>(k); p.v = ptr<const uint16_t>(v);
  p.q_ts = q.stride(0); p.k_ts = k.stride(0); p.v_ts = v.stride(0);
  p.g = ptr<const float>(g); p.g_ts = g.stride(0);
  p.beta = beta.data_ptr(); p.beta_ts = beta.stride(0); p.beta_mode = int(beta_mode);
  p.cu = ptr<const int>(cu); p.ckpt = ptr<const float>(ckpt); p.cp_base = ptr<const int>(cp_base);
  p.dO = ptr<const uint16_t>(dO); p.dO_ts = dO.stride(0);
  p.dht = dht ? ptr<const float>(*dht) : nullptr;
  p.dq = ptr<uint16_t>(dq); p.dk = ptr<uint16_t>(dk); p.dv = ptr<uint16_t>(dv);
  p.dg = ptr<float>(dg); p.dbeta = ptr<float>(dbeta); p.dh0 = dh0 ? ptr<float>(*dh0) : nullptr;
  p.scratch = ptr<float>(scratch);
  p.N = int(cu.numel() - 1); p.H = int(H);
  if (p.N > 0) bwd_launch(p, int(grid), at::cuda::getCurrentCUDAStream());
}

void leaf_tree4_launch(const float *x, int L, int64_t n, float *out_f32, uint16_t *out_bf16, cudaStream_t st);
// parts fp32 [L, ..., n_inner] contiguous (L a power of two <= 8) -> out (fp32 root or bf16), numel % 4 == 0
void leaf_tree(const at::Tensor &parts, at::Tensor &out) {
  const int64_t L = parts.size(0), n = out.numel();
  TORCH_CHECK(parts.scalar_type() == at::kFloat && parts.is_contiguous() && parts.numel() == L * n && out.is_contiguous() &&
                  (L & (L - 1)) == 0 && L >= 1 && L <= 8 && n % 4 == 0 &&
                  reinterpret_cast<uintptr_t>(parts.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0 &&
                  (out.scalar_type() == at::kFloat || out.scalar_type() == at::kBFloat16), "kda_rec leaf_tree: fp32 parts, 16B aligned");
  leaf_tree4_launch(ptr<const float>(parts), int(L), n, out.scalar_type() == at::kFloat ? ptr<float>(out) : nullptr,
                    out.scalar_type() == at::kBFloat16 ? ptr<uint16_t>(out) : nullptr, at::cuda::getCurrentCUDAStream());
}

void gate_launch(const uint16_t *f, int64_t f_ts, const float *dt_bias, const float *A_log, float lb, float *out,
                 int64_t n_tok, int HK, cudaStream_t st);
void beta_launch(const uint16_t *b, uint16_t *out, int64_t n, cudaStream_t st);
// out fp32 [T, H*128] = forget gate of the raw bf16 projection f [T, H*128]
void gate(const at::Tensor &f, const at::Tensor &dt_bias, const at::Tensor &A_log, double lb, at::Tensor &out) {
  TORCH_CHECK(f.scalar_type() == at::kBFloat16 && f.dim() == 2 && f.stride(1) == 1 && out.scalar_type() == at::kFloat &&
                  out.is_contiguous() && out.sizes() == f.sizes() && dt_bias.scalar_type() == at::kFloat &&
                  A_log.scalar_type() == at::kFloat, "kda_rec gate: bf16 f [T, H*128] -> fp32");
  gate_launch(ptr<const uint16_t>(f), f.stride(0), ptr<const float>(dt_bias), ptr<const float>(A_log), float(lb),
              ptr<float>(out), f.size(0), int(f.size(1)), at::cuda::getCurrentCUDAStream());
}
void beta(const at::Tensor &b, at::Tensor &out) {
  TORCH_CHECK(b.scalar_type() == at::kBFloat16 && b.is_contiguous() && out.scalar_type() == at::kBFloat16 &&
                  out.is_contiguous() && out.numel() == b.numel(), "kda_rec beta: contiguous bf16");
  beta_launch(ptr<const uint16_t>(b), ptr<uint16_t>(out), b.numel(), at::cuda::getCurrentCUDAStream());
}
}  // namespace kda_rec

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rec", &kda_rec::rec);
  m.def("gate", &kda_rec::gate);
  m.def("leaf_tree", &kda_rec::leaf_tree);
  m.def("bwd", &kda_rec::bwd);
  m.def("beta", &kda_rec::beta);
}
