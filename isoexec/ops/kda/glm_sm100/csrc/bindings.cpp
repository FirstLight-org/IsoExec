// SPDX-License-Identifier: Apache-2.0
// Torch bindings of the SM100 KDA kernels. Shape and dtype checks live in ops.py; this file marshals
// pointers and launches on the current stream of the inputs' device (CUDAGuard), and checks every launch.
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include "params.cuh"

namespace kda_sm100 {

void plan_launch(const Plan &p, cudaStream_t stream);
int chunk_len();
void intra_launch(const FwdParams &p, cudaStream_t stream);
void state_launch(const FwdParams &p, cudaStream_t stream);
struct DecodeParams {
  const uint16_t *q, *k, *v;
  const void *g, *beta;
  int g_bf16, beta_bf16;
  int N, H;
  const float *h0;
  float *ht;
  uint16_t *o;
  uint16_t *W, *U, *QG, *KD, *MQ;
  float *GL;
};
void decode_launch(const DecodeParams &p, cudaStream_t stream);
struct DecRecParams {
  const uint16_t *q, *k, *v;
  const void *g, *beta;
  int g_bf16, beta_bf16;
  int N, H;
  RecRef rec;
  const int *slots;
  uint16_t *o;
  uint16_t *wW, *wQG, *wU, *wMQ;
  int *wpos;
  int null_slot;
  // Fused-input form (0 / nullptr = the plain form above). qkv_ld: elements between consecutive tokens of
  // q, k, v (their head rows stay 128-contiguous), e.g. the columns of one wider projection output.
  // gate_f: when set, p.g is the raw bf16 forget projection f [N][g_ld] and the row kernel evaluates
  // the gate stage itself (layer.cu gate_fwd: lb (*) sigmoid(e(A_log_h) (*) (bf(f) (+) dt_bias_c))).
  // beta_raw: when set, p.beta is the raw bf16 beta projection b [N][beta_ld] and the row kernel applies
  // layer.cu beta_fwd's rn(sigmoid(bf(b))). Same operations in the same order: the same bits.
  int64_t qkv_ld;
  int gate_f;
  int64_t g_ld;
  const float *dt_bias, *A_log;
  float lb;
  int beta_raw;
  int64_t beta_ld;
};
void decode_rec_launch(const DecRecParams &p, cudaStream_t stream);
void record_state_launch(const RecRef &rr, const int *slots, int N, float *out, cudaStream_t stream);
void bwd_state_launch(const BwdParams &p, cudaStream_t stream);
void bwd_inter_launch(const BwdParams &p, cudaStream_t stream);
void bwd_wy_launch(const BwdParams &p, cudaStream_t stream);
void bwd_qkg_launch(const BwdParams &p, cudaStream_t stream);
void gate_fwd_launch(const void *f, int f_bf16, const float *dt_bias, const float *A_log, float lb, float *out,
                     int64_t n, int HK, cudaStream_t s);
void gate_bwd_launch(const float *dld, const void *f, int f_bf16, void *df, int df_bf16, const float *dt_bias,
                     const float *A_log, float lb, float *pdt, float *pA, float *colA, int64_t T, int HK, int H,
                     float *ddt, float *dA, cudaStream_t s);
void beta_fwd_launch(const uint16_t *b, uint16_t *out, int64_t n, cudaStream_t s);
void leaf_tree_launch(const float *x, int L, int64_t n, float *out_f32, uint16_t *out_bf16, cudaStream_t s);
void beta_bwd_launch(const void *dbeta, int dbeta_bf16, const uint16_t *y, uint16_t *db, int64_t n, cudaStream_t s);
void gnorm_fwd_launch(const uint16_t *x, const uint16_t *z, const void *w, int w_bf16, float eps, uint16_t *y,
                      float *rstd, int64_t rows, cudaStream_t s);
void gnorm_bwd_launch(const uint16_t *dy, const uint16_t *x, const uint16_t *z, const void *w, int w_bf16,
                      const float *rstd, uint16_t *dx, uint16_t *dz, float *pw, void *dw, int dw_bf16, int64_t rows,
                      cudaStream_t s);
int seg_rows();

namespace {

template <class T>
T *ptr(const at::Tensor &t) {
  return reinterpret_cast<T *>(t.data_ptr());
}
template <class T>
T *opt(const c10::optional<at::Tensor> &t) {
  return (t.has_value() && t->defined() && t->numel() > 0) ? reinterpret_cast<T *>(t->data_ptr()) : nullptr;
}

Plan make_plan(const at::Tensor &cu, const at::Tensor &plan_buf, int maxch) {
  // plan_buf: int32 [1 + 5 * maxch + 2 * nseq + 1]
  Plan p{};
  p.cu = ptr<const int>(cu);
  p.nseq = int(cu.size(0)) - 1;
  p.maxch = maxch;
  int *b = ptr<int>(plan_buf);
  p.count = b;
  p.chunk_tok = b + 1;
  p.chunk_rows = b + 1 + maxch;
  p.chunk_seq = b + 1 + 2 * maxch;
  p.seq_chunk0 = b + 1 + 3 * maxch;
  p.chunk_lo = p.seq_chunk0 + p.nseq + 1;
  p.chunk_wb = p.chunk_lo + maxch;
  p.seq_pos = p.chunk_wb + maxch;
  return p;
}
// Engine records: rec [slots][H][Rec::BYTES] uint8 (empty: functional mode). The heads of a slot are
// contiguous records; the slot stride is free (an engine may place each slot in its own page).
int64_t rec_slot_stride(const at::Tensor &rec, int H) {
  TORCH_CHECK(rec.scalar_type() == at::kByte && rec.dim() == 3 && rec.size(1) == H && rec.size(2) == Rec::BYTES &&
                  rec.stride(2) == 1 && (rec.size(1) == 1 || rec.stride(1) == Rec::BYTES) &&
                  (rec.size(0) == 1 || rec.stride(0) >= int64_t(H) * Rec::BYTES) && rec.storage_offset() % 16 == 0 &&
                  rec.stride(0) % 16 == 0,
              "kda records: uint8 [slots, H, rec_bytes] with contiguous heads and a 16-byte aligned slot stride");
  return rec.size(0) == 1 ? int64_t(H) * Rec::BYTES : rec.stride(0);
}
void bind_records(Plan &p, const c10::optional<at::Tensor> &rec, const c10::optional<at::Tensor> &slot_in, int H) {
  p.rec.base = opt<char>(rec);
  p.rec.H = H;
  p.rec.slot_stride = rec ? rec_slot_stride(*rec, H) : 0;
  p.slot_in = p.rec.base ? opt<const int>(slot_in) : nullptr;
}

}  // namespace

void plan(const at::Tensor &cu, at::Tensor &plan_buf, int64_t maxch, const c10::optional<at::Tensor> &rec,
          const c10::optional<at::Tensor> &slot_in, int64_t H) {
  const c10::cuda::CUDAGuard guard(cu.device());
  Plan p = make_plan(cu, plan_buf, int(maxch));
  bind_records(p, rec, slot_in, int(H));
  plan_launch(p, at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
int64_t rec_bytes() { return Rec::BYTES; }
void decode_rec(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, const at::Tensor &g,
                const at::Tensor &beta, at::Tensor &rec, const at::Tensor &slots, at::Tensor &o,
                std::vector<at::Tensor> hand, int64_t null_slot) {
  const c10::cuda::CUDAGuard guard(q.device());
  DecRecParams p{};
  p.q = ptr<const uint16_t>(q);
  p.k = ptr<const uint16_t>(k);
  p.v = ptr<const uint16_t>(v);
  p.g = g.data_ptr();
  p.beta = beta.data_ptr();
  p.g_bf16 = g.scalar_type() == at::kBFloat16;
  p.beta_bf16 = beta.scalar_type() == at::kBFloat16;
  p.N = int(q.size(0));
  p.H = int(q.size(1));
  p.rec.base = ptr<char>(rec);
  p.rec.H = p.H;
  p.rec.slot_stride = rec_slot_stride(rec, p.H);
  p.slots = ptr<const int>(slots);
  p.o = ptr<uint16_t>(o);
  p.wW = ptr<uint16_t>(hand[0]);
  p.wQG = ptr<uint16_t>(hand[1]);
  p.wU = ptr<uint16_t>(hand[2]);
  p.wMQ = ptr<uint16_t>(hand[3]);
  p.wpos = ptr<int>(hand[4]);
  p.null_slot = int(null_slot);
  decode_rec_launch(p, at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
// decode_rec over fused inputs: q, k, v strided token rows of one projection output (head rows 128-contiguous,
// one common token stride), the raw bf16 forget projection f [N, H*128] (row stride free) and the raw bf16
// beta projection b [N, H] (row stride free); the gate and beta stages run inside the row kernel.
void decode_rec_fused(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, const at::Tensor &f,
                      const at::Tensor &b, const at::Tensor &dt_bias, const at::Tensor &A_log, double lb,
                      at::Tensor &rec, const at::Tensor &slots, at::Tensor &o, std::vector<at::Tensor> hand,
                      int64_t null_slot) {
  const c10::cuda::CUDAGuard guard(q.device());
  const int64_t N = q.size(0), H = q.size(1);
  for (const at::Tensor *t : {&q, &k, &v})
    TORCH_CHECK(t->scalar_type() == at::kBFloat16 && t->dim() == 3 && t->size(0) == N && t->size(1) == H &&
                    t->size(2) == 128 && t->stride(2) == 1 && t->stride(1) == 128 && t->stride(0) == q.stride(0) &&
                    q.stride(0) % 4 == 0,
                "decode_rec_fused: q, k, v bf16 [N, H, 128] with 128-contiguous heads and one token stride");
  TORCH_CHECK(f.scalar_type() == at::kBFloat16 && f.dim() == 2 && f.size(0) == N && f.size(1) == H * 128 &&
                  f.stride(1) == 1, "decode_rec_fused: f bf16 [N, H*128], unit column stride");
  TORCH_CHECK(b.scalar_type() == at::kBFloat16 && b.dim() == 2 && b.size(0) == N && b.size(1) == H && b.stride(1) == 1,
              "decode_rec_fused: b bf16 [N, H], unit column stride");
  TORCH_CHECK(dt_bias.scalar_type() == at::kFloat && dt_bias.is_contiguous() && dt_bias.numel() == H * 128 &&
                  A_log.scalar_type() == at::kFloat && A_log.is_contiguous() && A_log.numel() == H,
              "decode_rec_fused: fp32 dt_bias [H*128], A_log [H]");
  TORCH_CHECK(o.is_contiguous() && o.numel() == N * H * 128, "decode_rec_fused: contiguous output");
  DecRecParams p{};
  p.q = ptr<const uint16_t>(q);
  p.k = ptr<const uint16_t>(k);
  p.v = ptr<const uint16_t>(v);
  p.g = f.data_ptr();
  p.beta = b.data_ptr();
  p.g_bf16 = 1;
  p.beta_bf16 = 1;
  p.N = int(N);
  p.H = int(H);
  p.rec.base = ptr<char>(rec);
  p.rec.H = p.H;
  p.rec.slot_stride = rec_slot_stride(rec, p.H);
  p.slots = ptr<const int>(slots);
  p.o = ptr<uint16_t>(o);
  p.wW = ptr<uint16_t>(hand[0]);
  p.wQG = ptr<uint16_t>(hand[1]);
  p.wU = ptr<uint16_t>(hand[2]);
  p.wMQ = ptr<uint16_t>(hand[3]);
  p.wpos = ptr<int>(hand[4]);
  p.null_slot = int(null_slot);
  p.qkv_ld = q.stride(0);
  p.gate_f = 1;
  p.g_ld = f.stride(0);
  p.dt_bias = ptr<const float>(dt_bias);
  p.A_log = ptr<const float>(A_log);
  p.lb = float(lb);
  p.beta_raw = 1;
  p.beta_ld = b.stride(0);
  decode_rec_launch(p, at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void record_state(at::Tensor &rec, const at::Tensor &slots, at::Tensor &out) {
  const c10::cuda::CUDAGuard guard(rec.device());
  RecRef rr{ptr<char>(rec), int(rec.size(1)), rec_slot_stride(rec, int(rec.size(1)))};
  record_state_launch(rr, ptr<const int>(slots), int(slots.size(0)), ptr<float>(out), at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
int64_t carry_bytes() { return Carry::BYTES; }

void fwd(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, const at::Tensor &g, const at::Tensor &beta,
         const at::Tensor &cu, const at::Tensor &plan_buf, int64_t maxch, const at::Tensor &W, const at::Tensor &U,
         const at::Tensor &QG, const at::Tensor &KD, const at::Tensor &MQ, const at::Tensor &GL,
         const c10::optional<at::Tensor> &h0, const c10::optional<at::Tensor> &ht, at::Tensor &o,
         const c10::optional<at::Tensor> &save_S, const c10::optional<at::Tensor> &save_d, bool run_intra,
         bool run_state,
         const c10::optional<at::Tensor> &save_X, const c10::optional<at::Tensor> &save_KK,
         const c10::optional<at::Tensor> &save_Xu, const c10::optional<at::Tensor> &save_Kg,
         const c10::optional<at::Tensor> &rec, const c10::optional<at::Tensor> &slot_in,
         const c10::optional<at::Tensor> &slot_out, const c10::optional<at::Tensor> &carry) {
  const c10::cuda::CUDAGuard guard(q.device());
  FwdParams p{};
  p.q = ptr<const uint16_t>(q);
  p.k = ptr<const uint16_t>(k);
  p.v = ptr<const uint16_t>(v);
  p.g = g.data_ptr();
  p.beta = beta.data_ptr();
  p.g_bf16 = g.scalar_type() == at::kBFloat16;
  p.beta_bf16 = beta.scalar_type() == at::kBFloat16;
  p.H = int(q.size(1));
  p.T = q.size(0);
  p.plan = make_plan(cu, plan_buf, int(maxch));
  p.ws = Work{ptr<uint16_t>(W), ptr<uint16_t>(U), ptr<uint16_t>(QG), ptr<uint16_t>(KD), ptr<uint16_t>(MQ),
              ptr<float>(GL)};
  p.h0 = opt<const float>(h0);
  p.ht = opt<float>(ht);
  p.o = ptr<uint16_t>(o);
  p.save_S = opt<uint16_t>(save_S);
  p.save_d = opt<uint16_t>(save_d);
  p.save_X = opt<float>(save_X);
  p.save_KK = opt<float>(save_KK);
  p.save_Xu = opt<uint16_t>(save_Xu);
  p.save_Kg = opt<uint16_t>(save_Kg);
  bind_records(p.plan, rec, slot_in, p.H);
  p.slot_out = opt<const int>(slot_out);
  p.carry = opt<char>(carry);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (run_intra && maxch > 0) {
    intra_launch(p, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  if (run_state && p.plan.nseq > 0) {
    state_launch(p, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

void decode(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, const at::Tensor &g,
            const at::Tensor &beta, const c10::optional<at::Tensor> &h0, at::Tensor &ht, at::Tensor &o,
            std::vector<at::Tensor> rows) {
  const c10::cuda::CUDAGuard guard(q.device());
  DecodeParams p{};
  p.q = ptr<const uint16_t>(q);
  p.k = ptr<const uint16_t>(k);
  p.v = ptr<const uint16_t>(v);
  p.g = g.data_ptr();
  p.beta = beta.data_ptr();
  p.g_bf16 = g.scalar_type() == at::kBFloat16;
  p.beta_bf16 = beta.scalar_type() == at::kBFloat16;
  p.N = int(q.size(0));
  p.H = int(q.size(1));
  p.h0 = opt<const float>(h0);
  p.ht = ptr<float>(ht);
  p.o = ptr<uint16_t>(o);
  p.W = ptr<uint16_t>(rows[0]);
  p.U = ptr<uint16_t>(rows[1]);
  p.QG = ptr<uint16_t>(rows[2]);
  p.KD = ptr<uint16_t>(rows[3]);
  p.MQ = ptr<uint16_t>(rows[4]);
  p.GL = ptr<float>(rows[5]);
  if (p.N > 0) {
    decode_launch(p, at::cuda::getCurrentCUDAStream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

// Backward: `fw` = the forward recompute workspace and saves, `iw` = the intermediates, `out` = grads.
// fw: [W, U, QG, KD, MQ, GL, S, d, X, KK, Xu, Kg] ; iw: [dd, dSn, dW, dQG, dKD, dMQ, dGLs, dkg, dKK, dbeta_p]
// out: [dq, dk, dv, dg, dbeta, dh0]
void bwd(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, const at::Tensor &g, const at::Tensor &beta,
         const at::Tensor &cu, const at::Tensor &plan_buf, int64_t maxch, std::vector<at::Tensor> fw,
         const at::Tensor &dO, const c10::optional<at::Tensor> &dht, std::vector<at::Tensor> iw,
         std::vector<at::Tensor> out) {
  const c10::cuda::CUDAGuard guard(q.device());
  BwdParams p{};
  p.q = ptr<const uint16_t>(q);
  p.k = ptr<const uint16_t>(k);
  p.v = ptr<const uint16_t>(v);
  p.g = g.data_ptr();
  p.beta = beta.data_ptr();
  p.g_bf16 = g.scalar_type() == at::kBFloat16;
  p.beta_bf16 = beta.scalar_type() == at::kBFloat16;
  p.H = int(q.size(1));
  p.T = q.size(0);
  p.plan = make_plan(cu, plan_buf, int(maxch));
  p.ws = Work{ptr<uint16_t>(fw[0]), ptr<uint16_t>(fw[1]), ptr<uint16_t>(fw[2]), ptr<uint16_t>(fw[3]),
              ptr<uint16_t>(fw[4]), ptr<float>(fw[5])};
  p.S = ptr<const uint16_t>(fw[6]);
  p.d = ptr<const uint16_t>(fw[7]);
  p.X = ptr<const float>(fw[8]);
  p.KK = ptr<const float>(fw[9]);
  p.Xu = ptr<const uint16_t>(fw[10]);
  p.Kg = ptr<const uint16_t>(fw[11]);
  p.dO = ptr<const uint16_t>(dO);
  p.dht = opt<const float>(dht);
  p.dd = ptr<uint16_t>(iw[0]);
  p.dSn = ptr<uint16_t>(iw[1]);
  p.dW = ptr<uint16_t>(iw[2]);
  p.dQG = ptr<uint16_t>(iw[3]);
  p.dKD = ptr<uint16_t>(iw[4]);
  p.dMQ = ptr<float>(iw[5]);
  p.dGLs = ptr<float>(iw[6]);
  p.dkg = ptr<uint16_t>(iw[7]);
  p.dKK = ptr<float>(iw[8]);
  p.dbeta_p = ptr<float>(iw[9]);
  p.dq = ptr<uint16_t>(out[0]);
  p.dk = ptr<uint16_t>(out[1]);
  p.dv = ptr<uint16_t>(out[2]);
  p.dg = out[3].data_ptr();
  p.dbeta = out[4].data_ptr();
  p.dh0 = ptr<float>(out[5]);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (p.plan.nseq > 0) {
    bwd_state_launch(p, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  if (maxch > 0) {
    bwd_inter_launch(p, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    bwd_wy_launch(p, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    bwd_qkg_launch(p, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

// ---- layer stages (shapes and dtypes checked in integration.py)
bool isbf(const at::Tensor &t) { return t.scalar_type() == at::kBFloat16; }
void gate_fwd(const at::Tensor &f, const at::Tensor &dt_bias, const at::Tensor &A_log, double lb, at::Tensor &out) {
  const c10::cuda::CUDAGuard guard(f.device());
  gate_fwd_launch(f.data_ptr(), isbf(f), ptr<const float>(dt_bias), ptr<const float>(A_log), float(lb),
                  ptr<float>(out), f.numel(), int(dt_bias.numel()), at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void gate_bwd(const at::Tensor &dld, const at::Tensor &f, at::Tensor &df, const at::Tensor &dt_bias,
              const at::Tensor &A_log, double lb, at::Tensor &pdt, at::Tensor &pA, at::Tensor &colA,
              at::Tensor &ddt, at::Tensor &dA) {
  const c10::cuda::CUDAGuard guard(f.device());
  const int HK = int(dt_bias.numel());
  gate_bwd_launch(ptr<const float>(dld), f.data_ptr(), isbf(f), df.data_ptr(), isbf(df), ptr<const float>(dt_bias),
                  ptr<const float>(A_log), float(lb), ptr<float>(pdt), ptr<float>(pA), ptr<float>(colA),
                  f.numel() / HK, HK, int(A_log.numel()), ptr<float>(ddt), ptr<float>(dA),
                  at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void leaf_tree(const at::Tensor &stacked, at::Tensor &out) {
  TORCH_CHECK(stacked.scalar_type() == at::kFloat && stacked.is_contiguous() && out.is_contiguous() &&
                  (out.scalar_type() == at::kFloat || out.scalar_type() == at::kBFloat16) &&
                  stacked.numel() == stacked.size(0) * out.numel(),
              "leaf_tree: fp32 [L, ...] leaves and an fp32 / bf16 output of one leaf's shape");
  const int L = int(stacked.size(0));
  TORCH_CHECK(L >= 1 && L <= 8 && (L & (L - 1)) == 0, "leaf_tree: L must be a power of two <= 8");
  const bool f32 = out.scalar_type() == at::kFloat;
  const c10::cuda::CUDAGuard guard(stacked.device());
  leaf_tree_launch(ptr<const float>(stacked), L, out.numel(), f32 ? ptr<float>(out) : nullptr,
                   f32 ? nullptr : ptr<uint16_t>(out), at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void beta_fwd(const at::Tensor &b, at::Tensor &out) {
  const c10::cuda::CUDAGuard guard(b.device());
  beta_fwd_launch(ptr<const uint16_t>(b), ptr<uint16_t>(out), b.numel(), at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void beta_bwd(const at::Tensor &dbeta, const at::Tensor &y, at::Tensor &db) {
  const c10::cuda::CUDAGuard guard(y.device());
  beta_bwd_launch(dbeta.data_ptr(), isbf(dbeta), ptr<const uint16_t>(y), ptr<uint16_t>(db), y.numel(),
                  at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void gnorm_fwd(const at::Tensor &x, const at::Tensor &z, const at::Tensor &w, double eps, at::Tensor &y,
               at::Tensor &rstd) {
  const c10::cuda::CUDAGuard guard(x.device());
  gnorm_fwd_launch(ptr<const uint16_t>(x), ptr<const uint16_t>(z), w.data_ptr(), isbf(w), float(eps),
                   ptr<uint16_t>(y), ptr<float>(rstd), x.numel() / 128, at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void gnorm_bwd(const at::Tensor &dy, const at::Tensor &x, const at::Tensor &z, const at::Tensor &w,
               const at::Tensor &rstd, at::Tensor &dx, at::Tensor &dz, at::Tensor &pw, at::Tensor &dw) {
  const c10::cuda::CUDAGuard guard(x.device());
  gnorm_bwd_launch(ptr<const uint16_t>(dy), ptr<const uint16_t>(x), ptr<const uint16_t>(z), w.data_ptr(), isbf(w),
                   ptr<const float>(rstd), ptr<uint16_t>(dx), ptr<uint16_t>(dz), ptr<float>(pw), dw.data_ptr(),
                   isbf(dw), x.numel() / 128, at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
int64_t seg() { return seg_rows(); }

}  // namespace kda_sm100

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("plan", &kda_sm100::plan);
  m.def("rec_bytes", &kda_sm100::rec_bytes);
  m.def("chunk_len", &kda_sm100::chunk_len);
  m.def("decode_rec", &kda_sm100::decode_rec);
  m.def("decode_rec_fused", &kda_sm100::decode_rec_fused);
  m.def("record_state", &kda_sm100::record_state);
  m.def("carry_bytes", &kda_sm100::carry_bytes);
  m.def("fwd", &kda_sm100::fwd);
  m.def("decode", &kda_sm100::decode);
  m.def("bwd", &kda_sm100::bwd);
  m.def("gate_fwd", &kda_sm100::gate_fwd);
  m.def("gate_bwd", &kda_sm100::gate_bwd);
  m.def("beta_fwd", &kda_sm100::beta_fwd);
  m.def("leaf_tree", &kda_sm100::leaf_tree);
  m.def("beta_bwd", &kda_sm100::beta_bwd);
  m.def("gnorm_fwd", &kda_sm100::gnorm_fwd);
  m.def("gnorm_bwd", &kda_sm100::gnorm_bwd);
  m.def("seg", &kda_sm100::seg);
}
