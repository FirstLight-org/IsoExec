// SPDX-License-Identifier: Apache-2.0
// Torch bindings for the SM100 GDN core. Shape and dtype checking lives in ops.py; this file marshals
// pointers and strides and launches. One kernel serves prefill, chunked prefill, the trainer forward and
// decode (csrc/core.cu).
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include "params.cuh"

namespace gdn_cpr_sm100 {

void core_launch(const Params &p, bool save, bool decode, int plan, int items, cudaStream_t stream);
void prep_launch(const Params &p, char *hand, char *handk, char *carry, int *table, int *base, int maxch,
                 int64_t hstride, int plan, cudaStream_t stream);
void decode_rec_launch(const Params &p, int split, int items, cudaStream_t stream);
void decode_rows(const at::Tensor &slots, const std::vector<at::Tensor> &maps, std::vector<at::Tensor> rows32s,
                 std::vector<at::Tensor> lane32s, const std::vector<at::Tensor> &clocks,
                 const std::vector<at::Tensor> &last_useds);
int64_t decode_rec_smem_bytes();
int64_t decode_rec_blocks_per_sm();
int64_t prep_hand_bytes();
int64_t prep_hand_short_bytes();
int64_t prep_handk_bytes();
int64_t prep_carry_bytes();
int64_t prep_a_smem_bytes();
int64_t prep_b_smem_bytes();
int64_t prep_table_ints();
int64_t prep_a_blocks_per_sm();
void chunk_base_launch(const int *cu, int nseq, int *base, cudaStream_t stream);
int64_t core_smem_bytes();

namespace {

const uint16_t *bf16_ptr(const at::Tensor &t) {
  return reinterpret_cast<const uint16_t *>(t.data_ptr<at::BFloat16>());
}
uint16_t *bf16_ptr_mut(at::Tensor &t) { return reinterpret_cast<uint16_t *>(t.data_ptr<at::BFloat16>()); }
uint16_t *opt_bf16(const c10::optional<at::Tensor> &t) {
  if (!t.has_value() || t->numel() == 0) return nullptr;
  at::Tensor x = *t;
  return reinterpret_cast<uint16_t *>(x.data_ptr<at::BFloat16>());
}
float *opt_f32(const c10::optional<at::Tensor> &t) {
  if (!t.has_value() || t->numel() == 0) return nullptr;
  at::Tensor x = *t;
  return x.data_ptr<float>();
}

void fill_common(Params &p, const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
                 const at::Tensor &a, const at::Tensor &b, const at::Tensor &A_log, const at::Tensor &dt_bias,
                 at::Tensor &out, double scale) {
  p.q = bf16_ptr(q);
  p.k = bf16_ptr(k);
  p.v = bf16_ptr(v);
  p.a = bf16_ptr(a);
  p.b = bf16_ptr(b);
  p.qs0 = q.stride(0);
  p.qs1 = q.stride(1);
  p.ks0 = k.stride(0);
  p.ks1 = k.stride(1);
  p.vs0 = v.stride(0);
  p.vs1 = v.stride(1);
  p.as0 = a.stride(0);
  p.as1 = a.stride(1);
  p.bs0 = b.stride(0);
  p.bs1 = b.stride(1);
  p.A_log = A_log.data_ptr();
  p.dt_bias = dt_bias.data_ptr();
  p.param_bf16 = A_log.scalar_type() == at::kBFloat16 ? 1 : 0;
  p.out = bf16_ptr_mut(out);
  p.os0 = out.stride(0);
  p.os1 = out.stride(1);
  p.hk = int(q.size(1));
  p.hv = int(v.size(1));
  p.T = q.size(0);
  p.scale = float(scale);
  p.order = nullptr;
  p.chunk_base = nullptr;
  p.cu = nullptr;
  p.nseq = 0;
  p.st.base = nullptr;
  p.st.hv = p.hv;
  p.slot_in = nullptr;
  p.slot_out = nullptr;
  p.sv = SaveParams{nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
}

void fill_state(Params &p, const c10::optional<at::Tensor> &state, const c10::optional<at::Tensor> &slot_in,
                const c10::optional<at::Tensor> &slot_out) {
  if (!state.has_value()) return;
  at::Tensor s = *state;
  p.st.base = reinterpret_cast<char *>(s.data_ptr());
  p.st.hv = int(s.size(1));
  p.slot_in = (*slot_in).data_ptr<int>();
  p.slot_out = (*slot_out).data_ptr<int>();
}

}  // namespace

void prefill(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor a, at::Tensor b, at::Tensor A_log,
             at::Tensor dt_bias, at::Tensor cu_seqlens, c10::optional<at::Tensor> state,
             c10::optional<at::Tensor> slot_in, c10::optional<at::Tensor> slot_out, at::Tensor out, double scale,
             std::vector<at::Tensor> saves, c10::optional<at::Tensor> chunk_base,
             int64_t plan) {
  Params p;
  fill_common(p, q, k, v, a, b, A_log, dt_bias, out, scale);
  p.cu = cu_seqlens.data_ptr<int>();
  p.nseq = int(cu_seqlens.numel()) - 1;
  fill_state(p, state, slot_in, slot_out);
  bool save = false;
  if (!saves.empty()) {
    TORCH_CHECK(saves.size() == 9, "gdn_cpr_sm100: saves must hold 9 tensors");
    save = true;
    p.sv.qn = bf16_ptr_mut(saves[0]);
    p.sv.kn = bf16_ptr_mut(saves[1]);
    p.sv.q_rstd = saves[2].data_ptr<float>();
    p.sv.k_rstd = saves[3].data_ptr<float>();
    p.sv.v = bf16_ptr_mut(saves[4]);
    p.sv.A = bf16_ptr_mut(saves[5]);
    p.sv.w = saves[6].numel() ? bf16_ptr_mut(saves[6]) : nullptr;
    p.sv.h = saves[7].numel() ? bf16_ptr_mut(saves[7]) : nullptr;
    p.sv.v_new = saves[8].numel() ? bf16_ptr_mut(saves[8]) : nullptr;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  if (p.sv.h) {
    TORCH_CHECK(chunk_base.has_value(), "gdn_cpr_sm100: the h save needs a chunk_base buffer");
    at::Tensor cb = *chunk_base;
    p.chunk_base = cb.data_ptr<int>();
    chunk_base_launch(p.cu, p.nseq, cb.data_ptr<int>(), stream);
  }
  core_launch(p, save, false, int(plan), p.nseq * p.hv, stream);
}

// The PREPARED TWO-LAUNCH prefill (csrc/prepared.cu). The workspaces are allocated by the caller because
// their size depends on a chunk count only the device knows exactly; `ops.py` passes the host's upper
// bound `maxch` and the kernels fill the table.
void prefill_prepared(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor a, at::Tensor b, at::Tensor A_log,
                      at::Tensor dt_bias, at::Tensor cu_seqlens, c10::optional<at::Tensor> state,
                      c10::optional<at::Tensor> slot_in, c10::optional<at::Tensor> slot_out, at::Tensor out,
                      double scale, at::Tensor hand, c10::optional<at::Tensor> handk, at::Tensor carry,
                      at::Tensor table, int64_t plan) {
  Params p;
  fill_common(p, q, k, v, a, b, A_log, dt_bias, out, scale);
  p.cu = cu_seqlens.data_ptr<int>();
  p.nseq = int(cu_seqlens.numel()) - 1;
  fill_state(p, state, slot_in, slot_out);
  const int maxch = int(hand.size(0));
  const int64_t rec = prep_table_ints();
  TORCH_CHECK(table.numel() >= rec * maxch + p.nseq + 1, "gdn_cpr_sm100: table too small");
  int *tb = table.data_ptr<int>();
  TORCH_CHECK((reinterpret_cast<uintptr_t>(tb) & 15u) == 0,
              "gdn_cpr_sm100: the chunk table must be 16-byte aligned (it is read as two int4)");
  const int64_t hstride = hand.size(2);
  char *hkp = nullptr;
  if (handk.has_value() && handk->numel()) {
    at::Tensor t = *handk;
    TORCH_CHECK(t.size(0) == maxch && t.size(1) == p.hk && t.size(2) == prep_handk_bytes(),
                "gdn_cpr_sm100: the (chunk, key head) hand-off has the wrong shape");
    TORCH_CHECK(hstride == prep_hand_short_bytes(),
                "gdn_cpr_sm100: with the pre-kernel the (chunk, value head) hand-off is the SHORT stride");
    hkp = reinterpret_cast<char *>(t.data_ptr());
  } else {
    TORCH_CHECK(hstride == prep_hand_bytes(),
                "gdn_cpr_sm100: without the pre-kernel the hand-off is the full stride");
  }
  prep_launch(p, reinterpret_cast<char *>(hand.data_ptr()), hkp, reinterpret_cast<char *>(carry.data_ptr()),
              tb, tb + rec * maxch, maxch, hstride, int(plan), at::cuda::getCurrentCUDAStream());
}

// THE DECODE STEP IS THE RECURRENT ONE UNLESS A PLAN ASKS FOR THE CHUNK FORM. `plan < 0` takes the
// deployment rule (the recurrent kernel, `csrc/decode.cu`, at the value-row split `ops.choose_decode_split`
// picked); a plan with bit 12 set forces it explicitly; any other plan value runs the CHUNK-FORM decode of
// `csrc/core.cu`, which is the previous shipped kernel and this pass's before column. Both evaluate the
// same recipe and `tests/test_identity_gpu.py` gates them bit-identical, which is the whole licence for
// choosing between them on speed.
void decode(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor a, at::Tensor b, at::Tensor A_log,
            at::Tensor dt_bias, at::Tensor state, at::Tensor slot_in, at::Tensor slot_out, at::Tensor out,
            double scale, int64_t plan, int64_t split) {
  Params p;
  fill_common(p, q, k, v, a, b, A_log, dt_bias, out, scale);
  p.nseq = int(q.size(0));
  fill_state(p, state, slot_in, slot_out);
  auto stream = at::cuda::getCurrentCUDAStream();
  const bool rec = plan < 0 || (plan & 0x1000) != 0;
  if (rec) decode_rec_launch(p, int(split), p.nseq * p.hv, stream);
  else core_launch(p, false, true, int(plan), p.nseq * p.hv, stream);
}

std::vector<std::pair<std::string, int64_t>> record_layout() {
  return {{"S", Rec::S},   {"sh", Rec::SH}, {"kb", Rec::KB},   {"vb", Rec::VB}, {"db", Rec::DB},
          {"xb", Rec::XB}, {"pb", Rec::PB}, {"gb", Rec::GB},   {"bb", Rec::BB}, {"pos", Rec::POS},
          {"bytes", Rec::BYTES}, {"npos", NPOS}, {"smem", core_smem_bytes()},
          {"prep_hand", prep_hand_bytes()}, {"prep_carry", prep_carry_bytes()},
          {"prep_hand_short", prep_hand_short_bytes()}, {"prep_handk", prep_handk_bytes()},
          {"prep_a_smem", prep_a_smem_bytes()}, {"prep_b_smem", prep_b_smem_bytes()},
          {"prep_table_ints", prep_table_ints()}, {"prep_a_blocks_per_sm", prep_a_blocks_per_sm()},
          {"decode_rec_smem", decode_rec_smem_bytes()},
          {"decode_rec_blocks_per_sm", decode_rec_blocks_per_sm()}};
}

}  // namespace gdn_cpr_sm100

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prefill", &gdn_cpr_sm100::prefill);
  m.def("prefill_prepared", &gdn_cpr_sm100::prefill_prepared);
  m.def("decode", &gdn_cpr_sm100::decode);
  m.def("record_layout", &gdn_cpr_sm100::record_layout);
  m.def("decode_rows", &gdn_cpr_sm100::decode_rows);
}
