// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stdint.h>

#include "state.cuh"

namespace gdn_cpr_sm100 {

// Trainer mode (stateless calls only): the forward's intermediates in the layouts FLA's chunked backward
// reads, so the backward recomputes none of the forward. Row-major and contiguous:
//   qn, kn [T][hk][128] bf16 (l2-normalized, qn without the attention scale), q_rstd, k_rstd [T][hk] fp32,
//   v [T][hv][128] bf16, A [T][hv][64] bf16 (rn X, row j of its chunk), w [T][hv][128] bf16,
//   h [NT][hv][128 v][128 d] bf16 (S^ at each chunk start, FLA's chunk order), v_new [T][hv][128] bf16.
struct SaveParams {
  uint16_t *qn, *kn, *v, *A, *w, *h, *v_new;
  float *q_rstd, *k_rstd;
};

struct Params {
  const uint16_t *q, *k, *v, *a, *b;
  int64_t qs0, qs1, ks0, ks1, vs0, vs1, as0, as1, bs0, bs1;
  const void *A_log, *dt_bias;
  int param_bf16;
  uint16_t *out;
  int64_t os0, os1;
  const int *cu;
  int nseq, hk, hv;
  int64_t T;
  float scale;
  StateRef st;
  const int *slot_in, *slot_out;
  const int *order;       // [N] sequence ids in dispatch order; nullptr (the identity) today.
                          // A longest-first permutation is an execution choice, not a numerical
                          // one -- it needs host lengths, and nothing supplies them yet.
  const int *chunk_base;  // [N] first FLA chunk of each sequence; only the trainer `h` save needs it
  SaveParams sv;
  __device__ __forceinline__ float A_log_of(int h) const {
    return param_bf16 ? bf(static_cast<const uint16_t *>(A_log)[h]) : static_cast<const float *>(A_log)[h];
  }
  __device__ __forceinline__ float dt_bias_of(int h) const {
    return param_bf16 ? bf(static_cast<const uint16_t *>(dt_bias)[h]) : static_cast<const float *>(dt_bias)[h];
  }
  __device__ __forceinline__ int key_head(int h) const { return h / (hv / hk); }
  __device__ __forceinline__ bool key_leader(int h) const { return h % (hv / hk) == 0; }
  __device__ __forceinline__ int slot_in_of(int n) const { return st.base ? slot_in[n] : -1; }
  __device__ __forceinline__ int slot_out_of(int n) const { return st.base ? slot_out[n] : -1; }
};

}  // namespace gdn_cpr_sm100
