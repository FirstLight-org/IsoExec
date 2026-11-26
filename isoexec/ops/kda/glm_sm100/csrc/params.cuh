// SPDX-License-Identifier: Apache-2.0
// Launch parameters of the SM100 KDA kernels (execution only; no arithmetic).
//
// Layouts (row-major, contiguous): q, k, v, g [T][H][128]; beta [T][H]; o [T][H][128] bf16;
// initial / final state [N][H][128 k][128 v] fp32. The chunk workspace is TOKEN-indexed so a call's
// footprint is proportional to its tokens, not to its chunk count: W, U, QG, KD [T][H][128] bf16,
// MQ [T][H][64] bf16 (row j of its chunk's Mq), GL [chunks][H][128] fp32 (e(G_L) of the chunk).
#pragma once
#include <stdint.h>

#include "record.cuh"

namespace kda_sm100 {

// A sequence starting at absolute position pos_in (from its input record; 0 in functional mode) has
// j0 = pos_in mod 64; its first chunk holds rows [0, j0) from the record and new rows from j0. Row i of
// chunk c is token chunk_tok[c] + i for i in [chunk_lo[c], chunk_rows[c]); the workspace row of (c, i) is
// chunk_wb[c] + i (in functional mode exactly the token index).
struct Plan {
  const int *cu;          // [N + 1] int32 token offsets
  int nseq;
  int maxch;              // upper bound of the chunk count (grid size of the chunk kernels)
  int *count;             // [1] chunk count
  int *chunk_tok;         // [maxch] token of row 0 of the chunk (may precede the sequence)
  int *chunk_rows;        // [maxch] r_hi: rows [0, r_hi) of the chunk exist after this call (1..64)
  int *chunk_seq;         // [maxch] sequence of the chunk
  int *seq_chunk0;        // [N + 1] first chunk of each sequence (exclusive prefix)
  int *chunk_lo;          // [maxch] r_lo: first row this call adds (j0 for a sequence's first chunk)
  int *chunk_wb;          // [maxch] workspace row base
  int *seq_pos;           // [N] absolute position of each sequence's first new token
  const int *slot_in;     // [N] input record slots (nullptr: functional mode, every pos_in = 0)
  RecRef rec;
};

struct Work {
  uint16_t *W, *U, *QG, *KD, *MQ;
  float *GL;
};

struct FwdParams {
  const uint16_t *q, *k, *v;
  const void *g;
  const void *beta;
  int g_bf16, beta_bf16;
  int H;
  int64_t T;
  Plan plan;
  Work ws;
  const float *h0;        // nullptr: zero initial state
  float *ht;              // nullptr: final state not requested
  uint16_t *o;
  // Engine records (plan.rec.base != nullptr): output slots (negative = discard) and phase 1's carry
  // [N][H][Carry::BYTES].
  const int *slot_out;
  char *carry;
  // Backward recompute saves (nullptr in the inference forward): S^ at each chunk start
  // [chunks][H][128 k][128 v] bf16 and the deltas d [T][H][128] bf16.
  uint16_t *save_S, *save_d;
  // Backward saves of phase 1 (nullptr in the inference forward), token-indexed rows of the chunk:
  // X [T][H][64] fp32 (+0 above the diagonal), KK [T][H][64] fp32 (+0 for i >= j), xu [T][H][64] bf16,
  // kg [T][H][128] bf16.
  float *save_X, *save_KK;
  uint16_t *save_Xu, *save_Kg;
};

}  // namespace kda_sm100

namespace kda_sm100 {

// The backward (bwd.cu). Gradients are of the mathematical function (roundings are identities); the
// forward intermediates are recomputed by the forward's own kernels (`ws`, `S`, `d`).
struct BwdParams {
  const uint16_t *q, *k, *v;
  const void *g, *beta;
  int g_bf16, beta_bf16;
  int H;
  int64_t T;
  Plan plan;
  Work ws;                  // recomputed forward workspace (W, U, QG, KD, MQ, GL)
  const uint16_t *S;        // S^ at each chunk start [chunk][H][128 k][128 v] bf16
  const uint16_t *d;        // deltas [T][H][128] bf16
  const uint16_t *dO;       // output gradient [T][H][128] bf16
  const float *dht;         // final-state gradient [N][H][128][128] fp32 (nullptr: zero)
  const float *X, *KK;       // saved X, KK [T][H][64] fp32 (phase 1 in save mode)
  const uint16_t *Xu, *Kg;   // saved xu [T][H][64], kg [T][H][128] bf16
  // intermediates (fp32 unless noted)
  uint16_t *dd;             // rn(dL/du) = rn(dL/dd) [T][H][128] bf16 (only ever an mma operand)
  uint16_t *dSn;            // rn(dS_{c+1}) of chunk c [chunk][H][128 k][128 v] bf16
  uint16_t *dW;             // rn(dL/dw) [T][H][128] bf16 (only ever an mma operand)
  uint16_t *dQG, *dKD;      // [T][H][128] bf16
  float *dMQ;               // [T][H][64]
  float *dGLs;              // sum_v S^ dS^_{c+1} [chunk][H][128]
  uint16_t *dkg;            // [T][H][128] bf16
  float *dKK;               // [T][H][64]
  float *dbeta_p;           // [T][H]
  // outputs
  uint16_t *dq, *dk, *dv;
  void *dg, *dbeta;
  float *dh0;               // [N][H][128][128]
};

}  // namespace kda_sm100
