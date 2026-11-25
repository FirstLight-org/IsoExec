// SPDX-License-Identifier: Apache-2.0
// Shared launch parameters and workspace layout of the prefill (execution only).
//
//   plan    prefill.cu  one CTA: each sequence's input position and chunk count, and the chunks' dense order
//   pass 1  wy.cu       per (chunk, value head), all chunks in parallel: k^, q^, gates, G, L,
//                       X = (I+L)^-1, u, w, M
//   pass 2  sweep.cu    per (sequence, head, value-row slab), sequential over chunks: d = u - S^ w, S_{c+1},
//                       and the output-slot commit; a separate launch computes outputs in parallel
//                       per (chunk, head, 64 value rows) after every sweep has completed
//
// Chunks are numbered densely: sequence n owns chunks [chunk0[n], chunk0[n + 1]) (its chunks_of(len, j0)
// chunks, in order), so a stateless pack's chunk order is FLA's. Pass 1 hands pass 2 whole chunk tiles,
// one per (chunk, head), stored as the byte image of the shared-memory tile that consumes them, so every
// hand-off is a single bulk copy:
//   W  [chunks][hv]  Wide<64>  w rows [i][d]        K  [chunks][hk]  Wide<64>  k^ rows [i][d]
//   Q  [chunks][hk]  Wide<64>  q^ rows [j][d]       M  [chunks][hv]  Tile<64>  M [j][i]
//   U  [chunks][hv]  Tile<64>  u, value-major [v][i] (128 rows)
//   GT [chunks][hv]  G, beta, e(G), e(G_63 - G) per row (fp32 [4][64])
// A tile holds all rows [0, r_hi) of its chunk (zeros elsewhere); for a continuing sequence's first
// chunk, rows [0, r_lo) of K and GT are the input slot's open-chunk rows, placed there by pass 1. Pass 2
// reads the input slot's delta rows itself: each of its CTAs reads and later rewrites only its own value
// columns. Per-sequence side buffers from pass 1 to pass 2's commit (stateful calls):
//   FX [N][hv] fp32 [64][64]  X rows [0, r_hi) of the last chunk
//   FP [N][hv] fp32 [16][64]  P rows of the last chunk's open 16-row block
// and from the sweep to the output units, per (chunk, head, value-row slab), as the images of the
// sweep's shared-memory tiles, made visible by stream-ordered launch completion:
//   SH [chunks][hv] S^ = rn(S_c) [128 v][128 d]          DT [chunks][hv] deltas [128 v][64 i]
// in slabs of the sweep's width (Wide<rows> / Tile<64> images of rows [v0, v0 + rows)). A sequence's
// first chunk without an input slot starts from the zero state: its S^ is zero and its deltas are its u
// (d = rn(bf(u) (-) CHAIN128(0, w; +0)) = u, the chain of zeros seeded with +0 being +0), so the sweep
// does not publish them and its output units read the U tile and ZEROS instead, waiting on nothing.
#pragma once
#include <stdint.h>

#include "state.cuh"

namespace gdn_cpr {

constexpr int TILE_BYTES = C * D * 2;      // a [64][128] bf16 tile
constexpr int MT_BYTES = C * C * 2;        // a [64][64] bf16 tile
constexpr int GT_BYTES = 4 * C * 4;        // G, beta, e(G), e(G_last - G)
constexpr int FX_BYTES = C * C * 4;        // X rows fp32 [64][64]
constexpr int FP_BYTES = 16 * C * 4;       // P rows fp32 [16][64]
constexpr int ZERO_BYTES = 64 * D * 2;     // a zero S^ operand of an output unit (64 value rows)
constexpr int OUTPUT_ORDER_MAX_SEQS = 64;  // larger packs use dense chunk order for output units

struct GateTile {   // the GT tile
  float G[C], beta[C], gam[C], lam[C];
};

// Trainer mode (stateless calls only): the forward's intermediates in the layouts of FLA's chunked
// backward (integration.py), so the backward recomputes none of the forward. All row-major, contiguous:
//   qn, kn [T][hk][128] bf16 (l2-normalized, qn without the 128^-1/2 scale), q_rstd, k_rstd [T][hk] fp32,
//   v [T][hv][128] bf16 (the values), A [T][hv][64] bf16 (rn X, row j of its chunk), w [T][hv][128]
//   bf16, h [NT][hv][128 v][128 d] bf16 (S^ at the start of each chunk, FLA's chunk order: sequences'
//   ceil(len / 64) chunks back to back), v_new [T][hv][128] bf16 (the deltas). Null: not saved.
struct SaveParams {
  uint16_t *qn, *kn, *v, *A, *w, *h, *v_new;
  float *q_rstd, *k_rstd;
};

struct PrefillParams {
  // inputs (bf16 bits; q/k [T, hk, 128], v [T, hv, 128], a/b [T, hv], arbitrary row strides)
  const uint16_t *q, *k, *v, *a, *b;
  int64_t qs0, qs1, ks0, ks1, vs0, vs1, as0, as1, bs0, bs1;
  const void *A_log, *dt_bias;   // [hv] fp32 or bf16
  int param_bf16;
  uint16_t *out;                 // [T, hv, 128]
  int64_t os0, os1;
  const int *cu;                 // [N + 1]
  int nseq, hk, hv;
  int full_chunks;               // stateless pack proven to contain only complete 64-token chunks
  int64_t T;
  float scale;
  // state pool (st.base == nullptr: stateless, zero initial state, nothing written back)
  StateRef st;
  const int *slot_in, *slot_out;
  // the plan (prefill.cu)
  int *chunk0;       // [N + 1] first dense chunk of each sequence; chunk0[N] = the pack's chunks
  int *pos;          // [N] input positions (0: fresh), read from the input slots before anyone writes them
  int *seq_of;       // [chunks bound] the sequence owning each dense chunk; -1 past the pack's chunks
  // workspace
  char *ws_w, *ws_u, *ws_m, *ws_k, *ws_q, *ws_gt, *ws_fx, *ws_fp;
  char *ws_sh, *ws_d;
  uint16_t *zeros;   // ZERO_BYTES of zeros (the plan writes them)
  // Output work allocation (sweep.cu), zeroed by the plan. No cross-CTA progress flags.
  int *unit_next;    // the next output unit
  int units;         // output units bound (chunks bound x hv x 2 value halves); past the plan's: none
  int groups;        // sweep roles per (head, slab): group g chains sequences g, g + groups, ... (sweep.cu)
  // the fused prefill (fused.cu)
  int *order;        // fused: [N] sequences longest first; two-pass: [chunks] output traversal order
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
  __device__ __forceinline__ int pos_in(int n) const { return pos[n]; }
  __device__ __forceinline__ int chunks(int n) const { return chunk0[n + 1] - chunk0[n]; }
  __device__ __forceinline__ int seq_of_chunk(int c) const { return seq_of[c]; }
  // chunk k of sequence n starts from the zero state, known without the sweep (see the header)
  __device__ __forceinline__ bool zero_start(int n, int k) const { return k == 0 && slot_in_of(n) < 0; }
  __device__ __forceinline__ uint16_t *w_tile(int s, int h) const {
    return reinterpret_cast<uint16_t *>(ws_w + (int64_t(s) * hv + h) * TILE_BYTES);
  }
  __device__ __forceinline__ uint16_t *u_tile(int s, int h) const {
    return reinterpret_cast<uint16_t *>(ws_u + (int64_t(s) * hv + h) * TILE_BYTES);
  }
  __device__ __forceinline__ uint16_t *m_tile(int s, int h) const {
    return reinterpret_cast<uint16_t *>(ws_m + (int64_t(s) * hv + h) * MT_BYTES);
  }
  __device__ __forceinline__ uint16_t *k_tile(int s, int hq) const {
    return reinterpret_cast<uint16_t *>(ws_k + (int64_t(s) * hk + hq) * TILE_BYTES);
  }
  __device__ __forceinline__ uint16_t *q_tile(int s, int hq) const {
    return reinterpret_cast<uint16_t *>(ws_q + (int64_t(s) * hk + hq) * TILE_BYTES);
  }
  __device__ __forceinline__ GateTile *gt_tile(int s, int h) const {
    return reinterpret_cast<GateTile *>(ws_gt + (int64_t(s) * hv + h) * GT_BYTES);
  }
  __device__ __forceinline__ float *fx_tile(int n, int h) const {
    return reinterpret_cast<float *>(ws_fx + (int64_t(n) * hv + h) * FX_BYTES);
  }
  __device__ __forceinline__ float *fp_tile(int n, int h) const {
    return reinterpret_cast<float *>(ws_fp + (int64_t(n) * hv + h) * FP_BYTES);
  }
  // the slab of value rows [v0, v0 + rows) of chunk slot s, head h (rows is the sweep's width)
  __device__ __forceinline__ uint16_t *sh_tile(int s, int h, int v0) const {
    return reinterpret_cast<uint16_t *>(ws_sh + ((int64_t(s) * hv + h) * D + v0) * (D * 2));
  }
  __device__ __forceinline__ uint16_t *d_tile(int s, int h, int v0) const {
    return reinterpret_cast<uint16_t *>(ws_d + ((int64_t(s) * hv + h) * D + v0) * (C * 2));
  }
};

// The output slot's open-chunk rows a call writes (state.cuh: a record means rows [0, pos mod 64)), from
// the sequence's last chunk g: rows [r_first, jf). Rows [0, r_new) predate the call (only when no new
// chunk was started); they are already in the output slot when it is the input slot.
struct OpenRows {
  int jf, r_new, r_first;
};
__device__ __forceinline__ OpenRows open_rows(const ChunkGeo &g, int si, int so) {
  OpenRows o;
  o.jf = (g.j0 + g.len) & (C - 1);
  o.r_new = g.nch <= 1 ? min(g.j0, o.jf) : 0;
  o.r_first = so == si ? o.r_new : 0;
  return o;
}

// Pass-2 tile helpers (sweep.cu, fused.cu). The m64n32 accumulator -> a [32 rows][cols] tile, transposed
// (rows of the tile = accumulator columns): element (r, c) of warp w's accumulator (r = 16w + g (+8), c = 8 nt + 2t (+1)) lands at row c, column
// col0 + r of `t`. Four stmatrix.trans, each two n8 tiles x two row halves.
template <class T, int NA>
__device__ __forceinline__ void store_transposed(const T &t, int col0, const float (&acc)[NA], int warp, int lane) {
  #pragma unroll
  for (int np = 0; np < NA / 8; ++np) {   // n8 tiles 2np, 2np + 1
    const int m = lane >> 3, r = lane & 7;
    const int row = 8 * (2 * np + (m >> 1)) + r, chunk = (col0 + 16 * warp) / 8 + (m & 1);
    stsm_x4_t(smem_u32(t.chunk(row, chunk)), pack(rn(acc[8 * np]), rn(acc[8 * np + 1])),
              pack(rn(acc[8 * np + 2]), rn(acc[8 * np + 3])), pack(rn(acc[8 * np + 4]), rn(acc[8 * np + 5])),
              pack(rn(acc[8 * np + 6]), rn(acc[8 * np + 7])));
  }
}
// The same for bf16 bits already packed per accumulator pair: bits[4 nt + 2 eh + el]
template <class T, int NA>
__device__ __forceinline__ void store_transposed_bits(const T &t, int col0, const uint16_t (&bits)[NA], int warp, int lane) {
  #pragma unroll
  for (int np = 0; np < NA / 8; ++np) {
    const int m = lane >> 3, r = lane & 7;
    const int row = 8 * (2 * np + (m >> 1)) + r, chunk = (col0 + 16 * warp) / 8 + (m & 1);
    stsm_x4_t(smem_u32(t.chunk(row, chunk)), pack(bits[8 * np], bits[8 * np + 1]), pack(bits[8 * np + 2], bits[8 * np + 3]),
              pack(bits[8 * np + 4], bits[8 * np + 5]), pack(bits[8 * np + 6], bits[8 * np + 7]));
  }
}

}  // namespace gdn_cpr
