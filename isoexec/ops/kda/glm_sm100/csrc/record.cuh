// SPDX-License-Identifier: Apache-2.0
// The engine state record: one per (slot, head), chunks anchored at ABSOLUTE sequence position 0.
//
// A sequence at position pos is S_c (the fp32 state at its last chunk boundary, c = pos div 64) plus the
// open chunk's rows [0, pos mod 64). Every section is what the chunk recipe needs to evaluate the next
// rows of that chunk exactly as a single whole-sequence call would:
//   SH  bf16 [128 v][128 k]   S^ = rn(S_c), k contiguous (the B operand image of the d / o-seed chains)
//   SL  u16  [128 k][128 v]   the LOW 16 bits of S_c's fp32 pattern
//   SC  bits, 8x8 blocks (2,048 B)   per element: SH != the top 16 bits of S_c (rn rounded the
//                             magnitude up). S_c is ((SH - SC) << 16) | SL, bit for bit, for every finite or
//                             infinite pattern: the fp32 state is stored once, split between the operand
//                             image the step reads (SH) and the part only a chunk close or a record read
//                             needs (SL, SC). Layout in record.cuh "State planes" below.
//   KB  bf16 [64][128]        k^ rows of the open chunk (at a close: kd^T [128 c][64 i], the A operand of
//                             the state fold, in the same bytes)
//   VB  bf16 [64 i][128]      v rows
//   DB  bf16 [63 i][128]      d rows (row 63 is the closing step's own, taken from shared memory)
//   KGB bf16 [63 i][128]      kg = rn(k^ (*) E(G)) rows (row 63 is never read: the chunk closes with it)
//   KI  bf16 [48][128]        ki^b = rn(k^_i (*) E(G_16b (-) G_i)), i < 16 b, of the open block b (written
//                             when the block has rows; a decode step entering a new block computes it).
//                             KDA_REC_RECOMPUTE_KI: not stored; every step of block b >= 1 computes it
//                             exactly as the entering step does.
//   GB  fp32 [64][128]        G rows (chunk-local cumulative log2 decay)
//   XB  fp32 X rows [0, 63), lower-triangular packed: row i holds columns [0, 4 (i/4 + 1)) at xrow_off(i)
//                             (X_im = +0 for m > i is never stored or read; 2112 floats)
//   PB  fp32 [16][48]         P rows of the open 16-row block (row 16 B + r at r; columns < 16 B <= 48
//                             are live, +0 past 16 B)
//   BB  fp32 [64]             beta
//   POS int32                 tokens consumed (16 bytes)
// Rows at or past pos mod 64 may hold anything and are never read as data. A fresh slot is all zeros.
// Prefill (intra.cu / state.cu) RECOMPUTES the open rows from k^, v, G, beta and S_c -- the recipe is
// row-separable, so those rows get the bits they had -- and writes the record through a carry buffer
// (phase 1 never writes a record byte: another chunk of the same call may still be reading it). Decode
// (decode_rec.cu) evaluates row pos mod 64 incrementally from all sections.
#pragma once
#include <stdint.h>

#include "record_profile.cuh"

namespace kda_sm100 {

// X rows lower-triangular packed in groups of four rows: rows of group G = i / 4 hold 4 (G + 1) floats
// (every column m <= i, a 16-byte multiple), row i at xrow_off(i) floats; rows [0, 64) take X_FLOATS (the
// carry's staging), the record keeps rows [0, 63) (X_REC_FLOATS): the closing row's X is never read.
constexpr int X_FLOATS = 2176;
constexpr int P_COLS = 48;   // P_jm lives at m < 16 B <= 48
__host__ __device__ __forceinline__ constexpr int xrow_off(int i) { return 4 * ((i >> 2) + 1) * (2 * (i >> 2) + (i & 3)); }
__host__ __device__ __forceinline__ constexpr int xrow_len(int i) { return 4 * ((i >> 2) + 1); }
static_assert(xrow_off(64) == X_FLOATS, "packed X rows");
constexpr int X_REC_FLOATS = xrow_off(63);
static_assert(X_REC_FLOATS == 2112, "record X rows");

// The open block's decay-factored earlier keys: stored (default) or recomputed by every step of the block.
// Same function of the same stored rows either way (decode_rec.cu's block entry), so the bits do not change;
// the step pays the exponentials a block entry pays, every step, to save 12,288 bytes a head.
#ifdef KDA_REC_RECOMPUTE_KI
constexpr bool KI_STORED = false;
#else
constexpr bool KI_STORED = true;
#endif

// State planes. The fp32 state S_c [128 k][128 v] is held as the bf16 image the step reads (SH, [v][k]) plus
// the 16 bits and one carry bit per element that complete it (S = ((SH - carry) << 16) | SL). SL is [k][v]
// u16 (the accumulator pair of an element is one 32-bit word). SC is a bitmap of 8 x 8 blocks (cpr_planes.cuh, shared with
// the GDN record): block (k / 8, v / 8) is the 64-bit word (v / 8) * 16 + k / 8, bit (k % 8) * 8 + (v % 8).
constexpr int64_t SH_BYTES = 128 * 128 * 2, SL_BYTES = 128 * 128 * 2, SC_BYTES = 8 * 16 * 16;   // SC: 256 64-bit 8x8 carry blocks (cpr_planes.cuh)

struct Rec {
  static constexpr int64_t SH = 0;
  static constexpr int64_t SL = SH + SH_BYTES;
  static constexpr int64_t SC = SL + SL_BYTES;
  static constexpr int64_t KB = SC + SC_BYTES;
  static constexpr int64_t VB = KB + 64 * 128 * 2;
  static constexpr int64_t DB = VB + 64 * 128 * 2;
  static constexpr int64_t KGB = DB + 63 * 128 * 2;
  static constexpr int64_t KI = KGB + 63 * 128 * 2;
  static constexpr int64_t GB = KI + (KI_STORED ? 48 * 128 * 2 : 0);
  static constexpr int64_t XB = GB + 64 * 128 * 4;
  static constexpr int64_t PB = XB + X_REC_FLOATS * 4;
  static constexpr int64_t BB = PB + 16 * P_COLS * 4;
  static constexpr int64_t POS = BB + 64 * 4;
  static constexpr int64_t BYTES = POS + 16;
  static_assert(BYTES % 16 == 0 && SC % 16 == 0 && KB % 16 == 0 && VB % 16 == 0 && DB % 16 == 0 && KGB % 16 == 0 &&
                    KI % 16 == 0 && GB % 16 == 0 && XB % 16 == 0 && PB % 16 == 0 && BB % 16 == 0 && POS % 16 == 0,
                "record sections are 16-byte aligned (cp.async / uint4)");
};

// Phase 1's carry of a sequence's last chunk (rows [0, jf) of the chunk that stays open), per (seq, head).
struct Carry {
  static constexpr int64_t KB = 0;
  static constexpr int64_t VB = KB + 64 * 128 * 2;
  static constexpr int64_t KGB = VB + 64 * 128 * 2;
  static constexpr int64_t KI = KGB + 64 * 128 * 2;
  static constexpr int64_t GB = KI + 48 * 128 * 2;
  static constexpr int64_t XB = GB + 64 * 128 * 4;
  static constexpr int64_t PB = XB + X_FLOATS * 4;
  static constexpr int64_t BB = PB + 16 * P_COLS * 4;
  static constexpr int64_t BYTES = BB + 256;
};

struct RecRef {
  char *base;      // [slots][H][Rec::BYTES] (slot stride free: an engine's page); nullptr: functional mode
  int H;
  int64_t slot_stride;   // bytes between consecutive slots (>= H * Rec::BYTES)
  __device__ __forceinline__ char *rec(int slot, int h) const {
    return base + int64_t(slot) * slot_stride + int64_t(h) * Rec::BYTES;
  }
  __device__ __forceinline__ int *pos(int slot, int h) const { return reinterpret_cast<int *>(rec(slot, h) + Rec::POS); }
};

}  // namespace kda_sm100
