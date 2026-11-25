// SPDX-License-Identifier: Apache-2.0
// The recurrent state record and the chunk geometry of a packed batch (execution only; no arithmetic).
//
// State record (format 3), one per (slot, value head), 16-byte aligned sections:
//   sh   [128 v][128 d] bf16   S^ = rn(S_c), the operand every decode step reads (one 16-byte load per k16 step)
//   lo   [128 v][128 d] u16    the low 16 bits of the fp32 boundary state S_c
//   tie  [128 v x 128 d] bits  one bit per element, meaningful only where lo = 0x8000 (below)
//                              all three in the mma A-fragment image order (s_index): 16-row value tile t, k16
//                              step ks, lane, the lane's 8 elements; the tie byte at (t ks lane) holds exactly
//                              that lane's 8 elements
//   The fp32 S_c is ENCODED, exactly: rn() moves the upper 16 bits h of S's bit pattern by at most one (b = h or
//   h + 1, as 16-bit integers, for either sign), upward exactly when lo > 0x8000 or lo = 0x8000 with h odd. So
//   h = b when lo < 0x8000, h = b - 1 when lo > 0x8000, and at lo = 0x8000 (b even either way) the tie bit says
//   which (1: h = b - 1). S = (h << 16) | lo. Decode reads 32 KiB (sh) per head; the boundary update and the
//   prefill commits encode, the readers of S_c decode (s_decode / s_encode below). Formats: 1 = fp32 S + bf16
//   image (168,704 B), 2 = fp32 S only (135,936 B), 3 = this (137,984 B).
//   kb   [64 i][128 d]  bf16   k^ rows of the open chunk            (the byte image of a Wide<64> tile)
//   vb   [128 v][64 i]  bf16   the values v, value-major            (the byte image of a Cols<128>)
//   db   [128 v][64 i]  bf16   delta, value-major                   (the byte image of a Cols<128>)
//   xb   [64 j][64 m]   fp32   rows of X = (I + L)^-1
//   pb   [16][64]       fp32   rows of P of the open 16-row block (row 16B + r at r; recipe.cuh)
//   gb   [64]           fp32   G (chunk-local cumulative log decay)
//   bb   [64]           fp32   beta
//   pos  int32 [NPOS]          tokens consumed so far (NPOS equal copies: a decode CTA reads and writes
//                              only its own copies, so the CTAs of a head need no ordering between
//                              them); the open chunk holds rows [0, pos mod 64)
// The bf16 buffers are stored exactly as the shared-memory tiles decode reads them into (tile.cuh), so a
// decode step moves them with a few bulk copies. Only rows (value-major: columns) [0, pos mod 64) are
// meaningful; the rest may hold anything (stale rows, NaN) -- readers zero the absent rows they use as
// operands and select, never multiply, past them. A slot is fresh when pos == 0 and S == S^ == 0 -- a
// NULL (negative) input slot stands for exactly that without touching memory.
#pragma once
#include <stdint.h>

#include "recipe.cuh"
#include "tile.cuh"

namespace gdn_cpr {

constexpr int NPOS = 8;   // copies of pos: one per CTA of the widest decode split

struct Rec {
  static constexpr int FORMAT = 3;
  static constexpr int64_t SH = 0;
  static constexpr int64_t LO = SH + int64_t(D) * D * 2;
  static constexpr int64_t TIE = LO + int64_t(D) * D * 2;
  static constexpr int64_t KB = TIE + int64_t(D) * D / 8;
  static constexpr int64_t VB = KB + int64_t(C) * D * 2;
  static constexpr int64_t DB = VB + int64_t(C) * D * 2;
  static constexpr int64_t XB = DB + int64_t(C) * D * 2;
  static constexpr int64_t PB = XB + int64_t(C) * C * 4;
  static constexpr int64_t GB = PB + int64_t(16) * C * 4;
  static constexpr int64_t BB = GB + int64_t(C) * 4;
  static constexpr int64_t POS = BB + int64_t(C) * 4;
  static constexpr int64_t BYTES = (POS + 4 * 8 + 255) / 256 * 256;   // per (slot, head)
};

struct StateRef {
  char *base;   // [P, hv, Rec::BYTES]; nullptr = stateless (trainer)
  int hv;
  __device__ __forceinline__ char *rec(int slot, int h) const { return base + (int64_t(slot) * hv + h) * Rec::BYTES; }
  __device__ __forceinline__ uint16_t *sh(int slot, int h) const { return reinterpret_cast<uint16_t *>(rec(slot, h) + Rec::SH); }
  __device__ __forceinline__ uint16_t *lo(int slot, int h) const { return reinterpret_cast<uint16_t *>(rec(slot, h) + Rec::LO); }
  __device__ __forceinline__ uint8_t *tie(int slot, int h) const { return reinterpret_cast<uint8_t *>(rec(slot, h) + Rec::TIE); }
  __device__ __forceinline__ uint16_t *kb(int slot, int h) const { return reinterpret_cast<uint16_t *>(rec(slot, h) + Rec::KB); }
  __device__ __forceinline__ uint16_t *vb(int slot, int h) const { return reinterpret_cast<uint16_t *>(rec(slot, h) + Rec::VB); }
  __device__ __forceinline__ uint16_t *db(int slot, int h) const { return reinterpret_cast<uint16_t *>(rec(slot, h) + Rec::DB); }
  __device__ __forceinline__ float *xb(int slot, int h) const { return reinterpret_cast<float *>(rec(slot, h) + Rec::XB); }
  __device__ __forceinline__ float *pb(int slot, int h) const { return reinterpret_cast<float *>(rec(slot, h) + Rec::PB); }
  __device__ __forceinline__ float *gb(int slot, int h) const { return reinterpret_cast<float *>(rec(slot, h) + Rec::GB); }
  __device__ __forceinline__ float *bb(int slot, int h) const { return reinterpret_cast<float *>(rec(slot, h) + Rec::BB); }
  __device__ __forceinline__ int *pos(int slot, int h) const { return reinterpret_cast<int *>(rec(slot, h) + Rec::POS); }
  __device__ __forceinline__ Wide<64> kb_tile(int slot, int h) const { return Wide<64>{kb(slot, h)}; }
  __device__ __forceinline__ Cols<D> vb_tile(int slot, int h) const { return Cols<D>{vb(slot, h)}; }
  __device__ __forceinline__ Cols<D> db_tile(int slot, int h) const { return Cols<D>{db(slot, h)}; }
};


// Element index of S (v, d) in the record's A-fragment image: lane 4 (v mod 8) + (d / 2) mod 4, register
// (v / 8) mod 2 + 2 ((d / 8) mod 2), element d mod 2 of the register's pair; a 16-row value tile is one contiguous
// 2048-element block, so whole tiles of value rows are contiguous runs, and (v, d), (v, d + 1) (d even) are
// adjacent.
__host__ __device__ __forceinline__ int s_index(int v, int d) {
  const int lane = 4 * (v & 7) + ((d >> 1) & 3), reg = ((v >> 3) & 1) + 2 * ((d >> 3) & 1);
  return ((((v >> 4) * 8 + (d >> 4)) * 32 + lane) * 8) + reg * 2 + (d & 1);
}

// ---- the exact fp32 <-> (sh, lo, tie) code (format 3)
__device__ __forceinline__ float s_decode(uint16_t b, uint16_t l, unsigned t) {
  const uint16_t h = l < 0x8000u ? b : (l > 0x8000u ? uint16_t(b - 1) : uint16_t(b - (t & 1u)));
  return __uint_as_float((uint32_t(h) << 16) | l);
}
struct SCode {
  uint16_t b, l;
  unsigned t;
};
__device__ __forceinline__ SCode s_encode(float x) {
  const uint32_t u = __float_as_uint(x);
  const uint16_t b = rn(x), l = uint16_t(u & 0xffffu), h = uint16_t(u >> 16);
  return SCode{b, l, unsigned(l == 0x8000u && h != b)};
}

// One element at image index i (prefill readers and writers). The tie bit is written with a single-bit atomic
// (OR to set, AND to clear), so threads writing different elements of one tie word never race.
__device__ __forceinline__ float s_load(const StateRef &st, int slot, int h, int i) {
  return s_decode(st.sh(slot, h)[i], st.lo(slot, h)[i], unsigned(st.tie(slot, h)[i >> 3] >> (i & 7)));
}
__device__ __forceinline__ void s_store(const StateRef &st, int slot, int h, int i, float x) {
  const SCode c = s_encode(x);
  st.sh(slot, h)[i] = c.b;
  st.lo(slot, h)[i] = c.l;
  unsigned *word = reinterpret_cast<unsigned *>(st.tie(slot, h)) + (i >> 5);
  const unsigned bit = 1u << (i & 31);
  if (c.t) atomicOr(word, bit);
  else atomicAnd(word, ~bit);
}

// The 8 elements of one lane's fragment (image indices base .. base + 7, base = 8 ((t 8 + ks) 32 + lane)):
// one thread owns the whole tie byte, so plain stores suffice (decode's boundary update).
__device__ __forceinline__ void s_load8(const StateRef &st, int slot, int h, int base, float (&x)[8]) {
  const uint4 b = *reinterpret_cast<const uint4 *>(st.sh(slot, h) + base);
  const uint4 l = *reinterpret_cast<const uint4 *>(st.lo(slot, h) + base);
  const unsigned t = st.tie(slot, h)[base >> 3];
  const uint32_t bw[4] = {b.x, b.y, b.z, b.w}, lw[4] = {l.x, l.y, l.z, l.w};
  #pragma unroll
  for (int r = 0; r < 4; ++r) {
    x[2 * r] = s_decode(lo16(bw[r]), lo16(lw[r]), t >> (2 * r));
    x[2 * r + 1] = s_decode(hi16(bw[r]), hi16(lw[r]), t >> (2 * r + 1));
  }
}
__device__ __forceinline__ void s_store8(const StateRef &st, int slot, int h, int base, const float (&x)[8]) {
  uint32_t bw[4], lw[4];
  unsigned t = 0;
  #pragma unroll
  for (int r = 0; r < 4; ++r) {
    const SCode c0 = s_encode(x[2 * r]), c1 = s_encode(x[2 * r + 1]);
    bw[r] = pack(c0.b, c1.b);
    lw[r] = pack(c0.l, c1.l);
    t |= (c0.t << (2 * r)) | (c1.t << (2 * r + 1));
  }
  *reinterpret_cast<uint4 *>(st.sh(slot, h) + base) = make_uint4(bw[0], bw[1], bw[2], bw[3]);
  *reinterpret_cast<uint4 *>(st.lo(slot, h) + base) = make_uint4(lw[0], lw[1], lw[2], lw[3]);
  st.tie(slot, h)[base >> 3] = uint8_t(t);
}

// Copy (or zero, when src is null) the S sections of the value rows [v0, v0 + rows) -- whole 16-row tiles, so
// three contiguous runs -- in 16-byte pieces split over `nthreads` threads starting at `tid`.
__device__ __forceinline__ void s_move_rows(const StateRef &st, int src_slot, int dst_slot, int h, int v0, int rows,
                                            int tid, int nthreads) {
  const int n16 = rows * D * 2 / 16, t16 = rows * D / 8 / 16;   // 16-byte pieces of sh (= of lo) and of tie
  uint4 *dsh = reinterpret_cast<uint4 *>(st.sh(dst_slot, h) + v0 * D), *dlo = reinterpret_cast<uint4 *>(st.lo(dst_slot, h) + v0 * D);
  uint4 *dtie = reinterpret_cast<uint4 *>(st.tie(dst_slot, h) + v0 * D / 8);
  const uint4 z = make_uint4(0u, 0u, 0u, 0u);
  const bool src = src_slot >= 0;
  const uint4 *ssh = src ? reinterpret_cast<const uint4 *>(st.sh(src_slot, h) + v0 * D) : nullptr;
  const uint4 *slo = src ? reinterpret_cast<const uint4 *>(st.lo(src_slot, h) + v0 * D) : nullptr;
  const uint4 *stie = src ? reinterpret_cast<const uint4 *>(st.tie(src_slot, h) + v0 * D / 8) : nullptr;
  for (int x = tid; x < n16; x += nthreads) {
    dsh[x] = src ? ssh[x] : z;
    dlo[x] = src ? slo[x] : z;
  }
  for (int x = tid; x < t16; x += nthreads) dtie[x] = src ? stie[x] : z;
}

// Chunks of a packed batch. Sequence n (tokens [cs, ce) of the pack, starting at absolute position
// pos_in) covers chunks k = 0 .. nch-1; tile row i of chunk k is token cs - j0 + 64k + i, where
// j0 = pos_in mod 64, and the rows present in this call are [r_lo, r_hi).
struct ChunkGeo {
  int n, k;          // sequence, chunk within this call
  int cs, len, j0;   // packed start, tokens in this call, first open row
  int r_lo, r_hi;    // rows present
  int nch;           // chunks of this sequence in this call
  __device__ __forceinline__ int64_t token(int i) const { return int64_t(cs) - j0 + 64 * k + i; }   // packed row
  __device__ __forceinline__ bool first() const { return k == 0; }
  __device__ __forceinline__ bool last() const { return k == nch - 1; }
  __device__ __forceinline__ bool complete() const { return r_hi == C; }
};

__host__ __device__ __forceinline__ int chunks_of(int len, int j0) { return len > 0 ? (j0 + len + C - 1) / C : 0; }

__device__ __forceinline__ ChunkGeo chunk_geo(int n, int k, int cs, int len, int j0) {
  ChunkGeo g;
  g.n = n;
  g.k = k;
  g.cs = cs;
  g.len = len;
  g.j0 = j0;
  g.nch = chunks_of(len, j0);
  g.r_lo = k == 0 ? j0 : 0;
  int hi = j0 + len - 64 * k;
  g.r_hi = hi < C ? hi : C;
  return g;
}

}  // namespace gdn_cpr
