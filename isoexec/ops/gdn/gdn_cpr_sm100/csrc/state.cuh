// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stdint.h>

#include "recipe.cuh"
#include "tile.cuh"

namespace gdn_cpr_sm100 {

constexpr int NPOS = 8;   // copies of pos: one per CTA of the widest decode split

using KTile = Tile<C, D>;    // k^ [i][d], v [i][v], deltas [i][v], w [j][d], u [j][v], q^ [j][d]
using MTile = Tile<C, C>;    // M [j][i], xu / xw [j][i]
using ShTile = Tile<D, D>;   // S^ [v][d]

struct Rec {
  static constexpr int64_t S = 0;
  static constexpr int64_t SH = S + int64_t(D) * D * 4;
  static constexpr int64_t KB = SH + int64_t(D) * D * 2;
  static constexpr int64_t VB = KB + int64_t(C) * D * 2;
  static constexpr int64_t DB = VB + int64_t(C) * D * 2;
  static constexpr int64_t XB = DB + int64_t(C) * D * 2;
  static constexpr int64_t PB = XB + int64_t(C) * C * 4;
  static constexpr int64_t GB = PB + int64_t(16) * C * 4;
  static constexpr int64_t BB = GB + int64_t(C) * 4;
  static constexpr int64_t POS = BB + int64_t(C) * 4;
  static constexpr int64_t BYTES = (POS + 4 * NPOS + 255) / 256 * 256;   // per (slot, head)
};

struct StateRef {
  char *base;   // [P, hv, Rec::BYTES]; nullptr = stateless (trainer)
  int hv;
  __device__ __forceinline__ char *rec(int slot, int h) const {
    return base + (int64_t(slot) * hv + h) * Rec::BYTES;
  }
  __device__ __forceinline__ float *S(int slot, int h) const {
    return reinterpret_cast<float *>(rec(slot, h) + Rec::S);
  }
  __device__ __forceinline__ uint16_t *sh(int slot, int h) const {
    return reinterpret_cast<uint16_t *>(rec(slot, h) + Rec::SH);
  }
  __device__ __forceinline__ uint16_t *kb(int slot, int h) const {
    return reinterpret_cast<uint16_t *>(rec(slot, h) + Rec::KB);
  }
  __device__ __forceinline__ uint16_t *vb(int slot, int h) const {
    return reinterpret_cast<uint16_t *>(rec(slot, h) + Rec::VB);
  }
  __device__ __forceinline__ uint16_t *db(int slot, int h) const {
    return reinterpret_cast<uint16_t *>(rec(slot, h) + Rec::DB);
  }
  __device__ __forceinline__ float *xb(int slot, int h) const {
    return reinterpret_cast<float *>(rec(slot, h) + Rec::XB);
  }
  __device__ __forceinline__ float *pb(int slot, int h) const {
    return reinterpret_cast<float *>(rec(slot, h) + Rec::PB);
  }
  __device__ __forceinline__ float *gb(int slot, int h) const {
    return reinterpret_cast<float *>(rec(slot, h) + Rec::GB);
  }
  __device__ __forceinline__ float *bb(int slot, int h) const {
    return reinterpret_cast<float *>(rec(slot, h) + Rec::BB);
  }
  __device__ __forceinline__ int *pos(int slot, int h) const {
    return reinterpret_cast<int *>(rec(slot, h) + Rec::POS);
  }
};

// Chunks of a packed batch. Sequence n (tokens [cs, cs + len) of the pack, starting at absolute position
// pos_in) covers chunks k = 0 .. nch-1; tile row i of chunk k is token cs - j0 + 64k + i, where
// j0 = pos_in mod 64, and the rows present in this call are [r_lo, r_hi).
struct ChunkGeo {
  int n, k;
  int cs, len, j0;
  int r_lo, r_hi;
  int nch;
  __device__ __forceinline__ int64_t token(int i) const { return int64_t(cs) - j0 + int64_t(C) * k + i; }
  __device__ __forceinline__ bool first() const { return k == 0; }
  __device__ __forceinline__ bool last() const { return k == nch - 1; }
  __device__ __forceinline__ bool complete() const { return r_hi == C; }
};

__host__ __device__ __forceinline__ int chunks_of(int len, int j0) {
  return len > 0 ? (j0 + len + C - 1) / C : 0;
}

__device__ __forceinline__ ChunkGeo chunk_geo(int n, int k, int cs, int len, int j0) {
  ChunkGeo g;
  g.n = n;
  g.k = k;
  g.cs = cs;
  g.len = len;
  g.j0 = j0;
  g.nch = chunks_of(len, j0);
  g.r_lo = k == 0 ? j0 : 0;
  const int hi = j0 + len - C * k;
  g.r_hi = hi < C ? hi : C;
  return g;
}

// The open-chunk rows a call leaves in the output slot, from the sequence's last chunk g: rows
// [r_first, jf). Rows [0, r_new) predate the call (only when no new chunk was started); they are already
// in the output slot when it is the input slot.
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

}  // namespace gdn_cpr_sm100
