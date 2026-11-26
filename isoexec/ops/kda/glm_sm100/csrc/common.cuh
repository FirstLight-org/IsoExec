// SPDX-License-Identifier: Apache-2.0
// ISA-level primitives of the SM100 KDA kernels (execution only; the arithmetic is recipe.cuh).
//
// These are the gdn_cpr_sm100 primitives (gdn/gdn_cpr_sm100/csrc/tile.cuh and recipe.cuh): the same
// swizzled bf16 tile, the same ldmatrix / stmatrix fragment maps for mma.sync.m16n8k16, the same paired
// bf16 narrowing and the same mma instruction. They are restated here, not #included across packages,
// because a kernel's identity is the digest of ITS csrc tree (build_cache.sm100_profile_digest); a header
// pulled from another package would change this kernel's arithmetic without changing its digest. The
// fragment formulas are ISA-forced (one address map per operand and majorness), so they are identical in
// both packages; the one extension is `Tile<ROWS, 32>`, a 64-byte row used for value-column slices.
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

namespace kda_sm100 {

constexpr int D = 128;   // head_k_dim == head_v_dim (GLM-5.3-Flash linear_head_dim)
constexpr int C = 64;    // storage height of a chunk tile (rows); the chunk LENGTH of the schedule is CL below
constexpr int BLK = 16;  // sub-block of the chunk (decay reference granularity): m16 of mma.sync.m16n8k16
// The chunk length of the schedule (tokens folded by one step of the reducer, anchored at absolute position 0 of
// the sequence). It is NOT a literal of this kernel: the build passes -DKDA_CL=<S> with S taken from
// isoexec/ops/schedule_policy.no_merge(block) by isoexec/ops/kda/schedule.py (cuda_defines). A chunk of CL < C
// rows is evaluated as a partial chunk of the 64-row tile (rows >= CL are +0).
#ifndef KDA_CL
#error "KDA_CL (the chunk length) must come from isoexec/ops/kda/schedule.py through the build; it has no default here"
#endif
constexpr int CL = KDA_CL;
static_assert(CL % BLK == 0 && CL >= BLK && CL <= C, "chunk length: whole 16-row blocks, at most the 64-row tile");

// ------------------------------------------------------------------------------ scalar operations
__device__ __forceinline__ float bf(uint16_t bits) { return __bfloat162float(__ushort_as_bfloat16(bits)); }
__device__ __forceinline__ uint16_t rn(float x) { return __bfloat16_as_ushort(__float2bfloat16_rn(x)); }
__device__ __forceinline__ float fadd(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ float fsub(float a, float b) { return __fsub_rn(a, b); }
__device__ __forceinline__ float fmul(float a, float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ float ffma(float a, float b, float c) { return __fmaf_rn(a, b, c); }
__device__ __forceinline__ uint32_t pack(uint16_t lo, uint16_t hi) { return uint32_t(lo) | (uint32_t(hi) << 16); }
// rn2(a, b) == pack(rn(a), rn(b)) for every input pair (cvt.rn.bf16x2.f32 rounds each half independently;
// gdn_cpr_sm100/tests/test_mma_facts_gpu.py proves it exhaustively).
__device__ __forceinline__ uint32_t rn2(float lo, float hi) {
  const __nv_bfloat162 p = __floats2bfloat162_rn(lo, hi);
  uint32_t r;
  memcpy(&r, &p, 4);
  return r;
}
__device__ __forceinline__ uint16_t lo16(uint32_t x) { return uint16_t(x & 0xffffu); }
__device__ __forceinline__ uint16_t hi16(uint32_t x) { return uint16_t(x >> 16); }

// ------------------------------------------------------------------------------ swizzled bf16 tiles
// Tile<ROWS, COLS>: bf16 rows of COLS elements as COLS/8 sixteen-byte chunks. For 128- and 64-wide rows
// chunk c of row r sits at chunk position c ^ (r & 7) (gdn_cpr_sm100's layout); for 32-wide rows (64-byte
// rows, two per 128-byte bank line) at c ^ ((r >> 1) & 3). Either way the eight rows one ldmatrix phase
// touches land in eight distinct 16-byte bank groups.
template <int ROWS, int COLS>
struct Tile {
  static constexpr int CH = COLS / 8;
  static constexpr int BYTES = ROWS * COLS * 2;
  static_assert(COLS == 32 || COLS == 64 || COLS == 128, "tile rows are 32, 64 or 128 bf16");
  uint16_t *base;
  __device__ __forceinline__ static int swz(int r, int c) { return CH >= 8 ? (c ^ (r & 7)) : (c ^ ((r >> 1) & 3)); }
  __device__ __forceinline__ uint32_t chunk_offset(int r, int c) const { return uint32_t((r * CH + swz(r, c)) * 16); }
  __device__ __forceinline__ uint16_t *chunk(int r, int c) const {
    return reinterpret_cast<uint16_t *>(reinterpret_cast<char *>(base) + chunk_offset(r, c));
  }
  __device__ __forceinline__ uint16_t &at(int r, int col) const { return chunk(r, col >> 3)[col & 7]; }
  __device__ __forceinline__ uint32_t saddr(int r, int c) const {
    return uint32_t(__cvta_generic_to_shared(base)) + chunk_offset(r, c);
  }
};

__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ void ldsm4t(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ void stsm4(uint32_t addr, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
  asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1,%2,%3,%4};" ::"r"(addr), "r"(r0), "r"(r1),
               "r"(r2), "r"(r3) : "memory");
}
__device__ __forceinline__ void stsm4t(uint32_t addr, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
  asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1,%2,%3,%4};" ::"r"(addr), "r"(r0),
               "r"(r1), "r"(r2), "r"(r3) : "memory");
}

// A[m0..m0+15][k0..k0+15] from a tile that stores A ([M][K], k contiguous).
template <class TT>
__device__ __forceinline__ void frag_a(uint32_t (&a)[4], const TT &t, int m0, int k0, int lane) {
  ldsm4(a, t.saddr(m0 + (lane & 15), (k0 >> 3) + (lane >> 4)));
}
// The same fragment when the tile stores A transposed ([K][M], m contiguous).
template <class TT>
__device__ __forceinline__ void frag_a_t(uint32_t (&a)[4], const TT &t, int m0, int k0, int lane) {
  ldsm4t(a, t.saddr(k0 + (lane & 7) + ((lane >> 4) << 3), (m0 >> 3) + ((lane >> 3) & 1)));
}
// B fragments of two adjacent n8 tiles (n0..n0+15, k0..k0+15) when the tile stores B transposed
// ([N][K], k contiguous): r[0..1] is the n8 tile at n0, r[2..3] the one at n0 + 8.
template <class TT>
__device__ __forceinline__ void frag_b2(uint32_t (&r)[4], const TT &t, int n0, int k0, int lane) {
  ldsm4(r, t.saddr(n0 + (lane & 7) + ((lane >> 4) << 3), (k0 >> 3) + ((lane >> 3) & 1)));
}
// The same two n8 tiles when the tile stores B itself ([K][N], n contiguous).
template <class TT>
__device__ __forceinline__ void frag_b2_t(uint32_t (&r)[4], const TT &t, int n0, int k0, int lane) {
  ldsm4t(r, t.saddr(k0 + (lane & 7) + (((lane >> 3) & 1) << 3), (n0 >> 3) + (lane >> 4)));
}
// A 16-row operand whose rows 1..15 are +0 (a one-row chunk): row 0 has its own storage (chunk c at
// position c, i.e. row 0 of a Tile), every other row reads one shared zero row. The fragment loaders see
// the same values a zero-filled Tile<16, COLS> would give them.
template <int COLS>
struct Row0Tile {
  uint16_t *base;        // COLS bf16
  const uint16_t *zero;  // >= COLS bf16 of +0
  __device__ __forceinline__ uint16_t &at(int, int col) const { return base[col]; }
  __device__ __forceinline__ uint32_t saddr(int r, int c) const {
    return uint32_t(__cvta_generic_to_shared(r == 0 ? base : zero)) + uint32_t(c * 16);
  }
};

// Two adjacent n8 accumulators (rows m0.., columns n0..n0+15) stored as bf16; w = packed pairs
// (tile0 rows g, tile0 rows g+8, tile1 rows g, tile1 rows g+8).
template <int R, int CO>
__device__ __forceinline__ void store_acc(const Tile<R, CO> &t, int m0, int n0, const uint32_t (&w)[4], int lane) {
  const int mm = lane >> 3, rr = lane & 7;
  stsm4(t.saddr(m0 + rr + 8 * (mm & 1), (n0 >> 3) + (mm >> 1)), w[0], w[1], w[2], w[3]);
}
// The same, transposed: element (m, n) lands at tile row n, column m.
template <int R, int CO>
__device__ __forceinline__ void store_acc_t(const Tile<R, CO> &t, int m0, int n0, const uint32_t (&w)[4],
                                            int lane) {
  const int mm = lane >> 3, rr = lane & 7;
  stsm4t(t.saddr(n0 + rr + 8 * (mm >> 1), (m0 >> 3) + (mm & 1)), w[0], w[1], w[2], w[3]);
}

// ------------------------------------------------------------------------------ async copies, barriers
// `valid == false` zero-fills: the recipe's zero extension (absent rows enter as +0, which is finite).
__device__ __forceinline__ void cp16(uint32_t saddr, const void *gptr, bool valid) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(saddr), "l"(gptr), "r"(valid ? 16 : 0)
               : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
}
__device__ __forceinline__ uint32_t smem_u32(const void *p) { return uint32_t(__cvta_generic_to_shared(p)); }

// ------------------------------------------------------------------------------ tensor-core step
// One m16n8k16 bf16 MMA into fp32 d (g = lane / 4, t = lane % 4):
//   a[0] = A[g][2t..], a[1] = A[g+8][2t..], a[2] = A[g][2t+8..], a[3] = A[g+8][2t+8..]
//   b[0] = B[2t..2t+1][g], b[1] = B[2t+8..2t+9][g]
//   d[0] = D[g][2t], d[1] = D[g][2t+1], d[2] = D[g+8][2t], d[3] = D[g+8][2t+1]
__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

}  // namespace kda_sm100
