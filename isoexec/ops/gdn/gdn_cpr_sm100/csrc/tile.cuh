// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

#include "recipe.cuh"

namespace gdn_cpr_sm100 {

template <int ROWS, int COLS>
struct Tile {
  static constexpr int CH = COLS / 8;          // 16-byte chunks per row
  static constexpr int BYTES = ROWS * COLS * 2;
  static_assert(COLS == 64 || COLS == 128, "tile rows are 64 or 128 bf16");
  uint16_t *base;
  __device__ __forceinline__ uint32_t chunk_offset(int r, int c) const {
    return uint32_t((r * CH + (c ^ (r & 7))) * 16);
  }
  __device__ __forceinline__ uint16_t *chunk(int r, int c) const {
    return reinterpret_cast<uint16_t *>(reinterpret_cast<char *>(base) + chunk_offset(r, c));
  }
  __device__ __forceinline__ uint16_t &at(int r, int col) const { return chunk(r, col >> 3)[col & 7]; }
  __device__ __forceinline__ uint32_t saddr(int r, int c) const {
    return uint32_t(__cvta_generic_to_shared(base)) + chunk_offset(r, c);
  }
};

// Storage offset (in bf16 elements) of element (r, col) in a COLS-wide tile image: the host-side index
// map of the same layout (ops.py mirrors this for the record's logical views).
template <int COLS>
__host__ __device__ __forceinline__ int tile_offset(int r, int col) {
  constexpr int CH = COLS / 8;
  return (r * CH + ((col >> 3) ^ (r & 7))) * 8 + (col & 7);
}

__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
// The same load under an EXPLICIT guard predicate. `need == 0` leaves `r` indeterminate, which is what
// the caller wants: the only reader is a `mma_p` carrying the same predicate. See `recipe.cuh::mma_p` for
// why the guard is written here rather than as `if (...) continue;` around the pair.
__device__ __forceinline__ void ldsm4_p(uint32_t (&r)[4], uint32_t addr, unsigned need) {
  asm volatile("{\n\t"
               ".reg .pred pg;\n\t"
               "setp.ne.b32 pg, %5, 0;\n\t"
               "@pg ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n\t"
               "}"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr), "r"(need));
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

// A[m0..m0+15][k0..k0+15] from a tile that stores A itself ([M][K], k contiguous).
template <int R, int CO>
__device__ __forceinline__ void frag_a(uint32_t (&a)[4], const Tile<R, CO> &t, int m0, int k0, int lane) {
  ldsm4(a, t.saddr(m0 + (lane & 15), (k0 >> 3) + (lane >> 4)));
}
// The same A fragment under a guard predicate (`ldsm4_p`).
template <int R, int CO>
__device__ __forceinline__ void frag_a_p(uint32_t (&a)[4], const Tile<R, CO> &t, int m0, int k0, int lane,
                                         unsigned need) {
  ldsm4_p(a, t.saddr(m0 + (lane & 15), (k0 >> 3) + (lane >> 4)), need);
}
// The same fragment when the tile stores A transposed ([K][M], m contiguous).
template <int R, int CO>
__device__ __forceinline__ void frag_a_t(uint32_t (&a)[4], const Tile<R, CO> &t, int m0, int k0, int lane) {
  ldsm4t(a, t.saddr(k0 + (lane & 7) + ((lane >> 4) << 3), (m0 >> 3) + ((lane >> 3) & 1)));
}
// B fragments of the two adjacent n8 tiles (n0..n0+15, k0..k0+15) when the tile stores B transposed
// ([N][K], k contiguous): r[0..1] is the n8 tile at n0, r[2..3] the one at n0 + 8.
template <int R, int CO>
__device__ __forceinline__ void frag_b2(uint32_t (&r)[4], const Tile<R, CO> &t, int n0, int k0, int lane) {
  ldsm4(r, t.saddr(n0 + (lane & 7) + ((lane >> 4) << 3), (k0 >> 3) + ((lane >> 3) & 1)));
}
// The same two n8 tiles when the tile stores B itself ([K][N], n contiguous).
template <int R, int CO>
__device__ __forceinline__ void frag_b2_t(uint32_t (&r)[4], const Tile<R, CO> &t, int n0, int k0, int lane) {
  ldsm4t(r, t.saddr(k0 + (lane & 7) + (((lane >> 3) & 1) << 3), (n0 >> 3) + (lane >> 4)));
}

// Two adjacent n8 accumulators (16 rows m0.., 16 columns n0..) stored as bf16.
// `w[k]` is the PAIR of accumulator elements (2k, 2k+1) of n8 tile k / 2, already packed -- element e of
// tile 0 for e < 4 and of tile 1 for e >= 4, in the accumulator's own order
// (D[g][2t], D[g][2t+1], D[g+8][2t], D[g+8][2t+1]). It is a pair rather than eight `uint16_t` because the
// two elements of every pair are exactly what one `rn2` produces (recipe.cuh): the callers here were
// paying two `F2F.BF16.F32` and a `PRMT` for what is one `F2FP.BF16.F32.PACK_AB`, on the SM's quarter-rate
// XU unit, roughly 930 times a thread a chunk.
template <int R, int CO>
__device__ __forceinline__ void store_acc(const Tile<R, CO> &t, int m0, int n0, const uint32_t (&w)[4],
                                          int lane) {
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

// ---------------------------------------------------------------------------- 16-byte async copies
// `valid == false` zero-fills the destination, which is how the recipe's zero extension is realized on
// the load path: an absent row's operands must be finite, and +0 is finite.
__device__ __forceinline__ void cp16(uint32_t saddr, const void *gptr, bool valid) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(saddr), "l"(gptr), "r"(valid ? 16 : 0)
               : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
}

// ---------------------------------------------------------------------------- shared-memory helpers
__device__ __forceinline__ uint32_t smem_u32(const void *p) { return uint32_t(__cvta_generic_to_shared(p)); }

__device__ __forceinline__ void fence_async_smem() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }

// A named barrier over `n` threads: warpgroup-local synchronization without stalling the CTA.
__device__ __forceinline__ void bar_sync(int id, int n) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory");
}

}  // namespace gdn_cpr_sm100
