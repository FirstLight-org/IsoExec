// SPDX-License-Identifier: Apache-2.0
// Shared-memory tiles, asynchronous copies and MMA fragment loads (execution only; no arithmetic).
//
// Tiles hold bf16 rows of 64 or 128 elements (8 or 16 chunks of 16 bytes). Chunk c of row r lives at
// chunk position c ^ (r & 7), so the eight row addresses of every ldmatrix phase land in eight
// distinct 16-byte bank groups (conflict-free), whichever column the phase reads. A 64-wide tile with a
// 1024-byte aligned base is exactly the hardware's 128-byte-swizzle layout, which WGMMA descriptors read;
// 128-wide tiles that WGMMA consumes are stored as two such 64-wide halves (Wide).
#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

namespace gdn_cpr {

template <int COLS>  // bf16 elements per row: 64 or 128
struct Tile {
  static constexpr int CHUNKS = COLS / 8;
  static_assert(CHUNKS == 8 || CHUNKS == 16, "tile rows are 64 or 128 bf16");
  uint16_t *base;
  __device__ __forceinline__ uint32_t chunk_offset(int r, int c) const {  // bytes
    return uint32_t((r * CHUNKS + (c ^ (r & 7))) * 16);
  }
  __device__ __forceinline__ uint16_t *chunk(int r, int c) const {
    return reinterpret_cast<uint16_t *>(reinterpret_cast<char *>(base) + chunk_offset(r, c));
  }
  __device__ __forceinline__ uint16_t &at(int r, int col) const { return chunk(r, col >> 3)[col & 7]; }
  __device__ __forceinline__ uint32_t saddr(int r, int c) const {
    return uint32_t(__cvta_generic_to_shared(base)) + chunk_offset(r, c);
  }
};


// A [ROWS][128] bf16 tile stored as two [ROWS][64] 128-byte-swizzle halves (columns 0..63, 64..127).
template <int ROWS>
struct Wide {
  uint16_t *base;
  __device__ __forceinline__ Tile<64> half(int h) const { return Tile<64>{base + h * ROWS * 64}; }
  __device__ __forceinline__ uint16_t *chunk(int r, int c) const { return half(c >> 3).chunk(r, c & 7); }
  __device__ __forceinline__ uint16_t &at(int r, int col) const { return half(col >> 6).at(r, col & 63); }
};

// A [ROWS][64] bf16 tile stored in 16-row slabs, each chunk-major: slab s holds chunk 0 of its 16 rows
// (16 bytes each, consecutive), then chunk 1, ... The eight rows of an ldmatrix phase are 128 contiguous
// bytes (conflict-free), and the column prefix [0, 8n) of a slab is one contiguous run of 256n bytes --
// a column prefix of any 16-row-aligned row range moves with one bulk copy per slab.
template <int ROWS>
struct Cols {
  uint16_t *base;
  __device__ __forceinline__ uint32_t chunk_offset(int r, int c) const {
    return uint32_t(((((r >> 4) << 3) + c) * 16 + (r & 15)) * 16);
  }
  __device__ __forceinline__ uint16_t *chunk(int r, int c) const {
    return reinterpret_cast<uint16_t *>(reinterpret_cast<char *>(base) + chunk_offset(r, c));
  }
  __device__ __forceinline__ uint16_t &at(int r, int col) const { return chunk(r, col >> 3)[col & 7]; }
  __device__ __forceinline__ uint32_t saddr(int r, int c) const {
    return uint32_t(__cvta_generic_to_shared(base)) + chunk_offset(r, c);
  }
};

__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}

// stmatrix: four 8x8 bf16 matrices from the mma accumulator/A-fragment layout to shared memory (lanes
// 8m..8m+7 address the rows of matrix m); .trans stores each matrix transposed.
__device__ __forceinline__ void stsm_x4(uint32_t addr, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
  asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1,%2,%3,%4};" ::"r"(addr), "r"(r0), "r"(r1), "r"(r2),
               "r"(r3) : "memory");
}
__device__ __forceinline__ void stsm_x4_t(uint32_t addr, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
  asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1,%2,%3,%4};" ::"r"(addr), "r"(r0), "r"(r1),
               "r"(r2), "r"(r3) : "memory");
}

// A fragment (16 x 16 at rows m0.., k-columns k0..) of a tile stored [M][K] (row-major A).
template <int COLS>
__device__ __forceinline__ void frag_a(uint32_t (&a)[4], const Tile<COLS> &t, int m0, int k0, int lane) {
  ldsm_x4(a, t.saddr(m0 + (lane & 15), (k0 >> 3) + (lane >> 4)));
}
// A fragment of A = T^T, from a tile stored [K][M] (e.g. A = bv^T with bv stored [i][v]).
template <int COLS>
__device__ __forceinline__ void frag_a_t(uint32_t (&a)[4], const Tile<COLS> &t, int m0, int k0, int lane) {
  ldsm_x4_t(a, t.saddr(k0 + (lane & 7) + ((lane >> 4) << 3), (m0 >> 3) + ((lane >> 3) & 1)));
}
// B fragments of two adjacent n8 tiles (n0..n0+15, k0..k0+15) from a tile stored [N][K]:
// b0 = {r[0], r[1]} for columns n0..n0+7, b1 = {r[2], r[3]} for n0+8..n0+15.
template <int COLS>
__device__ __forceinline__ void frag_b2(uint32_t (&r)[4], const Tile<COLS> &t, int n0, int k0, int lane) {
  ldsm_x4(r, t.saddr(n0 + (lane & 7) + ((lane >> 4) << 3), (k0 >> 3) + ((lane >> 3) & 1)));
}
// The same two n8 tiles from a tile stored [K][N] (B = T itself, e.g. k^ rows as B[i][d]).
template <int COLS>
__device__ __forceinline__ void frag_b2_t(uint32_t (&r)[4], const Tile<COLS> &t, int n0, int k0, int lane) {
  ldsm_x4_t(r, t.saddr(k0 + (lane & 7) + (((lane >> 3) & 1) << 3), (n0 >> 3) + (lane >> 4)));
}


template <int ROWS>
__device__ __forceinline__ void frag_a(uint32_t (&a)[4], const Cols<ROWS> &t, int m0, int k0, int lane) {
  ldsm_x4(a, t.saddr(m0 + (lane & 15), (k0 >> 3) + (lane >> 4)));
}
template <int ROWS>
__device__ __forceinline__ void frag_a(uint32_t (&a)[4], const Wide<ROWS> &t, int m0, int k0, int lane) {
  frag_a(a, t.half(k0 >> 6), m0, k0 & 63, lane);
}
template <int ROWS>
__device__ __forceinline__ void frag_a_t(uint32_t (&a)[4], const Wide<ROWS> &t, int m0, int k0, int lane) {
  frag_a_t(a, t.half(m0 >> 6), m0 & 63, k0, lane);
}
template <int ROWS>
__device__ __forceinline__ void frag_b2(uint32_t (&r)[4], const Wide<ROWS> &t, int n0, int k0, int lane) {
  frag_b2(r, t.half(k0 >> 6), n0, k0 & 63, lane);
}
template <int ROWS>
__device__ __forceinline__ void frag_b2_t(uint32_t (&r)[4], const Wide<ROWS> &t, int n0, int k0, int lane) {
  frag_b2_t(r, t.half(n0 >> 6), n0 & 63, k0, lane);
}

// 16-byte asynchronous global->shared copy; `valid == false` zero-fills the destination.
__device__ __forceinline__ void cp16(uint32_t saddr, const void *gptr, bool valid) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(saddr), "l"(gptr), "r"(valid ? 16 : 0)
               : "memory");
}
__device__ __forceinline__ void cp4(uint32_t saddr, const void *gptr, bool valid) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::"r"(saddr), "l"(gptr), "r"(valid ? 4 : 0)
               : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }


// ---------------------------------------------------------------------------- bulk copies (TMA, 1-D)
__device__ __forceinline__ uint32_t smem_u32(const void *p) { return uint32_t(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void mbar_init(uint64_t *bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_inval(uint64_t *bar) {
  asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ int smid() {
  int id;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(id));
  return id;
}
__device__ __forceinline__ void mbar_expect(uint64_t *bar, int bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t *bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ bool mbar_test(uint64_t *bar, int phase) {   // non-blocking: has `phase` completed?
  uint32_t done;
  asm volatile("{ .reg .pred p; mbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
               : "=r"(done) : "r"(smem_u32(bar)), "r"(phase) : "memory");
  return done != 0;
}
__device__ __forceinline__ void mbar_wait(uint64_t *bar, int phase) {
  asm volatile(
      "{ .reg .pred done; retry: mbarrier.try_wait.parity.shared::cta.b64 done, [%0], %1; @!done bra retry; }"
      ::"r"(smem_u32(bar)), "r"(phase) : "memory");
}
// global -> shared, completing on `bar`; bytes and both addresses are multiples of 16
__device__ __forceinline__ void bulk_load(void *dst, const void *src, int bytes, uint64_t *bar) {
  if (bytes > 0)
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 ::"r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");
}
// shared -> global, tracked by the issuing thread's bulk groups
__device__ __forceinline__ void bulk_store(void *dst, const void *src, int bytes) {
  if (bytes > 0)
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(dst), "r"(smem_u32(src)), "r"(bytes)
                 : "memory");
}
__device__ __forceinline__ void bulk_commit() { asm volatile("cp.async.bulk.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ void bulk_wait_read() {
  asm volatile("cp.async.bulk.wait_group.read %0;" ::"n"(N) : "memory");
}
template <int N> __device__ __forceinline__ void bulk_wait() { asm volatile("cp.async.bulk.wait_group %0;" ::"n"(N) : "memory"); }
// generic-proxy shared-memory writes become visible to the async (bulk-copy) proxy
__device__ __forceinline__ void fence_async_smem() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }


// ---------------------------------------------------------------------------- cross-CTA flags (global)
// A producer publishes what it (or, ordered before it by a CTA barrier, its CTA) wrote with one flag; a
// consumer that acquires the flag may read it through either proxy.
// Flags are monotonic counters: publish sets `value`, acquire waits until the counter reaches `value`.
__device__ __forceinline__ void flag_publish(int *flag, int value) {
  asm volatile("fence.proxy.async.global;" ::: "memory");   // the thread's completed bulk stores, then
  asm volatile("st.release.gpu.global.b32 [%0], %1;" ::"l"(flag), "r"(value) : "memory");   // a release
}
__device__ __forceinline__ int ld_relaxed(const int *p) {
  int v;
  asm volatile("ld.relaxed.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void flag_acquire(const int *flag, int value) {
  int v;
  for (;;) {
    asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(flag) : "memory");
    if (v >= value) break;
    __nanosleep(256);
  }
  asm volatile("fence.proxy.async.global;" ::: "memory");
}


// ---------------------------------------------------------------------------- WGMMA (warpgroup MMA)
// One warpgroup (4 warps) computes D[64 x N] += A[64 x 16] B[16 x N]; warp w owns rows 16w..16w+15 and
// holds A in the mma.sync A-fragment layout, D in the mma.sync accumulator layout per n8 tile. B is read
// from shared memory through a descriptor. The bits equal the mma.sync chain over the same k16 steps
// (tests/test_mma_facts_gpu.py), which is what lets decode (mma.sync) and prefill (WGMMA) agree.
//
// K-major B (B[k][n] stored as [n][k] rows, e.g. W rows [i][d] for B = W^T) in a Tile<64> half:
// descriptor start at the half, + 32 bytes per k16 step; stride between 8-row groups 1024 bytes.
__device__ __forceinline__ uint64_t desc_kmajor(const void *half, int k0) {
  const uint64_t a = smem_u32(half) + 2 * k0;
  return ((a & 0x3FFFF) >> 4) | (uint64_t(1) << 16) | (uint64_t(1024 >> 4) << 32) | (uint64_t(1) << 62);
}
// MN-major B (B[k][n] stored as [k][n] rows, e.g. k^ rows [i][d] for B = K^) in a Wide tile whose halves
// are the two 64-column n blocks: + 16 rows (2 KB) per k16 step; stride between the halves ROWS * 128.
template <int ROWS>
__device__ __forceinline__ uint64_t desc_mnmajor(const Wide<ROWS> &t, int k0) {
  const uint64_t a = smem_u32(t.base) + k0 * 128;
  return ((a & 0x3FFFF) >> 4) | (uint64_t((ROWS * 128) >> 4) << 16) | (uint64_t(1024 >> 4) << 32) | (uint64_t(1) << 62);
}
// MN-major A (A[m][k] stored as [k][m] rows, e.g. k^ rows [i][d] read as A = k^T[d][i]) in one Tile<64>
// half (m = the 64 columns): + 16 rows (2 KB) per k16 step; one 128-byte swizzle atom spans all of m.
__device__ __forceinline__ uint64_t desc_mnmajor_a(const void *half, int k0) {
  const uint64_t a = smem_u32(half) + k0 * 128;
  return ((a & 0x3FFFF) >> 4) | (uint64_t((64 * 128 * 2) >> 4) << 16) | (uint64_t(1024 >> 4) << 32) | (uint64_t(1) << 62);
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(N) : "memory"); }
// Accumulators must not be touched between issue and wait; this keeps the compiler from moving them.
template <int N> __device__ __forceinline__ void wg_pin(float (&d)[N]) {
  #pragma unroll
  for (int e = 0; e < N; ++e) asm volatile("" : "+f"(d[e])::"memory");
}

// D[64 x 64] (+)= A[64 x 16] B[16 x 64], shared memory, B K-major; A K-major or MN-major.
template <int TA = 0>
__device__ __forceinline__ void wgmma_n64_ss(float (&d)[32], uint64_t adesc, uint64_t bdesc, bool acc) {
  asm volatile(
      "{ .reg .pred p; setp.ne.b32 p, %34, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"
      " %32, %33, p, 1, 1, %35, 0; }"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]),
        "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]),
        "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]),
        "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "l"(adesc), "l"(bdesc), "r"(int(acc)), "n"(TA));
}
// D[64 x 32] (+)= A[64 x 16] B[16 x 32], both in shared memory, B K-major; A K-major (TA = 0) or MN-major
// (TA = 1).
template <int TA>
__device__ __forceinline__ void wgmma_n32_ss(float (&d)[16], uint64_t adesc, uint64_t bdesc, bool acc) {
  asm volatile(
      "{ .reg .pred p; setp.ne.b32 p, %18, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, %16, %17, p, 1, 1, %19, 0; }"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]),
        "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
      : "l"(adesc), "l"(bdesc), "r"(int(acc)), "n"(TA));
}
// D[64 x 16] (+)= A[64 x 16] B[16 x 16], as wgmma_n32_ss.
template <int TA>
__device__ __forceinline__ void wgmma_n16_ss(float (&d)[8], uint64_t adesc, uint64_t bdesc, bool acc) {
  asm volatile(
      "{ .reg .pred p; setp.ne.b32 p, %10, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, %8, %9, p, 1, 1, %11, 0; }"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
      : "l"(adesc), "l"(bdesc), "r"(int(acc)), "n"(TA));
}
// m64 x N (N = 16, 32, 64) by the width's instruction
template <int N, int TA, int NR>
__device__ __forceinline__ void wgmma_ss(float (&d)[NR], uint64_t adesc, uint64_t bdesc, bool acc) {
  static_assert(NR == N / 2, "accumulator registers: N / 2");
  static_assert(N == 16 || N == 32 || N == 64, "supported WGMMA width");
  if constexpr (N == 64) wgmma_n64_ss<TA>(d, adesc, bdesc, acc);
  else if constexpr (N == 32) wgmma_n32_ss<TA>(d, adesc, bdesc, acc);
  else wgmma_n16_ss<TA>(d, adesc, bdesc, acc);
}
// D[64 x 128] (+)= A[64 x 16] B[16 x 128], both in shared memory, A K-major; B K-major (TB = 0) or MN-major
// (TB = 1).
template <int TB = 0>
__device__ __forceinline__ void wgmma_n128_ss(float (&d)[64], uint64_t adesc, uint64_t bdesc, bool acc) {
  asm volatile(
      "{ .reg .pred p; setp.ne.b32 p, %66, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
      "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63},"
      " %64, %65, p, 1, 1, 0, %67; }"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]),
        "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]),
        "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]),
        "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]),
        "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]),
        "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]),
        "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]),
        "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
      : "l"(adesc), "l"(bdesc), "r"(int(acc)), "n"(TB));
}

}  // namespace gdn_cpr
