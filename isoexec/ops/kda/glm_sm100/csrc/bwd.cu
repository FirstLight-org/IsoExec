// SPDX-License-Identifier: Apache-2.0
// The parallel backward kernels, one CTA per (chunk, head), 256 threads, no atomics (every output element
// has exactly one writer and every sum is a fixed-order chain, tree or mma chain):
//
//   bwd_inter   contractions over the value columns with the chunk's states:
//                 dw = -dd S^T (bf16)   dqg = dO S^T   dkd = d dS_{c+1}^T   dMq = dO d^T (i <= j)
//                 dGLs_k = sum_v S^_kv dS_{c+1,kv}
//   bwd_wy      from phase 1's saved X, xu, kg (no recompute), with Y = X diag(beta):
//                 dY = dd v^T + dw kg^T,  dv = Y^T dd (OUTPUT),  dkg = Y^T dw,
//                 dX = dY diag(beta),  dL = -X^T dX X^T,  dKK = diag(beta) dL,  dbeta (partial)
//               the two 64^3 products of dL by mma with hi/lo-split bf16 operands for X and dX X^T.
//   bwd_qkg     recompute k^, q^, G; the decay-weighted products' backward with A = dKK, B = dMq:
//                 Rk_j = sum_i A_ji k_i e^(G_j - G_i),  Rq_j = sum_i B_ji k_i e^(..)
//                 Ck_i = sum_j (A_ji k_j + B_ji q_j) e^(G_j - G_i)
//               off-diagonal blocks through the forward's block references (mma), diagonal blocks by the
//               per-token product chains of delta (as the forward), then the qg / kg / kd / E(G_L) terms,
//               the reverse cumulative sum to dg, the l2 norm backward to dq, dk, and dbeta.
#include <mutex>
#include <cuda_runtime.h>

#include <type_traits>

#include "intra_common.cuh"
#include "params.cuh"

namespace kda_sm100 {

namespace {

// Store one warp's [16 x 8 NTILE] fp32 accumulator block to a token-indexed global matrix (rows < r).
template <int NTILE>
__device__ __forceinline__ void store_rows_f32(float *dst, const float (&acc)[NTILE][4], int t0, int r, int m0, int n0,
                                               int64_t rowstride, int64_t off, int lane) {
  const int g = lane >> 2, t = lane & 3;
  #pragma unroll
  for (int nt = 0; nt < NTILE; ++nt)
    #pragma unroll
    for (int eh = 0; eh < 2; ++eh) {
      const int row = m0 + g + 8 * eh, col = n0 + 8 * nt + 2 * t;
      if (row < r)
        *reinterpret_cast<float2 *>(dst + (int64_t(t0) + row) * rowstride + off + col) =
            make_float2(acc[nt][2 * eh], acc[nt][2 * eh + 1]);
    }
}

// Store one warp's [16 x 8 NTILE] fp32 accumulator block as bf16 rows (rows < r).
template <int NTILE>
__device__ __forceinline__ void store_rows_bf16(uint16_t *dst, const float (&acc)[NTILE][4], int t0, int r, int m0,
                                                int n0, int64_t rowstride, int64_t off, int lane) {
  const int g = lane >> 2, t = lane & 3;
  #pragma unroll
  for (int nt = 0; nt < NTILE; ++nt)
    #pragma unroll
    for (int eh = 0; eh < 2; ++eh) {
      const int row = m0 + g + 8 * eh, col = n0 + 8 * nt + 2 * t;
      if (row < r)
        *reinterpret_cast<uint32_t *>(dst + (int64_t(t0) + row) * rowstride + off + col) =
            rn2(acc[nt][2 * eh], acc[nt][2 * eh + 1]);
    }
}

// cp.async a [64][64] bf16 row block into a Tile<64, 64>. No commit.
__device__ __forceinline__ void load_tile64(const T64 &dst, const uint16_t *src, int t0, int r, int64_t rowstride,
                                            int64_t off, int tid) {
  #pragma unroll
  for (int it = 0; it < 2; ++it) {
    const int idx = it * NT + tid, row = idx >> 3, ch = idx & 7;
    const bool valid = row < r;
    cp16(dst.saddr(row, ch), src + (int64_t(t0) + (valid ? row : 0)) * rowstride + off + ch * 8, valid);
  }
}
// cp.async a [64][64] fp32 row block into a [64][LS] smem matrix. No commit.
__device__ __forceinline__ void load_f32x64(float *dst, const float *src, int t0, int r, int64_t rowstride,
                                            int64_t off, int tid) {
  #pragma unroll
  for (int it = 0; it < 4; ++it) {
    const int idx = it * NT + tid, row = idx >> 4, ch = idx & 15;
    const bool valid = row < r;
    cp16(smem_u32(dst + row * LS + 4 * ch), src + (int64_t(t0) + (valid ? row : 0)) * rowstride + off + 4 * ch, valid);
  }
}
// hi / lo bf16 split of an fp32 value pair: hi = rn(x), lo = rn(x - bf(hi)).
__device__ __forceinline__ void split2(float a, float b, uint32_t &hi, uint32_t &lo) {
  hi = rn2(a, b);
  lo = rn2(fsub(a, bf(lo16(hi))), fsub(b, bf(hi16(hi))));
}

}  // namespace

// ============================================================================ bwd_inter
namespace inter {
constexpr int OFF_S = 0;            // Tile<128 k, 128 v> S^
constexpr int OFF_DSN = 32768;      // Tile<128 k, 128 v> rn(dS_{c+1})
constexpr int OFF_DO = 65536;       // Tile<64, 128> dO
constexpr int OFF_D = 81920;        // Tile<64, 128> d
constexpr int OFF_DD = 98304;       // Tile<64, 128> rn(dd)
constexpr int SMEM = 114688;
}  // namespace inter

__global__ void __launch_bounds__(NT, 2) bwd_inter_kernel(BwdParams p) {
  extern __shared__ __align__(128) char smem[];
  using namespace inter;
  const int chunk = blockIdx.x, h = blockIdx.y;
  if (chunk >= p.plan.count[0]) return;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int t0 = p.plan.chunk_tok[chunk], r = p.plan.chunk_rows[chunk];
  const int H = p.H;
  const int64_t rowH = int64_t(H) * D, hoff = int64_t(h) * D;
  const Tile<128, 128> Sh{reinterpret_cast<uint16_t *>(smem + OFF_S)};
  const Tile<128, 128> dSn{reinterpret_cast<uint16_t *>(smem + OFF_DSN)};
  const T128 DO{reinterpret_cast<uint16_t *>(smem + OFF_DO)};
  const T128 Dt{reinterpret_cast<uint16_t *>(smem + OFF_D)};
  const T128 DD{reinterpret_cast<uint16_t *>(smem + OFF_DD)};

  const uint16_t *sS = p.S + (int64_t(chunk) * H + h) * D * D;
  const uint16_t *sN = p.dSn + (int64_t(chunk) * H + h) * D * D;
  #pragma unroll
  for (int it = 0; it < 8; ++it) {
    const int idx = it * NT + tid, row = idx >> 4, ch = idx & 15;
    cp16(Sh.saddr(row, ch), sS + row * D + ch * 8, true);
    cp16(dSn.saddr(row, ch), sN + row * D + ch * 8, true);
  }
  load_tile128(DO, p.dO, t0, r, rowH, hoff, tid);
  load_tile128(Dt, p.d, t0, r, rowH, hoff, tid);
  load_tile128(DD, p.dd, t0, r, rowH, hoff, tid);
  cp_commit();
  cp_wait<0>();
  __syncthreads();

  const int mt = warp & 3, nh = warp >> 2, g = lane >> 2, t = lane & 3;
  {   // dw = -dd S^T (bf16 out), dqg = dO S^T: rows j in block mt, key columns 64 nh .. + 63
    float aw[8][4] = {}, aq[8][4] = {};
    #pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      uint32_t an[4], ao[4];
      frag_a(an, DD, 16 * mt, 16 * ks, lane);
      #pragma unroll
      for (int e = 0; e < 4; ++e) an[e] ^= 0x80008000u;   // -rn(dd)
      frag_a(ao, DO, 16 * mt, 16 * ks, lane);
      #pragma unroll
      for (int np = 0; np < 4; ++np) {
        uint32_t b[4];
        frag_b2(b, Sh, 64 * nh + 16 * np, 16 * ks, lane);
        mma(aw[2 * np], an, b[0], b[1]);
        mma(aw[2 * np + 1], an, b[2], b[3]);
        mma(aq[2 * np], ao, b[0], b[1]);
        mma(aq[2 * np + 1], ao, b[2], b[3]);
      }
    }
    #pragma unroll
    for (int nt = 0; nt < 8; ++nt)
      #pragma unroll
      for (int eh = 0; eh < 2; ++eh) {
        const int row = 16 * mt + g + 8 * eh, col = 64 * nh + 8 * nt + 2 * t;
        if (row < r)
          *reinterpret_cast<uint32_t *>(p.dW + (int64_t(t0) + row) * rowH + hoff + col) =
              rn2(aw[nt][2 * eh], aw[nt][2 * eh + 1]);
      }
    store_rows_bf16<8>(p.dQG, aq, t0, r, 16 * mt, 64 * nh, rowH, hoff, lane);
  }
  {   // dkd = d dS_{c+1}^T
    float ak[8][4] = {};
    #pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      uint32_t a[4];
      frag_a(a, Dt, 16 * mt, 16 * ks, lane);
      #pragma unroll
      for (int np = 0; np < 4; ++np) {
        uint32_t b[4];
        frag_b2(b, dSn, 64 * nh + 16 * np, 16 * ks, lane);
        mma(ak[2 * np], a, b[0], b[1]);
        mma(ak[2 * np + 1], a, b[2], b[3]);
      }
    }
    store_rows_bf16<8>(p.dKD, ak, t0, r, 16 * mt, 64 * nh, rowH, hoff, lane);
  }
  {   // dMq = dO d^T (i <= j, +0 above): rows j in block mt, columns i 32 nh .. + 31
    float am[4][4] = {};
    #pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      uint32_t a[4];
      frag_a(a, DO, 16 * mt, 16 * ks, lane);
      #pragma unroll
      for (int np = 0; np < 2; ++np) {
        uint32_t b[4];
        frag_b2(b, Dt, 32 * nh + 16 * np, 16 * ks, lane);
        mma(am[2 * np], a, b[0], b[1]);
        mma(am[2 * np + 1], a, b[2], b[3]);
      }
    }
    #pragma unroll
    for (int nt = 0; nt < 4; ++nt)
      #pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int j = 16 * mt + g + 8 * (e >> 1), i = 32 * nh + 8 * nt + 2 * t + (e & 1);
        if (i > j) am[nt][e] = 0.f;
      }
    store_rows_f32<4>(p.dMQ, am, t0, r, 16 * mt, 32 * nh, int64_t(H) * 64, int64_t(h) * 64, lane);
  }
  if (tid < D) {   // dGLs_k = sum_v S^_kv dS^_kv, ascending v
    float acc = 0.f;
    #pragma unroll 8
    for (int v = 0; v < D; ++v) acc = ffma(bf(Sh.at(tid, v)), bf(dSn.at(tid, v)), acc);
    p.dGLs[(int64_t(chunk) * H + h) * D + tid] = acc;
  }
}

// ============================================================================ bwd_wy
namespace wy {
constexpr int OFF_KG = 0;                        // Tile<64,128> kg [i][c]        ; then dX Tile<64,64>
constexpr int OFF_V = OFF_KG + 16384;            // Tile<64,128> v [i][v]
constexpr int OFF_DD = OFF_V + 16384;            // Tile<64,128> rn(dd) [j][v]    ; then PMh, PMl Tile<64,64>
constexpr int OFF_DW = OFF_DD + 16384;           // Tile<64,128> rn(dw) [j][c]
constexpr int OFF_XU = OFF_DW + 16384;           // Tile<64,64> xu [j][i]
constexpr int OFF_XH = OFF_XU + 8192;            // Tile<64,64> rn(X) [j][i]
constexpr int OFF_XL = OFF_XH + 8192;            // Tile<64,64> rn(X - rn(X))
constexpr int OFF_DY = OFF_XL + 8192;            // fp32 [64][LS] dY ; then dL
constexpr int OFF_BETA = OFF_DY + 64 * LS * 4;   // float [64]
constexpr int OFF_DB = OFF_BETA + 256;           // float [64] dbeta part 1
constexpr int SMEM = OFF_DB + 256;
}  // namespace wy

__global__ void __launch_bounds__(NT, 2) bwd_wy_kernel(BwdParams p) {
  extern __shared__ __align__(128) char smem[];
  using namespace wy;
  const int chunk = blockIdx.x, h = blockIdx.y;
  if (chunk >= p.plan.count[0]) return;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int t0 = p.plan.chunk_tok[chunk], r = p.plan.chunk_rows[chunk];
  const int H = p.H;
  const int64_t rowH = int64_t(H) * D, hoff = int64_t(h) * D, rH64 = int64_t(H) * 64, h64 = int64_t(h) * 64;
  const T128 Kg{reinterpret_cast<uint16_t *>(smem + OFF_KG)};
  const T128 Vt{reinterpret_cast<uint16_t *>(smem + OFF_V)};
  const T128 DU{reinterpret_cast<uint16_t *>(smem + OFF_DD)};
  const T128 DWt{reinterpret_cast<uint16_t *>(smem + OFF_DW)};
  const T64 Xu{reinterpret_cast<uint16_t *>(smem + OFF_XU)};
  const T64 XH{reinterpret_cast<uint16_t *>(smem + OFF_XH)};
  const T64 XL{reinterpret_cast<uint16_t *>(smem + OFF_XL)};
  float *DY = reinterpret_cast<float *>(smem + OFF_DY);
  float *beta = reinterpret_cast<float *>(smem + OFF_BETA);
  float *db1 = reinterpret_cast<float *>(smem + OFF_DB);

  load_tile128(Kg, p.Kg, t0, r, rowH, hoff, tid);
  load_tile128(Vt, p.v, t0, r, rowH, hoff, tid);
  load_tile128(DU, p.dd, t0, r, rowH, hoff, tid);
  load_tile128(DWt, p.dW, t0, r, rowH, hoff, tid);
  load_tile64(Xu, p.Xu, t0, r, rH64, h64, tid);
  cp_commit();
  if (tid < 64) beta[tid] = tid < r ? ldg_val(p.beta, (int64_t(t0) + tid) * H + h, p.beta_bf16) : 0.f;
  #pragma unroll 4
  for (int it = 0; it < 8; ++it) {   // X -> hi / lo tiles
    const int idx = it * NT + tid, j = idx >> 5, i0 = 2 * (idx & 31);
    float2 x = make_float2(0.f, 0.f);
    if (j < r) x = *reinterpret_cast<const float2 *>(p.X + (int64_t(t0) + j) * rH64 + h64 + i0);
    uint32_t hi, lo;
    split2(x.x, x.y, hi, lo);
    *reinterpret_cast<uint32_t *>(&XH.at(j, i0)) = hi;
    *reinterpret_cast<uint32_t *>(&XL.at(j, i0)) = lo;
  }
  cp_wait<0>();
  __syncthreads();

  const int mt = warp & 3, nh = warp >> 2;
  {   // dY = dd v^T + dw kg^T: rows j in block mt, columns i 32 nh .. + 31
    float acc[4][4] = {};
    #pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      uint32_t a[4], a2[4];
      frag_a(a, DU, 16 * mt, 16 * ks, lane);
      frag_a(a2, DWt, 16 * mt, 16 * ks, lane);
      #pragma unroll
      for (int np = 0; np < 2; ++np) {
        uint32_t b[4], b2[4];
        frag_b2(b, Vt, 32 * nh + 16 * np, 16 * ks, lane);
        frag_b2(b2, Kg, 32 * nh + 16 * np, 16 * ks, lane);
        mma(acc[2 * np], a, b[0], b[1]);
        mma(acc[2 * np + 1], a, b[2], b[3]);
        mma(acc[2 * np], a2, b2[0], b2[1]);
        mma(acc[2 * np + 1], a2, b2[2], b2[3]);
      }
    }
    #pragma unroll
    for (int nt = 0; nt < 4; ++nt)
      #pragma unroll
      for (int e = 0; e < 4; ++e)
        DY[(16 * mt + g + 8 * (e >> 1)) * LS + 32 * nh + 8 * nt + 2 * t + (e & 1)] = acc[nt][e];
  }
  {   // dv = Y^T dd (output) and dkg = Y^T dw: rows i in block mt, columns 64 nh .. + 63
    float av[8][4] = {}, ak[8][4] = {};
    #pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      if (ks < mt) continue;                     // Y_ji = 0 for j < i
      uint32_t a[4];
      frag_a_t(a, Xu, 16 * mt, 16 * ks, lane);
      #pragma unroll
      for (int np = 0; np < 4; ++np) {
        uint32_t b[4], b2[4];
        frag_b2_t(b, DU, 64 * nh + 16 * np, 16 * ks, lane);
        frag_b2_t(b2, DWt, 64 * nh + 16 * np, 16 * ks, lane);
        mma(av[2 * np], a, b[0], b[1]);
        mma(av[2 * np + 1], a, b[2], b[3]);
        mma(ak[2 * np], a, b2[0], b2[1]);
        mma(ak[2 * np + 1], a, b2[2], b2[3]);
      }
    }
    store_rows_bf16<8>(p.dkg, ak, t0, r, 16 * mt, 64 * nh, rowH, hoff, lane);
    #pragma unroll
    for (int nt = 0; nt < 8; ++nt)
      #pragma unroll
      for (int eh = 0; eh < 2; ++eh) {
        const int row = 16 * mt + g + 8 * eh, col = 64 * nh + 8 * nt + 2 * t;
        if (row < r)
          *reinterpret_cast<uint32_t *>(p.dv + (int64_t(t0) + row) * rowH + hoff + col) =
              rn2(av[nt][2 * eh], av[nt][2 * eh + 1]);
      }
  }
  __syncthreads();
  // dbeta part 1 = sum_{j >= i} dY_ji X_ji ; dX = rn(dY_ji beta_i) (j >= i) into kg's space
  const T64 DX{reinterpret_cast<uint16_t *>(smem + OFF_KG)};
  if (tid < 64) {
    float acc = 0.f;
    for (int j = tid; j < 64; ++j) acc = ffma(DY[j * LS + tid], fadd(bf(XH.at(j, tid)), bf(XL.at(j, tid))), acc);
    db1[tid] = acc;
  }
  for (int idx = tid; idx < 64 * 32; idx += NT) {
    const int j = idx >> 5, i0 = 2 * (idx & 31);
    const float x0 = i0 <= j ? fmul(DY[j * LS + i0], beta[i0]) : 0.f;
    const float x1 = i0 + 1 <= j ? fmul(DY[j * LS + i0 + 1], beta[i0 + 1]) : 0.f;
    *reinterpret_cast<uint32_t *>(&DX.at(j, i0)) = rn2(x0, x1);
  }
  __syncthreads();
  // PM = dX X^T: rows j in block mt, columns m 32 nh .. + 31 ; split into hi / lo tiles in dd's space
  const T64 PMh{reinterpret_cast<uint16_t *>(smem + OFF_DD)};
  const T64 PMl{reinterpret_cast<uint16_t *>(smem + OFF_DD + 8192)};
  {
    float acc[4][4] = {};
    #pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      if (ks > mt) break;                        // dX_ji = 0 for i > j
      uint32_t a[4];
      frag_a(a, DX, 16 * mt, 16 * ks, lane);
      #pragma unroll
      for (int np = 0; np < 2; ++np) {
        uint32_t bh[4], bl[4];
        frag_b2(bh, XH, 32 * nh + 16 * np, 16 * ks, lane);
        frag_b2(bl, XL, 32 * nh + 16 * np, 16 * ks, lane);
        mma(acc[2 * np], a, bh[0], bh[1]);
        mma(acc[2 * np + 1], a, bh[2], bh[3]);
        mma(acc[2 * np], a, bl[0], bl[1]);
        mma(acc[2 * np + 1], a, bl[2], bl[3]);
      }
    }
    #pragma unroll
    for (int np = 0; np < 2; ++np) {
      uint32_t wh[4], wl[4];
      #pragma unroll
      for (int h2 = 0; h2 < 2; ++h2)
        #pragma unroll
        for (int eh = 0; eh < 2; ++eh)
          split2(acc[2 * np + h2][2 * eh], acc[2 * np + h2][2 * eh + 1], wh[2 * h2 + eh], wl[2 * h2 + eh]);
      store_acc(PMh, 16 * mt, 32 * nh + 16 * np, wh, lane);
      store_acc(PMl, 16 * mt, 32 * nh + 16 * np, wl, lane);
    }
  }
  __syncthreads();
  // dL = -X^T PM (a > b): rows a in block mt, columns b 32 nh .. + 31 ; dKK = beta_a dL ; dL into DY
  {
    float acc[4][4] = {};
    #pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      if (ks < mt) continue;                     // X_ja = 0 for j < a
      uint32_t ah[4], al[4];
      frag_a_t(ah, XH, 16 * mt, 16 * ks, lane);
      frag_a_t(al, XL, 16 * mt, 16 * ks, lane);
      #pragma unroll
      for (int np = 0; np < 2; ++np) {
        uint32_t bh[4], bl[4];
        frag_b2_t(bh, PMh, 32 * nh + 16 * np, 16 * ks, lane);
        frag_b2_t(bl, PMl, 32 * nh + 16 * np, 16 * ks, lane);
        mma(acc[2 * np], ah, bh[0], bh[1]);
        mma(acc[2 * np + 1], ah, bh[2], bh[3]);
        mma(acc[2 * np], ah, bl[0], bl[1]);
        mma(acc[2 * np + 1], ah, bl[2], bl[3]);
        mma(acc[2 * np], al, bh[0], bh[1]);
        mma(acc[2 * np + 1], al, bh[2], bh[3]);
      }
    }
    #pragma unroll
    for (int nt = 0; nt < 4; ++nt)
      #pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int a = 16 * mt + g + 8 * (e >> 1), b = 32 * nh + 8 * nt + 2 * t + (e & 1);
        const float dl = b < a ? -acc[nt][e] : 0.f;
        DY[a * LS + b] = dl;
        if (a < r) p.dKK[(int64_t(t0) + a) * rH64 + h64 + b] = fmul(beta[a], dl);
      }
  }
  __syncthreads();
  if (tid < r) {   // dbeta_a = part 1 + sum_{b < a} dL_ab KK_ab
    float acc = db1[tid];
    const float *kk = p.KK + (int64_t(t0) + tid) * rH64 + h64;
    for (int b = 0; b < tid; ++b) acc = ffma(DY[tid * LS + b], kk[b], acc);
    p.dbeta_p[(int64_t(t0) + tid) * H + h] = acc;
  }
}

// ============================================================================ bwd_qkg
namespace qkg {
constexpr int OFF_KH = 0;                        // k^
constexpr int OFF_QH = OFF_KH + 16384;           // q^
constexpr int OFF_G = OFF_QH + 16384;            // G fp32 [64][128] ; then delta
constexpr int OFF_W1 = OFF_G + 32768;            // dKK fp32 [64][LS]
constexpr int OFF_W2 = OFF_W1 + 64 * LS * 4;     // dMq fp32 [64][LS]
constexpr int OFF_W1B = OFF_W2 + 64 * LS * 4;    // rn(dKK) Tile<64, 64>
constexpr int OFF_W2B = OFF_W1B + 8192;          // rn(dMq) Tile<64, 64>
constexpr int OFF_KI1 = OFF_W2B + 8192;          // Tile<16, 128> ki^1 ; raw bf16 g lives here first
constexpr int OFF_KI2 = OFF_KI1 + 4096;          // Tile<32, 128> ki^2
constexpr int OFF_KI3 = OFF_KI2 + 8192;          // Tile<48, 128> ki^3
constexpr int OFF_KR = OFF_KI3 + 12288;          // Tile<64, 128> kr (rows >= 16)
constexpr int OFF_QR = OFF_KR + 16384;           // Tile<64, 128> qr
constexpr int OFF_ST = OFF_QR + 16384;           // fp32 [64][128] staging
constexpr int OFF_INV = OFF_ST + 32768;          // inverse norms rk[64], rq[64]
constexpr int OFF_DGL = OFF_INV + 512;           // dGL [128]
constexpr int OFF_GL = OFF_DGL + 512;            // G_L [128]
constexpr int SMEM = OFF_GL + 512;
static_assert(OFF_KI1 + 16384 <= OFF_ST, "raw bf16 g fits in the factored tiles' space");
}  // namespace qkg

__global__ void __launch_bounds__(NT, 1) bwd_qkg_kernel(BwdParams p) {
  extern __shared__ __align__(128) char smem[];
  using namespace qkg;
  const int chunk = blockIdx.x, h = blockIdx.y;
  if (chunk >= p.plan.count[0]) return;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int t0 = p.plan.chunk_tok[chunk], r = p.plan.chunk_rows[chunk];
  
  const int H = p.H;
  const int64_t rowH = int64_t(H) * D, hoff = int64_t(h) * D, rH64 = int64_t(H) * 64, h64 = int64_t(h) * 64;
  const T128 Kh{reinterpret_cast<uint16_t *>(smem + OFF_KH)};
  const T128 Qh{reinterpret_cast<uint16_t *>(smem + OFF_QH)};
  float *G = reinterpret_cast<float *>(smem + OFF_G);
  float *W1 = reinterpret_cast<float *>(smem + OFF_W1);
  float *W2 = reinterpret_cast<float *>(smem + OFF_W2);
  const T64 W1b{reinterpret_cast<uint16_t *>(smem + OFF_W1B)};
  const T64 W2b{reinterpret_cast<uint16_t *>(smem + OFF_W2B)};
  const Tile<16, 128> Ki1{reinterpret_cast<uint16_t *>(smem + OFF_KI1)};
  const Tile<32, 128> Ki2{reinterpret_cast<uint16_t *>(smem + OFF_KI2)};
  const Tile<48, 128> Ki3{reinterpret_cast<uint16_t *>(smem + OFF_KI3)};
  const T128 Kr{reinterpret_cast<uint16_t *>(smem + OFF_KR)};
  const T128 Qr{reinterpret_cast<uint16_t *>(smem + OFF_QR)};
  float *ST = reinterpret_cast<float *>(smem + OFF_ST);
  float *rk = reinterpret_cast<float *>(smem + OFF_INV);
  float *rq = rk + 64;
  float *dGL = reinterpret_cast<float *>(smem + OFF_DGL);
  float *GLs = reinterpret_cast<float *>(smem + OFF_GL);
  // the gate-term gradients, bf16 rows [64][128], prefetched into the factored tiles' space once it is dead
  const uint16_t *gQG = reinterpret_cast<const uint16_t *>(smem + OFF_KI1);
  const uint16_t *gKG = gQG + 64 * D;
  const uint16_t *gKD = gKG + 64 * D;
  static_assert(OFF_KI1 + 3 * 64 * D * 2 <= OFF_ST, "gate-term rows fit in the factored tiles' space");

  // ---- P0: loads (all asynchronous)
  load_tile128(Kh, p.k, t0, r, rowH, hoff, tid);
  load_tile128(Qh, p.q, t0, r, rowH, hoff, tid);
  load_g(p.g, p.g_bf16, G, reinterpret_cast<uint16_t *>(smem + OFF_KI1), t0, r, rowH, hoff, tid);
  load_f32x64(W1, p.dKK, t0, r, rH64, h64, tid);
  load_f32x64(W2, p.dMQ, t0, r, rH64, h64, tid);
  cp_commit();
  cp_wait<0>();
  __syncthreads(); 
  // ---- P1: k^, q^ (with inverse norms), G, bf16 weights
  l2norm_tile(Kh, false, rk, warp, lane);
  l2norm_tile(Qh, true, rq, warp, lane);
  g_chain(G, reinterpret_cast<const uint16_t *>(smem + OFF_KI1), p.g_bf16, tid);
  for (int idx = tid; idx < 64 * 32; idx += NT) {
    const int j = idx >> 5, i0 = 2 * (idx & 31);
    *reinterpret_cast<uint32_t *>(&W1b.at(j, i0)) = rn2(W1[j * LS + i0], W1[j * LS + i0 + 1]);
    *reinterpret_cast<uint32_t *>(&W2b.at(j, i0)) = rn2(W2[j * LS + i0], W2[j * LS + i0 + 1]);
  }
  __syncthreads(); 
  // ---- P2: decay-factored operand tiles (bf16): ki^b, kr, qr ; G_L
  #pragma unroll 2
  for (int it = 0; it < 16; ++it) {
    const int task = it * NT + tid, j = task >> 6, c0 = 2 * (task & 63);
    const uint32_t kp = *reinterpret_cast<const uint32_t *>(&Kh.at(j, c0));
    const float2 Gj = *reinterpret_cast<const float2 *>(G + j * D + c0);
    if (j >= 16) {
      const uint32_t qp = *reinterpret_cast<const uint32_t *>(&Qh.at(j, c0));
      const float2 Gn = *reinterpret_cast<const float2 *>(G + (j & ~15) * D + c0);
      const float e0 = decay(Gj.x, Gn.x), e1 = decay(Gj.y, Gn.y);
      *reinterpret_cast<uint32_t *>(&Kr.at(j, c0)) = scale2e(kp, e0, e1);
      *reinterpret_cast<uint32_t *>(&Qr.at(j, c0)) = scale2e(qp, e0, e1);
    }
    #pragma unroll
    for (int b = 1; b <= 3; ++b)
      if (j < 16 * b) {
        const float2 Gb = *reinterpret_cast<const float2 *>(G + 16 * b * D + c0);
        const uint32_t val = scale2(kp, Gb, Gj);
        uint16_t *dst = b == 1 ? &Ki1.at(j, c0) : (b == 2 ? &Ki2.at(j, c0) : &Ki3.at(j, c0));
        *reinterpret_cast<uint32_t *>(dst) = val;
      }
  }
  if (tid < D) {
    GLs[tid] = G[(r - 1) * D + tid];
    dGL[tid] = 0.f;
  }
  __syncthreads(); 

  // ---- owner accumulators: row x = tid >> 2, channel quarter qp: c = 16 m + 4 qp + e
  const int x = tid >> 2, qp = tid & 3;
  float dq_r[32], dk_r[32], dG_r[32];
  #pragma unroll
  for (int e = 0; e < 32; ++e) dq_r[e] = dk_r[e] = dG_r[e] = 0.f;
  auto row4 = [&](const T128 &tt, int row, int m, float (&o)[4]) {
    const uint2 w2 = *reinterpret_cast<const uint2 *>(&tt.at(row, 16 * m + 4 * qp));
    o[0] = bf(lo16(w2.x)); o[1] = bf(hi16(w2.x)); o[2] = bf(lo16(w2.y)); o[3] = bf(hi16(w2.y));
  };

  // ---- P3: off-diagonal R (rows j in blocks 1..3), staged, owners combine
  #pragma unroll 1
  for (int which = 0; which < 2; ++which) {           // 0: Rk (A = dKK), 1: Rq (B = dMq)
    const T64 &Wb = which == 0 ? W1b : W2b;
    for (int task = warp; task < 24; task += 8) {     // (j-block 1..3) x 8 column pairs
      const int bj = 1 + task / 8, c0 = 16 * (task % 8);
      float acc[2][4] = {};
      for (int ks = 0; ks < bj; ++ks) {
        uint32_t a[4], b[4];
        frag_a(a, Wb, 16 * bj, 16 * ks, lane);
        if (bj == 1) frag_b2_t(b, Ki1, c0, 16 * ks, lane);
        else if (bj == 2) frag_b2_t(b, Ki2, c0, 16 * ks, lane);
        else frag_b2_t(b, Ki3, c0, 16 * ks, lane);
        mma(acc[0], a, b[0], b[1]);
        mma(acc[1], a, b[2], b[3]);
      }
      #pragma unroll
      for (int nt = 0; nt < 2; ++nt)
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int j = 16 * bj + g + 8 * (e >> 1), c = c0 + 8 * nt + 2 * t + (e & 1);
          ST[j * D + c] = fmul(acc[nt][e], decay(G[j * D + c], G[16 * bj * D + c]));
        }
    }
    __syncthreads(); 
    if (x >= 16) {
      #pragma unroll
      for (int m = 0; m < 8; ++m) {
        const float4 s4 = *reinterpret_cast<const float4 *>(ST + x * D + 16 * m + 4 * qp);
        const float sv[4] = {s4.x, s4.y, s4.z, s4.w};
        float xv[4];
        row4(which == 0 ? Kh : Qh, x, m, xv);
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
          if (which == 0) dk_r[4 * m + e] = fadd(dk_r[4 * m + e], sv[e]);
          else dq_r[4 * m + e] = fadd(dq_r[4 * m + e], sv[e]);
          dG_r[4 * m + e] = ffma(xv[e], sv[e], dG_r[4 * m + e]);
        }
      }
    }
    __syncthreads(); 
  }
  // ---- off-diagonal C (rows i in blocks 0..2): sum over j-blocks bj > bi, ascending
  for (int task = warp; task < 24; task += 8) {       // (i-block 0..2) x 8 column pairs
    const int bi = task / 8, c0 = 16 * (task % 8);
    float tot[2][4] = {};
    for (int bj = bi + 1; bj < 4; ++bj) {
      float acc[2][4] = {};
      uint32_t a[4], a2[4], b[4], b2[4];
      frag_a_t(a, W1b, 16 * bi, 16 * bj, lane);
      frag_a_t(a2, W2b, 16 * bi, 16 * bj, lane);
      frag_b2_t(b, Kr, c0, 16 * bj, lane);
      frag_b2_t(b2, Qr, c0, 16 * bj, lane);
      mma(acc[0], a, b[0], b[1]);
      mma(acc[1], a, b[2], b[3]);
      mma(acc[0], a2, b2[0], b2[1]);
      mma(acc[1], a2, b2[2], b2[3]);
      #pragma unroll
      for (int nt = 0; nt < 2; ++nt)
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int i = 16 * bi + g + 8 * (e >> 1), c = c0 + 8 * nt + 2 * t + (e & 1);
          tot[nt][e] = ffma(acc[nt][e], decay(G[16 * bj * D + c], G[i * D + c]), tot[nt][e]);
        }
    }
    #pragma unroll
    for (int nt = 0; nt < 2; ++nt)
      #pragma unroll
      for (int e = 0; e < 4; ++e)
        ST[(16 * bi + g + 8 * (e >> 1)) * D + c0 + 8 * nt + 2 * t + (e & 1)] = tot[nt][e];
  }
  __syncthreads(); 
  // the factored tiles are dead: prefetch the gate-term gradient rows behind the C combine
  #pragma unroll
  for (int it = 0; it < 12; ++it) {
    const int idx = it * NT + tid, which = idx >> 10, row = (idx >> 4) & 63, ch = idx & 15;
    const bool valid = row < r;
    const uint16_t *src = (which == 0 ? p.dQG : (which == 1 ? p.dkg : p.dKD)) +
                          (int64_t(t0) + (valid ? row : 0)) * rowH + hoff + ch * 8;
    cp16(smem_u32(gQG + which * 64 * D + row * D + ch * 8), src, valid);
  }
  cp_commit();
  if (x < 48) {
    #pragma unroll
    for (int m = 0; m < 8; ++m) {
      const float4 s4 = *reinterpret_cast<const float4 *>(ST + x * D + 16 * m + 4 * qp);
      const float sv[4] = {s4.x, s4.y, s4.z, s4.w};
      float kv[4];
      row4(Kh, x, m, kv);
      #pragma unroll
      for (int e = 0; e < 4; ++e) {
        dk_r[4 * m + e] = fadd(dk_r[4 * m + e], sv[e]);
        dG_r[4 * m + e] = fsub(dG_r[4 * m + e], fmul(kv[e], sv[e]));
      }
    }
  }
  cp_wait<0>();
  __syncthreads(); 

  // ---- P4: qg, kg, kd terms (owner rows < r); kd's G_L column parts staged for the column sum
  #pragma unroll
  for (int m = 0; m < 8; ++m) {
    const int c = 16 * m + 4 * qp;
    float tL[4] = {0.f, 0.f, 0.f, 0.f};
    if (x < r) {
      float kv[4], qv[4], av[4], bv[4], cv[4];
      row4(Kh, x, m, kv);
      row4(Qh, x, m, qv);
      const uint2 aw = *reinterpret_cast<const uint2 *>(gQG + x * D + c);
      const uint2 bw = *reinterpret_cast<const uint2 *>(gKG + x * D + c);
      const uint2 cw = *reinterpret_cast<const uint2 *>(gKD + x * D + c);
      av[0] = bf(lo16(aw.x)); av[1] = bf(hi16(aw.x)); av[2] = bf(lo16(aw.y)); av[3] = bf(hi16(aw.y));
      bv[0] = bf(lo16(bw.x)); bv[1] = bf(hi16(bw.x)); bv[2] = bf(lo16(bw.y)); bv[3] = bf(hi16(bw.y));
      cv[0] = bf(lo16(cw.x)); cv[1] = bf(hi16(cw.x)); cv[2] = bf(lo16(cw.y)); cv[3] = bf(hi16(cw.y));
      const float4 g4 = *reinterpret_cast<const float4 *>(G + x * D + c);
      const float gx[4] = {g4.x, g4.y, g4.z, g4.w};
      #pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int k = 4 * m + e;
        const float eg = decay0(gx[e]), eL = decay(GLs[c + e], gx[e]);
        dq_r[k] = ffma(av[e], eg, dq_r[k]);
        dk_r[k] = ffma(bv[e], eg, ffma(cv[e], eL, dk_r[k]));
        const float tq = fmul(fmul(av[e], qv[e]), eg), tk = fmul(fmul(bv[e], kv[e]), eg);
        tL[e] = fmul(fmul(cv[e], kv[e]), eL);
        dG_r[k] = fsub(fadd(dG_r[k], fadd(tq, tk)), tL[e]);
      }
    }
    *reinterpret_cast<float4 *>(ST + x * D + c) = make_float4(tL[0], tL[1], tL[2], tL[3]);
  }
  __syncthreads(); 
  if (tid < D) {   // dG_L = E(G_L) dGLs + sum_j tL_j (ascending j)
    float acc = fmul(decay0(GLs[tid]), p.dGLs[(int64_t(chunk) * H + h) * D + tid]);
    for (int j = 0; j < r; ++j) acc = fadd(acc, ST[j * D + tid]);
    dGL[tid] = acc;
  }
  // ---- P5: delta in place over G (its last readers, P3 / P4, are behind the barriers)
  delta_in_place(G, tid);
  __syncthreads(); 

  // ---- P6: diagonal blocks by the product chains of delta, owner quads; per channel half, R then C pass
  const int B0 = x & ~15;
  auto diag_half = [&](auto half_c) {
    constexpr int M0 = 4 * decltype(half_c)::value;   // channels 16 m + 4 qp + e, m in [M0, M0 + 4)
    {
      float Rk[16], Rq[16], dec[16];
      #pragma unroll
      for (int e = 0; e < 16; ++e) { Rk[e] = Rq[e] = 0.f; dec[e] = 1.f; }
      #pragma unroll 1
      for (int i = x; i >= B0; --i) {   // e^(G_x - G_i) = prod_{t = i+1 .. x} delta_t, built descending
        const float a = W1[x * LS + i], b = W2[x * LS + i];
        #pragma unroll
        for (int mm = 0; mm < 4; ++mm) {
          const int m = M0 + mm;
          if (i < x) {
            const float4 d4 = *reinterpret_cast<const float4 *>(G + (i + 1) * D + 16 * m + 4 * qp);
            dec[4 * mm] = fmul(dec[4 * mm], d4.x); dec[4 * mm + 1] = fmul(dec[4 * mm + 1], d4.y);
            dec[4 * mm + 2] = fmul(dec[4 * mm + 2], d4.z); dec[4 * mm + 3] = fmul(dec[4 * mm + 3], d4.w);
          }
          float ki[4];
          row4(Kh, i, m, ki);
          #pragma unroll
          for (int e = 0; e < 4; ++e) {
            const float ke = fmul(ki[e], dec[4 * mm + e]);
            Rk[4 * mm + e] = ffma(a, ke, Rk[4 * mm + e]);
            Rq[4 * mm + e] = ffma(b, ke, Rq[4 * mm + e]);
          }
        }
      }
      #pragma unroll
      for (int mm = 0; mm < 4; ++mm) {
        float kv[4], qv[4];
        row4(Kh, x, M0 + mm, kv);
        row4(Qh, x, M0 + mm, qv);
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int k = 4 * (M0 + mm) + e, kk = 4 * mm + e;
          dq_r[k] = fadd(dq_r[k], Rq[kk]);
          dk_r[k] = fadd(dk_r[k], Rk[kk]);
          dG_r[k] = fadd(dG_r[k], ffma(kv[e], Rk[kk], fmul(qv[e], Rq[kk])));
        }
      }
    }
    {
      float Ck[16], dec[16];
      #pragma unroll
      for (int e = 0; e < 16; ++e) { Ck[e] = 0.f; dec[e] = 1.f; }
      #pragma unroll 1
      for (int j = x; j < B0 + 16; ++j) {   // e^(G_j - G_x) = prod_{t = x+1 .. j} delta_t, built ascending
        const float a = W1[j * LS + x], b = W2[j * LS + x];
        #pragma unroll
        for (int mm = 0; mm < 4; ++mm) {
          const int m = M0 + mm;
          if (j > x) {
            const float4 d4 = *reinterpret_cast<const float4 *>(G + j * D + 16 * m + 4 * qp);
            dec[4 * mm] = fmul(dec[4 * mm], d4.x); dec[4 * mm + 1] = fmul(dec[4 * mm + 1], d4.y);
            dec[4 * mm + 2] = fmul(dec[4 * mm + 2], d4.z); dec[4 * mm + 3] = fmul(dec[4 * mm + 3], d4.w);
          }
          float kj[4], qj[4];
          row4(Kh, j, m, kj);
          row4(Qh, j, m, qj);
          #pragma unroll
          for (int e = 0; e < 4; ++e)
            Ck[4 * mm + e] = ffma(ffma(a, kj[e], fmul(b, qj[e])), dec[4 * mm + e], Ck[4 * mm + e]);
        }
      }
      #pragma unroll
      for (int mm = 0; mm < 4; ++mm) {
        float kv[4];
        row4(Kh, x, M0 + mm, kv);
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int k = 4 * (M0 + mm) + e;
          dk_r[k] = fadd(dk_r[k], Ck[4 * mm + e]);
          dG_r[k] = fsub(dG_r[k], fmul(kv[e], Ck[4 * mm + e]));
        }
      }
    }
  };
  diag_half(std::integral_constant<int, 0>{});
  diag_half(std::integral_constant<int, 1>{});
  
  // ---- P7: dG staged; reverse cumulative sum per channel in two 32-row halves -> dg
  #pragma unroll
  for (int m = 0; m < 8; ++m)
    *reinterpret_cast<float4 *>(ST + x * D + 16 * m + 4 * qp) =
        make_float4(dG_r[4 * m], dG_r[4 * m + 1], dG_r[4 * m + 2], dG_r[4 * m + 3]);
  __syncthreads(); 
  {
    const int c = tid & (D - 1), half = tid >> 7;
    const int lo = 32 * half, hi = min(r, lo + 32);
    float acc = half == 1 ? dGL[c] : 0.f;
    for (int j = hi - 1; j >= lo; --j) {   // suffix sums within the half (in place)
      acc = fadd(acc, ST[j * D + c]);
      ST[j * D + c] = acc;
    }
    __syncthreads();
    const float carry = (half == 0 && r > 32) ? ST[32 * D + c] : (half == 0 ? dGL[c] : 0.f);
    for (int j = lo; j < hi; ++j) {
      const float v = half == 0 ? fadd(ST[j * D + c], carry) : ST[j * D + c];
      const int64_t gi = (int64_t(t0) + j) * rowH + hoff + c;
      if (p.g_bf16) static_cast<uint16_t *>(p.dg)[gi] = rn(v);
      else static_cast<float *>(p.dg)[gi] = v;
    }
  }
  // ---- l2 norm backward from the normalized rows: dx = r (dxh - xh (xh . dxh)), xh = x^ / s, dxh = s dx^
  if (x < r) {
    #pragma unroll
    for (int which = 0; which < 2; ++which) {
      const float rr = which == 0 ? rk[x] : rq[x];
      const float s = which == 0 ? 1.f : QSCALE, inv_s = which == 0 ? 1.f : 11.313708498984761f;
      float xh[32], dxh[32];
      #pragma unroll
      for (int m = 0; m < 8; ++m) {
        float xv[4];
        row4(which == 0 ? Kh : Qh, x, m, xv);
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
          xh[4 * m + e] = fmul(xv[e], inv_s);
          dxh[4 * m + e] = fmul(s, which == 0 ? dk_r[4 * m + e] : dq_r[4 * m + e]);
        }
      }
      float dot = 0.f;
      #pragma unroll
      for (int e = 0; e < 32; ++e) dot = ffma(xh[e], dxh[e], dot);
      dot = quad_t4(dot);
      uint16_t *dst = (which == 0 ? p.dk : p.dq) + (int64_t(t0) + x) * rowH + hoff;
      #pragma unroll
      for (int m = 0; m < 8; ++m) {
        float o[4];
        #pragma unroll
        for (int e = 0; e < 4; ++e) o[e] = fmul(rr, fsub(dxh[4 * m + e], fmul(xh[4 * m + e], dot)));
        uint2 w2;
        w2.x = rn2(o[0], o[1]);
        w2.y = rn2(o[2], o[3]);
        *reinterpret_cast<uint2 *>(dst + 16 * m + 4 * qp) = w2;
      }
    }
    if (qp == 0) {
      const int64_t bi = (int64_t(t0) + x) * H + h;
      const float db = p.dbeta_p[bi];
      if (p.beta_bf16) static_cast<uint16_t *>(p.dbeta)[bi] = rn(db);
      else static_cast<float *>(p.dbeta)[bi] = db;
    }
  }
  
}

// ============================================================================ launches
void bwd_inter_launch(const BwdParams &p, cudaStream_t stream) {
  static std::once_flag attr[64];  // the attribute is per device
  int attr_dev = 0;
  cudaGetDevice(&attr_dev);
  std::call_once(attr[unsigned(attr_dev) % 64u], [&] {
    cudaFuncSetAttribute(bwd_inter_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, inter::SMEM);
  });
  bwd_inter_kernel<<<dim3(p.plan.maxch, p.H), NT, inter::SMEM, stream>>>(p);
}
void bwd_wy_launch(const BwdParams &p, cudaStream_t stream) {
  static std::once_flag attr[64];  // the attribute is per device
  int attr_dev = 0;
  cudaGetDevice(&attr_dev);
  std::call_once(attr[unsigned(attr_dev) % 64u], [&] {
    cudaFuncSetAttribute(bwd_wy_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, wy::SMEM);
  });
  bwd_wy_kernel<<<dim3(p.plan.maxch, p.H), NT, wy::SMEM, stream>>>(p);
}
void bwd_qkg_launch(const BwdParams &p, cudaStream_t stream) {
  static std::once_flag attr[64];  // the attribute is per device
  int attr_dev = 0;
  cudaGetDevice(&attr_dev);
  std::call_once(attr[unsigned(attr_dev) % 64u], [&] {
    cudaFuncSetAttribute(bwd_qkg_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, qkg::SMEM);
  });
  bwd_qkg_kernel<<<dim3(p.plan.maxch, p.H), NT, qkg::SMEM, stream>>>(p);
}

}  // namespace kda_sm100
