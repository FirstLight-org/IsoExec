// SPDX-License-Identifier: Apache-2.0
// mHC split backward. Kernels (all atomic-free; every cross-thread/CTA reduction is a fixed tree):
//   bwd_dcolx   : d_pre_collapse[row][s] = sum_h dcol[h] * x_s[h]       (tree over h; README "Backward")
//   bwd_tail    : per row, recompute the scalar tail from (proj, rstd), back-propagate Sinkhorn / softmax / sigmoid,
//                 emit dproj (three exact bf16 pieces), per-row scale/base gradients, c1 = d_rstd * rstd^3 / (4H)
//   bwd_main    : tensor-core dgrad  dx = dproj . fn  (+ pre*dcol - c1*x)  and the d_fn token reduction
//                 (fixed 512-row segments accumulated in fixed order, partial per segment)
//   bwd_reduce_fn, bwd_reduce_rows : fixed-tree reductions of the segment partials / per-row scale+base gradients
//
// dproj[row][j] (fp32) is split into three bf16 pieces hi, mid, lo (each subtraction exact), so every tensor-core
// product dproj-piece * x (bf16) is exact; fp32 accumulation order is fixed (pieces hi, mid, lo; ascending tokens).
// For an fp32 fn (DeepSeek) fn is likewise split into three pieces and only the products with i + j <= 2 are summed
// (the dropped ones are below 2^-24 of the result).
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <type_traits>

#include "common.cuh"
#include "tail.cuh"

namespace mhc {

constexpr int kSegRows = 512;     // d_fn / d_scale / d_base token segment (deployment constant)
constexpr int kTile = 32;         // rows per bwd_main tile (kSegRows is a multiple)
constexpr int kBwdSlices = 64;    // column slices of bwd_main (H / 64 columns per stream per CTA)
constexpr int kBwdStages = 2;     // bwd_main input ring depth (tiles): kBwdStages - 1 tiles in flight ahead of the one computed

// ---------------------------------------------------------------------------------------------------------
// d_pre from the collapse: one 128-thread block per row.
// thread j owns chunks j, j+128, ... of 8 h (ascending, sequential fp32); butterfly over the warp, then (w0+w1)+(w2+w3).
__global__ void __launch_bounds__(128) bwd_dcolx(const __nv_bfloat16 *__restrict__ x, const __nv_bfloat16 *__restrict__ dcol,
                                                 float *__restrict__ out, int H) {
  __shared__ float red[4][4];
  const int row = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  for (int c = tid; c < H / 8; c += 128) {
    const int h = c * 8;
    const uint4 q = *reinterpret_cast<const uint4 *>(dcol + (size_t)row * H + h);
    const uint32_t gw[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const uint4 xq = *reinterpret_cast<const uint4 *>(x + ((size_t)row * 4 + s) * H + h);
      const uint32_t xw[4] = {xq.x, xq.y, xq.z, xq.w};
      float a = acc[s];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        a = __fadd_rn(a, __fmul_rn(bf_lo(gw[j]), bf_lo(xw[j])));
        a = __fadd_rn(a, __fmul_rn(bf_hi(gw[j]), bf_hi(xw[j])));
      }
      acc[s] = a;
    }
  }
#pragma unroll
  for (int s = 0; s < 4; ++s)
#pragma unroll
    for (int m = 16; m >= 1; m >>= 1) acc[s] = __fadd_rn(acc[s], __shfl_xor_sync(FULL, acc[s], m));
  if (lane == 0) {
#pragma unroll
    for (int s = 0; s < 4; ++s) red[warp][s] = acc[s];
  }
  __syncthreads();
  if (tid < 4) out[(size_t)row * 4 + tid] = __fadd_rn(__fadd_rn(red[0][tid], red[1][tid]), __fadd_rn(red[2][tid], red[3][tid]));
}

// ---------------------------------------------------------------------------------------------------------
struct TailBwdParams {
  const float *proj, *rstd;        // [T][24], [T]
  const float *dpre_in;            // [T][4] gradient of the pre output (may be null)
  const float *dcolx;              // [T][4] gradient of pre through the collapse (null if the collapse is not differentiated)
  const float *ext_pre;            // [T][4] pre used by the collapse when it is the shifted one (null: use own pre)
  const float *dpost, *dcomb;      // [T][4], [T][16]
  const float *scale, *base;
  __nv_bfloat16 *dpa;              // [T][3][32] dproj pieces (j >= 24 zero)
  float *rowaux;                   // [T][8]: pre used by the dx direct term (4), c1, pad
  float *da;                       // [T][24] gradient of the pre-activation logits (= d base per row)
  float *dsc;                      // [T][4]: per-row d scale (3) + pad
  int T, H, iters;
  float eps;
};

// One warp per row; smem history of the Sinkhorn chain (Y and D after every step).
__global__ void __launch_bounds__(128) bwd_tail(const TailBwdParams p) {
  extern __shared__ float hist[];                       // [4 warps][steps][2][16]
  const int lane = threadIdx.x & 31, wi = threadIdx.x >> 5;
  const int row = blockIdx.x * 4 + wi;
  if (row >= p.T) return;
  const int steps = 2 * p.iters - 1;
  float *Yh = hist + (size_t)wi * steps * 32, *Dh = Yh + 16;   // Yh[k*32 + idx], Dh[k*32 + idx]
  const float rstd = p.rstd[row];
  const float val = lane < 24 ? p.proj[(size_t)row * 24 + lane] : 0.f;
  const float mix = __fmul_rn(val, rstd);
  const float mix_c = __shfl_sync(FULL, mix, 8 + (lane & 15));
  const int idx = lane & 15;
  // ---- forward recompute (same operations as tail_forward) ----
  float sg = 0.f;                                        // sigmoid value, lanes 0..7
  {
    const float a = __fadd_rn(__fmul_rn(mix, p.scale[lane < 4 ? 0 : 1]), p.base[lane & 7]);
    sg = sigmoid_rn(a);
  }
  const float ac = __fadd_rn(__fmul_rn(mix_c, p.scale[2]), p.base[8 + idx]);
  float m = fmaxf(ac, __shfl_xor_sync(FULL, ac, 1));
  m = fmaxf(m, __shfl_xor_sync(FULL, m, 2));
  const float e = expf(__fsub_rn(ac, m));
  const float psm = __fdiv_rn(e, sum4_row(e));
  float q = __fadd_rn(psm, p.eps);
  {
    const float D = __fadd_rn(sum4_col(q), p.eps);
    q = __fdiv_rn(q, D);
    if (lane >= 16) { Yh[idx] = q; Dh[idx] = D; }
  }
  for (int it = 1; it < p.iters; ++it) {
    float D = __fadd_rn(sum4_row(q), p.eps);
    q = __fdiv_rn(q, D);
    if (lane >= 16) { Yh[(2 * it - 1) * 32 + idx] = q; Dh[(2 * it - 1) * 32 + idx] = D; }
    D = __fadd_rn(sum4_col(q), p.eps);
    q = __fdiv_rn(q, D);
    if (lane >= 16) { Yh[(2 * it) * 32 + idx] = q; Dh[(2 * it) * 32 + idx] = D; }
  }
  __syncwarp();
  // ---- backward ----
  float dq = lane >= 16 ? p.dcomb[(size_t)row * 16 + idx] : 0.f;
  for (int k = steps - 1; k >= 0; --k) {
    const float Y = Yh[k * 32 + idx], D = Dh[k * 32 + idx];
    const float t = __fmul_rn(dq, Y);
    const float s = (k & 1) ? sum4_row(t) : sum4_col(t);   // odd steps normalise rows, even steps columns
    dq = __fdiv_rn(__fsub_rn(dq, s), D);
  }
  // softmax backward (dq is d(p + eps) = dp)
  const float sdot = sum4_row(__fmul_rn(dq, psm));
  const float da_c = __fmul_rn(psm, __fsub_rn(dq, sdot));
  // pre / post
  float dpp = 0.f;
  if (lane < 4) {
    dpp = 0.f;
    if (p.dpre_in) dpp = p.dpre_in[(size_t)row * 4 + lane];
    if (p.dcolx) dpp = __fadd_rn(dpp, p.dcolx[(size_t)row * 4 + lane]);
  } else if (lane < 8) {
    dpp = p.dpost[(size_t)row * 4 + lane - 4];
  }
  const float fac = lane < 4 ? 1.0f : 2.0f;
  const float da_pp = __fmul_rn(__fmul_rn(dpp, fac), __fmul_rn(sg, __fsub_rn(1.0f, sg)));
  // unify into lanes 0..23
  const float da_moved = __shfl_sync(FULL, da_c, (lane + 8) & 31);   // executed by all lanes
  const float da24 = lane < 8 ? da_pp : da_moved;
  // (lane j in 8..23 reads lane j + 8 in 16..31)
  const bool act = lane < 24;
  const int grp = lane < 4 ? 0 : (lane < 8 ? 1 : 2);
  const float dm = act ? __fmul_rn(da24, p.scale[grp]) : 0.f;
  const float dproj = __fmul_rn(dm, rstd);
  // d_rstd = sum_j dm_j * proj_j (butterfly over the warp, lanes >= 24 contribute 0)
  float dr = act ? __fmul_rn(dm, val) : 0.f;
#pragma unroll
  for (int mm = 16; mm >= 1; mm >>= 1) dr = __fadd_rn(dr, __shfl_xor_sync(FULL, dr, mm));
  const float r3 = __fmul_rn(__fmul_rn(rstd, rstd), rstd);
  const float c1 = __fdiv_rn(__fmul_rn(dr, r3), (float)(4 * p.H));
  // per-row d scale: three group sums of da * mix
  float tj = act ? __fmul_rn(da24, mix) : 0.f;
  float ds[3];
#pragma unroll
  for (int gi = 0; gi < 3; ++gi) {
    float u = (grp == gi && act) ? tj : 0.f;
#pragma unroll
    for (int mm = 16; mm >= 1; mm >>= 1) u = __fadd_rn(u, __shfl_xor_sync(FULL, u, mm));
    ds[gi] = u;
  }
  // outputs
  __nv_bfloat16 p0, p1, p2;
  split3(dproj, p0, p1, p2);
  __nv_bfloat16 *dpa = p.dpa + (size_t)row * 96;
  dpa[lane] = p0; dpa[32 + lane] = p1; dpa[64 + lane] = p2;   // lanes >= 24 hold exact zeros
  if (act) p.da[(size_t)row * 24 + lane] = da24;
  if (lane < 3) p.dsc[(size_t)row * 4 + lane] = ds[lane];
  // the pre that the dx direct term uses: own (recomputed) or the shifted one
  float prev = 0.f;
  if (lane < 4) prev = p.ext_pre ? p.ext_pre[(size_t)row * 4 + lane] : __fadd_rn(sg, p.eps);
  if (lane < 4) p.rowaux[(size_t)row * 8 + lane] = prev;
  if (lane == 4) p.rowaux[(size_t)row * 8 + 4] = c1;
  if (lane == 5 || lane == 6 || lane == 7) p.rowaux[(size_t)row * 8 + lane] = 0.f;
}


// ---------------------------------------------------------------------------------------------------------
struct MainParams {
  const __nv_bfloat16 *x;      // [T][4][H]
  const __nv_bfloat16 *dcol;   // [T][H] or null
  const void *fn;              // [24][4H] bf16 or fp32
  const __nv_bfloat16 *dpa;    // [T][3][32]
  const float *rowaux;         // [T][8]
  __nv_bfloat16 *dx;           // [T][4][H]
  float *dfn_part;             // [nseg][24][4H]
  int T, H, ntiles, nseg;
};

template <int HSB>
struct MainLayout {
  static constexpr int XS = 4 * HSB * 2 + 16;      // x / staging row stride (bytes)
  static constexpr int DS = HSB * 2 + 16;          // dcol row stride
  static constexpr int PS = 192 + 16;              // dpa row stride
  static constexpr int AS = 32;                    // aux row stride
  static constexpr int BUF = kTile * (XS + DS + PS + AS);
  static constexpr int BYTES = kBwdStages * BUF + kTile * XS;
};

template <int HSB, bool FN32>
__global__ void __launch_bounds__(256, 2) bwd_main(const MainParams p) {
  using L = MainLayout<HSB>;
  constexpr int NT = HSB / 16, H4 = 4;
  // SW: d_fn with the features as the MMA M dimension and the 24 logits as three 8-wide N tiles (no padding of j to 32). Needs the
  // warp's feature span (HSB / 2) to be a multiple of 16. Same products, same per-element accumulation order (tokens ascending in
  // 16-token steps, pieces hi, mid, lo), 25% fewer MMAs.
  constexpr bool SW = (HSB / 2) % 16 == 0;
  constexpr int MF = SW ? HSB / 32 : 2;          // accD tiles along M (features if SW, else 16-j blocks)
  constexpr int NJ = SW ? 3 : NT;                // accD tiles along N (logits if SW, else 8-feature blocks)
  constexpr int P = FN32 ? 3 : 1;
  const int H = p.H;
  extern __shared__ __align__(16) unsigned char smem[];
  unsigned char *stage_out = smem + kBwdStages * L::BUF;
  auto xbuf = [&](int b) { return smem + (size_t)b * L::BUF; };
  auto dbuf = [&](int b) { return smem + (size_t)b * L::BUF + kTile * L::XS; };
  auto pbuf = [&](int b) { return smem + (size_t)b * L::BUF + kTile * (L::XS + L::DS); };
  auto abuf = [&](int b) { return smem + (size_t)b * L::BUF + kTile * (L::XS + L::DS + L::PS); };

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int stream = warp >> 1, col0 = (warp & 1) * (HSB / 2);
  const int slice = blockIdx.x % kBwdSlices, sg0 = blockIdx.x / kBwdSlices, G = gridDim.x / kBwdSlices;
  const int hbase = slice * HSB;
  const size_t Hf = (size_t)H;

  // ---- fn operand (dgrad B fragments), resident -------------------------------------------------------
  uint32_t Bfn[NT][2][P][2];
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
    const size_t c = (size_t)stream * Hf + hbase + col0 + nt * 8 + g;
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      const int j0 = ks * 16 + 2 * t, j1 = j0 + 8;
      float f[4];
      auto ld = [&](int j) -> float {
        if (j >= 24) return 0.f;
        if constexpr (FN32) return reinterpret_cast<const float *>(p.fn)[(size_t)j * 4 * Hf + c];
        else return __bfloat162float(reinterpret_cast<const __nv_bfloat16 *>(p.fn)[(size_t)j * 4 * Hf + c]);
      };
      f[0] = ld(j0); f[1] = ld(j0 + 1); f[2] = ld(j1); f[3] = ld(j1 + 1);
      if constexpr (!FN32) {
        Bfn[nt][ks][0][0] = pack_f2_bf16(f[0], f[1]);
        Bfn[nt][ks][0][1] = pack_f2_bf16(f[2], f[3]);
      } else {
        uint32_t lo[3], hi[3];
        split3_trunc_pair(f[0], f[1], lo);
        split3_trunc_pair(f[2], f[3], hi);
#pragma unroll
        for (int pc = 0; pc < 3; ++pc) { Bfn[nt][ks][pc][0] = lo[pc]; Bfn[nt][ks][pc][1] = hi[pc]; }
      }
    }
  }

  // ---- my tile sequence: segments sg0, sg0 + G, ...; kSegRows / kTile tiles each -------------------------
  constexpr int TPS = kSegRows / kTile;
  auto tile_of = [&](int it) -> int {
    const int sgi = sg0 + (it / TPS) * G;
    const int tile = sgi * TPS + (it % TPS);
    return (sgi < p.nseg && tile < p.ntiles) ? tile : -1;
  };
  auto stage = [&](int tile, int b) {
    const int row0 = tile * kTile;
    constexpr int XC = HSB / 8;  // 16B chunks per (row, stream)
    unsigned char *xb = xbuf(b), *db = dbuf(b), *pb = pbuf(b), *ab = abuf(b);
    for (int i = tid; i < kTile * 4 * XC; i += 256) {
      const int r = i / (4 * XC), rem = i - r * 4 * XC, s = rem / XC, c = rem - s * XC;
      const bool v = row0 + r < p.T;
      cp_async16(xb + (size_t)r * L::XS + (s * HSB + c * 8) * 2,
                 p.x + ((size_t)(v ? row0 + r : 0) * 4 + s) * Hf + hbase + c * 8, v);
    }
    if (p.dcol) {
      for (int i = tid; i < kTile * XC; i += 256) {
        const int r = i / XC, c = i - r * XC;
        const bool v = row0 + r < p.T;
        cp_async16(db + (size_t)r * L::DS + c * 16, p.dcol + (size_t)(v ? row0 + r : 0) * Hf + hbase + c * 8, v);
      }
    }
    for (int i = tid; i < kTile * 12; i += 256) {
      const int r = i / 12, c = i - r * 12;
      const bool v = row0 + r < p.T;
      cp_async16(pb + (size_t)r * L::PS + c * 16, p.dpa + (size_t)(v ? row0 + r : 0) * 96 + c * 8, v);
    }
    for (int i = tid; i < kTile * 2; i += 256) {
      const int r = i >> 1, c = i & 1;
      const bool v = row0 + r < p.T;
      cp_async16(ab + (size_t)r * L::AS + c * 16, p.rowaux + (size_t)(v ? row0 + r : 0) * 8 + c * 4, v);
    }
  };

  float accD[MF][NJ][4];
#pragma unroll
  for (int a = 0; a < MF; ++a)
#pragma unroll
    for (int n = 0; n < NJ; ++n) accD[a][n][0] = accD[a][n][1] = accD[a][n][2] = accD[a][n][3] = 0.f;

  int buf = 0;
#pragma unroll
  for (int s = 0; s < kBwdStages - 1; ++s) {
    if (tile_of(s) >= 0) stage(tile_of(s), s);
    cp_async_commit();
  }
  for (int it = 0;; ++it) {
    const int tile = tile_of(it);
    if (tile < 0) break;
    const int nxt = tile_of(it + 1);
    {
      const int ahead = tile_of(it + kBwdStages - 1);
      if (ahead >= 0) stage(ahead, (buf + kBwdStages - 1) % kBwdStages);
    }
    cp_async_commit();
    cp_async_wait_group_n<kBwdStages - 1>();
    __syncthreads();
    const int row0 = tile * kTile;
    const unsigned char *xb = xbuf(buf), *db = dbuf(buf), *pb = pbuf(buf), *ab = abuf(buf);

    // d_fn MMAs. SW (features as M): issued after the dgrad MMAs so the epilogue ALU work overlaps the tensor pipe; otherwise first.
    auto dfn_mma = [&]() {
    if constexpr (SW) {
      // ---- d_fn: accD[mf][jt] += x^T[feature][tok] . dprojPiece[tok][j]  (M = feature, N = j (3 tiles), K = tokens)
#pragma unroll
      for (int kt = 0; kt < 2; ++kt) {
        uint32_t Ax[MF][4];
        const int mat = lane >> 3, rr = lane & 7;
#pragma unroll
        for (int mf = 0; mf < MF; ++mf)       // matrices a0..a3: (tok0-7,f0-7) (tok0-7,f8-15) (tok8-15,f0-7) (tok8-15,f8-15)
          ldsm_x4_trans(Ax[mf], xb + (size_t)(kt * 16 + (mat >> 1) * 8 + rr) * L::XS +
                                    (stream * HSB + col0 + mf * 16 + (mat & 1) * 8) * 2);
#pragma unroll
        for (int pc = 0; pc < 3; ++pc) {
          uint32_t Bd[3][2];
          {
            uint32_t r[4];                    // (tok0-7, jt0) (tok8-15, jt0) (tok0-7, jt1) (tok8-15, jt1)
            ldsm_x4_trans(r, pb + (size_t)(kt * 16 + (mat & 1) * 8 + rr) * L::PS + (pc * 32 + (mat >> 1) * 8) * 2);
            Bd[0][0] = r[0]; Bd[0][1] = r[1]; Bd[1][0] = r[2]; Bd[1][1] = r[3];
          }
          {
            uint32_t r[2];
            ldsm_x2_trans(r, pb + (size_t)(kt * 16 + ((lane >> 3) & 1) * 8 + rr) * L::PS + (pc * 32 + 16) * 2);
            Bd[2][0] = r[0]; Bd[2][1] = r[1];
          }
#pragma unroll
          for (int mf = 0; mf < MF; ++mf)
#pragma unroll
            for (int jt = 0; jt < 3; ++jt) mma_bf16(accD[mf][jt], Ax[mf], Bd[jt][0], Bd[jt][1]);
        }
      }
    } else {
    // ---- d_fn: accD[mj][nt] += dprojT[j][tok] . x[tok][feature]  (M = j, N = feature, K = tokens)
#pragma unroll
    for (int kt = 0; kt < 2; ++kt) {
      uint32_t Bx[NT][2];
#pragma unroll
      for (int nt = 0; nt < NT; nt += 2) {
        const int f0 = stream * HSB + col0 + nt * 8;
        if (nt + 1 < NT) {
          uint32_t r[4];
          const int mat = lane >> 3, rr = lane & 7;       // matrices: (nt,tok0-7) (nt,tok8-15) (nt+1,tok0-7) (nt+1,tok8-15)
          const unsigned char *a = xb + (size_t)(kt * 16 + (mat & 1) * 8 + rr) * L::XS + (f0 + (mat >> 1) * 8) * 2;
          ldsm_x4_trans(r, a);
          Bx[nt][0] = r[0]; Bx[nt][1] = r[1]; Bx[nt + 1][0] = r[2]; Bx[nt + 1][1] = r[3];
        } else {
          uint32_t r[2];
          const int mat = (lane >> 3) & 1, rr = lane & 7;
          ldsm_x2_trans(r, xb + (size_t)(kt * 16 + mat * 8 + rr) * L::XS + f0 * 2);
          Bx[nt][0] = r[0]; Bx[nt][1] = r[1];
        }
      }
#pragma unroll
      for (int mj = 0; mj < 2; ++mj)
#pragma unroll
        for (int pc = 0; pc < 3; ++pc) {
          uint32_t A[4];
          const int mat = lane >> 3, rr = lane & 7;       // a0: (tok0-7, j0-7) a1: (tok0-7, j8-15) a2: (tok8-15, j0-7) a3: (tok8-15, j8-15)
          ldsm_x4_trans(A, pb + (size_t)(kt * 16 + (mat >> 1) * 8 + rr) * L::PS + (pc * 32 + mj * 16 + (mat & 1) * 8) * 2);
#pragma unroll
          for (int nt = 0; nt < NT; ++nt) mma_bf16(accD[mj][nt], A, Bx[nt][0], Bx[nt][1]);
        }
    }

    }

    };
    if constexpr (!SW) dfn_mma();
    asm volatile("" ::: "memory");   // keep the MMA phases in program order (the compiler otherwise interleaves them into a spilling schedule)
    // ---- dgrad MMAs (both 16-token m-tiles), then d_fn MMAs, then the epilogue: the epilogue ALU work does not depend on the
    // d_fn products, so the tensor pipe stays busy while it runs
    float accg[2][NT][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
      float (&acc)[NT][4] = accg[mt];
#pragma unroll
      for (int n = 0; n < NT; ++n) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f;
#pragma unroll
      for (int ks = 0; ks < 2; ++ks) {
        uint32_t A[3][4];
#pragma unroll
        for (int ip = 0; ip < 3; ++ip) {
          const int mat = lane >> 3, rr = lane & 7;       // a0: (tok0-7, j0-7) a1: (tok8-15, j0-7) a2: (tok0-7, j8-15) a3: (tok8-15, j8-15)
          if (ks == 0) {
            ldsm_x4(A[ip], pb + (size_t)(mt * 16 + (mat & 1) * 8 + rr) * L::PS + (ip * 32 + (mat >> 1) * 8) * 2);
          } else {                                        // j = 16..23 only (24..31 are exact zeros): a0, a1 of a k8 MMA
            uint32_t r2[2];
            ldsm_x2(r2, pb + (size_t)(mt * 16 + ((lane >> 3) & 1) * 8 + rr) * L::PS + (ip * 32 + 16) * 2);
            A[ip][0] = r2[0]; A[ip][1] = r2[1]; A[ip][2] = 0u; A[ip][3] = 0u;
          }
        }
#pragma unroll
        for (int jp = 0; jp < P; ++jp)
#pragma unroll
          for (int ip = 0; ip + jp < 3; ++ip)
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
              if (ks == 0) mma_bf16(acc[nt], A[ip], Bfn[nt][ks][jp][0], Bfn[nt][ks][jp][1]);
              else mma_bf16_k8(acc[nt], A[ip][0], A[ip][1], Bfn[nt][ks][jp][0]);
            }
      }
    }
    asm volatile("" ::: "memory");
    if constexpr (SW) dfn_mma();
    asm volatile("" ::: "memory");
    // ---- epilogue per 16-token m-tile
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
      float (&acc)[NT][4] = accg[mt];
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {
        const int r = mt * 16 + g + hf * 8;
        const float *aux = reinterpret_cast<const float *>(ab + (size_t)r * L::AS);
        const float prev = aux[stream], c1 = aux[4];
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
          const int lc = col0 + nt * 8 + 2 * t;
          const uint32_t xw = *reinterpret_cast<const uint32_t *>(xb + (size_t)r * L::XS + (stream * HSB + lc) * 2);
          float v0 = acc[nt][hf * 2], v1 = acc[nt][hf * 2 + 1];
          if (p.dcol) {
            const uint32_t dw = *reinterpret_cast<const uint32_t *>(db + (size_t)r * L::DS + lc * 2);
            v0 = __fadd_rn(__fmul_rn(prev, bf_lo(dw)), v0);
            v1 = __fadd_rn(__fmul_rn(prev, bf_hi(dw)), v1);
          }
          v0 = __fsub_rn(v0, __fmul_rn(c1, bf_lo(xw)));
          v1 = __fsub_rn(v1, __fmul_rn(c1, bf_hi(xw)));
          *reinterpret_cast<uint32_t *>(stage_out + (size_t)r * L::XS + (stream * HSB + lc) * 2) = pack_f2_bf16(v0, v1);
        }
      }
    }
    __syncthreads();   // staging complete; every read of this tile's buffers is done
    {
      constexpr int XC = HSB / 8;
      for (int i = tid; i < kTile * 4 * XC; i += 256) {
        const int r = i / (4 * XC), rem = i - r * 4 * XC, s = rem / XC, c = rem - s * XC;
        if (row0 + r >= p.T) continue;
        *reinterpret_cast<uint4 *>(p.dx + ((size_t)(row0 + r) * 4 + s) * Hf + hbase + c * 8) =
            *reinterpret_cast<const uint4 *>(stage_out + (size_t)r * L::XS + (s * HSB + c * 8) * 2);
      }
    }
    // ---- segment boundary: flush the d_fn partial
    const bool last_of_seg = (it % TPS) == TPS - 1 || nxt < 0 || (nxt / TPS == tile / TPS) == false;
    if ((it % TPS) == TPS - 1 || tile_of(it + 1) < 0) {
      const int sgi = sg0 + (it / TPS) * G;
      (void)last_of_seg;
      if constexpr (SW) {
#pragma unroll
        for (int mf = 0; mf < MF; ++mf)
#pragma unroll
          for (int jt = 0; jt < 3; ++jt)
#pragma unroll
            for (int hf = 0; hf < 2; ++hf) {
              const size_t c = (size_t)stream * Hf + hbase + col0 + mf * 16 + g + hf * 8;     // feature (row of the accumulator)
              const int j0 = jt * 8 + 2 * t;                                                    // logits (columns 2t, 2t + 1)
              p.dfn_part[((size_t)sgi * 24 + j0) * 4 * Hf + c] = accD[mf][jt][hf * 2];
              p.dfn_part[((size_t)sgi * 24 + j0 + 1) * 4 * Hf + c] = accD[mf][jt][hf * 2 + 1];
              accD[mf][jt][hf * 2] = 0.f;
              accD[mf][jt][hf * 2 + 1] = 0.f;
            }
      } else {
#pragma unroll
      for (int mj = 0; mj < 2; ++mj)
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
          const int j = mj * 16 + g + hf * 8;
          if (j < 24) {
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
              const size_t c = (size_t)stream * Hf + hbase + col0 + nt * 8 + 2 * t;
              *reinterpret_cast<float2 *>(p.dfn_part + ((size_t)sgi * 24 + j) * 4 * Hf + c) =
                  make_float2(accD[mj][nt][hf * 2], accD[mj][nt][hf * 2 + 1]);
            }
          }
          if (true) {
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) { accD[mj][nt][hf * 2] = 0.f; accD[mj][nt][hf * 2 + 1] = 0.f; }
          }
        }
      }
    }
    buf = buf + 1 == kBwdStages ? 0 : buf + 1;
  }
}

// ---------------------------------------------------------------------------------------------------------
// d_fn = fixed tree over the segment partials (balanced adjacent pairs, odd tail carried), cast to fn's dtype.
template <typename OutT, int MAXN>
__global__ void __launch_bounds__(256) bwd_reduce_fn(const float *__restrict__ part, OutT *__restrict__ out, int nseg,
                                                     int64_t n) {
  const int64_t i = (int64_t)blockIdx.x * 256 + threadIdx.x;
  if (i >= n) return;
  float v[MAXN];
#pragma unroll
  for (int s = 0; s < MAXN; ++s) v[s] = s < nseg ? part[(int64_t)s * n + i] : 0.f;
  const float tot = tree_sum<MAXN>(v, nseg);
  if constexpr (std::is_same<OutT, float>::value) out[i] = tot;
  else out[i] = __float2bfloat16_rn(tot);
}

// d_scale / d_base: 27 columns (24 da + 3 dscale). Segment partial = butterfly over 32 lanes, lane l summing rows
// seg*512 + 16 l .. +15 ascending; then the balanced tree over segments.
__global__ void __launch_bounds__(32) bwd_rows_partial(const float *__restrict__ da, const float *__restrict__ dsc,
                                                       float *__restrict__ part, int T, int nseg) {
  const int col = blockIdx.x, seg = blockIdx.y, lane = threadIdx.x;
  float a = 0.f;
  const int r0 = seg * kSegRows + lane * 16;
  for (int r = r0; r < r0 + 16 && r < T; ++r)
    a = __fadd_rn(a, col < 24 ? da[(size_t)r * 24 + col] : dsc[(size_t)r * 4 + col - 24]);
#pragma unroll
  for (int m = 16; m >= 1; m >>= 1) a = __fadd_rn(a, __shfl_xor_sync(FULL, a, m));
  if (lane == 0) part[(size_t)col * nseg + seg] = a;
}
__global__ void __launch_bounds__(32) bwd_rows_final(const float *__restrict__ part, float *__restrict__ dscale,
                                                     float *__restrict__ dbase, int nseg) {
  const int col = threadIdx.x;
  if (col >= 27) return;
  float v[128];
#pragma unroll
  for (int s = 0; s < 128; ++s) v[s] = s < nseg ? part[(size_t)col * nseg + s] : 0.f;
  const float tot = tree_sum<128>(v, nseg);
  if (col < 24) dbase[col] = tot; else dscale[col - 24] = tot;
}

}  // namespace mhc

namespace mhc_host {

using namespace mhc;


template <int HSB, bool FN32>
static void launch_main(const MainParams &p, int nseg, cudaStream_t st, int groups_override) {
  auto kern = bwd_main<HSB, FN32>;
  constexpr int smem = MainLayout<HSB>::BYTES;
  cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  int dev = 0, sms = 0, per_sm = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, 256, smem);
  int G = std::max(1, std::min(nseg, sms * std::max(per_sm, 1) / kBwdSlices));
  if (groups_override > 0) G = std::min(nseg, groups_override);   // ownership only (tests); never changes a value
  kern<<<G * kBwdSlices, 256, smem, st>>>(p);
}

// d_pre = sum_h dcol * x_s for a standalone (shifted) collapse. [T, 4] fp32.
at::Tensor dcolx_cuda(const at::Tensor &x, const at::Tensor &dcol) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() == 3 && x.size(1) == 4 && x.scalar_type() == at::kBFloat16, "bad x");
  const int64_t T = x.size(0), H = x.size(2);
  TORCH_CHECK(H % 8 == 0 && dcol.is_contiguous() && dcol.scalar_type() == at::kBFloat16 && dcol.sizes() == at::IntArrayRef({T, H}), "bad dcol");
  const c10::cuda::CUDAGuard guard(x.device());
  at::Tensor out = at::empty({T, 4}, x.options().dtype(at::kFloat));
  if (T == 0) return out;
  bwd_dcolx<<<(unsigned)T, 128, 0, at::cuda::getCurrentCUDAStream().stream()>>>(
      reinterpret_cast<const __nv_bfloat16 *>(x.data_ptr()), reinterpret_cast<const __nv_bfloat16 *>(dcol.data_ptr()),
      out.data_ptr<float>(), (int)H);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// returns (dx [T,4,H] bf16, dfn [24,4H] (fn dtype), dscale [3], dbase [24], dcolx [T,4] fp32)
std::vector<at::Tensor> split_backward_cuda(
    const at::Tensor &x, const at::Tensor &fn, const at::Tensor &scale, const at::Tensor &base,
    const at::Tensor &proj, const at::Tensor &rstd, const c10::optional<at::Tensor> &dpre_in,
    const at::Tensor &dpost, const at::Tensor &dcomb, const c10::optional<at::Tensor> &dcol,
    const c10::optional<at::Tensor> &ext_pre, double eps, int64_t iters, int64_t groups) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() == 3 && x.size(1) == 4 && x.scalar_type() == at::kBFloat16, "bad x");
  const int64_t T = x.size(0), H = x.size(2);
  TORCH_CHECK(H == 4096 || H == 5120, "unsupported H");
  const bool fn32 = fn.scalar_type() == at::kFloat;
  TORCH_CHECK(fn.is_contiguous() && fn.sizes() == at::IntArrayRef({24, 4 * H}), "bad fn");
  TORCH_CHECK(iters >= 1 && iters <= 32, "iters must be in [1, 32]");
  TORCH_CHECK(T <= (int64_t)128 * kSegRows, "split backward serves up to ", 128 * kSegRows, " rows per call");
  const c10::cuda::CUDAGuard guard(x.device());
  auto f32 = x.options().dtype(at::kFloat);
  at::Tensor dx = at::empty_like(x);
  at::Tensor dfn = at::empty({24, 4 * H}, fn.options());
  at::Tensor dscale = at::zeros({3}, f32), dbase = at::zeros({24}, f32);
  at::Tensor dcolx = at::empty({T, 4}, f32);
  if (T == 0) { dfn.zero_(); return {dx, dfn, dscale, dbase, dcolx}; }
  auto st = at::cuda::getCurrentCUDAStream().stream();
  const bool has_dcol = dcol.has_value();
  if (has_dcol) {
    TORCH_CHECK(dcol->is_contiguous() && dcol->scalar_type() == at::kBFloat16 && dcol->sizes() == at::IntArrayRef({T, H}), "bad dcol");
    bwd_dcolx<<<(unsigned)T, 128, 0, st>>>(reinterpret_cast<const __nv_bfloat16 *>(x.data_ptr()),
                                           reinterpret_cast<const __nv_bfloat16 *>(dcol->data_ptr()), dcolx.data_ptr<float>(), (int)H);
  } else {
    dcolx.zero_();
  }
  const int nseg = (int)((T + kSegRows - 1) / kSegRows);
  at::Tensor dpa = at::empty({T, 3, 32}, x.options());
  at::Tensor rowaux = at::empty({T, 8}, f32), da = at::empty({T, 24}, f32), dsc = at::empty({T, 4}, f32);
  TailBwdParams tp{};
  tp.proj = proj.data_ptr<float>(); tp.rstd = rstd.data_ptr<float>();
  tp.dpre_in = dpre_in.has_value() ? dpre_in->data_ptr<float>() : nullptr;
  // the own-pre collapse feeds d_pre; the shifted (external-pre) collapse does not
  tp.dcolx = (has_dcol && !ext_pre.has_value()) ? dcolx.data_ptr<float>() : nullptr;
  tp.ext_pre = ext_pre.has_value() ? ext_pre->data_ptr<float>() : nullptr;
  tp.dpost = dpost.data_ptr<float>(); tp.dcomb = dcomb.data_ptr<float>();
  tp.scale = scale.data_ptr<float>(); tp.base = base.data_ptr<float>();
  tp.dpa = reinterpret_cast<__nv_bfloat16 *>(dpa.data_ptr());
  tp.rowaux = rowaux.data_ptr<float>(); tp.da = da.data_ptr<float>(); tp.dsc = dsc.data_ptr<float>();
  tp.T = (int)T; tp.H = (int)H; tp.iters = (int)iters; tp.eps = (float)eps;
  const size_t hist = (size_t)4 * (2 * iters - 1) * 32 * sizeof(float);
  cudaFuncSetAttribute(bwd_tail, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)hist);
  bwd_tail<<<(unsigned)((T + 3) / 4), 128, hist, st>>>(tp);

  MainParams mp{};
  mp.x = reinterpret_cast<const __nv_bfloat16 *>(x.data_ptr());
  mp.dcol = has_dcol ? reinterpret_cast<const __nv_bfloat16 *>(dcol->data_ptr()) : nullptr;
  mp.fn = fn.data_ptr();
  mp.dpa = tp.dpa; mp.rowaux = tp.rowaux;
  mp.dx = reinterpret_cast<__nv_bfloat16 *>(dx.data_ptr());
  at::Tensor part = at::empty({nseg, 24, 4 * H}, f32);
  mp.dfn_part = part.data_ptr<float>();
  mp.T = (int)T; mp.H = (int)H; mp.ntiles = (int)((T + kTile - 1) / kTile); mp.nseg = nseg;
  if (H == 4096) fn32 ? launch_main<64, true>(mp, nseg, st, (int)groups) : launch_main<64, false>(mp, nseg, st, (int)groups);
  else fn32 ? launch_main<80, true>(mp, nseg, st, (int)groups) : launch_main<80, false>(mp, nseg, st, (int)groups);

  const int64_t n = 24 * 4 * H;
  auto reduce_fn = [&](auto tag, auto out_ptr) {
    using OutT = std::remove_pointer_t<decltype(out_ptr)>;
    const unsigned grid = (unsigned)((n + 255) / 256);
#define RF(M) if (nseg <= M) { bwd_reduce_fn<OutT, M><<<grid, 256, 0, st>>>(part.data_ptr<float>(), out_ptr, nseg, n); } else
    RF(1) RF(2) RF(4) RF(8) RF(16) RF(32) RF(64) { bwd_reduce_fn<OutT, 128><<<grid, 256, 0, st>>>(part.data_ptr<float>(), out_ptr, nseg, n); }
#undef RF
  };
  if (fn32) reduce_fn(0, dfn.data_ptr<float>());
  else reduce_fn(0, reinterpret_cast<__nv_bfloat16 *>(dfn.data_ptr()));
  at::Tensor rpart = at::empty({27, nseg}, f32);
  bwd_rows_partial<<<dim3(27, nseg), 32, 0, st>>>(da.data_ptr<float>(), dsc.data_ptr<float>(), rpart.data_ptr<float>(), (int)T, nseg);
  bwd_rows_final<<<1, 32, 0, st>>>(rpart.data_ptr<float>(), dscale.data_ptr<float>(), dbase.data_ptr<float>(), nseg);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dx, dfn, dscale, dbase, dcolx};
}

}  // namespace mhc_host
