// SPDX-License-Identifier: Apache-2.0
// One decode step per sequence: the chunk recipe of a ONE-ROW chunk (r = 1), in two launches.
//
//   decode_prep    one warp per (sequence, head) row: k^, q^ (FOLD32), G = 0 (+) g LOG2E, kg, qg, kd, E(G_L),
//                  Mq_00 = rn(DIAG(q^, k^)) (T_00 = k^, eighths, T8), xu_00 = rn(1 (*) beta), and u, w by the
//                  same single-k-step mma chains as the chunk path (CHAIN64^b(0)); published as rows.
//   decode_state   a persistent grid over (sequence, head, 64-column value half): the state slice and the
//                  rows of item i + grid stream into shared memory (cp.async) while item i runs
//                    S^ = rn(S);  d = rn(bf(u) (-) CHAIN128(w, S^));  o = rn(CHAIN64^b(0)(Mq, d; CHAIN128(qg, S^)))
//                    S' = CHAIN64^b(0)(kd^T, d; S (*) E(G_L))
//                  -- three barriers an item, nothing of the row preparation on its critical path.
//
// WHY THIS IS NOT A SECOND RECIPE. Every quantity is evaluated through the recipe.cuh definitions on the
// same operand values the two-phase chunk path gives the instruction for a one-row chunk: absent rows
// (1..63) are +0 there (zero-filled loads, +0 coefficients) and +0 here (a shared zero row), and every
// chain is the recipe's block-bounded chain (one k-step over the token index for row 0, eight over
// channels), so each output and state element is the same instruction sequence on the same operands.
// "decode(x) == forward(x) for one-token sequences, outputs and state, bit for bit" is the battery's gate
// (tests/test_relations_gpu.py).
#include <cuda_runtime.h>

#include "params.cuh"
#include "recipe.cuh"

namespace kda_sm100 {

struct DecodeParams {
  const uint16_t *q, *k, *v;       // [N][H][128]
  const void *g, *beta;            // [N][H][128], [N][H]
  int g_bf16, beta_bf16;
  int N, H;
  const float *h0;                 // [N][H][128][128] (nullptr: zero state)
  float *ht;                       // [N][H][128][128] (may alias h0)
  uint16_t *o;                     // [N][H][128]
  // the prep rows: w, u, qg, kd [N][H][128] bf16, E(G_L) [N][H][128] fp32, Mq_00 [N][H][2] bf16 (padded)
  uint16_t *W, *U, *QG, *KD, *MQ;
  float *GL;
};

namespace {

constexpr int THREADS = 256;
constexpr int VS = 64;                           // value columns per work item

// ---------------------------------------------------------------------------- decode_prep
// shared per warp: k^, q^ (fp32 of bf16), G [128] each; v, kg rows (bf16, 256 B each); xu row 0.
struct PrepWarp {
  float kf[128], qf[128], G[128];
  uint16_t v[128], kg[128];
  uint32_t xu[64];                               // Row0 of xu: element 0 live, rest +0
};

}  // namespace

__global__ void __launch_bounds__(THREADS) decode_prep_kernel(DecodeParams p) {
  __shared__ __align__(16) PrepWarp pw[THREADS / 32];
  __shared__ __align__(16) uint16_t zero[128];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  if (threadIdx.x < 64) reinterpret_cast<uint32_t *>(zero)[threadIdx.x] = 0u;
  const int64_t rowid = int64_t(blockIdx.x) * (THREADS / 32) + warp;
  const bool live = rowid < int64_t(p.N) * p.H;
  PrepWarp &s = pw[warp];
  const int64_t row = (live ? rowid : 0) * D;
  // k^, q^ (FOLD32: lane owns 4 lane .. 4 lane + 3), v, G
  #pragma unroll
  for (int which = 0; which < 2; ++which) {
    const uint2 w2 = *reinterpret_cast<const uint2 *>((which == 0 ? p.k : p.q) + row + 4 * lane);
    float x[4] = {bf(lo16(w2.x)), bf(hi16(w2.x)), bf(lo16(w2.y)), bf(hi16(w2.y))};
    float inv = inv_norm(fold32(x));
    if (which == 1) inv = fmul(inv, QSCALE);
    float *dst = which == 0 ? s.kf : s.qf;
    #pragma unroll
    for (int e = 0; e < 4; ++e) dst[4 * lane + e] = bf(rn(fmul(x[e], inv)));
  }
  *reinterpret_cast<uint2 *>(s.v + 4 * lane) = *reinterpret_cast<const uint2 *>(p.v + row + 4 * lane);
  #pragma unroll
  for (int e = 0; e < 4; ++e) {
    const int c = 4 * lane + e;
    const float gv = p.g_bf16 ? bf(static_cast<const uint16_t *>(p.g)[row + c]) : static_cast<const float *>(p.g)[row + c];
    s.G[c] = cum_gate(0.f, gv);
  }
  s.xu[lane] = 0u;
  s.xu[32 + lane] = 0u;
  __syncwarp();
  // kg, qg, kd, E(G_L) (row 0; G_L = G_0): lane owns channels 4 lane .. + 3 as two pairs
  #pragma unroll
  for (int hp = 0; hp < 2; ++hp) {
    const int c0 = 4 * lane + 2 * hp;
    const float k0 = s.kf[c0], k1 = s.kf[c0 + 1], q0 = s.qf[c0], q1 = s.qf[c0 + 1];
    const float G0 = s.G[c0], G1 = s.G[c0 + 1];
    const float eg0 = decay0(G0), eg1 = decay0(G1);
    *reinterpret_cast<uint32_t *>(s.kg + c0) = rn2(fmul(k0, eg0), fmul(k1, eg1));
    if (live) {
      *reinterpret_cast<uint32_t *>(p.QG + row + c0) = rn2(fmul(q0, eg0), fmul(q1, eg1));
      *reinterpret_cast<uint32_t *>(p.KD + row + c0) = rn2(fmul(k0, decay(G0, G0)), fmul(k1, decay(G1, G1)));
      *reinterpret_cast<float2 *>(p.GL + row + c0) = make_float2(decay0(G0), decay0(G1));
    }
  }
  // Mq_00 = rn(DIAG(q^, k^)): lane p = lane & 7 runs eighth p's chain (T_00 = k^), T8 over lanes 0..7
  float kq = 0.f;
  {
    const int qp = lane & 7;
    #pragma unroll
    for (int m = 0; m < 4; ++m) {
      const float4 q4 = *reinterpret_cast<const float4 *>(s.qf + 32 * m + 4 * qp);
      const float4 k4 = *reinterpret_cast<const float4 *>(s.kf + 32 * m + 4 * qp);
      kq = ffma(q4.x, k4.x, kq);
      kq = ffma(q4.y, k4.y, kq);
      kq = ffma(q4.z, k4.z, kq);
      kq = ffma(q4.w, k4.w, kq);
    }
    kq = oct_t8(kq);
  }
  const float beta = live ? (p.beta_bf16 ? bf(static_cast<const uint16_t *>(p.beta)[rowid])
                                         : static_cast<const float *>(p.beta)[rowid]) : 0.f;
  if (lane == 0) {
    reinterpret_cast<uint16_t *>(s.xu)[0] = rn(fmul(1.f, beta));
    if (live) {
      p.MQ[2 * rowid] = rn(kq);
      p.MQ[2 * rowid + 1] = 0;
    }
  }
  __syncthreads();   // xu, kg, v rows visible (the zero row too)
  // u, w (row 0): CHAIN64^b(0) -- one k-step over the token index
  const Row0Tile<64> XU{reinterpret_cast<uint16_t *>(s.xu), zero};
  const Row0Tile<128> Vt{s.v, zero}, KGt{s.kg, zero};
  uint32_t a[4];
  frag_a(a, XU, 0, 0, lane);
  #pragma unroll
  for (int np = 0; np < 8; ++np) {
    uint32_t bv[4], bk[4];
    frag_b2_t(bv, Vt, 16 * np, 0, lane);
    frag_b2_t(bk, KGt, 16 * np, 0, lane);
    float u[2][4] = {}, w[2][4] = {};
    mma(u[0], a, bv[0], bv[1]);
    mma(u[1], a, bv[2], bv[3]);
    mma(w[0], a, bk[0], bk[1]);
    mma(w[1], a, bk[2], bk[3]);
    if (g == 0 && live) {   // row 0 lives in lanes 0..3, elements 0 and 1 of each n8 tile
      #pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        const int c = 16 * np + 8 * nt + 2 * t;
        *reinterpret_cast<uint32_t *>(p.U + row + c) = rn2(u[nt][0], u[nt][1]);
        *reinterpret_cast<uint32_t *>(p.W + row + c) = rn2(w[nt][0], w[nt][1]);
      }
    }
  }
}

// ---------------------------------------------------------------------------- decode_state
namespace {
constexpr int OFF_SB = 0;                        // fp32 [128 k][64 v] state slice, 16-byte chunks swizzled
constexpr int OFF_R0 = OFF_SB + 32768;           // two row buffers (double-buffered with the items)
constexpr int R_W = 0, R_QG = 256, R_KD = 512, R_U = 768, R_GL = 896, R_MQ = 1408, RBUF = 1424;
constexpr int OFF_SH = OFF_R0 + 2 * RBUF;        // Tile<64 v, 128 k>  S^ transposed
constexpr int OFF_DT = OFF_SH + 16384;           // Tile<64 v, 32 i>   d transposed (column 0 live)
constexpr int OFF_ZERO = OFF_DT + 4096;          // 256 B of +0
constexpr int OFF_MQT = OFF_ZERO + 256;          // Row0 of Mq (element 0 live, 128 B)
constexpr int SMEM = OFF_MQT + 128;

__device__ __forceinline__ uint32_t sb_off(int k, int v) {
  return uint32_t(k * 256 + ((((v >> 2) ^ (k & 7))) << 4) + (v & 3) * 4);
}

// Stage item `it`'s state slice and rows (cp.async; a NULL initial state is zero-filled).
__device__ __forceinline__ void prefetch(const DecodeParams &p, char *smem, char *rb, int it, int tid) {
  const int n = it / (2 * p.H), rem = it % (2 * p.H), h = rem >> 1, vs = rem & 1;
  const int64_t nh = int64_t(n) * p.H + h;
  const int64_t sbase = nh * D * D + vs * VS;
  const uint32_t sb = smem_u32(smem + OFF_SB);
  #pragma unroll
  for (int i = 0; i < 8; ++i) {                  // 128 rows x 16 chunks
    const int idx = i * THREADS + tid, k = idx >> 4, c = idx & 15;
    const float *src = p.h0 ? p.h0 + sbase + int64_t(k) * D + 4 * c : nullptr;
    cp16(sb + uint32_t(k * 256 + ((c ^ (k & 7)) << 4)), p.h0 ? (const void *)src : (const void *)p.W, p.h0 != nullptr);
  }
  const int64_t row = nh * D;
  if (tid < 48) {                                // w, qg, kd rows: 16 chunks each
    const int which = tid >> 4, c = tid & 15;
    const uint16_t *src = (which == 0 ? p.W : (which == 1 ? p.QG : p.KD)) + row + 8 * c;
    cp16(smem_u32(rb + which * 256 + 16 * c), src, true);
  } else if (tid < 56) {                         // u slice: 8 chunks
    const int c = tid - 48;
    cp16(smem_u32(rb + R_U + 16 * c), p.U + row + vs * VS + 8 * c, true);
  } else if (tid < 88) {                         // E(G_L): 32 chunks
    const int c = tid - 56;
    cp16(smem_u32(rb + R_GL + 16 * c), p.GL + row + 4 * c, true);
  } else if (tid == 88) {                        // Mq_00 (padded to 4 bytes)
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(smem_u32(rb + R_MQ)), "l"(p.MQ + 2 * nh) : "memory");
  }
}

}  // namespace

__global__ void __launch_bounds__(THREADS, 3) decode_state_kernel(DecodeParams p) {
  extern __shared__ __align__(128) char smem[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int items = p.N * p.H * 2;
  const Tile<64, 128> Sh{reinterpret_cast<uint16_t *>(smem + OFF_SH)};
  const Tile<64, 32> dT{reinterpret_cast<uint16_t *>(smem + OFF_DT)};
  const uint16_t *zr = reinterpret_cast<const uint16_t *>(smem + OFF_ZERO);
  const Row0Tile<64> MQt{reinterpret_cast<uint16_t *>(smem + OFF_MQT), zr};

  for (int i = tid; i < (SMEM - OFF_DT) / 4; i += THREADS) reinterpret_cast<uint32_t *>(smem + OFF_DT)[i] = 0u;
  if (blockIdx.x < items) prefetch(p, smem, smem + OFF_R0, blockIdx.x, tid);
  cp_commit();

  int local = 0;
  #pragma unroll 1
  for (int it = blockIdx.x; it < items; it += gridDim.x, ++local) {
    const int n = it / (2 * p.H), rem = it % (2 * p.H), h = rem >> 1, vs = rem & 1;
    const int64_t row = (int64_t(n) * p.H + h) * D;
    const int64_t sbase = (int64_t(n) * p.H + h) * D * D + vs * VS;
    char *rb = smem + OFF_R0 + (local & 1) * RBUF;
    const Row0Tile<128> Wt{reinterpret_cast<uint16_t *>(rb + R_W), zr};
    const Row0Tile<128> QGt{reinterpret_cast<uint16_t *>(rb + R_QG), zr};
    const Row0Tile<128> KDt{reinterpret_cast<uint16_t *>(rb + R_KD), zr};
    const uint16_t *us = reinterpret_cast<const uint16_t *>(rb + R_U);
    const float *GLs = reinterpret_cast<const float *>(rb + R_GL);
    cp_wait<0>();
    __syncthreads();         // item it staged; every warp is done with the previous item

    // state slice -> registers: S[16 warp + g (+8)][8 nt + 2 t (+1)] ; S^ = rn(S) -> Sh[v][k]
    float S[8][4];
    #pragma unroll
    for (int nt = 0; nt < 8; ++nt)
      #pragma unroll
      for (int eh = 0; eh < 2; ++eh) {
        const float2 s2 = *reinterpret_cast<const float2 *>(smem + OFF_SB + sb_off(16 * warp + g + 8 * eh, 8 * nt + 2 * t));
        S[nt][2 * eh] = s2.x;
        S[nt][2 * eh + 1] = s2.y;
      }
    #pragma unroll
    for (int np = 0; np < 4; ++np) {
      const uint32_t w4[4] = {rn2(S[2 * np][0], S[2 * np][1]), rn2(S[2 * np][2], S[2 * np][3]),
                              rn2(S[2 * np + 1][0], S[2 * np + 1][1]), rn2(S[2 * np + 1][2], S[2 * np + 1][3])};
      store_acc_t(Sh, 16 * warp, 16 * np, w4, lane);
    }
    if (tid == 0) reinterpret_cast<uint16_t *>(smem + OFF_MQT)[0] = *reinterpret_cast<const uint16_t *>(rb + R_MQ);
    __syncthreads();         // Sh ready, the state buffer consumed: stream the next item in
    if (it + gridDim.x < items) prefetch(p, smem, smem + OFF_R0 + ((local + 1) & 1) * RBUF, it + gridDim.x, tid);
    cp_commit();

    // d and the o seed: slice value columns 16 warp .. + 15 (warps 0..3)
    float oacc[2][4] = {};
    if (warp < 4) {
      float dacc[2][4] = {};
      #pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        uint32_t aw[4], aq[4], b[4];
        frag_a(aw, Wt, 0, 16 * ks, lane);
        frag_a(aq, QGt, 0, 16 * ks, lane);
        frag_b2(b, Sh, 16 * warp, 16 * ks, lane);
        mma(dacc[0], aw, b[0], b[1]);
        mma(dacc[1], aw, b[2], b[3]);
        mma(oacc[0], aq, b[0], b[1]);
        mma(oacc[1], aq, b[2], b[3]);
      }
      // only row 0 of d is stored: the tile's other columns hold +0, the chunk path's absent-row d
      if (g == 0) {
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
          const int vc = 16 * warp + 8 * nt + 2 * t;
          const uint32_t up = *reinterpret_cast<const uint32_t *>(us + vc);
          const uint32_t dp = rn2(fsub(bf(lo16(up)), dacc[nt][0]), fsub(bf(hi16(up)), dacc[nt][1]));
          dT.at(vc, 0) = lo16(dp);
          dT.at(vc + 1, 0) = hi16(dp);
        }
      }
    }
    __syncthreads(); 
    // o = rn(CHAIN64^b(0)(Mq, d; seed))
    if (warp < 4) {
      uint32_t am[4], b[4];
      frag_a(am, MQt, 0, 0, lane);
      frag_b2(b, dT, 16 * warp, 0, lane);
      mma(oacc[0], am, b[0], b[1]);
      mma(oacc[1], am, b[2], b[3]);
      if (g == 0) {
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
          const int vc = 16 * warp + 8 * nt + 2 * t;
          *reinterpret_cast<uint32_t *>(p.o + row + vs * VS + vc) = rn2(oacc[nt][0], oacc[nt][1]);
        }
      }
    }
    // S' = CHAIN64^b(0)(kd^T, d; seed = S (*) E(G_L))
    {
      const float gl0 = GLs[16 * warp + g], gl1 = GLs[16 * warp + g + 8];
      #pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        S[nt][0] = fmul(S[nt][0], gl0);
        S[nt][1] = fmul(S[nt][1], gl0);
        S[nt][2] = fmul(S[nt][2], gl1);
        S[nt][3] = fmul(S[nt][3], gl1);
      }
      uint32_t ak[4];
      frag_a_t(ak, KDt, 16 * warp, 0, lane);
      #pragma unroll
      for (int np = 0; np < 4; ++np) {
        uint32_t b[4];
        frag_b2(b, dT, 16 * np, 0, lane);
        mma(S[2 * np], ak, b[0], b[1]);
        mma(S[2 * np + 1], ak, b[2], b[3]);
      }
    }
    #pragma unroll
    for (int nt = 0; nt < 8; ++nt)
      #pragma unroll
      for (int eh = 0; eh < 2; ++eh) {
        const int kr = 16 * warp + g + 8 * eh, vc = 8 * nt + 2 * t;
        *reinterpret_cast<float2 *>(p.ht + sbase + int64_t(kr) * D + vc) = make_float2(S[nt][2 * eh], S[nt][2 * eh + 1]);
      }
  }
  cp_wait<0>();
}

void decode_launch(const DecodeParams &p, cudaStream_t stream) {
  static int grid_cap = 0;
  if (!grid_cap) {
    cudaFuncSetAttribute(decode_state_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    int dev = 0, sms = 0, per = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, decode_state_kernel, THREADS, SMEM);
    grid_cap = sms * (per > 0 ? per : 1);
  }
  const long rows = long(p.N) * p.H;
  decode_prep_kernel<<<unsigned((rows + 7) / 8), THREADS, 0, stream>>>(p);
  const long items = rows * 2;
  const int grid = int(items < grid_cap ? items : grid_cap);
  decode_state_kernel<<<grid, THREADS, SMEM, stream>>>(p);
}

}  // namespace kda_sm100
