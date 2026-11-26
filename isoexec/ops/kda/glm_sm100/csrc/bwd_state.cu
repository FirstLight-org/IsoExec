// SPDX-License-Identifier: Apache-2.0
// Backward of the chunk step, reverse over each sequence's chunks: one CTA per (sequence, head, 64-wide
// value slice) with the fp32 state gradient dS[128 k][64 v] in mma accumulators (value columns never
// meet, as in the forward). For chunk c, entered with dS = dL/dS_{c+1}:
//   dd   = Mq^T dO + kd rn(dS)                               (dL/dd = dL/du; stored as rn(dd))
//   dS_c = dS (*) E(G_L) + qg^T dO - w^T rn(dd)
// and rn(dS_{c+1}) is stored per chunk for the parallel contraction kernel. No atomics: every output
// element has one writer.
#include <mutex>
#include <cuda_runtime.h>

#include "params.cuh"
#include "recipe.cuh"

namespace kda_sm100 {

namespace {

constexpr int THREADS = 256;
constexpr int BV = 64;

constexpr int ST_W = 0;
constexpr int ST_QG = ST_W + 16384;
constexpr int ST_KD = ST_QG + 16384;
constexpr int ST_MQ = ST_KD + 16384;
constexpr int ST_DO = ST_MQ + 8192;
constexpr int ST_GL = ST_DO + 8192;
constexpr int STAGE = ST_GL + 512;
constexpr int OFF_DSH = 2 * STAGE;               // Tile<64 v, 128 k>: rn(dS) transposed
constexpr int OFF_NDD = OFF_DSH + 16384;         // Tile<64 j, 64 v>: rn(-dd)
constexpr int SMEM = OFF_NDD + 8192;

struct Stage {
  Tile<64, 128> W, QG, KD;
  Tile<64, 64> MQ, DO;
  float *GL;
  __device__ Stage(char *base)
      : W{reinterpret_cast<uint16_t *>(base + ST_W)}, QG{reinterpret_cast<uint16_t *>(base + ST_QG)},
        KD{reinterpret_cast<uint16_t *>(base + ST_KD)}, MQ{reinterpret_cast<uint16_t *>(base + ST_MQ)},
        DO{reinterpret_cast<uint16_t *>(base + ST_DO)}, GL{reinterpret_cast<float *>(base + ST_GL)} {}
};

__device__ __forceinline__ void load_stage(const BwdParams &p, const Stage &s, int chunk, int h, int vs, int tid) {
  const int t0 = p.plan.chunk_tok[chunk], r = p.plan.chunk_rows[chunk];
  const int64_t rowH = int64_t(p.H) * D, hoff = int64_t(h) * D;
  #pragma unroll
  for (int it = 0; it < 12; ++it) {             // W, QG, KD: 3 x 64 rows x 16 chunks
    const int idx = it * THREADS + tid;
    const int which = idx >> 10, row = (idx >> 4) & 63, ch = idx & 15;
    const bool valid = row < r;
    const uint16_t *base = which == 0 ? p.ws.W : (which == 1 ? p.ws.QG : p.ws.KD);
    const Tile<64, 128> &dst = which == 0 ? s.W : (which == 1 ? s.QG : s.KD);
    cp16(dst.saddr(row, ch), base + (int64_t(t0) + (valid ? row : 0)) * rowH + hoff + ch * 8, valid);
  }
  #pragma unroll
  for (int it = 0; it < 4; ++it) {              // MQ, dO slice: 2 x 64 rows x 8 chunks
    const int idx = it * THREADS + tid;
    const int which = idx >> 9, row = (idx >> 3) & 63, ch = idx & 7;
    const bool valid = row < r;
    const int rr = valid ? row : 0;
    const uint16_t *src = which == 0
        ? p.ws.MQ + (int64_t(t0) + rr) * (int64_t(p.H) * 64) + int64_t(h) * 64 + ch * 8
        : p.dO + (int64_t(t0) + rr) * rowH + hoff + vs * BV + ch * 8;
    cp16(which == 0 ? s.MQ.saddr(row, ch) : s.DO.saddr(row, ch), src, valid);
  }
  if (tid < 32) cp16(smem_u32(s.GL + 4 * tid), p.ws.GL + (int64_t(chunk) * p.H + h) * D + 4 * tid, true);
}

}  // namespace

__global__ void __launch_bounds__(THREADS, 1) bwd_state_kernel(BwdParams p) {
  extern __shared__ __align__(128) char smem[];
  const int n = blockIdx.y;
  const int h = blockIdx.x / (D / BV), vs = blockIdx.x % (D / BV);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int c_first = p.plan.seq_chunk0[n], nch = p.plan.seq_chunk0[n + 1] - c_first;
  const int H = p.H;
  const Stage st[2] = {Stage(smem), Stage(smem + STAGE)};
  const Tile<64, 128> dSh{reinterpret_cast<uint16_t *>(smem + OFF_DSH)};
  const Tile<64, 64> NDD{reinterpret_cast<uint16_t *>(smem + OFF_NDD)};

  // dS[16 warp + g (+8)][8 nt + 2 t (+1)]
  float dS[8][4];
  const int64_t sbase = (int64_t(n) * H + h) * D * D + vs * BV;
  #pragma unroll
  for (int nt = 0; nt < 8; ++nt)
    #pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int kr = 16 * warp + g + 8 * (e >> 1), vc = 8 * nt + 2 * t + (e & 1);
      dS[nt][e] = p.dht ? p.dht[sbase + int64_t(kr) * D + vc] : 0.f;
    }

  if (nch > 0) load_stage(p, st[(nch - 1) & 1], c_first + nch - 1, h, vs, tid);
  cp_commit();
  const int mt = warp & 3, nh = warp >> 2;
  #pragma unroll 1
  for (int q = nch - 1; q >= 0; --q) {
    const Stage &s = st[q & 1];
    const int chunk = c_first + q;
    const int t0 = p.plan.chunk_tok[chunk], r = p.plan.chunk_rows[chunk];
    __syncthreads();                       // every warp is done with chunk q + 1
    if (q > 0) load_stage(p, st[(q - 1) & 1], chunk - 1, h, vs, tid);
    cp_commit();
    // rn(dS_{c+1}) -> dSh[v][k] and the per-chunk save [chunk][h][k][v]
    {
      uint16_t *dst = p.dSn + (int64_t(chunk) * H + h) * D * D + vs * BV;
      #pragma unroll
      for (int np = 0; np < 4; ++np) {
        const uint32_t w4[4] = {rn2(dS[2 * np][0], dS[2 * np][1]), rn2(dS[2 * np][2], dS[2 * np][3]),
                                rn2(dS[2 * np + 1][0], dS[2 * np + 1][1]), rn2(dS[2 * np + 1][2], dS[2 * np + 1][3])};
        store_acc_t(dSh, 16 * warp, 16 * np, w4, lane);
        #pragma unroll
        for (int h2 = 0; h2 < 2; ++h2)
          #pragma unroll
          for (int eh = 0; eh < 2; ++eh)
            *reinterpret_cast<uint32_t *>(dst + int64_t(16 * warp + g + 8 * eh) * D + 16 * np + 8 * h2 + 2 * t) =
                w4[2 * h2 + eh];
      }
    }
    cp_wait<1>();
    __syncthreads();

    // dd = Mq^T dO + kd rn(dS): rows i in block mt, slice columns 32 nh .. + 31
    {
      float acc[4][4] = {};
      #pragma unroll
      for (int ks = 0; ks < 4; ++ks) {
        if (ks < mt) continue;                    // Mq_ji = 0 for j < i
        uint32_t a[4];
        frag_a_t(a, s.MQ, 16 * mt, 16 * ks, lane);
        #pragma unroll
        for (int np = 0; np < 2; ++np) {
          uint32_t b[4];
          frag_b2_t(b, s.DO, 32 * nh + 16 * np, 16 * ks, lane);
          mma(acc[2 * np], a, b[0], b[1]);
          mma(acc[2 * np + 1], a, b[2], b[3]);
        }
      }
      #pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        uint32_t a[4];
        frag_a(a, s.KD, 16 * mt, 16 * ks, lane);
        #pragma unroll
        for (int np = 0; np < 2; ++np) {
          uint32_t b[4];
          frag_b2(b, dSh, 32 * nh + 16 * np, 16 * ks, lane);
          mma(acc[2 * np], a, b[0], b[1]);
          mma(acc[2 * np + 1], a, b[2], b[3]);
        }
      }
      #pragma unroll
      for (int np = 0; np < 2; ++np) {
        const uint32_t w4[4] = {rn2(-acc[2 * np][0], -acc[2 * np][1]), rn2(-acc[2 * np][2], -acc[2 * np][3]),
                                rn2(-acc[2 * np + 1][0], -acc[2 * np + 1][1]),
                                rn2(-acc[2 * np + 1][2], -acc[2 * np + 1][3])};
        store_acc(NDD, 16 * mt, 32 * nh + 16 * np, w4, lane);
        #pragma unroll
        for (int h2 = 0; h2 < 2; ++h2)
          #pragma unroll
          for (int eh = 0; eh < 2; ++eh) {
            const int i = 16 * mt + g + 8 * eh, vc = 32 * nh + 16 * np + 8 * h2 + 2 * t;
            if (i < r)
              *reinterpret_cast<uint32_t *>(p.dd + (int64_t(t0) + i) * (int64_t(H) * D) + int64_t(h) * D + vs * BV + vc) =
                  w4[2 * h2 + eh] ^ 0x80008000u;   // rn(dd) = -rn(-dd): RN is sign-symmetric
          }
      }
    }
    __syncthreads();

    // dS_c = dS (*) E(G_L) + qg^T dO - w^T rn(dd)
    {
      const float gl0 = s.GL[16 * warp + g], gl1 = s.GL[16 * warp + g + 8];
      #pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        dS[nt][0] = fmul(dS[nt][0], gl0);
        dS[nt][1] = fmul(dS[nt][1], gl0);
        dS[nt][2] = fmul(dS[nt][2], gl1);
        dS[nt][3] = fmul(dS[nt][3], gl1);
      }
      const int kmax = (r - 1) >> 4;
      #pragma unroll
      for (int ks = 0; ks < 4; ++ks) {
        if (ks > kmax) break;
        uint32_t aq[4], aw[4];
        frag_a_t(aq, s.QG, 16 * warp, 16 * ks, lane);
        frag_a_t(aw, s.W, 16 * warp, 16 * ks, lane);
        #pragma unroll
        for (int np = 0; np < 4; ++np) {
          uint32_t bo[4], bd[4];
          frag_b2_t(bo, s.DO, 16 * np, 16 * ks, lane);
          frag_b2_t(bd, NDD, 16 * np, 16 * ks, lane);
          mma(dS[2 * np], aq, bo[0], bo[1]);
          mma(dS[2 * np + 1], aq, bo[2], bo[3]);
          mma(dS[2 * np], aw, bd[0], bd[1]);
          mma(dS[2 * np + 1], aw, bd[2], bd[3]);
        }
      }
    }
  }
  cp_wait<0>();
  #pragma unroll
  for (int nt = 0; nt < 8; ++nt)
    #pragma unroll
    for (int eh = 0; eh < 2; ++eh) {
      const int kr = 16 * warp + g + 8 * eh, vc = 8 * nt + 2 * t;
      *reinterpret_cast<float2 *>(p.dh0 + sbase + int64_t(kr) * D + vc) = make_float2(dS[nt][2 * eh], dS[nt][2 * eh + 1]);
    }
}

void bwd_state_launch(const BwdParams &p, cudaStream_t stream) {
  static std::once_flag attr[64];  // the attribute is per device
  int attr_dev = 0;
  cudaGetDevice(&attr_dev);
  std::call_once(attr[unsigned(attr_dev) % 64u], [&] {
    cudaFuncSetAttribute(bwd_state_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
  });
  dim3 grid(p.H * (D / BV), p.plan.nseq);
  bwd_state_kernel<<<grid, THREADS, SMEM, stream>>>(p);
}

}  // namespace kda_sm100
