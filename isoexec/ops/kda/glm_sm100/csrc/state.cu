// SPDX-License-Identifier: Apache-2.0
// Phase 2 of the chunk recipe: the chunk step from S_c (recipe.cuh, "Chunk step"), one CTA per
// (sequence, head, 64-wide value-column slice), walking the sequence's chunks in order with its fp32 state
// slice S[128 k][64 v] resident in mma accumulators. Value columns never meet in any contraction, so the
// slice width is execution only. 256 threads (8 warps); the workspace of chunk c + 1 is prefetched with
// cp.async while chunk c is evaluated.
//
// Warp decomposition: the state and its update -- warp w owns key rows 16w..16w+15 x 64 value columns;
// d and o -- warp w owns rows j in block w & 3 x value columns 32 (w >> 2) .. +31.
#include <mutex>
#include <cuda_runtime.h>

#include "params.cuh"
#include "recipe.cuh"
#include "record_state.cuh"

namespace kda_sm100 {

namespace {

constexpr int THREADS = 256;
constexpr int BV = 64;

constexpr int ST_W = 0;
constexpr int ST_QG = ST_W + 16384;
constexpr int ST_KD = ST_QG + 16384;
constexpr int ST_MQ = ST_KD + 16384;
constexpr int ST_U = ST_MQ + 8192;
constexpr int ST_GL = ST_U + 8192;
constexpr int STAGE = ST_GL + 512;
constexpr int OFF_SH = 2 * STAGE;                // Tile<64 v, 128 k>: S^ transposed
constexpr int OFF_DT = OFF_SH + 16384;           // Tile<64 v, 64 i>: d transposed
constexpr int SMEM = OFF_DT + 8192;

struct Stage {
  Tile<64, 128> W, QG, KD;
  Tile<64, 64> MQ, U;
  float *GL;
  __device__ Stage(char *base)
      : W{reinterpret_cast<uint16_t *>(base + ST_W)}, QG{reinterpret_cast<uint16_t *>(base + ST_QG)},
        KD{reinterpret_cast<uint16_t *>(base + ST_KD)}, MQ{reinterpret_cast<uint16_t *>(base + ST_MQ)},
        U{reinterpret_cast<uint16_t *>(base + ST_U)}, GL{reinterpret_cast<float *>(base + ST_GL)} {}
};

__device__ __forceinline__ void load_stage(const FwdParams &p, const Stage &s, int chunk, int h, int vs, int tid) {
  const int t0 = p.plan.chunk_wb[chunk], r = p.plan.chunk_rows[chunk];   // workspace rows
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
  for (int it = 0; it < 4; ++it) {              // MQ, U slice: 2 x 64 rows x 8 chunks
    const int idx = it * THREADS + tid;
    const int which = idx >> 9, row = (idx >> 3) & 63, ch = idx & 7;
    const bool valid = row < r;
    const int rr = valid ? row : 0;
    const uint16_t *src = which == 0
        ? p.ws.MQ + (int64_t(t0) + rr) * (int64_t(p.H) * 64) + int64_t(h) * 64 + ch * 8
        : p.ws.U + (int64_t(t0) + rr) * rowH + hoff + vs * BV + ch * 8;
    cp16(which == 0 ? s.MQ.saddr(row, ch) : s.U.saddr(row, ch), src, valid);
  }
  if (tid < 32) cp16(smem_u32(s.GL + 4 * tid), p.ws.GL + (int64_t(chunk) * p.H + h) * D + 4 * tid, true);
}

}  // namespace

// REC as intra_kernel's: the instantiation without records has no record reads or writes (lo = 0).
template <bool REC>
__global__ void __launch_bounds__(THREADS, 1) state_kernel(FwdParams p) {
  extern __shared__ __align__(128) char smem[];
  const int n = blockIdx.y;
  const int h = blockIdx.x / (D / BV), vs = blockIdx.x % (D / BV);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int c_first = p.plan.seq_chunk0[n], nch = p.plan.seq_chunk0[n + 1] - c_first;
  const bool records = REC;
  const int s_in = records ? p.plan.slot_in[n] : -1, s_out = records ? p.slot_out[n] : -1;
  const char *rin = s_in >= 0 ? p.plan.rec.rec(s_in, h) : nullptr;
  char *rout = s_out >= 0 ? p.plan.rec.rec(s_out, h) : nullptr;
  const int H = p.H;
  const Stage st[2] = {Stage(smem), Stage(smem + STAGE)};
  const Tile<64, 128> Sh{reinterpret_cast<uint16_t *>(smem + OFF_SH)};
  const Tile<64, 64> dT{reinterpret_cast<uint16_t *>(smem + OFF_DT)};

  // The state slice: S[16 warp + g (+8)][8 nt + 2 t (+1)] in acc[nt][e].
  float S[8][4];
  const int64_t sbase = (int64_t(n) * H + h) * D * D + vs * BV;
  if (rin) {   // the record's state planes (record_state.cuh): tile (warp, 8 vs + nt)
    #pragma unroll
    for (int nt = 0; nt < 8; ++nt) s_load_tile(S[nt], rin, warp, vs * (BV / 8) + nt, lane);
  } else {
    #pragma unroll
    for (int nt = 0; nt < 8; ++nt)
      #pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int kr = 16 * warp + g + 8 * (e >> 1), vc = 8 * nt + 2 * t + (e & 1);
        S[nt][e] = p.h0 ? p.h0[sbase + int64_t(kr) * D + vc] : 0.f;
      }
  }
  bool saved_sc = false;   // the record's S was written from S_c of an open last chunk

  if (nch > 0) load_stage(p, st[0], c_first, h, vs, tid);
  cp_commit();
  const int mt = warp & 3, nh = warp >> 2;
  #pragma unroll 1
  for (int q = 0; q < nch; ++q) {
    const Stage &s = st[q & 1];
    const int chunk = c_first + q;
    __syncthreads();                       // every warp is done with chunk q - 1 (its stage, Sh, dT)
    if (q + 1 < nch) load_stage(p, st[(q + 1) & 1], chunk + 1, h, vs, tid);
    cp_commit();
    // S^ = rn(S), transposed into Sh[v][k]
    #pragma unroll
    for (int np = 0; np < 4; ++np) {
      const uint32_t w4[4] = {rn2(S[2 * np][0], S[2 * np][1]), rn2(S[2 * np][2], S[2 * np][3]),
                              rn2(S[2 * np + 1][0], S[2 * np + 1][1]), rn2(S[2 * np + 1][2], S[2 * np + 1][3])};
      store_acc_t(Sh, 16 * warp, 16 * np, w4, lane);
    }
    if (p.save_S) {   // backward recompute: S^ of this chunk start, [chunk][h][k][v]
      uint16_t *dst = p.save_S + (int64_t(chunk) * H + h) * D * D + vs * BV;
      #pragma unroll
      for (int nt = 0; nt < 8; ++nt)
        #pragma unroll
        for (int eh = 0; eh < 2; ++eh) {
          const int kr = 16 * warp + g + 8 * eh, vc = 8 * nt + 2 * t;
          *reinterpret_cast<uint32_t *>(dst + int64_t(kr) * D + vc) = rn2(S[nt][2 * eh], S[nt][2 * eh + 1]);
        }
    }
    cp_wait<1>();
    __syncthreads();

    // d and the o seed: rows j in block mt, value columns 32 nh .. + 31
    float dacc[4][4] = {}, oacc[4][4] = {};
    #pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      uint32_t aw[4], aq[4];
      frag_a(aw, s.W, 16 * mt, 16 * ks, lane);
      frag_a(aq, s.QG, 16 * mt, 16 * ks, lane);
      #pragma unroll
      for (int np = 0; np < 2; ++np) {
        uint32_t b[4];
        frag_b2(b, Sh, 32 * nh + 16 * np, 16 * ks, lane);
        mma(dacc[2 * np], aw, b[0], b[1]);
        mma(dacc[2 * np + 1], aw, b[2], b[3]);
        mma(oacc[2 * np], aq, b[0], b[1]);
        mma(oacc[2 * np + 1], aq, b[2], b[3]);
      }
    }
    #pragma unroll
    for (int np = 0; np < 2; ++np) {
      uint32_t w4[4];
      #pragma unroll
      for (int h2 = 0; h2 < 2; ++h2)
        #pragma unroll
        for (int eh = 0; eh < 2; ++eh) {
          const int j = 16 * mt + g + 8 * eh, vc = 32 * nh + 16 * np + 8 * h2 + 2 * t;
          const uint32_t up = *reinterpret_cast<const uint32_t *>(&s.U.at(j, vc));
          const float *a = dacc[2 * np + h2];
          w4[2 * h2 + eh] = rn2(fsub(bf(lo16(up)), a[2 * eh]), fsub(bf(hi16(up)), a[2 * eh + 1]));
        }
      store_acc_t(dT, 16 * mt, 32 * nh + 16 * np, w4, lane);
      if (p.save_d) {
        const int t0 = p.plan.chunk_tok[chunk], r = p.plan.chunk_rows[chunk];
        #pragma unroll
        for (int h2 = 0; h2 < 2; ++h2)
          #pragma unroll
          for (int eh = 0; eh < 2; ++eh) {
            const int j = 16 * mt + g + 8 * eh, vc = 32 * nh + 16 * np + 8 * h2 + 2 * t;
            if (j < r)
              *reinterpret_cast<uint32_t *>(p.save_d + (int64_t(t0) + j) * (int64_t(H) * D) + int64_t(h) * D +
                                            vs * BV + vc) = w4[2 * h2 + eh];
          }
      }
    }
    __syncthreads();

    // o = rn(CHAIN64_i(Mq, d; seed)) ; staged in the stage's U tile (U's last reader was before the barrier)
    #pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
#ifndef KDA_PLAIN_CHAINS
      if (ks > mt) break;                          // CHAIN64^b(j)
#endif
      uint32_t am[4];
      frag_a(am, s.MQ, 16 * mt, 16 * ks, lane);
      #pragma unroll
      for (int np = 0; np < 2; ++np) {
        uint32_t b[4];
        frag_b2(b, dT, 32 * nh + 16 * np, 16 * ks, lane);
        mma(oacc[2 * np], am, b[0], b[1]);
        mma(oacc[2 * np + 1], am, b[2], b[3]);
      }
    }
    #pragma unroll
    for (int np = 0; np < 2; ++np) {
      const uint32_t w4[4] = {rn2(oacc[2 * np][0], oacc[2 * np][1]), rn2(oacc[2 * np][2], oacc[2 * np][3]),
                              rn2(oacc[2 * np + 1][0], oacc[2 * np + 1][1]),
                              rn2(oacc[2 * np + 1][2], oacc[2 * np + 1][3])};
      store_acc(s.U, 16 * mt, 32 * nh + 16 * np, w4, lane);
    }
    // S_{c+1} = CHAIN64_i(kd^T, d; seed = S (*) e(G_L))
    if (rout && q == nch - 1 && p.plan.chunk_rows[chunk] < CL) {
      // the sequence stops inside this chunk: its record keeps S_c and the open rows' d
      saved_sc = true;
      #pragma unroll
      for (int nt = 0; nt < 8; ++nt) s_store_tile(rout, S[nt], warp, vs * (BV / 8) + nt, lane);
      const int r = p.plan.chunk_rows[chunk];
      uint16_t *dstD = reinterpret_cast<uint16_t *>(rout + Rec::DB);   // d rows [i][v] of this slice
      for (int idx = tid; idx < r * 32; idx += THREADS) {
        const int i = idx >> 5, v2 = 2 * (idx & 31);
        *reinterpret_cast<uint32_t *>(dstD + i * D + vs * BV + v2) = pack(dT.at(v2, i), dT.at(v2 + 1, i));
      }
    }
    if (!REC || q < nch - 1 || p.plan.chunk_rows[chunk] == CL || p.ht) {
      const float gl0 = s.GL[16 * warp + g], gl1 = s.GL[16 * warp + g + 8];
      #pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        S[nt][0] = fmul(S[nt][0], gl0);
        S[nt][1] = fmul(S[nt][1], gl0);
        S[nt][2] = fmul(S[nt][2], gl1);
        S[nt][3] = fmul(S[nt][3], gl1);
      }
      const int kmax = (p.plan.chunk_rows[chunk] - 1) >> 4;   // CHAIN64^b(L)
      #pragma unroll
      for (int ks = 0; ks < 4; ++ks) {
#ifndef KDA_PLAIN_CHAINS
        if (ks > kmax) break;
#endif
        uint32_t ak[4];
        frag_a_t(ak, s.KD, 16 * warp, 16 * ks, lane);
        #pragma unroll
        for (int np = 0; np < 4; ++np) {
          uint32_t b[4];
          frag_b2(b, dT, 16 * np, 16 * ks, lane);
          mma(S[2 * np], ak, b[0], b[1]);
          mma(S[2 * np + 1], ak, b[2], b[3]);
        }
      }
    }
    __syncthreads();
    {
      const int t0 = p.plan.chunk_tok[chunk], r = p.plan.chunk_rows[chunk], lo = REC ? p.plan.chunk_lo[chunk] : 0;
      for (int idx = tid; idx < 64 * 8; idx += THREADS) {
        const int row = idx >> 3, ch = idx & 7;
        if (row >= lo && row < r)
          *reinterpret_cast<uint4 *>(p.o + (int64_t(t0) + row) * (int64_t(H) * D) + int64_t(h) * D + vs * BV + ch * 8) =
              *reinterpret_cast<const uint4 *>(s.U.chunk(row, ch));
      }
    }
  }
  cp_wait<0>();
  if (rout) {
    const int len = p.plan.cu[n + 1] - p.plan.cu[n];
    if (nch == 0 && rin == rout) {
      // nothing to do: the record is its own continuation
    } else if (nch == 0 && rin) {   // a zero-token move: copy the record, half a CTA pair each
      constexpr int64_t units = Rec::BYTES / 16, half = (units + 1) / 2;   // the whole record, POS included
      const uint4 *src = reinterpret_cast<const uint4 *>(rin);
      uint4 *dst = reinterpret_cast<uint4 *>(rout);
      for (int64_t i = vs * half + tid; i < (vs ? units : half); i += THREADS) dst[i] = src[i];
    } else {
      if (!saved_sc) {   // a complete last chunk or a fresh empty sequence: S, S^ of the current S
        #pragma unroll
        for (int nt = 0; nt < 8; ++nt) s_store_tile(rout, S[nt], warp, vs * (BV / 8) + nt, lane);
        __syncthreads();
        #pragma unroll
        for (int np = 0; np < 4; ++np) {
          const uint32_t w4[4] = {rn2(S[2 * np][0], S[2 * np][1]), rn2(S[2 * np][2], S[2 * np][3]),
                                  rn2(S[2 * np + 1][0], S[2 * np + 1][1]), rn2(S[2 * np + 1][2], S[2 * np + 1][3])};
          store_acc_t(Sh, 16 * warp, 16 * np, w4, lane);
        }
        __syncthreads();
      }
      // S^ image rows [v][k] of this slice from Sh (rn(S_c) of an open chunk, else rn(S))
      uint16_t *dstH = reinterpret_cast<uint16_t *>(rout + Rec::SH);
      for (int idx = tid; idx < 64 * 16; idx += THREADS) {
        const int v = idx >> 4, ch = idx & 15;
        *reinterpret_cast<uint4 *>(dstH + int64_t(vs * BV + v) * D + ch * 8) = *reinterpret_cast<const uint4 *>(Sh.chunk(v, ch));
      }
      // the open rows (from phase 1's carry), split between the two value-slice CTAs of the head
      const int jf = saved_sc ? p.plan.chunk_rows[c_first + nch - 1] : 0;
      if (jf > 0) {
        const char *cr = p.carry + (int64_t(n) * p.H + h) * Carry::BYTES;
        auto copy = [&](int64_t dofs, int64_t sofs, int bytes) {
          for (int i = tid; i < bytes / 16; i += THREADS)
            reinterpret_cast<uint4 *>(rout + dofs)[i] = reinterpret_cast<const uint4 *>(cr + sofs)[i];
        };
        if (vs == 0) {
          copy(Rec::KB, Carry::KB, jf * D * 2);
          copy(Rec::GB, Carry::GB, jf * D * 4);
          copy(Rec::BB, Carry::BB, 256);
          copy(Rec::VB, Carry::VB, jf * D * 2);
        } else {
          copy(Rec::KGB, Carry::KGB, jf * D * 2);
          if (KI_STORED && ((jf - 1) >> 4) >= 1 && (jf & 15)) copy(Rec::KI, Carry::KI, 16 * ((jf - 1) >> 4) * D * 2);
          copy(Rec::XB, Carry::XB, xrow_off(jf) * 4);
          copy(Rec::PB, Carry::PB, 16 * P_COLS * 4);
        }
      }
      if (vs == 0 && tid == 0) *p.plan.rec.pos(s_out, h) = p.plan.seq_pos[n] + len;
    }
  }
  if (p.ht) {
    #pragma unroll
    for (int nt = 0; nt < 8; ++nt)
      #pragma unroll
      for (int eh = 0; eh < 2; ++eh) {
        const int kr = 16 * warp + g + 8 * eh, vc = 8 * nt + 2 * t;
        *reinterpret_cast<float2 *>(p.ht + sbase + int64_t(kr) * D + vc) = make_float2(S[nt][2 * eh], S[nt][2 * eh + 1]);
      }
  }
}

void state_launch(const FwdParams &p, cudaStream_t stream) {
  static std::once_flag attr[64];  // the attribute is per device
  int attr_dev = 0;
  cudaGetDevice(&attr_dev);
  std::call_once(attr[unsigned(attr_dev) % 64u], [&] {
    cudaFuncSetAttribute(state_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    cudaFuncSetAttribute(state_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
  });
  dim3 grid(p.H * (D / BV), p.plan.nseq);
  if (p.plan.rec.base) state_kernel<true><<<grid, THREADS, SMEM, stream>>>(p);
  else state_kernel<false><<<grid, THREADS, SMEM, stream>>>(p);
}

}  // namespace kda_sm100
