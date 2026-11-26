// SPDX-License-Identifier: Apache-2.0
// Phase 1 of the chunk recipe: everything of one (chunk, head) that does not depend on the state --
// k^, q^, G, the decay-factored operands, KK / KQ, X = (I + L)^-1, u, w -- published to the token-indexed
// workspace (params.cuh). 256 threads, one CTA per (chunk, head), 105 KB of shared memory so that two
// CTAs share an SM (most phases are latency-bound; the second CTA is what hides them). The phases are
// intra_common.cuh's, which the backward recompute calls too.
//
// Warp decomposition (execution only):
//   l2 norm     warp w folds rows 8w..8w+7 of k and of q (lane l owns channels 4l..4l+3).
//   G           threads 0..127 own one channel each and run the chunk's ascending chain.
//   factors     thread per (row, channel pair) for kg, qg, kd.
//   off-diag    warp w < 6 owns one block pair, decay-factored operands formed in registers.
//   diagonal    a quad of lanes per (j, i) pair, lane p of the quad owning channel quarter p.
//   X solve     phase a: thread per column of a diagonal block; phase b: rounds B = 1, 2, 3.
//   u, w        warp w: row block w & 3, column half w >> 2.
#include <mutex>
#include <cuda_runtime.h>

#include "intra_common.cuh"
#include "params.cuh"

namespace kda_sm100 {

namespace {

constexpr int OFF_KH = 0;                        // k^ ; u staging (P8)
constexpr int OFF_QH = OFF_KH + 16384;           // q^ ; v (loaded after the products)
constexpr int OFF_KG = OFF_QH + 16384;           // raw bf16 g (P0) ; kg
constexpr int OFF_G = OFF_KG + 16384;            // float [64][128] G ; then X, P, xu
constexpr int OFF_BETA = OFF_G + 64 * 128 * 4;   // float [64]
constexpr int OFF_MQ = OFF_BETA + 256;           // Tile<64, 64>
constexpr int OFF_KK = OFF_MQ + 8192;            // float [64][LS] KK ; w staging (P8)
constexpr int SMEM = OFF_KK + 64 * LS * 4;
constexpr int OFF_X = OFF_G;                     // float [64][LS]
constexpr int OFF_P = OFF_X + 64 * LS * 4;       // float [16][LS]
constexpr int OFF_XU = OFF_P + 16 * LS * 4;      // Tile<64, 64>
static_assert(OFF_XU + 8192 <= OFF_BETA, "X, P, xu fit in G's space");
static_assert(64 * LS * 4 >= 16384, "w staging fits in KK's space");

}  // namespace

// REC: the call has engine records (launched as REC = plan.rec.base != nullptr). Without records every
// chunk starts at row 0 (lo = 0), so the instantiation drops the record and carry paths; the arithmetic of
// every row is the same.
template <bool REC>
__global__ void __launch_bounds__(NT, 2) intra_kernel(FwdParams p) {
  extern __shared__ __align__(128) char smem[];
  const int chunk = blockIdx.x, h = blockIdx.y;
  if (chunk >= p.plan.count[0]) return;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  
  const int t0 = p.plan.chunk_tok[chunk];
  const int r = p.plan.chunk_rows[chunk], lo = REC ? p.plan.chunk_lo[chunk] : 0, wb = p.plan.chunk_wb[chunk];
  const int seq = p.plan.chunk_seq[chunk];
  const bool records = REC;
  // the open chunk's rows [0, lo) come from the input record; the last chunk's open rows go to the carry
  const char *rin = (records && lo > 0) ? p.plan.rec.rec(p.plan.slot_in[seq], h) : nullptr;
  const bool is_last = chunk == p.plan.seq_chunk0[seq + 1] - 1;
  char *cr = (records && is_last && r < CL) ? p.carry + (int64_t(seq) * p.H + h) * Carry::BYTES : nullptr;
  const int H = p.H;
  const int64_t rowH = int64_t(H) * D;   // elements between consecutive tokens of one head
  const int64_t hoff = int64_t(h) * D;

  const T128 Kh{reinterpret_cast<uint16_t *>(smem + OFF_KH)};
  const T128 Qh{reinterpret_cast<uint16_t *>(smem + OFF_QH)};
  const T128 Kg{reinterpret_cast<uint16_t *>(smem + OFF_KG)};
  float *G = reinterpret_cast<float *>(smem + OFF_G);
  float *beta = reinterpret_cast<float *>(smem + OFF_BETA);
  const T64 Mq{reinterpret_cast<uint16_t *>(smem + OFF_MQ)};
  float *KK = reinterpret_cast<float *>(smem + OFF_KK);

  // ---------------------------------------------------------------- P0: loads and zero fills
  load_rows128(Kh, p.k, t0, lo, r, rowH, hoff, tid);
  load_rows128(Qh, p.q, t0, lo, r, rowH, hoff, tid);
  for (int idx = tid; idx < lo * 16; idx += NT)   // q of the record's rows: +0 (no output of theirs is made)
    *reinterpret_cast<uint4 *>(Qh.chunk(idx >> 4, idx & 15)) = make_uint4(0u, 0u, 0u, 0u);
  load_g(p.g, p.g_bf16, G, Kg.base, t0, r, rowH, hoff, tid, lo);
  if (rin) {   // the record's k^ and G rows [0, lo)
    load_rec_rows128(Kh, reinterpret_cast<const uint16_t *>(rin + Rec::KB), lo, tid);
    const float *gr = reinterpret_cast<const float *>(rin + Rec::GB);
    for (int idx = tid; idx < lo * 32; idx += NT) cp16(smem_u32(G + 4 * idx), gr + 4 * idx, true);
  }
  cp_commit();
  for (int i = tid; i < 64 * LS; i += NT) KK[i] = 0.f;
  for (int i = tid; i < 64 * 64 / 2; i += NT) reinterpret_cast<uint32_t *>(Mq.base)[i] = 0u;
  if (tid < 64)
    beta[tid] = tid < lo ? reinterpret_cast<const float *>(rin + Rec::BB)[tid]
              : tid < r ? ldg_val(p.beta, (int64_t(t0) + tid) * H + h, p.beta_bf16) : 0.f;
  cp_wait<0>();
  __syncthreads(); 

  // ---------------------------------------------------------------- P1: l2 norms (in place), G chain
  l2norm_tile(Kh, false, nullptr, warp, lane, lo);
  l2norm_tile(Qh, true, nullptr, warp, lane);
  g_chain(G, Kg.base, p.g_bf16, tid, lo);
  __syncthreads(); 

  // ---------------------------------------------------------------- P3: kg, qg, kd, E(G_L)
  #pragma unroll 4
  for (int it = 0; it < 16; ++it) {
    const int task = it * NT + tid;
    const int j = task >> 6, c0 = 2 * (task & 63);
    const uint32_t kp = *reinterpret_cast<const uint32_t *>(&Kh.at(j, c0));
    const float2 Gj = *reinterpret_cast<const float2 *>(G + j * D + c0);
    const float eg0 = decay0(Gj.x), eg1 = decay0(Gj.y);
    *reinterpret_cast<uint32_t *>(&Kg.at(j, c0)) = scale2e(kp, eg0, eg1);
    if (j < r) {
      const uint32_t qp = *reinterpret_cast<const uint32_t *>(&Qh.at(j, c0));
      const float2 GL = *reinterpret_cast<const float2 *>(G + (r - 1) * D + c0);
      const int64_t gi = (int64_t(wb) + j) * rowH + hoff + c0;
      *reinterpret_cast<uint32_t *>(p.ws.QG + gi) = scale2e(qp, eg0, eg1);
      *reinterpret_cast<uint32_t *>(p.ws.KD + gi) = scale2(kp, GL, Gj);
      if (cr) {   // the open rows' k^ and G ; the open block's ki^b (rows below the block)
        *reinterpret_cast<uint32_t *>(cr + Carry::KB + (j * D + c0) * 2) = kp;
        *reinterpret_cast<float2 *>(cr + Carry::GB + (j * D + c0) * 4) = Gj;
        const int bo = (r - 1) >> 4;
        if (KI_STORED && bo >= 1 && (r & 15) && j < 16 * bo)
          *reinterpret_cast<uint32_t *>(cr + Carry::KI + (j * D + c0) * 2) =
              scale2(kp, *reinterpret_cast<const float2 *>(G + 16 * bo * D + c0), Gj);
      }
    }
  }
  if (tid < D) p.ws.GL[(int64_t(chunk) * H + h) * D + tid] = decay0(G[(r - 1) * D + tid]);
  if (cr && tid < 64) reinterpret_cast<float *>(cr + Carry::BB)[tid] = tid < r ? beta[tid] : 0.f;
  if (cr && (r - 1) >> 4 == 0)   // no earlier block: the P rows are all +0
    for (int i = tid; i < 16 * P_COLS; i += NT) reinterpret_cast<float *>(cr + Carry::PB)[i] = 0.f;

  // ---------------------------------------------------------------- P5: off-diagonal KK and Mq (G's last use)
  offdiag_products<true>(Kh, Qh, G, KK, Mq, warp, lane);
  __syncthreads(); 
  // ---------------------------------------------------------------- P4: diagonal blocks (delta over G)
  delta_in_place(G, tid);
  __syncthreads();
  diag_products<true>(Kh, Qh, G, KK, Mq, warp, lane);
  __syncthreads(); 

  // ---------------------------------------------------------------- v into q^'s space (async), Mq out
  load_rows128(Qh, p.v, t0, lo, r, rowH, hoff, tid);
  if (rin) load_rec_rows128(Qh, reinterpret_cast<const uint16_t *>(rin + Rec::VB), lo, tid);
  cp_commit();
  for (int idx = tid; idx < 64 * 8; idx += NT) {
    const int row = idx >> 3, ch = idx & 7;
    if (row < r)
      *reinterpret_cast<uint4 *>(p.ws.MQ + (int64_t(wb) + row) * (int64_t(H) * 64) + int64_t(h) * 64 + ch * 8) =
          *reinterpret_cast<const uint4 *>(Mq.chunk(row, ch));
  }
  // ---------------------------------------------------------------- P6/P7: X solve (G is dead), xu
  float *X = reinterpret_cast<float *>(smem + OFF_X);
  float *P = reinterpret_cast<float *>(smem + OFF_P);
  x_solve(KK, beta, X, P, tid, cr ? reinterpret_cast<float *>(cr + Carry::PB) : nullptr, cr ? (r - 1) >> 4 : -1);
  
  const T64 Xu{reinterpret_cast<uint16_t *>(smem + OFF_XU)};
  make_xu(X, beta, Xu, tid);
  if (p.save_X) {   // backward recompute: X, KK (fp32), kg rows
    const int64_t rH64 = int64_t(H) * 64, h64 = int64_t(h) * 64;
    for (int idx = tid; idx < 64 * 64; idx += NT) {
      const int j = idx >> 6, i = idx & 63;
      if (j < r) {
        const int64_t gi = (int64_t(t0) + j) * rH64 + h64 + i;
        p.save_X[gi] = i <= j ? X[j * LS + i] : 0.f;
        p.save_KK[gi] = KK[j * LS + i];
      }
    }
    for (int idx = tid; idx < 64 * 16; idx += NT) {
      const int row = idx >> 4, ch = idx & 15;
      if (row < r)
        *reinterpret_cast<uint4 *>(p.save_Kg + (int64_t(t0) + row) * rowH + hoff + ch * 8) =
            *reinterpret_cast<const uint4 *>(Kg.chunk(row, ch));
    }
  }
  if (cr)
    for (int idx = tid; idx < r * 64; idx += NT) {   // packed rows (record layout)
      const int j = idx >> 6, i = idx & 63;
      if (i < xrow_len(j)) reinterpret_cast<float *>(cr + Carry::XB)[xrow_off(j) + i] = i <= j ? X[j * LS + i] : 0.f;
    }
  cp_wait<0>();
  __syncthreads(); 
  if (cr)   // the open rows' v (q^'s space holds v now) and kg
    for (int idx = tid; idx < r * 16; idx += NT) {
      const int row = idx >> 4, ch = idx & 15;
      *reinterpret_cast<uint4 *>(cr + Carry::VB + (row * D + ch * 8) * 2) = *reinterpret_cast<const uint4 *>(Qh.chunk(row, ch));
      *reinterpret_cast<uint4 *>(cr + Carry::KGB + (row * D + ch * 8) * 2) = *reinterpret_cast<const uint4 *>(Kg.chunk(row, ch));
    }
  if (p.save_Xu) {
    for (int idx = tid; idx < 64 * 8; idx += NT) {
      const int row = idx >> 3, ch = idx & 7;
      if (row < r)
        *reinterpret_cast<uint4 *>(p.save_Xu + (int64_t(t0) + row) * (int64_t(H) * 64) + int64_t(h) * 64 + ch * 8) =
            *reinterpret_cast<const uint4 *>(Xu.chunk(row, ch));
    }
  }

  // ---------------------------------------------------------------- P8: u, w (CHAIN64^b(j))
  const T128 &Vt = Qh;
  const T128 Ws{reinterpret_cast<uint16_t *>(smem + OFF_KK)};
  {
    const int mt = warp & 3, ch = warp >> 2;
    float u[8][4] = {}, w[8][4] = {};
    #pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
#ifndef KDA_PLAIN_CHAINS
      if (ks > mt) break;
#endif
      uint32_t a[4];
      frag_a(a, Xu, 16 * mt, 16 * ks, lane);
      #pragma unroll
      for (int np = 0; np < 4; ++np) {
        uint32_t bv[4], bk[4];
        frag_b2_t(bv, Vt, 64 * ch + 16 * np, 16 * ks, lane);
        frag_b2_t(bk, Kg, 64 * ch + 16 * np, 16 * ks, lane);
        mma(u[2 * np], a, bv[0], bv[1]);
        mma(u[2 * np + 1], a, bv[2], bv[3]);
        mma(w[2 * np], a, bk[0], bk[1]);
        mma(w[2 * np + 1], a, bk[2], bk[3]);
      }
    }
    // Kh (last read by the products) stages u; KK (last read by the X solve) stages w.
    #pragma unroll
    for (int np = 0; np < 4; ++np) {
      const uint32_t wu[4] = {rn2(u[2 * np][0], u[2 * np][1]), rn2(u[2 * np][2], u[2 * np][3]),
                              rn2(u[2 * np + 1][0], u[2 * np + 1][1]), rn2(u[2 * np + 1][2], u[2 * np + 1][3])};
      const uint32_t ww[4] = {rn2(w[2 * np][0], w[2 * np][1]), rn2(w[2 * np][2], w[2 * np][3]),
                              rn2(w[2 * np + 1][0], w[2 * np + 1][1]), rn2(w[2 * np + 1][2], w[2 * np + 1][3])};
      store_acc(Kh, 16 * mt, 64 * ch + 16 * np, wu, lane);
      store_acc(Ws, 16 * mt, 64 * ch + 16 * np, ww, lane);
    }
  }
  __syncthreads(); 
  for (int idx = tid; idx < 2 * 64 * 16; idx += NT) {
    const int which = idx >> 10, row = (idx >> 4) & 63, ch = idx & 15;
    if (row < r) {
      const T128 &src = which == 0 ? Kh : Ws;
      uint16_t *dst = (which == 0 ? p.ws.U : p.ws.W) + (int64_t(wb) + row) * rowH + hoff + ch * 8;
      *reinterpret_cast<uint4 *>(dst) = *reinterpret_cast<const uint4 *>(src.chunk(row, ch));
    }
  }
}

void intra_launch(const FwdParams &p, cudaStream_t stream) {
  static std::once_flag attr[64];  // the attribute is per device
  int attr_dev = 0;
  cudaGetDevice(&attr_dev);
  std::call_once(attr[unsigned(attr_dev) % 64u], [&] {
    cudaFuncSetAttribute(intra_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    cudaFuncSetAttribute(intra_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
  });
  dim3 grid(p.plan.maxch, p.H);
  if (p.plan.rec.base) intra_kernel<true><<<grid, NT, SMEM, stream>>>(p);
  else intra_kernel<false><<<grid, NT, SMEM, stream>>>(p);
}

}  // namespace kda_sm100
