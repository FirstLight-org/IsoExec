// SPDX-License-Identifier: Apache-2.0
// Engine decode: one token per sequence at absolute position pos, evaluating ROW j = pos mod 64 of the
// open chunk incrementally from the record (csrc/record.cuh) -- the same quantities the chunk kernels
// compute for that row, through the same recipe.cuh / intra_common.cuh arithmetic:
//   KK_j., KQ_j.   off-diagonal blocks: the forward's block-pair mma (A = row j's block of k^ / q^ with
//                  the decay references, B = the earlier blocks' k^), row j taken from the output tile;
//                  diagonal block: the decayed-key product chains T_ji = T_{j-1,i} (*) delta_j over the
//                  open block's rows, T8 over the eighths.
//   X_j.           the block solve's row j: diagonal part from the record's X rows, earlier blocks via
//                  P_j. (T16) and the record's P rows of the open block.
//   u_j, w_j, d_j, o_j   the forward's mma chains with row j placed in row 0 of the A tile. Their B
//                  operands (v, kg, S^, d) are the record's transposed sections, read as fragments straight
//                  into registers (row j patched in, rows past j +0).
// When j == 63 the chunk closes: S_{c+1} = CHAIN64(kd^T, d; S (*) E(G_63)) over the record's rows.
// Two launches: the S-independent row kernel (several rows in flight per SM) and the persistent state pass
// over 64-column value halves (value columns never meet in any chain).
// Row placement inside an mma tile does not enter the bits (gdn_cpr_sm100/tests/test_mma_facts_gpu.py);
// "N decode steps == one N-token prefill" is gated bit for bit by tests/test_engine_gpu.py.
#include <mutex>
#include <cuda_runtime.h>

#include "intra_common.cuh"
#include "params.cuh"
#include "record_state.cuh"

namespace kda_sm100 {

// layer.cu's sigm (gate / beta stages), repeated for the fused-input decode form.
__device__ __forceinline__ float fused_sigm(float y) { return __fdiv_rn(1.f, fadd(1.f, expf(-y))); }

struct DecRecParams {
  const uint16_t *q, *k, *v;       // [N][H][128]
  const void *g, *beta;            // [N][H][128], [N][H]
  int g_bf16, beta_bf16;
  int N, H;
  RecRef rec;
  const int *slots;                // [N] (negative: NULL lane, output +0, nothing written)
  uint16_t *o;                     // [N][H][128]
  // the row kernel's hand-off to the state pass, per (sequence, head): w, qg, u [128] bf16, Mq row [64]
  // bf16, and pos (-1: NULL lane)
  uint16_t *wW, *wQG, *wU, *wMQ;
  int *wpos;
  int null_slot;                   // a slot id that is also a NULL lane (an engine's null block; -1: none)
  // Fused-input form (0 / nullptr = the plain form above). qkv_ld: elements between consecutive tokens of
  // q, k, v (their head rows stay 128-contiguous), e.g. the columns of one wider projection output.
  // gate_f: when set, g is the raw bf16 forget projection f [N][g_ld] and the row kernel evaluates the
  // gate stage itself (layer.cu gate_fwd: lb (*) sigmoid(e(A_log_h) (*) (bf(f) (+) dt_bias_c))).
  // beta_raw: when set, beta is the raw bf16 beta projection b [N][beta_ld] and the row kernel applies
  // layer.cu beta_fwd's rn(sigmoid(bf(b))). Same operations in the same order: the same bits.
  int64_t qkv_ld;
  int gate_f;
  int64_t g_ld;
  const float *dt_bias, *A_log;
  float lb;
  int beta_raw;
  int64_t beta_ld;
};

namespace {

// A 16-row operand whose only live row is `row` (other rows read the zero row).
template <int COLS>
struct OneRowTile {
  const uint16_t *base, *zero;
  int row;
  __device__ __forceinline__ uint32_t saddr(int r, int c) const {
    return uint32_t(__cvta_generic_to_shared(r == row ? base : zero)) + uint32_t(c * 16);
  }
};

// frag_a of a one-live-row operand without the ldmatrix: the same four registers (rows g and g + 8 of
// the m16n8k16 A map, columns k0 + 2t and k0 + 8 + 2t), the live row's words or +0.
__device__ __forceinline__ void frag_a_row(uint32_t (&a)[4], const uint16_t *base, int row, int k0, int lane) {
  const int g = lane >> 2, t = lane & 3;
  const uint32_t lo = (g == row || g + 8 == row) ? *reinterpret_cast<const uint32_t *>(base + k0 + 2 * t) : 0u;
  const uint32_t hi = (g == row || g + 8 == row) ? *reinterpret_cast<const uint32_t *>(base + k0 + 8 + 2 * t) : 0u;
  a[0] = g == row ? lo : 0u;
  a[1] = g + 8 == row ? lo : 0u;
  a[2] = g == row ? hi : 0u;
  a[3] = g + 8 == row ? hi : 0u;
}
template <int COLS>
__device__ __forceinline__ void frag_a(uint32_t (&a)[4], const OneRowTile<COLS> &t, int m0, int k0, int lane) {
  frag_a_row(a, t.base, t.row - m0, k0, lane);
}
template <int COLS>
__device__ __forceinline__ void frag_a(uint32_t (&a)[4], const Row0Tile<COLS> &t, int m0, int k0, int lane) {
  frag_a_row(a, t.base, -m0, k0, lane);
}

}  // namespace

// The B fragments (two n8 tiles at columns n0, n0 + 8; k-step ks) of a transposed [col][64 i] bf16 section,
// rows i > j set to +0 and row j replaced from `rowj` (bf16 by column).
__device__ __forceinline__ void frag_bt(uint32_t (&r)[4], const uint16_t *sec, int n0, int ks, int j,
                                        const uint16_t *rowj, int lane) {
  const int g = lane >> 2, t = lane & 3;
  #pragma unroll
  for (int q = 0; q < 4; ++q) {
    const int col = n0 + g + 8 * (q >> 1), i = 16 * ks + 2 * t + 8 * (q & 1);
    const uint32_t w = *reinterpret_cast<const uint32_t *>(sec + col * 64 + i);
    const uint16_t lo = i < j ? lo16(w) : (i == j ? rowj[col] : uint16_t(0));
    const uint16_t hi = i + 1 < j ? hi16(w) : (i + 1 == j ? rowj[col] : uint16_t(0));
    r[q] = pack(lo, hi);
  }
}

// The same fragments from 4-byte words already loaded (frag_bt's address map), patched for row j.
__device__ __forceinline__ void load_bt(uint32_t (&r)[4], const uint16_t *sec, int n0, int ks, int lane) {
  const int g = lane >> 2, t = lane & 3;
  #pragma unroll
  for (int q = 0; q < 4; ++q)
    r[q] = *reinterpret_cast<const uint32_t *>(sec + (n0 + g + 8 * (q >> 1)) * 64 + 16 * ks + 2 * t + 8 * (q & 1));
}
__device__ __forceinline__ void patch_bt(uint32_t (&r)[4], int n0, int ks, int j, const uint16_t *rowj, int lane) {
  const int g = lane >> 2, t = lane & 3;
  #pragma unroll
  for (int q = 0; q < 4; ++q) {
    const int col = n0 + g + 8 * (q >> 1), i = 16 * ks + 2 * t + 8 * (q & 1);
    const uint16_t lo = i < j ? lo16(r[q]) : (i == j ? rowj[col] : uint16_t(0));
    const uint16_t hi = i + 1 < j ? hi16(r[q]) : (i + 1 == j ? rowj[col] : uint16_t(0));
    r[q] = pack(lo, hi);
  }
}

// ============================================================================ the row kernel
// Everything of row j that does not depend on the state, one CTA of 128 threads per (sequence, head),
// ~41 KB of shared memory so that five rows are in flight on an SM (the step is a chain of short
// dependent phases; concurrency, not bandwidth, is what it needs). Publishes w_j, qg_j, u_j, Mq_j. and pos
// to the hand-off and appends the row's k^, G, X, P, beta, v, kg to the record; at a chunk close it
// replaces the (now dead) k^ section by kd^T = rn(k^_i (*) E(G_63 (-) G_i)) transposed [c][i], the A
// operand of the state pass's S_{c+1} chain.
namespace rowk {
constexpr int RT = 128;
constexpr int OFF_KHB = 0;                       // Tile<16,128> k^ of the open block (row r = i - n0)
constexpr int OFF_GB = OFF_KHB + 4096;           // fp32 [16][128] G of the open block ; delta in place
constexpr int OFF_X = OFF_GB + 8192;             // fp32 X rows, lower-triangular packed (xoff)
constexpr int OFF_P = OFF_X + 4 * X_FLOATS;      // fp32 [16][64] P rows of the open block (48 live columns)
constexpr int OFF_KI = OFF_P + 4096;             // Tile<48,128> ki^b
constexpr int OFF_KGT = OFF_P;                   // Tile<64 i, 128 c> kg rows over P + ki^b once both are dead
constexpr int OFF_ROW = OFF_KI + 12288;
static_assert(OFF_KGT + 16384 <= OFF_ROW, "kg rows fit over P and ki^b");
constexpr int R_Q = 0, R_KR = 256, R_QR = 512, R_QG = 768, R_ZERO = 1024, R_XU = 1280, R_MQ = 1408, R_KK = 1536,
              R_XR = 1792, R_PR = 2048, R_BETA = 2304, R_V = 2560, R_KG = 2816, R_GJ = 3072, R_KJ = 3584,
              R_END = 3840;
constexpr int SMEM = OFF_ROW + R_END;
// X rows in shared memory: the record's packed layout (record.cuh xrow_off), loaded as one contiguous run
__device__ __forceinline__ int xoff(int i) { return xrow_off(i); }
}  // namespace rowk

__global__ void __launch_bounds__(rowk::RT, 5) decode_row_kernel(DecRecParams p) {
  extern __shared__ __align__(128) char smem[];
  using namespace rowk;
  const int n = blockIdx.y, h = blockIdx.x;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int slot = p.slots[n];
  const int64_t nh = int64_t(n) * p.H + h, row = nh * D;
  const int64_t in_row = p.qkv_ld ? int64_t(n) * p.qkv_ld + int64_t(h) * D : row;
  if (slot < 0 || slot == p.null_slot) {   // NULL lane: output +0, no record byte touched
    if (tid == 0) p.wpos[nh] = -1;
    return;
  }
  char *rec = p.rec.rec(slot, h);
  // ---- the token's row (independent of the position)
  uint2 rowin = make_uint2(0u, 0u);
  if (warp < 3) rowin = *reinterpret_cast<const uint2 *>((warp == 0 ? p.k : (warp == 1 ? p.q : p.v)) + in_row + 4 * lane);
  float gin, bin;
  if (p.gate_f) {   // layer.cu gate_fwd_kernel, channel c = h * 128 + tid
    const int c = h * D + tid;
    const float x = fadd(bf(static_cast<const uint16_t *>(p.g)[int64_t(n) * p.g_ld + c]), p.dt_bias[c]);
    gin = fmul(p.lb, fused_sigm(fmul(expf(p.A_log[h]), x)));
  } else {
    gin = p.g_bf16 ? bf(static_cast<const uint16_t *>(p.g)[row + tid]) : static_cast<const float *>(p.g)[row + tid];
  }
  if (p.beta_raw) { // layer.cu beta_fwd_kernel's bf16 output, read back as bf16
    bin = bf(rn(fused_sigm(bf(static_cast<const uint16_t *>(p.beta)[int64_t(n) * p.beta_ld + h]))));
  } else {
    bin = p.beta_bf16 ? bf(static_cast<const uint16_t *>(p.beta)[nh]) : static_cast<const float *>(p.beta)[nh];
  }
  const int pos = *reinterpret_cast<const int *>(rec + Rec::POS);
  const int j = pos % CL, b = j >> 4, n0 = 16 * b, jj = j - n0;
  const bool entry = b >= 1 && j == n0, close = j == CL - 1;

  const Tile<16, 128> Kb{reinterpret_cast<uint16_t *>(smem + OFF_KHB)};
  float *GB = reinterpret_cast<float *>(smem + OFF_GB);
  float *X = reinterpret_cast<float *>(smem + OFF_X);
  float *P = reinterpret_cast<float *>(smem + OFF_P);
  char *rb = smem + OFF_ROW;
  uint16_t *qrow = reinterpret_cast<uint16_t *>(rb + R_Q), *vrow = reinterpret_cast<uint16_t *>(rb + R_V);
  uint16_t *kgrow = reinterpret_cast<uint16_t *>(rb + R_KG), *kjrow = reinterpret_cast<uint16_t *>(rb + R_KJ);
  const uint16_t *zr = reinterpret_cast<const uint16_t *>(rb + R_ZERO);
  const Row0Tile<64> XU{reinterpret_cast<uint16_t *>(rb + R_XU), zr};
  float *KKr = reinterpret_cast<float *>(rb + R_KK), *Xr = reinterpret_cast<float *>(rb + R_XR);
  float *Pr = reinterpret_cast<float *>(rb + R_PR), *beta = reinterpret_cast<float *>(rb + R_BETA);
  float *Gj = reinterpret_cast<float *>(rb + R_GJ);
  const float *grec = reinterpret_cast<const float *>(rec + Rec::GB);

  // ---- the record rows this step reads
  {
    const uint16_t *kb = reinterpret_cast<const uint16_t *>(rec + Rec::KB);
    #pragma unroll
    for (int it = 0; it < 2; ++it) {
      const int idx = it * RT + tid, r = idx >> 4, ch = idx & 15;
      cp16(Kb.saddr(r, ch), kb + (n0 + (r < jj ? r : 0)) * D + ch * 8, r < jj);
    }
    for (int idx = tid; idx < jj * 32; idx += RT) cp16(smem_u32(GB + 4 * idx), grec + n0 * D + 4 * idx, true);
    if (tid < 16) cp16(smem_u32(beta + 4 * tid), reinterpret_cast<const float *>(rec + Rec::BB) + 4 * tid, true);
    cp_commit();   // group 1: what row j's own work needs; group 2 (X, P, ki^b) lands behind it
    const float *xr = reinterpret_cast<const float *>(rec + Rec::XB);
    for (int idx = tid; idx < xrow_off(j) / 4; idx += RT) cp16(smem_u32(X + 4 * idx), xr + 4 * idx, true);
    const float *pr = reinterpret_cast<const float *>(rec + Rec::PB);   // [16][P_COLS] -> smem [16][64]
    for (int idx = tid; idx < 16 * P_COLS / 4; idx += RT) {
      const int i = idx / (P_COLS / 4), c = idx - i * (P_COLS / 4);
      cp16(smem_u32(P + i * 64 + 4 * c), pr + i * P_COLS + 4 * c, true);
    }
    if (KI_STORED && b >= 1 && !entry) {
      const Tile<48, 128> KI{reinterpret_cast<uint16_t *>(smem + OFF_KI)};
      const uint16_t *ki = reinterpret_cast<const uint16_t *>(rec + Rec::KI);
      for (int idx = tid; idx < n0 * 16; idx += RT) cp16(KI.saddr(idx >> 4, idx & 15), ki + idx * 8, true);
    }
    cp_commit();
  }
  const float gprev = j == 0 ? 0.f : (jj == 0 ? grec[(j - 1) * D + tid] : 0.f);
  if (tid < 64) reinterpret_cast<uint32_t *>(rb + R_ZERO)[tid] = 0u;
  if (tid < 32) {
    reinterpret_cast<uint32_t *>(rb + R_XU)[tid] = 0u;
    reinterpret_cast<uint32_t *>(rb + R_MQ)[tid] = 0u;
  }
  if (tid < 64) KKr[tid] = 0.f;
  cp_wait<1>();
  __syncthreads();
  // ---- row j: k^, q^ (FOLD32), v, beta_j, G_j
  if (warp < 2) {
    float x[4] = {bf(lo16(rowin.x)), bf(hi16(rowin.x)), bf(lo16(rowin.y)), bf(hi16(rowin.y))};
    float inv = inv_norm(fold32(x));
    if (warp == 1) inv = fmul(inv, QSCALE);
    uint2 o2;
    o2.x = rn2(fmul(x[0], inv), fmul(x[1], inv));
    o2.y = rn2(fmul(x[2], inv), fmul(x[3], inv));
    if (warp == 0) {
      *reinterpret_cast<uint2 *>(&Kb.at(jj, 4 * lane)) = o2;
      *reinterpret_cast<uint2 *>(kjrow + 4 * lane) = o2;
    } else {
      *reinterpret_cast<uint2 *>(qrow + 4 * lane) = o2;
    }
  } else if (warp == 2) {
    *reinterpret_cast<uint2 *>(vrow + 4 * lane) = rowin;
  } else if (lane == 0) {
    beta[j] = bin;
  }
  {
    const float gj = cum_gate(jj > 0 ? GB[(jj - 1) * D + tid] : gprev, gin);
    GB[jj * D + tid] = gj;
    Gj[tid] = gj;
  }
  __syncthreads();

  // ---- kg_j, qg_j, kr_j, qr_j ; ki^b when entering a block
  if (tid < 64) {
    const int c0 = 2 * tid;
    const float2 G2 = *reinterpret_cast<const float2 *>(Gj + c0);
    const float e0 = decay0(G2.x), e1 = decay0(G2.y);
    const uint32_t kp = *reinterpret_cast<const uint32_t *>(kjrow + c0), qp = *reinterpret_cast<const uint32_t *>(qrow + c0);
    *reinterpret_cast<uint32_t *>(kgrow + c0) = scale2e(kp, e0, e1);
    *reinterpret_cast<uint32_t *>(rb + R_QG + 2 * c0) = scale2e(qp, e0, e1);
    if (b >= 1) {
      const float2 Gn = *reinterpret_cast<const float2 *>(GB + c0);   // block row 0 is G_n
      const float r0 = decay(G2.x, Gn.x), r1 = decay(G2.y, Gn.y);
      *reinterpret_cast<uint32_t *>(rb + R_KR + 2 * c0) = scale2e(kp, r0, r1);
      *reinterpret_cast<uint32_t *>(rb + R_QR + 2 * c0) = scale2e(qp, r0, r1);
    }
  }
  // ki^b: computed by the entering step (stored for the block's later steps), or by every step of the block
  // (KDA_REC_RECOMPUTE_KI). The reference G_n is the block's first row: Gj at the entry, GB row 0 after (the
  // same value: the entering step wrote Gj to both the record and GB row 0).
  if (entry || (!KI_STORED && b >= 1))   // eight channels a thread-step (16-byte loads in flight); each element its own scale2
    #pragma unroll 2
    for (int idx = tid; idx < n0 * 16; idx += RT) {
      const int r = idx >> 4, ch = idx & 15, c0 = 8 * ch;
      const uint4 k8 = *reinterpret_cast<const uint4 *>(rec + Rec::KB + (r * D + c0) * 2);
      const float4 ga = *reinterpret_cast<const float4 *>(grec + r * D + c0), gb = *reinterpret_cast<const float4 *>(grec + r * D + c0 + 4);
      const float *gref = entry ? Gj : GB;
      const float4 ja = *reinterpret_cast<const float4 *>(gref + c0), jb = *reinterpret_cast<const float4 *>(gref + c0 + 4);
      const uint4 v8 = make_uint4(scale2(k8.x, make_float2(ja.x, ja.y), make_float2(ga.x, ga.y)),
                                  scale2(k8.y, make_float2(ja.z, ja.w), make_float2(ga.z, ga.w)),
                                  scale2(k8.z, make_float2(jb.x, jb.y), make_float2(gb.x, gb.y)),
                                  scale2(k8.w, make_float2(jb.z, jb.w), make_float2(gb.z, gb.w)));
      *reinterpret_cast<uint4 *>(Tile<48, 128>{reinterpret_cast<uint16_t *>(smem + OFF_KI)}.chunk(r, ch)) = v8;
      if (KI_STORED) *reinterpret_cast<uint4 *>(rec + Rec::KI + (r * D + c0) * 2) = v8;
    }
  cp_wait<0>();   // group 2: X, P, ki^b
  __syncthreads();

  {   // delta_t = E(G_t (-) G_{t-1}) over the open block's G rows, in place, descending (channel tid)
    float hi = GB[jj * D + tid];
    for (int r = jj; r >= 1; --r) {
      const float lo = GB[(r - 1) * D + tid];
      GB[r * D + tid] = decay(hi, lo);
      hi = lo;
    }
  }
  // ---- KK_j. / KQ_j. : off-diagonal block pairs (warps < b), then the diagonal block (all warps)
  if (warp < b) {
    const OneRowTile<128> Kt{reinterpret_cast<const uint16_t *>(rb + R_KR), zr, jj};
    const OneRowTile<128> Qt{reinterpret_cast<const uint16_t *>(rb + R_QR), zr, jj};
    const int i0 = 16 * warp;
    float kk[2][4] = {}, kq[2][4] = {};
    #pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      uint32_t ak[4], aq[4], bb[4];
      frag_a(ak, Kt, 0, 16 * ks, lane);
      frag_a(aq, Qt, 0, 16 * ks, lane);
      frag_b2(bb, (Tile<48, 128>{reinterpret_cast<uint16_t *>(smem + OFF_KI)}), i0, 16 * ks, lane);
      mma(kk[0], ak, bb[0], bb[1]);
      mma(kk[1], ak, bb[2], bb[3]);
      mma(kq[0], aq, bb[0], bb[1]);
      mma(kq[1], aq, bb[2], bb[3]);
    }
    if (g == (jj & 7)) {
      const bool hi8 = jj >> 3;
      #pragma unroll
      for (int nt = 0; nt < 2; ++nt)
        #pragma unroll
        for (int el = 0; el < 2; ++el) {
          const int i = i0 + 8 * nt + 2 * t + el;
          KKr[i] = hi8 ? kk[nt][2 + el] : kk[nt][el];
          reinterpret_cast<uint16_t *>(rb + R_MQ)[i] = rn(hi8 ? kq[nt][2 + el] : kq[nt][el]);
        }
    }
  }
  __syncthreads();   // delta complete
  {
    const int cc = 4 * warp + (lane >> 3), p8 = lane & 7, i = n0 + cc;
    float T[16], kj[16], qj[16];
    #pragma unroll
    for (int m = 0; m < 4; ++m) {
      const uint2 w2 = *reinterpret_cast<const uint2 *>(&Kb.at(cc, 32 * m + 4 * p8));
      T[4 * m] = bf(lo16(w2.x)); T[4 * m + 1] = bf(hi16(w2.x)); T[4 * m + 2] = bf(lo16(w2.y)); T[4 * m + 3] = bf(hi16(w2.y));
      const uint2 k2 = *reinterpret_cast<const uint2 *>(kjrow + 32 * m + 4 * p8);
      kj[4 * m] = bf(lo16(k2.x)); kj[4 * m + 1] = bf(hi16(k2.x)); kj[4 * m + 2] = bf(lo16(k2.y)); kj[4 * m + 3] = bf(hi16(k2.y));
      const uint2 q2 = *reinterpret_cast<const uint2 *>(qrow + 32 * m + 4 * p8);
      qj[4 * m] = bf(lo16(q2.x)); qj[4 * m + 1] = bf(hi16(q2.x)); qj[4 * m + 2] = bf(lo16(q2.y)); qj[4 * m + 3] = bf(hi16(q2.y));
    }
    for (int tt = cc + 1; tt <= jj; ++tt)
      #pragma unroll
      for (int m = 0; m < 4; ++m) {
        const float4 d4 = *reinterpret_cast<const float4 *>(GB + tt * D + 32 * m + 4 * p8);
        T[4 * m] = fmul(T[4 * m], d4.x);
        T[4 * m + 1] = fmul(T[4 * m + 1], d4.y);
        T[4 * m + 2] = fmul(T[4 * m + 2], d4.z);
        T[4 * m + 3] = fmul(T[4 * m + 3], d4.w);
      }
    float kk = 0.f, kq = 0.f;
    #pragma unroll
    for (int e = 0; e < 16; ++e) {
      kk = ffma(kj[e], T[e], kk);
      kq = ffma(qj[e], T[e], kq);
    }
    kk = oct_t8(kk);
    kq = oct_t8(kq);
    if (p8 == 0 && i <= j) {
      if (i < j) KKr[i] = kk;
      reinterpret_cast<uint16_t *>(rb + R_MQ)[i] = rn(kq);
    }
  }
  __syncthreads();

  // ---- X_j. : diagonal part and P_j. (independent), then the earlier blocks
  const float bj = beta[j];
  if (tid < 16) {
    const int mm = tid, m = n0 + mm;
    if (mm < jj) {
      float lrow[16], xcol[16];
      #pragma unroll
      for (int i = 0; i < 16; ++i) {
        lrow[i] = i < jj ? fmul(KKr[n0 + i], bj) : fmul(0.f, bj);
        xcol[i] = i < mm ? 0.f : (i == mm ? 1.f : (i < jj ? X[xoff(n0 + i) + m] : 0.f));
      }
      Xr[m] = x_block(lrow, xcol);
    }
  } else if (tid >= 64 && tid - 64 < n0) {
    const int m = tid - 64;
    Pr[m] = p_entry(b, [&](int i) { return fmul(KKr[i], bj); },
                    [&](int i) { return i < m ? 0.f : X[xoff(i) + m]; });
  }
  __syncthreads();
  if (tid < n0) {
    const int m = tid;
    float y[16], pc[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
      y[i] = i < jj ? Xr[n0 + i] : (i == jj ? 1.f : 0.f);
      pc[i] = i < jj ? P[i * 64 + m] : (i == jj ? Pr[m] : 0.f);
    }
    Xr[m] = x_block(y, pc);
  }
  if (tid == 0) Xr[j] = 1.f;
  __syncthreads();   // P and ki^b are dead: stream the kg rows in over them (rows past j +0)
  {
    const T128 KGb{reinterpret_cast<uint16_t *>(smem + OFF_KGT)};
    const uint16_t *kgb = reinterpret_cast<const uint16_t *>(rec + Rec::KGB);
    for (int idx = tid; idx < 16 * (b + 1) * 16; idx += RT) {   // rows of the blocks up to j's
      const int r = idx >> 4, ch = idx & 15;
      if (r != j) cp16(KGb.saddr(r, ch), kgb + (r < j ? r : 0) * D + ch * 8, r < j);
      else *reinterpret_cast<uint4 *>(KGb.chunk(j, ch)) = *reinterpret_cast<const uint4 *>(kgrow + 8 * ch);
    }
    cp_commit();
  }
  if (tid < 32) {   // xu_j. = rn(X_ji (*) beta_i) (i <= j)
    const int i0 = 2 * tid;
    const float x0 = i0 <= j ? Xr[i0] : 0.f, x1 = i0 + 1 <= j ? Xr[i0 + 1] : 0.f;
    reinterpret_cast<uint32_t *>(rb + R_XU)[tid] =
        rn2(fmul(x0, i0 <= j ? beta[i0] : 0.f), fmul(x1, i0 + 1 <= j ? beta[i0 + 1] : 0.f));
  }
  __syncthreads();

  cp_wait<0>();
  __syncthreads();
  // ---- w_j: CHAIN64^b(j) ; warp w: columns 32 w .. + 31 ; published to the hand-off (u_j is the state
  // pass's: it needs only that pass's value columns)
  {
    const T128 KGb{reinterpret_cast<uint16_t *>(smem + OFF_KGT)};
    #pragma unroll
    for (int np = 0; np < 2; ++np) {
      float w[2][4] = {};
      #pragma unroll
      for (int ks = 0; ks < 4; ++ks) {
        if (ks > b) break;
        uint32_t a[4], bk[4];
        frag_a(a, XU, 0, 16 * ks, lane);
        frag_b2_t(bk, KGb, 32 * warp + 16 * np, 16 * ks, lane);
        mma(w[0], a, bk[0], bk[1]);
        mma(w[1], a, bk[2], bk[3]);
      }
      if (g == 0)
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
          const int c = 32 * warp + 16 * np + 8 * nt + 2 * t;
          *reinterpret_cast<uint32_t *>(p.wW + row + c) = rn2(w[nt][0], w[nt][1]);
        }
    }
  }
  if (tid < 32) *reinterpret_cast<uint32_t *>(p.wU + nh * 64 + 2 * tid) = *reinterpret_cast<const uint32_t *>(rb + R_XU + 4 * tid);
  if (tid < 64) *reinterpret_cast<uint32_t *>(p.wQG + row + 2 * tid) = *reinterpret_cast<const uint32_t *>(rb + R_QG + 4 * tid);
  if (tid < 32) *reinterpret_cast<uint32_t *>(p.wMQ + nh * 64 + 2 * tid) = *reinterpret_cast<const uint32_t *>(rb + R_MQ + 4 * tid);
  if (tid == 0) p.wpos[nh] = pos;
  // ---- append row j to the record
  if (!close && tid < 16)
    *reinterpret_cast<uint4 *>(rec + Rec::KB + (j * D + 8 * tid) * 2) = *reinterpret_cast<const uint4 *>(kjrow + 8 * tid);
  if (tid >= 16 && tid < 48) {
    const int c4 = tid - 16;
    *reinterpret_cast<float4 *>(rec + Rec::GB + (j * D + 4 * c4) * 4) = *reinterpret_cast<const float4 *>(Gj + 4 * c4);
  }
  if (tid >= 64) {
    const int m = tid - 64;
    if (m < xrow_len(j) && !close) reinterpret_cast<float *>(rec + Rec::XB)[xrow_off(j) + m] = m <= j ? Xr[m] : 0.f;   // row 63 is never read
    if (b >= 1 && m < P_COLS) reinterpret_cast<float *>(rec + Rec::PB)[jj * P_COLS + m] = m < n0 ? Pr[m] : 0.f;
  }
  if (tid == 48) reinterpret_cast<float *>(rec + Rec::BB)[j] = bj;
  if (tid >= 96 && tid < 112)
    *reinterpret_cast<uint4 *>(rec + Rec::VB + (j * D + 8 * (tid - 96)) * 2) = *reinterpret_cast<const uint4 *>(vrow + 8 * (tid - 96));
  else if (tid >= 112 && !close)   // row 63 is never read
    *reinterpret_cast<uint4 *>(rec + Rec::KGB + (j * D + 8 * (tid - 112)) * 2) = *reinterpret_cast<const uint4 *>(kgrow + 8 * (tid - 112));
  if (close) {
    // kd_ic = rn(k^_i (*) E(G_63 (-) G_i)) staged [i][c] in the (dead) kg rows' space, then published
    // transposed [c][i]
    __syncthreads();
    uint16_t *kd = reinterpret_cast<uint16_t *>(smem + OFF_KGT);
    #pragma unroll 2
    for (int idx = tid; idx < 64 * 16; idx += RT) {
      const int r = idx >> 4, c0 = 8 * (idx & 15);
      const uint4 k8 = r == CL - 1 ? *reinterpret_cast<const uint4 *>(kjrow + c0)
                    : r < CL ? *reinterpret_cast<const uint4 *>(rec + Rec::KB + (r * D + c0) * 2)
                    : make_uint4(0u, 0u, 0u, 0u);   // rows past the chunk length are absent: +0
      const float *gr = (r >= CL - 1) ? Gj : grec + r * D;
      const float4 ga = *reinterpret_cast<const float4 *>(gr + c0), gb = *reinterpret_cast<const float4 *>(gr + c0 + 4);
      const float4 ja = *reinterpret_cast<const float4 *>(Gj + c0), jb = *reinterpret_cast<const float4 *>(Gj + c0 + 4);
      *reinterpret_cast<uint4 *>(kd + r * D + c0) =
          make_uint4(scale2(k8.x, make_float2(ja.x, ja.y), make_float2(ga.x, ga.y)),
                     scale2(k8.y, make_float2(ja.z, ja.w), make_float2(ga.z, ga.w)),
                     scale2(k8.z, make_float2(jb.x, jb.y), make_float2(gb.x, gb.y)),
                     scale2(k8.w, make_float2(jb.z, jb.w), make_float2(gb.z, gb.w)));
    }
    __syncthreads();
    uint16_t *kdt = reinterpret_cast<uint16_t *>(rec + Rec::KB);
    for (int idx = tid; idx < 128 * 32; idx += RT) {
      const int c = idx >> 5, i0 = 2 * (idx & 31);
      *reinterpret_cast<uint32_t *>(kdt + c * 64 + i0) = pack(kd[i0 * D + c], kd[(i0 + 1) * D + c]);
    }
  }
}

// ============================================================================ the state pass
// The part of the step that reads the state: a persistent grid over (sequence, head, 32-column value slice),
// the item's S^ slice, d slice and hand-off rows streaming into shared memory (cp.async, double-buffered)
// while the previous item runs:
//   d_j = rn(bf(u_j) (-) CHAIN128(w_j, S^))   o_j = rn(CHAIN64^b(j)(Mq_j., d; CHAIN128(qg_j, S^)))
//   at a close: S_{c+1} = CHAIN64(kd^T, d; S (*) E(G_63))
// Value columns never meet, so the split is execution only (the chains of an element are the same).
namespace stk {
constexpr int ST = 128, VS = 32, NV = D / VS;      // 4 warps ; 32-column value slices, NV per (seq, head)
constexpr int S_SH = 0;                          // Tile<32 v, 128 k> S^ slice
constexpr int S_DT = S_SH + VS * D * 2;          // Tile<64 i, 32 v> d slice (row j added)
constexpr int S_VT = S_DT + 64 * VS * 2;         // Tile<64 i, 32 v> v slice (row j already appended)
constexpr int S_W = S_VT + 64 * VS * 2, S_QG = S_W + 256, S_U = S_QG + 256, S_MQ = S_U + 128, S_POS = S_MQ + 128;
constexpr int STAGE = S_POS + 128;
constexpr int NSTAGE = 2, MINB = 6;               // the id pipeline of decode_state_rec_kernel is two-stage
constexpr int OFF_ZERO = NSTAGE * STAGE;
constexpr int OFF_D = OFF_ZERO + 256;            // d_j of the slice (bf16 [32]) ; u_j of the slice at +128
constexpr int SMEM = OFF_D + 256;
}  // namespace stk

// The close's two touches of the state planes, out of line: the state pass runs at its register cap (six CTAs an
// SM), and the planes' loads and ballots are only ever needed once in 64 steps, so their live ranges must not
// widen the hot path's allocation.
// S_c of warp's 32 k rows x the slice's 32 v columns (acc layout of the close's fold), scaled by exp2(G_63):
// S[mt][nt][e], tile (2 warp + mt, 4 vs + nt).
__device__ __noinline__ void close_join(float (&S)[2][4][4], const Tile<stk::VS, 128> Sh, const char *rec, const float *G63, int warp,
                                        int vs, int lane) {
  const int g = lane >> 2, t = lane & 3;
  #pragma unroll
  for (int mt = 0; mt < 2; ++mt) {
    const int k0 = 32 * warp + 16 * mt;
    const float gl0 = decay0(G63[k0 + g]), gl1 = decay0(G63[k0 + g + 8]);
    #pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      uint16_t shv[4];
      #pragma unroll
      for (int e = 0; e < 4; ++e) shv[e] = Sh.at(8 * nt + 2 * t + (e & 1), k0 + g + 8 * (e >> 1));
      float a[4];
      s_join_tile(a, shv, rec, 2 * warp + mt, vs * (stk::VS / 8) + nt, lane);
      #pragma unroll
      for (int e = 0; e < 4; ++e) S[mt][nt][e] = fmul(a[e], (e >> 1) ? gl1 : gl0);
    }
  }
}
__device__ __noinline__ void close_store(char *rec, const float (&S)[2][4][4], int warp, int vs, int lane) {
  #pragma unroll
  for (int mt = 0; mt < 2; ++mt)
    #pragma unroll
    for (int nt = 0; nt < 4; ++nt) s_store_tile(rec, S[mt][nt], 2 * warp + mt, vs * (stk::VS / 8) + nt, lane);
}

// (slot, pos) of item it: loaded one item before its prefetch is issued, so their latency is off the path
__device__ __forceinline__ int2 st_ids(const DecRecParams &p, int it) {
  const int nh32 = it / stk::NV;
  return make_int2(p.slots[int(unsigned(nh32) / unsigned(p.H))], p.wpos[nh32]);
}
__device__ __forceinline__ void st_prefetch(const DecRecParams &p, char *st, int it, int tid, int2 ids) {
  using namespace stk;
  const int vs = it % NV, nh32 = it / NV;   // 32-bit index math (N H NV < 2^31)
  const int h = nh32 % p.H;
  const int64_t nh = nh32, row = nh * D;
  const int slot = ids.x, pp = ids.y;
  const Tile<VS, 128> Sh{reinterpret_cast<uint16_t *>(st + S_SH)};
  const Tile<64, VS> Dt{reinterpret_cast<uint16_t *>(st + S_DT)};
  if (slot >= 0 && pp >= 0) {
    const int nrow = 16 * (((pp % CL) >> 4) + 1);   // rows of the blocks up to j's
    const char *rec = p.rec.rec(slot, h);
    const uint16_t *sh = reinterpret_cast<const uint16_t *>(rec + Rec::SH) + vs * VS * D;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int idx = i * ST + tid, r = idx >> 4, ch = idx & 15;
      cp16(Sh.saddr(r, ch), sh + r * D + ch * 8, true);
    }
    const int jq = pp % CL;
    const uint16_t *db = reinterpret_cast<const uint16_t *>(rec + Rec::DB) + vs * VS;
    const uint16_t *vb = reinterpret_cast<const uint16_t *>(rec + Rec::VB) + vs * VS;
    const Tile<64, VS> Vt{reinterpret_cast<uint16_t *>(st + S_VT)};
    for (int idx = tid; idx < nrow * 4; idx += ST) {   // 4 chunks a row
      const int r = idx >> 2, ch = idx & 3;
      cp16(Dt.saddr(r, ch), db + (r < jq ? r : 0) * D + ch * 8, r < jq);    // d rows < j (row j: this step)
      cp16(Vt.saddr(r, ch), vb + (r <= jq ? r : 0) * D + ch * 8, r <= jq);  // v rows <= j (row j appended)
    }
  }
  if (tid < 16) cp16(smem_u32(st + S_W + 16 * tid), p.wW + row + 8 * tid, true);
  else if (tid < 32) cp16(smem_u32(st + S_QG + 16 * (tid - 16)), p.wQG + row + 8 * (tid - 16), true);
  else if (tid < 40) cp16(smem_u32(st + S_U + 16 * (tid - 32)), p.wU + nh * 64 + 8 * (tid - 32), true);
  else if (tid < 48) cp16(smem_u32(st + S_MQ + 16 * (tid - 40)), p.wMQ + nh * 64 + 8 * (tid - 40), true);
  else if (tid == 48)
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(smem_u32(st + S_POS)), "l"(p.wpos + nh) : "memory");
}

__global__ void __launch_bounds__(stk::ST, stk::MINB) decode_state_rec_kernel(DecRecParams p) {
  extern __shared__ __align__(128) char smem[];
  using namespace stk;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int items = p.N * p.H * NV;
  const uint16_t *zr = reinterpret_cast<const uint16_t *>(smem + OFF_ZERO);
  uint16_t *drow = reinterpret_cast<uint16_t *>(smem + OFF_D);
  uint16_t *us = reinterpret_cast<uint16_t *>(smem + OFF_D + 128);
  if (tid < 64) reinterpret_cast<uint32_t *>(smem + OFF_ZERO)[tid] = 0u;
  static_assert(NSTAGE == 2, "the id pipeline below is written for two stages");
  int2 ids = blockIdx.x < items ? st_ids(p, blockIdx.x) : make_int2(-1, -1);
  if (blockIdx.x < items) st_prefetch(p, smem, blockIdx.x, tid, ids);
  cp_commit();
  int slot_cur = ids.x;
  ids = blockIdx.x + gridDim.x < items ? st_ids(p, blockIdx.x + gridDim.x) : make_int2(-1, -1);
  int local = 0;
  #pragma unroll 1
  for (int it = blockIdx.x; it < items; it += gridDim.x, ++local) {
    char *st = smem + (local % NSTAGE) * STAGE;
    const int vs = it % NV, nh32 = it / NV;
    const int h = nh32 % p.H;
    const int64_t nh = nh32, row = nh * D;
    cp_wait<NSTAGE - 2>();
    __syncthreads();                     // item it staged; every warp is done with the previous item
    if (it + gridDim.x < items) st_prefetch(p, smem + ((local + 1) & 1) * STAGE, it + gridDim.x, tid, ids);
    cp_commit();
    const int slot = slot_cur;
    slot_cur = ids.x;
    if (it + 2 * gridDim.x < items) ids = st_ids(p, it + 2 * gridDim.x);
    const int pos = *reinterpret_cast<const int *>(st + S_POS);
    if (pos < 0) {   // NULL lane
      if (tid < VS / 2) reinterpret_cast<uint32_t *>(p.o + row + vs * VS)[tid] = 0u;
      continue;
    }
    const int j = pos % CL, b = j >> 4;
    char *rec = p.rec.rec(slot, h);
    const Tile<VS, 128> Sh{reinterpret_cast<uint16_t *>(st + S_SH)};
    const Tile<64, VS> Dt{reinterpret_cast<uint16_t *>(st + S_DT)};
    const Tile<64, VS> Vt{reinterpret_cast<uint16_t *>(st + S_VT)};
    const Row0Tile<128> Wr{reinterpret_cast<uint16_t *>(st + S_W), zr}, QGr{reinterpret_cast<uint16_t *>(st + S_QG), zr};
    const Row0Tile<64> MQ{reinterpret_cast<uint16_t *>(st + S_MQ), zr};
    const Row0Tile<64> XU{reinterpret_cast<uint16_t *>(st + S_U), zr};
    // warps 0, 1: u_j then d_j over slice value columns 16 w .. + 15 ; warps 2, 3: the o seed
    // CHAIN128(qg_j, S^) over columns 16 (w - 2) .. + 15, then o_j
    const int cw = 16 * (warp & 1);
    float oacc[2][4] = {};
    if (warp < 2) {
      float u[2][4] = {};   // u_j = rn(CHAIN64^b(j)(xu_j., v)) (v rows past j are +0: their xu coefficients are +0)
      #pragma unroll
      for (int ks = 0; ks < 4; ++ks) {
        if (ks > b) break;
        uint32_t a[4], bv[4];
        frag_a(a, XU, 0, 16 * ks, lane);
        frag_b2_t(bv, Vt, cw, 16 * ks, lane);
        mma(u[0], a, bv[0], bv[1]);
        mma(u[1], a, bv[2], bv[3]);
      }
      float dacc[2][4] = {};
      #pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        uint32_t aw[4], bb[4];
        frag_a(aw, Wr, 0, 16 * ks, lane);
        frag_b2(bb, Sh, cw, 16 * ks, lane);
        mma(dacc[0], aw, bb[0], bb[1]);
        mma(dacc[1], aw, bb[2], bb[3]);
      }
      if (g == 0)
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
          const int vc = cw + 8 * nt + 2 * t;
          const uint32_t up = rn2(u[nt][0], u[nt][1]);
          *reinterpret_cast<uint32_t *>(us + vc) = up;
          *reinterpret_cast<uint32_t *>(drow + vc) =
              rn2(fsub(bf(lo16(up)), dacc[nt][0]), fsub(bf(hi16(up)), dacc[nt][1]));
        }
    } else {
      #pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        uint32_t aq[4], bb[4];
        frag_a(aq, QGr, 0, 16 * ks, lane);
        frag_b2(bb, Sh, cw, 16 * ks, lane);
        mma(oacc[0], aq, bb[0], bb[1]);
        mma(oacc[1], aq, bb[2], bb[3]);
      }
    }
    __syncthreads();
    // the d slice's row j = d_j (rows past j are +0: their Mq coefficients are +0)
    if (tid < VS / 8) *reinterpret_cast<uint4 *>(Dt.chunk(j, tid)) = *reinterpret_cast<const uint4 *>(drow + 8 * tid);
    __syncthreads();
    // o_j = rn(CHAIN64^b(j)(Mq_j., d; seed))
    if (warp >= 2) {
      #pragma unroll
      for (int ks = 0; ks < 4; ++ks) {
        if (ks > b) break;
        uint32_t am[4], bb[4];
        frag_a(am, MQ, 0, 16 * ks, lane);
        frag_b2_t(bb, Dt, cw, 16 * ks, lane);
        mma(oacc[0], am, bb[0], bb[1]);
        mma(oacc[1], am, bb[2], bb[3]);
      }
      if (g == 0)
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
          const int vc = cw + 8 * nt + 2 * t;
          *reinterpret_cast<uint32_t *>(p.o + row + vs * VS + vc) = rn2(oacc[nt][0], oacc[nt][1]);
        }
    } else if (tid < VS / 8 && j != CL - 1) {   // the slice's d row j into the record (row 63 is read from shared memory)
      *reinterpret_cast<uint4 *>(rec + Rec::DB + (j * D + vs * VS + 8 * tid) * 2) =
          *reinterpret_cast<const uint4 *>(drow + 8 * tid);
    }
    if (j == CL - 1) {
      // S_{c+1} = CHAIN64(kd^T, d; seed = S (*) E(G_63)) ; warp w: k rows 32 w .. + 31 of the slice
      const float *G63 = reinterpret_cast<const float *>(rec + Rec::GB) + (CL - 1) * D;
      const uint16_t *kdt = reinterpret_cast<const uint16_t *>(rec + Rec::KB);   // kd^T [c][i] at a close
      float S[2][4][4];
      close_join(S, Sh, rec, G63, warp, vs, lane);
      #pragma unroll
      for (int ks = 0; ks < CL / 16; ++ks) {   // CHAIN^b(L), L = CL - 1
        uint32_t bb[2][4];
        #pragma unroll
        for (int np = 0; np < 2; ++np) frag_b2_t(bb[np], Dt, 16 * np, 16 * ks, lane);
        #pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
          const int k0 = 32 * warp + 16 * mt;
          uint32_t ak[4];
          #pragma unroll
          for (int q = 0; q < 4; ++q)   // A[m = c][k = i] = kd^T[c][i]: m16n8k16 A fragment map
            ak[q] = *reinterpret_cast<const uint32_t *>(kdt + (k0 + g + 8 * (q & 1)) * 64 + 16 * ks + 2 * t + 8 * (q >> 1));
          #pragma unroll
          for (int np = 0; np < 2; ++np) {
            mma(S[mt][2 * np], ak, bb[np][0], bb[np][1]);
            mma(S[mt][2 * np + 1], ak, bb[np][2], bb[np][3]);
          }
        }
      }
      close_store(rec, S, warp, vs, lane);
      __syncthreads();   // every warp is done with the stage's S^ (the d / o-seed chains)
      #pragma unroll
      for (int mt = 0; mt < 2; ++mt)
        #pragma unroll
        for (int np = 0; np < 2; ++np) {
          const uint32_t w4[4] = {rn2(S[mt][2 * np][0], S[mt][2 * np][1]), rn2(S[mt][2 * np][2], S[mt][2 * np][3]),
                                  rn2(S[mt][2 * np + 1][0], S[mt][2 * np + 1][1]), rn2(S[mt][2 * np + 1][2], S[mt][2 * np + 1][3])};
          store_acc_t(Sh, 32 * warp + 16 * mt, 16 * np, w4, lane);
        }
      __syncthreads();
      uint16_t *sh = reinterpret_cast<uint16_t *>(rec + Rec::SH) + vs * VS * D;
      for (int idx = tid; idx < VS * 16; idx += ST) {
        const int r = idx >> 4, ch = idx & 15;
        *reinterpret_cast<uint4 *>(sh + r * D + ch * 8) = *reinterpret_cast<const uint4 *>(Sh.chunk(r, ch));
      }
    }
    if (vs == 0 && tid == 0) *reinterpret_cast<int *>(rec + Rec::POS) = pos + 1;
  }
  cp_wait<0>();
}
// The state a record stands for: S_c, or for an open chunk S_c (*) E(G_L) + CHAIN64^b(L)(kd^T, d) with
// L = pos mod 64 - 1 -- the forward's final state of a sequence ending at that row, by the same chains.
// CTA per (slot, head, 64-column value half).
__global__ void __launch_bounds__(NT, 1) record_state_kernel(RecRef rr, const int *slots, float *out) {
  extern __shared__ __align__(128) char smem[];
  const int n = blockIdx.y, h = blockIdx.x >> 1, vs = blockIdx.x & 1;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const char *rec = rr.rec(slots[n], h);
  const int jf = *reinterpret_cast<const int *>(rec + Rec::POS) % CL;
  const T128 KD{reinterpret_cast<uint16_t *>(smem)};
  const Tile<64, 64> Ds{reinterpret_cast<uint16_t *>(smem + 16384)};
  {
    const uint16_t *db = reinterpret_cast<const uint16_t *>(rec + Rec::DB) + 64 * vs;
    for (int idx = tid; idx < 64 * 8; idx += NT) {
      const int r = idx >> 3, ch = idx & 7;
      cp16(Ds.saddr(r, ch), db + (r < jf ? r : 0) * D + ch * 8, r < jf);
    }
    cp_commit();
  }
  const float *Gr = reinterpret_cast<const float *>(rec + Rec::GB);
  for (int idx = tid; idx < 64 * 64; idx += NT) {
    const int r = idx >> 6, c0 = 2 * (idx & 63);
    uint32_t kd = 0u;
    if (r < jf) {
      const uint32_t kp = *reinterpret_cast<const uint32_t *>(rec + Rec::KB + (r * D + c0) * 2);
      kd = scale2(kp, *reinterpret_cast<const float2 *>(Gr + (jf - 1) * D + c0), *reinterpret_cast<const float2 *>(Gr + r * D + c0));
    }
    *reinterpret_cast<uint32_t *>(&KD.at(r, c0)) = kd;
  }
  cp_wait<0>();
  __syncthreads();
  float S[8][4];
  const float gl0 = jf ? decay0(Gr[(jf - 1) * D + 16 * warp + g]) : 1.f;
  const float gl1 = jf ? decay0(Gr[(jf - 1) * D + 16 * warp + g + 8]) : 1.f;
  #pragma unroll
  for (int nt = 0; nt < 8; ++nt) {
    float a[4];
    s_load_tile(a, rec, warp, 8 * vs + nt, lane);
    #pragma unroll
    for (int e = 0; e < 4; ++e) S[nt][e] = jf ? fmul(a[e], (e >> 1) ? gl1 : gl0) : a[e];
  }
  if (jf) {
    const int kmax = (jf - 1) >> 4;
    #pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      if (ks > kmax) break;
      uint32_t ak[4];
      frag_a_t(ak, KD, 16 * warp, 16 * ks, lane);
      #pragma unroll
      for (int np = 0; np < 4; ++np) {
        uint32_t bb[4];
        frag_b2_t(bb, Ds, 16 * np, 16 * ks, lane);
        mma(S[2 * np], ak, bb[0], bb[1]);
        mma(S[2 * np + 1], ak, bb[2], bb[3]);
      }
    }
  }
  float *o = out + (int64_t(n) * rr.H + h) * D * D;
  #pragma unroll
  for (int nt = 0; nt < 8; ++nt)
    #pragma unroll
    for (int eh = 0; eh < 2; ++eh)
      *reinterpret_cast<float2 *>(o + (16 * warp + g + 8 * eh) * D + 64 * vs + 8 * nt + 2 * t) =
          make_float2(S[nt][2 * eh], S[nt][2 * eh + 1]);
}

void decode_rec_launch(const DecRecParams &p, cudaStream_t stream) {
  static int grid_cap_by_dev[64] = {};  // per device: the attribute and the SM count are both per device
  int cur_dev = 0;
  cudaGetDevice(&cur_dev);
  int& grid_cap = grid_cap_by_dev[unsigned(cur_dev) % 64u];
  if (!grid_cap) {
    cudaFuncSetAttribute(decode_row_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, rowk::SMEM);
    cudaFuncSetAttribute(decode_state_rec_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, stk::SMEM);
    int dev = 0, sms = 0, per = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, decode_state_rec_kernel, stk::ST, stk::SMEM);
    grid_cap = sms * (per > 0 ? per : 1);
  }
  if (p.N <= 0) return;
  decode_row_kernel<<<dim3(p.H, p.N), rowk::RT, rowk::SMEM, stream>>>(p);
  const long items = long(p.N) * p.H * stk::NV;
  decode_state_rec_kernel<<<int(items < grid_cap ? items : grid_cap), stk::ST, stk::SMEM, stream>>>(p);
}

void record_state_launch(const RecRef &rr, const int *slots, int N, float *out, cudaStream_t stream) {
  static std::once_flag attr[64];  // the attribute is per device
  int attr_dev = 0;
  cudaGetDevice(&attr_dev);
  std::call_once(attr[unsigned(attr_dev) % 64u], [&] {
    cudaFuncSetAttribute(record_state_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 16384 + 8192);
  });
  if (N > 0) record_state_kernel<<<dim3(rr.H * 2, N), NT, 16384 + 8192, stream>>>(rr, slots, out);
}

}  // namespace kda_sm100
