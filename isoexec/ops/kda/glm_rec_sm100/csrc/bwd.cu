// SPDX-License-Identifier: Apache-2.0
// Trainer backward of recipe sm100_kda_rec_v1: the gradient of the token recurrence (recipe.cuh) with the roundings
// treated as identities. Deterministic (every reduction is a fixed chain / fixed tree, one writer per output
// element, no atomics); it may depend on T (the forward saves one fp32 state per 64 tokens -- DESIGN.md section 6).
//
// CTA = (sequence, head), 512 threads: thread (cgx = tid >> 7, c = tid & 127) owns channel c of the 32 value columns
// j = 32 cgx + i of its group. A persistent grid loops over the (sequence, head) items. Per 64-token block, last to
// first: (R) restore the block's checkpoint state, re-run the block forward writing the entering state of every token
// to a per-CTA global scratch and the per-token scalars kS_j, delta_j to shared memory; (B) walk the block's tokens in
// reverse with the adjoint state dS in registers (see DESIGN.md for the adjoint equations).
#include <cuda_runtime.h>

#include "recipe.cuh"

namespace kda_rec {

struct BwdParams {
  const uint16_t *q, *k, *v;
  int64_t q_ts, k_ts, v_ts;
  const float *g;
  int64_t g_ts;
  const void *beta;
  int64_t beta_ts;
  int beta_mode;                     // 0 bf16, 1 fp32 (post-sigmoid)
  const int *cu;
  const float *ckpt;                 // [blocks, H, 128, 128]
  const int *cp_base;
  const uint16_t *dO;                // bf16 [T, H, 128], token stride dO_ts
  int64_t dO_ts;
  const float *dht;                  // [N, H, 128, 128] v-first or nullptr
  uint16_t *dq, *dk, *dv;            // bf16, token stride d_ts (dense [T, H, 128])
  float *dg;                         // fp32 [T, H, 128]
  float *dbeta;                      // fp32 [T, H]
  float *dh0;                        // fp32 [N, H, 128, 128] or nullptr
  float *scratch;                    // [gridDim.x][64][128][128]
  int N, H;
};

namespace {
constexpr int NT = 512, CI = 32, BLK = 64;

struct Smem {
  float kh[D], qh[D], av[D], rk_raw[D], rq_raw[D];
  float scal[8];                       // rk, rq, beta, dbeta partials...
  float red[4][CI][4];                 // channel-reduction partials [group][col][warp]
  float nred[2][4][4];                 // norm / l2-backward partials [which][group][warp]
  float part[3][4][D];                 // dq^, dk^, da partials by group
  float sdO[D], sv[D];
  float dbp[4];
  float kS[BLK][D], dl[BLK][D];        // per token, per column: kS_j and delta_j
  // the block's raw rows, staged by cp.async before the block's two sweeps (no per-token global round trips)
  uint16_t rk[BLK][D], rq[BLK][D], rv[BLK][D], rdO[BLK][D];
  float rg[BLK][D];
  float rb[BLK];
};

// sum over the 32 lanes of a warp, fixed xor order 16, 8, 4, 2, 1 (all lanes end with the sum's bits)
__device__ __forceinline__ float wsum(float v) {
  v = fadd(v, __shfl_xor_sync(0xffffffffu, v, 16));
  v = fadd(v, __shfl_xor_sync(0xffffffffu, v, 8));
  v = fadd(v, __shfl_xor_sync(0xffffffffu, v, 4));
  v = fadd(v, __shfl_xor_sync(0xffffffffu, v, 2));
  v = fadd(v, __shfl_xor_sync(0xffffffffu, v, 1));
  return v;
}

// out[i] = sum over the 128 channels (all four warps of the group, fixed order) of x[i]; every thread gets all 32.
// Within a warp the 32 column sums come from a butterfly reduce-scatter (31 shuffles: lane c ends with column c's warp sum),
// the four warps' sums are combined in shared memory in a fixed order.
__device__ __forceinline__ void chan_reduce(Smem &sm, int cgx, int wc, int lane, const float (&x)[CI], float (&out)[CI]) {
  float v[CI];
  #pragma unroll
  for (int i = 0; i < CI; ++i) v[i] = x[i];
  #pragma unroll
  for (int o = 16; o >= 1; o >>= 1) {
    const bool hi = (lane & o) != 0;
    #pragma unroll
    for (int i = 0; i < o; ++i) {
      const float lo_v = v[i], hi_v = v[i + o];
      const float send = hi ? lo_v : hi_v, keep = hi ? hi_v : lo_v;
      v[i] = fadd(keep, __shfl_xor_sync(0xffffffffu, send, o));
    }
  }
  sm.red[cgx][lane][wc] = v[0];
  __syncthreads();
  #pragma unroll
  for (int i = 0; i < CI; ++i) out[i] = fadd(fadd(sm.red[cgx][i][0], sm.red[cgx][i][1]), fadd(sm.red[cgx][i][2], sm.red[cgx][i][3]));
  __syncthreads();
}

// Token preparation by the group-0 threads into shared memory: k^, q^, a and the raw rows / norm scalars.
__device__ __forceinline__ void prep(const BwdParams &p, Smem &sm, int lt, int tid) {
  if (tid < D) {
    const int c = tid, wc = c >> 5, lane = c & 31;
    const float k = bf(sm.rk[lt][c]), q = bf(sm.rq[lt][c]);
    float sk = wsum(fmul(k, k)), sq = wsum(fmul(q, q));
    if (lane == 0) { sm.nred[0][0][wc] = sk; sm.nred[1][0][wc] = sq; }
    sm.rk_raw[c] = k; sm.rq_raw[c] = q;
  }
  __syncthreads();
  if (tid < D) {
    const int c = tid;
    const float sk = fadd(fadd(sm.nred[0][0][0], sm.nred[0][0][1]), fadd(sm.nred[0][0][2], sm.nred[0][0][3]));
    const float sq = fadd(fadd(sm.nred[1][0][0], sm.nred[1][0][1]), fadd(sm.nred[1][0][2], sm.nred[1][0][3]));
    const float rk = __frsqrt_rn(fadd(sk, 1e-6f)), rq = __frsqrt_rn(fadd(sq, 1e-6f));
    sm.kh[c] = fmul(sm.rk_raw[c], rk);
    sm.qh[c] = fmul(sm.rq_raw[c], fmul(rq, QSCALE));
    sm.av[c] = ex2(fmul(sm.rg[lt][c], LOG2E));
    if (c == 0) { sm.scal[0] = rk; sm.scal[1] = rq; }
  }
  __syncthreads();
}
}  // namespace

__global__ void __launch_bounds__(NT, 1) kda_rec_bwd(BwdParams p) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  Smem &sm = *reinterpret_cast<Smem *>(smem_raw);
  const int tid = threadIdx.x, cgx = tid >> 7, c = tid & 127, wc = c >> 5, lane = c & 31;
  float *scratch = p.scratch + int64_t(blockIdx.x) * BLK * D * D;
  for (int item = blockIdx.x; item < p.N * p.H; item += gridDim.x) {
    const int n = item / p.H, h = item % p.H;
    const int t0 = p.cu[n], t1 = p.cu[n + 1], Tn = t1 - t0;
    float dS[CI];
    #pragma unroll
    for (int i = 0; i < CI; ++i)
      dS[i] = p.dht ? p.dht[((int64_t(n) * p.H + h) * D + cgx * CI + i) * D + c] : 0.f;
    const int nblk = (Tn + BLK - 1) / BLK;
    for (int b = nblk - 1; b >= 0; --b) {
      const int L = min(BLK, Tn - b * BLK);
      const int tb = t0 + b * BLK;
      // ---------------------------------------------------------------- stage the block's raw rows
      {
        constexpr int PCS = 16 * 4 + 32;                // 16-byte pieces per token: k, q, v, dO (16 each) + g (32)
        for (int piece = tid; piece < L * PCS; piece += NT) {
          const int lt2 = piece / PCS, r = piece % PCS, t2 = tb + lt2;
          const void *src;
          void *dst;
          if (r < 16) { src = p.k + int64_t(t2) * p.k_ts + h * D + r * 8; dst = &sm.rk[lt2][r * 8]; }
          else if (r < 32) { src = p.q + int64_t(t2) * p.q_ts + h * D + (r - 16) * 8; dst = &sm.rq[lt2][(r - 16) * 8]; }
          else if (r < 48) { src = p.v + int64_t(t2) * p.v_ts + h * D + (r - 32) * 8; dst = &sm.rv[lt2][(r - 32) * 8]; }
          else if (r < 64) { src = p.dO + int64_t(t2) * p.dO_ts + h * D + (r - 48) * 8; dst = &sm.rdO[lt2][(r - 48) * 8]; }
          else { src = p.g + int64_t(t2) * p.g_ts + h * D + (r - 64) * 4; dst = &sm.rg[lt2][(r - 64) * 4]; }
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(uint32_t(__cvta_generic_to_shared(dst))), "l"(src) : "memory");
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
        if (tid < L)
          sm.rb[tid] = p.beta_mode == 1 ? static_cast<const float *>(p.beta)[int64_t(tb + tid) * p.beta_ts + h]
                                        : bf(static_cast<const uint16_t *>(p.beta)[int64_t(tb + tid) * p.beta_ts + h]);
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
      }
      // ---------------------------------------------------------------- (R) recompute the block forward
      float S[CI];
      {
        const float *src = p.ckpt + (int64_t(p.cp_base[n]) + b) * int64_t(p.H) * D * D + int64_t(h) * D * D;
        #pragma unroll
        for (int i = 0; i < CI; ++i) S[i] = src[(cgx * CI + i) * D + c];
      }
      for (int lt = 0; lt < L; ++lt) {
        const int t = tb + lt;
        #pragma unroll
        for (int i = 0; i < CI; ++i) scratch[(int64_t(lt) * D + cgx * CI + i) * D + c] = S[i];
        prep(p, sm, lt, tid);
        if (tid < D) sm.sv[tid] = bf(sm.rv[lt][tid]);
        if (tid == 0) sm.scal[2] = sm.rb[lt];
        __syncthreads();
        const float a = sm.av[c], kh = sm.kh[c], beta = sm.scal[2];
        float A[CI], part[CI], kS[CI];
        #pragma unroll
        for (int i = 0; i < CI; ++i) { A[i] = fmul(S[i], a); part[i] = fmul(A[i], kh); }
        chan_reduce(sm, cgx, wc, lane, part, kS);
        #pragma unroll
        for (int i = 0; i < CI; ++i) {
          const float delta = fmul(beta, fsub(sm.sv[cgx * CI + i], kS[i]));
          S[i] = ffma(delta, kh, A[i]);
          if (c == i) { sm.kS[lt][cgx * CI + i] = kS[i]; sm.dl[lt][cgx * CI + i] = delta; }
        }
        __syncthreads();
      }
      // ---------------------------------------------------------------- (B) reverse sweep of the block
      for (int lt = L - 1; lt >= 0; --lt) {
        const int t = tb + lt;
        prep(p, sm, lt, tid);
        if (tid < D) {
          sm.sdO[tid] = bf(sm.rdO[lt][tid]);
          sm.sv[tid] = bf(sm.rv[lt][tid]);
        }
        if (tid == 0) sm.scal[2] = sm.rb[lt];
        __syncthreads();
        const float a = sm.av[c], kh = sm.kh[c], qh = sm.qh[c], beta = sm.scal[2];
        float pd[CI], dd[CI];
        float dqh = 0.f;
        #pragma unroll
        for (int i = 0; i < CI; ++i) {
          const float s = scratch[(int64_t(lt) * D + cgx * CI + i) * D + c];
          const float A = fmul(s, a);
          const float Sp = ffma(sm.dl[lt][cgx * CI + i], kh, A);
          const float dO = sm.sdO[cgx * CI + i];
          dqh = ffma(dO, Sp, dqh);
          const float G = fadd(dS[i], fmul(dO, qh));
          dS[i] = G;
          pd[i] = fmul(G, kh);
        }
        chan_reduce(sm, cgx, wc, lane, pd, dd);
        float dkh = 0.f, da = 0.f, dbp = 0.f;
        #pragma unroll
        for (int i = 0; i < CI; ++i) {
          const float s = scratch[(int64_t(lt) * D + cgx * CI + i) * D + c];
          const float A = fmul(s, a);
          const float delta = sm.dl[lt][cgx * CI + i];
          const float dkS = fmul(-beta, dd[i]);
          const float G = dS[i];
          const float dA = ffma(dkS, kh, G);
          dkh = ffma(G, delta, ffma(dkS, A, dkh));
          da = ffma(dA, s, da);
          dS[i] = fmul(dA, a);
          dbp = ffma(dd[i], fsub(sm.sv[cgx * CI + i], sm.kS[lt][cgx * CI + i]), dbp);
          if (c == i) p.dv[int64_t(t) * D * p.H + int64_t(h) * D + cgx * CI + i] = rn(fmul(beta, dd[i]));
        }
        sm.part[0][cgx][c] = dqh; sm.part[1][cgx][c] = dkh; sm.part[2][cgx][c] = da;
        if (c == 0) sm.dbp[cgx] = dbp;
        __syncthreads();
        float dqc = 0.f, dkc = 0.f, dac = 0.f;
        if (tid < D) {
          dqc = fadd(fadd(sm.part[0][0][c], sm.part[0][1][c]), fadd(sm.part[0][2][c], sm.part[0][3][c]));
          dkc = fadd(fadd(sm.part[1][0][c], sm.part[1][1][c]), fadd(sm.part[1][2][c], sm.part[1][3][c]));
          dac = fadd(fadd(sm.part[2][0][c], sm.part[2][1][c]), fadd(sm.part[2][2][c], sm.part[2][3][c]));
          const float xq = wsum(fmul(sm.rq_raw[c], dqc)), xk = wsum(fmul(sm.rk_raw[c], dkc));
          if (lane == 0) { sm.nred[0][1][wc] = xk; sm.nred[1][1][wc] = xq; }
        }
        __syncthreads();
        if (tid < D) {
          const float xk = fadd(fadd(sm.nred[0][1][0], sm.nred[0][1][1]), fadd(sm.nred[0][1][2], sm.nred[0][1][3]));
          const float xq = fadd(fadd(sm.nred[1][1][0], sm.nred[1][1][1]), fadd(sm.nred[1][1][2], sm.nred[1][1][3]));
          const float rk = sm.scal[0], rq = sm.scal[1];
          const float r3k = fmul(fmul(rk, rk), rk), r3q = fmul(fmul(rq, rq), rq);
          const float dk = fsub(fmul(rk, dkc), fmul(fmul(sm.rk_raw[c], r3k), xk));
          const float dq = fmul(QSCALE, fsub(fmul(rq, dqc), fmul(fmul(sm.rq_raw[c], r3q), xq)));
          p.dq[int64_t(t) * D * p.H + int64_t(h) * D + c] = rn(dq);
          p.dk[int64_t(t) * D * p.H + int64_t(h) * D + c] = rn(dk);
          p.dg[int64_t(t) * D * p.H + int64_t(h) * D + c] = fmul(dac, a);
          if (c == 0) p.dbeta[int64_t(t) * p.H + h] = fadd(fadd(sm.dbp[0], sm.dbp[1]), fadd(sm.dbp[2], sm.dbp[3]));
        }
        __syncthreads();
      }
    }
    if (p.dh0) {
      #pragma unroll
      for (int i = 0; i < CI; ++i) p.dh0[((int64_t(n) * p.H + h) * D + cgx * CI + i) * D + c] = dS[i];
    }
  }
}

size_t bwd_smem_bytes() { return sizeof(Smem); }
void bwd_launch(const BwdParams &p, int grid, cudaStream_t st) {
  static bool attr = false;
  if (!attr) {
    cudaFuncSetAttribute(kda_rec_bwd, cudaFuncAttributeMaxDynamicSharedMemorySize, int(sizeof(Smem)));
    attr = true;
  }
  kda_rec_bwd<<<grid, NT, sizeof(Smem), st>>>(p);
}

}  // namespace kda_rec
