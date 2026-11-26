// SPDX-License-Identifier: Apache-2.0
// The recurrent KDA kernel of recipe sm100_kda_rec_v1 (recipe.cuh is the contract). One kernel body serves
// prefill, decode and the trainer forward: a sequence of T tokens is `step` applied T times in order, with
// the fp32 state of the CTA's value columns held in registers across tokens.
//
// CTA = (value slice of VB columns, head, sequence). Thread = (segment s = tid & 7, column group tid >> 3);
// a column group owns NC consecutive value columns, a segment owns 16 channels c = 32 m + 4 s + e.
#include <cuda_runtime.h>

#include "recipe.cuh"

namespace kda_rec {

struct Params {
  const uint16_t *q, *k, *v;        // bf16 token rows: element (t, h, c) at p + t * *_ts + h * 128 + c
  int64_t q_ts, k_ts, v_ts;
  const float *g;                   // unfused log decay fp32, token stride g_ts
  int64_t g_ts;
  const uint16_t *f;                // fused gate: raw bf16 forget projection, token stride f_ts
  int64_t f_ts;
  const float *dt_bias, *A_log;     // fused gate: [H*128], [H]
  float lb;
  const void *beta;                 // beta_mode 0: bf16 post-sigmoid, 1: fp32, 2: raw bf16 (rn(sigmoid))
  int64_t beta_ts;
  int beta_mode;
  const int *cu;                    // [N+1] tokens, or nullptr: one token per sequence (token = n)
  const int *slot_in, *slot_out;    // [N]; slot_in < 0 / == null_slot: zero state; slot_out < 0 / == null_slot: no state write
  int null_slot;
  float *state;                     // [slots][H][128 v][128 k] fp32
  int64_t slot_stride;              // floats
  uint16_t *o;                      // bf16, token stride o_ts
  int64_t o_ts;
  int N, H;
  float *ckpt;                      // trainer forward: the state entering every 64th token of each sequence (nullptr: none)
  const int *cp_base;               // [N] first checkpoint index of each sequence
};

// Pair operations: one packed fp32x2 instruction (sm_100 FFMA2/FMUL2/FADD2), or -- PK = false, the literal scalar
// reference path the tests compare against -- the same RN operation on each component. Bit-identical by IEEE.
template <bool PK> __device__ __forceinline__ float2 mul2(float2 x, float2 y) {
  if (PK) return __fmul2_rn(x, y);
  return make_float2(fmul(x.x, y.x), fmul(x.y, y.y));
}
template <bool PK> __device__ __forceinline__ float2 fma2(float2 x, float2 y, float2 z) {
  if (PK) return __ffma2_rn(x, y, z);
  return make_float2(ffma(x.x, y.x, z.x), ffma(x.y, y.y, z.y));
}
template <bool PK> __device__ __forceinline__ float2 add2(float2 x, float2 y) {
  if (PK) return __fadd2_rn(x, y);
  return make_float2(fadd(x.x, y.x), fadd(x.y, y.y));
}

// L lanes per value column (8, 16 or 32), NC columns per lane group, ML = 32 / L m-values (4 channels each) per lane.
template <int L, int NC, bool FG>
struct Raw {
  static constexpr int ML = 32 / L;
  uint2 k[ML], q[ML];
  float4 g[ML];         // unfused
  uint2 f;              // fused: this lane's gate slice (local m = its group index in the warp)
  uint32_t vv[(NC + 1) / 2];
  float beta;
};

// R128 root from this lane's ML partials: Q_s pairing, then xor 8 / 16 (m pairs), then the s tree.
template <int L>
__device__ __forceinline__ float root(const float (&P)[32 / L]) {
  float x;
  if (L == 8) x = fadd(fadd(P[0], P[1]), fadd(P[2], P[3]));
  else if (L == 16) x = fadd(P[0], P[1]);
  else x = P[0];
  if (L >= 16) x = fadd(x, __shfl_xor_sync(0xffffffffu, x, 8));
  if (L >= 32) x = fadd(x, __shfl_xor_sync(0xffffffffu, x, 16));
  return tree8(x);
}

// G independent R128 roots at once (same tree per value; interleaved so the shuffle latencies overlap).
template <int L, int G>
__device__ __forceinline__ void roots(const float (&P)[G][32 / L], float (&out)[G]) {
  #pragma unroll
  for (int g = 0; g < G; ++g) {
    if (L == 8) out[g] = fadd(fadd(P[g][0], P[g][1]), fadd(P[g][2], P[g][3]));
    else if (L == 16) out[g] = fadd(P[g][0], P[g][1]);
    else out[g] = P[g][0];
  }
  if (L >= 16) {
    #pragma unroll
    for (int g = 0; g < G; ++g) out[g] = fadd(out[g], __shfl_xor_sync(0xffffffffu, out[g], 8));
  }
  if (L >= 32) {
    #pragma unroll
    for (int g = 0; g < G; ++g) out[g] = fadd(out[g], __shfl_xor_sync(0xffffffffu, out[g], 16));
  }
  #pragma unroll
  for (int g = 0; g < G; ++g) out[g] = fadd(out[g], __shfl_xor_sync(0xffffffffu, out[g], 1));
  #pragma unroll
  for (int g = 0; g < G; ++g) out[g] = fadd(out[g], __shfl_xor_sync(0xffffffffu, out[g], 2));
  #pragma unroll
  for (int g = 0; g < G; ++g) out[g] = fadd(out[g], __shfl_xor_sync(0xffffffffu, out[g], 4));
}

// ---------------------------------------------------------------------------------------------- flat kernel
// One-token (decode) steps: every thread loads and prepares its own token (latency of a single step is what matters, and
// there is no next token to prefetch). The same recipe functions as the warp-specialized kernel below.
template <int L, int NC, int VB, bool FG, bool PK>
__global__ void __launch_bounds__((VB / NC) * L) kda_rec_flat(Params p) {
  constexpr int ML = 32 / L, CL = 4 * ML, NG = VB / NC;
  const int tid = threadIdx.x, lane = tid & 31, l = tid % L, s = l & 7, mh = l >> 3, cg = tid / L;
  const int h = blockIdx.y, n = blockIdx.z;
  const int j0 = blockIdx.x * VB + cg * NC;      // first owned value column
  const int sin = p.slot_in[n], sout = p.slot_out[n];
  const bool null_lane = sout == p.null_slot;    // padded lane (vLLM NULL_BLOCK_ID): zero outputs, nothing touched
  const bool wr = sout >= 0 && !null_lane;
  int t0 = n, t1 = n + 1;
  if (p.cu != nullptr) { t0 = p.cu[n]; t1 = p.cu[n + 1]; }
  if (null_lane) {
    for (int t = t0; t < t1; ++t)
      if (l == 0) {
        #pragma unroll
        for (int i = 0; i < NC; ++i) p.o[t * p.o_ts + h * D + j0 + i] = 0;
      }
    return;
  }
  // channel of (local m', e): 32 (mh ML + m') + 4 s + e
  static_assert(NC % 2 == 0, "columns are processed in pairs (packed fp32x2 instructions)");
  float2 S[NC / 2][CL];                          // .x = column j0 + 2 pair, .y = the next
  #pragma unroll
  for (int i = 0; i < NC; ++i) {
    #pragma unroll
    for (int mm = 0; mm < ML; ++mm) {
      float4 x = make_float4(0.f, 0.f, 0.f, 0.f);
      if (sin >= 0 && sin != p.null_slot)
        x = *reinterpret_cast<const float4 *>(p.state + int64_t(sin) * p.slot_stride + (int64_t(h) * D + j0 + i) * D +
                                              32 * (mh * ML + mm) + 4 * s);
      float2 *d = S[i >> 1];
      if (i & 1) { d[4 * mm].y = x.x; d[4 * mm + 1].y = x.y; d[4 * mm + 2].y = x.z; d[4 * mm + 3].y = x.w; }
      else       { d[4 * mm].x = x.x; d[4 * mm + 1].x = x.y; d[4 * mm + 2].x = x.z; d[4 * mm + 3].x = x.w; }
    }
  }
  float ea = 0.f;
  if (FG) ea = expf(p.A_log[h]);
  const int gi = (lane / L);                      // group index within the warp (== local m this lane evaluates the gate for)

  Raw<L, NC, FG> cur, nxt;
  auto load = [&](Raw<L, NC, FG> &r, int t) {
    const uint16_t *kp = p.k + int64_t(t) * p.k_ts + h * D + 32 * mh * ML + 4 * s;
    const uint16_t *qp = p.q + int64_t(t) * p.q_ts + h * D + 32 * mh * ML + 4 * s;
    #pragma unroll
    for (int mm = 0; mm < ML; ++mm) {
      r.k[mm] = *reinterpret_cast<const uint2 *>(kp + 32 * mm);
      r.q[mm] = *reinterpret_cast<const uint2 *>(qp + 32 * mm);
    }
    if (FG) {
      r.f = *reinterpret_cast<const uint2 *>(p.f + int64_t(t) * p.f_ts + h * D + 32 * (mh * ML + gi) + 4 * s);
    } else {
      const float *gp = p.g + int64_t(t) * p.g_ts + h * D + 32 * mh * ML + 4 * s;
      #pragma unroll
      for (int mm = 0; mm < ML; ++mm) r.g[mm] = *reinterpret_cast<const float4 *>(gp + 32 * mm);
    }
    const uint16_t *vp = p.v + int64_t(t) * p.v_ts + h * D + j0;
    #pragma unroll
    for (int i = 0; i < NC / 2; ++i) r.vv[i] = *reinterpret_cast<const uint32_t *>(vp + 2 * i);
    const int64_t bi = int64_t(t) * p.beta_ts + h;
    r.beta = p.beta_mode == 1 ? static_cast<const float *>(p.beta)[bi] : bf(static_cast<const uint16_t *>(p.beta)[bi]);
  };
  if (t0 < t1) load(nxt, t0);
  for (int t = t0; t < t1; ++t) {
    cur = nxt;
    if (t + 1 < t1) load(nxt, t + 1);
    // ---- token prep
    float kf[CL], qf[CL], a[CL];
    {
      float Pk[ML], Pq[ML];
      #pragma unroll
      for (int mm = 0; mm < ML; ++mm) {
        const uint32_t w[4] = {cur.k[mm].x, cur.k[mm].y, cur.q[mm].x, cur.q[mm].y};
        kf[4 * mm + 0] = bf(w[0] & 0xffffu); kf[4 * mm + 1] = bf(w[0] >> 16);
        kf[4 * mm + 2] = bf(w[1] & 0xffffu); kf[4 * mm + 3] = bf(w[1] >> 16);
        qf[4 * mm + 0] = bf(w[2] & 0xffffu); qf[4 * mm + 1] = bf(w[2] >> 16);
        qf[4 * mm + 2] = bf(w[3] & 0xffffu); qf[4 * mm + 3] = bf(w[3] >> 16);
        float ak = 0.f, aq = 0.f;
        #pragma unroll
        for (int e = 0; e < 4; ++e) { ak = ffma(kf[4 * mm + e], kf[4 * mm + e], ak); aq = ffma(qf[4 * mm + e], qf[4 * mm + e], aq); }
        Pk[mm] = ak; Pq[mm] = aq;
      }
      const float ssk = root<L>(Pk), ssq = root<L>(Pq);
      const float ik = __frsqrt_rn(fadd(ssk, 1e-6f));
      const float iq = fmul(__frsqrt_rn(fadd(ssq, 1e-6f)), QSCALE);
      #pragma unroll
      for (int c = 0; c < CL; ++c) { kf[c] = fmul(kf[c], ik); qf[c] = fmul(qf[c], iq); }
    }
    if (FG) {
      float gm[4];
      const uint32_t w0 = cur.f.x, w1 = cur.f.y;
      const float fv[4] = {bf(w0 & 0xffffu), bf(w0 >> 16), bf(w1 & 0xffffu), bf(w1 >> 16)};
      #pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int c = 32 * (mh * ML + gi) + 4 * s + e;
        gm[e] = fmul(p.lb, sigm(fmul(ea, fadd(fv[e], p.dt_bias[h * D + c]))));
      }
      #pragma unroll
      for (int mm = 0; mm < ML; ++mm) {
        #pragma unroll
        for (int e = 0; e < 4; ++e)
          a[4 * mm + e] = ex2(fmul(__shfl_sync(0xffffffffu, gm[e], mm * L + l), LOG2E));
      }
    }
    if (!FG) {
      #pragma unroll
      for (int mm = 0; mm < ML; ++mm) {
        a[4 * mm + 0] = ex2(fmul(cur.g[mm].x, LOG2E)); a[4 * mm + 1] = ex2(fmul(cur.g[mm].y, LOG2E));
        a[4 * mm + 2] = ex2(fmul(cur.g[mm].z, LOG2E)); a[4 * mm + 3] = ex2(fmul(cur.g[mm].w, LOG2E));
      }
    }
    float beta = cur.beta;
    if (p.beta_mode == 2) beta = bf(rn(sigm(beta)));
    // ---- the update, per pair of owned columns (packed pairs when PK; each lane op is the scalar RN op)
    float o[NC];
    #pragma unroll
    for (int pr = 0; pr < NC / 2; ++pr) {
      float2 A[CL];
      float2 P2[ML];
      #pragma unroll
      for (int mm = 0; mm < ML; ++mm) {
        float2 acc = make_float2(0.f, 0.f);
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
          A[4 * mm + e] = mul2<PK>(S[pr][4 * mm + e], make_float2(a[4 * mm + e], a[4 * mm + e]));
          acc = fma2<PK>(A[4 * mm + e], make_float2(kf[4 * mm + e], kf[4 * mm + e]), acc);
        }
        P2[mm] = acc;
      }
      float Px[ML], Py[ML];
      #pragma unroll
      for (int mm = 0; mm < ML; ++mm) { Px[mm] = P2[mm].x; Py[mm] = P2[mm].y; }
      const float2 kS = make_float2(root<L>(Px), root<L>(Py));
      const uint32_t vw = cur.vv[pr];
      const float2 vj = make_float2(bf(vw & 0xffffu), bf(vw >> 16));
      const float2 delta = mul2<PK>(make_float2(beta, beta), add2<PK>(vj, make_float2(-kS.x, -kS.y)));
      #pragma unroll
      for (int mm = 0; mm < ML; ++mm) {
        float2 acc = make_float2(0.f, 0.f);
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
          S[pr][4 * mm + e] = fma2<PK>(delta, make_float2(kf[4 * mm + e], kf[4 * mm + e]), A[4 * mm + e]);
          acc = fma2<PK>(S[pr][4 * mm + e], make_float2(qf[4 * mm + e], qf[4 * mm + e]), acc);
        }
        P2[mm] = acc;
      }
      #pragma unroll
      for (int mm = 0; mm < ML; ++mm) { Px[mm] = P2[mm].x; Py[mm] = P2[mm].y; }
      o[2 * pr] = root<L>(Px);
      o[2 * pr + 1] = root<L>(Py);
    }
    if (l == 0) {
      uint16_t *op = p.o + int64_t(t) * p.o_ts + h * D + j0;
      if (NC == 2) *reinterpret_cast<uint32_t *>(op) = uint32_t(rn(o[0])) | (uint32_t(rn(o[1])) << 16);
      else if (NC == 4)
        *reinterpret_cast<uint2 *>(op) = make_uint2(uint32_t(rn(o[0])) | (uint32_t(rn(o[1])) << 16),
                                                    uint32_t(rn(o[2])) | (uint32_t(rn(o[3])) << 16));
      else {
        #pragma unroll
        for (int i = 0; i < NC; ++i) op[i] = rn(o[i]);
      }
    }
  }
  if (wr) {
    #pragma unroll
    for (int i = 0; i < NC; ++i) {
      #pragma unroll
      for (int mm = 0; mm < ML; ++mm)
      {
        const float2 *d = S[i >> 1];
        const float4 x = (i & 1) ? make_float4(d[4 * mm].y, d[4 * mm + 1].y, d[4 * mm + 2].y, d[4 * mm + 3].y)
                                 : make_float4(d[4 * mm].x, d[4 * mm + 1].x, d[4 * mm + 2].x, d[4 * mm + 3].x);
        *reinterpret_cast<float4 *>(p.state + int64_t(sout) * p.slot_stride + (int64_t(h) * D + j0 + i) * D +
                                    32 * (mh * ML + mm) + 4 * s) = x;
      }
    }
  }
}


// Warp-specialized kernel. CTA = C consumer warps (NG column groups x L lanes, they own the state of the CTA's value
// columns) + 2 producer warps (token parity). The producers run ahead of the consumers: they fetch each token's raw rows
// (cp.async ring), prepare k^, q^, a and beta ONCE per CTA (not once per column group) and publish them through a stage
// ring in shared memory guarded by named barriers (FULL / EMPTY per stage). Execution only: every value is the same
// function (recipe.cuh) of the same inputs, evaluated by the same instruction sequences.
template <int L, int NC, int VB, bool FG, bool PK, bool CK>
__global__ void __launch_bounds__((VB / NC * L / 32 + 2) * 32) kda_rec(Params p) {
  constexpr int ML = 32 / L, CL = 4 * ML, NG = VB / NC;
  constexpr int CW = NG * L / 32;                 // consumer warps
  static_assert(NG * L % 32 == 0 && NC % 2 == 0, "consumer groups fill whole warps; columns in pairs");
  constexpr int ST = 4, NB = (1 + CW) * 32;      // stage ring depth; threads counted by each stage barrier
  const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31;
  const int h = blockIdx.y, n = blockIdx.z;
  const int sin = p.slot_in[n], sout = p.slot_out[n];
  const bool null_lane = sout == p.null_slot;    // padded lane (vLLM NULL_BLOCK_ID): zero outputs, nothing touched
  const bool wr = sout >= 0 && !null_lane;
  int t0 = n, t1 = n + 1;
  if (p.cu != nullptr) { t0 = p.cu[n]; t1 = p.cu[n + 1]; }
  const int jb = blockIdx.x * VB;                 // first value column of the CTA

  struct alignas(16) Stage { float kf[D], qf[D], a[D]; float beta; uint32_t vv[VB / 2]; };
  __shared__ __align__(16) Stage stg[ST];
  constexpr int RAWB = 4;                         // producer raw ring depth (per producer warp)
  struct alignas(16) RawSlot { uint2 k, q; float4 g; uint2 f; uint32_t beta; };
  __shared__ __align__(16) RawSlot raw[2][RAWB][32];
  __shared__ __align__(16) uint32_t rawv[2][RAWB][VB / 2];

  if (null_lane) {
    if (wid < CW) {
      const int l = tid % L, cg = tid / L;
      for (int t = t0; t < t1; ++t)
        if (l == 0) {
          #pragma unroll
          for (int i = 0; i < NC; ++i) p.o[t * p.o_ts + h * D + jb + cg * NC + i] = 0;
        }
    }
    return;
  }

  auto bar_sync = [](int id) { asm volatile("bar.sync %0, %1;" ::"r"(id), "n"(NB) : "memory"); };
  auto bar_arrive = [](int id) { asm volatile("bar.arrive %0, %1;" ::"r"(id), "n"(NB) : "memory"); };

  if (wid >= CW) {
    // ============================================================================================== producer warp
    const int pw = wid - CW;                      // 0: tokens t0, t0 + 2, ...; 1: t0 + 1, t0 + 3, ...
    const int s = lane & 7, mh = lane >> 3;       // L = 32 mapping: channels 32 mh + 4 s + e
    float ea = 0.f;
    if (FG) ea = expf(p.A_log[h]);
    auto cp_ = [&](int sz, void *sm, const void *gp) {
      const uint32_t a_ = uint32_t(__cvta_generic_to_shared(sm));
      if (sz == 16) asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" ::"r"(a_), "l"(gp) : "memory");
      else if (sz == 8) asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" ::"r"(a_), "l"(gp) : "memory");
      else asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(a_), "l"(gp) : "memory");
    };
    auto issue = [&](int u) {                     // raw rows of token u into this warp's ring
      if (u < t1) {
        RawSlot &r = raw[pw][((u - t0) >> 1) % RAWB][lane];
        cp_(8, &r.k, p.k + int64_t(u) * p.k_ts + h * D + 32 * mh + 4 * s);
        cp_(8, &r.q, p.q + int64_t(u) * p.q_ts + h * D + 32 * mh + 4 * s);
        if (FG) cp_(8, &r.f, p.f + int64_t(u) * p.f_ts + h * D + 32 * mh + 4 * s);
        else cp_(16, &r.g, p.g + int64_t(u) * p.g_ts + h * D + 32 * mh + 4 * s);
        #pragma unroll
        for (int w = lane; w < VB / 2; w += 32)
          cp_(4, &rawv[pw][((u - t0) >> 1) % RAWB][w], p.v + int64_t(u) * p.v_ts + h * D + jb + 2 * w);
        const int64_t bi = int64_t(u) * p.beta_ts + h;
        if (p.beta_mode == 1) cp_(4, &r.beta, static_cast<const float *>(p.beta) + bi);
        else cp_(4, &r.beta, static_cast<const uint16_t *>(p.beta) + (bi & ~int64_t(1)));
      }
      asm volatile("cp.async.commit_group;" ::: "memory");
    };
    issue(t0 + pw); issue(t0 + pw + 2); issue(t0 + pw + 4);
    for (int u = t0 + pw; u < t1; u += 2) {
      asm volatile("cp.async.wait_group 2;" ::: "memory");
      issue(u + 6);
      // (issue after the wait: the oldest group is the one for token u; two newer groups stay in flight)
      const RawSlot r = raw[pw][((u - t0) >> 1) % RAWB][lane];
      float kf[4], qf[4], a[4];
      {
        const uint32_t w[4] = {r.k.x, r.k.y, r.q.x, r.q.y};
        kf[0] = bf(w[0] & 0xffffu); kf[1] = bf(w[0] >> 16); kf[2] = bf(w[1] & 0xffffu); kf[3] = bf(w[1] >> 16);
        qf[0] = bf(w[2] & 0xffffu); qf[1] = bf(w[2] >> 16); qf[2] = bf(w[3] & 0xffffu); qf[3] = bf(w[3] >> 16);
        float Pk[1], Pq[1], ak = 0.f, aq = 0.f;
        #pragma unroll
        for (int e = 0; e < 4; ++e) { ak = ffma(kf[e], kf[e], ak); aq = ffma(qf[e], qf[e], aq); }
        Pk[0] = ak; Pq[0] = aq;
        const float ssk = root<32>(Pk), ssq = root<32>(Pq);
        const float ik = __frsqrt_rn(fadd(ssk, 1e-6f));
        const float iq = fmul(__frsqrt_rn(fadd(ssq, 1e-6f)), QSCALE);
        #pragma unroll
        for (int c = 0; c < 4; ++c) { kf[c] = fmul(kf[c], ik); qf[c] = fmul(qf[c], iq); }
      }
      if (FG) {
        const uint32_t w0 = r.f.x, w1 = r.f.y;
        const float fv[4] = {bf(w0 & 0xffffu), bf(w0 >> 16), bf(w1 & 0xffffu), bf(w1 >> 16)};
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float gv = fmul(p.lb, sigm(fmul(ea, fadd(fv[e], p.dt_bias[h * D + 32 * mh + 4 * s + e]))));
          a[e] = ex2(fmul(gv, LOG2E));
        }
      } else {
        a[0] = ex2(fmul(r.g.x, LOG2E)); a[1] = ex2(fmul(r.g.y, LOG2E)); a[2] = ex2(fmul(r.g.z, LOG2E)); a[3] = ex2(fmul(r.g.w, LOG2E));
      }
      float beta;
      if (p.beta_mode == 1) beta = __uint_as_float(r.beta);
      else beta = bf((((int64_t(u) * p.beta_ts + h) & 1) ? (r.beta >> 16) : (r.beta & 0xffffu)));
      if (p.beta_mode == 2) beta = bf(rn(sigm(beta)));
      const int st = (u - t0) % ST;
      bar_sync(1 + ST + st);                      // EMPTY[st]: the consumers released this stage
      Stage &S_ = stg[st];
      *reinterpret_cast<float4 *>(&S_.kf[32 * mh + 4 * s]) = make_float4(kf[0], kf[1], kf[2], kf[3]);
      *reinterpret_cast<float4 *>(&S_.qf[32 * mh + 4 * s]) = make_float4(qf[0], qf[1], qf[2], qf[3]);
      *reinterpret_cast<float4 *>(&S_.a[32 * mh + 4 * s]) = make_float4(a[0], a[1], a[2], a[3]);
      if (lane == 0) S_.beta = beta;
      #pragma unroll
      for (int w = lane; w < VB / 2; w += 32) S_.vv[w] = rawv[pw][((u - t0) >> 1) % RAWB][w];
      bar_arrive(1 + st);                         // FULL[st]
    }
    return;
  }

  // ================================================================================================ consumer warps
  const int l = tid % L, s = l & 7, mh = l >> 3, cg = tid / L;
  const int j0 = jb + cg * NC;                    // first owned value column
  float2 S[NC / 2][CL];                           // .x = column j0 + 2 pair, .y = the next
  #pragma unroll
  for (int i = 0; i < NC; ++i) {
    #pragma unroll
    for (int mm = 0; mm < ML; ++mm) {
      float4 x = make_float4(0.f, 0.f, 0.f, 0.f);
      if (sin >= 0 && sin != p.null_slot)
        x = *reinterpret_cast<const float4 *>(p.state + int64_t(sin) * p.slot_stride + (int64_t(h) * D + j0 + i) * D +
                                              32 * (mh * ML + mm) + 4 * s);
      float2 *d = S[i >> 1];
      if (i & 1) { d[4 * mm].y = x.x; d[4 * mm + 1].y = x.y; d[4 * mm + 2].y = x.z; d[4 * mm + 3].y = x.w; }
      else       { d[4 * mm].x = x.x; d[4 * mm + 1].x = x.y; d[4 * mm + 2].x = x.z; d[4 * mm + 3].x = x.w; }
    }
  }
  #pragma unroll
  for (int st = 0; st < ST; ++st) bar_arrive(1 + ST + st);    // every stage starts empty
  for (int t = t0; t < t1; ++t) {
    if (CK && ((t - t0) & 63) == 0) {    // exec-only extra stores: the exact entering fp32 state
      float *dst = p.ckpt + (int64_t(p.cp_base[n]) + ((t - t0) >> 6)) * int64_t(p.H) * D * D;
      #pragma unroll
      for (int i = 0; i < NC; ++i) {
        #pragma unroll
        for (int mm = 0; mm < ML; ++mm) {
          const float2 *d = S[i >> 1];
          const float4 x = (i & 1) ? make_float4(d[4 * mm].y, d[4 * mm + 1].y, d[4 * mm + 2].y, d[4 * mm + 3].y)
                                   : make_float4(d[4 * mm].x, d[4 * mm + 1].x, d[4 * mm + 2].x, d[4 * mm + 3].x);
          *reinterpret_cast<float4 *>(dst + (int64_t(h) * D + j0 + i) * D + 32 * (mh * ML + mm) + 4 * s) = x;
        }
      }
    }
    const int st = (t - t0) % ST;
    bar_sync(1 + st);                              // FULL[st]
    float kf[CL], qf[CL], a[CL];
    #pragma unroll
    for (int mm = 0; mm < ML; ++mm) {
      const int o4 = 32 * (mh * ML + mm) + 4 * s;
      const float4 x = *reinterpret_cast<const float4 *>(&stg[st].kf[o4]);
      const float4 y = *reinterpret_cast<const float4 *>(&stg[st].qf[o4]);
      const float4 z = *reinterpret_cast<const float4 *>(&stg[st].a[o4]);
      kf[4 * mm] = x.x; kf[4 * mm + 1] = x.y; kf[4 * mm + 2] = x.z; kf[4 * mm + 3] = x.w;
      qf[4 * mm] = y.x; qf[4 * mm + 1] = y.y; qf[4 * mm + 2] = y.z; qf[4 * mm + 3] = y.w;
      a[4 * mm] = z.x; a[4 * mm + 1] = z.y; a[4 * mm + 2] = z.z; a[4 * mm + 3] = z.w;
    }
    const float beta = stg[st].beta;
    uint32_t vv[NC / 2];
    #pragma unroll
    for (int i = 0; i < NC / 2; ++i) vv[i] = stg[st].vv[(cg * NC) / 2 + i];
    bar_arrive(1 + ST + st);                       // EMPTY[st]: everything of this stage is in registers
    // ---- the update, in groups of GS owned columns (packed pairs when PK); within a group every phase runs for all
    // columns before the next starts, so the shuffle trees of the columns overlap (execution order only)
    constexpr int GS = NC < 4 ? NC : 4;
    float o[NC];
    #pragma unroll
    for (int gb = 0; gb < NC; gb += GS) {
      float2 A[GS / 2][CL];
      float Pk[GS][ML], Pq[GS][ML];
      #pragma unroll
      for (int pp = 0; pp < GS / 2; ++pp) {
        const int pr = gb / 2 + pp;
        #pragma unroll
        for (int mm = 0; mm < ML; ++mm) {
          float2 acc = make_float2(0.f, 0.f);
          #pragma unroll
          for (int e = 0; e < 4; ++e) {
            A[pp][4 * mm + e] = mul2<PK>(S[pr][4 * mm + e], make_float2(a[4 * mm + e], a[4 * mm + e]));
            acc = fma2<PK>(A[pp][4 * mm + e], make_float2(kf[4 * mm + e], kf[4 * mm + e]), acc);
          }
          Pk[2 * pp][mm] = acc.x; Pk[2 * pp + 1][mm] = acc.y;
        }
      }
      float kS[GS];
      roots<L, GS>(Pk, kS);
      float2 delta[GS / 2];
      #pragma unroll
      for (int pp = 0; pp < GS / 2; ++pp) {
        const uint32_t vw = vv[gb / 2 + pp];
        const float2 vj = make_float2(bf(vw & 0xffffu), bf(vw >> 16));
        delta[pp] = mul2<PK>(make_float2(beta, beta), add2<PK>(vj, make_float2(-kS[2 * pp], -kS[2 * pp + 1])));
      }
      #pragma unroll
      for (int pp = 0; pp < GS / 2; ++pp) {
        const int pr = gb / 2 + pp;
        #pragma unroll
        for (int mm = 0; mm < ML; ++mm) {
          float2 acc = make_float2(0.f, 0.f);
          #pragma unroll
          for (int e = 0; e < 4; ++e) {
            S[pr][4 * mm + e] = fma2<PK>(delta[pp], make_float2(kf[4 * mm + e], kf[4 * mm + e]), A[pp][4 * mm + e]);
            acc = fma2<PK>(S[pr][4 * mm + e], make_float2(qf[4 * mm + e], qf[4 * mm + e]), acc);
          }
          Pq[2 * pp][mm] = acc.x; Pq[2 * pp + 1][mm] = acc.y;
        }
      }
      float og[GS];
      roots<L, GS>(Pq, og);
      #pragma unroll
      for (int g2 = 0; g2 < GS; ++g2) o[gb + g2] = og[g2];
    }
    if (l == 0) {
      uint16_t *op = p.o + int64_t(t) * p.o_ts + h * D + j0;
      if (NC == 2) *reinterpret_cast<uint32_t *>(op) = uint32_t(rn(o[0])) | (uint32_t(rn(o[1])) << 16);
      else if (NC == 4)
        *reinterpret_cast<uint2 *>(op) = make_uint2(uint32_t(rn(o[0])) | (uint32_t(rn(o[1])) << 16),
                                                    uint32_t(rn(o[2])) | (uint32_t(rn(o[3])) << 16));
      else {
        #pragma unroll
        for (int i = 0; i < NC; ++i) op[i] = rn(o[i]);
      }
    }
  }
  if (wr) {
    #pragma unroll
    for (int i = 0; i < NC; ++i) {
      #pragma unroll
      for (int mm = 0; mm < ML; ++mm) {
        const float2 *d = S[i >> 1];
        const float4 x = (i & 1) ? make_float4(d[4 * mm].y, d[4 * mm + 1].y, d[4 * mm + 2].y, d[4 * mm + 3].y)
                                 : make_float4(d[4 * mm].x, d[4 * mm + 1].x, d[4 * mm + 2].x, d[4 * mm + 3].x);
        *reinterpret_cast<float4 *>(p.state + int64_t(sout) * p.slot_stride + (int64_t(h) * D + j0 + i) * D +
                                    32 * (mh * ML + mm) + 4 * s) = x;
      }
    }
  }
}

template <int L, int NC, int VB, bool PKOK, bool CKOK>
static bool launch_nc(const Params &p, bool fused, bool packed, bool ws, cudaStream_t st) {
  if (!ws) {   // flat kernel (decode)
    const dim3 grid(D / VB, p.H, p.N), block((VB / NC) * L);
    if (packed || p.ckpt != nullptr) return false;
    if (fused) kda_rec_flat<L, NC, VB, true, false><<<grid, block, 0, st>>>(p);
    else kda_rec_flat<L, NC, VB, false, false><<<grid, block, 0, st>>>(p);
    return true;
  }
  const dim3 grid(D / VB, p.H, p.N), block(((VB / NC) * L / 32 + 2) * 32);
  if (p.ckpt != nullptr) {      // trainer forward: also saves the state entering every 64th token
    if constexpr (CKOK) {
      if (packed) return false;
      if (fused) kda_rec<L, NC, VB, true, false, true><<<grid, block, 0, st>>>(p);
      else kda_rec<L, NC, VB, false, false, true><<<grid, block, 0, st>>>(p);
      return true;
    }
    return false;
  }
  if (packed) {
    if constexpr (PKOK) {   // the packed fp32x2 path is instantiated for a few shapes only (compile time)
      if (fused) kda_rec<L, NC, VB, true, true, false><<<grid, block, 0, st>>>(p);
      else kda_rec<L, NC, VB, false, true, false><<<grid, block, 0, st>>>(p);
      return true;
    }
    return false;
  }
  if (fused) kda_rec<L, NC, VB, true, false, false><<<grid, block, 0, st>>>(p);   // the literal scalar recipe
  else kda_rec<L, NC, VB, false, false, false><<<grid, block, 0, st>>>(p);
  return true;
}

// Launch shapes (lanes per column, columns per lane group, columns per CTA). `ws` selects the warp-specialized kernel
// (prefill) over the flat one (decode); the two are the same arithmetic.
#define KDA_CASE(L_, NC_, VB_, PK_, CK_) if (lanes == L_ && nc == NC_ && vb == VB_) return launch_nc<L_, NC_, VB_, PK_, CK_>(p, fused, packed, true, st);
#define KDA_FLAT(L_, NC_, VB_) if (!ws && lanes == L_ && nc == NC_ && vb == VB_) return launch_nc<L_, NC_, VB_, false, false>(p, fused, packed, false, st);
bool rec_launch(const Params &p, bool fused, bool packed, bool ws, int lanes, int nc, int vb, cudaStream_t st) {
  if (p.N == 0 || p.H == 0) return true;
  KDA_FLAT(8, 4, 16) KDA_FLAT(8, 2, 8) KDA_FLAT(8, 4, 32) KDA_FLAT(8, 8, 32) KDA_FLAT(16, 4, 16) KDA_FLAT(32, 4, 16)
  KDA_FLAT(8, 4, 64) KDA_FLAT(8, 2, 32) KDA_FLAT(8, 2, 64) KDA_FLAT(8, 8, 64) KDA_FLAT(16, 4, 32) KDA_FLAT(8, 4, 128)
  if (!ws) return false;
  KDA_CASE(8, 4, 16, true, false) KDA_CASE(8, 4, 32, false, false) KDA_CASE(8, 2, 8, false, false) KDA_CASE(8, 8, 32, false, false)
  KDA_CASE(16, 4, 16, false, false) KDA_CASE(16, 2, 8, false, false)
  KDA_CASE(32, 4, 16, false, false) KDA_CASE(32, 4, 8, true, false) KDA_CASE(32, 2, 8, false, false)
  KDA_CASE(8, 2, 16, false, false) KDA_CASE(8, 2, 32, false, true) KDA_CASE(8, 2, 64, false, false) KDA_CASE(8, 4, 64, false, true)
  KDA_CASE(8, 4, 128, false, false) KDA_CASE(16, 2, 32, false, false) KDA_CASE(16, 4, 64, false, false)
  return false;
}

// ---------------------------------------------------------------------------- element-wise stages
// log_decay = lb (*) sigmoid(expf(A_log_h) (*) (bf(f) (+) dt_bias_c)) (fp32); beta = rn(sigmoid(bf(b))). The same
// definitions as the fused kernel and as glm_sm100's layer.cu, so all three agree bit for bit.
__global__ void gate_kernel(const uint16_t *f, int64_t f_ts, const float *dt_bias, const float *A_log, float lb,
                            float *out, int64_t n_tok, int HK) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n_tok * HK) return;
  const int64_t t = i / HK;
  const int c = int(i % HK);
  const float x = fadd(bf(f[t * f_ts + c]), dt_bias[c]);
  out[i] = fmul(lb, sigm(fmul(expf(A_log[c / D]), x)));
}
__global__ void beta_kernel(const uint16_t *b, uint16_t *out, int64_t n) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = rn(sigm(bf(b[i])));
}
void gate_launch(const uint16_t *f, int64_t f_ts, const float *dt_bias, const float *A_log, float lb, float *out,
                 int64_t n_tok, int HK, cudaStream_t st) {
  const int64_t n = n_tok * HK;
  if (n) gate_kernel<<<unsigned((n + 255) / 256), 256, 0, st>>>(f, f_ts, dt_bias, A_log, lb, out, n_tok, HK);
}
void beta_launch(const uint16_t *b, uint16_t *out, int64_t n, cudaStream_t st) {
  if (n) beta_kernel<<<unsigned((n + 255) / 256), 256, 0, st>>>(b, out, n);
}

// ---------------------------------------------------------------------------- o_proj leaf tree (decode glue)
// FP32 adjacent-pair tree over L (a power of two <= 8) leaves [L][n] -- level 0 pairs (0,1), (2,3), ... -- written as the FP32
// root or rounded once to bf16: glm_sm100's leaf_tree_kernel, four elements per thread (16-byte accesses; the same adds on the
// same operands, so the same bits).
__global__ void leaf_tree4_kernel(const float4 *x, int L, int64_t n4, float4 *out_f32, uint2 *out_bf16) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n4; i += int64_t(gridDim.x) * blockDim.x) {
    float4 a[8];
    #pragma unroll
    for (int l = 0; l < 8; ++l) a[l] = l < L ? x[int64_t(l) * n4 + i] : make_float4(0.f, 0.f, 0.f, 0.f);
    for (int m = L; m > 1; m >>= 1)
      #pragma unroll
      for (int k = 0; k < 4; ++k)
        if (k < m / 2) {
          a[k] = make_float4(fadd(a[2 * k].x, a[2 * k + 1].x), fadd(a[2 * k].y, a[2 * k + 1].y),
                             fadd(a[2 * k].z, a[2 * k + 1].z), fadd(a[2 * k].w, a[2 * k + 1].w));
        }
    if (out_f32) out_f32[i] = a[0];
    else out_bf16[i] = make_uint2(uint32_t(rn(a[0].x)) | (uint32_t(rn(a[0].y)) << 16), uint32_t(rn(a[0].z)) | (uint32_t(rn(a[0].w)) << 16));
  }
}
void leaf_tree4_launch(const float *x, int L, int64_t n, float *out_f32, uint16_t *out_bf16, cudaStream_t st) {
  const int64_t n4 = n / 4;
  if (n4) leaf_tree4_kernel<<<unsigned((n4 + 255) / 256 < 65536 ? (n4 + 255) / 256 : 65536), 256, 0, st>>>(
      reinterpret_cast<const float4 *>(x), L, n4, reinterpret_cast<float4 *>(out_f32), reinterpret_cast<uint2 *>(out_bf16));
}

}  // namespace kda_rec
