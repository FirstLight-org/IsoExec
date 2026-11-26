// SPDX-License-Identifier: Apache-2.0
// GLM Kimi Delta Attention on SM100 -- numerical recipe `sm100_kda_rec_v1`, version 1 ("recurrent everywhere").
//
// THIS HEADER IS THE NUMERICAL CONTRACT. Prefill, decode and the trainer forward evaluate the SAME per-token
// update below, T times in order for T tokens. Built with the shared SM100 profile (sm_100a, --fmad=false,
// --ftz=false: build_cache NUMERICS_FLAGS). Everything not written here (which warp/CTA owns which value
// columns, prefetching, launch shape) is execution and never enters the bits.
//
// Geometry: K = V = 128 per head. State S[v][k] (V-first, k contiguous: vLLM's native GLM KDA state layout),
// fp32, +0 for a fresh sequence.
// Notation: (+) (-) (*) RN fp32; fma = one RN fused multiply-add; rn() RN fp32->bf16; bf() exact widening;
// E(x) = ex2.approx.f32; rsqrt_rn = __frsqrt_rn.
//
// R128(a, b)  (the ONLY reduction form; every dot product of the recipe)
//   channel c = 32 m + 4 s + e, m in 0..3, s in 0..7, e in 0..3.
//   P(m,s) = ascending-e fma chain seeded +0 of fma(a_c, b_c, acc)                  (4 channels = 32 partials)
//   Q_s    = (P(0,s) (+) P(1,s)) (+) (P(2,s) (+) P(3,s))
//   R128   = ((Q_0 (+) Q_1) (+) (Q_2 (+) Q_3)) (+) ((Q_4 (+) Q_5) (+) (Q_6 (+) Q_7))
//   Depends only on the deployment-fixed head dim 128; never on batch, T, occupancy or CTA shape. No atomics.
//   EXECUTION MAPPINGS (bit-identical by construction): 8 / 16 / 32 lanes per value column. A lane owns the
//   partials of its (m-block, s): with L lanes per column, lane = 8 mh + s, mh in 0..L/8-1 owns m in
//   [mh 32/L, (mh + 1) 32/L); it adds its own partials locally in the Q_s pairing and the cross-lane steps
//   are xor-shuffles 8, 16 (m pairs) then 1, 2, 4 (adjacent s pairs). fadd is commutative, so every lane
//   ends with the root's bits.

// Token update (per head; q, k, v bf16 post-convolution; g fp32 log decay; beta bf16 post-sigmoid):
//   k^_c = bf(k_c) (*) rsqrt_rn(R128(k, k) (+) 1e-6)                (fp32, not rounded to bf16)
//   q^_c = bf(q_c) (*) (rsqrt_rn(R128(q, q) (+) 1e-6) (*) 128^-1/2)
//   a_c  = E(g_c (*) LOG2E)
//   per value column j (columns never meet, so how many columns a CTA / thread owns is execution):
//     A_jc    = S_jc (*) a_c
//     kS_j    = R128(A_j., k^)                      [fma(A_jc, k^_c, acc)]
//     delta_j = bf(beta) (*) (bf(v_j) (-) kS_j)
//     S'_jc   = fma(delta_j, k^_c, A_jc)
//     o_j     = R128(S'_j., q^)                     output rn(o_j) bf16
//   S := S'.
// Sequence of T tokens = this update T times in order. A fresh sequence's state is +0. Prefill, decode and the
// trainer forward evaluate exactly this; there is no other path.
// The fused-gate form evaluates g_c = lb (*) sigmoid(expf(A_log_h) (*) (bf(f_c) (+) dt_bias_c)) and
// beta = rn(sigmoid(bf(b))), sigmoid(y) = 1 (/) (1 (+) expf(-y)): the glm_sm100 layer.cu gate_fwd / beta_fwd
// definitions, so it equals the unfused ops bit for bit.
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace kda_rec {

constexpr int D = 128;
constexpr float QSCALE = 0.08838834764831845f;   // 128^-1/2 rounded to fp32
constexpr float LOG2E = 1.4426950408889634f;

__device__ __forceinline__ float bf(uint32_t bits16) { return __uint_as_float(bits16 << 16); }
__device__ __forceinline__ uint16_t rn(float x) { return __bfloat16_as_ushort(__float2bfloat16_rn(x)); }
__device__ __forceinline__ float fadd(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ float fsub(float a, float b) { return __fsub_rn(a, b); }
__device__ __forceinline__ float fmul(float a, float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ float ffma(float a, float b, float c) { return __fmaf_rn(a, b, c); }
__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float sigm(float y) { return __fdiv_rn(1.f, fadd(1.f, expf(-y))); }

// The s tree over the 8 lanes (lane & 7 == s); every lane ends with the root's bits.
__device__ __forceinline__ float tree8(float p) {
  p = fadd(p, __shfl_xor_sync(0xffffffffu, p, 1));
  p = fadd(p, __shfl_xor_sync(0xffffffffu, p, 2));
  p = fadd(p, __shfl_xor_sync(0xffffffffu, p, 4));
  return p;
}

}  // namespace kda_rec
