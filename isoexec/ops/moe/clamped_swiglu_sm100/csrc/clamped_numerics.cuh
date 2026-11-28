// SPDX-License-Identifier: Apache-2.0
//
// Scalar programs of the GLM clamped FP8 recipes (README "Numerical program"), written once and
// called from every kernel that evaluates them, so that fused and unfused evaluations are one
// program under the shared --ftz=false --fmad=false profile.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace glmfp8 {

// Packing helper for the f32x2 instructions (two independent IEEE operations, lane for lane the
// scalar program).
__device__ __forceinline__ unsigned long long pack2(float a, float b) {
  return (unsigned long long)__float_as_uint(a) | ((unsigned long long)__float_as_uint(b) << 32);
}

// ---------------------------------------------------------------------------------------------
// Clamped SwiGLU (recipe glm_clamped_swiglu_fp32_bf16_v1). NaN propagates exactly as
// torch.clamp_max / torch.clamp do; the boundary value itself is returned unchanged (g == L -> L).
//   g' = g > L ? L : g
//   u' = u > L ? L : (u < -L ? -L : u)
//   s  = g' / (1 + expf(-g'))            (ATen's silu formula: CUDA expf, IEEE division)
//   h  = RNE_bf16(s * u')                 (one fp32 product, one bf16 rounding)
// With a pre-down route weight w (the DeepSeek placement, a recipe parameter):
//   h  = RNE_bf16((s * u') * w)
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ float clamp_gate(float g, float limit) { return g > limit ? limit : g; }
__device__ __forceinline__ float clamp_up(float u, float limit) {
  return u > limit ? limit : (u < -limit ? -limit : u);
}
__device__ __forceinline__ float silu(float g) {
  const float d = __fadd_rn(1.0f, expf(-g));
  return __fdiv_rn(g, d);
}
__device__ __forceinline__ float clamped_swiglu_f32(float g, float u, float limit) {
  return __fmul_rn(silu(clamp_gate(g, limit)), clamp_up(u, limit));
}

// ---------------------------------------------------------------------------------------------
// Training recipe v2: the FC1 projection saved for the backward as BF16 with the forward's clamp
// decision encoded losslessly (glm_clamped_fp8_master_dense_surrogate_v2):
//   gate: g > L (fp32) -> +inf; else v = RNE_bf16(g), pulled down to RD_bf16(L) if it rounded past L
//   up:   u > L -> +inf, u < -L -> -inf; else RNE_bf16(u), pulled back inside [-RD(L), RD(L)]
// so on the stored value v: (v <= L) == (g <= L) and (-L <= v <= L) == (-L <= u <= L) exactly, and
// clamp_gate / clamp_up(v) give the clamped value (L, +-L) for the clamped elements. NaN stays NaN.
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ __nv_bfloat16 raw_gate_bf16(float g, float limit) {
  if (g > limit) return __float2bfloat16_rn(__int_as_float(0x7f800000));
  __nv_bfloat16 v = __float2bfloat16_rn(g);
  if (__bfloat162float(v) > limit) v = __float2bfloat16_rd(limit);
  return v;
}
__device__ __forceinline__ __nv_bfloat16 raw_up_bf16(float u, float limit) {
  if (u > limit) return __float2bfloat16_rn(__int_as_float(0x7f800000));
  if (u < -limit) return __float2bfloat16_rn(__int_as_float(0xff800000));
  __nv_bfloat16 v = __float2bfloat16_rn(u);
  const float f = __bfloat162float(v);
  if (f > limit) v = __float2bfloat16_rd(limit);
  if (f < -limit) v = __hneg(__float2bfloat16_rd(limit));
  return v;
}

// ---------------------------------------------------------------------------------------------
// Activation quantizer: deepseek_v41_act_quant_v1, E4M3 codes with an FP32 per-row 1x128 scale
// (ops/quant/checkpoint_sm100/csrc/quant.cu, SF == 0), applied to BF16 values:
//   amax  = max_i |x_i|                     (exact; finite domain)
//   scale = RN(max(amax, 1e-4) * RN(1/448))
//   code  = e4m3_satfinite_rne(min(448, max(-448, x / scale)))   (IEEE division)
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ float act_scale(float amax) {
  return __fmul_rn(fmaxf(amax, 1.0e-4f), 1.0f / 448.0f);
}
__device__ __forceinline__ uint8_t act_code(float x, float scale) {
  const float v = fminf(448.0f, fmaxf(-448.0f, __fdiv_rn(x, scale)));
  return static_cast<uint8_t>(__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3));
}

// bf16 round trip as a float (the value the quantizer reads).
__device__ __forceinline__ float bf16_round(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

}  // namespace glmfp8
