// SPDX-License-Identifier: MIT
// IsoExec deepseek_v41_cache_sm100: the scalar graph of every row operation of the V4.1 attention
// composition (recipe deepseek_v41_attn_rows_v1).  Every rounding point
// is an explicit RN intrinsic; the translation unit is built with --fmad=false --ftz=false.
//
// Row layout (fixed, part of the recipe): one warp owns one row of D elements, lane l owns the E = D/32
// CONTIGUOUS elements [l*E, (l+1)*E).  Reductions over the row are (a) a sequential ascending chain over
// the lane's E elements, then (b) the XOR butterfly 16, 8, 4, 2, 1 -- i.e. a balanced tree over lanes
// whose first level pairs (l, l+16).  reference.py emulates exactly this tree.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <stdint.h>

namespace v41 {

// The V4.1 cache program (CACHE-SPEC: vLLM 0.33 FLASHMLA_SPARSE_DSV41 records). Every kind is FP8 E4M3 with a
// power-of-two (UE8M0) scale; they differ in the group width and in how the exponent is formed:
//  * Q_FP8_B32_UE8M0      window KV, 32 per scale, BF16 input: e = exact ceil(log2(max(amax,1e-4) * fl(1/448)))
//  * Q_FP8_B32_UE8M0_F32  compressed KV, 32 per scale, FP32 input (the RoPE output is not rounded to BF16):
//                         e = ceilf(log2f(fl(max(amax,1e-4) * fl(1/448)))) with libdevice log2f, as vLLM's Triton
//                         writer; differs from the exact ceil only when that product is within a few ulps above a
//                         power of two, which a BF16 input can never be
//  * Q_FP8_B128_UE8M0     index keys (and index queries), one scale per 128, BF16 input, exact ceil; the record
//                         carries the exponent as four equal UE8M0 bytes
enum QuantKind : int { Q_NONE = 0, Q_FP8_B32_UE8M0 = 1, Q_FP8_B32_UE8M0_F32 = 2, Q_FP8_B128_UE8M0 = 3 };

__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

// ---- RMSNorm (DeepSeek convention: y = bf16((x * inv) * w), inv = 1 / sqrt(mean(x^2) + eps)) --------
template <int E>
__device__ __forceinline__ float row_sum_squares(const float (&v)[E]) {
    float s = __fmul_rn(v[0], v[0]);
#pragma unroll
    for (int i = 1; i < E; ++i) s = __fadd_rn(s, __fmul_rn(v[i], v[i]));
#pragma unroll
    for (int d = 16; d; d >>= 1) s = __fadd_rn(s, __shfl_xor_sync(0xffffffffu, s, d));
    return s;
}

template <int D>
__device__ __forceinline__ float rms_inv(float ss, float eps) {
    return __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(ss, float(D)), eps)));
}

// ---- adjacent-pair RoPE (x[2j], x[2j+1]) as a complex number times (c + i s) ------------------------
__device__ __forceinline__ void rope_pair(float &a, float &b, float c, float s, bool inverse) {
    if (!inverse) {
        const float re = __fsub_rn(__fmul_rn(a, c), __fmul_rn(b, s));
        const float im = __fadd_rn(__fmul_rn(a, s), __fmul_rn(b, c));
        a = re; b = im;
    } else {  // multiply by the conjugate (c - i s)
        const float re = __fadd_rn(__fmul_rn(a, c), __fmul_rn(b, s));
        const float im = __fsub_rn(__fmul_rn(b, c), __fmul_rn(a, s));
        a = re; b = im;
    }
}

// ---- the V4.1 cache-record quantizer (port/CACHE-SPEC.md sections 1-3 and 6) ---------------------------
template <int Q> struct QuantTraits;
template <> struct QuantTraits<Q_NONE>            { static constexpr int G = 1;   static constexpr bool F32_IN = false; };
template <> struct QuantTraits<Q_FP8_B32_UE8M0>     { static constexpr int G = 32;  static constexpr bool F32_IN = false; };
template <> struct QuantTraits<Q_FP8_B32_UE8M0_F32> { static constexpr int G = 32;  static constexpr bool F32_IN = true; };
template <> struct QuantTraits<Q_FP8_B128_UE8M0>    { static constexpr int G = 128; static constexpr bool F32_IN = false; };

// Returns the power-of-two scale 2^e; scale_byte = e + 127 (the UE8M0 byte). amax ignores NaN (fmaxf); an Inf amax
// gives byte 255 and scale +Inf, as vLLM's writers.
template <int Q>
__device__ __forceinline__ float quant_scale(float amax, uint8_t &scale_byte) {
    amax = fmaxf(amax, 1.0e-4f);
    const float r = __fmul_rn(amax, 1.0f / 448.0f);
    uint32_t exponent;
    if constexpr (QuantTraits<Q>::F32_IN) {
        const float e = fminf(fmaxf(__fadd_rn(ceilf(log2f(r)), 127.0f), 0.0f), 255.0f);
        exponent = uint32_t(e);
    } else {
        const uint32_t b = __float_as_uint(r);
        exponent = ((b >> 23) & 255) + ((b & 0x7fffff) != 0);
    }
    scale_byte = (uint8_t)exponent;
    return __uint_as_float(exponent << 23);
}

// The E4M3 code of x at this scale (round to nearest even, saturating; NaN -> -448 as vLLM's clamp).
__device__ __forceinline__ uint8_t quant_code(float x, float scale) {
    const float value = fminf(448.0f, fmaxf(-448.0f, __fdiv_rn(x, scale)));
    return (uint8_t)__nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
}

// The decoded code value (unscaled); the dequantized element is bf16(decoded * scale), exact for every reachable scale.
__device__ __forceinline__ float quant_decode(float x, float scale) {
    return __half2float(__nv_cvt_fp8_to_halfraw(quant_code(x, scale), __NV_E4M3));
}

// ---- deepseek_channel_softmax_pool_fp32_v1 (attention/channel_pool_sm100/csrc/pool.cu), verbatim ----
// One channel over `ratio` tokens visited in ascending order.
__device__ __forceinline__ float pool_channel(const float *v, const float *s, int64_t stride, int ratio) {
    float maximum = -INFINITY;
    for (int j = 0; j < ratio; ++j) maximum = fmaxf(maximum, s[j * stride]);
    float denominator = 0.f;
    for (int j = 0; j < ratio; ++j) denominator += expf(s[j * stride] - maximum);
    float result = 0.f;
    for (int j = 0; j < ratio; ++j) {
        const float p = expf(s[j * stride] - maximum) / denominator;
        result += p * v[j * stride];
    }
    return result;
}

}  // namespace v41
