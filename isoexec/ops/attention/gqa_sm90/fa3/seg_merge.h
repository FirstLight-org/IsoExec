// IsoExec segment split (recipe sm90_fa3_segsplit, gqa_sm90/spec.py): THE merge of two segment results.
// Not part of upstream FlashAttention-3. Included by both places that fold segments -- the attention
// kernel's in-CTA fold (mainloop_fwd_sm90_tma_gmma_ws.hpp, the saturated path) and the combine kernel
// (seg_combine.cu, the split path) -- so the two run one definition, op for op.
//
// A segment result is FA3's finalized pair for one row: O (fp32, already multiplied by FA3's 1/L) and
// lse = M * scale * ln2 + log(L) (natural log, -inf when the segment holds no key of the row).
// The fold over a row's segments 0..n-1 is a RIGHT fold in descending segment order, the order in which
// FA3 walks KV tiles:
//
//     state = seg[n-1];  for s = n-2 .. 0:  state = merge(seg[s], state)
//
// merge(a = lower segment, b = upper segment), per row, then per element:
//     b empty (lse_b == -inf): a, verbatim          a empty: b, verbatim
//     m   = max(lse_a, lse_b)
//     w_a = ex2.approx.ftz((lse_a - m) * log2(e)),  w_b likewise       (__fsub_rn, __fmul_rn)
//     S   = w_a + w_b;  inv = 1 / S (IEEE);  c_a = w_a * inv;  c_b = w_b * inv
//     o   = fma(c_b, o_b, c_a * o_a)                                      (__fmul_rn, then __fmaf_rn)
//     lse = m + lg2.approx.ftz(S) * ln2
// Every operation is an explicit intrinsic or PTX instruction, so neither FA3's --use_fast_math profile
// nor FMA contraction can make the two call sites differ.

#pragma once

#include <cmath>

namespace isoexec_seg {

__device__ __forceinline__ float ex2_approx_ftz(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float lg2_approx_ftz(float x) {
    float y;
    asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

// The per-row part of merge(a, b) for two non-empty results: the two weights and the merged lse.
struct MergeCoef {
    float c_a, c_b, lse;
};

__device__ __forceinline__ MergeCoef merge_coef(float lse_a, float lse_b) {
    constexpr float kLog2e = 1.4426950408889634f;
    constexpr float kLn2 = 0.6931471805599453f;
    float const m = fmaxf(lse_a, lse_b);
    float const w_a = ex2_approx_ftz(__fmul_rn(__fsub_rn(lse_a, m), kLog2e));
    float const w_b = ex2_approx_ftz(__fmul_rn(__fsub_rn(lse_b, m), kLog2e));
    float const S = __fadd_rn(w_a, w_b);
    float const inv = __fdiv_rn(1.f, S);
    return {__fmul_rn(w_a, inv), __fmul_rn(w_b, inv), __fadd_rn(m, __fmul_rn(lg2_approx_ftz(S), kLn2))};
}

// The per-element part: a is the lower segment's element, b the upper's.
__device__ __forceinline__ float merge_elem(MergeCoef const& c, float o_a, float o_b) {
    return __fmaf_rn(c.c_b, o_b, __fmul_rn(c.c_a, o_a));
}

}  // namespace isoexec_seg
