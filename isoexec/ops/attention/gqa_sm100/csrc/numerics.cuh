// SPDX-License-Identifier: BSD-3-Clause
// The scalar graph of the op. Every instruction here is part of the identity: packed-lane rounding
// mode, the exp2 implementation and its per-column selection pattern, the max and row-sum orders,
// the merge operator and the single rounding at the root.
//
// Derived from FlashAttention-4 (flash_attn/cute/{softmax.py,utils.py}) via TileFoundry's CUDA port
// of it. Copyright (c) 2025, Tri Dao et al., BSD-3-Clause; see ../README.md.
#pragma once

#include <cuda_bf16.h>
#include <math_constants.h>

#include "params.h"

namespace gqa_sm100 {

//: Packed fp32 lane pair. The two lanes keep independent round-to-nearest arithmetic; `.ftz` is
//: explicit in every packed instruction and is part of the recipe (the build profile's --ftz=false
//: governs the compiler's scalar choices, not these).
struct FloatPair { float lo, hi; };

__device__ __forceinline__ FloatPair add_pair(FloatPair a, FloatPair b) {
    FloatPair c;
    asm("{ .reg .b64 x,y,z; mov.b64 x,{%2,%3}; mov.b64 y,{%4,%5}; "
        "add.rn.ftz.f32x2 z,x,y; mov.b64 {%0,%1},z; }"
        : "=f"(c.lo), "=f"(c.hi) : "f"(a.lo), "f"(a.hi), "f"(b.lo), "f"(b.hi));
    return c;
}

__device__ __forceinline__ FloatPair sub_pair(FloatPair a, FloatPair b) {
    FloatPair c;
    asm("{ .reg .b64 x,y,z; mov.b64 x,{%2,%3}; mov.b64 y,{%4,%5}; "
        "sub.rn.ftz.f32x2 z,x,y; mov.b64 {%0,%1},z; }"
        : "=f"(c.lo), "=f"(c.hi) : "f"(a.lo), "f"(a.hi), "f"(b.lo), "f"(b.hi));
    return c;
}

__device__ __forceinline__ FloatPair sub_bcast_pair(FloatPair a, float c) {
    FloatPair out;
    asm("{ .reg .b64 x,y,z; mov.b64 x,{%2,%3}; mov.b64 y,{%4,%4}; "
        "sub.rn.ftz.f32x2 z,x,y; mov.b64 {%0,%1},z; }"
        : "=f"(out.lo), "=f"(out.hi) : "f"(a.lo), "f"(a.hi), "f"(c));
    return out;
}

//: p = x * scale + bias, both lanes, one instruction.
__device__ __forceinline__ FloatPair affine_pair(float lo, float hi, float scale, float bias) {
    FloatPair c;
    asm("{ .reg .b64 x,y,z,w; mov.b64 x,{%2,%3}; mov.b64 y,{%4,%4}; "
        "mov.b64 z,{%5,%5}; fma.rn.ftz.f32x2 w,x,y,z; mov.b64 {%0,%1},w; }"
        : "=f"(c.lo), "=f"(c.hi) : "f"(lo), "f"(hi), "f"(scale), "f"(bias));
    return c;
}

__device__ __forceinline__ FloatPair fma_pair(FloatPair a, FloatPair b, float c) {
    FloatPair out;
    asm("{ .reg .b64 x,y,z,w; mov.b64 x,{%2,%3}; mov.b64 y,{%4,%5}; mov.b64 z,{%6,%6}; "
        "fma.rn.ftz.f32x2 w,x,y,z; mov.b64 {%0,%1},w; }"
        : "=f"(out.lo), "=f"(out.hi)
        : "f"(a.lo), "f"(a.hi), "f"(b.lo), "f"(b.hi), "f"(c));
    return out;
}

//: Round toward minus infinity, so the exp2 emulation's fraction stays in [0, 1).
__device__ __forceinline__ FloatPair add_down_pair(FloatPair a, float c) {
    FloatPair out;
    asm("{ .reg .b64 x,y,z; mov.b64 x,{%2,%3}; mov.b64 y,{%4,%4}; "
        "add.rm.ftz.f32x2 z,x,y; mov.b64 {%0,%1},z; }"
        : "=f"(out.lo), "=f"(out.hi) : "f"(a.lo), "f"(a.hi), "f"(c));
    return out;
}

__device__ __forceinline__ void multiply_pair(float& lo, float& hi, float alpha) {
    asm("{ .reg .b64 a,b,c; mov.b64 a,{%2,%3}; mov.b64 b,{%4,%4}; "
        "mul.rn.ftz.f32x2 c,a,b; mov.b64 {%0,%1},c; }"
        : "=f"(lo), "=f"(hi) : "f"(lo), "f"(hi), "f"(alpha));
}

//: PTX puts its FIRST input in the HIGH half, so the operands are swapped here on purpose.
__device__ __forceinline__ uint32_t pack_bf16_pair(float lo, float hi) {
    uint32_t packed;
    asm("cvt.rn.bf16x2.f32 %0,%1,%2;" : "=r"(packed) : "f"(hi), "f"(lo));
    return packed;
}

//: The hardware exp2. One of the op's two declared exp2 implementations.
__device__ __forceinline__ float exp2_native(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0,%1;" : "=f"(y) : "f"(x));
    return y;
}

//: Fold the integer part back in: shift it into the exponent field and add. `add.s32` keeps the
//: operation on the ALU pipe (FA4's combine_int_frac_ex2).
__device__ __forceinline__ float combine_exponent(float rounded, float fraction) {
    float out;
    asm("{ .reg .b32 a,b,c; mov.b32 a,%1; mov.b32 b,%2; shl.b32 c,a,23;"
        " add.s32 c,c,b; mov.b32 %0,c; }"
        : "=f"(out) : "f"(rounded), "f"(fraction));
    return out;
}

//: The op's second declared exp2 implementation: FA4's degree-three polynomial emulation
//: (`ex2_emulation_2`), exp2(x) = 2^floor(x) * poly3(x - floor(x)), on packed lanes. It moves work
//: off the quarter-rate SFU and onto the fp32 FMA pipe. Coefficients retain FA4's bits.
__device__ __forceinline__ FloatPair polynomial_exp2(float lo, float hi) {
    constexpr float magic = 12582912.0f;   // 2^23 + 2^22
    const FloatPair clamped = {fmaxf(lo, -127.0f), fmaxf(hi, -127.0f)};
    const FloatPair rounded = add_down_pair(clamped, magic);
    const FloatPair fraction = sub_pair(clamped, sub_bcast_pair(rounded, magic));
    FloatPair value = {0.077119089663028717041015625f, 0.077119089663028717041015625f};
    value = fma_pair(value, fraction, 0.227564394474029541015625f);
    value = fma_pair(value, fraction, 0.695146143436431884765625f);
    value = fma_pair(value, fraction, 1.0f);
    return {combine_exponent(rounded.lo, value.lo), combine_exponent(rounded.hi, value.hi)};
}

//: WHICH of the two exp2 implementations serves column pair `pair` of a 128-key tile. The pattern is
//: a function of the column index inside the fixed KV tile and of nothing else, so it is
//: deployment-fixed. FA4's rule (softmax.py apply_exp2_convert) at freq = 8, res = 4, start = 1:
//: emulate when `k % freq >= freq - res` inside fragments 1 and 2 of four. 32 of 128 columns.
//: (FA4's own HD128 tuning entry is freq = 16; freq = 8 measured faster at this tile shape.)
__host__ __device__ __forceinline__ constexpr bool emulated_exp2(int pair) {
    const int fragment = pair / 16, k = 2 * (pair % 16);
    return fragment >= 1 && fragment <= 2 && k % 8 >= 8 - 4;
}

// --- the merge operator and the root -------------------------------------------------------------
//
// State = (O[D] fp32, L fp32, M fp32). M is the segment's EXACT running score maximum in the RAW
// (unscaled) frame, exactly as the leaf's running max holds it; the scale enters the merge the same
// way it enters the leaf's own rescale, as a multiply of the max difference. L is the segment's
// unnormalized denominator. Neither is ever normalized before the merge.
//
//   merge(a, b)   with a the LOWER segment index, always the left operand:
//     M  = fmaxf(a.M, b.M)
//     sa = ex2.approx.ftz.f32((a.M - M) * scale_log2)
//     sb = ex2.approx.ftz.f32((b.M - M) * scale_log2)
//     O  = a.O * sa + b.O * sb          per element, left term first, products rounded separately
//     L  = a.L * sa + b.L * sb
//   root(s):
//     L' = (s.L == 0 || s.L != s.L) ? 1 : s.L
//     inv = 1 / L'                       one IEEE divide per row
//     out = (bf16)(s.O * inv)            the one rounding
//     lse = (s.M * scale_log2 + log2(L')) * ln2,  or -inf when L' was the sentinel

struct MergeScales { float sa, sb, m; };

__device__ __forceinline__ MergeScales merge_scales(float ma, float mb, float scale_log2) {
    MergeScales s;
    s.m = fmaxf(ma, mb);
    s.sa = exp2_native(__fmul_rn(__fsub_rn(ma, s.m), scale_log2));
    s.sb = exp2_native(__fmul_rn(__fsub_rn(mb, s.m), scale_log2));
    return s;
}

__device__ __forceinline__ float merge_l(float la, float lb, float sa, float sb) {
    return __fadd_rn(__fmul_rn(la, sa), __fmul_rn(lb, sb));
}

//: O_a <- O_a * sa + O_b * sb on a packed lane pair, left term first.
__device__ __forceinline__ void merge_o_pair(float& alo, float& ahi, float blo, float bhi,
                                            float sa, float sb) {
    FloatPair a = {alo, ahi};
    FloatPair b = {blo, bhi};
    multiply_pair(a.lo, a.hi, sa);
    multiply_pair(b.lo, b.hi, sb);
    const FloatPair r = add_pair(a, b);
    alo = r.lo;
    ahi = r.hi;
}

//: `1 / L'`, the reciprocal the root multiplies by. One IEEE round-to-nearest divide per row.
__device__ __forceinline__ float root_inverse(float l) {
    const float safe = (l == 0.0f || l != l) ? 1.0f : l;
    return __fdiv_rn(1.0f, safe);
}

__device__ __forceinline__ float root_lse(float l, float m, float scale_log2) {
    if (l == 0.0f || l != l) return -CUDART_INF_F;
    return __fmul_rn(__fadd_rn(__fmul_rn(m, scale_log2), log2f(l)), 0.693147180559945309417f);
}

// --- shared / TMEM plumbing ----------------------------------------------------------------------

__device__ __forceinline__ uint32_t shared_address(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

//: FA4 passes a suspendTimeHint on every try_wait, so ptxas emits NANOSLEEP.SYNCS and a blocked warp
//: is suspended by hardware instead of hot-spinning.
__device__ __forceinline__ void bar_wait(uint64_t* barrier, int phase) {
    asm volatile("{ .reg .pred p; WAITLBL: mbarrier.try_wait.parity.shared::cta.b64 "
                 "p,[%0],%1,10000000; @!p bra WAITLBL; }"
                 :: "r"(shared_address(barrier)), "r"(phase) : "memory");
}

__device__ __forceinline__ void arrive_local(uint64_t* barrier) {
    asm volatile("mbarrier.arrive.relaxed.cta.shared::cta.b64 _,[%0];"
                 :: "r"(shared_address(barrier)) : "memory");
}

//: One elected lane of a fully populated warp arrives.
__device__ __forceinline__ void arrive_warp_local(uint64_t* barrier) {
    asm volatile("{ .reg .pred e; elect.sync _|e,0xffffffff; "
                 "@e mbarrier.arrive.relaxed.cta.shared::cta.b64 _,[%0]; }"
                 :: "r"(shared_address(barrier)) : "memory");
}

//: A RELEASE arrive. `values_ready` is armed after the publisher warp has written zeros into the V
//: buffer through the generic proxy; a relaxed arrive does not order those stores before the arrive as
//: the acquiring MMA thread observes them.
__device__ __forceinline__ void arrive_local_release(uint64_t* barrier) {
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _,[%0];"
                 :: "r"(shared_address(barrier)) : "memory");
}

//: Per-warpgroup register budget, FA4's mechanism (`flash_fwd_sm100.py:107`), which also works from
//: CUDA C++. ptxas allocates
//: ONE static count for the whole kernel -- the sub-partition file divided by the warps that land on
//: it -- but it honours `setmaxnreg` and will allocate a branch that raises its budget past that
//: count. The budgets of the warps resident in one sub-partition must sum to at most 512 and the
//: increases and the decreases must balance exactly, or the increase waits for registers no warp will
//: release. Execution only: a register budget cannot move a bit.
template <int N> __device__ __forceinline__ void registers_increase() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(N));
}
template <int N> __device__ __forceinline__ void registers_decrease() {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" :: "n"(N));
}

__device__ __forceinline__ void commit_one(uint64_t* barrier) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                 :: "r"(shared_address(barrier)) : "memory");
}

__device__ __forceinline__ void synchronize_cta() {
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

//: UMMA shared-operand descriptor: 128 B swizzle, 1024 B eight-row stride. `mn` selects the
//: MN-major PV B operand (V), whose 64-feature chunk is 128 keys x 128 B.
__device__ __forceinline__ uint64_t descriptor(const void* p, bool mn) {
    return uint64_t(shared_address(p) >> 4) | (uint64_t(mn ? 1024 : 1) << 16) |
           (uint64_t(64) << 32) | (uint64_t(1) << 46) | (uint64_t(2) << 61);
}

}  // namespace gqa_sm100
