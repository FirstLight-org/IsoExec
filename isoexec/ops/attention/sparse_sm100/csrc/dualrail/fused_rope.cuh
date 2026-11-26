// SPDX-License-Identifier: MIT
// IsoExec sparse_sm100: the RoPE steps the dual-rail leaf can run in place of two separate row launches (IsoExec code).
//
//   * Q RoPE (prologue): the query's last 64 dims (GPT-J adjacent pairs) rotated at the row's position in the Q tile's
//     shared memory before it is copied to TMEM -- the scalar graph of deepseek_v41_cache_sm100 `rope_tail_inplace`
//     (rows.cu): a = float(q[2j]), b = float(q[2j+1]); re = a*c - b*s, im = a*s + b*c, each product and sum rounded
//     separately; bf16 RNE of both.
//   * inverse RoPE (epilogue): after the leaf's bf16 rounding of the output, the conjugate rotation of the last 64
//     dims from those bf16 values, bf16 again -- the scalar graph of the row kernel ROW_OUT_INV_ROPE (rows.cu
//     row_body, ROPE == 2): re = a*c + b*s, im = b*c - a*s.
// rows.cu is built --ftz=false --fmad=false; this extension is built --ftz=true, so the arithmetic is written as
// explicit non-FTZ PTX (mul.rn / add.rn / sub.rn .f32): the same instructions, so the same bits on subnormals too.
#pragma once

#include <cuda_bf16.h>
#include <cstdint>

namespace sparse_sm100::dualrail {

struct RopeParams {
    const float *cosv = nullptr, *sinv = nullptr;   // [positions, 32] fp32 (the layer's table, ops.rope_table)
    const int *pos = nullptr;                       // [rows] int32 absolute position of each query row
    int q = 0, o = 0;                               // apply the Q RoPE / the output's inverse RoPE
};

__device__ __forceinline__ float rope_mul(float a, float b) { float r; asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float rope_add(float a, float b) { float r; asm("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float rope_sub(float a, float b) { float r; asm("sub.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }

// rows.cu rope_pair, operation for operation
__device__ __forceinline__ __nv_bfloat162 rope_pair_bf16(__nv_bfloat162 x, float c, float s, bool inverse) {
    const float a = __bfloat162float(x.x), b = __bfloat162float(x.y);
    float re, im;
    if (!inverse) {
        re = rope_sub(rope_mul(a, c), rope_mul(b, s));
        im = rope_add(rope_mul(a, s), rope_mul(b, c));
    } else {
        re = rope_add(rope_mul(a, c), rope_mul(b, s));
        im = rope_sub(rope_mul(b, c), rope_mul(a, s));
    }
    return __floats2bfloat162_rn(re, im);
}

// Q RoPE of a Q tile in shared memory: one warp, lane l rotates heads l and l + 32 (< h_q) over the 32 tail pairs.
// sQ: a cute tensor (heads, 512) over the swizzled tile (element access resolves the swizzle).
template<typename SQ>
__device__ __forceinline__ void rope_q_tile(SQ &sQ, int h_q, const RopeParams &rp, int row, int lane) {
    const int64_t position = __ldg(rp.pos + row);
    const float *c = rp.cosv + position * 32, *s = rp.sinv + position * 32;
    for (int h = lane; h < h_q; h += 32) {
#pragma unroll 4
        for (int j = 0; j < 32; ++j) {
            // an adjacent pair never straddles a 16-byte swizzle chunk: one 4-byte word
            __nv_bfloat162 *w = reinterpret_cast<__nv_bfloat162 *>(&sQ(h, 448 + 2 * j));
            *w = rope_pair_bf16(*w, __ldg(c + j), __ldg(s + j), false);
        }
    }
}

// Inverse RoPE of one thread's bf16 output chunk: columns [col0, col0 + 2 * N2) of one head row, after the leaf's bf16
// rounding. Pairs in the tail [448, 512) are rotated; the condition is uniform over a warp (all its lanes share col0).
template<int N2>
__device__ __forceinline__ void rope_o_chunk(__nv_bfloat162 (&o)[N2], int col0, const RopeParams &rp, int row) {
    if (col0 + 2 * N2 <= 448) return;
    const int64_t position = __ldg(rp.pos + row);
    const float *c = rp.cosv + position * 32, *s = rp.sinv + position * 32;
#pragma unroll
    for (int j = 0; j < N2; ++j) {
        const int col = col0 + 2 * j;
        if (col >= 448) {
            const int k = (col - 448) / 2;
            o[j] = rope_pair_bf16(o[j], __ldg(c + k), __ldg(s + k), true);
        }
    }
}

}  // namespace sparse_sm100::dualrail
