// SPDX-License-Identifier: BSD-3-Clause
// tcgen05 access and MMA issue for the HD128 leaf. Execution only.
// Derived from TileFoundry's CUDA port of FlashAttention-4. Copyright (c) 2025, Tri Dao et al.,
// BSD-3-Clause.
#pragma once

#include <cuda.h>
#include <cstdint>

#include "numerics.cuh"

namespace gqa_sm100 {

//: The warp's TMEM lane base sits in the high half of the address.
#define GQA_LANE_BASE (((threadIdx.x / 32) % 4) * 32) << 16

__device__ __forceinline__ void store16_bits(uint32_t address, const uint32_t* v) {
    address += GQA_LANE_BASE;
    asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32 "
                 "[%0],{%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};"
                 :: "r"(address),"r"(v[0]),"r"(v[1]),"r"(v[2]),"r"(v[3]),
                    "r"(v[4]),"r"(v[5]),"r"(v[6]),"r"(v[7]),
                    "r"(v[8]),"r"(v[9]),"r"(v[10]),"r"(v[11]),
                    "r"(v[12]),"r"(v[13]),"r"(v[14]),"r"(v[15]) : "memory");
}

__device__ __forceinline__ void load32(uint32_t address, float (&v)[32]) {
    address += GQA_LANE_BASE;
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,"
                 "%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},[%32];"
                 : "=f"(v[0]),"=f"(v[1]),"=f"(v[2]),"=f"(v[3]),"=f"(v[4]),"=f"(v[5]),"=f"(v[6]),
                   "=f"(v[7]),"=f"(v[8]),"=f"(v[9]),"=f"(v[10]),"=f"(v[11]),"=f"(v[12]),"=f"(v[13]),
                   "=f"(v[14]),"=f"(v[15]),"=f"(v[16]),"=f"(v[17]),"=f"(v[18]),"=f"(v[19]),
                   "=f"(v[20]),"=f"(v[21]),"=f"(v[22]),"=f"(v[23]),"=f"(v[24]),"=f"(v[25]),
                   "=f"(v[26]),"=f"(v[27]),"=f"(v[28]),"=f"(v[29]),"=f"(v[30]),"=f"(v[31])
                 : "r"(address) : "memory");
}

__device__ __forceinline__ void load16(uint32_t address, float (&v)[16]) {
    address += GQA_LANE_BASE;
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15},[%16];"
                 : "=f"(v[0]),"=f"(v[1]),"=f"(v[2]),"=f"(v[3]),
                   "=f"(v[4]),"=f"(v[5]),"=f"(v[6]),"=f"(v[7]),
                   "=f"(v[8]),"=f"(v[9]),"=f"(v[10]),"=f"(v[11]),
                   "=f"(v[12]),"=f"(v[13]),"=f"(v[14]),"=f"(v[15])
                 : "r"(address) : "memory");
}


__device__ __forceinline__ void load8(uint32_t address, float (&v)[8]) {
    address += GQA_LANE_BASE;
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7},[%8];"
                 : "=f"(v[0]),"=f"(v[1]),"=f"(v[2]),"=f"(v[3]),
                   "=f"(v[4]),"=f"(v[5]),"=f"(v[6]),"=f"(v[7])
                 : "r"(address) : "memory");
}

__device__ __forceinline__ void store32(uint32_t address, const float (&v)[32]) {
    address += GQA_LANE_BASE;
    asm volatile("tcgen05.st.sync.aligned.32x32b.x32.b32 "
                 "[%0],{%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,"
                 "%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32};"
                 :: "r"(address),"f"(v[0]),"f"(v[1]),"f"(v[2]),"f"(v[3]),"f"(v[4]),"f"(v[5]),
                    "f"(v[6]),"f"(v[7]),"f"(v[8]),"f"(v[9]),"f"(v[10]),"f"(v[11]),"f"(v[12]),
                    "f"(v[13]),"f"(v[14]),"f"(v[15]),"f"(v[16]),"f"(v[17]),"f"(v[18]),"f"(v[19]),
                    "f"(v[20]),"f"(v[21]),"f"(v[22]),"f"(v[23]),"f"(v[24]),"f"(v[25]),"f"(v[26]),
                    "f"(v[27]),"f"(v[28]),"f"(v[29]),"f"(v[30]),"f"(v[31]) : "memory");
}

__device__ __forceinline__ void store16(uint32_t address, const float (&v)[16]) {
    address += GQA_LANE_BASE;
    asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32 "
                 "[%0],{%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};"
                 :: "r"(address),"f"(v[0]),"f"(v[1]),"f"(v[2]),"f"(v[3]),
                    "f"(v[4]),"f"(v[5]),"f"(v[6]),"f"(v[7]),
                    "f"(v[8]),"f"(v[9]),"f"(v[10]),"f"(v[11]),
                    "f"(v[12]),"f"(v[13]),"f"(v[14]),"f"(v[15]) : "memory");
}

//: The STATS LANE of a score slot: columns 64-66 hold this tile's `alpha` (64) and, at a
//: segment's last tile, the segment's `(L, M)` (65-66), written by the softmax warp that owns the
//: row and read by the correction warp that owns the same row -- FA4 hd256's `tmem_vec` form. They
//: are free from the moment the softmax has read S into registers until the next QK into that slot,
//: and the barrier protocol issues that QK only after the correction has read them.
__device__ __forceinline__ void store1(uint32_t address, float a) {
    address += GQA_LANE_BASE;
    asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0],{%1};"
                 :: "r"(address), "f"(a) : "memory");
}
__device__ __forceinline__ void store2(uint32_t address, float a, float b) {
    address += GQA_LANE_BASE;
    asm volatile("tcgen05.st.sync.aligned.32x32b.x2.b32 [%0],{%1,%2};"
                 :: "r"(address), "f"(a), "f"(b) : "memory");
}
__device__ __forceinline__ float load1(uint32_t address) {
    address += GQA_LANE_BASE;
    float a;
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0},[%1];"
                 : "=f"(a) : "r"(address) : "memory");
    return a;
}
__device__ __forceinline__ void load2(uint32_t address, float& a, float& b) {
    address += GQA_LANE_BASE;
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x2.b32 {%0,%1},[%2];"
                 : "=f"(a), "=f"(b) : "r"(address) : "memory");
}

//: Chunk-width-generic TMEM row access, so the correction warpgroup's merge/spill/rescale can run at
//: 16 columns under the two-wave 72-register budget and at 32 under the one-wave 160.
template <int C> __device__ __forceinline__ void tload(uint32_t address, float (&v)[C]) {
    if constexpr (C == 32) load32(address, v); else load16(address, v);
}
template <int C> __device__ __forceinline__ void tstore(uint32_t address, const float (&v)[C]) {
    if constexpr (C == 32) store32(address, v); else store16(address, v);
}

//: Rescale the 128-column fp32 output accumulator by the softmax correction factor: FOUR sequential
//: 32-column read-modify-writes. The same 128 columns and the same 64 packed multiplies in the same
//: order as FA4's eight-x16 `correction_rescale`, with half the TMEM instructions and half the
//: read-after-load fences -- the scores are dead by the time this runs, so the 64 registers a 32-wide
//: chunk needs are free.
template <int C = 32>
__device__ __forceinline__ void correction128(uint32_t base, float alpha) {
    //: `O *= alpha` ONLY when `alpha != 1`, which is what README.md section 1.4 step 7 states and what
    //: makes the warp ballot bit-neutral. It is not cosmetic: `mul.rn.ftz.f32x2` states `.ftz`, so
    //: multiplying by exactly 1.0 FLUSHES a denormal accumulator element to zero. A lane that kept its
    //: running max has `alpha == 1` and must not be touched -- the ballot makes the whole warp run this
    //: function whenever ANY of its lanes advanced its max, and which query rows share a warp depends
    //: on `q_len mod tile_tokens`, that is on the chunking. Skipping the multiply per lane makes the
    //: ballot bit-neutral unconditionally instead of only on accumulators that happen to hold no
    //: denormal. The tcgen05 store is warp-wide and cannot be skipped per lane, so a kept lane writes
    //: back exactly the bits it loaded.
    const bool scale = alpha != 1.0f;
    #pragma unroll 1
    for (int chunk = 0; chunk < 128 / C; ++chunk) {
        float values[C];
        tload<C>(base + chunk * C, values);
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        if (scale) {
            #pragma unroll
            for (int j = 0; j < C; j += 2) multiply_pair(values[j], values[j + 1], alpha);
        }
        tstore<C>(base + chunk * C, values);
    }
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}

//: One K=128 QK update as eight K16 tcgen05 MMAs issued by one elected lane. Q is 128 rows x 128
//: features in two 64-feature chunks (16384 B apart, +1024 descriptor units); K likewise. These are
//: FA4's own eight B offsets for its HD128 QK run.
__device__ __forceinline__ void qk_sequence128(uint32_t dst, uint64_t q, uint64_t k,
                                               uint32_t instruction) {
    asm volatile(
        "{ .reg .b64 a,b; .reg .pred p,leader; elect.sync _|leader,1; @!leader bra QKDONE;\n"
        "setp.ne.b32 p,0,0;\n"
        "add.u64 a,%1,0;\n add.u64 b,%2,0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0],a,b,%3,p;\n"
        "setp.ne.b32 p,1,0;\n"
        "add.u64 a,%1,2;\n add.u64 b,%2,2;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0],a,b,%3,p;\n"
        "add.u64 a,%1,4;\n add.u64 b,%2,4;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0],a,b,%3,p;\n"
        "add.u64 a,%1,6;\n add.u64 b,%2,6;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0],a,b,%3,p;\n"
        "add.u64 a,%1,1024;\n add.u64 b,%2,1024;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0],a,b,%3,p;\n"
        "add.u64 a,%1,1026;\n add.u64 b,%2,1026;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0],a,b,%3,p;\n"
        "add.u64 a,%1,1028;\n add.u64 b,%2,1028;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0],a,b,%3,p;\n"
        "add.u64 a,%1,1030;\n add.u64 b,%2,1030;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0],a,b,%3,p;\n"
        "QKDONE: }\n"
        :: "r"(dst), "l"(q), "l"(k), "r"(instruction) : "memory");
}

//: One K16 step of a PV run: `bo` is the B-operand offset in 16 B descriptor units (128 B per key,
//: k * 8) and `ao` the A-operand TMEM column offset (two bf16 probabilities per column, k / 2).
#define GQA_PV_STEP(bo, ao, pred)                                                    \
    "add.u64 b,%2," #bo ";\n"                                                        \
    "add.s32 a,%1," #ao ";\n"                                                        \
    "tcgen05.mma.cta_group::1.kind::f16 [%0],[a],b,%3," #pred ";\n"

//: Issue a run of PV K16 updates from one asm block so ptxas hoists every descriptor into a uniform
//: register and emits the updates back to back.
#define GQA_PV_BLOCK(name, steps)                                                    \
__device__ __forceinline__ void name(uint32_t dst, uint32_t a, uint64_t b,            \
                                     uint32_t instruction, bool accumulate) {        \
    asm volatile("{ .reg .b64 b; .reg .b32 a; .reg .pred p,q,leader;\n"               \
                 "elect.sync _|leader,1; @!leader bra PVDONE;\n"                      \
                 "setp.ne.b32 p,%4,0;\n"                                             \
                 "setp.ne.b32 q,1,0;\n" steps "PVDONE: }\n"                           \
                 :: "r"(dst), "r"(a), "l"(b), "r"(instruction),                       \
                    "r"(int(accumulate)) : "memory");                                 \
}

GQA_PV_BLOCK(pv_keys_0_63, GQA_PV_STEP(0, 0, p) GQA_PV_STEP(128, 8, q)
                           GQA_PV_STEP(256, 16, q) GQA_PV_STEP(384, 24, q))
GQA_PV_BLOCK(pv_keys_64_127, GQA_PV_STEP(512, 32, q) GQA_PV_STEP(640, 40, q)
                             GQA_PV_STEP(768, 48, q) GQA_PV_STEP(896, 56, q))

//: Five-dimensional TMA tile: (feature 64, feature chunk, head, token, batch).
__device__ __forceinline__ void load_tile(const CUtensorMap& map, uint16_t* target,
                                          uint64_t* barrier, int feature_chunk,
                                          int head, int token) {
    asm volatile("cp.async.bulk.tensor.5d.shared::cta.global.mbarrier::complete_tx::bytes "
                 "[%0],[%1,{%2,%3,%4,%5,%6}],[%7];"
                 :: "r"(shared_address(target)), "l"(reinterpret_cast<uint64_t>(&map)),
                    "r"(0), "r"(feature_chunk), "r"(head), "r"(token), "r"(0),
                    "r"(shared_address(barrier)) : "memory");
}

}  // namespace gqa_sm100
