// SPDX-License-Identifier: BSD-3-Clause
// tcgen05 `cta_group::2` access and MMA issue for the HD256 leaf. EXECUTION ONLY -- nothing in this
// header can move a bit of the numerical program, which is stated in ../README.md section 1 and
// implemented by numerics.cuh (shared verbatim with the HD128 leaf).
//
// Derived from TileFoundry's CUDA port of FlashAttention-4.
// Copyright (c) 2025, Tri Dao, Siyu Wang, Shengbin Di, Yuxi Chi, Johnsonms, Linfeng Zheng,
// Haoyan Huang, Lanbo Li, Yun Zhong, Man Yuan, Minmin Sun, Yong Li, Wei Lin, Jay Shah,
// Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani. Distributed under BSD-3-Clause.
//
// The two-CTA geometry, in one place:
//
//   cluster        two CTAs, rank = blockIdx.x & 1, item = blockIdx.x >> 1
//   per-CTA Q      128 packed query rows x 256 features, four 64-feature chunks   (65536 B)
//   per-CTA K      64 of the tile's 128 keys x 256 features, four chunks          (32768 B / slot)
//   per-CTA V      128 keys x 128 of the 256 features, two chunks                 (32768 B / slot)
//   QK             tcgen05.mma.cta_group::2 M256 N128 K16 x 16 steps  -> S, 128 TMEM columns
//   PV             tcgen05.mma.cta_group::2 M256 N256 K16 x 8 steps   -> O, 256 TMEM columns
//   TMEM           S0 @ 0, S1 @ 128, O @ 256..511. 512 columns: FULL.
//
// M = 256 means the pair's 256 packed query rows: each CTA's TMEM holds ITS OWN 128 rows (128 lanes)
// for every N column. N is what the pair splits across the two CTAs' shared operands -- 64 keys each
// for QK, 128 features each for PV. So a CTA owns 128 query rows and ALL 256 output features, which
// is why the epilogue and the fp32 export are purely local and need no cross-CTA reduction.
#pragma once

#include <cuda.h>
#include <cstdint>

#include "numerics.cuh"
#include "tmem.cuh"

namespace gqa_sm100 {

// --- cluster-scope barrier arrivals --------------------------------------------------------------

//: The address of `barrier` inside cluster rank 0. Every cross-CTA rendezvous lands on rank 0's copy,
//: because rank 0 is the only CTA that issues MMAs for the pair.
__device__ __forceinline__ uint32_t rank0_address(const uint64_t* barrier) {
    uint32_t remote;
    asm volatile("mapa.shared::cluster.u32 %0,%1,0;"
                 : "=r"(remote) : "r"(shared_address(barrier)));
    return remote;
}

//: One arrival at rank 0's copy, from the calling thread. Relaxed: the data this announces was
//: published either by a TMA (whose `mbarrier::complete_tx` carries async-proxy visibility) or by
//: tcgen05 traffic fenced with `tcgen05.fence::before_thread_sync`. The arrival supplies cluster
//: EXECUTION ordering, not generic-proxy visibility.
__device__ __forceinline__ void arrive_cluster(uint64_t* barrier) {
    const uint32_t remote = rank0_address(barrier);
    asm volatile("mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _,[%0];"
                 :: "r"(remote) : "memory");
}

//: One elected lane of a fully populated warp arrives at rank 0's copy. `tcgen05` accesses and their
//: waits are warp-synchronous, so one lane's arrival after the warp-wide `tcgen05.wait::st` plus
//: `fence::before_thread_sync` carries exactly the same information as thirty-two arrivals with one
//: sixteenth of the cluster-remote barrier traffic.
__device__ __forceinline__ void arrive_warp_cluster(uint64_t* barrier) {
    const uint32_t remote = rank0_address(barrier);
    asm volatile("{ .reg .pred e; elect.sync _|e,0xffffffff; "
                 "@e mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _,[%0]; }"
                 :: "r"(remote) : "memory");
}

//: A RELEASE arrival at rank 0's copy, from the calling thread. `values_ready` is the one barrier
//: this kernel arms after writing through the GENERIC proxy: the publisher warp zeroes the value
//: rows at or past `seqused_k` before the PV MMA may read them, and at D256 the reader is the MMA
//: thread of the OTHER CTA. A relaxed arrival does not order those stores before the arrival as that
//: thread observes them, and `fence.proxy.async.shared::cta` only reaches this CTA's own async
//: proxy. Pairs with `bar_wait_cluster`.
//:
//: No `elect.sync` here: the caller is already a single lane (the publisher's lane 0, after the
//: warp-wide `fence.proxy.async.shared::cluster` and a `__syncwarp`), and `elect.sync` is a
//: `.sync` instruction whose whole member mask must execute it.
__device__ __forceinline__ void arrive_cluster_release(uint64_t* barrier) {
    const uint32_t remote = rank0_address(barrier);
    asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _,[%0];"
                 :: "r"(remote) : "memory");
}

//: Order this CTA's generic-proxy shared writes before an async-proxy (UMMA) read made by ANY CTA of
//: the cluster. The `.shared::cluster` form is what makes the peer CTA's tensor-core read of our V
//: buffer see the zeros; `.shared::cta` would only cover our own.
__device__ __forceinline__ void fence_async_cluster() {
    asm volatile("fence.proxy.async.shared::cluster;" ::: "memory");
}

//: The acquiring half of `arrive_cluster_release`. `bar_wait` (numerics.cuh) defaults to
//: `.acquire.cta`, which does not order a peer CTA's generic writes; this is `.acquire.cluster`.
__device__ __forceinline__ void bar_wait_cluster(uint64_t* barrier, int phase) {
    asm volatile("{ .reg .pred p; CWAITLBL: "
                 "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 "
                 "p,[%0],%1,10000000; @!p bra CWAITLBL; }"
                 :: "r"(shared_address(barrier)), "r"(phase) : "memory");
}

//: A cluster-wide join. Both CTAs' TMEM and generic shared traffic retires first.
__device__ __forceinline__ void synchronize_pair() {
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

//: The UMMA completion arrival, MULTICAST to both CTAs of the pair (mask 0b11). Each CTA's copy of
//: the barrier is initialised with ONE arrival and receives exactly one.
__device__ __forceinline__ void commit_pair(uint64_t* barrier) {
    asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster."
                 "multicast::cluster.b64 [%0],%1;"
                 :: "r"(shared_address(barrier)), "h"(uint16_t(3)) : "memory");
}

// --- MMA issue -----------------------------------------------------------------------------------

//: One K=256 QK update as SIXTEEN K16 `cta_group::2` MMAs issued by one elected lane of the rank-0
//: MMA warp. A is Q (128 rows x 256 features per CTA) in four 64-feature chunks 16384 B apart
//: (+1024 descriptor units); B is K (64 keys x 256 features per CTA) in four chunks 8192 B apart
//: (+512 units). These are FA4's own offsets for its D256 two-CTA QK run.
//:
//: The contraction is over the 256 head-dim features in ascending order, four K16 steps per chunk,
//: chunks in order -- the SAME rule as the HD128 leaf's eight steps over 128 features, at twice the
//: extent. There is no D256 alternative order to choose between, so this is not a second recipe.
__device__ __forceinline__ void qk_sequence256(uint32_t dst, uint64_t q, uint64_t k,
                                               uint32_t instruction) {
#define GQA_QK256_STEP(ao, bo)                                                       \
    "add.u64 a,%1," #ao ";\n add.u64 b,%2," #bo ";\n"                                \
    "tcgen05.mma.cta_group::2.kind::f16 [%0],a,b,%3,{%4,%4,%4,%4,%4,%4,%4,%4},p;\n"
    asm volatile(
        //: Member mask ONE, not 0xffffffff: this is called by the MMA issuer THREAD, which is lane 0
        //: of its warp with the other 31 lanes diverged. `elect.sync` synchronizes the threads in the
        //: mask, so naming lanes that never execute it can hang. `tmem.cuh`'s `qk_sequence128` and
        //: FA4's `qk_sequence_scoped` both name lane 0 alone for the same reason.
        "{ .reg .b64 a,b; .reg .pred p,leader; elect.sync _|leader,1;\n"
        "@!leader bra QK256DONE;\n"
        "setp.ne.b32 p,0,0;\n"
        GQA_QK256_STEP(0, 0)
        "setp.ne.b32 p,1,0;\n"
        GQA_QK256_STEP(2, 2)
        GQA_QK256_STEP(4, 4)
        GQA_QK256_STEP(6, 6)
        GQA_QK256_STEP(1024, 512)
        GQA_QK256_STEP(1026, 514)
        GQA_QK256_STEP(1028, 516)
        GQA_QK256_STEP(1030, 518)
        GQA_QK256_STEP(2048, 1024)
        GQA_QK256_STEP(2050, 1026)
        GQA_QK256_STEP(2052, 1028)
        GQA_QK256_STEP(2054, 1030)
        GQA_QK256_STEP(3072, 1536)
        GQA_QK256_STEP(3074, 1538)
        GQA_QK256_STEP(3076, 1540)
        GQA_QK256_STEP(3078, 1542)
        "QK256DONE: }\n"
        :: "r"(dst), "l"(q), "l"(k), "r"(instruction), "r"(0) : "memory");
#undef GQA_QK256_STEP
}

//: One K16 step of a D256 PV run. `bo` is the B-operand offset in 16 B descriptor units (128 B per
//: key of the CTA's own 64-feature half-pair, so k * 8) and `ao` the A-operand TMEM column offset
//: (two bf16 probabilities per column, k / 2). Identical addressing to the HD128 leaf; only N and
//: the CTA group differ.
#define GQA_PV256_STEP(bo, ao, pred)                                                 \
    "add.u64 b,%2," #bo ";\n"                                                        \
    "add.s32 a,%1," #ao ";\n"                                                        \
    "tcgen05.mma.cta_group::2.kind::f16 [%0],[a],b,%3,{%5,%5,%5,%5,%5,%5,%5,%5}," #pred ";\n"

//: A run of PV K16 updates from one asm block, so ptxas hoists every descriptor into a uniform
//: register and emits the updates back to back. Member mask ONE for the same reason as
//: `qk_sequence256`: the caller is a single thread, lane 0 of a warp whose other lanes diverged.
//:
//: The PV contraction is over the 128 KEYS of the tile -- eight K16 steps in ascending key order,
//: exactly as at HD128. D256 splits the FEATURE (N) axis across the CTA pair, never the key axis, so
//: the fp32 accumulation order of every output element is bit-for-bit the HD128 leaf's.
#define GQA_PV256_BLOCK(name, steps)                                                 \
__device__ __forceinline__ void name(uint32_t dst, uint32_t a, uint64_t b,           \
                                     uint32_t instruction, bool accumulate) {        \
    asm volatile("{ .reg .b64 b; .reg .b32 a; .reg .pred p,q,leader;\n"              \
                 "elect.sync _|leader,1; @!leader bra PV256DONE;\n"                 \
                 "setp.ne.b32 p,%4,0;\n"                                             \
                 "setp.ne.b32 q,1,0;\n" steps "PV256DONE: }\n"                       \
                 :: "r"(dst), "r"(a), "l"(b), "r"(instruction),                      \
                    "r"(int(accumulate)), "r"(0) : "memory");                        \
}

GQA_PV256_BLOCK(pv256_keys_0_63, GQA_PV256_STEP(0, 0, p) GQA_PV256_STEP(128, 8, q)
                                 GQA_PV256_STEP(256, 16, q) GQA_PV256_STEP(384, 24, q))
GQA_PV256_BLOCK(pv256_keys_64_127, GQA_PV256_STEP(512, 32, q) GQA_PV256_STEP(640, 40, q)
                                   GQA_PV256_STEP(768, 48, q) GQA_PV256_STEP(896, 56, q))

// --- TMEM accessors ------------------------------------------------------------------------------

//: Rescale the 256-column fp32 output accumulator by the softmax correction factor: EIGHT sequential
//: 32-column read-modify-writes. Elementwise `mul.rn.ftz.f32x2` with one broadcast scalar -- there is
//: no reduction here, so the chunking is free and the bits are those of the HD128 leaf's
//: `correction128` over the first 128 columns, extended over the second 128.
//:
//: `O *= alpha` ONLY when `alpha != 1`, exactly as `correction128` does and for the same reason:
//: `mul.rn.ftz.f32x2` states `.ftz`, so multiplying by exactly 1.0 FLUSHES a denormal accumulator
//: element to zero. A lane that kept its running max has `alpha == 1` and must not be touched, or
//: the warp-wide rescale stops being bit-neutral on an accumulator that happens to hold a denormal.
//: The tcgen05 store is warp-wide and cannot be skipped per lane, so a kept lane writes back exactly
//: the bits it loaded.
__device__ __forceinline__ void correction256(uint32_t base, float alpha) {
    const bool scale = alpha != 1.0f;
    #pragma unroll 1
    for (int chunk = 0; chunk < 8; ++chunk) {
        float values[32];
        load32(base + chunk * 32, values);
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        if (scale) {
            #pragma unroll
            for (int j = 0; j < 32; j += 2) multiply_pair(values[j], values[j + 1], alpha);
        }
        store32(base + chunk * 32, values);
    }
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}

}  // namespace gqa_sm100
