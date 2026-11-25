// SPDX-License-Identifier: Apache-2.0
// Tensor-memory allocation and CTA-pair handshake: the ONE ordering every sm100 kernel uses.
//
// Why this header exists. Two production failures were the same race:
//  * DS SGLang TP4 hang: mm_sm100 / checkpoint_tc issued `tcgen05.alloc.cta_group::2` before the
//    peer CTA was known to have started. ptxas lowers the pair allocation to a software protocol
//    (SASS `UTCATOMSWS.2CTA.FIND_AND_SET`): the even CTA allocates for both SMs, stores its own
//    slot, then does a remote mbarrier arrive + `STAS` into the PEER's driver-reserved shared
//    window and slot. A DSMEM access is defined only once the peer has started; a peer that starts
//    late re-initialises its window over the delivered handshake and waits in tcgen05.alloc forever.
//  * GLM "misaligned address": mm_sm100/general pre-stored a 0xffffffff sentinel in the slot and
//    allocated before a RELAXED cluster barrier; a late rank-1 CTA overwrote the delivered index
//    with the sentinel and its half of the protocol scaled it to 0xffffffe0 -> LDTM fault, and the
//    bogus dealloc corrupted the TPC's allocation state (later allocations spin = hang).
//
// THE RULES (a kernel that follows them cannot hit either failure):
//   R1 peer started  -- a cta_group::2 allocation is issued only after a COMPLETED cluster barrier
//                       (`peer_started_sync()`, or any earlier barrier.cluster arrive+wait that all
//                       threads of the cluster pass). Nothing may touch a peer's shared memory
//                       (remote arrive, st.async, multicast TMA, mapa+st) before that either.
//   R2 one writer    -- nothing but tcgen05.alloc writes the slot: no sentinel, no zero-fill, no
//                       other data aliased onto it before the allocation.
//   R3 publish       -- allocating warp: alloc [, relinquish], tcgen05.fence::before_thread_sync;
//                       then a barrier WITH memory ordering (canonical: the full CTA barrier,
//                       __syncthreads; a named barrier over the same readers or a releasing
//                       cluster barrier are equivalent -- never barrier.cluster.arrive.relaxed)
//                       between the allocating warp and every thread that reads the slot or issues
//                       tcgen05 ops; those threads then tcgen05.fence::after_thread_sync and read the
//                       slot (each CTA its OWN slot: cta_group::2 returns the same address to both).
//                       Use the slot's value: never assume column 0. Concurrent tcgen05 allocators on
//                       the same SM (other streams) make the base non-zero, and dealloc of a base the
//                       CTA does not own corrupts the SM's allocation state (later allocs spin).
//   R4 teardown      -- every thread that touched tensor memory: tcgen05.fence::before_thread_sync;
//                       a pair-wide releasing barrier (cluster arrive.release + wait.acquire for
//                       cta_group::2, a CTA barrier for cta_group::1); the allocating warp:
//                       tcgen05.fence::after_thread_sync, then dealloc of the base read in R3.
//                       (An mbarrier hand-off -- release arrive / acquire wait by the deallocating
//                       warp -- is an equivalent barrier; mm_sm100_leaf uses one.)
//
// All of this is synchronization only: no value, order, address or ownership changes, so a kernel
// moved onto these helpers produces the same bits. Raw PTX only (no cute/cutlass dependency), so
// hand-written kernels can use it too. Include it by relative path from a package csrc
// and pass `extra_sources=(build_cache.SM100_COMMON,)` to build_sm100_extension so the build key
// folds this file. `tests/test_tmem_pair_lint_cpu.py` (ops/sm100_common/tests) fails on any new
// 2-CTA allocation site that is not listed there with the evidence it follows R1-R4.
#pragma once

#include <cstdint>

namespace isoexec_sm100 {
namespace tmem {

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void fence_before_thread_sync() {
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void fence_after_thread_sync() {
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

// Every thread of every CTA of the cluster, convergent. Release on arrive, acquire on wait: plain
// and DSMEM stores (and fence.mbarrier_init) before it are visible after it, cluster-wide. Its
// completion proves every CTA of the cluster has started (R1).
__device__ __forceinline__ void cluster_sync_release_acquire() {
  asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}

// R1. Use where the kernel has no earlier completed cluster barrier in front of the allocation.
__device__ __forceinline__ void peer_started_sync() { cluster_sync_release_acquire(); }

// R2 + R3 (allocating side). Whole warp, convergent (`.sync.aligned`). kCtaGroup 2 must follow R1.
template <uint32_t kCtaGroup, bool kRelinquish = true>
__device__ __forceinline__ void alloc(uint32_t* slot, uint32_t cols) {
  static_assert(kCtaGroup == 1 || kCtaGroup == 2, "cta_group is 1 or 2");
  const uint32_t s = smem_addr(slot);
  if constexpr (kCtaGroup == 1) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(s), "r"(cols) : "memory");
    if constexpr (kRelinquish) asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
  } else {
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(s), "r"(cols) : "memory");
    if constexpr (kRelinquish) asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;" ::: "memory");
  }
  fence_before_thread_sync();
}

// R3 (reading side), after the ordering barrier.
__device__ __forceinline__ uint32_t read_base(const uint32_t* slot) {
  fence_after_thread_sync();
  uint32_t v;
  asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(smem_addr(slot)) : "memory");
  return v;
}

// R1 + R2 + R3 for the common case -- exactly the mm_sm100/general gen.cu sequence:
// releasing cluster barrier (pair only; it also publishes mbarrier inits made before it), allocation by
// one warp + before-fence, a full CTA barrier, after-fence, each CTA reads its OWN slot. Every thread
// of the CTA calls it, convergent. Skip the cluster barrier with kPeerStarted = true only when an
// earlier completed cluster barrier already proved the peer started AND published the inits.
template <uint32_t kCtaGroup, bool kPeerStarted = false, bool kRelinquish = true>
__device__ __forceinline__ uint32_t setup(uint32_t* slot, uint32_t cols, bool alloc_warp) {
  if constexpr (kCtaGroup == 2 && !kPeerStarted) peer_started_sync();
  if (alloc_warp) alloc<kCtaGroup, kRelinquish>(slot, cols);
  __syncthreads();
  return read_base(slot);
}

// R4. Every thread of the CTA, convergent, after its last tensor-memory access.
template <uint32_t kCtaGroup>
__device__ __forceinline__ void dealloc(uint32_t base, uint32_t cols) {
  if constexpr (kCtaGroup == 1) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(base), "r"(cols) : "memory");
  } else {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" ::"r"(base), "r"(cols) : "memory");
  }
}

template <uint32_t kCtaGroup>
__device__ __forceinline__ void teardown(uint32_t base, uint32_t cols, bool alloc_warp) {
  fence_before_thread_sync();
  if constexpr (kCtaGroup == 2) {
    cluster_sync_release_acquire();
  } else {
    __syncthreads();
  }
  if (alloc_warp) {
    fence_after_thread_sync();
    dealloc<kCtaGroup>(base, cols);
  }
}

}  // namespace tmem
}  // namespace isoexec_sm100
