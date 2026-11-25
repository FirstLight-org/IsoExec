// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

__device__ __forceinline__ void barrier_init(uint64_t *bar) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(
                   (uint32_t)__cvta_generic_to_shared(bar))
               : "memory");
}

__device__ __forceinline__ void barrier_expect(uint64_t *bar, int bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(
                   (uint32_t)__cvta_generic_to_shared(bar)),
               "r"(bytes)
               : "memory");
}

__device__ __forceinline__ void barrier_wait(uint64_t *bar, int phase) {
  asm volatile(
      "{ .reg .pred done; retry: "
      "mbarrier.try_wait.parity.shared::cta.b64 done, [%0], %1; "
      "@!done bra retry; }" ::"r"((uint32_t)__cvta_generic_to_shared(bar)),
      "r"(phase)
      : "memory");
}

__device__ __forceinline__ void tma2(void *dst, const CUtensorMap *desc,
                                     uint64_t *bar, int k, int m) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::"
               "complete_tx::bytes "
               "[%0], [%1, {%3, %4}], [%2];" ::"r"(
                   (uint32_t)__cvta_generic_to_shared(dst)),
               "l"(desc), "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(k),
               "r"(m)
               : "memory");
}

__device__ __forceinline__ void tma3(void *dst, const CUtensorMap *desc,
                                     uint64_t *bar, int k, int n, int e) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::"
               "complete_tx::bytes "
               "[%0], [%1, {%3, %4, %5}], [%2];" ::"r"(
                   (uint32_t)__cvta_generic_to_shared(dst)),
               "l"(desc), "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(k),
               "r"(n), "r"(e)
               : "memory");
}
