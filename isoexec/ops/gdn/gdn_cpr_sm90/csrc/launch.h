// SPDX-License-Identifier: Apache-2.0
// Host-side launch helpers shared by the family's kernels.
#pragma once
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include <atomic>

namespace gdn_cpr {

// SMs of the current device (cached per device ordinal < 64); an error is raised, never guessed.
inline int device_sms() {
  static std::atomic<int> cache[64] = {};
  int dev = 0;
  C10_CUDA_CHECK(cudaGetDevice(&dev));
  if (dev < 64) {
    const int n = cache[dev].load(std::memory_order_relaxed);
    if (n > 0) return n;
  }
  int n = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev));
  TORCH_CHECK(n > 0, "gdn_cpr_sm90: the device reports no SMs");
  if (dev < 64) cache[dev].store(n, std::memory_order_relaxed);
  return n;
}

// Raise a kernel's dynamic shared memory limit once per device (the attribute is a per-device setting;
// setting it on every launch costs microseconds of host time on the decode path). Returns the error.
template <auto Kernel>   // the kernel itself: one flag set per kernel (instantiations share a type)
cudaError_t allow_smem(int bytes) {
  static std::atomic<uint64_t> done{0};   // one bit per device ordinal < 64
  int dev = 0;
  cudaError_t err = cudaGetDevice(&dev);
  if (err != cudaSuccess) return err;
  const uint64_t bit = dev < 64 ? uint64_t(1) << dev : 0;
  if (bit && (done.load(std::memory_order_acquire) & bit)) return cudaSuccess;
  err = cudaFuncSetAttribute(Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
  if (err == cudaSuccess && bit) done.fetch_or(bit, std::memory_order_release);
  return err;
}

}  // namespace gdn_cpr
