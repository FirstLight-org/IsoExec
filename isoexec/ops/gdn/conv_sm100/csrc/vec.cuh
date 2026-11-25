// SPDX-License-Identifier: Apache-2.0
// Vector widths for the SM100 conv kernels. Deliberately *not* in common.cuh: that file is the
// numerical recipe and is byte-identical to conv_sm90's scalar graph, while per-thread vector width
// is pure data movement and is exactly the kind of thing docs/sm100-bringup.md says an SM100 kernel
// is free to choose. Nothing here rounds, widens or narrows anything.
#pragma once
#include "common.cuh"

namespace conv_sm100 {

/// Natural alignment for a contiguous run of ``BYTES`` bytes, capped at one 128-bit access.
constexpr int vec_align(int bytes) {
  return bytes >= 16 ? 16 : bytes >= 8 ? 8 : bytes >= 4 ? 4 : 2;
}

/// ``N`` contiguous elements loaded/stored as one packed global memory operation.
/// ``packv<T, 8>`` is layout- and alignment-compatible with common.cuh's ``pack8<T>``.
template <class T, int N>
struct __align__(vec_align(int(sizeof(T)) * N)) packv {
  T v[N];
};

/// Widen a packed vector to fp32. **Bit-identical to widening each element with common.cuh's
/// ``widen``, by construction, not by approximation**: bf16 *is* the top 16 bits of the fp32 with
/// the low 16 zeroed, for every bit pattern including NaN payloads, infinities, signed zeros and
/// subnormals, and `__bfloat162float` is that placement. Doing it on the two halves of each 32-bit
/// word of the load -- `lo = w << 16`, `hi = w & 0xffff0000` -- is one instruction per element
/// instead of the extract-then-place pair (PRMT + SHF) ptxas emits when the 16-bit members are read
/// one at a time. Pure data movement; no rounding happens anywhere in here.
template <class T, int N>
__device__ __forceinline__ void widen_vec(const packv<T, N> &p, float (&v)[N]) {
  if constexpr (std::is_same_v<T, bf16> && N % 2 == 0) {
    const uint32_t *u = reinterpret_cast<const uint32_t *>(&p);
#pragma unroll
    for (int k = 0; k < N / 2; ++k) {
      const uint32_t w = u[k];
      v[2 * k] = __uint_as_float(w << 16);          // low half: the earlier channel
      v[2 * k + 1] = __uint_as_float(w & 0xffff0000u);
    }
  } else {
#pragma unroll
    for (int j = 0; j < N; ++j)
      v[j] = widen(p.v[j]);
  }
}

__device__ __forceinline__ unsigned long long f2bits(float2 a) {
  return (unsigned long long)__float_as_uint(a.x) | ((unsigned long long)__float_as_uint(a.y) << 32);
}
__device__ __forceinline__ float2 bits2f(unsigned long long u) {
  return make_float2(__uint_as_float((unsigned)u), __uint_as_float((unsigned)(u >> 32)));
}
__device__ __forceinline__ float2 mul2(float2 a, float2 b) {
  unsigned long long r;
  asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(r) : "l"(f2bits(a)), "l"(f2bits(b)));
  return bits2f(r);
}
__device__ __forceinline__ float2 add2(float2 a, float2 b) {
  unsigned long long r;
  asm("add.rn.f32x2 %0, %1, %2;" : "=l"(r) : "l"(f2bits(a)), "l"(f2bits(b)));
  return bits2f(r);
}
__device__ __forceinline__ float2 sub2(float2 a, float2 b) {
  unsigned long long r;
  asm("sub.rn.f32x2 %0, %1, %2;" : "=l"(r) : "l"(f2bits(a)), "l"(f2bits(b)));
  return bits2f(r);
}

}  // namespace conv_sm100
