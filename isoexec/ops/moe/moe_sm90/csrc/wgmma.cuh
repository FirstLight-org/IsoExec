// SPDX-License-Identifier: Apache-2.0
// Hopper K-major, 128-byte swizzled shared operands; no CUTLASS dependency.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <type_traits>

template <class T> __device__ __forceinline__ T cast(float x) {
  if constexpr (std::is_same_v<T, __nv_bfloat16>)
    return __float2bfloat16_rn(x);
  else if constexpr (std::is_same_v<T, __half>)
    return __float2half_rn(x);
  else
    return x;
}

__device__ __forceinline__ uint64_t descriptor(const void *p) {
  const uint64_t a = __cvta_generic_to_shared(p);
  return ((a & 0x3ffff) >> 4) | (uint64_t(1) << 16) | (uint64_t(64) << 32) |
         (uint64_t(1) << 62);
}

template <class T> __device__ __forceinline__ float rounded(float x) {
  if constexpr (std::is_same_v<T, __nv_bfloat16>)
    return __bfloat162float(__float2bfloat16_rn(x));
  else
    return __half2float(__float2half_rn(x));
}

template <class T>
__device__ __forceinline__ uint32_t pack_leaf(float a, float b) {
  if constexpr (std::is_same_v<T, __nv_bfloat16>)
    return uint32_t(__bfloat16_as_ushort(__float2bfloat16_rn(a))) |
           (uint32_t(__bfloat16_as_ushort(__float2bfloat16_rn(b))) << 16);
  else
    return uint32_t(__half_as_ushort(__float2half_rn(a))) |
           (uint32_t(__half_as_ushort(__float2half_rn(b))) << 16);
}

template <class T>
__device__ __forceinline__ float unpack_leaf(uint32_t pair, int high) {
  uint16_t bits = pair >> (high * 16);
  if constexpr (std::is_same_v<T, __nv_bfloat16>)
    return __bfloat162float(__ushort_as_bfloat16(bits));
  else
    return __half2float(__ushort_as_half(bits));
}

template <class T>
__device__ __forceinline__ void mma(float (&d)[32], uint64_t a, uint64_t b,
                                    int accumulate) {
#define OUTPUTS                                                                \
  "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),      \
      "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]),             \
      "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),         \
      "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]),         \
      "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]),         \
      "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]),         \
      "+f"(d[31])
#define INSTRUCTION(TYPE)                                                      \
  "{ .reg .pred p; setp.ne.b32 p, %34, 0; "                                    \
  "wgmma.mma_async.sync.aligned.m64n64k16.f32." TYPE "." TYPE " "              \
  "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"                    \
  "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"          \
  "%32,%33,p,1,1,0,0; }"
  if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    asm volatile(INSTRUCTION("bf16")
                 : OUTPUTS
                 : "l"(a), "l"(b), "r"(accumulate));
  } else {
    asm volatile(INSTRUCTION("f16")
                 : OUTPUTS
                 : "l"(a), "l"(b), "r"(accumulate));
  }
#undef INSTRUCTION
#undef OUTPUTS
}

__device__ __forceinline__ void copy16(void *dst, const void *src, int bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(
                   (uint32_t)__cvta_generic_to_shared(dst)),
               "l"(src), "r"(bytes)
               : "memory");
}

template <int N> __device__ __forceinline__ void fence_acc(float (&d)[N]) {
  // Make accumulator dependencies visible to the compiler across async
  // instructions.
#pragma unroll
  for (int i = 0; i < N; ++i)
    asm volatile("" : "+f"(d[i])::"memory");
}

// A single n128 instruction computes gate and up with one shared A operand.
template <class T>
__device__ __forceinline__ void mma_glu(float (&d)[32], float (&u)[32],
                                        uint64_t a, uint64_t b, int acc) {
  if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    asm volatile(
        "{ .reg .pred p; setp.ne.b32 p, %66, 0; "
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,"
        "%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,"
        "%24,%25,%26,%27,%28,%29,%30,%31,"
        "%32,%33,%34,%35,%36,%37,%38,%39,"
        "%40,%41,%42,%43,%44,%45,%46,%47,"
        "%48,%49,%50,%51,%52,%53,%54,%55,"
        "%56,%57,%58,%59,%60,%61,%62,%63},"
        "%64,%65,p,1,1,0,0; }"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]),
          "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]),
          "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]),
          "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
          "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]),
          "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]),
          "+f"(d[30]), "+f"(d[31]), "+f"(u[0]), "+f"(u[1]), "+f"(u[2]),
          "+f"(u[3]), "+f"(u[4]), "+f"(u[5]), "+f"(u[6]), "+f"(u[7]),
          "+f"(u[8]), "+f"(u[9]), "+f"(u[10]), "+f"(u[11]), "+f"(u[12]),
          "+f"(u[13]), "+f"(u[14]), "+f"(u[15]), "+f"(u[16]), "+f"(u[17]),
          "+f"(u[18]), "+f"(u[19]), "+f"(u[20]), "+f"(u[21]), "+f"(u[22]),
          "+f"(u[23]), "+f"(u[24]), "+f"(u[25]), "+f"(u[26]), "+f"(u[27]),
          "+f"(u[28]), "+f"(u[29]), "+f"(u[30]), "+f"(u[31])
        : "l"(a), "l"(b), "r"(acc));
  } else {
    asm volatile(
        "{ .reg .pred p; setp.ne.b32 p, %66, 0; "
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,"
        "%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,"
        "%24,%25,%26,%27,%28,%29,%30,%31,"
        "%32,%33,%34,%35,%36,%37,%38,%39,"
        "%40,%41,%42,%43,%44,%45,%46,%47,"
        "%48,%49,%50,%51,%52,%53,%54,%55,"
        "%56,%57,%58,%59,%60,%61,%62,%63},"
        "%64,%65,p,1,1,0,0; }"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]),
          "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]),
          "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]),
          "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
          "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]),
          "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]),
          "+f"(d[30]), "+f"(d[31]), "+f"(u[0]), "+f"(u[1]), "+f"(u[2]),
          "+f"(u[3]), "+f"(u[4]), "+f"(u[5]), "+f"(u[6]), "+f"(u[7]),
          "+f"(u[8]), "+f"(u[9]), "+f"(u[10]), "+f"(u[11]), "+f"(u[12]),
          "+f"(u[13]), "+f"(u[14]), "+f"(u[15]), "+f"(u[16]), "+f"(u[17]),
          "+f"(u[18]), "+f"(u[19]), "+f"(u[20]), "+f"(u[21]), "+f"(u[22]),
          "+f"(u[23]), "+f"(u[24]), "+f"(u[25]), "+f"(u[26]), "+f"(u[27]),
          "+f"(u[28]), "+f"(u[29]), "+f"(u[30]), "+f"(u[31])
        : "l"(a), "l"(b), "r"(acc));
  }
}

// Transposed 16-token tile: weight channels form M, token rows form N.
template <class T>
__device__ __forceinline__ void mma_small(float (&d)[8], uint64_t a, uint64_t b,
                                          int acc) {
#define SMALL_OUT                                                              \
  "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),      \
      "+f"(d[6]), "+f"(d[7])
#define SMALL_OP(TYPE)                                                         \
  "{ .reg .pred p; setp.ne.b32 p,%10,0; "                                      \
  "wgmma.mma_async.sync.aligned.m64n16k16.f32." TYPE "." TYPE                  \
  " {%0,%1,%2,%3,%4,%5,%6,%7},%8,%9,p,1,1,0,0; }"
  if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    asm volatile(SMALL_OP("bf16") : SMALL_OUT : "l"(a), "l"(b), "r"(acc));
  } else {
    asm volatile(SMALL_OP("f16") : SMALL_OUT : "l"(a), "l"(b), "r"(acc));
  }
#undef SMALL_OP
#undef SMALL_OUT
}
