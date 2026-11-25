// SPDX-License-Identifier: Apache-2.0
//
// Producer-side push of symm_tree_allreduce_v1's tagged one-shot (DESIGN.md section 1). A kernel that
// produces this rank's FP32 subtree root (for example a row-parallel GEMM epilogue) may push it straight to
// every peer with push4(); the reduction is then launched in "prepushed" mode (poll + tree only), so the
// push overlaps the producer instead of sitting on the collective's critical path.
//
// Contract (identical to the standalone kernel, so the bits and the safety argument are unchanged):
//   * element offset c (multiple of 4, c + 3 < n <= tslot) of the flattened root, value v = root[c .. c+3];
//   * every element of the root is pushed exactly once per call, before the prepushed reduction launches
//     in stream order on the same stream;
//   * the tag and parity half come from the reduction CTA that owns c: b = (c / kTile) % kTagGrid,
//     e = tepoch[b] + 1 (tepoch is advanced only by the reduction kernel, after this push).
// Header-only, no dependencies beyond CUDA.
#pragma once

#include <cstdint>

namespace detreduce_push {

constexpr int kTile = 1024;      // must equal detreduce.cu kTile
constexpr int kTagGrid = 256;    // must equal detreduce.cu kTagGrid
constexpr int kMaxWorld = 16;

struct PushDesc {
  uint64_t* tbox[kMaxWorld];     // each rank's tagged box (peer-mapped)
  int64_t tcap;                  // words per parity half
  int64_t tslot;                 // words per sender slot
  const int* tepoch;             // this rank's tagged epochs [kTagGrid]
  float* lbox[kMaxWorld];        // each rank's Lamport box (peer-mapped), for push4_lamport
  int64_t lcap;                  // words per parity half
  int64_t lslot;                 // words per sender slot
  int rank, world;
};

__device__ __forceinline__ void push4(const PushDesc& d, int64_t c, float4 v) {
  const int b = static_cast<int>((c / kTile) % kTagGrid);
  const int e = *reinterpret_cast<const volatile int*>(d.tepoch + b) + 1;
  const uint64_t t = static_cast<uint64_t>(static_cast<uint32_t>(e)) << 32;
  const uint64_t w0 = __float_as_uint(v.x) | t, w1 = __float_as_uint(v.y) | t;
  const uint64_t w2 = __float_as_uint(v.z) | t, w3 = __float_as_uint(v.w) | t;
  const int64_t off = (e & 1) * d.tcap + static_cast<int64_t>(d.rank) * d.tslot + c;
  for (int q = 0; q < d.world; ++q) {
    if (q == d.rank) continue;
    uint64_t* p = d.tbox[q] + off;
    asm volatile("st.relaxed.sys.global.v2.b64 [%0], {%1, %2};" ::"l"(p), "l"(w0), "l"(w1) : "memory");
    asm volatile("st.relaxed.sys.global.v2.b64 [%0], {%1, %2};" ::"l"(p + 2), "l"(w2), "l"(w3) : "memory");
  }
}

// Lamport form (kernel mode 6 = poll only): one 32-bit word per element, sentinel 0xFFFFFFFF; a root word that
// equals the sentinel is sent as 0x7FFFFFFF (bit-neutral: every root enters an FP32 add, which returns the
// canonical NaN for any NaN operand). Same tag/epoch source (tepoch of the CTA that owns c) and the same
// contract as push4, with n <= lslot.
__device__ __forceinline__ uint32_t lamport_word(float v) {
  const uint32_t b = __float_as_uint(v);
  return b == 0xFFFFFFFFu ? 0x7FFFFFFFu : b;
}
__device__ __forceinline__ void push4_lamport(const PushDesc& d, int64_t c, float4 v) {
  const int b = static_cast<int>((c / kTile) % kTagGrid);
  const int e = *reinterpret_cast<const volatile int*>(d.tepoch + b) + 1;
  const int64_t off = (e & 1) * d.lcap + static_cast<int64_t>(d.rank) * d.lslot + c;
  for (int q = 0; q < d.world; ++q) {
    if (q == d.rank) continue;
    float* p = d.lbox[q] + off;
    asm volatile("st.relaxed.sys.global.v4.b32 [%0], {%1, %2, %3, %4};" ::"l"(p), "r"(lamport_word(v.x)),
                 "r"(lamport_word(v.y)), "r"(lamport_word(v.z)), "r"(lamport_word(v.w))
                 : "memory");
  }
}

}  // namespace detreduce_push
