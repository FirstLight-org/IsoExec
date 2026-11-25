// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "common.cuh"

namespace rope_sm100 {

// Bytes one access moves: the widest of {16, 8, 4} that divides the V-element block.
template <int NB> struct chunk_of {
  static constexpr int bytes = (NB % 16 == 0) ? 16 : ((NB % 8 == 0) ? 8 : 4);
};

// V adjacent elements -> V floats, widened exactly (rd2 per pair).
// With WIDE, `p` must be aligned to min(V*sizeof(T), 16) bytes and the caller's host code gates that;
// with WIDE = false every access is 4 bytes, i.e. exactly rope_sm90's, which needs no gate. The two
// return the same floats -- the gate decides how many instructions fetch them, nothing else.
template <class T, int V, bool WIDE = true>
__device__ __forceinline__ void rdv(const T *p, float (&o)[V]) {
  static_assert(V >= 2 && (V & (V - 1)) == 0, "V must be a power of two >= 2");
  constexpr int NB = V * int(sizeof(T));
  constexpr int CB = WIDE ? chunk_of<NB>::bytes : 4;
  if constexpr (CB == 16) {
    union { uint4 c[NB / 16]; T e[V]; } b;
    const uint4 *s = reinterpret_cast<const uint4 *>(p);
#pragma unroll
    for (int i = 0; i < NB / 16; ++i)
      b.c[i] = s[i];
#pragma unroll
    for (int i = 0; i < V; i += 2)
      rd2(&b.e[i], o[i], o[i + 1]);
  } else if constexpr (CB == 8) {
    union { uint2 c[NB / 8]; T e[V]; } b;
    const uint2 *s = reinterpret_cast<const uint2 *>(p);
#pragma unroll
    for (int i = 0; i < NB / 8; ++i)
      b.c[i] = s[i];
#pragma unroll
    for (int i = 0; i < V; i += 2)
      rd2(&b.e[i], o[i], o[i + 1]);
  } else {
#pragma unroll
    for (int i = 0; i < V; i += 2)
      rd2(p + i, o[i], o[i + 1]);
  }
}

// As rdv, but always 4-byte accesses -- for tiny operands (per-head weights) that are L1-resident
// anyway and whose alignment the host does not gate.
template <class T, int V>
__device__ __forceinline__ void rdv_narrow(const T *p, float (&o)[V]) {
#pragma unroll
  for (int i = 0; i < V; i += 2)
    rd2(p + i, o[i], o[i + 1]);
}

// V floats -> V adjacent elements, each RN-narrowed by wr2.
template <class T, int V, bool WIDE = true>
__device__ __forceinline__ void wrv(T *p, const float (&x)[V]) {
  static_assert(V >= 2 && (V & (V - 1)) == 0, "V must be a power of two >= 2");
  constexpr int NB = V * int(sizeof(T));
  constexpr int CB = WIDE ? chunk_of<NB>::bytes : 4;
  if constexpr (CB == 16) {
    union { uint4 c[NB / 16]; T e[V]; } b;
#pragma unroll
    for (int i = 0; i < V; i += 2)
      wr2(&b.e[i], x[i], x[i + 1]);
    uint4 *d = reinterpret_cast<uint4 *>(p);
#pragma unroll
    for (int i = 0; i < NB / 16; ++i)
      d[i] = b.c[i];
  } else if constexpr (CB == 8) {
    union { uint2 c[NB / 8]; T e[V]; } b;
#pragma unroll
    for (int i = 0; i < V; i += 2)
      wr2(&b.e[i], x[i], x[i + 1]);
    uint2 *d = reinterpret_cast<uint2 *>(p);
#pragma unroll
    for (int i = 0; i < NB / 8; ++i)
      d[i] = b.c[i];
  } else {
#pragma unroll
    for (int i = 0; i < V; i += 2)
      wr2(p + i, x[i], x[i + 1]);
  }
}

// Bit copy of V adjacent elements (pass-through lanes).
template <class T, int V>
__device__ __forceinline__ void cpv(T *dst, const T *src) {
  constexpr int NB = V * int(sizeof(T));
  constexpr int CB = chunk_of<NB>::bytes;
  if constexpr (CB == 16) {
    uint4 *d = reinterpret_cast<uint4 *>(dst);
    const uint4 *s = reinterpret_cast<const uint4 *>(src);
#pragma unroll
    for (int i = 0; i < NB / 16; ++i)
      d[i] = s[i];
  } else if constexpr (CB == 8) {
    uint2 *d = reinterpret_cast<uint2 *>(dst);
    const uint2 *s = reinterpret_cast<const uint2 *>(src);
#pragma unroll
    for (int i = 0; i < NB / 8; ++i)
      d[i] = s[i];
  } else {
#pragma unroll
    for (int i = 0; i < V; i += 2)
      cp2(dst + i, src + i);
  }
}

// Balanced binary tree over the V squares of `x`, lower index on the left -- the same shape
// (and therefore the same bits) as rope_sm90's per-thread pair add followed by the ascending-offset
// warp butterfly, restricted to one lane's contiguous V-element block.
template <int V>
__device__ __forceinline__ float sumsq_tree(const float (&x)[V]) {
  float a[V];
#pragma unroll
  for (int i = 0; i < V; ++i)
    a[i] = __fmul_rn(x[i], x[i]);
#pragma unroll
  for (int span = 1; span < V; span *= 2)
#pragma unroll
    for (int g = 0; g + span < V; g += 2 * span)
      a[g] = __fadd_rn(a[g], a[g + span]);
  return a[0];
}

} // namespace rope_sm100
