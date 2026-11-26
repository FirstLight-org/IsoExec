// SPDX-License-Identifier: Apache-2.0
// The fp32 state of a record, held as three planes (record.cuh "State planes"): SH = rn(S) as bf16 [v][k] (the operand
// image every step reads), SL = the low 16 bits of S's pattern [k][v], SC = one carry bit per element (cpr_planes.cuh; the
// magnitude of rn(S) is one above the truncated top half). S is ((SH - SC) << 16) | SL, bit for bit.
// Everything here is bit manipulation of values the recipe already produces (`rn` is the recipe's bf16 narrowing); no
// arithmetic enters. Not-a-number patterns are not preserved (their top half is rn's quiet pattern); a state that holds
// one has already poisoned every output.
//
// All functions work on one lane's accumulator fragment tile (16 k x 8 v of an mma.m16n8 accumulator,
// a[e]: k = 16 ktile + lane / 4 + 8 (e / 2), v = 8 ntile + 2 (lane % 4) + (e % 2)) and must be called by all
// 32 lanes of the warp together (the carries are a warp ballot).
#pragma once
#include "common.cuh"
#include "cpr_planes.cuh"
#include "record.cuh"

namespace kda_sm100 {

// The element's SH value in the global SH plane [v][k].
__device__ __forceinline__ uint16_t sh_global(const char *rec, int k, int v) {
  return reinterpret_cast<const uint16_t *>(rec + Rec::SH)[v * D + k];
}

// a[0..3] of tile (ktile, ntile) from its four SH values (sh[e]) and the SL / SC planes of `rec`.
__device__ __forceinline__ void s_join_tile(float (&a)[4], const uint16_t (&sh)[4], const char *rec, int ktile, int ntile,
                                            int lane) {
  const int g = lane >> 2, t = lane & 3;
  uint32_t c[4];
  cpr_planes::carry_frag(rec + Rec::SC, ktile, ntile, lane, c);
  #pragma unroll
  for (int eh = 0; eh < 2; ++eh) {
    const int k = 16 * ktile + g + 8 * eh, v = 8 * ntile + 2 * t;
    const uint32_t lo = *reinterpret_cast<const uint32_t *>(rec + Rec::SL + (int64_t(k) * D + v) * 2);
    a[2 * eh] = __uint_as_float(((uint32_t(sh[2 * eh]) - c[2 * eh]) << 16) | (lo & 0xffffu));
    a[2 * eh + 1] = __uint_as_float(((uint32_t(sh[2 * eh + 1]) - c[2 * eh + 1]) << 16) | (lo >> 16));
  }
  asm volatile("" ::: "memory");   // one tile's loads at a time: the close runs inside a register-capped kernel
}

// The same with the SH values read from the record's global plane.
__device__ __forceinline__ void s_load_tile(float (&a)[4], const char *rec, int ktile, int ntile, int lane) {
  const int g = lane >> 2, t = lane & 3;
  uint16_t sh[4];
  #pragma unroll
  for (int e = 0; e < 4; ++e) sh[e] = sh_global(rec, 16 * ktile + g + 8 * (e >> 1), 8 * ntile + 2 * t + (e & 1));
  s_join_tile(a, sh, rec, ktile, ntile, lane);
}

// SL and SC of tile (ktile, ntile) from the accumulators (the SH plane is written by the caller's own
// transposed copy of rn(a); the carries are computed from the same narrowing).
__device__ __forceinline__ void s_store_tile(char *rec, const float (&a)[4], int ktile, int ntile, int lane) {
  const int g = lane >> 2, t = lane & 3;
  uint32_t m[4];
  #pragma unroll
  for (int eh = 0; eh < 2; ++eh) {
    const int k = 16 * ktile + g + 8 * eh, v = 8 * ntile + 2 * t;
    const uint32_t b0 = __float_as_uint(a[2 * eh]), b1 = __float_as_uint(a[2 * eh + 1]);
    *reinterpret_cast<uint32_t *>(rec + Rec::SL + (int64_t(k) * D + v) * 2) = (b0 & 0xffffu) | (b1 << 16);
    m[2 * eh] = uint32_t(rn(a[2 * eh]) != (b0 >> 16));
    m[2 * eh + 1] = uint32_t(rn(a[2 * eh + 1]) != (b1 >> 16));
  }
  cpr_planes::store_carry_frag<false>(rec + Rec::SC, ktile, ntile, lane, m);
  asm volatile("" ::: "memory");
}

}  // namespace kda_sm100
