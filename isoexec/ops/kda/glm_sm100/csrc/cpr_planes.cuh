// SPDX-License-Identifier: Apache-2.0
// The carry plane (SC) of the state planes, shared by the CPR record families (kda/glm_sm100, gdn/gdn_cpr_sm100).
// THIS FILE IS RESTATED, NOT INCLUDED, ACROSS PACKAGES (a kernel's identity is the digest of its own csrc tree): the copy in
// each package must be byte-identical, which ops/cpr_record/tests/test_spec_cpu.py enforces.
//
// The fp32 state S (128 x 128) is held as SH = rn(S) (bf16, the operand image), SL = the low 16 bits of S's pattern and
// SC = one carry bit per element, so S = ((SH - SC) << 16) | SL bit for bit. SC is a bitmap of 8 x 8 blocks over
// (r, c) = (key / d index, value index): block (r / 8, c / 8) is the 64-bit word (c / 8) * 16 + r / 8 of the plane, and
// bit (r % 8) * 8 + (c % 8) of that word is the element's carry (little endian). One element's carry is a shift of
// one word, whichever thread holds it (`carry_bit`). A warp holding an mma accumulator fragment writes whole words:
// a ballot per accumulator element, a bit spread (lane 4 g + t -> bit 8 g + 2 t) and, for a fragment whose rows are the
// VALUE index, an 8 x 8 bit transpose. Pure bit manipulation; no arithmetic enters.
#pragma once
#include <stdint.h>

namespace cpr_planes {

constexpr int SC_BYTES = 8 * 16 * 16;   // 256 block words

__device__ __forceinline__ uint64_t spread32(uint32_t x) {   // bit i -> bit 2 i
  uint64_t v = x;
  v = (v | (v << 16)) & 0x0000FFFF0000FFFFULL;
  v = (v | (v << 8)) & 0x00FF00FF00FF00FFULL;
  v = (v | (v << 4)) & 0x0F0F0F0F0F0F0F0FULL;
  v = (v | (v << 2)) & 0x3333333333333333ULL;
  v = (v | (v << 1)) & 0x5555555555555555ULL;
  return v;
}

__device__ __forceinline__ uint64_t transpose8x8(uint64_t x) {   // bit 8 a + b -> bit 8 b + a
  uint64_t t = (x ^ (x >> 7)) & 0x00AA00AA00AA00AAULL;
  x = x ^ t ^ (t << 7);
  t = (x ^ (x >> 14)) & 0x0000CCCC0000CCCCULL;
  x = x ^ t ^ (t << 14);
  t = (x ^ (x >> 28)) & 0x00000000F0F0F0F0ULL;
  return x ^ t ^ (t << 28);
}

__device__ __forceinline__ const uint64_t *sc_words(const char *sc) { return reinterpret_cast<const uint64_t *>(sc); }

// The carry of element (r, c).
__device__ __forceinline__ uint32_t carry_bit(const char *sc, int r, int c) {
  const uint64_t w = sc_words(sc)[(c >> 3) * 16 + (r >> 3)];
  return uint32_t(w >> (((r & 7) << 3) | (c & 7))) & 1u;
}

// The four carries of one lane's accumulator fragment, rows = r (row g and g + 8 of the 16-row tile `rtile`), columns =
// c (2 t, 2 t + 1 of the 8-column tile `ctile`): c[2 eh + x] is the element at row g + 8 eh, column 2 t + x.
__device__ __forceinline__ void carry_frag(const char *sc, int rtile, int ctile, int lane, uint32_t (&c)[4]) {
  const int g = lane >> 2, t = lane & 3;
  const uint4 w = *reinterpret_cast<const uint4 *>(sc + (int64_t(ctile) * 16 + 2 * rtile) * 8);   // blocks 2 rtile, 2 rtile + 1
  const uint64_t w0 = (uint64_t(w.y) << 32) | w.x, w1 = (uint64_t(w.w) << 32) | w.z;
  const int sh = g * 8 + 2 * t;
  c[0] = uint32_t(w0 >> sh) & 1u;
  c[1] = uint32_t(w0 >> (sh + 1)) & 1u;
  c[2] = uint32_t(w1 >> sh) & 1u;
  c[3] = uint32_t(w1 >> (sh + 1)) & 1u;
}

// Store the carries of a fragment (all 32 lanes together). TRANSPOSED = false: the fragment's rows are r (tile `a` =
// r / 16 of 16 rows, `b` = c / 8). TRANSPOSED = true: its rows are the value index c (tile `b` = c / 16 of 16 rows) and its
// columns r (`a` = r / 8). `m[e]` is this lane's carry for element e, as in `carry_frag`.
template <bool TRANSPOSED>
__device__ __forceinline__ void store_carry_frag(char *sc, int a, int b, int lane, const uint32_t (&m)[4]) {
  uint64_t w0 = spread32(__ballot_sync(0xffffffffu, m[0])) | (spread32(__ballot_sync(0xffffffffu, m[1])) << 1);
  uint64_t w1 = spread32(__ballot_sync(0xffffffffu, m[2])) | (spread32(__ballot_sync(0xffffffffu, m[3])) << 1);
  __syncwarp();   // every lane has read the blocks it is about to replace (a close folds in place)
  if (lane == 0) {
    uint64_t *w = reinterpret_cast<uint64_t *>(sc);
    if (!TRANSPOSED) {   // blocks (2 a, b), (2 a + 1, b)
      *reinterpret_cast<uint4 *>(w + b * 16 + 2 * a) =
          make_uint4(uint32_t(w0), uint32_t(w0 >> 32), uint32_t(w1), uint32_t(w1 >> 32));
    } else {             // blocks (a, 2 b), (a, 2 b + 1)
      w0 = transpose8x8(w0);
      w1 = transpose8x8(w1);
      w[(2 * b) * 16 + a] = w0;
      w[(2 * b + 1) * 16 + a] = w1;
    }
  }
}

}  // namespace cpr_planes
