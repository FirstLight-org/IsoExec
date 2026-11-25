// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <ATen/ATen.h>
#include <c10/util/Exception.h>
#include <cstdint>

namespace norms_sm100 {

// Changing this table changes the arithmetic and requires a recipe version bump.
struct Lanes {
  int L, K;
};

inline bool fwd_supported(int64_t n) { return n > 0 && (n % 256 == 0 || n == 128 || n == 384); }

inline Lanes fwd_lanes(int64_t n) {
  if (n % 256 == 0)
    return {32, int(n / 256)};
  TORCH_CHECK(n == 128 || n == 384, "norms_sm100: unsupported width ", n,
              " (forward recipe v1 covers N % 256 == 0, 128 and 384)");
  return {16, int(n / 128)};
}

inline Lanes bwd_lanes(int64_t n) {
  if (n % 1024 == 0)
    return {128, int(n / 1024)};
  if (n % 256 == 0)
    return {32, int(n / 256)};
  TORCH_CHECK(n == 128 || n == 384, "norms_sm100: unsupported width ", n);
  return {16, int(n / 128)};
}

inline Lanes fwd_lanes_v2(int64_t n, int lmax) {
  TORCH_CHECK(fwd_supported(n) && n % 8 == 0, "norms_sm100: unsupported width ", n, " for recipe v2");
  int64_t e = n / 8; // leaves per row; L * K == e
  int L = 1;
  while (L * 2 <= lmax && e % int64_t(L * 2) == 0)
    L *= 2;
  return {L, int(e / L)};
}

inline int fwd_v2_lmax(int64_t n) {
  switch (n) {
  case 128:
  case 384:
  case 512:
  case 768:
  case 1536:
  case 2048:
  case 4096:
  case 5120:
  case 7168:
    return 128; // measured; see the table above. The rule then gives 16/16/64/32/128/128/128.
  default:
    return 32; // unswept width: v2 resolves to v1's own lanes rather than to an unmeasured fold
  }
}

inline int fwd_recipe_default(int64_t n) {
  switch (n) {
  case 1536:
  case 4096:
  case 5120:
  case 7168:
    return 2;
  default:
    return 1;
  }
}

inline int fwd_recipe(int64_t n) { return fwd_recipe_default(n); }

inline Lanes fwd_lanes_active(int64_t n) {
  if (fwd_recipe(n) != 2)
    return fwd_lanes(n);
  int lmax = fwd_v2_lmax(n);
  return fwd_lanes_v2(n, lmax);
}

enum FwdVar {
  VAR_KEEP = 0,      // all 8K values stay in registers; the weight row loads in the epilogue
  VAR_PREFETCH = 1,  // as KEEP, plus the weight row loads BEFORE the fold (one round trip, not two)
  VAR_STREAM = 2,    // x is read a second time in the epilogue instead of being held (fewest regs)
  VAR_STREAM_PF = 3, // stream x AND issue the weight load before the fold
  VAR_KEEP_L2 = 4,   // KEEP, plus a register-free `prefetch.global.L2` of the weight row up front
  VAR_STREAM_L2 = 5, // STREAM, plus that prefetch
};

struct FwdPlan {
  int threads, var;
};

inline int fwd_var_legal(Lanes l, int var) {
  if (l.L > 32) {
    return (var == VAR_PREFETCH || var == VAR_STREAM_PF) ? VAR_PREFETCH : VAR_KEEP;
  }
  if (8 * l.K < 16)
    return VAR_KEEP;
  if (var == VAR_PREFETCH && l.K > 8)
    return VAR_STREAM;
  if (var == VAR_STREAM_PF && l.K > 20)
    return VAR_STREAM;
  if (var == VAR_KEEP_L2 && (l.K < 8 || l.K > 20))
    return l.K < 8 ? VAR_PREFETCH : VAR_STREAM_L2;
  if (var == VAR_STREAM_L2 && l.K < 8)
    return VAR_STREAM;
  return var;
}

inline FwdPlan fwd_plan(Lanes l, int64_t rows) {
  int threads, var;
  const int64_t n = int64_t(l.L) * 8 * l.K;
  {
    int64_t rpc = 1;
    while (rpc < 8 && (rpc * 2) * n <= 2048)
      rpc *= 2;
    while (rpc > 1 && rpc > rows)
      rpc /= 2; // never more row slots than there are rows to fill them
    threads = int(rpc) * l.L;
    while (threads < 32)
      threads *= 2; // a sub-warp CTA still occupies a whole warp slot
    if (l.L > 32)
      var = (rows < 1024 || l.K == 3) ? VAR_PREFETCH : VAR_KEEP; // K == 3 is N = 1536 only: see README round 4
    else if (l.K == 1)
      var = VAR_KEEP; // v[8] is 8 registers; there is nothing to trade
    else if (l.K <= 3)
      var = VAR_PREFETCH; // 46-62 registers: never enough pressure for the extra read to win
    else if (l.K <= 8)
      var = rows < 1024 ? VAR_PREFETCH : VAR_KEEP;
    else if (l.K <= 20)
      var = rows >= 1024 ? VAR_STREAM : VAR_KEEP;
    else
      var = VAR_STREAM; // 24+ chunks: VAR_KEEP spills 40-240 bytes per thread, so never keep
  }
  if (threads < l.L)
    threads = l.L;
  threads -= threads % l.L;
  if (l.L > 32)
    threads = l.L;
  return {threads, fwd_var_legal(l, var)};
}

inline int fwd_threads(Lanes l, int64_t rows) { return fwd_plan(l, rows).threads; }

inline bool aligned16(const at::Tensor &z) {
  return uintptr_t(z.data_ptr()) % 16 == 0 && (z.stride(0) * 2) % 16 == 0 && (z.stride(1) * 2) % 16 == 0;
}

struct Bands {
  int64_t bands;
  int iters;
};

inline Bands bwd_bands(Lanes l, int64_t rows, int64_t n, bool gated) {
  int64_t rpi = l.L <= 32 ? 128 / l.L : 1;
  int64_t units = (rows + rpi - 1) / rpi;
  int64_t target = rows * n / (gated ? 16384 : 65536), p2 = 256;
  while (p2 < 1024 && p2 * 3 / 2 < target)
    p2 *= 2;
  int64_t bands = units < p2 ? units : p2;
  int64_t iters = (units + bands - 1) / bands;
  bands = (units + iters - 1) / iters;
  return {bands, int(iters)};
}

enum BwdVar { BWD_KEEP = 0, BWD_STREAM = 1, BWD_RING = 2 };

inline bool bwd_ring_legal(Lanes l) { return l.L == 128 && l.K <= 5; }

inline int bwd_var(Lanes l, int64_t rows) {
  return (bwd_ring_legal(l) && rows >= 1024) ? BWD_RING : BWD_KEEP;
}

enum FoldMode { FOLD_REGISTERS = 0, FOLD_SHARED = 1 };

struct FoldPlan {
  int threads, cpt, unroll, mode;
};

inline FoldPlan fold_plan(int64_t bands, int64_t width) {
  int threads, cpt = 1, unroll, mode;
  if (bands >= 128) {
    mode = FOLD_SHARED;
    threads = 256;                        // 8 loading warps per 1 summing warp
    unroll = bands >= 256 ? 256 : 128;    // staged bands; 256 * 32 * 4 = 32 KiB of shared memory
  } else {
    mode = FOLD_REGISTERS;
    threads = 128;
    unroll = 1;
    while (unroll < 64 && unroll * 2 <= bands)
      unroll *= 2; // past the band count the unrolled body never runs and the tail does the work
  }
  if (width % cpt)
    cpt = 1;
  return {threads, cpt, unroll, mode};
}

} // namespace norms_sm100
