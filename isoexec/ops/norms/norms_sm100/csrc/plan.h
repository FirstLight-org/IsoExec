// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <ATen/ATen.h>
#include <c10/util/Exception.h>
#include <cstdint>

namespace norms_sm100 {

// Execution geometry of one row: L threads, each holding K 16-byte loads (N = 8 * L * K). It decides which thread
// loads which segment and nothing else. The numerical schedule (segments of S consecutive columns from
// isoexec/ops/schedule_policy.py, adjacent tree; common.cuh::segment_tree) does not read L or K, so a different
// geometry gives the same bits.
struct Lanes {
  int L, K;
};

inline bool fwd_supported(int64_t n) { return n > 0 && (n % 256 == 0 || n == 128 || n == 384); }

inline Lanes fwd_lanes(int64_t n) {
  switch (n) { // widths timed with one row per 64- or 128-thread CTA
  case 1536:
    return {64, 3};
  case 4096:
  case 5120:
  case 7168:
    return {128, int(n / 1024)};
  default:
    break;
  }
  if (n % 256 == 0)
    return {32, int(n / 256)};
  TORCH_CHECK(n == 128 || n == 384, "norms_sm100: unsupported width ", n,
              " (the forward covers N % 256 == 0, 128 and 384)");
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

enum BwdVar { BWD_KEEP = 0, BWD_STREAM = 1, BWD_RING = 2 };

inline bool bwd_ring_legal(Lanes l) { return l.L == 128 && l.K <= 5; }

inline int bwd_var(Lanes l, int64_t rows) {
  return (bwd_ring_legal(l) && rows >= 1024) ? BWD_RING : BWD_KEEP;
}

} // namespace norms_sm100
