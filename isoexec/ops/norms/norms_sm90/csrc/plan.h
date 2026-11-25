// SPDX-License-Identifier: Apache-2.0
// Host-side recipe table and launch-plan helpers (no device code).
#pragma once
#include <ATen/ATen.h>
#include <c10/util/Exception.h>
#include <cstdint>

namespace norms_sm90 {

// RECIPE v1 (forward): lanes per row L and chunks per lane K are a function of N only.
// Changing this table changes the arithmetic and requires a recipe version bump.
struct Lanes {
  int L, K;
};

inline bool fwd_supported(int64_t n) { return n > 0 && (n % 256 == 0 || n == 128 || n == 384); }

inline Lanes fwd_lanes(int64_t n) {
  if (n % 256 == 0)
    return {32, int(n / 256)};
  TORCH_CHECK(n == 128 || n == 384, "norms_sm90: unsupported width ", n,
              " (forward recipe v1 covers N % 256 == 0, 128 and 384)");
  return {16, int(n / 128)};
}

// Backward lanes: a shape-dependent execution choice; fixed for identical inputs.
inline Lanes bwd_lanes(int64_t n) {
  if (n % 1024 == 0)
    return {128, int(n / 1024)};
  if (n % 256 == 0)
    return {32, int(n / 256)};
  TORCH_CHECK(n == 128 || n == 384, "norms_sm90: unsupported width ", n);
  return {16, int(n / 128)};
}

// EXECUTION PLAN (forward): threads per CTA (rows per block = threads / L). Never enters the
// arithmetic. Narrow rows pack 8 rows into a 128-thread CTA once there are enough of them.
inline int fwd_threads(Lanes l, int64_t rows) { return l.L >= 32 || rows < 256 ? 32 : 128; }

// A 16-byte-aligned gate row (base and both strides) may use 128-bit loads; otherwise the kernel
// reads it with 32-bit loads (the host has already ensured 4-byte alignment).
inline bool aligned16(const at::Tensor &z) {
  return uintptr_t(z.data_ptr()) % 16 == 0 && (z.stride(0) * 2) % 16 == 0 && (z.stride(1) * 2) % 16 == 0;
}

struct Bands {
  int64_t bands;
  int iters;
};

// EXECUTION PLAN (backward): retain the existing measured band plan. Each 128-thread CTA
// iterates `iters` times over 128/L rows (one row for L = 128). The deterministic dW change
// does not retune this plan; its <=1024 bands bound FP32 partial storage to 4096*N bytes.
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

} // namespace norms_sm90
