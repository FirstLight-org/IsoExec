// SPDX-License-Identifier: Apache-2.0
// Decode-step row bookkeeping for GDN CPR pools, in one launch per step (the same kernel as gdn_cpr_sm100).
//
// Per pool it computes what the decode step otherwise runs as an ATen/Triton chain per layer:
//   rows   = slot2row[clamp(slots, 0, map_n - 1)]
//   lane32 = where(rows32 > 0, rows32, -1)                 (the null row 0 is the core's NULL lane)
//   clock += 1; last_used[rows] = clock                    (the LRU stamp)
// Index arithmetic only: no value is rounded. The LRU stamp is written before the recurrent decode
// rather than after it; nothing in between reads `clock` or `last_used` (the host reads them only
// between steps) and `slot2row` is written only by the host between steps, so the state after the
// step is the same. No atomics: duplicate rows (the null row 0 of padded lanes) store the same value.
//
// One CTA per pool, up to kMaxPools pools per launch, all sharing the step's slot ids.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

namespace gdn_cpr {
namespace {

constexpr int kMaxPools = 40;

struct PoolRows {
  const int64_t *map;
  int64_t map_n;
  int *rows32;
  int *lane32;
  int64_t *clock;
  int64_t *last_used;
};

struct BatchRows {
  PoolRows p[kMaxPools];
};

template <class S>
__global__ void decode_rows_kernel(const S *__restrict__ slots, int64_t sstride, int n,
                                   const __grid_constant__ BatchRows B) {
  const PoolRows &P = B.p[blockIdx.x];
  // Every thread reads the old clock before thread 0 advances it (the barrier below).
  const int64_t c = P.clock[0] + 1;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    int64_t s = static_cast<int64_t>(slots[int64_t(i) * sstride]);
    s = s < 0 ? 0 : (s > P.map_n - 1 ? P.map_n - 1 : s);
    const int64_t r = P.map[s];
    const int r32 = static_cast<int>(r);
    P.rows32[i] = r32;
    P.lane32[i] = r32 > 0 ? r32 : -1;
    P.last_used[r] = c;
  }
  __syncthreads();
  if (threadIdx.x == 0)
    P.clock[0] = c;
}

PoolRows pool_of(const at::Tensor &map, at::Tensor &rows32, at::Tensor &lane32, const at::Tensor &clock,
                 const at::Tensor &last_used, int64_t n) {
  TORCH_CHECK(map.is_cuda() && map.dim() == 1 && map.scalar_type() == at::kLong && map.numel() > 0 &&
                  map.stride(0) == 1,
              "gdn_cpr_sm90.decode_rows: slot2row must be a non-empty contiguous CUDA int64 vector");
  TORCH_CHECK(rows32.scalar_type() == at::kInt && lane32.scalar_type() == at::kInt && rows32.numel() == n &&
                  lane32.numel() == n && rows32.is_contiguous() && lane32.is_contiguous(),
              "gdn_cpr_sm90.decode_rows: rows32/lane32 must be contiguous int32 [n]");
  TORCH_CHECK(clock.scalar_type() == at::kLong && clock.numel() == 1 && last_used.scalar_type() == at::kLong &&
                  last_used.dim() == 1 && last_used.is_contiguous(),
              "gdn_cpr_sm90.decode_rows: clock must be an int64 scalar and last_used a contiguous int64 vector");
  return PoolRows{map.data_ptr<int64_t>(), map.numel(), rows32.data_ptr<int>(), lane32.data_ptr<int>(),
                  clock.data_ptr<int64_t>(), last_used.data_ptr<int64_t>()};
}

}  // namespace

// maps / rows32s / lane32s / clocks / last_useds: one entry per pool, in the same order.
void decode_rows(const at::Tensor &slots, const std::vector<at::Tensor> &maps, std::vector<at::Tensor> rows32s,
                 std::vector<at::Tensor> lane32s, const std::vector<at::Tensor> &clocks,
                 const std::vector<at::Tensor> &last_useds) {
  TORCH_CHECK(slots.is_cuda() && slots.dim() == 1 &&
                  (slots.scalar_type() == at::kInt || slots.scalar_type() == at::kLong),
              "gdn_cpr_sm90.decode_rows: slots must be a 1-D CUDA int32/int64 tensor");
  const size_t L = maps.size();
  TORCH_CHECK(rows32s.size() == L && lane32s.size() == L && clocks.size() == L && last_useds.size() == L,
              "gdn_cpr_sm90.decode_rows: one map/rows32/lane32/clock/last_used per pool");
  const int64_t n = slots.numel();
  if (n == 0 || L == 0)
    return;
  const c10::cuda::CUDAGuard guard(slots.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  // Lanes beyond one block loop in-thread: one CTA per pool, so each clock advances exactly once.
  const int threads = n >= 1024 ? 1024 : (n <= 32 ? 32 : int((n + 31) / 32 * 32));
  for (size_t base = 0; base < L; base += kMaxPools) {
    const int cnt = int(L - base < size_t(kMaxPools) ? L - base : size_t(kMaxPools));
    BatchRows B{};
    for (int j = 0; j < cnt; ++j)
      B.p[j] = pool_of(maps[base + j], rows32s[base + j], lane32s[base + j], clocks[base + j],
                       last_useds[base + j], n);
    if (slots.scalar_type() == at::kInt)
      decode_rows_kernel<int><<<cnt, threads, 0, stream>>>(slots.data_ptr<int>(), slots.stride(0), int(n), B);
    else
      decode_rows_kernel<int64_t><<<cnt, threads, 0, stream>>>(slots.data_ptr<int64_t>(), slots.stride(0),
                                                                int(n), B);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

}  // namespace gdn_cpr
