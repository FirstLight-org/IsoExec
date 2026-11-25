// SPDX-License-Identifier: Apache-2.0
// Prefill / chunked prefill / trainer forward: one workspace allocation, the plan, then the two passes
// (wy.cu, sweep.cu) or the fused prefill (fused.cu) -- the mode an execution choice of the host's.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cub/block/block_scan.cuh>

#include <map>
#include <string>
#include <vector>

#include "prefill.cuh"
#include "launch.h"

namespace gdn_cpr {
int prefill_groups(int nseq, int64_t max_chunks, int roles_per_group, int sms);
void launch_wy(const PrefillParams &p, int64_t max_chunks, cudaStream_t stream);
void launch_sweep(const PrefillParams &p, int64_t max_chunks, int rows, bool on_demand, int ctas_per_sm,
                  bool one_group, cudaStream_t stream);
void launch_fused(const PrefillParams &p, cudaStream_t stream);
void launch_prepared(const PrefillParams &p, cudaStream_t stream);
}  // namespace gdn_cpr

namespace {

constexpr int ORDER_BUCKETS = 256;   // the fused order sorts by chunk count, the longest bucket 255+

constexpr int PLAN_THREADS = 1024;

// The plan, one CTA: each sequence's input position (from its input slot, before any pass writes a
// record), its chunk count and the exclusive prefix of the counts (the dense chunk order). A malformed
// pack -- cu_seqlens not ending at T, a negative length, more chunks than the workspace holds -- stops the
// kernel with a device-side trap rather than running on it. Fused mode: also the order the items are
// taken in -- the sequences by decreasing chunk count (ties in any order: it is only a schedule).
__global__ __launch_bounds__(PLAN_THREADS) void plan_kernel(const __grid_constant__ gdn_cpr::PrefillParams p,
                                                            int64_t max_chunks, bool sequential) {
  using Scan = cub::BlockScan<int, PLAN_THREADS>;
  __shared__ typename Scan::TempStorage scan;
  __shared__ int carry;
  __shared__ int bucket[ORDER_BUCKETS];
  if (threadIdx.x == 0 && p.unit_next) *p.unit_next = 0;
  if (p.zeros)
    for (int x = threadIdx.x; x < gdn_cpr::ZERO_BYTES / 16; x += PLAN_THREADS)
      reinterpret_cast<uint4 *>(p.zeros)[x] = make_uint4(0u, 0u, 0u, 0u);
  if (threadIdx.x == 0) {
    carry = 0;
    if (p.cu[0] != 0 || p.cu[p.nseq] != p.T) __trap();
  }
  __syncthreads();
  for (int base = 0; base < p.nseq; base += PLAN_THREADS) {
    const int n = base + int(threadIdx.x);
    int nch = 0;
    if (n < p.nseq) {
      const int len = p.cu[n + 1] - p.cu[n];
      if (len < 0) __trap();
      const int si = p.slot_in_of(n);
      const int pos = si >= 0 ? *p.st.pos(si, 0) : 0;
      p.pos[n] = pos;
      nch = gdn_cpr::chunks_of(len, pos & (gdn_cpr::C - 1));
    }
    int before, total;
    Scan(scan).ExclusiveSum(nch, before, total);
    if (n < p.nseq) {
      p.chunk0[n] = carry + before;
      if (carry + before + nch > max_chunks) __trap();
      for (int c = 0; c < nch; ++c) p.seq_of[carry + before + c] = n;
    }
    __syncthreads();
    if (threadIdx.x == 0) carry += total;
    __syncthreads();
  }
  if (threadIdx.x == 0) p.chunk0[p.nseq] = carry;
  for (int64_t c = carry + threadIdx.x; c < max_chunks; c += PLAN_THREADS) p.seq_of[c] = -1;
  if (!sequential) {
    if (p.nseq > gdn_cpr::OUTPUT_ORDER_MAX_SEQS || p.groups == 1) return;
    __syncthreads();  // every chunk0 entry and seq_of entry is complete
    // Precompute the same group-round, k-major output order the workers previously searched for on
    // every unit. A chunk's rank is the earlier rounds, all earlier levels of this round, and the
    // earlier sequences present at this level. Different chunks have distinct ranks, including empty
    // sequences and unequal lengths. Every head and value half reuses this one small permutation.
    for (int c = threadIdx.x; c < carry; c += PLAN_THREADS) {
      const int n = p.seq_of[c], k = c - p.chunk0[n];
      const int n0 = (n / p.groups) * p.groups, n1 = min(n0 + p.groups, p.nseq);
      int rank = p.chunk0[n0];
      for (int peer = n0; peer < n1; ++peer) {
        const int nch = p.chunk0[peer + 1] - p.chunk0[peer];
        rank += min(nch, k) + int(peer < n && nch > k);
      }
      p.order[rank] = c;
    }
    return;
  }
  // counting sort, descending: bucket b holds chunk count ORDER_BUCKETS - 1 - b (the last: that many or more)
  for (int b = threadIdx.x; b < ORDER_BUCKETS; b += PLAN_THREADS) bucket[b] = 0;
  __syncthreads();
  auto key = [&](int n) { return ORDER_BUCKETS - 1 - min(p.chunk0[n + 1] - p.chunk0[n], ORDER_BUCKETS - 1); };
  for (int n = threadIdx.x; n < p.nseq; n += PLAN_THREADS) atomicAdd(&bucket[key(n)], 1);
  __syncthreads();
  if (threadIdx.x == 0)
    for (int b = 0, at = 0; b < ORDER_BUCKETS; ++b) {
      const int c = bucket[b];
      bucket[b] = at;
      at += c;
    }
  __syncthreads();
  for (int n = threadIdx.x; n < p.nseq; n += PLAN_THREADS) p.order[atomicAdd(&bucket[key(n)], 1)] = n;
}

struct Carve {
  int64_t bytes = 0;
  int64_t take(int64_t n) {
    const int64_t at = bytes;
    bytes += (n + 255) / 256 * 256;
    return at;
  }
};

}  // namespace

void gdn_cpr_prefill(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, const at::Tensor &a,
                     const at::Tensor &b, const at::Tensor &A_log, const at::Tensor &dt_bias, const at::Tensor &cu,
                     const c10::optional<at::Tensor> &state, const c10::optional<at::Tensor> &slot_in,
                     const c10::optional<at::Tensor> &slot_out, at::Tensor out, double scale, int64_t sweep_rows,
                     int64_t max_chunks, const std::vector<at::Tensor> &save, int64_t schedule, int64_t mode) {
  using namespace gdn_cpr;
  const c10::cuda::CUDAGuard guard(q.device());
  const int64_t T = q.size(0);
  const int nseq = int(cu.size(0)) - 1;
  const bool stateful = state.has_value() && state->defined();
  if (nseq <= 0 || (T <= 0 && !stateful)) return;   // an empty stateful pack still moves its slots
  const int hk = int(q.size(1)), hv = int(v.size(1));
  // The workspace holds the pack's chunks: a bound from the host (exact when it knows the lengths), else
  // the bound for any lengths -- sum_n ceil((j0_n + len_n) / 64) <= T / 64 + 2N.
  const int64_t bound = T / C + 2 * int64_t(nseq);
  TORCH_CHECK(max_chunks <= bound, "gdn_cpr_sm90: max_chunks exceeds the bound for this pack");
  const int64_t chunks = max_chunks >= 0 ? max_chunks : bound;

  TORCH_CHECK(mode >= 0 && mode <= 2, "gdn_cpr_sm90: prefill mode must be 0 (two passes), 1 (fused), or 2 (prepared)");
  const bool fused = mode == 1;
  const bool prepared = mode == 2, sequential = mode != 0;
  // The two passes hand every chunk's tiles over through the workspace; the fused prefill keeps them on
  // chip, including the fp32 state between chunks.
  const int64_t tiles = fused ? 0 : chunks;
  Carve cv;
  const int64_t o_w = cv.take(tiles * hv * TILE_BYTES), o_u = cv.take(tiles * hv * TILE_BYTES),
                o_m = cv.take(tiles * hv * MT_BYTES), o_k = cv.take(tiles * hk * TILE_BYTES),
                o_q = cv.take(tiles * hk * TILE_BYTES), o_gt = cv.take(tiles * hv * GT_BYTES);
  int64_t o_fx = 0, o_fp = 0;
  if (stateful && !fused) {
    o_fx = cv.take(int64_t(nseq) * hv * FX_BYTES);
    o_fp = cv.take(int64_t(nseq) * hv * FP_BYTES);
  }
  const int64_t o_pos = cv.take(int64_t(nseq) * 4), o_chunk0 = cv.take((int64_t(nseq) + 1) * 4),
                o_seq_of = cv.take(chunks * 4), o_order = cv.take((sequential ? int64_t(nseq) : chunks) * 4);
  // Prepared mode retains WY tiles, but its register-state walk needs neither state/delta hand-offs
  // nor cross-CTA coordination. Fully fused mode needs only the small plan.
  const int64_t handoff_chunks = sequential ? 0 : chunks;
  const int64_t o_sh = cv.take(handoff_chunks * hv * D * D * 2), o_d = cv.take(handoff_chunks * hv * D * C * 2),
                o_counter = cv.take(sequential ? 0 : sizeof(int)), o_zero = cv.take(sequential ? 0 : ZERO_BYTES);
  at::Tensor ws = at::empty({cv.bytes}, q.options().dtype(at::kByte));
  char *w = static_cast<char *>(ws.data_ptr());

  PrefillParams p{};
  p.q = static_cast<const uint16_t *>(q.data_ptr());
  p.k = static_cast<const uint16_t *>(k.data_ptr());
  p.v = static_cast<const uint16_t *>(v.data_ptr());
  p.a = static_cast<const uint16_t *>(a.data_ptr());
  p.b = static_cast<const uint16_t *>(b.data_ptr());
  p.qs0 = q.stride(0), p.qs1 = q.stride(1), p.ks0 = k.stride(0), p.ks1 = k.stride(1);
  p.vs0 = v.stride(0), p.vs1 = v.stride(1), p.as0 = a.stride(0), p.as1 = a.stride(1);
  p.bs0 = b.stride(0), p.bs1 = b.stride(1);
  p.A_log = A_log.data_ptr();
  p.dt_bias = dt_bias.data_ptr();
  p.param_bf16 = int(A_log.scalar_type() == at::kBFloat16);
  p.out = static_cast<uint16_t *>(out.data_ptr());
  p.os0 = out.stride(0), p.os1 = out.stride(1);
  p.cu = cu.data_ptr<int>();
  p.nseq = nseq, p.hk = hk, p.hv = hv, p.T = T;
  p.scale = float(scale);
  // With no input state, sum ceil(len / C) == T / C proves that every nonempty sequence consists
  // entirely of complete chunks. The plan checks this bound against the device lengths before WY.
  p.full_chunks = !stateful && T % C == 0 && chunks == T / C;
  p.st = StateRef{stateful ? static_cast<char *>(state->data_ptr()) : nullptr, hv};
  p.slot_in = stateful ? slot_in->data_ptr<int>() : nullptr;
  p.slot_out = stateful ? slot_out->data_ptr<int>() : nullptr;
  p.ws_w = w + o_w, p.ws_u = w + o_u, p.ws_m = w + o_m, p.ws_k = w + o_k, p.ws_q = w + o_q, p.ws_gt = w + o_gt;
  if (stateful) {
    p.ws_fx = w + o_fx;
    p.ws_fp = w + o_fp;
  }
  p.pos = reinterpret_cast<int *>(w + o_pos);
  p.chunk0 = reinterpret_cast<int *>(w + o_chunk0);
  p.seq_of = reinterpret_cast<int *>(w + o_seq_of);
  p.ws_sh = w + o_sh, p.ws_d = w + o_d;
  p.zeros = sequential ? nullptr : reinterpret_cast<uint16_t *>(w + o_zero);
  p.order = reinterpret_cast<int *>(w + o_order);
  if (!save.empty()) {   // trainer mode (ops.trainer_forward): qn, kn, q_rstd, k_rstd, v, A, w, h, v_new
    TORCH_CHECK(save.size() == 9 && !stateful, "gdn_cpr_sm90: the backward's tensors need a stateless call");
    for (const auto &t : save) TORCH_CHECK(t.is_cuda() && t.is_contiguous(), "gdn_cpr_sm90: save tensors: contiguous CUDA");
    // an empty tensor: not saved (w, h, v_new are optional together: the backward then recomputes them)
    auto b16 = [&](int i) { return save[i].numel() ? static_cast<uint16_t *>(save[i].data_ptr()) : nullptr; };
    p.sv = SaveParams{b16(0), b16(1), b16(4), b16(5), b16(6), b16(7), b16(8), save[2].data_ptr<float>(),
                      save[3].data_ptr<float>()};
    TORCH_CHECK(p.sv.qn && p.sv.kn && p.sv.v && p.sv.A && !p.sv.w == !p.sv.h && !p.sv.h == !p.sv.v_new,
                "gdn_cpr_sm90: save tensors: qn, kn, v, A always; w, h, v_new together");
  }
  if (!sequential) {
    p.unit_next = reinterpret_cast<int *>(w + o_counter);
    const int rows = sweep_rows ? int(sweep_rows) : (int64_t(nseq) * hv * 8 <= 128 ? 16 : 32);
    TORCH_CHECK(rows == 16 || rows == 32, "gdn_cpr_sm90: sweep rows must be 16 or 32, got ", rows);
    p.groups = (schedule & 4) ? 1 : prefill_groups(nseq, chunks, hv * (D / rows), device_sms());
  }
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  plan_kernel<<<1, PLAN_THREADS, 0, stream>>>(p, chunks, sequential);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (fused) {
    launch_fused(p, stream);
    return;
  }
  // schedule (tests only, ops._sweep_schedule): bit 0 sweep roles on demand, bit 1 an oversubscribed grid,
  // bit 2 every sequence of a head and slab in one sweep role
  launch_wy(p, chunks, stream);
  if (prepared) {
    launch_prepared(p, stream);
    return;
  }
  launch_sweep(p, chunks, int(sweep_rows), (schedule & 1) != 0, (schedule & 2) ? 8 : 2, (schedule & 4) != 0, stream);
}

std::map<std::string, int64_t> gdn_cpr_record_layout() {
  using R = gdn_cpr::Rec;
  return {{"sh", R::SH}, {"lo", R::LO}, {"tie", R::TIE}, {"format", R::FORMAT}, {"kb", R::KB}, {"vb", R::VB},   {"db", R::DB},       {"xb", R::XB}, {"pb", R::PB},
          {"gb", R::GB}, {"bb", R::BB}, {"pos", R::POS}, {"bytes", R::BYTES}};
}
