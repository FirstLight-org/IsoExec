// SPDX-License-Identifier: Apache-2.0
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <atomic>

#include "slab.cuh"

namespace gdn_cpr_sm100 {
namespace {

template <bool SAVE, bool DECODE, int PLAN>
__global__ __launch_bounds__(THREADS) void gdn_kernel(Params p) {
  constexpr bool DEAL = (PLAN & 1) != 0, WIDTHS = (PLAN & 2) != 0;
  // Bit 2 is the phase-2 `@p` predication and it is a DECODE bit: it costs a prefill
  // kernel about 1,200 bytes of extra local frame (the partial-chunk instantiation
  // hoists four tiles' fragments where the branchy form hoisted one) and a prefill
  // runs at most one partial chunk per sequence.
  constexpr bool PRED = DECODE && (PLAN & 4) != 0;
  extern __shared__ __align__(1024) char raw[];
  Smem &sm = *reinterpret_cast<Smem *>(raw);
  const int item = blockIdx.x;
  const int nn = item / p.hv, h = item - nn * p.hv;
  const int n = p.order ? p.order[nn] : nn;
  const int cs = DECODE ? n : p.cu[n];
  const int len = DECODE ? 1 : p.cu[n + 1] - cs;
  const int si = p.slot_in_of(n), so = p.slot_out_of(n);
  const int pos_in = si >= 0 ? p.st.pos(si, h)[0] : 0;
  const int j0 = pos_in & (C - 1);
  const int nch = chunks_of(len, j0);

  if (nch == 0) {
    if (so >= 0 && so != si) move_record(p, h, si, so, pos_in);
    return;
  }

  float ST[8][2][4];
  for (int k = 0; k < nch; ++k) {
    const ChunkGeo g = chunk_geo(n, k, cs, len, j0);
    const bool more = k + 1 < nch;
    // The record's S^ image serves when nothing will consume the fp32 state: a single incomplete chunk
    // that stays in its own slot. That is 63 decode steps out of 64, and it reads 32 KB per head instead
    // of 96 KB. Any later chunk follows a complete one, so the register state is then always valid.
    const bool sh_cached = k == 0 && !state_in_regs(g, si, so, more) && si >= 0;
    // ALL: this call evaluates the WHOLE chunk, which is every chunk of a prefill but the last. Both
    // halves are instantiated for it with every row test folded away -- a runtime predicate between two
    // dependent MMAs stops ptxas interleaving the chains, and that cost more than the work the predicates
    // save (chunk.cuh, slab.cuh; the l2 norm is where it was first measured).
    const SlabTiles tl{sm.k(), sm.q(), sm.w(), sm.u(), sm.m(), sm.sh(), sm.dt(),
                       SlabT{sm.slab().dl}, &sm.gt()};
    const bool all_rows = g.r_lo == 0 && g.r_hi == C;
    if (all_rows) {
      wy_chunk<SAVE, true, DEAL, WIDTHS>(p, sm, g, h, si, so, sh_cached);
      __syncthreads();
      slab_step<SAVE, true, false>(p, tl, g, h, si, so, more, sh_cached, ST);
    } else {
      wy_chunk<SAVE, false, DEAL, WIDTHS>(p, sm, g, h, si, so, sh_cached);
      __syncthreads();
      slab_step<SAVE, false, PRED>(p, tl, g, h, si, so, more, sh_cached, ST);
    }
    __syncthreads();
  }
  if (so >= 0 && threadIdx.x < NPOS) p.st.pos(so, h)[threadIdx.x] = pos_in + len;
}

// FLA's chunk numbering for the trainer `h` save: sequence n owns
// [chunk_base[n], chunk_base[n] + ceil(len / 64)). One thread; only save_states needs it.
__global__ void chunk_base_kernel(const int *cu, int nseq, int *base) {
  if (threadIdx.x != 0) return;
  int acc = 0;
  for (int n = 0; n < nseq; ++n) {
    base[n] = acc;
    acc += chunks_of(cu[n + 1] - cu[n], 0);
  }
}

// 174 KB of dynamic shared memory is far above the 48 KB default, so the opt-in is mandatory -- and it is
// a PER-(kernel, device) attribute. The bitmask therefore lives inside THIS function template, one
// instantiation per (SAVE, DECODE) kernel, with one bit per device ordinal. A single process-wide flag, or
// a helper whose template parameter deduces to the same function type for all three kernels, leaves every
// kernel but the first -- and every device but the first -- at the default and the launch fails with
// cudaErrorInvalidValue. Both mistakes were made and caught here; the tensor-parallel case is exactly the
// second one.
template <bool SAVE, bool DECODE, int PLAN>
void launch(const Params &p, int items, cudaStream_t stream) {
  auto *fn = gdn_kernel<SAVE, DECODE, PLAN>;
  static std::atomic<uint64_t> raised{0};
  int device = 0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  const uint64_t bit = uint64_t(1) << (device & 63);
  if (!(raised.load(std::memory_order_acquire) & bit)) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void *>(fn),
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, int(sizeof(Smem))));
    raised.fetch_or(bit, std::memory_order_release);
  }
  fn<<<items, THREADS, sizeof(Smem), stream>>>(p);
  C10_CUDA_CHECK(cudaGetLastError());
}

}  // namespace

void core_launch(const Params &p, bool save, bool decode, int plan, int items, cudaStream_t stream) {
  if (items <= 0) return;
  if (plan < 0) plan = decode ? 4 : 3;
  if (decode) {
    switch (plan) {
      case 0: launch<false, true, 0>(p, items, stream); break;
      case 1: launch<false, true, 1>(p, items, stream); break;
      case 3: launch<false, true, 3>(p, items, stream); break;
      case 4: launch<false, true, 4>(p, items, stream); break;
      default: launch<false, true, 7>(p, items, stream); break;
    }
  } else if (save) {
    launch<true, false, 3>(p, items, stream);
  } else if ((plan & 3) == 0) {
    launch<false, false, 0>(p, items, stream);
  } else if ((plan & 3) == 1) {
    launch<false, false, 1>(p, items, stream);
  } else {
    launch<false, false, 3>(p, items, stream);
  }
}

void chunk_base_launch(const int *cu, int nseq, int *base, cudaStream_t stream) {
  chunk_base_kernel<<<1, 32, 0, stream>>>(cu, nseq, base);
  C10_CUDA_CHECK(cudaGetLastError());
}

int64_t core_smem_bytes() { return int64_t(sizeof(Smem)); }

}  // namespace gdn_cpr_sm100
