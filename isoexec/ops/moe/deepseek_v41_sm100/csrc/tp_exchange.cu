// SPDX-License-Identifier: Apache-2.0
//
// Column all-gather of the TP-sharded experts over NVLink symmetric memory, one launch per call (the PIK owner
// all-reduce's conventions: per-block epochs kept locally, per-block flags in every rank's symmetric array, release /
// acquire at system scope, no host state per call: graph-replay safe).
//
// Region of one rank (symmetric, identical layout on every rank):
//   [0, HEADER)       arrive[kMaxBlocks][kMaxWorld] u32, leave[kMaxBlocks][kMaxWorld] u32
//   [HEADER, ...)     data[slice_bytes]: this rank's slice (the producer may write it there directly: in place)
// plus a LOCAL (not symmetric) epoch[kMaxBlocks] u32. Block b of call e = epoch[b] + 1 owns the row chunk b:
//   1. (copy mode) copy my chunk b of x into data; __syncthreads;
//   2. arrive_t[b][rank] = e for every peer t; wait arrive[b][t] >= e for every t; __syncthreads. Copy mode: release /
//      acquire at system scope (this launch wrote the data); in place: relaxed (the producer kernel's writes are
//      complete before this launch, as in vLLM's custom all-gather);
//   3. pull chunk b of every rank's data into out[:, t C : (t + 1) C] (16-byte vectors, peers staggered by rank);
//   4. __syncthreads; leave_t[b][rank] = e for every owner t (relaxed: my reads are done); wait leave[b][t] >= e for
//      every t: no peer still reads my chunk when this launch ends, so the next producer may overwrite data.
// Two flag round trips per call (as vLLM's custom all-gather); every byte crosses NVLink once (a pull); no arithmetic.
// A wait that exceeds the bound traps (a hang becomes a launch error). Every rank launches the same grid.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cstring>
#include <vector>

namespace isoexec_tp_exchange {
namespace {

constexpr int kThreads = 512;
constexpr int kMaxWorld = 8;
constexpr int kMaxBlocks = 256;
constexpr long long kSpinBound = 1ll << 31;

__device__ __forceinline__ uint32_t ld_acquire_sys(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ uint32_t ld_relaxed(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_relaxed_sys(uint32_t* p, uint32_t v) {
  asm volatile("st.relaxed.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ void st_release_sys(uint32_t* p, uint32_t v) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
// relaxed flags (vLLM's custom all-gather barriers): used where the guarded data was written by an EARLIER kernel
// (in place: the producer's writes are visible once its launch completed) or for the final "reads done" round
__device__ __forceinline__ void wait_geq_relaxed(const uint32_t* p, uint32_t e) {
  long long spins = 0;
  while (int32_t(ld_relaxed(p) - e) < 0) {
    if (++spins > kSpinBound) __trap();
  }
}
// spin with relaxed loads (cheap), then ONE acquire load that reads the flag value (>= e): the acquire synchronizes
// with the release (or, in place, follows the kernel boundary) and orders every later load after it
__device__ __forceinline__ void wait_geq(const uint32_t* p, uint32_t e) {
  long long spins = 0;
  while (int32_t(ld_relaxed(p) - e) < 0) {
    if (++spins > kSpinBound) __trap();
  }
  (void)ld_acquire_sys(p);
}

struct Args {
  char* base[kMaxWorld];      // every rank's region
  uint32_t* epoch;            // local [kMaxBlocks]
  const int4* x;              // [rows, cols] bf16 as 16-byte vectors
  int4* out;                  // [rows, world * cols]
  int world, rank, rows, vcols;  // vcols = cols / 8
  long long header, slice_bytes;
};

__device__ __forceinline__ uint32_t* arrive_of(char* base, int b, int src) {
  return reinterpret_cast<uint32_t*>(base) + (b * kMaxWorld + src);
}
__device__ __forceinline__ uint32_t* leave_of(char* base, int b, int src) {
  return reinterpret_cast<uint32_t*>(base) + kMaxBlocks * kMaxWorld + (b * kMaxWorld + src);
}

__global__ __launch_bounds__(kThreads) void gather_cols_kernel(Args a) {
  const int b = blockIdx.x, nb = gridDim.x;
  const uint32_t e = a.epoch[b] + 1u;
  const int r0 = int((long long)a.rows * b / nb), r1 = int((long long)a.rows * (b + 1) / nb);
  char* mine = a.base[a.rank];
  const long long n = (long long)(r1 - r0) * a.vcols;
  const long long off = (long long)r0 * a.vcols;
  if (a.x != nullptr) {  // copy mode
    int4* mydata = reinterpret_cast<int4*>(mine + a.header);
    for (long long i = threadIdx.x; i < n; i += kThreads) mydata[off + i] = a.x[off + i];
  }
  __syncthreads();
  if (threadIdx.x < a.world) {
    if (a.x != nullptr) {   // copy mode: this launch wrote the data: release / acquire
      st_release_sys(arrive_of(a.base[threadIdx.x], b, a.rank), e);
      wait_geq(arrive_of(mine, b, threadIdx.x), e);
    } else {                // in place: written by an earlier kernel
      st_relaxed_sys(arrive_of(a.base[threadIdx.x], b, a.rank), e);
      wait_geq(arrive_of(mine, b, threadIdx.x), e);   // acquire poll: orders our pulls after the flag
    }
  }
  __syncthreads();
  const int W = a.world, vc = a.vcols;
  for (int t = 0; t < W; ++t) {
    const int src = (a.rank + t) % W;   // stagger the peers across ranks
    const int4* sd = reinterpret_cast<const int4*>(a.base[src] + a.header);
    for (long long i = threadIdx.x; i < n; i += kThreads) {
      const long long rr = r0 + i / vc, cc = i % vc;
      a.out[rr * (long long)W * vc + (long long)src * vc + cc] = sd[off + i];
    }
  }
  __syncthreads();
  if (threadIdx.x < a.world) {
    st_relaxed_sys(leave_of(a.base[threadIdx.x], b, a.rank), e);   // final round: only "my reads are done"
    wait_geq_relaxed(leave_of(mine, b, threadIdx.x), e);
  }
  if (threadIdx.x == 0) a.epoch[b] = e;
}

}  // namespace

void gather_cols(std::vector<int64_t> bases, at::Tensor epoch, int64_t rank, int64_t header, int64_t slice_bytes,
                 at::Tensor x, at::Tensor out, int64_t blocks) {
  const c10::cuda::CUDAGuard guard(x.device());
  const int world = int(bases.size());
  TORCH_CHECK(world >= 1 && world <= kMaxWorld, "tp_exchange: 1..8 ranks");
  TORCH_CHECK(blocks >= 1 && blocks <= kMaxBlocks, "tp_exchange: block count");
  TORCH_CHECK(x.is_contiguous() && x.scalar_type() == at::kBFloat16 && x.dim() == 2, "tp_exchange: x bf16 [R, C]");
  const bool in_place = reinterpret_cast<int64_t>(x.data_ptr()) == bases[rank] + header;
  const int64_t rows = x.size(0), cols = x.size(1);
  TORCH_CHECK(cols % 8 == 0, "tp_exchange: slice width must be a multiple of 8 bf16 (16 bytes)");
  TORCH_CHECK(out.is_contiguous() && out.scalar_type() == at::kBFloat16 && out.numel() == rows * world * cols,
              "tp_exchange: out bf16 [R, W C]");
  TORCH_CHECK(rows * cols * 2 <= slice_bytes, "tp_exchange: slice exceeds the prepared capacity");
  TORCH_CHECK(epoch.is_cuda() && epoch.scalar_type() == at::kInt && epoch.numel() >= kMaxBlocks, "tp_exchange: epoch");
  if (rows == 0) return;
  Args a{};
  for (int r = 0; r < world; ++r) a.base[r] = reinterpret_cast<char*>(bases[r]);
  a.epoch = reinterpret_cast<uint32_t*>(epoch.data_ptr());
  a.x = in_place ? nullptr : reinterpret_cast<const int4*>(x.data_ptr());
  a.out = reinterpret_cast<int4*>(out.data_ptr());
  a.world = world;
  a.rank = int(rank);
  a.rows = int(rows);
  a.vcols = int(cols / 8);
  a.header = header;
  a.slice_bytes = slice_bytes;
  gather_cols_kernel<<<int(blocks), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ======================================================================================================================
// Lamport multicast column all-gather (NVLink SHARP multicast push, no flag round trip).
//
// Derived from vLLM's `mnnvl_lamport_all_gather` (vllm-project/vllm, Apache-2.0; csrc/custom_all_gather_reduce_scatter.cuh
// and csrc/custom_collective_common.cuh as of commit d79b52b18219ecbaba780f72ae0a4aa0546ac00f, read in a tree at
// v0.33.0rc0-910-g437c8dfe09; provenance and the list of changes in PROVENANCE.md next to this file). Kept from it:
// every rank pushes its slice once with multimem.st into a multicast mapping of a region that every rank owns a
// unicast alias of; the receiver spins on the payload words themselves in its own memory (no flag round trip); three
// rotating stages, the stage of the previous call cleaned while the current one is received; the stage state lives on
// the device and is advanced by the last block of the grid; PDL (griddepcontrol.wait first, launch_dependents after
// the publish); 128 threads a block, 16-byte vectors.
//
// Changed (each for byte-exactness or for a bounded wait):
//  1. No payload sentinel. vLLM marks an unwritten 32-bit word with 0x80000000 and rewrites payload words equal to it
//     ("sanitize"), which changes bits. Here every 32-bit word on the wire is (tag << 16) | u, u one 16-bit payload
//     unit (a bf16 element, or half of any wider element: the exchange is a byte copy of 16-bit units), tag a 16-bit
//     per-call stage tag that is never 0. A cleaned word is 0, which no tag equals. Every payload bit pattern travels
//     unchanged in the low half; the receiver takes the low half of words whose high half equals the call's tag.
//     Wire bytes are 2x the payload (the price of a validity field in every single-copy-atomic unit).
//  2. Tags are chosen so every tagged word is a normal finite IEEE f32 (high half = sign | exponent in [1, 254] |
//     7 mantissa bits): `multimem.st` has only floating vector forms (.v4.f32 here), and a word that is a normal
//     float is unaffected by any NaN canonicalisation or flush-to-zero a float path could apply. 65024 tags cycle.
//  3. Column layout: the receiver writes rank r's 16-byte vector (row i, column vector c) to out[i, r C + c] (the
//     contract of NcclExchange's all-gather + reorder), not vLLM's rank-major [W, n] result.
//  4. Grid-stride with a grid capped by the host (<= one block per SM by default) and every block publishing all its
//     vectors before it waits on any: once every block of every rank has started, every word is published, and a
//     grid that fits on the device cannot wait on one of its own unscheduled blocks.
//  5. The stage advance uses one acq_rel atomic: the last-arriving block advances (vLLM's block 0 spins on a counter).
//  6. Every wait is bounded by %globaltimer (timeout_ns): on expiry the first failing thread records (rank, stage,
//     call, tag counter, tag, block, thread, source rank, vector, the words seen, row vectors) in a mapped host record
//     and prints it, then the kernel traps: a lost peer becomes a launch error, never a hang.
//  7. The receiver polls every peer of a vector in one pass (vLLM's concurrent poll), the own slice comes from the input.
//  8. Own slot never cleaned nor read: the own multimem copy that lands in this rank's memory is never consumed.
//
// Region of one rank (symmetric, identical layout on every rank; the multicast mapping covers the same bytes):
//   stage s in {0,1,2}, slot r in [0, W): slot_vecs tagged 16-byte vectors at ((s W + r) slot_vecs) * 16
// Local (unicast, not symmetric) state, int32[16]: [0] stage, [1] arrival counter, [2] tag counter, [3] calls,
//   [4..6] tagged vectors per slot written in the last call of stage 0..2, [8] error claim.
//
// Call e (every rank runs the same sequence of calls on a region with the same row count each call; the stage is
// e mod 3 and the tag counter e mod 65024 on every rank, whatever the row counts):
//   P. publish: for each of my vectors i: two multimem.st.relaxed.sys.v4.f32 of the tagged words of the 8 units to
//      slot `rank`, vectors 2i and 2i+1, stage s, in every rank's memory; griddepcontrol.launch_dependents.
//   A. arrive: after __syncthreads, thread 0 adds 1 (atom.acq_rel.gpu) to the arrival counter; the block that brings
//      it to gridDim advances the state (stage, tag counter, calls, used[s] = 2 n, counter = 0) with plain stores.
//      Every block read the state before it arrived, so no block reads an advanced state; the next call reads it
//      after griddepcontrol.wait / the kernel boundary.
//   C. clean: zero (st.relaxed.sys.v4) the peer slots of stage (s + 2) mod 3 = the previous call's stage, up to the
//      vectors that call wrote.
//   R. receive: for each vector i and each peer src: load (ld.relaxed.sys.v4.u32) its two tagged vectors in my own
//      memory until all 8 words carry the tag; out[row, src C + col] = the 8 low halves. Own vector from the input.
//
// Ordering argument (PTX memory model; every access to a word is a 32-bit aligned, sys-scope, morally strong relaxed
// access: the multimem.st vector elements, the ld.relaxed.sys vector elements and the clean's st.relaxed.sys
// elements, so each word is single-copy atomic and a load returns exactly one store's 32 bits):
//   Which store publishes: peer src's multimem.st of a word at call e (one store per word and call; it carries both
//   the tag and the payload unit, so no ordering between two locations is needed: Lamport). Which load observes:
//   my ld.relaxed.sys of that word in my memory. Scope: sys (the store crosses NVLink into my memory).
//   Claim (byte-exactness): a word I accept at call e (high half == tag(e)) holds src's unit of call e. Writers of
//   slot src in my memory are only (a) src's multimem stores (slot src is written by rank src alone) and (b) my own
//   clean, which writes 0 (never a tag). So an accepted word was stored by src at some call e' with stage(e') = stage(e)
//   and tag(e') = tag(e), i.e. e' = e mod lcm(3, 65024) = 195072 (65024 = 2^9 127 is prime to 3). e' > e is
//   impossible: src starts call e + 3 only after finishing e + 2, which needs my publish of e + 2, after my call e.
//   e' < e: src's words of call e' lie in the vectors I read at e' (same row count on all ranks for a call); I
//   observed each of them at e' (I waited for it), then cleaned it at e' + 1 (a later kernel on my device, after the
//   read in causality order on my own memory, hence after src's store in coherence order: read-from + causality ->
//   coherence). So it was overwritten with 0 before call e and cannot be what I read at e (any later nonzero value
//   is again src's store of a later call of the same stage, by induction the one of call e). Stale words from a
//   larger earlier call of the same stage were cleaned the same way (the clean covers what that call wrote).
//   Liveness only (no effect on bytes): src's store of call e must land after my clean of the same words, done in
//   call e - 2 (which cleans stage(e - 3) = stage(e)). src stores at e only after finishing e - 1, which needed my
//   publish of e - 1, issued after my kernel e - 2 (with the clean) completed (stream order; griddepcontrol.wait under
//   PDL). Were that ever reordered, the word would read 0, the wait would time out and trap: never wrong bytes.
//   A peer may run one call ahead of me (it finished e, I did not): it then writes stage(e + 1) = stage(e - 2), which
//   I cleaned at e - 1, never the stage I read (e) or clean (e - 1): three stages suffice.
//   The own unit is copied from the input; the input is read after griddepcontrol.wait (the producer is complete).
//   The output is written with ordinary stores; the consumer is a later kernel (kernel boundary or griddepcontrol.wait).
// Preconditions (the same as NCCL's all-gather): every rank issues the same sequence of calls on a region with the
// same row count and width per call. A lost or diverging rank makes the others time out and trap.

namespace {

constexpr int kLThreads = 128;
constexpr uint32_t kLStages = 3;
constexpr uint32_t kTagHalf = 254u * 128u;                 // 32512 tags of each sign
constexpr uint32_t kTagPeriod = 2u * kTagHalf;            // 65024
enum : int { kStStage = 0, kStArrive = 1, kStTag = 2, kStCalls = 3, kStUsed = 4, kStErrClaim = 8, kStWords = 16 };
enum : int { kErrWords = 16 };

__host__ __device__ constexpr uint32_t lamport_tag(uint32_t c) {
  return ((c / kTagHalf) << 15) | ((1u + (c % kTagHalf) / 128u) << 7) | (c % 128u);
}
static_assert(lamport_tag(0) == 0x0080u && lamport_tag(kTagPeriod - 1) == 0xFF7Fu, "tag range");
static_assert(((lamport_tag(0) >> 7) & 0xFFu) == 1u && ((lamport_tag(kTagHalf - 1) >> 7) & 0xFFu) == 254u,
              "every tagged word is a normal f32");

__device__ __forceinline__ void mc_st_v4(void* p, uint4 v) {
  asm volatile("multimem.st.relaxed.sys.global.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(p), "r"(v.x), "r"(v.y),
               "r"(v.z), "r"(v.w)
               : "memory");
}
__device__ __forceinline__ uint4 ld_relaxed_sys_v4(const void* p) {
  uint4 v;
  asm volatile("ld.relaxed.sys.global.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p)
               : "memory");
  return v;
}
__device__ __forceinline__ void st_relaxed_sys_v4(void* p, uint4 v) {
  asm volatile("st.relaxed.sys.global.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z),
               "r"(v.w)
               : "memory");
}
__device__ __forceinline__ uint64_t global_ns() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
// 8 16-bit units of v -> two vectors of tagged words (unit k of v is word k: low halves of v.x, v.y, ... first)
__device__ __forceinline__ void tag_pack(uint4 v, uint32_t tag, uint4& lo, uint4& hi) {
  const uint32_t t = tag << 16;
  lo = make_uint4(t | (v.x & 0xFFFFu), t | (v.x >> 16), t | (v.y & 0xFFFFu), t | (v.y >> 16));
  hi = make_uint4(t | (v.z & 0xFFFFu), t | (v.z >> 16), t | (v.w & 0xFFFFu), t | (v.w >> 16));
}
__device__ __forceinline__ bool tag_ok(uint4 a, uint32_t tag) {
  return ((a.x >> 16) == tag) & ((a.y >> 16) == tag) & ((a.z >> 16) == tag) & ((a.w >> 16) == tag);
}
__device__ __forceinline__ uint4 tag_unpack(uint4 lo, uint4 hi) {
  return make_uint4((lo.x & 0xFFFFu) | (lo.y << 16), (lo.z & 0xFFFFu) | (lo.w << 16), (hi.x & 0xFFFFu) | (hi.y << 16),
                    (hi.z & 0xFFFFu) | (hi.w << 16));
}

struct LArgs {
  const uint4* x;            // [rows * vc] this rank's slice (16-byte vectors)
  uint4* out;                // [rows, world * vc]
  uint4* local;              // this rank's region (unicast)
  char* mc;                  // the region's multicast address
  uint32_t* state;           // local state [kStWords]
  uint32_t* err;             // mapped host error record [kErrWords] (device pointer) or null
  long long slot_vecs;       // tagged vectors per (stage, rank) slot
  long long nvec;            // rows * vc
  unsigned long long timeout_ns;
  int world, rank, vc;
};

__device__ __noinline__ void lamport_fail(const LArgs& a, uint32_t stage, uint32_t calls, uint32_t ctr, uint32_t tag,
                                          int src, long long i, uint4 lo, uint4 hi) {
  if (atomicCAS(a.state + kStErrClaim, 0u, 1u) == 0u) {
    if (a.err != nullptr) {
      volatile uint32_t* e = a.err;
      e[1] = uint32_t(a.rank);
      e[2] = stage;
      e[3] = calls;
      e[4] = ctr;
      e[5] = tag;
      e[6] = blockIdx.x;
      e[7] = threadIdx.x;
      e[8] = uint32_t(src);
      e[9] = uint32_t(i);
      e[10] = lo.x;
      e[11] = hi.w;
      e[12] = uint32_t(a.nvec);
      e[13] = uint32_t(a.world);
      __threadfence_system();
      e[0] = 1u;
      __threadfence_system();
    }
    printf("isoexec tp_exchange (multicast gather): wait timed out: rank %d stage %u call %u tag counter %u tag 0x%04x "
           "block %d thread %d source rank %d vector %lld of %lld, words seen 0x%08x 0x%08x; trapping\n",
           a.rank, stage, calls, ctr, tag, int(blockIdx.x), int(threadIdx.x), src, i, a.nvec, lo.x, hi.w);
  }
  __trap();
}

__global__ __launch_bounds__(kLThreads) void lamport_gather_cols_kernel(LArgs a) {
  asm volatile("griddepcontrol.wait;" ::: "memory");   // PDL: the producer's writes and the last call's state
  const uint32_t stage = a.state[kStStage], ctr = a.state[kStTag], calls = a.state[kStCalls];
  const uint32_t dirty = (stage + 2u) % kLStages;
  const long long dirty_vecs = a.state[kStUsed + dirty];
  const uint32_t tag = lamport_tag(ctr);
  const int W = a.world, me = a.rank;
  const long long stage_vecs = (long long)W * a.slot_vecs;
  const long long tid = (long long)blockIdx.x * kLThreads + threadIdx.x;
  const long long stride = (long long)gridDim.x * kLThreads;

  // P. publish my slice into slot `me` of stage `stage` of every rank
  char* pub = a.mc + ((long long)stage * stage_vecs + (long long)me * a.slot_vecs) * 16;
  for (long long i = tid; i < a.nvec; i += stride) {
    uint4 lo, hi;
    tag_pack(a.x[i], tag, lo, hi);
    mc_st_v4(pub + (2 * i) * 16, lo);
    mc_st_v4(pub + (2 * i + 1) * 16, hi);
  }
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

  // A. arrive; the last block advances the state for the next call
  __syncthreads();
  if (threadIdx.x == 0) {
    uint32_t n;
    asm volatile("atom.add.acq_rel.gpu.global.u32 %0, [%1], 1;" : "=r"(n) : "l"(a.state + kStArrive) : "memory");
    if (n == gridDim.x - 1) {
      a.state[kStUsed + stage] = uint32_t(2 * a.nvec);
      a.state[kStStage] = (stage + 1u) % kLStages;
      a.state[kStTag] = (ctr + 1u) % kTagPeriod;
      a.state[kStCalls] = calls + 1u;
      a.state[kStArrive] = 0u;
    }
  }

  // C. clean the peer slots of the previous call's stage
  if (dirty_vecs > 0 && W > 1) {
    char* base = reinterpret_cast<char*>(a.local) + (long long)dirty * stage_vecs * 16;
    const long long n = (long long)(W - 1) * dirty_vecs;
    for (long long j = tid; j < n; j += stride) {
      int s = int(j / dirty_vecs);
      s += (s >= me);
      st_relaxed_sys_v4(base + ((long long)s * a.slot_vecs + j % dirty_vecs) * 16, make_uint4(0u, 0u, 0u, 0u));
    }
  }

  // R. receive every peer's vectors from my own memory
  const char* cur = reinterpret_cast<const char*>(a.local) + (long long)stage * stage_vecs * 16;
  const uint32_t peers = ((W >= 32 ? 0xFFFFFFFFu : ((1u << W) - 1u))) & ~(1u << me);
  const uint64_t t0 = global_ns();
  for (long long i = tid; i < a.nvec; i += stride) {
    const long long row = i / a.vc, col = i % a.vc;
    uint4* o = a.out + row * (long long)W * a.vc + col;
    o[(long long)me * a.vc] = a.x[i];
    uint32_t pending = peers;
    uint32_t spins = 0;
    while (pending != 0u) {
#pragma unroll
      for (int src = 0; src < kMaxWorld; ++src) {
        if (((pending >> src) & 1u) == 0u) continue;
        const char* p = cur + ((long long)src * a.slot_vecs + 2 * i) * 16;
        const uint4 lo = ld_relaxed_sys_v4(p), hi = ld_relaxed_sys_v4(p + 16);
        if (tag_ok(lo, tag) && tag_ok(hi, tag)) {
          o[(long long)src * a.vc] = tag_unpack(lo, hi);
          pending &= ~(1u << src);
        } else if ((++spins & 255u) == 0u && global_ns() - t0 > a.timeout_ns) {
          lamport_fail(a, stage, calls, ctr, tag, src, i, lo, hi);
        }
      }
    }
  }
}

// Construction-time probe: rank `rank` stores two marked vectors through the multicast address (stage 0 slot `rank`
// first vector, stage 2 slot `rank` last vector); the host then checks every rank's marks in its unicast alias.
__global__ void lamport_probe_kernel(char* mc, long long slot_vecs, int world, int rank) {
  if (threadIdx.x != 0) return;
  const long long stage_vecs = (long long)world * slot_vecs;
  const uint32_t m = 0x3F800000u | (uint32_t(rank) << 8);   // normal f32 words
  mc_st_v4(mc + ((long long)rank * slot_vecs) * 16, make_uint4(m | 1u, m | 2u, m | 3u, m | 4u));
  mc_st_v4(mc + (2 * stage_vecs + (long long)rank * slot_vecs + slot_vecs - 1) * 16,
           make_uint4(m | 5u, m | 6u, m | 7u, m | 8u));
}

}  // namespace

void lamport_gather_cols(at::Tensor x, at::Tensor out, int64_t local, int64_t mc, at::Tensor state, int64_t err,
                         int64_t world, int64_t rank, int64_t slot_vecs, int64_t blocks, int64_t timeout_ns,
                         bool pdl) {
  const c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(world >= 2 && world <= kMaxWorld, "tp_exchange (multicast): 2..8 ranks");
  TORCH_CHECK(rank >= 0 && rank < world, "tp_exchange (multicast): rank");
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() == 2, "tp_exchange (multicast): x [R, C] contiguous");
  TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.scalar_type() == x.scalar_type(),
              "tp_exchange (multicast): out contiguous, x's dtype");
  const int64_t rows = x.size(0), row_bytes = x.size(1) * int64_t(x.element_size());
  TORCH_CHECK(row_bytes % 16 == 0, "tp_exchange (multicast): slice row bytes must be a multiple of 16");
  TORCH_CHECK(out.numel() * int64_t(out.element_size()) == rows * world * row_bytes,
              "tp_exchange (multicast): out [R, W C]");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0,
              "tp_exchange (multicast): 16-byte aligned tensors");
  TORCH_CHECK(state.is_cuda() && state.scalar_type() == at::kInt && state.numel() >= kStWords && state.is_contiguous(),
              "tp_exchange (multicast): state int32[16]");
  TORCH_CHECK(local != 0 && mc != 0 && local % 16 == 0 && mc % 16 == 0, "tp_exchange (multicast): region pointers");
  const int64_t vc = row_bytes / 16, nvec = rows * vc;
  TORCH_CHECK(2 * nvec <= slot_vecs && slot_vecs < (int64_t(1) << 31), "tp_exchange (multicast): slice exceeds the capacity");
  TORCH_CHECK(blocks >= 1 && blocks <= 65535 && timeout_ns > 0, "tp_exchange (multicast): blocks / timeout");
  if (rows == 0 || vc == 0) return;
  LArgs a{};
  a.x = reinterpret_cast<const uint4*>(x.data_ptr());
  a.out = reinterpret_cast<uint4*>(out.data_ptr());
  a.local = reinterpret_cast<uint4*>(local);
  a.mc = reinterpret_cast<char*>(mc);
  a.state = reinterpret_cast<uint32_t*>(state.data_ptr());
  a.err = reinterpret_cast<uint32_t*>(err);
  a.slot_vecs = slot_vecs;
  a.nvec = nvec;
  a.timeout_ns = static_cast<unsigned long long>(timeout_ns);
  a.world = int(world);
  a.rank = int(rank);
  a.vc = int(vc);
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(unsigned(blocks));
  cfg.blockDim = dim3(kLThreads);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, lamport_gather_cols_kernel, a));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void lamport_probe(int64_t mc, int64_t slot_vecs, int64_t world, int64_t rank, int64_t device) {
  const c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device));
  TORCH_CHECK(mc != 0 && mc % 16 == 0 && slot_vecs >= 1 && world >= 2 && world <= kMaxWorld && rank >= 0 && rank < world,
              "tp_exchange (multicast): probe arguments");
  lamport_probe_kernel<<<1, 32, 0, at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<char*>(mc), slot_vecs,
                                                                        int(world), int(rank));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Mapped pinned host record the kernel writes on a timeout (readable after the context is lost).
std::vector<int64_t> error_record_alloc(int64_t device) {
  const c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device));
  void* h = nullptr;
  void* d = nullptr;
  C10_CUDA_CHECK(cudaHostAlloc(&h, kErrWords * sizeof(uint32_t), cudaHostAllocMapped | cudaHostAllocPortable));
  std::memset(h, 0, kErrWords * sizeof(uint32_t));
  C10_CUDA_CHECK(cudaHostGetDevicePointer(&d, h, 0));
  return {reinterpret_cast<int64_t>(h), reinterpret_cast<int64_t>(d)};
}

std::vector<int64_t> error_record_read(int64_t host) {
  const volatile uint32_t* p = reinterpret_cast<const volatile uint32_t*>(host);
  std::vector<int64_t> v(kErrWords);
  for (int k = 0; k < kErrWords; ++k) v[k] = int64_t(p[k]);
  return v;
}

void error_record_free(int64_t host) {
  if (host != 0) (void)cudaFreeHost(reinterpret_cast<void*>(host));
}

}  // namespace isoexec_tp_exchange

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gather_cols", &isoexec_tp_exchange::gather_cols);
  m.def("lamport_gather_cols", &isoexec_tp_exchange::lamport_gather_cols);
  m.def("lamport_probe", &isoexec_tp_exchange::lamport_probe);
  m.def("error_record_alloc", &isoexec_tp_exchange::error_record_alloc);
  m.def("error_record_read", &isoexec_tp_exchange::error_record_read);
  m.def("error_record_free", &isoexec_tp_exchange::error_record_free);
}
