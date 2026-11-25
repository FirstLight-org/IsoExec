// SPDX-License-Identifier: Apache-2.0
//
// detreduce_sm100: one-round, TP/EP-size-invariant deterministic reductions over symmetric memory
// (DESIGN.md). Transports only -- the arithmetic is the frozen recipes' arithmetic:
//
//   tree_allreduce   (symm_tree_allreduce_v1)  out = BF16(tree(root_0, .., root_{W-1})), the adjacent-pair
//                    balanced FP32 tree of leaf_tree (kda/glm_sm100 layer.cu, deepseek_v41_cache_sm100
//                    rows.cu) over the ranks' FP32 subtree roots in rank order.
//   terms_combine    (glm_ep_symm_terms_fold_v1)  out[t] = BF16(p_0 + p_1 + ... + p_{S-1}) folded left in slot
//                    order (ascending expert id, shared last), p_s = RN(w_s * r_s) computed by the owner of
//                    the slot's expert, w = 1.0 for shared slots. Any placement, any EP.
//   moe_prep         GLM ep_prep's expert-major plan + deepseek_v41_act_quant_v1, generalized to any expert
//                    placement (owner tables) and a per-rank shared-expert row range.
//
// Synchronization: integer flags only (st.release.sys / ld.acquire.sys), monotone per-CTA epochs held on
// the device, a device call counter (advanced by the last CTA with an INTEGER atomic) selecting the
// parity half of the double-buffered staging. No floating-point atomics, no arithmetic in transit.
#include "../../../moe/clamped_swiglu_sm100/csrc/clamped_numerics.cuh"
#include "detreduce_push.cuh"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

#include <vector>

namespace detreduce {
namespace {

constexpr int kMaxWorld = 16;
constexpr int kMaxCta = 1024;
constexpr int kThreads = 256;
constexpr int kTile = kThreads * 4;       // elements (columns) per tile
constexpr int kMaxSlots = 32;
constexpr int kMaxGroups = 1024;
constexpr int kTagGrid = 256;               // fixed grid of the tagged kernels (location -> CTA is fixed)
static_assert(kTagGrid == detreduce_push::kTagGrid && kThreads * 4 == detreduce_push::kTile, "push header geometry");

// ------------------------------------------------------------------------------------- sync helpers
__device__ __forceinline__ int ld_acquire_sys(const int* p) {
  int v;
  asm volatile("ld.acquire.sys.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_release_sys(int* p, int v) {
  asm volatile("st.release.sys.global.b32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ uint64_t global_ns() {
  uint64_t t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ void spin_until(const int* flag, int e) {
  const uint64_t start = global_ns();
  while (ld_acquire_sys(flag) < e) {
    if (global_ns() - start > 20000000000ull) {
      printf("detreduce: peer flag timeout (cta %d, want %d, have %d)\n", blockIdx.x, e, ld_acquire_sys(flag));
      __trap();
    }
  }
}
__device__ __forceinline__ float4 ldcg4(const float* p) { return __ldcg(reinterpret_cast<const float4*>(p)); }
__device__ __forceinline__ uint2 pack_bf16x4(float4 a) {
  __nv_bfloat162 lo = __floats2bfloat162_rn(a.x, a.y);
  __nv_bfloat162 hi = __floats2bfloat162_rn(a.z, a.w);
  uint2 r;
  r.x = *reinterpret_cast<uint32_t*>(&lo);
  r.y = *reinterpret_cast<uint32_t*>(&hi);
  return r;
}
__device__ __forceinline__ float4 add4(float4 a, float4 b) {
  return make_float4(__fadd_rn(a.x, b.x), __fadd_rn(a.y, b.y), __fadd_rn(a.z, b.z), __fadd_rn(a.w, b.w));
}
__device__ __forceinline__ float4 mul4(float4 a, float p) {
  return make_float4(__fmul_rn(a.x, p), __fmul_rn(a.y, p), __fmul_rn(a.z, p), __fmul_rn(a.w, p));
}

struct SyncState {
  int* flags[kMaxWorld];   // each rank's flag base, [2 phases][kMaxCta][kMaxWorld] int32 (peer-mapped)
  int* epoch;              // local [kMaxCta]
  int* call;               // local [1]: calls completed (parity source)
  int* finished;           // local [1]: CTAs finished in the current call
  int* tepoch;             // local [kTagGrid]: per-CTA call count of the tagged kernels
  int world, rank;
};

__device__ __forceinline__ int flag_index(int phase, int cta, int src) {
  return (phase * kMaxCta + cta) * kMaxWorld + src;
}
// Credit block of the tagged terms kernel, after the two flag phases: [kTagGrid][kMaxWorld] int32 per rank.
// Word (b, src) of rank q holds the last tagged call (CTA b's epoch) that rank src's CTA b has finished
// reading. Written only by rank src with st.release.sys; read by rank q with ld.acquire.sys.
__device__ __forceinline__ int credit_index(int cta, int src) {
  return 2 * kMaxCta * kMaxWorld + cta * kMaxWorld + src;
}
// Per-CTA credits of the tagged terms kernel. Its parity halves are protected only by the data: a writer can reach
// call e only once the reader has started call e - 1 -- unless the writer needs NOTHING from that reader in call e - 1
// (the reader owns no slot of any row the writer folds; at M = 1, W = 4 about 10% per peer and call). Then the writer
// can finish e - 1 at once and in call e store into the half the reader still polls for call e - 2 (a different tag:
// "tagged word timeout"). Credits restore the bound: every CTA publishes "finished call e" to the same CTA on every
// peer after its polls, and a CTA that stores into peers in call e first waits until every peer finished e - 2 (the
// previous user of the same half). Progress: the wait at e depends only on calls <= e - 2.
// The dense fixed-grid kernels (tagged / Lamport / Lamport two-shot) do not need this: every element of every call
// combines every rank's root, so a rank's call e - 1 grid completes only after every peer pushed call e - 1, i.e. after
// every peer's call e - 2 grid fully completed (griddepcontrol.wait / stream order), reads and re-arms included. (A
// zero-element call launches nothing and advances no epoch.) Adding credits there cost 5-7 us per call (measured).
// `stores`: this CTA stores into peers' boxes in call e (block-uniform).
__device__ __forceinline__ void credit_wait(const SyncState& s, int e, bool stores) {
  if (!stores) return;
  const int q = threadIdx.x;
  if (q < s.world && q != s.rank) spin_until(s.flags[s.rank] + credit_index(blockIdx.x, q), e - 2);
  __syncthreads();
}
// after every read / re-arm of call e by this CTA (bar.sync orders all threads' accesses before the cumulative release)
__device__ __forceinline__ void credit_post(const SyncState& s, int e) {
  __syncthreads();
  const int q = threadIdx.x;
  if (q < s.world && q != s.rank) st_release_sys(s.flags[q] + credit_index(blockIdx.x, s.rank), e);
}

// threads [0, world) publish this CTA's phase flag to every peer, after the CTA's writes (bar.sync
// orders them before the release, which is cumulative).
__device__ __forceinline__ void publish(const SyncState& s, int phase, int e) {
  __syncthreads();
  const int q = threadIdx.x;
  if (q < s.world && q != s.rank) st_release_sys(s.flags[q] + flag_index(phase, blockIdx.x, s.rank), e);
}
// threads [0, world) each wait for one peer's flag of the same CTA index; then the CTA proceeds.
__device__ __forceinline__ void await(const SyncState& s, int phase, int e) {
  const int q = threadIdx.x;
  if (q < s.world && q != s.rank) spin_until(s.flags[s.rank] + flag_index(phase, blockIdx.x, q), e);
  __syncthreads();
}
__device__ __forceinline__ void begin(const SyncState& s, int& e, int& par) {
  __shared__ int e_s, par_s;
  if (threadIdx.x == 0) {
    e_s = s.epoch[blockIdx.x] + 1;
    par_s = *reinterpret_cast<volatile int*>(s.call) & 1;
  }
  __syncthreads();
  e = e_s;
  par = par_s;
}
__device__ __forceinline__ void finish(const SyncState& s, int e) {
  __syncthreads();
  if (threadIdx.x == 0) {
    s.epoch[blockIdx.x] = e;   // read only by later kernels (stream order): no fence needed
    const int done = atomicAdd(s.finished, 1);   // integer atomic (CTA completion count)
    if (done == static_cast<int>(gridDim.x) - 1) {
      *s.finished = 0;
      *s.call += 1;
    }
  }
}

// ------------------------------------------------------------------------- fixed tree all-reduce
struct TreeArgs {
  const float* in;                    // [n] this rank's FP32 subtree root
  __nv_bfloat16* out;                 // [n]
  float* inbox[kMaxWorld];            // each rank's FP32 inbox: 2 halves of `cap`, W slots of `slot` per half
  __nv_bfloat16* ybox[kMaxWorld];     // each rank's BF16 row inbox (two-shot): 2 halves of `cap`
  uint64_t* tbox[kMaxWorld];          // each rank's tagged inbox (zero-initialized, tagged words only): 2 halves of `tcap`
  float* lbox[kMaxWorld];             // each rank's lamport box (FP32 words, sentinel-initialized): 2 halves of `lcap`
  uint32_t* lybox[kMaxWorld];         // each rank's Lamport BF16 box (BF16 pairs as 32-bit words, sentinel-initialized): 2 halves of `lycap` elements
  int64_t n, cap, slot, tcap, tslot, lcap, lslot, lycap;
  int two_shot;
};

// The leaf_tree program on W values in rank order: level by level, node k = node 2k (+) node 2k+1.
template <int W>
__device__ __forceinline__ float4 tree4(float4 (&v)[W]) {
#pragma unroll
  for (int m = W; m > 1; m >>= 1) {
#pragma unroll
    for (int k = 0; k < m / 2; ++k) v[k] = add4(v[2 * k], v[2 * k + 1]);
  }
  return v[0];
}
template <int W>
__device__ __forceinline__ float tree1(float (&v)[W]) {
#pragma unroll
  for (int m = W; m > 1; m >>= 1) {
#pragma unroll
    for (int k = 0; k < m / 2; ++k) v[k] = __fadd_rn(v[2 * k], v[2 * k + 1]);
  }
  return v[0];
}

// Push form. One-shot: every rank pushes its root into slot `rank` of every peer's inbox; every rank
// evaluates the whole tree from local memory. Two-shot: tile i is reduced by rank i mod W; its root
// tiles are pushed to that owner only, at the compact offset (i / W) * kTile of the sender's slot; the
// owner rounds once and pushes the BF16 tile to every peer's ybox. Inboxes have one writer per word.
template <int W>
__global__ __launch_bounds__(kThreads) void tree_allreduce_kernel(TreeArgs a, SyncState s) {
  asm volatile("griddepcontrol.wait;" ::: "memory");   // PDL: the root producer is complete
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  int e, par;
  begin(s, e, par);
  const int me = s.rank;
  const int64_t tiles = (a.n + kTile - 1) / kTile;
  const int64_t half = par * a.cap;
  float* const local = a.inbox[me] + half;
  for (int64_t tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
    const int owner = a.two_shot ? static_cast<int>(tile % W) : -1;
    if (owner == me) continue;
    const int64_t c = tile * kTile + threadIdx.x * 4;
    const int64_t dst = (a.two_shot ? (tile / W) * kTile + threadIdx.x * 4 : c) + me * a.slot + half;
    if (c + 3 < a.n) {
      const float4 v = __ldg(reinterpret_cast<const float4*>(a.in + c));
      if (a.two_shot) {
        *reinterpret_cast<float4*>(a.inbox[owner] + dst) = v;
      } else {
#pragma unroll
        for (int q = 0; q < W; ++q)
          if (q != me) *reinterpret_cast<float4*>(a.inbox[q] + dst) = v;
      }
    } else {
      for (int64_t i = c; i < a.n; ++i) {
        const float v = a.in[i];
        if (a.two_shot) {
          a.inbox[owner][dst + (i - c)] = v;
        } else {
          for (int q = 0; q < W; ++q)
            if (q != me) a.inbox[q][dst + (i - c)] = v;
        }
      }
    }
  }
  publish(s, 0, e);
  await(s, 0, e);
  for (int64_t tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
    if (a.two_shot && tile % W != me) continue;
    const int64_t c = tile * kTile + threadIdx.x * 4;
    const int64_t src = (a.two_shot ? (tile / W) * kTile + threadIdx.x * 4 : c);
    if (c + 3 < a.n) {
      float4 v[W];
#pragma unroll
      for (int q = 0; q < W; ++q)
        v[q] = q == me ? __ldg(reinterpret_cast<const float4*>(a.in + c)) : ldcg4(local + q * a.slot + src);
      const uint2 r = pack_bf16x4(tree4<W>(v));
      *reinterpret_cast<uint2*>(a.out + c) = r;
      if (a.two_shot)
#pragma unroll
        for (int q = 0; q < W; ++q)
          if (q != me) *reinterpret_cast<uint2*>(a.ybox[q] + half + c) = r;
    } else {
      for (int64_t i = c; i < a.n; ++i) {
        float v[W];
#pragma unroll
        for (int q = 0; q < W; ++q) v[q] = q == me ? a.in[i] : __ldcg(local + q * a.slot + src + (i - c));
        const __nv_bfloat16 r = __float2bfloat16_rn(tree1<W>(v));
        a.out[i] = r;
        if (a.two_shot)
          for (int q = 0; q < W; ++q)
            if (q != me) a.ybox[q][half + i] = r;
      }
    }
  }
  if (a.two_shot) {
    publish(s, 1, e);
    await(s, 1, e);
    const __nv_bfloat16* ylocal = a.ybox[me] + half;
    for (int64_t tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
      if (tile % W == me) continue;
      const int64_t c = tile * kTile + threadIdx.x * 4;
      if (c + 3 < a.n) {
        *reinterpret_cast<uint2*>(a.out + c) = __ldcg(reinterpret_cast<const uint2*>(ylocal + c));
      } else {
        for (int64_t i = c; i < a.n; ++i) a.out[i] = ylocal[i];
      }
    }
  }
  finish(s, e);
}

// Tagged one-shot (no flags): every FP32 travels as one 64-bit word {value bits, call tag} written with
// a single-copy-atomic 8-byte store; the reader polls each word until its tag is this call's tag.
// The tag is the device call counter + 1 (identical on every rank and CTA, strictly increasing), so a
// word left over from any earlier call can never match. Parity halves keep a writer that is one call
// ahead out of the half being read (it cannot get two calls ahead: it needs this rank's next data).
__device__ __forceinline__ void st_tag(uint64_t* p, float v, uint32_t tag) {
  const uint64_t w = static_cast<uint64_t>(__float_as_uint(v)) | (static_cast<uint64_t>(tag) << 32);
  asm volatile("st.relaxed.sys.global.b64 [%0], %1;" ::"l"(p), "l"(w) : "memory");
}
__device__ __forceinline__ float ld_tag(const uint64_t* p, uint32_t tag) {
  uint64_t w;
  const uint64_t start = global_ns();
  while (true) {
    asm volatile("ld.relaxed.sys.global.b64 %0, [%1];" : "=l"(w) : "l"(p) : "memory");
    if (static_cast<uint32_t>(w >> 32) == tag) break;
    if (global_ns() - start > 20000000000ull) {
      printf("detreduce: tagged word timeout (cta %d, want %u, have %u)\n", blockIdx.x, tag, static_cast<uint32_t>(w >> 32));
      __trap();
    }
  }
  return __uint_as_float(static_cast<uint32_t>(w));
}

// Four tagged words as two 16-byte vector accesses (each 64-bit element is its own single-copy-atomic access).
__device__ __forceinline__ void st_tag4(uint64_t* p, float4 v, uint32_t tag) {
  const uint64_t t = static_cast<uint64_t>(tag) << 32;
  const uint64_t w0 = __float_as_uint(v.x) | t, w1 = __float_as_uint(v.y) | t;
  const uint64_t w2 = __float_as_uint(v.z) | t, w3 = __float_as_uint(v.w) | t;
  asm volatile("st.relaxed.sys.global.v2.b64 [%0], {%1, %2};" ::"l"(p), "l"(w0), "l"(w1) : "memory");
  asm volatile("st.relaxed.sys.global.v2.b64 [%0], {%1, %2};" ::"l"(p + 2), "l"(w2), "l"(w3) : "memory");
}
__device__ __forceinline__ float4 ld_tag4(const uint64_t* p, uint32_t tag) {
  uint64_t w0, w1, w2, w3;
  const uint64_t start = global_ns();
  while (true) {
    asm volatile("ld.relaxed.sys.global.v2.b64 {%0, %1}, [%2];" : "=l"(w0), "=l"(w1) : "l"(p) : "memory");
    asm volatile("ld.relaxed.sys.global.v2.b64 {%0, %1}, [%2];" : "=l"(w2), "=l"(w3) : "l"(p + 2) : "memory");
    if (static_cast<uint32_t>(w0 >> 32) == tag && static_cast<uint32_t>(w1 >> 32) == tag &&
        static_cast<uint32_t>(w2 >> 32) == tag && static_cast<uint32_t>(w3 >> 32) == tag)
      break;
    if (global_ns() - start > 20000000000ull) {
      printf("detreduce: tagged word timeout (cta %d, want %u)\n", blockIdx.x, tag);
      __trap();
    }
  }
  return make_float4(__uint_as_float(static_cast<uint32_t>(w0)), __uint_as_float(static_cast<uint32_t>(w1)),
                     __uint_as_float(static_cast<uint32_t>(w2)), __uint_as_float(static_cast<uint32_t>(w3)));
}

template <int W>
__global__ __launch_bounds__(kThreads) void tree_tagged_kernel(TreeArgs a, SyncState s) {
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  // Fixed grid (kTagGrid) and fixed tile -> CTA map: every inbox word is always written and read by the
  // same CTA index, so that CTA's call count is a strictly increasing tag for the word and its parity
  // selects the half. No global counter, no atomics.
  const int e = *reinterpret_cast<volatile int*>(s.tepoch + blockIdx.x) + 1;
  const uint32_t tag = static_cast<uint32_t>(e);
  const int par = e & 1;
  const int me = s.rank;
  const int64_t tiles = (a.n + kTile - 1) / kTile;
  const int64_t slot64 = a.tslot;                       // 64-bit words per slot
  const int64_t half64 = par * a.tcap;
  if (a.two_shot != 3)   // 3: the producer already pushed (detreduce_push.cuh)
  for (int64_t tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
    const int64_t c = tile * kTile + threadIdx.x * 4;
    if (c >= a.n) continue;
    const int cnt = static_cast<int>(a.n - c < 4 ? a.n - c : 4);
    float v[4];
    if (cnt == 4) {
      const float4 x = __ldg(reinterpret_cast<const float4*>(a.in + c));
      v[0] = x.x, v[1] = x.y, v[2] = x.z, v[3] = x.w;
    } else {
      for (int j = 0; j < cnt; ++j) v[j] = a.in[c + j];
    }
#pragma unroll
    for (int q = 0; q < W; ++q) {
      if (q == me) continue;
      uint64_t* dst = a.tbox[q] + half64 + me * slot64 + c;
      if (cnt == 4) st_tag4(dst, make_float4(v[0], v[1], v[2], v[3]), tag);
      else for (int j = 0; j < cnt; ++j) st_tag(dst + j, v[j], tag);
    }
  }
  const uint64_t* local = a.tbox[me] + half64;
  for (int64_t tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
    const int64_t c = tile * kTile + threadIdx.x * 4;
    if (c >= a.n) continue;
    const int cnt = static_cast<int>(a.n - c < 4 ? a.n - c : 4);
    if (cnt == 4) {
      float4 v[W];
#pragma unroll
      for (int q = 0; q < W; ++q)
        v[q] = q == me ? __ldg(reinterpret_cast<const float4*>(a.in + c)) : ld_tag4(local + q * slot64 + c, tag);
      *reinterpret_cast<uint2*>(a.out + c) = pack_bf16x4(tree4<W>(v));
    } else {
      for (int j = 0; j < cnt; ++j) {
        float v[W];
#pragma unroll
        for (int q = 0; q < W; ++q) v[q] = q == me ? __ldg(a.in + c + j) : ld_tag(local + q * slot64 + c + j, tag);
        a.out[c + j] = __float2bfloat16_rn(tree1<W>(v));
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) s.tepoch[blockIdx.x] = e;
}

// Lamport FP32 one-shot (mode 5): one 32-bit word per element, sentinel 0xFFFFFFFF (a NaN bit pattern).
// A root word equal to the sentinel is sent as 0x7FFFFFFF instead: every root enters at least one
// __fadd_rn at W >= 2 and an FP32 add returns the canonical NaN for ANY NaN operand, so the output bits
// are unchanged. Box = 2 parity halves x W slots, initialized to the sentinel; the reader re-arms each
// word right after consuming it. Fixed grid / fixed tile -> CTA map and per-CTA epochs (as the tagged
// form): a writer can be at most one call ahead of a reader on the same CTA index, so it only ever
// writes the half the reader re-armed during its previous kernel (completed before the reader pushed
// the data that let the writer advance).
constexpr uint32_t kSentinel = 0xFFFFFFFFu;
__device__ __forceinline__ uint32_t lamport_word(float v) {
  const uint32_t b = __float_as_uint(v);
  return b == kSentinel ? 0x7FFFFFFFu : b;
}
__device__ __forceinline__ void st_lamport4(float* p, float4 v) {
  asm volatile("st.relaxed.sys.global.v4.b32 [%0], {%1, %2, %3, %4};" ::"l"(p), "r"(lamport_word(v.x)),
               "r"(lamport_word(v.y)), "r"(lamport_word(v.z)), "r"(lamport_word(v.w))
               : "memory");
}
__device__ __forceinline__ float4 ld_lamport4(float* p) {
  uint32_t a, b, c, d;
  const uint64_t start = global_ns();
  while (true) {
    asm volatile("ld.relaxed.sys.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "l"(p) : "memory");
    if (a != kSentinel && b != kSentinel && c != kSentinel && d != kSentinel) break;
    if (global_ns() - start > 20000000000ull) {
      printf("detreduce: lamport word timeout (cta %d)\n", blockIdx.x);
      __trap();
    }
  }
  // re-arm for the call two epochs later (local write, ordered before this kernel's completion)
  asm volatile("st.relaxed.sys.global.v4.b32 [%0], {%1, %1, %1, %1};" ::"l"(p), "r"(kSentinel) : "memory");
  return make_float4(__uint_as_float(a), __uint_as_float(b), __uint_as_float(c), __uint_as_float(d));
}

__device__ __forceinline__ void ld_lamport4_raw(const float* p, float4& r) {
  uint32_t a, b, c, d;
  asm volatile("ld.relaxed.sys.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "l"(p) : "memory");
  r = make_float4(__uint_as_float(a), __uint_as_float(b), __uint_as_float(c), __uint_as_float(d));
}
// r holds a first read of p; re-read until no element is the sentinel, then re-arm p.
__device__ __forceinline__ void lamport_settle4(float* p, float4& r) {
  const uint64_t start = global_ns();
  while (__float_as_uint(r.x) == kSentinel || __float_as_uint(r.y) == kSentinel ||
         __float_as_uint(r.z) == kSentinel || __float_as_uint(r.w) == kSentinel) {
    if (global_ns() - start > 20000000000ull) {
      printf("detreduce: lamport word timeout (cta %d)\n", blockIdx.x);
      __trap();
    }
    ld_lamport4_raw(p, r);
  }
  asm volatile("st.relaxed.sys.global.v4.b32 [%0], {%1, %1, %1, %1};" ::"l"(p), "r"(kSentinel) : "memory");
}

template <int W>
__global__ __launch_bounds__(kThreads) void tree_lamport_kernel(TreeArgs a, SyncState s) {
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  const int e = *reinterpret_cast<volatile int*>(s.tepoch + blockIdx.x) + 1;
  const int par = e & 1;
  const int me = s.rank;
  const int64_t tiles = a.n / kTile + (a.n % kTile ? 1 : 0);
  float* const box = a.lbox[me] + par * a.lcap;
  if (a.two_shot != 6) {   // 6: prepushed by the producer (detreduce_push.cuh push4_lamport)
    for (int64_t tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
      const int64_t c = tile * kTile + threadIdx.x * 4;
      if (c >= a.n) continue;
      const float4 v = __ldg(reinterpret_cast<const float4*>(a.in + c));
#pragma unroll
      for (int q = 0; q < W; ++q)
        if (q != me) st_lamport4(a.lbox[q] + par * a.lcap + me * a.lslot + c, v);
    }
  }
  // Poll phase with instruction-level parallelism: kU tiles x (W-1) peers of loads are in flight before
  // any sentinel check; each word that still holds the sentinel is re-read until it arrives. The tile ->
  // CTA map (tile = blockIdx.x + k * gridDim.x) is unchanged.
  constexpr int kU = 4;
  for (int64_t t0 = blockIdx.x; t0 < tiles; t0 += static_cast<int64_t>(kU) * gridDim.x) {
    float4 v[kU][W];
#pragma unroll
    for (int u = 0; u < kU; ++u) {
      const int64_t c = (t0 + static_cast<int64_t>(u) * gridDim.x) * kTile + threadIdx.x * 4;
      if (c >= a.n) continue;
#pragma unroll
      for (int q = 0; q < W; ++q)
        if (q == me) v[u][q] = __ldg(reinterpret_cast<const float4*>(a.in + c));
        else ld_lamport4_raw(box + q * a.lslot + c, v[u][q]);
    }
#pragma unroll
    for (int u = 0; u < kU; ++u) {
      const int64_t c = (t0 + static_cast<int64_t>(u) * gridDim.x) * kTile + threadIdx.x * 4;
      if (c >= a.n) continue;
#pragma unroll
      for (int q = 0; q < W; ++q)
        if (q != me) lamport_settle4(box + q * a.lslot + c, v[u][q]);
      *reinterpret_cast<uint2*>(a.out + c) = pack_bf16x4(tree4<W>(v[u]));
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) s.tepoch[blockIdx.x] = e;
}


// Lamport two-shot (mode 7). One-shot moves (W-1) x 4 B per element per rank; this moves (W-1)/W x 4 B of FP32 roots
// to the tile owners plus (W-1)/W x 2 B of BF16 results back. Tile (c / kTile) = b + k * G (G = kTagGrid, b = CTA) is
// reduced by rank (k + b) % W: the other ranks push its roots to the owner's Lamport box at the compact offset
// (b + (k / W) * G) * kTile of their sender slot; the owner polls them, evaluates the same tree on the same operands as
// one-shot, rounds once, and pushes the BF16 words to every peer's BF16 box; peers poll. Sentinel 0xFFFFFFFF in both
// boxes (a BF16 pair cannot equal it: the tree's NaN is canonical 0x7FFF; lamport_pair maps it away anyway). Same fixed
// grid, tile -> CTA map, parity halves and re-arm-after-consume argument as mode 5: every word has one writer and is
// read by the same CTA index on the reader, and a writer is at most one call ahead of any reader.
__device__ __forceinline__ uint32_t lamport_pair(float a, float b) {
  __nv_bfloat162 p = __floats2bfloat162_rn(a, b);
  const uint32_t w = *reinterpret_cast<uint32_t*>(&p);
  return w == kSentinel ? 0x7FFF7FFFu : w;
}
template <int W>
__global__ __launch_bounds__(kThreads) void tree_lamport2_kernel(TreeArgs a, SyncState s) {
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  constexpr int G = kTagGrid;
  constexpr int kU = 4;
  const int b = blockIdx.x;
  const int e = *reinterpret_cast<volatile int*>(s.tepoch + b) + 1;
  const int par = e & 1;
  const int me = s.rank;
  const int64_t tiles = (a.n + kTile - 1) / kTile;
  const int k0 = ((me - b) % W + W) % W;                     // rounds k with k % W == k0 are owned by this rank
  float* const box = a.lbox[me] + par * a.lcap;
  uint32_t* const ybox = a.lybox[me] + par * (a.lycap / 2);
  // phase A
  for (int k = 0; b + static_cast<int64_t>(k) * G < tiles; ++k) {
    if (k % W == k0) continue;
    const int64_t c = (b + static_cast<int64_t>(k) * G) * kTile + threadIdx.x * 4;
    if (c >= a.n) continue;
    const int owner = (k + b) % W;
    st_lamport4(a.lbox[owner] + par * a.lcap + me * a.lslot + (b + static_cast<int64_t>(k / W) * G) * kTile + threadIdx.x * 4,
                __ldg(reinterpret_cast<const float4*>(a.in + c)));
  }
  // phase B: owned tiles
  for (int k = k0; b + static_cast<int64_t>(k) * G < tiles; k += W * kU) {
    float4 v[kU][W];
#pragma unroll
    for (int u = 0; u < kU; ++u) {
      const int ku = k + u * W;
      const int64_t c = (b + static_cast<int64_t>(ku) * G) * kTile + threadIdx.x * 4;
      if (c >= a.n) continue;
#pragma unroll
      for (int q = 0; q < W; ++q)
        if (q == me) v[u][q] = __ldg(reinterpret_cast<const float4*>(a.in + c));
        else ld_lamport4_raw(box + q * a.lslot + (b + static_cast<int64_t>(ku / W) * G) * kTile + threadIdx.x * 4, v[u][q]);
    }
#pragma unroll
    for (int u = 0; u < kU; ++u) {
      const int ku = k + u * W;
      const int64_t c = (b + static_cast<int64_t>(ku) * G) * kTile + threadIdx.x * 4;
      if (c >= a.n) continue;
#pragma unroll
      for (int q = 0; q < W; ++q)
        if (q != me) lamport_settle4(box + q * a.lslot + (b + static_cast<int64_t>(ku / W) * G) * kTile + threadIdx.x * 4, v[u][q]);
      const float4 t = tree4<W>(v[u]);
      const uint2 r = make_uint2(lamport_pair(t.x, t.y), lamport_pair(t.z, t.w));
      *reinterpret_cast<uint2*>(a.out + c) = make_uint2(r.x, r.y);
#pragma unroll
      for (int q = 0; q < W; ++q)
        if (q != me)
          asm volatile("st.relaxed.sys.global.v2.b32 [%0], {%1, %2};" ::"l"(a.lybox[q] + par * (a.lycap / 2) + c / 2), "r"(r.x), "r"(r.y)
                       : "memory");
    }
  }
  // phase C: tiles reduced by peers
  int k = 0;
  while (b + static_cast<int64_t>(k) * G < tiles) {
    uint2 r[kU];
    int64_t cc[kU];
    int cnt = 0;
    while (cnt < kU && b + static_cast<int64_t>(k) * G < tiles) {
      if (k % W != k0) {
        const int64_t c = (b + static_cast<int64_t>(k) * G) * kTile + threadIdx.x * 4;
        if (c < a.n) {
          cc[cnt] = c;
          asm volatile("ld.relaxed.sys.global.v2.b32 {%0, %1}, [%2];" : "=r"(r[cnt].x), "=r"(r[cnt].y) : "l"(ybox + c / 2) : "memory");
          ++cnt;
        }
      }
      ++k;
    }
#pragma unroll
    for (int i = 0; i < kU; ++i) {
      if (i >= cnt) break;
      uint32_t* p = ybox + cc[i] / 2;
      const uint64_t start = global_ns();
      while (r[i].x == kSentinel || r[i].y == kSentinel) {
        if (global_ns() - start > 20000000000ull) {
          printf("detreduce: lamport2 word timeout (cta %d)\n", blockIdx.x);
          __trap();
        }
        asm volatile("ld.relaxed.sys.global.v2.b32 {%0, %1}, [%2];" : "=r"(r[i].x), "=r"(r[i].y) : "l"(p) : "memory");
      }
      asm volatile("st.relaxed.sys.global.v2.b32 [%0], {%1, %1};" ::"l"(p), "r"(kSentinel) : "memory");
      *reinterpret_cast<uint2*>(a.out + cc[i]) = r[i];
    }
  }
  (void)box;
  __syncthreads();
  if (threadIdx.x == 0) s.tepoch[b] = e;
}


// Isolated mode8: ordinary same-stream grid boundaries, no PDL.
template <int W>
__global__ __launch_bounds__(kThreads) void tree_phased_push_kernel(TreeArgs a, SyncState s) {
  constexpr int G = kTagGrid;
  constexpr int kU = 4;
  const int b = blockIdx.x;
  const int e = *reinterpret_cast<volatile int*>(s.tepoch + b) + 1;
  const int par = e & 1;
  const int me = s.rank;
  const int64_t tiles = (a.n + kTile - 1) / kTile;
  const int k0 = ((me - b) % W + W) % W;                     // rounds k with k % W == k0 are owned by this rank
  float* const box = a.lbox[me] + par * a.lcap;
  uint32_t* const ybox = a.lybox[me] + par * (a.lycap / 2);
  credit_wait(s, e, true);
  // phase A
  for (int k = 0; b + static_cast<int64_t>(k) * G < tiles; ++k) {
    if (k % W == k0) continue;
    const int64_t c = (b + static_cast<int64_t>(k) * G) * kTile + threadIdx.x * 4;
    if (c >= a.n) continue;
    const int owner = (k + b) % W;
    st_lamport4(a.lbox[owner] + par * a.lcap + me * a.lslot + (b + static_cast<int64_t>(k / W) * G) * kTile + threadIdx.x * 4,
                __ldg(reinterpret_cast<const float4*>(a.in + c)));
  }
}

template <int W>
__global__ __launch_bounds__(kThreads) void tree_phased_fold_kernel(TreeArgs a, SyncState s) {
  constexpr int G = kTagGrid;
  constexpr int kU = 4;
  const int b = blockIdx.x;
  const int e = *reinterpret_cast<volatile int*>(s.tepoch + b) + 1;
  const int par = e & 1;
  const int me = s.rank;
  const int64_t tiles = (a.n + kTile - 1) / kTile;
  const int k0 = ((me - b) % W + W) % W;                     // rounds k with k % W == k0 are owned by this rank
  float* const box = a.lbox[me] + par * a.lcap;
  uint32_t* const ybox = a.lybox[me] + par * (a.lycap / 2);
  // phase B: owned tiles
  for (int k = k0; b + static_cast<int64_t>(k) * G < tiles; k += W * kU) {
    float4 v[kU][W];
#pragma unroll
    for (int u = 0; u < kU; ++u) {
      const int ku = k + u * W;
      const int64_t c = (b + static_cast<int64_t>(ku) * G) * kTile + threadIdx.x * 4;
      if (c >= a.n) continue;
#pragma unroll
      for (int q = 0; q < W; ++q)
        if (q == me) v[u][q] = __ldg(reinterpret_cast<const float4*>(a.in + c));
        else ld_lamport4_raw(box + q * a.lslot + (b + static_cast<int64_t>(ku / W) * G) * kTile + threadIdx.x * 4, v[u][q]);
    }
#pragma unroll
    for (int u = 0; u < kU; ++u) {
      const int ku = k + u * W;
      const int64_t c = (b + static_cast<int64_t>(ku) * G) * kTile + threadIdx.x * 4;
      if (c >= a.n) continue;
#pragma unroll
      for (int q = 0; q < W; ++q)
        if (q != me) lamport_settle4(box + q * a.lslot + (b + static_cast<int64_t>(ku / W) * G) * kTile + threadIdx.x * 4, v[u][q]);
      const float4 t = tree4<W>(v[u]);
      const uint2 r = make_uint2(lamport_pair(t.x, t.y), lamport_pair(t.z, t.w));
      *reinterpret_cast<uint2*>(a.out + c) = make_uint2(r.x, r.y);
#pragma unroll
      for (int q = 0; q < W; ++q)
        if (q != me)
          asm volatile("st.relaxed.sys.global.v2.b32 [%0], {%1, %2};" ::"l"(a.lybox[q] + par * (a.lycap / 2) + c / 2), "r"(r.x), "r"(r.y)
                       : "memory");
    }
  }
}

template <int W>
__global__ __launch_bounds__(kThreads) void tree_phased_gather_kernel(TreeArgs a, SyncState s) {
  constexpr int G = kTagGrid;
  constexpr int kU = 4;
  const int b = blockIdx.x;
  const int e = *reinterpret_cast<volatile int*>(s.tepoch + b) + 1;
  const int par = e & 1;
  const int me = s.rank;
  const int64_t tiles = (a.n + kTile - 1) / kTile;
  const int k0 = ((me - b) % W + W) % W;                     // rounds k with k % W == k0 are owned by this rank
  float* const box = a.lbox[me] + par * a.lcap;
  uint32_t* const ybox = a.lybox[me] + par * (a.lycap / 2);
  // phase C: tiles reduced by peers
  int k = 0;
  while (b + static_cast<int64_t>(k) * G < tiles) {
    uint2 r[kU];
    int64_t cc[kU];
    int cnt = 0;
    while (cnt < kU && b + static_cast<int64_t>(k) * G < tiles) {
      if (k % W != k0) {
        const int64_t c = (b + static_cast<int64_t>(k) * G) * kTile + threadIdx.x * 4;
        if (c < a.n) {
          cc[cnt] = c;
          asm volatile("ld.relaxed.sys.global.v2.b32 {%0, %1}, [%2];" : "=r"(r[cnt].x), "=r"(r[cnt].y) : "l"(ybox + c / 2) : "memory");
          ++cnt;
        }
      }
      ++k;
    }
#pragma unroll
    for (int i = 0; i < kU; ++i) {
      if (i >= cnt) break;
      uint32_t* p = ybox + cc[i] / 2;
      const uint64_t start = global_ns();
      while (r[i].x == kSentinel || r[i].y == kSentinel) {
        if (global_ns() - start > 20000000000ull) {
          printf("detreduce: lamport2 word timeout (cta %d)\n", blockIdx.x);
          __trap();
        }
        asm volatile("ld.relaxed.sys.global.v2.b32 {%0, %1}, [%2];" : "=r"(r[i].x), "=r"(r[i].y) : "l"(p) : "memory");
      }
      asm volatile("st.relaxed.sys.global.v2.b32 [%0], {%1, %1};" ::"l"(p), "r"(kSentinel) : "memory");
      *reinterpret_cast<uint2*>(a.out + cc[i]) = r[i];
    }
  }
  credit_post(s, e);
  if (threadIdx.x == 0) s.tepoch[b] = e;
}


// ------------------------------------------------------------------ fused single-launch phased transport (mode 9)
// Same arithmetic as mode 8 (push / fold / gather): out = lamport_pair(tree4<W>(root_0 .. root_{W-1})) per element, rank
// order, FP32 __fadd_rn, one BF16 rounding, the same sentinel remapping (lamport_word on every pushed root word,
// lamport_pair on every BF16 pair). Only the schedule differs:
//   * ONE launch per chunk instead of three; the grid is sized to the payload (no 256-CTA floor) and either one-shot
//     (TWO = false: every rank pushes its root to every peer and evaluates every tree from local memory: one NVLink hop) or
//     two-shot (TWO = true: item owner folds and pushes BF16 back: two hops, (W-1)/W x 6 B per element instead of (W-1) x 4 B).
//   * Progress does not need co-residency. A CTA takes a dense start ticket ("slot", integer atomicAdd on a local counter)
//     and processes items slot, slot + G, slot + 2G, ... in three phases: (A) push all its items' roots, no waits; (B) fold the
//     items it owns, waiting only for the peers' phase-A words of the same slot; (C) two-shot only: gather the BF16 words of
//     items owned by peers, waiting only for the peers' phase B of the same slot. Slots start densely on each rank, so when
//     slot j waits for slot j of a peer, every slot < j of that peer has started; once slot j has started on every rank its
//     phases complete in (A, B, C) order. Hence the kernel completes if each rank eventually runs at least one CTA at a time,
//     whatever the residency or CTA scheduling order (the property mode 8 got from grid boundaries).
//   * Parity half = global call count & 1 (variable grids => per-CTA epochs would not line up). Call count and slot ticket
//     share one 64-bit integer word: the CTA that draws ticket G - 1 advances the call and resets the ticket (no tail
//     atomics, no completion fences on the critical path).
//   * Call-level credits replace mode 8's per-CTA credits: at the start of call e, slot 0 releases "finished e - 1" into
//     every peer's credit word (launch e - 1 completed before launch e started: one stream, no PDL), and every CTA first
//     waits for every peer's e - 2 (the previous user of the same half). In steady state: one local load.
// Every wait is bounded (20 s, then printf + __trap: fail closed). No PDL: the kernel neither starts before its producer
// completes nor lets a dependent (e.g. MegaMoE) start while it may still spin.
// State words (private mode-9 TreeWorkspace; mode 8's tepoch / credit block are untouched): s.epoch[0..1] as one 64-bit
// word {calls done, tickets drawn}. Credit word of peer src: flags[me] + fused_credit_index(src).
__device__ __forceinline__ int fused_credit_index(int src) { return flag_index(0, 0, src); }

template <int W, bool TWO, int U>
__global__ __launch_bounds__(kThreads) void tree_fused_kernel(TreeArgs a, SyncState s) {
  __shared__ int slot_s, e_s;
  const int me = s.rank;
  const int t = threadIdx.x;
  const int G = gridDim.x;
  if (t == 0) {
    // one 64-bit integer counter {calls done : 32 | tickets drawn in this call : 32}; the CTA that draws the last ticket
    // (all G drawn: atomics on one word are totally ordered) advances the call and resets the ticket for the next launch
    unsigned long long* word = reinterpret_cast<unsigned long long*>(s.epoch);
    const unsigned long long o = atomicAdd(word, 1ull);
    slot_s = static_cast<int>(o & 0xFFFFFFFFull);
    e_s = static_cast<int>(o >> 32) + 1;
    if (slot_s == G - 1) atomicAdd(word, (1ull << 32) - static_cast<unsigned long long>(G));
  }
  __syncthreads();
  const int j = slot_s;
  const int e = e_s;
  const int par = e & 1;
  // credits: wait until every peer finished call e - 2 (last user of half par); post "finished e - 1" for this rank (the
  // previous launch on this workspace completed before this one started: same stream, no PDL)
  if (t < W && t != me) spin_until(s.flags[me] + fused_credit_index(t), e - 2);
  if (j == 0 && t >= 32 && t < 32 + W && t - 32 != me) st_release_sys(s.flags[t - 32] + fused_credit_index(me), e - 1);
  __syncthreads();
  constexpr int64_t kItem = static_cast<int64_t>(U) * kTile;
  const int64_t items = (a.n + kItem - 1) / kItem;
  float* const box = a.lbox[me] + par * a.lcap;
  uint32_t* const ybox = a.lybox[me] + par * (a.lycap / 2);
  // item i = j + k * G; two-shot owner (k + j) % W, compact sender-slot index j + (k / W) * G (unique per owner)
  // phase A: push
  for (int64_t k = 0, i = j; i < items; ++k, i += G) {
    const int owner = static_cast<int>((k + j) % W);
    if (TWO && owner == me) continue;
    const int64_t dst = TWO ? (j + (k / W) * G) * kItem : i * kItem;
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int64_t c = i * kItem + u * kTile + t * 4;
      if (c >= a.n) break;
      const float4 v = __ldg(reinterpret_cast<const float4*>(a.in + c));
      const int64_t off = par * a.lcap + me * a.lslot + dst + u * kTile + t * 4;
      if (TWO) {
        st_lamport4(a.lbox[owner] + off, v);
      } else {
#pragma unroll
        for (int q = 0; q < W; ++q)
          if (q != me) st_lamport4(a.lbox[q] + off, v);
      }
    }
  }
  // phase B: fold (one-shot: every item; two-shot: owned items, then publish the BF16 pairs)
  for (int64_t k = 0, i = j; i < items; ++k, i += G) {
    if (TWO && static_cast<int>((k + j) % W) != me) continue;
    const int64_t src = TWO ? (j + (k / W) * G) * kItem : i * kItem;
    float4 v[U][W];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int64_t c = i * kItem + u * kTile + t * 4;
      if (c >= a.n) break;
#pragma unroll
      for (int q = 0; q < W; ++q)
        if (q == me) v[u][q] = __ldg(reinterpret_cast<const float4*>(a.in + c));
        else ld_lamport4_raw(box + q * a.lslot + src + u * kTile + t * 4, v[u][q]);
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int64_t c = i * kItem + u * kTile + t * 4;
      if (c >= a.n) break;
#pragma unroll
      for (int q = 0; q < W; ++q)
        if (q != me) lamport_settle4(box + q * a.lslot + src + u * kTile + t * 4, v[u][q]);
      const float4 r4 = tree4<W>(v[u]);
      const uint2 r = make_uint2(lamport_pair(r4.x, r4.y), lamport_pair(r4.z, r4.w));
      *reinterpret_cast<uint2*>(a.out + c) = r;
      if (TWO) {
#pragma unroll
        for (int q = 0; q < W; ++q)
          if (q != me)
            asm volatile("st.relaxed.sys.global.v2.b32 [%0], {%1, %2};" ::"l"(a.lybox[q] + par * (a.lycap / 2) + c / 2), "r"(r.x),
                         "r"(r.y)
                         : "memory");
      }
    }
  }
  // phase C (two-shot): gather items folded by peers
  if (TWO) {
    for (int64_t k = 0, i = j; i < items; ++k, i += G) {
      if (static_cast<int>((k + j) % W) == me) continue;
      uint2 r[U];
#pragma unroll
      for (int u = 0; u < U; ++u) {
        const int64_t c = i * kItem + u * kTile + t * 4;
        if (c >= a.n) break;
        asm volatile("ld.relaxed.sys.global.v2.b32 {%0, %1}, [%2];" : "=r"(r[u].x), "=r"(r[u].y) : "l"(ybox + c / 2) : "memory");
      }
#pragma unroll
      for (int u = 0; u < U; ++u) {
        const int64_t c = i * kItem + u * kTile + t * 4;
        if (c >= a.n) break;
        uint32_t* p = ybox + c / 2;
        const uint64_t start = global_ns();
        while (r[u].x == kSentinel || r[u].y == kSentinel) {
          if (global_ns() - start > 20000000000ull) {
            printf("detreduce: fused result word timeout (slot %d)\n", j);
            __trap();
          }
          asm volatile("ld.relaxed.sys.global.v2.b32 {%0, %1}, [%2];" : "=r"(r[u].x), "=r"(r[u].y) : "l"(p) : "memory");
        }
        asm volatile("st.relaxed.sys.global.v2.b32 [%0], {%1, %1};" ::"l"(p), "r"(kSentinel) : "memory");
        *reinterpret_cast<uint2*>(a.out + c) = r[u];
      }
    }
  }
}

// Standalone producer push (reference for detreduce_push.cuh; does not advance the tagged epochs).
__global__ __launch_bounds__(kThreads) void tree_push_kernel(const float* in, int64_t n, detreduce_push::PushDesc d, bool lamport) {
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  for (int64_t c = (static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x) * 4; c < n;
       c += static_cast<int64_t>(gridDim.x) * kThreads * 4)
    if (lamport) detreduce_push::push4_lamport(d, c, __ldg(reinterpret_cast<const float4*>(in + c)));
    else detreduce_push::push4(d, c, __ldg(reinterpret_cast<const float4*>(in + c)));
}

// ------------------------------------------------------------------------------ MoE terms combine
struct TermsArgs {
  const float* roots;        // [R, H] this rank's FP32 expert roots
  const int* dest;           // [M, S] root row of each (token, slot) this rank owns, -1 otherwise
  const float* weights;      // [M, k] routed weights; slots >= k (shared) use 1.0
  __nv_bfloat16* out;        // [M, H]
  float* terms[kMaxWorld];   // each rank's term inbox, 2 halves of [cap_rows, S, H] FP32
  __nv_bfloat16* y[kMaxWorld];  // each rank's BF16 row inbox, 2 halves of [cap_y, H] (two-shot)
  uint64_t* tterms[kMaxWorld];  // each rank's tagged term box (zero-initialized), 2 halves of [cap_t, S, H] words
  int64_t terms_half, y_half, tterms_half;   // elements (words) per half
  int M, H, S, k;
  int two_shot;
  int rows_per;              // two-shot: rows owned per rank (owner(t) = t / rows_per)
};

__device__ __forceinline__ float4 product(const TermsArgs& a, int t, int s, int r, int c) {
  const float p = s < a.k ? __ldg(a.weights + static_cast<int64_t>(t) * a.k + s) : 1.0f;
  return mul4(__ldg(reinterpret_cast<const float4*>(a.roots + static_cast<int64_t>(r) * a.H + c)), p);
}

// fold row t's tile at column c: own slots recomputed from local roots, others read from the inbox
// row `slot_row` (one-shot: t, two-shot: t - owner * rows_per) of this rank's half `par`.
__device__ __forceinline__ float4 fold(const TermsArgs& a, int rank, int par, int t, int slot_row, int c) {
  const float* inbox = a.terms[rank] + par * a.terms_half + static_cast<int64_t>(slot_row) * a.S * a.H + c;
  float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
  for (int s = 0; s < a.S; ++s) {
    const int r = __ldg(a.dest + static_cast<int64_t>(t) * a.S + s);
    const float4 m = r >= 0 ? product(a, t, s, r, c) : ldcg4(inbox + static_cast<int64_t>(s) * a.H);
    acc = s == 0 ? m : add4(acc, m);
  }
  return acc;
}

__global__ __launch_bounds__(kThreads) void terms_combine_kernel(TermsArgs a, SyncState s) {
  asm volatile("griddepcontrol.wait;" ::: "memory");   // PDL: the expert GEMMs are complete
  int e, par;
  begin(s, e, par);
  const int col_tiles = (a.H + kTile - 1) / kTile;
  const int tiles = a.M * col_tiles;
  // phase A: push the products of owned slots (one-shot: to every peer; two-shot: to the row owner)
  for (int tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
    const int t = tile / col_tiles;
    const int c = (tile % col_tiles) * kTile + threadIdx.x * 4;
    if (c >= a.H) continue;
    const int owner = a.two_shot ? t / a.rows_per : -1;
    if (owner == s.rank) continue;   // own rows fold from local roots
    const int slot_row = a.two_shot ? t - owner * a.rows_per : t;
    for (int sl = 0; sl < a.S; ++sl) {
      const int r = __ldg(a.dest + static_cast<int64_t>(t) * a.S + sl);
      if (r < 0) continue;
      const float4 m = product(a, t, sl, r, c);
      const int64_t at = par * a.terms_half + (static_cast<int64_t>(slot_row) * a.S + sl) * a.H + c;
      if (a.two_shot) {
        *reinterpret_cast<float4*>(a.terms[owner] + at) = m;
      } else {
        for (int q = 0; q < s.world; ++q)
          if (q != s.rank) *reinterpret_cast<float4*>(a.terms[q] + at) = m;
      }
    }
  }
  publish(s, 0, e);
  await(s, 0, e);
  for (int tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
    const int t = tile / col_tiles;
    const int c = (tile % col_tiles) * kTile + threadIdx.x * 4;
    if (c >= a.H) continue;
    const int owner = a.two_shot ? t / a.rows_per : s.rank;
    if (owner != s.rank) continue;
    const uint2 r = pack_bf16x4(fold(a, s.rank, par, t, a.two_shot ? t - owner * a.rows_per : t, c));
    const int64_t at = static_cast<int64_t>(t) * a.H + c;
    *reinterpret_cast<uint2*>(a.out + at) = r;
    if (a.two_shot)
      for (int q = 0; q < s.world; ++q)
        if (q != s.rank) *reinterpret_cast<uint2*>(a.y[q] + par * a.y_half + at) = r;
  }
  if (a.two_shot) {
    publish(s, 1, e);
    await(s, 1, e);
    for (int tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
      const int t = tile / col_tiles;
      const int c = (tile % col_tiles) * kTile + threadIdx.x * 4;
      if (c >= a.H || t / a.rows_per == s.rank) continue;
      const int64_t at = static_cast<int64_t>(t) * a.H + c;
      *reinterpret_cast<uint2*>(a.out + at) = __ldcg(reinterpret_cast<const uint2*>(a.y[s.rank] + par * a.y_half + at));
    }
  }
  finish(s, e);
}

// Tagged one-shot terms (decode): owners push each owned slot's product as tagged 64-bit words to every
// peer (call tag = CTA b's call count, dedicated zero-initialized box); receivers poll the words, then fold
// every row locally.
//
// Back-pressure (credits). Unlike the dense tagged form, a (CTA, peer) pair may exchange no word in a call
// (the peer owns none of the slots of CTA b's rows; at M = 1, W = 4 that is ~10% per peer and call), so the
// data alone does not keep a writer within one call of its reader: a writer could reach call e + 2 and
// overwrite the parity half (and tag) that a slower reader still polls for call e -- the "tagged word
// timeout" hang. Every CTA therefore (1) before pushing call e's words, waits until every peer's CTA b has
// finished call e - 2 (the previous user of this parity half), and (2) after its own polls, publishes
// "finished e" to every peer. The wait reads a local word that the peer released at least one full call
// earlier, so it is normally already satisfied (no extra round trip on the critical path). Progress: the
// wait at e depends only on calls <= e - 2 of the peers, which depend on pushes of calls <= e - 2.
__global__ __launch_bounds__(kThreads) void terms_tagged_kernel(TermsArgs a, SyncState s) {
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  // Fixed grid (kTagGrid) and fixed tile -> CTA map: every inbox word is always written and read by the
  // same CTA index, so that CTA's call count is a strictly increasing tag for the word and its parity
  // selects the half. No global counter, no atomics.
  const int e = *reinterpret_cast<volatile int*>(s.tepoch + blockIdx.x) + 1;
  const uint32_t tag = static_cast<uint32_t>(e);
  const int par = e & 1;
  const int col_tiles = (a.H + kTile - 1) / kTile;
  const int tiles = a.M * col_tiles;
  const int64_t half = par * a.tterms_half;
  credit_wait(s, e, static_cast<int>(blockIdx.x) < tiles);   // CTAs without tiles write nothing (and read nothing)
  for (int tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
    const int t = tile / col_tiles;
    const int c = (tile % col_tiles) * kTile + threadIdx.x * 4;
    if (c >= a.H) continue;
    for (int sl = 0; sl < a.S; ++sl) {
      const int r = __ldg(a.dest + static_cast<int64_t>(t) * a.S + sl);
      if (r < 0) continue;
      const float4 m = product(a, t, sl, r, c);
      const int64_t at = half + (static_cast<int64_t>(t) * a.S + sl) * a.H + c;
      for (int q = 0; q < s.world; ++q)
        if (q != s.rank) st_tag4(a.tterms[q] + at, m, tag);
    }
  }
  const uint64_t* local = a.tterms[s.rank] + half;
  for (int tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
    const int t = tile / col_tiles;
    const int c = (tile % col_tiles) * kTile + threadIdx.x * 4;
    if (c >= a.H) continue;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int sl = 0; sl < a.S; ++sl) {
      const int r = __ldg(a.dest + static_cast<int64_t>(t) * a.S + sl);
      const float4 m = r >= 0 ? product(a, t, sl, r, c)
                              : ld_tag4(local + (static_cast<int64_t>(t) * a.S + sl) * a.H + c, tag);
      acc = sl == 0 ? m : add4(acc, m);
    }
    *reinterpret_cast<uint2*>(a.out + static_cast<int64_t>(t) * a.H + c) = pack_bf16x4(acc);
  }
  credit_post(s, e);   // this CTA's reads of call e are complete: peers may reuse this half for call e + 2
  if (threadIdx.x == 0) s.tepoch[blockIdx.x] = e;
}

// ---------------------------------------------------------------------------------- MoE prep
// GLM ep_prep (moe/clamped_swiglu_sm100/csrc/ep_chain.cu) with the expert range test replaced by the
// local-group table `lidx[e]` (-1: not owned) and the shared group restricted to rows [s0, s1).
// Plan: groups in local order, rows of a group ascending by token. One CTA per token.
__global__ __launch_bounds__(1024) void moe_prep_kernel(const __nv_bfloat16* __restrict__ x, int M, int H,
                                                         const int* __restrict__ ids, int k, const int* __restrict__ lidx,
                                                         int E, int shared_group, int s0, int s1, int G,
                                                         uint8_t* __restrict__ codes, float* __restrict__ sa, int sa_ld,
                                                         int* __restrict__ offsets, int* __restrict__ dest,
                                                         int* __restrict__ done) {
  __shared__ int cnt_all[kMaxGroups];
  __shared__ int cnt_before[kMaxGroups];
  __shared__ int base[kMaxGroups + 1];
  __shared__ int dst[kMaxSlots];
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  const int t = blockIdx.x;
  const int S = k + 1;
  for (int g = threadIdx.x; g < G; g += blockDim.x) cnt_all[g] = 0, cnt_before[g] = 0;
  __syncthreads();
  const int n = M * k;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int e = __ldg(ids + i);
    const int g = (e >= 0 && e < E) ? __ldg(lidx + e) : -1;
    if (g >= 0) {
      atomicAdd(&cnt_all[g], 1);                     // integer shared-memory counts
      if (i / k < t) atomicAdd(&cnt_before[g], 1);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0 && shared_group >= 0) {
    cnt_all[shared_group] += s1 - s0;
    cnt_before[shared_group] += min(max(t - s0, 0), s1 - s0);
  }
  __syncthreads();
  if (threadIdx.x < 32) {
    const int lane = threadIdx.x;
    const int per = (G + 31) / 32;
    const int g0 = min(G, lane * per), g1 = min(G, g0 + per);
    int sum = 0;
    for (int g = g0; g < g1; ++g) sum += cnt_all[g];
    int inc = sum;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      const int v = __shfl_up_sync(0xffffffffu, inc, off);
      if (lane >= off) inc += v;
    }
    int run = inc - sum;
    for (int g = g0; g < g1; ++g) {
      base[g] = run;
      run += cnt_all[g];
    }
    if (lane == 31) base[G] = inc;
  }
  __syncthreads();
  if (t == 0) {
    for (int g = threadIdx.x; g <= G; g += blockDim.x) offsets[g] = base[g];
    if (done != nullptr)
      for (int g = threadIdx.x; g < G; g += blockDim.x) done[g] = 0;
  }
  if (threadIdx.x < S) {
    const int sl = threadIdx.x;
    int g = -1;
    if (sl < k) {
      const int e = __ldg(ids + static_cast<int64_t>(t) * k + sl);
      if (e >= 0 && e < E) g = __ldg(lidx + e);
    } else if (t >= s0 && t < s1) {
      g = shared_group;
    }
    const int row = g >= 0 ? base[g] + cnt_before[g] : -1;
    dst[sl] = row;
    dest[static_cast<int64_t>(t) * S + sl] = row;
  }
  __syncthreads();
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, warps = blockDim.x / 32;
  for (int g = warp; g < H / 128; g += warps) {
    const int c = g * 128 + lane * 4;
    const uint2 raw = *reinterpret_cast<const uint2*>(x + static_cast<int64_t>(t) * H + c);
    const float v0 = __uint_as_float(raw.x << 16), v1 = __uint_as_float(raw.x & 0xffff0000u);
    const float v2 = __uint_as_float(raw.y << 16), v3 = __uint_as_float(raw.y & 0xffff0000u);
    float amax = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
    const float scale = glmfp8::act_scale(amax);
    const uint32_t q = static_cast<uint32_t>(glmfp8::act_code(v0, scale)) |
                       (static_cast<uint32_t>(glmfp8::act_code(v1, scale)) << 8) |
                       (static_cast<uint32_t>(glmfp8::act_code(v2, scale)) << 16) |
                       (static_cast<uint32_t>(glmfp8::act_code(v3, scale)) << 24);
    for (int sl = 0; sl < S; ++sl) {
      const int row = dst[sl];
      if (row < 0) continue;
      *reinterpret_cast<uint32_t*>(codes + static_cast<int64_t>(row) * H + c) = q;
      if (lane == 0) sa[static_cast<int64_t>(g) * sa_ld + row] = scale;
    }
  }
}

template <typename... KArgs, typename... Args>
void launch_pdl(void (*kernel)(KArgs...), dim3 grid, dim3 block, cudaStream_t stream, Args&&... args) {
  cudaLaunchConfig_t config = {};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  config.attrs = attrs;
  config.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, std::forward<Args>(args)...));
}

template <typename K>
int64_t resident_grid(K kernel, int64_t want) {
  int dev = 0, sms = 0, per_sm = 0;
  C10_CUDA_CHECK(cudaGetDevice(&dev));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, kThreads, 0));
  TORCH_CHECK(per_sm >= 1, "detreduce kernel cannot be resident");
  return std::max<int64_t>(1, std::min<int64_t>({want, static_cast<int64_t>(per_sm) * sms, kMaxCta}));
}

// sync: [flags_0 .. flags_{W-1}] peer pointers; state tensor int32 [kMaxCta + 2] (epoch, call, finished)
SyncState make_sync(const std::vector<int64_t>& flag_ptrs, at::Tensor& state, int64_t rank) {
  TORCH_CHECK(state.is_cuda() && state.scalar_type() == at::kInt && state.is_contiguous() && state.numel() == kMaxCta + 2 + kTagGrid,
              "state: int32 [kMaxCta + 2 + kTagGrid]");
  const int world = static_cast<int>(flag_ptrs.size());
  TORCH_CHECK(world >= 1 && world <= kMaxWorld && rank >= 0 && rank < world, "world/rank");
  SyncState s{};
  for (int q = 0; q < world; ++q) s.flags[q] = reinterpret_cast<int*>(flag_ptrs[q]);
  s.epoch = state.data_ptr<int>();
  s.call = s.epoch + kMaxCta;
  s.finished = s.epoch + kMaxCta + 1;
  s.tepoch = s.epoch + kMaxCta + 2;
  s.world = world;
  s.rank = static_cast<int>(rank);
  return s;
}

}  // namespace

int64_t max_cta() { return kMaxCta; }
int64_t max_world() { return kMaxWorld; }
int64_t tile_elems() { return kTile; }
int64_t tag_grid() { return kTagGrid; }

// inbox_ptrs / ybox_ptrs: each rank's symmetric FP32 inbox / BF16 ybox base (2 halves of `cap` each).
int64_t tree_slot(int64_t cap, int64_t world) { return (cap / world) / kTile * kTile; }
// largest payload (elements) of one Lamport two-shot launch: the compact slot index b + (k / W) * G must stay inside a sender slot
int64_t lamport2_chunk(int64_t lcap, int64_t lycap, int64_t world) {
  const int64_t gt = static_cast<int64_t>(kTagGrid) * kTile;
  const int64_t rounds = tree_slot(lcap, world) / gt;     // compact rounds per slot
  return std::min<int64_t>(lycap, rounds * world * gt);
}

int64_t lamport2_chunk_host(int64_t lcap, int64_t lycap, int64_t world) { return lamport2_chunk(lcap, lycap, world); }

void tree_allreduce(const at::Tensor& in, at::Tensor& out, const std::vector<int64_t>& inbox_ptrs,
                    const std::vector<int64_t>& ybox_ptrs, int64_t cap, const std::vector<int64_t>& tbox_ptrs,
                    int64_t tcap, const std::vector<int64_t>& lbox_ptrs, int64_t lcap, const std::vector<int64_t>& lybox_ptrs, int64_t lycap,
                    const std::vector<int64_t>& flag_ptrs, at::Tensor& state, int64_t rank, int64_t two_shot, int64_t grid) {
  const at::cuda::CUDAGuard guard(in.device());
  TORCH_CHECK(in.is_cuda() && in.is_contiguous() && in.scalar_type() == at::kFloat, "in: contiguous FP32");
  TORCH_CHECK(out.is_contiguous() && out.scalar_type() == at::kBFloat16 && out.numel() == in.numel(), "out: BF16, same numel");
  const int64_t n = in.numel();
  TORCH_CHECK(reinterpret_cast<uintptr_t>(in.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(out.data_ptr()) % 8 == 0,
              "alignment");
  SyncState s = make_sync(flag_ptrs, state, rank);
  const int W = s.world;
  TORCH_CHECK(W == 2 || W == 4 || W == 8 || W == 16, "tree all-reduce serves W in {2, 4, 8, 16}");
  TORCH_CHECK(static_cast<int>(inbox_ptrs.size()) == W && static_cast<int>(ybox_ptrs.size()) == W, "pointer tables");
  TORCH_CHECK(cap % 4 == 0, "cap");
  const int64_t slot = tree_slot(cap, W);
  const int64_t tiles = (n + kTile - 1) / kTile;
  if (two_shot == 7) {
    TORCH_CHECK(static_cast<int>(lbox_ptrs.size()) == W && static_cast<int>(lybox_ptrs.size()) == W && n % 4 == 0 &&
                    n <= std::min<int64_t>(lycap, lamport2_chunk(lcap, lycap, W)),
                "lamport two-shot payload: whole float4 groups within lamport2_chunk");
  } else if (two_shot == 5 || two_shot == 6) {
    TORCH_CHECK(static_cast<int>(lbox_ptrs.size()) == W && n % 4 == 0 && n <= tree_slot(lcap, W),
                "lamport payload: whole float4 groups within the slot");
  } else if (two_shot == 2 || two_shot == 3) {
    TORCH_CHECK(static_cast<int>(tbox_ptrs.size()) == W && n <= tree_slot(tcap, W), "tagged one-shot payload exceeds the tagged slot");
    TORCH_CHECK(two_shot == 2 || n % 4 == 0, "prepushed payloads are whole float4 groups");
  } else if (two_shot) {
    TORCH_CHECK(n <= cap && (tiles + W - 1) / W * kTile <= slot, "two-shot payload exceeds the prepared inbox");
  } else {
    TORCH_CHECK(n <= slot, "one-shot payload exceeds the prepared inbox slot");
  }
  if (n == 0) return;
  TreeArgs a{};
  a.in = in.data_ptr<float>();
  a.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  for (int q = 0; q < W; ++q) {
    a.inbox[q] = reinterpret_cast<float*>(inbox_ptrs[q]);
    a.ybox[q] = reinterpret_cast<__nv_bfloat16*>(ybox_ptrs[q]);
  }
  a.n = n;
  a.cap = cap;
  a.slot = slot;
  for (int q = 0; q < static_cast<int>(tbox_ptrs.size()) && q < W; ++q) a.tbox[q] = reinterpret_cast<uint64_t*>(tbox_ptrs[q]);
  a.tcap = tcap;
  a.tslot = tree_slot(tcap, W);
  for (int q = 0; q < static_cast<int>(lbox_ptrs.size()) && q < W; ++q) a.lbox[q] = reinterpret_cast<float*>(lbox_ptrs[q]);
  a.lcap = lcap;
  a.lslot = tree_slot(lcap, W);
  for (int q = 0; q < static_cast<int>(lybox_ptrs.size()) && q < W; ++q) a.lybox[q] = reinterpret_cast<uint32_t*>(lybox_ptrs[q]);
  a.lycap = lycap;
  a.two_shot = static_cast<int>(two_shot);
  auto stream = at::cuda::getCurrentCUDAStream();
#define DETREDUCE_TREE(WW)                                                                            \
  case WW: {                                                                                          \
    auto kern = two_shot == 7 ? tree_lamport2_kernel<WW> : two_shot >= 5 ? tree_lamport_kernel<WW>                                             \
                : two_shot >= 2 ? tree_tagged_kernel<WW> : tree_allreduce_kernel<WW>;                     \
    const int64_t g = two_shot >= 2 ? kTagGrid : resident_grid(kern, std::min(tiles, grid > 0 ? grid : tiles)); \
    launch_pdl(kern, dim3(static_cast<unsigned>(g)), dim3(kThreads), stream, a, s);                   \
    break;                                                                                            \
  }
  switch (W) {
    DETREDUCE_TREE(2)
    DETREDUCE_TREE(4)
    DETREDUCE_TREE(8)
    DETREDUCE_TREE(16)
  }
#undef DETREDUCE_TREE
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void phased_entry(const at::Tensor& in, at::Tensor& out, const std::vector<int64_t>& inbox_ptrs,
                    const std::vector<int64_t>& ybox_ptrs, int64_t cap, const std::vector<int64_t>& tbox_ptrs,
                    int64_t tcap, const std::vector<int64_t>& lbox_ptrs, int64_t lcap, const std::vector<int64_t>& lybox_ptrs, int64_t lycap,
                    const std::vector<int64_t>& flag_ptrs, at::Tensor& state, int64_t rank, int64_t two_shot, int64_t grid) {
  TORCH_CHECK(two_shot == 8, "phased_entry requires exclusive mode8 workspace");
  TORCH_CHECK(out.device() == in.device() && state.device() == in.device(), "phased tensors must share device");
  const at::cuda::CUDAGuard guard(in.device());
  TORCH_CHECK(in.is_cuda() && in.is_contiguous() && in.scalar_type() == at::kFloat, "in: contiguous FP32");
  TORCH_CHECK(out.is_contiguous() && out.scalar_type() == at::kBFloat16 && out.numel() == in.numel(), "out: BF16, same numel");
  const int64_t n = in.numel();
  TORCH_CHECK(reinterpret_cast<uintptr_t>(in.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(out.data_ptr()) % 8 == 0,
              "alignment");
  SyncState s = make_sync(flag_ptrs, state, rank);
  const int W = s.world;
  TORCH_CHECK(W == 2 || W == 4 || W == 8 || W == 16, "tree all-reduce serves W in {2, 4, 8, 16}");
  TORCH_CHECK(static_cast<int>(inbox_ptrs.size()) == W && static_cast<int>(ybox_ptrs.size()) == W, "pointer tables");
  TORCH_CHECK(cap % 4 == 0, "cap");
  const int64_t slot = tree_slot(cap, W);
  const int64_t tiles = (n + kTile - 1) / kTile;
  if (two_shot == 8) {
    TORCH_CHECK(static_cast<int>(lbox_ptrs.size()) == W && static_cast<int>(lybox_ptrs.size()) == W && n % 4 == 0 &&
                    n <= std::min<int64_t>(lycap, lamport2_chunk(lcap, lycap, W)),
                "lamport two-shot payload: whole float4 groups within lamport2_chunk");
  } else if (two_shot == 5 || two_shot == 6) {
    TORCH_CHECK(static_cast<int>(lbox_ptrs.size()) == W && n % 4 == 0 && n <= tree_slot(lcap, W),
                "lamport payload: whole float4 groups within the slot");
  } else if (two_shot == 2 || two_shot == 3) {
    TORCH_CHECK(static_cast<int>(tbox_ptrs.size()) == W && n <= tree_slot(tcap, W), "tagged one-shot payload exceeds the tagged slot");
    TORCH_CHECK(two_shot == 2 || n % 4 == 0, "prepushed payloads are whole float4 groups");
  } else if (two_shot) {
    TORCH_CHECK(n <= cap && (tiles + W - 1) / W * kTile <= slot, "two-shot payload exceeds the prepared inbox");
  } else {
    TORCH_CHECK(n <= slot, "one-shot payload exceeds the prepared inbox slot");
  }
  if (n == 0) return;
  TreeArgs a{};
  a.in = in.data_ptr<float>();
  a.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  for (int q = 0; q < W; ++q) {
    a.inbox[q] = reinterpret_cast<float*>(inbox_ptrs[q]);
    a.ybox[q] = reinterpret_cast<__nv_bfloat16*>(ybox_ptrs[q]);
  }
  a.n = n;
  a.cap = cap;
  a.slot = slot;
  for (int q = 0; q < static_cast<int>(tbox_ptrs.size()) && q < W; ++q) a.tbox[q] = reinterpret_cast<uint64_t*>(tbox_ptrs[q]);
  a.tcap = tcap;
  a.tslot = tree_slot(tcap, W);
  for (int q = 0; q < static_cast<int>(lbox_ptrs.size()) && q < W; ++q) a.lbox[q] = reinterpret_cast<float*>(lbox_ptrs[q]);
  a.lcap = lcap;
  a.lslot = tree_slot(lcap, W);
  for (int q = 0; q < static_cast<int>(lybox_ptrs.size()) && q < W; ++q) a.lybox[q] = reinterpret_cast<uint32_t*>(lybox_ptrs[q]);
  a.lycap = lycap;
  a.two_shot = static_cast<int>(two_shot);
  auto stream = at::cuda::getCurrentCUDAStream();
#define DETREDUCE_TREE(WW) \
  case WW: { \
    tree_phased_push_kernel<WW><<<kTagGrid, kThreads, 0, stream>>>(a, s); \
    C10_CUDA_KERNEL_LAUNCH_CHECK(); \
    tree_phased_fold_kernel<WW><<<kTagGrid, kThreads, 0, stream>>>(a, s); \
    C10_CUDA_KERNEL_LAUNCH_CHECK(); \
    tree_phased_gather_kernel<WW><<<kTagGrid, kThreads, 0, stream>>>(a, s); \
    break; \
  }
  switch (W) {
    DETREDUCE_TREE(2)
    DETREDUCE_TREE(4)
    DETREDUCE_TREE(8)
    DETREDUCE_TREE(16)
  }
#undef DETREDUCE_TREE
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}



// Mode 9 (fused phased) host entry: one launch of tree_fused_kernel<W, two_shot, unroll> with `grid` CTAs.
// Payload limits: one-shot n <= lslot; two-shot G * ceil(ceil(items / G) / W) * unroll * kTile <= lslot (compact sender slots)
// and n <= lycap (BF16 half). The host (phased.py) chunks larger payloads.
int64_t fused_two_shot_chunk(int64_t lcap, int64_t lycap, int64_t world, int64_t grid, int64_t unroll) {
  const int64_t unit = grid * unroll * kTile;              // one round of G items
  const int64_t rounds = tree_slot(lcap, world) / unit;    // compact rounds per sender slot
  return std::min<int64_t>(lycap, rounds * world * unit);
}

void fused_entry(const at::Tensor& in, at::Tensor& out, const std::vector<int64_t>& lbox_ptrs, int64_t lcap,
                 const std::vector<int64_t>& lybox_ptrs, int64_t lycap, const std::vector<int64_t>& flag_ptrs, at::Tensor& state,
                 int64_t rank, int64_t two_shot, int64_t grid, int64_t unroll) {
  TORCH_CHECK(out.device() == in.device() && state.device() == in.device(), "fused tensors must share device");
  const at::cuda::CUDAGuard guard(in.device());
  TORCH_CHECK(in.is_cuda() && in.is_contiguous() && in.scalar_type() == at::kFloat, "in: contiguous FP32");
  TORCH_CHECK(out.is_contiguous() && out.scalar_type() == at::kBFloat16 && out.numel() == in.numel(), "out: BF16, same numel");
  const int64_t n = in.numel();
  TORCH_CHECK(reinterpret_cast<uintptr_t>(in.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(out.data_ptr()) % 8 == 0,
              "alignment");
  SyncState s = make_sync(flag_ptrs, state, rank);
  const int W = s.world;
  TORCH_CHECK(W == 2 || W == 4 || W == 8 || W == 16, "tree all-reduce serves W in {2, 4, 8, 16}");
  TORCH_CHECK(static_cast<int>(lbox_ptrs.size()) == W && static_cast<int>(lybox_ptrs.size()) == W, "pointer tables");
  TORCH_CHECK(unroll == 1 || unroll == 4, "unroll in {1, 4}");
  TORCH_CHECK(n % 4 == 0, "fused payload: whole float4 groups");
  if (n == 0) return;
  const int64_t item = unroll * kTile;
  const int64_t items = (n + item - 1) / item;
  TORCH_CHECK(grid >= 1 && grid <= items && grid <= kMaxCta, "fused grid in [1, min(items, kMaxCta)]");
  const int64_t lslot = tree_slot(lcap, W);
  if (two_shot) {
    const int64_t rounds = (items + grid - 1) / grid;
    TORCH_CHECK(grid * ((rounds + W - 1) / W) * item <= lslot && n <= lycap, "fused two-shot payload exceeds the compact slots");
  } else {
    TORCH_CHECK(n <= lslot, "fused one-shot payload exceeds the sender slot");
  }
  TreeArgs a{};
  a.in = in.data_ptr<float>();
  a.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  for (int q = 0; q < W; ++q) {
    a.lbox[q] = reinterpret_cast<float*>(lbox_ptrs[q]);
    a.lybox[q] = reinterpret_cast<uint32_t*>(lybox_ptrs[q]);
  }
  a.n = n;
  a.lcap = lcap;
  a.lslot = lslot;
  a.lycap = lycap;
  a.two_shot = 9;
  auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 g(static_cast<unsigned>(grid)), b(kThreads);
#define DETREDUCE_FUSED(WW)                                                     \
  case WW:                                                                      \
    if (two_shot) {                                                             \
      if (unroll == 4) tree_fused_kernel<WW, true, 4><<<g, b, 0, stream>>>(a, s);   \
      else tree_fused_kernel<WW, true, 1><<<g, b, 0, stream>>>(a, s);              \
    } else {                                                                    \
      if (unroll == 4) tree_fused_kernel<WW, false, 4><<<g, b, 0, stream>>>(a, s);  \
      else tree_fused_kernel<WW, false, 1><<<g, b, 0, stream>>>(a, s);             \
    }                                                                           \
    break;
  switch (W) {
    DETREDUCE_FUSED(2)
    DETREDUCE_FUSED(4)
    DETREDUCE_FUSED(8)
    DETREDUCE_FUSED(16)
  }
#undef DETREDUCE_FUSED
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void tree_push(const at::Tensor& in, const std::vector<int64_t>& tbox_ptrs, int64_t tcap, const std::vector<int64_t>& lbox_ptrs,
               int64_t lcap, bool lamport, at::Tensor& state, int64_t rank) {
  const at::cuda::CUDAGuard guard(in.device());
  TORCH_CHECK(in.is_cuda() && in.is_contiguous() && in.scalar_type() == at::kFloat && in.numel() % 4 == 0, "in: FP32, numel % 4 == 0");
  const int W = static_cast<int>(tbox_ptrs.size());
  TORCH_CHECK(W >= 2 && W <= kMaxWorld && rank >= 0 && rank < W, "world/rank");
  TORCH_CHECK(!lamport || static_cast<int>(lbox_ptrs.size()) == W, "lamport pointer table");
  TORCH_CHECK(state.numel() == kMaxCta + 2 + kTagGrid && state.scalar_type() == at::kInt, "state");
  const int64_t n = in.numel();
  TORCH_CHECK(n <= tree_slot(lamport ? lcap : tcap, W), "payload exceeds the slot");
  if (n == 0) return;
  detreduce_push::PushDesc d{};
  for (int q = 0; q < W; ++q) d.tbox[q] = reinterpret_cast<uint64_t*>(tbox_ptrs[q]);
  if (lamport)
    for (int q = 0; q < W; ++q) d.lbox[q] = reinterpret_cast<float*>(lbox_ptrs[q]);
  d.lcap = lcap;
  d.lslot = tree_slot(lcap, W);
  d.tcap = tcap;
  d.tslot = tree_slot(tcap, W);
  d.tepoch = state.data_ptr<int>() + kMaxCta + 2;
  d.rank = static_cast<int>(rank);
  d.world = W;
  const int64_t blocks = std::min<int64_t>((n / 4 + kThreads - 1) / kThreads, 1024);
  launch_pdl(tree_push_kernel, dim3(static_cast<unsigned>(blocks)), dim3(kThreads), at::cuda::getCurrentCUDAStream(),
             in.data_ptr<float>(), n, d, lamport);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void terms_combine(const at::Tensor& roots, const at::Tensor& dest, const at::Tensor& weights, at::Tensor& out,
                   const std::vector<int64_t>& terms_ptrs, int64_t terms_half, const std::vector<int64_t>& y_ptrs,
                   int64_t y_half, int64_t cap_rows, int64_t cap_y, const std::vector<int64_t>& tterms_ptrs,
                   int64_t tterms_half, int64_t cap_t, const std::vector<int64_t>& flag_ptrs,
                   at::Tensor& state, int64_t rank, int64_t two_shot, int64_t grid) {
  const at::cuda::CUDAGuard guard(roots.device());
  TORCH_CHECK(roots.is_contiguous() && roots.scalar_type() == at::kFloat && roots.dim() == 2, "roots FP32 [R, H]");
  TORCH_CHECK(out.is_contiguous() && out.scalar_type() == at::kBFloat16 && out.dim() == 2, "out BF16 [M, H]");
  const int M = static_cast<int>(out.size(0)), H = static_cast<int>(out.size(1));
  TORCH_CHECK(roots.size(1) == H && H % 4 == 0, "width");
  TORCH_CHECK(weights.is_contiguous() && weights.scalar_type() == at::kFloat && weights.dim() == 2 && weights.size(0) == M,
              "weights FP32 [M, k]");
  const int k = static_cast<int>(weights.size(1));
  TORCH_CHECK(dest.is_contiguous() && dest.scalar_type() == at::kInt && dest.dim() == 2 && dest.size(0) == M, "dest int32 [M, S]");
  const int S = static_cast<int>(dest.size(1));
  TORCH_CHECK(S >= k && S <= kMaxSlots, "slots");
  SyncState s = make_sync(flag_ptrs, state, rank);
  const int W = s.world;
  TORCH_CHECK(W >= 2, "terms combine needs EP >= 2");
  TORCH_CHECK(static_cast<int>(terms_ptrs.size()) == W && static_cast<int>(y_ptrs.size()) == W, "pointer tables");
  if (M == 0) return;
  TermsArgs a{};
  a.roots = roots.data_ptr<float>();
  a.dest = dest.data_ptr<int>();
  a.weights = weights.data_ptr<float>();
  a.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  for (int q = 0; q < W; ++q) {
    a.terms[q] = reinterpret_cast<float*>(terms_ptrs[q]);
    a.y[q] = reinterpret_cast<__nv_bfloat16*>(y_ptrs[q]);
  }
  a.terms_half = terms_half;
  a.y_half = y_half;
  a.M = M, a.H = H, a.S = S, a.k = k;
  a.two_shot = static_cast<int>(two_shot);
  a.rows_per = (M + W - 1) / W;
  if (two_shot == 2) {
    TORCH_CHECK(static_cast<int>(tterms_ptrs.size()) == W && M <= cap_t &&
                cap_t * S * static_cast<int64_t>(H) <= tterms_half, "tagged chunk exceeds the prepared tagged box");
    for (int q = 0; q < W; ++q) a.tterms[q] = reinterpret_cast<uint64_t*>(tterms_ptrs[q]);
    a.tterms_half = tterms_half;
  } else if (two_shot) {
    TORCH_CHECK(a.rows_per <= cap_rows && M <= cap_y, "two-shot chunk exceeds the prepared inboxes");
  } else {
    TORCH_CHECK(M <= cap_rows, "one-shot chunk exceeds the prepared inbox");
  }
  TORCH_CHECK(cap_rows * S * static_cast<int64_t>(H) <= terms_half && cap_y * static_cast<int64_t>(H) <= y_half, "halves");
  const int64_t tiles = static_cast<int64_t>(M) * ((H + kTile - 1) / kTile);
  auto kern = two_shot == 2 ? terms_tagged_kernel : terms_combine_kernel;
  const int64_t g = two_shot == 2 ? kTagGrid : resident_grid(kern, std::min(tiles, grid > 0 ? grid : tiles));
  launch_pdl(kern, dim3(static_cast<unsigned>(g)), dim3(kThreads), at::cuda::getCurrentCUDAStream(), a, s);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void moe_prep(const at::Tensor& x, const at::Tensor& ids, const at::Tensor& lidx, int64_t shared_group, int64_t s0,
              int64_t s1, at::Tensor& codes, at::Tensor& sa, at::Tensor& offsets, at::Tensor& dest,
              const c10::optional<at::Tensor>& done) {
  const at::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.is_contiguous() && x.scalar_type() == at::kBFloat16, "x: BF16 [M, H]");
  const int M = static_cast<int>(x.size(0)), H = static_cast<int>(x.size(1));
  TORCH_CHECK(H % 128 == 0, "H must be a multiple of 128");
  TORCH_CHECK(ids.is_contiguous() && ids.scalar_type() == at::kInt && ids.dim() == 2 && ids.size(0) == M, "ids: int32 [M, k]");
  const int k = static_cast<int>(ids.size(1)), S = k + 1;
  TORCH_CHECK(S <= kMaxSlots, "too many slots");
  TORCH_CHECK(lidx.is_contiguous() && lidx.scalar_type() == at::kInt && lidx.dim() == 1, "lidx: int32 [E]");
  const int E = static_cast<int>(lidx.size(0));
  const int G = static_cast<int>(offsets.numel()) - 1;
  TORCH_CHECK(G >= 1 && G < kMaxGroups && offsets.scalar_type() == at::kInt && offsets.is_contiguous(), "offsets: int32 [G + 1]");
  TORCH_CHECK(shared_group < G && 0 <= s0 && s0 <= s1 && s1 <= M, "shared group / rows");
  TORCH_CHECK(dest.is_contiguous() && dest.scalar_type() == at::kInt && dest.numel() == static_cast<int64_t>(M) * S, "dest: int32 [M, S]");
  const int64_t R = codes.size(0);
  TORCH_CHECK(codes.is_contiguous() && codes.scalar_type() == at::kFloat8_e4m3fn && codes.size(1) == H, "codes: E4M3 [R, H]");
  TORCH_CHECK(sa.is_contiguous() && sa.scalar_type() == at::kFloat && sa.size(0) == H / 128 && sa.size(1) == (R + 3) / 4 * 4,
              "sa: FP32 [H/128, R rounded up to 4]");
  int* done_p = nullptr;
  if (done.has_value()) {
    TORCH_CHECK(done->is_contiguous() && done->scalar_type() == at::kInt && done->numel() >= G, "done: int32 [G]");
    done_p = done->data_ptr<int>();
  }
  if (M == 0) return;
  // the kernel does 8-byte loads of x and 4-byte stores into codes: fail with a stack instead of an async cudaErrorMisalignedAddress
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 8 == 0 && reinterpret_cast<uintptr_t>(codes.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(sa.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(dest.data_ptr()) % 4 == 0,
              "moe_prep: misaligned x / codes / sa");
  launch_pdl(moe_prep_kernel, dim3(M), dim3(1024), at::cuda::getCurrentCUDAStream(),
             reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), M, H, static_cast<const int*>(ids.data_ptr<int>()), k,
             static_cast<const int*>(lidx.data_ptr<int>()), E, static_cast<int>(shared_group), static_cast<int>(s0),
             static_cast<int>(s1), G, reinterpret_cast<uint8_t*>(codes.data_ptr()), sa.data_ptr<float>(),
             static_cast<int>(sa.size(1)), offsets.data_ptr<int>(), dest.data_ptr<int>(), done_p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


// Explicit GPU-query API; never called during CPU prebuild.
template<typename K>
std::vector<int64_t> resource_row(K kernel, int phase) {
  cudaFuncAttributes a{}; int blocks=0;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&a, kernel));
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, kThreads, 0));
  return {phase, a.numRegs, static_cast<int64_t>(a.sharedSizeBytes), static_cast<int64_t>(a.localSizeBytes),
          a.maxThreadsPerBlock, blocks, kThreads, kTagGrid};
}
std::vector<std::vector<int64_t>> phased_resources(int64_t world) {
#define RESOURCE(W) case W: return {resource_row(tree_phased_push_kernel<W>,0), resource_row(tree_phased_fold_kernel<W>,1), resource_row(tree_phased_gather_kernel<W>,2), resource_row(tree_lamport2_kernel<W>,3)};
  switch(world) { RESOURCE(2) RESOURCE(4) RESOURCE(8) RESOURCE(16) }
#undef RESOURCE
  TORCH_CHECK(false, "world must be 2,4,8,16");
}
std::vector<int64_t> device_resources() {
  int device=0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp p{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&p, device));
  return {device, p.multiProcessorCount, p.regsPerMultiprocessor, p.regsPerBlock,
          p.maxThreadsPerMultiProcessor, p.maxThreadsPerBlock, p.warpSize,
          static_cast<int64_t>(p.sharedMemPerMultiprocessor), p.major, p.minor};
}
}  // namespace detreduce
