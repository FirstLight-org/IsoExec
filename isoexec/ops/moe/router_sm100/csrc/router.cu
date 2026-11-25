// SPDX-License-Identifier: Apache-2.0
//
// moe.route on SM100 for the DeepSeek-class routers:
//
//   route_select_only   the selection program (select.cuh) over caller-supplied fp32 logits.
//   route_fused         the router GEMM on tcgen05 (bf16 x bf16 -> fp32 in TMEM, swap-AB: experts
//                       are the UMMA M, tokens the N) with the selection in the same kernel. A
//                       cluster of C = ceil(E/128) CTAs covers one token tile: CTA r owns experts
//                       [128r, 128r+128), writes its fp32 accumulator column by column into the
//                       shared memory of the CTA that selects each token (distributed shared
//                       memory), and after one cluster barrier every CTA selects its share of the
//                       tile's tokens. Logits never leave the chip unless the caller asks for them.
//
// THE LOGITS PROGRAM (the identity of route_fused's GEMM): per (token, expert) one tcgen05
// accumulation chain over the hidden axis, K = 16 per instruction, instructions in ascending h from
// h = 0 (accumulate flag off on the first), one fp32 accumulator, never split, never re-associated.
// The tile over tokens (N) and which CTA owns which expert row are ownership only; the battery
// proves the bits do not move across N and across T.
#include "blackwell.cuh"
#include "select.cuh"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>
#include <string>
#include <vector>

namespace rt100 {
namespace {

// ------------------------------------------------------------------ select only ---

template <int MODE, int K, int EPL>
__global__ __launch_bounds__(256) void select_kernel(const float *__restrict__ logits,
                                                     int64_t stride, int T, SelectArgs a) {
  const int64_t t = int64_t(blockIdx.x) * 8 + int(threadIdx.x) / 32;
  const int lane = int(threadIdx.x & 31u);
  const bool valid = t < T;
  const int e0 = lane * EPL;
  float v[EPL];
  load_lane<EPL>(valid ? logits + t * stride + e0 : logits, valid ? max(0, min(EPL, a.E - e0)) : 0,
                 v);
  select_warp<MODE, K, EPL>(v, valid, t, a);
}

// ------------------------------------------------------------------------ fused ---
//
// THE LOGITS PROGRAM. The hidden axis is cut into S fixed segments of H/S (a multiple of 64); segment g is ONE
// tcgen05 accumulation chain from zero over its h in ascending order, K = 16 per instruction (cta_group::1,
// kind::f16, bf16 x bf16 -> fp32, both operands K-major, SW128); the logit is the balanced tree of the S fp32
// partials with lower-index subtrees on the left, one __fadd_rn per node. S IS THE SCHEDULE POLICY'S: the host
// passes ``schedule_policy.reduce(H, b = 16).G`` (``ops.declared_segments``), which is 1 -- a tensor-core sum
// publishes no shortest segment, so the logit is ONE chain over the whole hidden axis and the tree is the
// identity. The kernels stay templated on (CS, SL); only S = 1 (CS = SL = 1) is instantiated.
//
// OWNERSHIP (free, gated bit-for-bit): a cluster of CE x CS CTAs covers one token tile of N tokens.
// CTA (ce, cs) owns experts [MR*ce, MR*ce + MR) and segments [SL*cs, SL*cs + SL) (S = CS*SL), each
// segment in its own TMEM accumulator. The UMMA M is UM = 128 for MR = 128 and 64 for MR <= 64; a
// CTA may fill only its first MR rows of the A tile (rows above are never loaded, their lanes never
// read, and a UMMA row's arithmetic reads only that row). Every CTA writes its fp32 partials into the
// shared memory of the CTA that selects each token (distributed shared memory); after one cluster
// barrier that CTA evaluates the tree in the declared order and selects. The tree is evaluated by
// ONE thread per element from S separately stored partials, so no arrival order can reach it.

//: 8 warps for tiles of N <= 64 (decode: the selection is a handful of tokens and every extra warp
//: is barrier latency), 16 for N >= 128 (prefill: the selection is the long pole).
template <int N> struct Threads {
  static constexpr int v = N >= 128 ? 512 : 256;
};

template <int N, int MR> struct FusedShape {
  static constexpr uint32_t kUM = MR <= 64 ? 64u : 128u;
  static constexpr uint32_t kFullA = kUM * kAtomK * 2;   // what one UMMA reads of one A atom
  static constexpr uint32_t kAtomA = uint32_t(MR) * kAtomK * 2;
  static constexpr uint32_t kAtomB = uint32_t(N) * kAtomK * 2;
  //: accumulator rows per TMEM lane quarter: M = 128 puts row m in lane m; M = 64 (cta_group::1)
  //: puts rows 16q..16q+15 in lanes 32q..32q+15 of quarter q.
  static constexpr uint32_t kRowsPerQuarter = kUM == 128 ? 32u : 16u;
  //: two or four warps per TMEM lane quarter, each draining its share of the N columns.
  static constexpr uint32_t kParts = uint32_t(Threads<N>::v / 32) / 4u;
  static constexpr uint32_t kCols = uint32_t(N) / kParts;
  static constexpr uint32_t kCh = kCols < 16u ? kCols : 16u;   // TMEM columns per tcgen05.ld
};


template <uint32_t CH> struct TmemLdN;
template <> struct TmemLdN<4> {
  __device__ __forceinline__ static void copy(uint32_t addr, uint32_t *v) {
    cute::SM100::TMEM::LOAD::SM100_TMEM_LOAD_32dp32b4x::copy(addr, v[0], v[1], v[2], v[3]);
  }
};
template <> struct TmemLdN<8> {
  __device__ __forceinline__ static void copy(uint32_t addr, uint32_t *v) { TmemLd<8>::copy(addr, v); }
};
template <> struct TmemLdN<16> {
  __device__ __forceinline__ static void copy(uint32_t addr, uint32_t *v) { TmemLd<16>::copy(addr, v); }
};
template <> struct TmemLdN<32> {
  __device__ __forceinline__ static void copy(uint32_t addr, uint32_t *v) { TmemLd<32>::copy(addr, v); }
};

//: THE segment tree: balanced, lower-index subtrees on the left, one __fadd_rn a node.
template <int S> __device__ __forceinline__ float segment_tree(const float (&p)[S]) {
  float a[S];
#pragma unroll
  for (int i = 0; i < S; ++i)
    a[i] = p[i];
#pragma unroll
  for (int width = 1; width < S; width *= 2)
#pragma unroll
    for (int i = 0; i + width < S; i += 2 * width)
      a[i] = __fadd_rn(a[i], a[i + width]);
  return a[0];
}

// ------------------------------------------------------------ select from segment roots ---
//
// The selection of select_kernel on logits given as their S fp32 SEGMENT ROOTS ``parts [S, T, E]`` (segment g
// = one ascending tcgen05 K = 16 chain from zero over the hidden range [g*H/S, (g+1)*H/S), e.g. DeepGEMM segment
// GEMMs): each logit is segment_tree<S> of its S roots -- the fused kernel's tree, evaluated by one thread per
// element from separately stored partials -- then the same select_warp. With S = declared_segments(H) the
// table equals route_fused's bit for bit.
template <int MODE, int K, int EPL, int S>
__global__ __launch_bounds__(256) void select_tree_kernel(const float *__restrict__ parts, int64_t seg,
                                                          int64_t stride, int T, SelectArgs a) {
  // launched as a programmatic dependent of the segment GEMM (PDL): wait for its results before the first load
  asm volatile("griddepcontrol.wait;" ::: "memory");
  const int64_t t = int64_t(blockIdx.x) * 8 + int(threadIdx.x) / 32;
  const int lane = int(threadIdx.x & 31u);
  const bool valid = t < T;
  const int e0 = lane * EPL;
  const int n = valid ? max(0, min(EPL, a.E - e0)) : 0;
  float p[S][EPL];
#pragma unroll
  for (int g = 0; g < S; ++g)
    load_lane<EPL>(valid ? parts + g * seg + t * stride + e0 : parts, n, p[g]);
  float v[EPL];
#pragma unroll
  for (int u = 0; u < EPL; ++u) {
    float q[S];
#pragma unroll
    for (int g = 0; g < S; ++g)
      q[g] = p[g][u];
    v[u] = segment_tree<S>(q);
  }
  select_warp<MODE, K, EPL>(v, valid, t, a);
}

struct FusedArgs {
  int T, H, stages, ka;
  uint32_t region;   //: bytes of the stage ring (and of the partial buffer it aliases)
  float *logits_out;
};

template <int N, int MR, int CE, int CS, int SL, int MODE, int K, int EPL, int OCC>
__global__ __launch_bounds__(Threads<N>::v, OCC) void fused_kernel(const __grid_constant__ CUtensorMap wmap,
                                                       const __grid_constant__ CUtensorMap xmap,
                                                       FusedArgs f, SelectArgs a) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  using Sh = FusedShape<N, MR>;
  constexpr int S = CS * SL;                         // the declared segment count
  constexpr int C = CE * CS;                         // CTAs a cluster
  constexpr int OWN = (N + C - 1) / C;               // tokens each CTA of the cluster selects
  constexpr uint32_t kTmemCols = SL * N < 32 ? 32u : uint32_t(SL * N);
  static_assert(SL * N <= 512, "segment accumulators exceed tensor memory");
  const int ka = f.ka;
  const uint32_t stage_bytes = uint32_t(ka) * (Sh::kAtomA + Sh::kAtomB);
  extern __shared__ uint8_t smem_raw[];
  uint8_t *smem = reinterpret_cast<uint8_t *>(
      (reinterpret_cast<uintptr_t>(smem_raw) + 1023u) & ~uintptr_t(1023u));
  //: stage layout: [ka W atoms of MR rows][ka x atoms of N rows], each atom 128 B a row, SW128.
  auto smem_a = [&](uint32_t s) { return smem + s * stage_bytes; };
  auto smem_b = [&](uint32_t s) { return smem + s * stage_bytes + uint32_t(ka) * Sh::kAtomA; };
  auto bar_base = reinterpret_cast<Barrier *>(smem + f.region);
  auto full = [&](uint32_t s) { return bar_base + s; };
  auto empty = [&](uint32_t s) { return bar_base + f.stages + s; };
  Barrier *acc_full = bar_base + 2 * f.stages;
  uint32_t *tmem_slot = reinterpret_cast<uint32_t *>(bar_base + 2 * f.stages + 1);
  //: the partial buffer [S][OWN][E] ALIASES the operand stages; it is written only after cluster
  //: barrier 1, when every CTA of the cluster has consumed every stage.
  float *pbuf = reinterpret_cast<float *>(smem);
  const int E = a.E, T = f.T;

  const uint32_t warp = __shfl_sync(0xffffffffu, threadIdx.x / 32u, 0);
  const uint32_t lane = lane_id();
  const uint32_t rank = blockIdx.x;                  // cluster (C,1,1) over grid.x == C
  const int ce = int(rank) % CE, cs = int(rank) / CE;
  const int tok0 = int(blockIdx.y) * N;
  const int m0 = ce * MR;
  const int seg_atoms = f.H / int(kAtomK) / S;       // atoms a segment
  const int steps = seg_atoms / ka;                  // stages a segment

  if (warp == 0 && cute::elect_one_sync()) {
    cute::prefetch_tma_descriptor(&wmap);
    cute::prefetch_tma_descriptor(&xmap);
  }
  if (warp == 1) {
    for (uint32_t b = lane; b < uint32_t(2 * f.stages + 1); b += 32u)
      (bar_base + b)->init(1u);
    cutlass::arch::fence_barrier_init();
  }
  if (warp == 2) {
    // The allocation pattern and both fences are moe_sm100/csrc/experts.cu's (see its comment).
    if (cute::elect_one_sync())
      asm volatile("st.shared.u32 [%0], %1;" ::"l"(__cvta_generic_to_shared(tmem_slot)),
                   "r"(0xffffffffu));
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncwarp();
    TmemAllocator().allocate(kTmemCols, tmem_slot);
    TmemAllocator().release_allocation_lock();
    tcgen05_before_thread_sync();
  }
  __syncthreads();
  uint32_t tmem_base;
  while ((tmem_base = ld_shared_u32(tmem_slot)) == 0xffffffffu) {
  }

  if (warp == 0) {
    // ---- TMA producer: per stage, ka atoms of W (MR experts) and ka atoms of x (N tokens) ----
    if (cute::elect_one_sync()) {
      uint32_t slot = 0, phase = 0, issued = 0;
      const uint64_t hint = uint64_t(cute::TMA::CacheHintSm100::EVICT_NORMAL);
      for (int ls = 0; ls < SL; ++ls) {
        const int g = cs * SL + ls;
        for (int st = 0; st < steps; ++st, ++issued) {
          if (issued >= uint32_t(f.stages))
            empty(slot)->wait(phase ^ 1u);
          auto *bar = reinterpret_cast<uint64_t *>(full(slot));
          const int atom = g * seg_atoms + st * ka;
          cute::SM90_TMA_LOAD_3D::copy(&wmap, bar, hint, smem_a(slot), 0, m0, atom);
          cute::SM90_TMA_LOAD_3D::copy(&xmap, bar, hint, smem_b(slot), 0, tok0, atom);
          full(slot)->arrive_and_expect_tx(stage_bytes);
          if (++slot == uint32_t(f.stages)) {
            slot = 0;
            phase ^= 1u;
          }
        }
      }
    }
    __syncwarp();
  } else if (warp == 1) {
    // ---- UMMA issuer: segment ls of this CTA accumulates in TMEM columns [ls*N, ls*N + N) ----
    const uint64_t idesc = cute::UMMA::make_runtime_instr_desc(
        cute::UMMA::make_instr_desc<cutlass::bfloat16_t, cutlass::bfloat16_t, float, Sh::kUM, N,
                                    cute::UMMA::Major::K, cute::UMMA::Major::K>());
    auto a_desc = make_desc_k_major<__nv_bfloat16>(smem_a(0));
    auto b_desc = make_desc_k_major<__nv_bfloat16>(smem_b(0));
    const uint32_t a_lo0 = a_desc.lo, b_lo0 = b_desc.lo;
    uint32_t slot = 0, phase = 0;
    for (int ls = 0; ls < SL; ++ls) {
      const uint32_t acc = tmem_base + uint32_t(ls * N);
      for (int st = 0; st < steps; ++st) {
        full(slot)->wait(phase);
        tcgen05_after_thread_sync();
        if (cute::elect_one_sync()) {
          for (int at = 0; at < ka; ++at) {
            const uint32_t a_base = a_lo0 + ((slot * stage_bytes + uint32_t(at) * Sh::kAtomA) >> 4);
            const uint32_t b_base = b_lo0 + ((slot * stage_bytes + uint32_t(at) * Sh::kAtomB) >> 4);
#pragma unroll
            for (uint32_t kk = 0; kk < kAtomK / kUmmaK; ++kk) {
              a_desc.lo = advance_lo<__nv_bfloat16>(a_base, kk * kUmmaK);
              b_desc.lo = advance_lo<__nv_bfloat16>(b_base, kk * kUmmaK);
              Umma1Sm::fma(a_desc, b_desc, acc, (st > 0 || at > 0 || kk > 0) ? 1u : 0u, idesc);
            }
          }
        }
        cutlass::arch::umma_arrive(reinterpret_cast<uint64_t *>(empty(slot)));
        if (++slot == uint32_t(f.stages)) {
          slot = 0;
          phase ^= 1u;
        }
      }
    }
    cutlass::arch::umma_arrive(reinterpret_cast<uint64_t *>(acc_full));
    __syncwarp();
  }

  // ---- every thread: the accumulators are complete ----
  acc_full->wait(0u);
  tcgen05_after_thread_sync();
  // Cluster barrier 1: every CTA's operand stages are dead, so the aliased buffers may be written
  // from anywhere in the cluster.
  cluster_sync_all();
  {
    // Drain: thread (quarter q, lane) is expert m0 + q*rows + lane; the four warps of a quarter
    // take a quarter of the columns each. The CTA's SL local segments are folded HERE, one thread
    // per element, into their subtree node of the declared tree (SL is a power of two and the
    // segments are contiguous, so the node is exactly a subtree); CS nodes then travel.
    const uint32_t q = warp & 3u, part = warp >> 2;
    if (q * Sh::kRowsPerQuarter < uint32_t(MR)) {   // warp-uniform: this quarter holds experts
      const int el = int(q * Sh::kRowsPerQuarter + lane);
      const int e = m0 + el;
      const bool real = lane < Sh::kRowsPerQuarter && el < MR && e < E;
      const uint32_t lbuf = uint32_t(__cvta_generic_to_shared(pbuf));
      uint32_t rbuf[C];
#pragma unroll
      for (int r = 0; r < C; ++r)
        rbuf[r] = mapa_shared(lbuf, uint32_t(r));
      const uint32_t taddr = tmem_base + ((q * 32u) << 16) + part * Sh::kCols;
#pragma unroll 1
      for (uint32_t c0 = 0; c0 < Sh::kCols; c0 += Sh::kCh) {
        uint32_t v[SL][Sh::kCh];
#pragma unroll
        for (int ls = 0; ls < SL; ++ls)
          TmemLdN<Sh::kCh>::copy(taddr + uint32_t(ls * N) + c0, v[ls]);
        cutlass::arch::fence_view_async_tmem_load();
        if (real) {
#pragma unroll
          for (uint32_t j = 0; j < Sh::kCh; ++j) {
            const int t = int(part * Sh::kCols + c0 + j);
            float leaf[SL];
#pragma unroll
            for (int ls = 0; ls < SL; ++ls)
              leaf[ls] = __uint_as_float(v[ls][j]);
            const float node = segment_tree<SL>(leaf);
            const int owner = t / OWN;
            const int tl = t - owner * OWN;
            uint32_t dst = rbuf[0];
#pragma unroll
            for (int r = 1; r < C; ++r)
              dst = owner == r ? rbuf[r] : dst;
            st_cluster_f32(dst + uint32_t((cs * OWN + tl) * E + e) * 4u, node);
          }
        }
      }
    }
  }
  // Cluster barrier 2: every token's S partial rows are in its selecting CTA's shared memory.
  cluster_sync_all();
  // One warp per token, sixteen tokens a pass. The CS subtree nodes are folded by the top of the
  // same declared tree.
  {
    const int e0 = int(lane) * EPL;
    const int n = max(0, min(EPL, E - e0));
    constexpr int TPW = N >= 128 ? 2 : 1;            // tokens a warp interleaves (ILP only)
    constexpr int kPass = (Threads<N>::v / 32) * TPW;
#pragma unroll 1
    for (int base = 0; base < OWN; base += kPass) {
      float v[TPW][EPL];
      bool valid[TPW];
      int64_t tok[TPW];
#pragma unroll
      for (int tt = 0; tt < TPW; ++tt) {
        const int tl = base + int(warp) * TPW + tt;
        const int t = int(rank) * OWN + tl;
        tok[tt] = int64_t(tok0) + t;
        valid[tt] = tl < OWN && t < N && tok[tt] < T;
        float p[CS][EPL];
#pragma unroll
        for (int c = 0; c < CS; ++c)
          load_lane<EPL>(pbuf + (c * OWN + (tl < OWN ? tl : 0)) * E + e0, valid[tt] ? n : 0, p[c]);
#pragma unroll
        for (int u = 0; u < EPL; ++u) {
          float node[CS];
#pragma unroll
          for (int c = 0; c < CS; ++c)
            node[c] = p[c][u];
          v[tt][u] = segment_tree<CS>(node);
        }
        if (f.logits_out && valid[tt]) {
#pragma unroll
          for (int u = 0; u < EPL; ++u)
            if (u < n)
              f.logits_out[tok[tt] * E + e0 + u] = v[tt][u];
        }
      }
      select_warp_n<MODE, K, EPL, TPW>(v, valid, tok, a);
    }
  }
  tcgen05_before_thread_sync();
  __syncthreads();
  if (warp == 2)
    TmemAllocator().free(tmem_base, kTmemCols);
#endif
}

// ------------------------------------------------------------------- persistent ---
//
// The PREFILL schedule of the same program: a cluster of CE CTAs walks token tiles of 128 (CTA ce
// owns experts [128ce, 128ce + 128), all S segments in S TMEM accumulators of 128 columns; S = 1 under the
// schedule policy). Warp roles: 0 TMA, 1 UMMA, 4-7 drain (fold the S segment partials by the declared tree, one
// thread per element, and store the logits into the selecting CTA's buffer), the other ten select.
// The drain releases TMEM as soon as it has read it, so tile i+1's UMMAs run while tile i is being
// selected out of a double-buffered shared-memory logits buffer. Cluster mbarriers carry the
// handoffs: `sel_full[p]` (in the selecting CTA, CE x 4 drain-warp arrivals) and `sel_free[p]` (in
// each writing CTA, one arrival per selecting CTA). Who arrives first is never arithmetic.

constexpr int kPN = 128;            // tokens a tile
constexpr int kPThreads = 512;
constexpr int kPSelWarps = 10;      // warps 2, 3, 8..15

template <int CE, int MODE, int K, int EPL, int NB, int S>
__global__ __launch_bounds__(kPThreads, 1) void persist_kernel(
    const __grid_constant__ CUtensorMap wmap, const __grid_constant__ CUtensorMap xmap,
    FusedArgs f, SelectArgs a) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  using Sh = FusedShape<kPN, 128>;
  static_assert(S >= 1 && S * kPN <= 512, "segment accumulators exceed tensor memory");
  constexpr int C = CE;
  constexpr int OWN = (kPN + C - 1) / C;
  constexpr uint32_t kTmemCols = 512;
  const int ka = f.ka;
  const uint32_t stage_bytes = uint32_t(ka) * (Sh::kAtomA + Sh::kAtomB);
  extern __shared__ uint8_t smem_raw[];
  uint8_t *smem = reinterpret_cast<uint8_t *>(
      (reinterpret_cast<uintptr_t>(smem_raw) + 1023u) & ~uintptr_t(1023u));
  const int E = a.E, T = f.T;
  const uint32_t ring = uint32_t(f.stages) * stage_bytes;
  //: each CTA's drained logits of ITS OWN 128 experts for the whole tile, [NB][128 tokens][128];
  //: the selecting CTAs read the rows they need from every peer (distributed shared memory).
  float *sbuf = reinterpret_cast<float *>(smem + ring);
  const uint32_t sbuf_bytes = uint32_t(NB) * uint32_t(kPN) * 128u * 4u;
  auto bar_base = reinterpret_cast<Barrier *>(smem + ring + sbuf_bytes);
  auto smem_a = [&](uint32_t st) { return smem + st * stage_bytes; };
  auto smem_b = [&](uint32_t st) { return smem + st * stage_bytes + uint32_t(ka) * Sh::kAtomA; };
  auto full = [&](uint32_t st) { return bar_base + st; };
  auto empty = [&](uint32_t st) { return bar_base + f.stages + st; };
  Barrier *acc_full = bar_base + 2 * f.stages;
  Barrier *acc_empty = acc_full + 1;
  Barrier *sel_full = acc_full + 2;     // [2]
  Barrier *sel_free = acc_full + 4;     // [2]
  uint32_t *tmem_slot = reinterpret_cast<uint32_t *>(acc_full + 6);

  const uint32_t warp = __shfl_sync(0xffffffffu, threadIdx.x / 32u, 0);
  const uint32_t lane = lane_id();
  const uint32_t rank = blockIdx.x;
  const int m0 = int(rank) * 128;
  const int ntiles = (T + kPN - 1) / kPN;
  const int seg_atoms = f.H / int(kAtomK) / S;
  const int steps = seg_atoms / ka;

  if (warp == 0 && cute::elect_one_sync()) {
    cute::prefetch_tma_descriptor(&wmap);
    cute::prefetch_tma_descriptor(&xmap);
  }
  if (warp == 1) {
    for (uint32_t b = lane; b < uint32_t(2 * f.stages); b += 32u)
      (bar_base + b)->init(1u);
    if (lane == 0) {
      acc_full->init(1u);
      acc_empty->init(4u);
      sel_full[0].init(uint32_t(C * 4));
      sel_full[1].init(uint32_t(C * 4));
      sel_free[0].init(uint32_t(C));
      sel_free[1].init(uint32_t(C));
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (warp == 2) {
    if (cute::elect_one_sync())
      asm volatile("st.shared.u32 [%0], %1;" ::"l"(__cvta_generic_to_shared(tmem_slot)),
                   "r"(0xffffffffu));
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncwarp();
    TmemAllocator().allocate(kTmemCols, tmem_slot);
    TmemAllocator().release_allocation_lock();
    tcgen05_before_thread_sync();
  }
  __syncthreads();
  cluster_sync_all();                    // every CTA's barriers are initialised before any remote use
  uint32_t tmem_base;
  while ((tmem_base = ld_shared_u32(tmem_slot)) == 0xffffffffu) {
  }

  if (warp == 0) {
    if (cute::elect_one_sync()) {
      uint32_t slot = 0, phase = 0, issued = 0;
      const uint64_t hint = uint64_t(cute::TMA::CacheHintSm100::EVICT_NORMAL);
      for (int tile = blockIdx.y; tile < ntiles; tile += gridDim.y) {
        const int tok0 = tile * kPN;
        for (int g = 0; g < S; ++g)
          for (int st = 0; st < steps; ++st, ++issued) {
            if (issued >= uint32_t(f.stages))
              empty(slot)->wait(phase ^ 1u);
            auto *bar = reinterpret_cast<uint64_t *>(full(slot));
            const int atom = g * seg_atoms + st * ka;
            cute::SM90_TMA_LOAD_3D::copy(&wmap, bar, hint, smem_a(slot), 0, m0, atom);
            cute::SM90_TMA_LOAD_3D::copy(&xmap, bar, hint, smem_b(slot), 0, tok0, atom);
            full(slot)->arrive_and_expect_tx(stage_bytes);
            if (++slot == uint32_t(f.stages)) {
              slot = 0;
              phase ^= 1u;
            }
          }
      }
    }
    __syncwarp();
  } else if (warp == 1) {
    const uint64_t idesc = cute::UMMA::make_runtime_instr_desc(
        cute::UMMA::make_instr_desc<cutlass::bfloat16_t, cutlass::bfloat16_t, float, 128, kPN,
                                    cute::UMMA::Major::K, cute::UMMA::Major::K>());
    auto a_desc = make_desc_k_major<__nv_bfloat16>(smem_a(0));
    auto b_desc = make_desc_k_major<__nv_bfloat16>(smem_b(0));
    const uint32_t a_lo0 = a_desc.lo, b_lo0 = b_desc.lo;
    uint32_t slot = 0, phase = 0, it = 0;
    for (int tile = blockIdx.y; tile < ntiles; tile += gridDim.y, ++it) {
      if (it)
        acc_empty->wait((it - 1u) & 1u);
      tcgen05_after_thread_sync();
      for (int g = 0; g < S; ++g) {
        const uint32_t acc = tmem_base + uint32_t(g * kPN);
        for (int st = 0; st < steps; ++st) {
          full(slot)->wait(phase);
          tcgen05_after_thread_sync();
          if (cute::elect_one_sync()) {
            for (int at = 0; at < ka; ++at) {
              const uint32_t a_base = a_lo0 + ((slot * stage_bytes + uint32_t(at) * Sh::kAtomA) >> 4);
              const uint32_t b_base = b_lo0 + ((slot * stage_bytes + uint32_t(at) * Sh::kAtomB) >> 4);
#pragma unroll
              for (uint32_t kk = 0; kk < kAtomK / kUmmaK; ++kk) {
                a_desc.lo = advance_lo<__nv_bfloat16>(a_base, kk * kUmmaK);
                b_desc.lo = advance_lo<__nv_bfloat16>(b_base, kk * kUmmaK);
                Umma1Sm::fma(a_desc, b_desc, acc, (st > 0 || at > 0 || kk > 0) ? 1u : 0u, idesc);
              }
            }
          }
          cutlass::arch::umma_arrive(reinterpret_cast<uint64_t *>(empty(slot)));
          if (++slot == uint32_t(f.stages)) {
            slot = 0;
            phase ^= 1u;
          }
        }
      }
      cutlass::arch::umma_arrive(reinterpret_cast<uint64_t *>(acc_full));
    }
    __syncwarp();
  } else if (warp >= 4 && warp < 8) {
    // ---- drain: the S segment partials -> declared tree -> the selecting CTA's buffer ----
    const uint32_t q = warp & 3u;
    const int e = m0 + int(q * 32u + lane);
    const bool real = e < E;
    const uint32_t lfull = uint32_t(__cvta_generic_to_shared(sel_full));
    uint32_t it = 0;
    for (int tile = blockIdx.y; tile < ntiles; tile += gridDim.y, ++it) {
      const uint32_t p = it % NB, use = it / NB;
      acc_full->wait(it & 1u);
      tcgen05_after_thread_sync();
      mbar_wait_cluster(uint32_t(__cvta_generic_to_shared(&sel_free[p])), (use & 1u) ^ 1u);
      const uint32_t taddr = tmem_base + ((q * 32u) << 16);
#pragma unroll 1
      for (uint32_t c0 = 0; c0 < uint32_t(kPN); c0 += 16u) {
        uint32_t v[S][16];
#pragma unroll
        for (int g = 0; g < S; ++g)
          TmemLdN<16>::copy(taddr + uint32_t(g * kPN) + c0, v[g]);
        cutlass::arch::fence_view_async_tmem_load();
        if (real) {
#pragma unroll
          for (uint32_t j = 0; j < 16u; ++j) {
            const int t = int(c0 + j);
            float leaf[S];
#pragma unroll
            for (int g = 0; g < S; ++g)
              leaf[g] = __uint_as_float(v[g][j]);
            sbuf[(int(p) * kPN + t) * 128 + int(q * 32u + lane)] = segment_tree<S>(leaf);
          }
        }
      }
      tcgen05_before_thread_sync();
      fence_acq_rel_cluster();
      __syncwarp();
      if (lane == 0) {
        acc_empty->arrive();
#pragma unroll
        for (int r = 0; r < C; ++r)
          mbar_arrive_cluster(mapa_shared(lfull + p * 8u, uint32_t(r)));
      }
      __syncwarp();
    }
  } else if (warp == 2 || warp == 3 || warp >= 8) {
    // ---- select: this CTA's OWN tokens of the tile, out of buffer p ----
    const int sw = warp < 4 ? int(warp) - 2 : int(warp) - 6;   // 0..9
    const int e0 = int(lane) * EPL;
    const int n = max(0, min(EPL, E - e0));
    const uint32_t lfree = uint32_t(__cvta_generic_to_shared(sel_free));
    //: expert e lives in peer e / 128 at local column e % 128; a 4-group (or 2-group) never
    //: straddles a peer because 128 is a multiple of both.
    const uint32_t lsb = uint32_t(__cvta_generic_to_shared(sbuf));
    uint32_t it = 0;
    for (int tile = blockIdx.y; tile < ntiles; tile += gridDim.y, ++it) {
      const uint32_t p = it % NB, use = it / NB;
      mbar_wait_cluster(uint32_t(__cvta_generic_to_shared(&sel_full[p])), use & 1u);
      const int tok0 = tile * kPN;
#pragma unroll 1
      for (int tl = sw; tl < OWN; tl += kPSelWarps) {
        const int t = int(rank) * OWN + tl;
        const int64_t tok = int64_t(tok0) + t;
        const bool valid = t < kPN && tok < T;
        float v[EPL];
        {
          const int row = (int(p) * kPN + (t < kPN ? t : 0)) * 128;
          auto addr = [&](int e) {
            return mapa_shared(lsb, uint32_t(e / 128)) + uint32_t(row + e % 128) * 4u;
          };
          if constexpr (EPL % 4 == 0) {
#pragma unroll
            for (int u = 0; u < EPL; u += 4) {
              const float4 q4 = (valid && u < n) ? ld_cluster_v4(addr(e0 + u))
                                                 : make_float4(0.f, 0.f, 0.f, 0.f);
              v[u] = q4.x;
              v[u + 1] = q4.y;
              v[u + 2] = q4.z;
              v[u + 3] = q4.w;
            }
          } else {
#pragma unroll
            for (int u = 0; u < EPL; u += 2) {
              const float2 q2 = (valid && u < n) ? ld_cluster_v2(addr(e0 + u))
                                                 : make_float2(0.f, 0.f);
              v[u] = q2.x;
              v[u + 1] = q2.y;
            }
          }
        }
        if (f.logits_out && valid) {
#pragma unroll
          for (int u = 0; u < EPL; ++u)
            if (u < n)
              f.logits_out[tok * E + e0 + u] = v[u];
        }
        select_warp<MODE, K, EPL>(v, valid, tok, a);
      }
      named_bar_sync(1, kPSelWarps * 32);
      if (sw == 0 && lane == 0) {
#pragma unroll
        for (int r = 0; r < C; ++r)
          mbar_arrive_cluster(mapa_shared(lfree + p * 8u, uint32_t(r)));
      }
    }
  }
  __syncwarp();
  tcgen05_before_thread_sync();
  cluster_sync_all();                    // nobody leaves while a peer may still touch its memory
  if (warp == 2)
    TmemAllocator().free(tmem_base, kTmemCols);
#endif
}

// ------------------------------------------------------------------------- host ---

int pow2_at_least(int v) {
  int p = 1;
  while (p < v)
    p <<= 1;
  return p;
}

struct Geometry {
  int T, E, K, S, n_group, topk_group, EPL;
};

//: experts per lane: the smallest served EPL with 32 * EPL >= E.
int pick_epl(int E) {
  const int epls[6] = {2, 4, 8, 12, 16, 32};
  for (int i = 0; i < 6; ++i)
    if (32 * epls[i] >= E)
      return epls[i];
  return 0;
}

Geometry check_geometry(int T, int E, int K, int S, int n_group, int topk_group, int mode,
                        bool has_bias) {
  TORCH_CHECK(E >= 2 && E <= 1024, "router_sm100: 2 <= E <= 1024, got ", E);
  TORCH_CHECK(K >= 1 && K <= 16 && K <= E, "router_sm100: 1 <= top_k <= min(16, E), got ", K);
  TORCH_CHECK(S >= 0 && K + S <= 32, "router_sm100: top_k + shared slots <= 32");
  TORCH_CHECK(n_group >= 1 && topk_group >= 1 && topk_group <= n_group,
              "router_sm100: 1 <= topk_group <= n_group");
  const int EPL = pick_epl(E);
  if (mode == kSigmoid) {
    TORCH_CHECK(has_bias, "sigmoid_noaux_bias_select_grouped requires the correction bias");
    TORCH_CHECK(S == 0, "shared-as-routed slots belong to sqrtsoftplus_bias_select only");
    TORCH_CHECK(E % n_group == 0, "router_sm100: E = ", E, " is not a multiple of n_group = ",
                n_group, " (a group must be whole)");
    if (topk_group < n_group) {
      const int gs = E / n_group;
      TORCH_CHECK(gs >= 2, "group score is a top-2 sum: a group needs two experts");
      TORCH_CHECK(gs % EPL == 0 && ((gs / EPL) & (gs / EPL - 1)) == 0,
                  "router_sm100: group size ", gs, " must be a power-of-two multiple of ", EPL,
                  " experts (one warp lane's share at E = ", E, ")");
      TORCH_CHECK(K <= topk_group * gs, "top_k exceeds the experts the surviving groups hold");
    }
  } else {
    TORCH_CHECK(n_group == 1 && topk_group == 1, "sqrtsoftplus_bias_select is ungrouped");
  }
  return Geometry{T, E, K, S, n_group, topk_group, EPL};
}

#define RT_DISPATCH_EPL(EPL_, BODY)                                                          \
  do {                                                                                       \
    switch (EPL_) {                                                                          \
    case 2: { constexpr int ET = 2; BODY; } break;                                           \
    case 4: { constexpr int ET = 4; BODY; } break;                                           \
    case 8: { constexpr int ET = 8; BODY; } break;                                           \
    case 12: { constexpr int ET = 12; BODY; } break;                                         \
    case 16: { constexpr int ET = 16; BODY; } break;                                         \
    default: { constexpr int ET = 32; BODY; } break;                                         \
    }                                                                                        \
  } while (0)

#define RT_DISPATCH_K(K_, BODY)                                                              \
  do {                                                                                       \
    switch (K_) {                                                                            \
    case 1: { constexpr int KT = 1; BODY; } break;                                           \
    case 2: { constexpr int KT = 2; BODY; } break;                                           \
    case 4: { constexpr int KT = 4; BODY; } break;                                           \
    case 6: { constexpr int KT = 6; BODY; } break;                                           \
    case 8: { constexpr int KT = 8; BODY; } break;                                           \
    case 16: { constexpr int KT = 16; BODY; } break;                                         \
    default: TORCH_CHECK(false, "router_sm100: top_k ", K_, " is not in {1,2,4,6,8,16}");    \
    }                                                                                        \
  } while (0)


} // namespace

void select_cuda(const at::Tensor &logits, const c10::optional<at::Tensor> &bias,
                 const c10::optional<at::Tensor> &mask, int64_t mode, int64_t K, int64_t S,
                 int64_t n_group, int64_t topk_group, double scale, at::Tensor out_idx,
                 at::Tensor out_w) {
  TORCH_CHECK(logits.is_cuda() && logits.scalar_type() == at::kFloat && logits.dim() == 2 &&
                  logits.stride(1) == 1,
              "router_sm100: logits must be CUDA fp32 [T, E] with unit inner stride");
  const c10::cuda::CUDAGuard guard(logits.device());
  const int T = int(logits.size(0)), E = int(logits.size(1));
  const Geometry g = check_geometry(T, E, int(K), int(S), int(n_group), int(topk_group),
                                    int(mode), bias.has_value());
  SelectArgs a{bias ? bias->data_ptr<float>() : nullptr,
               mask ? reinterpret_cast<const uint8_t *>(mask->data_ptr<bool>()) : nullptr,
               E, int(n_group), int(topk_group), int(S), float(scale),
               out_idx.data_ptr<int>(), out_w.data_ptr<float>()};
  if (!T)
    return;
  auto stream = at::cuda::getCurrentCUDAStream();
  const int blocks = (T + 7) / 8;
#define RT_SEL(MODE_)                                                                        \
  RT_DISPATCH_K(g.K, RT_DISPATCH_EPL(g.EPL, (select_kernel<MODE_, KT, ET><<<blocks, 256, 0,  \
                                              stream>>>(logits.data_ptr<float>(),          \
                                                        logits.stride(0), T, a))))
  if (mode == kSigmoid)
    RT_SEL(kSigmoid);
  else
    RT_SEL(kSqrtSoftplus);
#undef RT_SEL
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <class Kern, class... Args> static void launch_pdl(cudaLaunchConfig_t &cfg, Kern kern, Args... args) {
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kern, args...));
}

void select_tree_cuda(const at::Tensor &parts, const c10::optional<at::Tensor> &bias,
                      const c10::optional<at::Tensor> &mask, int64_t mode, int64_t K, int64_t S,
                      int64_t n_group, int64_t topk_group, double scale, at::Tensor out_idx,
                      at::Tensor out_w) {
  TORCH_CHECK(parts.is_cuda() && parts.scalar_type() == at::kFloat && parts.dim() == 3 && parts.is_contiguous() &&
                  parts.size(0) == 1,
              "router_sm100: segment roots must be contiguous CUDA fp32 [1, T, E] (one segment: the schedule "
              "policy's count for the logits program)");
  const c10::cuda::CUDAGuard guard(parts.device());
  const int T = int(parts.size(1)), E = int(parts.size(2));
  const Geometry g = check_geometry(T, E, int(K), int(S), int(n_group), int(topk_group),
                                    int(mode), bias.has_value());
  SelectArgs a{bias ? bias->data_ptr<float>() : nullptr,
               mask ? reinterpret_cast<const uint8_t *>(mask->data_ptr<bool>()) : nullptr,
               E, int(n_group), int(topk_group), int(S), float(scale),
               out_idx.data_ptr<int>(), out_w.data_ptr<float>()};
  if (!T)
    return;
  auto stream = at::cuda::getCurrentCUDAStream();
  const int blocks = (T + 7) / 8;
  const int64_t seg = int64_t(T) * E;
  // PDL: the launch (and the scheduling of its CTAs) overlaps the producing GEMM's tail; griddepcontrol.wait in the
  // kernel orders every load after the GEMM's completion. Launch overlap only: the bits are unchanged.
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(blocks);
  cfg.blockDim = dim3(256);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = stream;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  const float *pp = parts.data_ptr<float>();
#define RT_SELT(MODE_)                                                                       \
  RT_DISPATCH_K(g.K, RT_DISPATCH_EPL(g.EPL, (launch_pdl(cfg, select_tree_kernel<MODE_, KT, ET, 1>, pp, seg, \
                                                         int64_t(E), T, a))))
  if (mode == kSigmoid)
    RT_SELT(kSigmoid);
  else
    RT_SELT(kSqrtSoftplus);
#undef RT_SELT
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

namespace {

//: the fused kernel's ownership at T: token tile N, experts a CTA MR, segment CTAs CS, atoms a
//: stage KA. OWNERSHIP ONLY -- the battery pins every served point at one T and compares bits.
struct Tile {
  int N, MR, CS, KA, OCC;   //: OCC: CTAs a multiprocessor (2 halves the smem and TMEM budget)
  int PERSIST;              //: 1 = the persistent prefill schedule (N = 128, MR = 128, CS = 1)
};
//: CS (segment CTAs a cluster) in the rungs below was measured for the 4-segment program this kernel ran before
//: the schedule policy. It is ownership only and is clamped to a divisor of the policy's S by the caller, so at
//: S = 1 every rung runs CS = 1; the (N, MR, KA) of the rungs are kept as measured.

Tile pick_tile(int T, int E, int H, int S) {
  // THE MEASURED RULE (sweep_tiles.py, README "Ownership sweep"): per expert-count class, the
  // fastest (N, MR, CS, KA) at each production row, chosen on the same GPU in one lease. T between
  // rows takes the bucket that contains it. Ownership only -- every point is gated bit-equal.
  (void)H;
  (void)S;
  struct Rung {
    int tmax;
    Tile t;
  };
  static const Rung e64[] = {{512, {16, 64, 4, 4, 1, 0}},   {1024, {32, 64, 4, 4, 1, 0}},
                             {2048, {32, 64, 2, 4, 1, 0}},  {4096, {64, 64, 2, 4, 1, 0}},
                             {8192, {128, 64, 2, 2, 1, 0}}, {1 << 30, {128, 64, 1, 2, 1, 0}}};
  // e256 / e384 / e512: KA in {1, 2} rungs from a re-sweep (README "Ownership sweep",
  // KA ladder): a rung moved only where the mean gain over the bucket's geometries was >= 2 % and no
  // geometry regressed. Ownership only -- the sweep gated every point bit-equal.
  static const Rung e256[] = {{256, {32, 128, 4, 4, 1, 0}},   {512, {64, 128, 4, 4, 1, 0}},
                              {1024, {32, 128, 2, 2, 1, 0}},  {2048, {64, 128, 2, 2, 1, 0}},
                              {4096, {128, 128, 2, 2, 1, 0}}, {8192, {128, 128, 1, 1, 1, 0}},
                              {1 << 30, {128, 128, 1, 1, 1, 1}}};
  // e384v4 (DeepSeek V4-class, E in (288, 384]): re-measured at ONE segment (CS = 1) on B200, H 5120, top-6 + shared
  // (policy_validation/port/dense/results/sweep_router_tile_fl.json: every (N, KA, persistent) point bit-equal to
  // the previous rung, then timed). With one segment a cluster is CE = 3 CTAs, so the small token tile carries
  // the parallelism the segment CTAs used to: N = 16 up to 1024 rows, 64 up to 2048, 128 above.
  static const Rung e384v4[] = {{1024, {16, 128, 1, 4, 1, 0}},  {2048, {64, 128, 1, 2, 1, 0}},
                                {4096, {128, 128, 1, 1, 1, 0}}, {1 << 30, {128, 128, 1, 1, 1, 1}}};
  static const Rung e384[] = {{256, {16, 128, 2, 4, 1, 0}},   {512, {32, 128, 2, 2, 1, 0}},
                              {1024, {64, 128, 2, 2, 1, 0}},  {2048, {128, 128, 2, 2, 1, 0}},
                              {4096, {128, 128, 1, 1, 1, 0}}, {1 << 30, {128, 128, 1, 1, 1, 1}}};
  static const Rung e512[] = {{256, {32, 128, 2, 4, 1, 0}},   {512, {64, 128, 2, 2, 1, 0}},
                              {1024, {32, 128, 1, 2, 1, 0}},  {2048, {64, 128, 1, 2, 1, 0}},
                              {4096, {128, 128, 1, 1, 1, 0}}, {8192, {128, 128, 1, 1, 1, 0}},
                              {1 << 30, {128, 128, 1, 1, 1, 1}}};
  // E in (288, 384] (DeepSeek V4-class) takes the ladder re-measured at one segment; the other classes keep the
  // (N, MR, KA) of their 4-segment sweeps with CS clamped to 1 (not re-measured).
  const Rung *r = E <= 64 ? e64 : E <= 256 ? e256 : E <= 288 ? e384 : E <= 384 ? e384v4 : e512;
  while (T > r->tmax)
    ++r;
  return r->t;
}

//: a rank-3 SW128 map over a [rows, H] 16-bit row-major tensor viewed as [H/64 atoms][rows][64]:
//: one box is `atoms` consecutive 64-wide K atoms of `box_rows` rows, laid out atom-major in
//: shared memory -- exactly `atoms` consecutive K-major SW128 operand tiles.
int make_atom_map(CUtensorMap *map, const void *base, uint64_t rows, uint64_t H,
                  uint64_t row_stride_elems, uint32_t box_rows, uint32_t atoms) {
  auto fn = tma_encode_fn();
  if (fn == nullptr)
    return -10;
  const uint64_t dims[3] = {64, rows, H / 64};
  const uint64_t strides[2] = {row_stride_elems * 2u, 128u};
  const uint32_t box[3] = {64u, box_rows, atoms};
  const uint32_t elem_strides[3] = {1, 1, 1};
  CUresult rc = fn(map, CU_TENSOR_MAP_DATA_TYPE_UINT16, 3, const_cast<void *>(base), dims,
                   strides, box, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
                   CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                   CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return rc == CUDA_SUCCESS ? 0 : -(20 + int(rc));
}

} // namespace

//: OWNERSHIP OVERRIDE for sweeps and the ownership relation (tests): N > 0 replaces the ladder's token tile and
//: atoms a stage, persist >= 0 its persistent flag. Ownership only: every point computes the same bits. Zero
//: (the default) leaves the measured ladder in force; nothing in the package sets it.
static int g_tile_n = 0, g_tile_ka = 0, g_tile_persist = -1;
void tile_override_cuda(int64_t n, int64_t ka, int64_t persist) {
  g_tile_n = int(n);
  g_tile_ka = int(ka);
  g_tile_persist = int(persist);
}

void fused_cuda(const at::Tensor &x, const at::Tensor &w, const c10::optional<at::Tensor> &bias,
                const c10::optional<at::Tensor> &mask, int64_t mode, int64_t K, int64_t S_,
                int64_t n_group, int64_t topk_group, double scale, at::Tensor out_idx,
                at::Tensor out_w, const c10::optional<at::Tensor> &logits_out, int64_t segments) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && x.scalar_type() == at::kBFloat16 &&
                  w.scalar_type() == at::kBFloat16 && x.dim() == 2 && w.dim() == 2 &&
                  x.stride(1) == 1 && w.stride(1) == 1 && x.size(1) == w.size(1),
              "router_sm100 fused: x [T,H] and W [E,H] must be CUDA bf16 with unit inner stride");
  const c10::cuda::CUDAGuard guard(x.device());
  const int T = int(x.size(0)), E = int(w.size(0)), H = int(x.size(1));
  TORCH_CHECK(H % 64 == 0 && H >= 64, "router_sm100 fused: H must be a positive multiple of 64");
  TORCH_CHECK((x.stride(0) * 2) % 16 == 0 && (w.stride(0) * 2) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0,
              "router_sm100 fused: TMA needs 16-byte aligned rows");
  const Geometry g = check_geometry(T, E, int(K), int(S_), int(n_group), int(topk_group),
                                    int(mode), bias.has_value());
  float *lo = nullptr;
  if (logits_out.has_value()) {
    TORCH_CHECK(logits_out->is_contiguous() && logits_out->scalar_type() == at::kFloat &&
                    logits_out->size(0) == T && logits_out->size(1) == E,
                "logits_out must be contiguous fp32 [T, E]");
    lo = logits_out->data_ptr<float>();
  }
  SelectArgs a{bias ? bias->data_ptr<float>() : nullptr,
               mask ? reinterpret_cast<const uint8_t *>(mask->data_ptr<bool>()) : nullptr,
               E, int(n_group), int(topk_group), int(S_), float(scale),
               out_idx.data_ptr<int>(), out_w.data_ptr<float>()};
  if (!T)
    return;
  //: the segment count of the logits program: the schedule policy's, passed by the host (ops.declared_segments)
  const int S = int(segments);
  TORCH_CHECK(S >= 1 && (H / 64) % S == 0, "router_sm100 fused: H/64 must divide into the segments");
  Tile tl = pick_tile(T, E, H, S);
  while (S % tl.CS)       // ownership: a cluster cannot hold more segment CTAs than there are segments
    tl.CS /= 2;
  if (g_tile_n > 0) {
    tl.N = g_tile_n;
    tl.KA = g_tile_ka > 0 ? g_tile_ka : tl.KA;
    tl.PERSIST = g_tile_persist >= 0 ? g_tile_persist : 0;
  }
  const int OCC = tl.OCC;
  const int N = tl.N, MR = tl.MR, CS = tl.CS;
  const int CE = (E + MR - 1) / MR;
  TORCH_CHECK(S % CS == 0, "router_sm100 fused: CS must divide the segment count");
  const int SL = S / CS;
  const int seg_atoms = H / 64 / S;
  int KA = std::min(tl.KA, seg_atoms);
  while (seg_atoms % KA)
    --KA;
  TORCH_CHECK(CE * CS <= 16, "router_sm100 fused: cluster CE x CS exceeds the B200 maximum 16");
  CUtensorMap wmap, xmap;
  TORCH_CHECK(make_atom_map(&wmap, w.data_ptr(), uint64_t(E), uint64_t(H), uint64_t(w.stride(0)),
                            uint32_t(MR), uint32_t(KA)) == 0,
              "router_sm100: W tensor map");
  TORCH_CHECK(make_atom_map(&xmap, x.data_ptr(), uint64_t(T), uint64_t(H), uint64_t(x.stride(0)),
                            uint32_t(N), uint32_t(KA)) == 0,
              "router_sm100: x tensor map");
  auto stream = at::cuda::getCurrentCUDAStream();
  bool launched = false;
  if (tl.PERSIST) {
    TORCH_CHECK(N == 128 && MR == 128 && CS == 1 && S == 1,
                "router_sm100 persistent: tile (128, 128, 1) at S = 1 only");
    using Sh = FusedShape<128, 128>;
    const uint32_t stage = uint32_t(KA) * (Sh::kAtomA + Sh::kAtomB);
    const int OWN = (128 + CE - 1) / CE;
    const int NB = 1;     // one logits buffer: selection of tile i still overlaps the MMA of i+1
    (void)OWN;
    const uint32_t sbuf = uint32_t(NB) * 128u * 128u * 4u;
    const uint32_t extra = 1024u + 1024u;
    TORCH_CHECK(sbuf + stage + extra <= 232448u, "router_sm100 persistent: buffers exceed smem");
    const int stages = int(std::min<uint32_t>(16u, (232448u - extra - sbuf) / stage));
    const uint32_t smem = uint32_t(stages) * stage + sbuf + extra;
#define RT_PERSIST(CE_, MODE_, K_, EPL_)                                                     \
    if (!launched && CE == CE_ && int(mode) == MODE_ && g.K == K_ && g.EPL == EPL_) {      \
      launched = true;                                                                       \
      auto kern = persist_kernel<CE_, MODE_, K_, EPL_, 1, 1>;                                \
      C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, \
                                          int(smem)));                                       \
      cudaLaunchConfig_t cfg{};                                                              \
      cfg.blockDim = dim3(kPThreads, 1, 1);                                                  \
      cfg.dynamicSmemBytes = smem;                                                           \
      cfg.stream = stream;                                                                   \
      cudaLaunchAttribute attr[1];                                                           \
      attr[0].id = cudaLaunchAttributeClusterDimension;                                      \
      attr[0].val.clusterDim.x = CE_;                                                        \
      attr[0].val.clusterDim.y = 1;                                                          \
      attr[0].val.clusterDim.z = 1;                                                          \
      cfg.attrs = attr;                                                                      \
      cfg.numAttrs = 1;                                                                      \
      cfg.gridDim = dim3(CE_, 1, 1);                                                         \
      int clusters = 0;                                                                      \
      C10_CUDA_CHECK(cudaOccupancyMaxActiveClusters(&clusters, kern, &cfg));                 \
      const int ntiles = (T + 127) / 128;                                                    \
      clusters = std::max(1, std::min(clusters, ntiles));                                    \
      cfg.gridDim = dim3(CE_, clusters, 1);                                                  \
      FusedArgs fa{T, H, stages, KA, 0u, lo};                                  \
      C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kern, wmap, xmap, fa, a));                     \
    }
    RT_PERSIST(1, kSigmoid, 4, 2)
    RT_PERSIST(2, kSigmoid, 8, 8)
    RT_PERSIST(3, kSigmoid, 8, 12)             // GLM-5.3-Flash, E = 288
    RT_PERSIST(2, kSqrtSoftplus, 6, 8)
    RT_PERSIST(3, kSqrtSoftplus, 6, 12)
    RT_PERSIST(4, kSqrtSoftplus, 6, 16)
#undef RT_PERSIST
    TORCH_CHECK(launched, "router_sm100 persistent: no served kernel for this geometry");
    return;
  }
#define RT_FUSED_O(N_, MR_, CE_, CS_, SL_, MODE_, K_, EPL_, OCC_)                            \
  if (!launched && N == N_ && MR == MR_ && CE == CE_ && CS == CS_ && SL == SL_ &&           \
      int(mode) == MODE_ && g.K == K_ && g.EPL == EPL_ && OCC == OCC_) {                     \
    launched = true;                                                                         \
    using Sh = FusedShape<N_, MR_>;                                                          \
    constexpr int OWN = (N_ + CE_ * CS_ - 1) / (CE_ * CS_);                                  \
    const uint32_t buf = uint32_t(CS_) * uint32_t(OWN) * uint32_t(E) * 4u;                   \
    const uint32_t stage = uint32_t(KA) * (Sh::kAtomA + Sh::kAtomB);                         \
    const uint32_t tail = Sh::kFullA > Sh::kAtomA ? Sh::kFullA - Sh::kAtomA : 0u;            \
    const uint32_t budget = 232448u / OCC_ - 1024u - 1024u - tail - (OCC_ - 1) * 1024u;      \
    const uint32_t cap = 32u;                      \
    int stages = int(std::min<uint32_t>(cap, budget / stage));                               \
    stages = std::min(stages, SL_ * (seg_atoms / KA));                                       \
    const uint32_t region = std::max(uint32_t(stages) * stage + tail, buf);                  \
    TORCH_CHECK(stages >= 1 && region <= budget + tail,                                      \
                "router_sm100 fused: tile does not fit shared memory");                     \
    const uint32_t smem = region + 1024u + 1024u;                                            \
    auto kern = fused_kernel<N_, MR_, CE_, CS_, SL_, MODE_, K_, EPL_, OCC_>;                 \
    C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize,   \
                                        int(smem)));                                         \
    if (CE_ * CS_ > 8)                                                                       \
      C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1)); \
    cudaLaunchConfig_t cfg{};                                                                \
    cfg.gridDim = dim3(CE_ * CS_, (T + N_ - 1) / N_, 1);                                     \
    cfg.blockDim = dim3(Threads<N_>::v, 1, 1);                                                     \
    cfg.dynamicSmemBytes = smem;                                                             \
    cfg.stream = stream;                                                                     \
    cudaLaunchAttribute attr[1];                                                             \
    attr[0].id = cudaLaunchAttributeClusterDimension;                                        \
    attr[0].val.clusterDim.x = CE_ * CS_;                                                    \
    attr[0].val.clusterDim.y = 1;                                                            \
    attr[0].val.clusterDim.z = 1;                                                            \
    cfg.attrs = attr;                                                                        \
    cfg.numAttrs = 1;                                                                        \
    FusedArgs fa{T, H, stages, KA, region, lo};                                \
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kern, wmap, xmap, fa, a));                       \
  }
#define RT_FUSED(N_, MR_, CE_, CS_, SL_, MODE_, K_, EPL_)                                    \
  RT_FUSED_O(N_, MR_, CE_, CS_, SL_, MODE_, K_, EPL_, 1)
  // The ownership ladder per served geometry at the policy's S = 1 (CS = SL = 1): N in {16, 32, 64, 128},
  // MR in {64, 128}. The (CS, SL) splits of the 4-segment program are no longer instantiated.
#define RT_NS(MR_, CE_, CS_, SL_, MODE_, K_, EPL_)                                           \
  RT_FUSED(16, MR_, CE_, CS_, SL_, MODE_, K_, EPL_)                                          \
  RT_FUSED(32, MR_, CE_, CS_, SL_, MODE_, K_, EPL_)                                          \
  RT_FUSED(64, MR_, CE_, CS_, SL_, MODE_, K_, EPL_)                                          \
  RT_FUSED(128, MR_, CE_, CS_, SL_, MODE_, K_, EPL_)
#define RT_GEOM(E_, MODE_, K_, EPL_)                                                         \
  RT_NS(128, (E_ + 127) / 128, 1, 1, MODE_, K_, EPL_)                                        \
  RT_NS(64, (E_ + 63) / 64, 1, 1, MODE_, K_, EPL_)
  RT_GEOM(64, kSigmoid, 4, 2)                 // GLM-4.7-Flash
  RT_GEOM(256, kSigmoid, 8, 8)                // DeepSeek V3-class, grouped or not
  RT_NS(128, 3, 1, 1, kSigmoid, 8, 12)        // GLM-5.3-Flash, E = 288 (partial expert tile)
  RT_GEOM(256, kSqrtSoftplus, 6, 8)           // V4-class
  RT_NS(128, 3, 1, 1, kSqrtSoftplus, 6, 12)   // V4-class E = 384
  RT_NS(128, 4, 1, 1, kSqrtSoftplus, 6, 16)   // V4-class E = 512
#undef RT_GEOM
#undef RT_NS
#undef RT_FUSED
#undef RT_FUSED_O
  TORCH_CHECK(launched, "router_sm100 fused: no served kernel for routing mode ", mode, ", E = ",
              E, ", top_k = ", g.K, ", S = ", S, ", tile (N = ", N, ", MR = ", MR, ", CS = ", CS,
              ")");
}

} // namespace rt100
