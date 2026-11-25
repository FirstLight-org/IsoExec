// SPDX-License-Identifier: Apache-2.0
// Grouped SM100 expert GEMMs. FC1 keeps gate/up accumulators separate; FC2
// resets each K segment and merges rounded leaves in the declared tree.
// Warp roles and mbarrier handoff follow the persistent tcgen05 pipeline.

#include "blackwell.cuh"
#include "numerics.cuh"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>
#include <cstdlib>
#include <string>

namespace moe100 {
namespace {

constexpr uint32_t kBlockM = 128;   // rows per CTA tile == the cta_group::1 UMMA M
constexpr uint32_t kEpiWarp0 = 4;   // epilogue warps 4..11 -> TMEM quarters (warp & 3)
//: EIGHT epilogue warps, not four, and this is the FC2 change.
constexpr uint32_t kEpiWarps = 8;
constexpr uint32_t kEpiThreads = 32 * kEpiWarps;
constexpr uint32_t kThreads = 32 * (kEpiWarp0 + kEpiWarps);

//: SIXTEEN epilogue warps are available as a per-shape choice, and ``EW`` is that choice.
constexpr uint32_t kWideEpiWarps = 16;

template <bool GLU, uint32_t BN> struct Shapes {
  static constexpr uint32_t kSmemA = kBlockM * kAtomK * 2;
  static constexpr uint32_t kSmemB = BN * kAtomK * 2 * (GLU ? 2 : 1);
  static constexpr uint32_t kPerStage = kSmemA + kSmemB;
};

// -------------------------------------------------------------------- kernel ---


//: ROUND TWO COLUMNS WITH ONE CONVERT.
template <class T> __device__ __forceinline__ void round_pair(float a, float b, float &ra, float &rb) {
  if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    const __nv_bfloat162 p = __floats2bfloat162_rn(a, b);
    ra = __low2float(p);
    rb = __high2float(p);
  } else {
    const __half2 p = __floats2half2_rn(a, b);
    ra = __low2float(p);
    rb = __high2float(p);
  }
}

template <class T, int L> __device__ __forceinline__ void leaf_tree_pair(const float *s0, const float *s1,
                                                                         float &o0, float &o1) {
  float n0[L], n1[L];
#pragma unroll
  for (int i = 0; i < L; ++i)
    round_pair<T>(s0[i], s1[i], n0[i], n1[i]);
#pragma unroll
  for (int width = 1; width < L; width *= 2)
#pragma unroll
    for (int i = 0; i < L; i += 2 * width) {
      n0[i] = __fadd_rn(n0[i], n0[i + width]);
      n1[i] = __fadd_rn(n1[i], n1[i + width]);
    }
  o0 = n0[0];
  o1 = n1[0];
}

//: the tree of every column of one chunk: ``raw`` is ``L`` accumulators of ``C`` columns.
template <class T, int L, int C, bool PACK>
__device__ __forceinline__ void tree_row(const uint32_t (&raw)[L][C], float (&out)[C]) {
#pragma unroll
  for (int i = 0; i < C; i += (PACK ? 2 : 1)) {
    float a[L], b[L];
#pragma unroll
    for (int x = 0; x < L; ++x) {
      a[x] = __uint_as_float(raw[x][i]);
      if constexpr (PACK)
        b[x] = __uint_as_float(raw[x][i + 1]);
    }
    if constexpr (PACK) {
      leaf_tree_pair<T, L>(a, b, out[i], out[i + 1]);
    } else {
      out[i] = leaf_tree<T, L>(a);
    }
  }
}

// ``G`` IS THE NUMBER OF SEGMENT GROUPS, and it is what buys the second accumulator
// generation at eight leaves.
template <class T, uint32_t L, bool GLU, uint32_t BN, uint32_t STAGES, bool NARROW = false,
          uint32_t G = 1, uint32_t EW = kEpiWarps, bool PACK = false, bool PAIR = false>
__global__ __launch_bounds__(32 * (kEpiWarp0 + EW), 1) void grouped_kernel(
    void *__restrict__ out, const int *__restrict__ tiles,
    const int *__restrict__ total, const float *__restrict__ probs, uint32_t n,
    uint32_t seg_atoms, uint32_t n_blocks, uint32_t gate_up_half, bool short_fetch,
    const __grid_constant__ CUtensorMap am8, const __grid_constant__ CUtensorMap am32,
    const __grid_constant__ CUtensorMap am64, const __grid_constant__ CUtensorMap am,
    const __grid_constant__ CUtensorMap bm) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  constexpr bool kTwoB = GLU || PAIR;                 // two B tiles per stage, two accumulators
  using Sh = Shapes<kTwoB, BN>;
  constexpr uint32_t kGroups = GLU ? 1u : G;         // segment groups per work item
  constexpr uint32_t kAcc = kTwoB ? 2u : L / kGroups;  // live accumulators, one per group segment
  static_assert(!PAIR || (!GLU && L == 1 && G == 1), "PAIR serves the single-segment raw projection");
  // TWO ACCUMULATOR GENERATIONS WHEREVER THEY FIT.
  constexpr uint32_t kGens = (2u * kAcc * BN <= 512u) ? 2u : 1u;
  constexpr uint32_t kTmemCols = kGens * kAcc * BN;
  constexpr uint32_t kChunk = GLU ? 16u : 8u;        // columns per tcgen05.ld
  //: how many epilogue warps share one TMEM quarter, and therefore how the column chunks split.
  //: Capped by the chunk count, so a narrow tile leaves the extra warps with no chunks rather
  //: than requiring the tile to be wide enough for them.
  constexpr uint32_t kChunks = BN / kChunk;
  constexpr uint32_t kSubs = kChunks < (EW / 4u) ? kChunks : (EW / 4u);
  //: how many chunk iterations one epilogue warp walks -- the depth of the register array that
  //: carries a group's subtree root across the group boundary, and a compile-time constant so
  //: that array is registers and not local memory.
  constexpr uint32_t kParts = kChunks / kSubs;
  static_assert(kTmemCols <= 512, "accumulator does not fit in tensor memory");
  static_assert(L % kGroups == 0, "the segment groups must partition the leaves evenly");
  static_assert(kGroups == 1 || kGroups == 2 || kGroups == 4, "one, two or four segment groups");
  static_assert(kGroups == 1 || !NARROW, "grouped storage is the fp32 tree epilogue");
  static_assert(BN % kChunk == 0, "column chunk must divide the tile width");
  static_assert(EW % 4 == 0, "the epilogue warps cover the four TMEM quarters evenly");
  static_assert(kChunks % kSubs == 0, "the epilogue warps of a quarter must split the chunks");
  static_assert(!GLU || L == 1, "the GLU epilogue is a single-segment program");
  static_assert(!NARROW || (L == 1 && !GLU), "narrow storage holds one rounded segment");

  extern __shared__ __align__(1024) uint8_t smem[];
  auto smem_a = [&](uint32_t s) { return smem + s * Sh::kPerStage; };
  auto smem_b = [&](uint32_t s) { return smem + s * Sh::kPerStage + Sh::kSmemA; };
  auto bar_base = reinterpret_cast<Barrier *>(smem + STAGES * Sh::kPerStage);
  auto full = [&](uint32_t s) { return bar_base + s; };
  auto empty = [&](uint32_t s) { return bar_base + STAGES + s; };
  auto acc_full = [&](uint32_t g) { return bar_base + 2 * STAGES + g; };
  auto acc_empty = [&](uint32_t g) { return bar_base + 2 * STAGES + kGens + g; };
  auto tmem_slot = reinterpret_cast<uint32_t *>(bar_base + 2 * STAGES + 2 * kGens);

  const uint32_t warp = __shfl_sync(0xffffffffu, threadIdx.x / 32u, 0);
  const uint32_t lane = lane_id();

  if (warp == 1) {
    if (lane < 2 * STAGES + 2 * kGens) {
      // One mbarrier per lane. full[] takes the TMA's single arrival, empty[] the UMMA's, and each
      // accumulator generation is one UMMA arrival against the epilogue's releases -- ONE ARRIVAL
      // PER WARP, not per thread: ``EW`` of them, behind a ``__syncwarp``, which is the pattern
      // CUTLASS's UMMA consumer pipelines use and is cheaper than one atomic update of the same
      // mbarrier per thread (what this kernel did through round four and what README gap item 6
      // named).
      (bar_base + lane)->init(lane < 2 * STAGES + kGens ? 1u : EW);
    }
    cutlass::arch::fence_barrier_init();
  }
  if (warp == 2) {
    // ``tcgen05.alloc`` is ``.sync.aligned``: one fully active warp, never one lane.
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
  uint32_t tmem_base = 0;
  if (warp == 1 || (warp >= kEpiWarp0 && warp < kEpiWarp0 + EW)) {
    while ((tmem_base = ld_shared_u32(tmem_slot)) == 0xffffffffu) {
    }
  }

  const uint32_t items = uint32_t(*total) * n_blocks;
  //: stages per work item, and per segment GROUP: the TMA producer walks all of an item's K
  //: atoms in one ascending sweep whatever the grouping, so only the consumers see the boundary.
  const uint32_t iters = (GLU ? 1u : L) * seg_atoms;
  const uint32_t giters = iters / kGroups;
  const uint32_t seg_elems = seg_atoms * kAtomK;

  // Work-item -> (tile, column block). Consecutive items share a row block, so one
  // wave covers few row blocks and all their column blocks: the expert's weight block
  // is read by the whole wave and stays resident.
  auto decode = [&](uint32_t item, uint32_t &tile, uint32_t &nb) {
    tile = item / n_blocks;
    nb = item - tile * n_blocks;
  };

  uint32_t slot = 0, phase = 0, issued = 0;

  if (warp == 0 && cute::elect_one_sync()) {
    // ---- TMA producer -------------------------------------------------------
    for (uint32_t item = blockIdx.x; item < items; item += gridDim.x) {
      uint32_t tile, nb;
      decode(item, tile, nb);
      const uint32_t expert = uint32_t(tiles[3 * tile]);
      const uint32_t first = uint32_t(tiles[3 * tile + 1]);
      const uint32_t end = min(uint32_t(tiles[3 * tile + 2]), first + kBlockM);
      const uint32_t n_idx = nb * BN;
      // THE A-FETCH HEIGHT. A TMA box taller than the tensor does not skip the missing rows:
      // it zero-fills them into shared memory, and those writes are issued and counted in the
      // mbarrier's transaction expectation whether or not anything reads them.
      const uint32_t height = short_fetch ? (end - first) : kBlockM;
      const CUtensorMap *a_map = &am;
      uint32_t a_bytes = Sh::kSmemA;
      if (height <= 8u) {
        a_map = &am8;
        a_bytes = 8u * kAtomK * 2u;
      } else if (height <= 32u) {
        a_map = &am32;
        a_bytes = 32u * kAtomK * 2u;
      } else if (height <= 64u) {
        //: THE 64-ROW RUNG, and the row count that asked for it: Qwen's 2048-token prefill puts
        //: exactly 64 routed rows in an expert (16384 rows over 256 experts), which the three-rung
        //: ladder rounded up to the full 128-row box and therefore zero-filled half of every A
        //: stage. Same ownership argument as the other rungs: rows above the box are past the
        //: expert's end, are never stored, and a UMMA row reads only its own row.
        a_map = &am64;
        a_bytes = 64u * kAtomK * 2u;
      }
      const uint32_t stage_bytes = a_bytes + Sh::kSmemB;
      for (uint32_t it = 0; it < iters; ++it) {
        if (issued >= STAGES)
          empty(slot)->wait(phase ^ 1u);
        const uint32_t seg = GLU ? 0u : it / seg_atoms;
        const uint32_t katom = GLU ? it : it - seg * seg_atoms;
        const uint32_t k_idx = seg * seg_elems + katom * kAtomK;
        auto *bar = reinterpret_cast<uint64_t *>(full(slot));
        const uint64_t hint = uint64_t(cute::TMA::CacheHintSm100::EVICT_NORMAL);
        cute::SM90_TMA_LOAD_2D::copy(a_map, bar, hint, smem_a(slot), k_idx, first);
        cute::SM90_TMA_LOAD_3D::copy(&bm, bar, hint, smem_b(slot), k_idx, n_idx, expert);
        if constexpr (kTwoB)
          cute::SM90_TMA_LOAD_3D::copy(&bm, bar, hint, smem_b(slot) + BN * kAtomK * 2,
                                       k_idx, gate_up_half + n_idx, expert);
        full(slot)->arrive_and_expect_tx(stage_bytes);
        ++issued;
        if (++slot == STAGES) {
          slot = 0;
          phase ^= 1u;
        }
      }
    }
  } else if (warp == 1) {
    // ---- UMMA issuer --------------------------------------------------------
    using Elem = std::conditional_t<std::is_same_v<T, __nv_bfloat16>, cutlass::bfloat16_t,
                                    cutlass::half_t>;
    const uint64_t idesc = cute::UMMA::make_runtime_instr_desc(
        cute::UMMA::make_instr_desc<Elem, Elem, float, kBlockM, BN, cute::UMMA::Major::K,
                                    cute::UMMA::Major::K>());
    auto a_desc = make_desc_k_major<T>(smem_a(0));
    auto b_desc = make_desc_k_major<T>(smem_b(0));
    const uint32_t a_lo0 = a_desc.lo, b_lo0 = b_desc.lo;
    uint32_t gen = 0;
    for (uint32_t item = blockIdx.x; item < items; item += gridDim.x) {
      // One generation handoff per SEGMENT GROUP. At kGroups == 1 this loop runs once and the
      // code below is the item scheduler verbatim.
      for (uint32_t grp = 0; grp < kGroups; ++grp) {
        const uint32_t g = kGens == 2u ? (gen & 1u) : 0u;
        const uint32_t use = gen / kGens;   // how many times THIS generation was used before
        const uint32_t acc0 = g * kAcc * BN;
        if (use)
          acc_empty(g)->wait((use - 1u) & 1u);
        tcgen05_after_thread_sync();
        for (uint32_t it = 0; it < giters; ++it) {
          full(slot)->wait(phase);
          tcgen05_after_thread_sync();
          //: the segment index INSIDE the group is the accumulator; the group's own offset is
          //: already in the K atom the TMA delivered, so nothing here reads it.
          const uint32_t seg = GLU ? 0u : it / seg_atoms;
          const uint32_t katom = GLU ? it : it - seg * seg_atoms;
          const uint32_t a_base = a_lo0 + slot * (Sh::kPerStage >> 4);
          const uint32_t b_base = b_lo0 + slot * (Sh::kPerStage >> 4);
          if (cute::elect_one_sync()) {
#pragma unroll
            for (uint32_t kk = 0; kk < kAtomK / kUmmaK; ++kk) {
              const uint32_t in_atom = kk * kUmmaK;
              a_desc.lo = advance_lo<T>(a_base, in_atom);
              const uint32_t scale = (katom > 0 || kk > 0) ? 1u : 0u;
              b_desc.lo = advance_lo<T>(b_base, in_atom);
              Umma1Sm::fma(a_desc, b_desc, tmem_base + acc0 + (GLU ? 0u : seg * BN), scale,
                           idesc);
              if constexpr (kTwoB) {
                b_desc.lo = advance_lo<T>(b_base + ((BN * kAtomK * 2) >> 4), in_atom);
                Umma1Sm::fma(a_desc, b_desc, tmem_base + acc0 + BN, scale, idesc);
              }
            }
          }
          cutlass::arch::umma_arrive(reinterpret_cast<uint64_t *>(empty(slot)));
          if (++slot == STAGES) {
            slot = 0;
            phase ^= 1u;
          }
        }
        cutlass::arch::umma_arrive(reinterpret_cast<uint64_t *>(acc_full(g)));
        ++gen;
      }
    }
  } else if (warp >= kEpiWarp0 && warp < kEpiWarp0 + EW) {
    // ---- epilogue -----------------------------------------------------------
    // The TMEM datapath quarter a warp reads is its warp index modulo four, so warps 4..11 cover
    // the 128 rows twice over and the second four take the odd column chunks.
    const uint32_t quarter = warp & 3u;
    const uint32_t sub = (warp - kEpiWarp0) >> 2;
    uint32_t gen = 0;
    for (uint32_t item = blockIdx.x; item < items; item += gridDim.x) {
      uint32_t tile, nb;
      decode(item, tile, nb);
      const uint32_t first = uint32_t(tiles[3 * tile + 1]);
      const uint32_t end = min(uint32_t(tiles[3 * tile + 2]), first + kBlockM);
      const uint32_t row = first + quarter * 32u + lane;
      const uint32_t n_idx = nb * BN;
      // The GLU epilogue's routing probability is a function of the ROW, not of the column chunk;
      // reading it inside the chunk loop cost eight global loads per item per thread and made
      // ``long_scoreboard`` the kernel's dominant stall in the ncu pass.
      const float prob = (GLU && probs && row < end) ? probs[row] : 0.f;
      //: group 0's subtree root, carried in registers across the group boundary. Constant
      //: indices only, which is what keeps it out of local memory; unused at kGroups == 1.
      float part[kGroups == 2u ? kParts : 1u][kChunk];
      //: at kGroups == 4 the same carry is two registers, one per quarter-subtree of the declared
      //: eight-leaf tree: ``part0`` accumulates the left half ((l0+l1)+(l2+l3)) across groups 0-1,
      //: ``part1`` holds the right half's first pair (l4+l5) from group 2. Constant indices only.
      float part0[kGroups == 4u ? kParts : 1u][kChunk];
      float part1[kGroups == 4u ? kParts : 1u][kChunk];
      // (Skipping the chunk loop entirely for TMEM quarters wholly past the tile's real rows --
      // ``first + quarter*32 >= end``, which is warp-uniform -- was built and measured NEUTRAL;
      // see README "Rejected variants". It is what says the epilogue's cost is its serialized
      // latency chain and not the arithmetic in it, which is what ``G = 2`` attacks.)
#pragma unroll
      for (uint32_t grp = 0; grp < kGroups; ++grp) {
      const uint32_t g = kGens == 2u ? (gen & 1u) : 0u;
      const uint32_t acc0 = g * kAcc * BN;
      acc_full(g)->wait((gen / kGens) & 1u);
      tcgen05_after_thread_sync();
      if constexpr (kGroups == 2u) {
        // TWO SEGMENT GROUPS. Group 0 computes the left subtree of the declared eight-leaf tree
        // and keeps its root; group 1 computes the right subtree and adds it with the tree's own
        // ``__fadd_rn``. Same leaves, same rounding, same shape -- and one generation each, so
        // group 1's mainloop runs while this runs.
#pragma unroll
        for (uint32_t ci = 0; ci < kParts; ++ci) {
          const uint32_t c = sub * kChunk + ci * kSubs * kChunk;
          uint32_t raw[kAcc][kChunk];
#pragma unroll
          for (uint32_t a = 0; a < kAcc; ++a)
            TmemLd<kChunk>::copy(tmem_base + acc0 + a * BN + c, raw[a]);
          cutlass::arch::fence_view_async_tmem_load();
          if (row < end) {
            float value[kChunk];
            float rt[kChunk];
            tree_row<T, int(kAcc), kChunk, PACK>(raw, rt);
#pragma unroll
            for (uint32_t i = 0; i < kChunk; ++i) {
              const float root = rt[i];
              if (grp == 0)
                part[ci][i] = root;
              else
                value[i] = __fadd_rn(part[ci][i], root);
            }
            if (grp) {
              auto *dst = static_cast<float *>(out) + int64_t(row) * n + n_idx + c;
              if (n_idx + c + kChunk <= n) {
#pragma unroll
                for (uint32_t i = 0; i < kChunk; i += 4)
                  *reinterpret_cast<float4 *>(dst + i) =
                      make_float4(value[i], value[i + 1], value[i + 2], value[i + 3]);
              } else {
                for (uint32_t i = 0; i < kChunk && n_idx + c + i < n; ++i)
                  dst[i] = value[i];
              }
            }
          }
        }
      } else if constexpr (kGroups == 4u) {
        // FOUR SEGMENT GROUPS, TWO LEAVES EACH: the declared eight-leaf tree as its four leaf
        // pairs.
#pragma unroll
        for (uint32_t ci = 0; ci < kParts; ++ci) {
          const uint32_t c = sub * kChunk + ci * kSubs * kChunk;
          uint32_t raw[kAcc][kChunk];
#pragma unroll
          for (uint32_t a = 0; a < kAcc; ++a)
            TmemLd<kChunk>::copy(tmem_base + acc0 + a * BN + c, raw[a]);
          cutlass::arch::fence_view_async_tmem_load();
          if (row < end) {
            float value[kChunk];
            float rt[kChunk];
            tree_row<T, int(kAcc), kChunk, PACK>(raw, rt);
#pragma unroll
            for (uint32_t i = 0; i < kChunk; ++i) {
              const float root = rt[i];
              if (grp == 0)
                part0[ci][i] = root;
              else if (grp == 1)
                part0[ci][i] = __fadd_rn(part0[ci][i], root);
              else if (grp == 2)
                part1[ci][i] = root;
              else
                value[i] = __fadd_rn(part0[ci][i], __fadd_rn(part1[ci][i], root));
            }
            if (grp == 3) {
              auto *dst = static_cast<float *>(out) + int64_t(row) * n + n_idx + c;
              if (n_idx + c + kChunk <= n) {
#pragma unroll
                for (uint32_t i = 0; i < kChunk; i += 4)
                  *reinterpret_cast<float4 *>(dst + i) =
                      make_float4(value[i], value[i + 1], value[i + 2], value[i + 3]);
              } else {
                for (uint32_t i = 0; i < kChunk && n_idx + c + i < n; ++i)
                  dst[i] = value[i];
              }
            }
          }
        }
      } else {
#pragma unroll 1
      for (uint32_t c = sub * kChunk; c < BN; c += kSubs * kChunk) {
        uint32_t raw[kAcc][kChunk];
#pragma unroll
        for (uint32_t a = 0; a < kAcc; ++a)
          TmemLd<kChunk>::copy(tmem_base + acc0 + a * BN + c, raw[a]);
        cutlass::arch::fence_view_async_tmem_load();
        if (row < end) {
          if constexpr (GLU) {
            const float p = prob;
            T value[kChunk];
#pragma unroll
            for (uint32_t i = 0; i < kChunk; ++i)
              value[i] = cast<T>(glu_epilogue<T>(__uint_as_float(raw[0][i]),
                                                 __uint_as_float(raw[1][i]), p,
                                                 probs != nullptr));
            auto *dst = static_cast<T *>(out) + int64_t(row) * n + n_idx + c;
            if (n_idx + c + kChunk <= n) {
#pragma unroll
              for (uint32_t i = 0; i < kChunk; i += 8)
                *reinterpret_cast<uint4 *>(dst + i) =
                    *reinterpret_cast<const uint4 *>(value + i);
            } else {
              for (uint32_t i = 0; i < kChunk && n_idx + c + i < n; ++i)
                dst[i] = value[i];
            }
          } else if constexpr (NARROW) {
#pragma unroll
            for (uint32_t a = 0; a < (PAIR ? 2u : 1u); ++a) {
              const uint32_t col = (a ? gate_up_half : 0u) + n_idx + c;
              T value[kChunk];
#pragma unroll
              for (uint32_t i = 0; i < kChunk; ++i)
                value[i] = cast<T>(__uint_as_float(raw[a][i]));
              auto *dst = static_cast<T *>(out) + int64_t(row) * n + col;
              if (col + kChunk <= n) {
                *reinterpret_cast<uint4 *>(dst) = *reinterpret_cast<const uint4 *>(value);
              } else {
                for (uint32_t i = 0; i < kChunk && col + i < n; ++i)
                  dst[i] = value[i];
              }
            }
          } else if constexpr (PAIR) {
#pragma unroll
            for (uint32_t a = 0; a < 2u; ++a) {
              const uint32_t col = (a ? gate_up_half : 0u) + n_idx + c;
              uint32_t one[1][kChunk];
#pragma unroll
              for (uint32_t i = 0; i < kChunk; ++i) one[0][i] = raw[a][i];
              float value[kChunk];
              tree_row<T, 1, kChunk, PACK>(one, value);
              auto *dst = static_cast<float *>(out) + int64_t(row) * n + col;
              if (col + kChunk <= n) {
#pragma unroll
                for (uint32_t i = 0; i < kChunk; i += 4)
                  *reinterpret_cast<float4 *>(dst + i) =
                      make_float4(value[i], value[i + 1], value[i + 2], value[i + 3]);
              } else {
                for (uint32_t i = 0; i < kChunk && col + i < n; ++i)
                  dst[i] = value[i];
              }
            }
          } else {
            float value[kChunk];
            tree_row<T, int(L), kChunk, PACK>(raw, value);
            auto *dst = static_cast<float *>(out) + int64_t(row) * n + n_idx + c;
            if (n_idx + c + kChunk <= n) {
#pragma unroll
              for (uint32_t i = 0; i < kChunk; i += 4)
                *reinterpret_cast<float4 *>(dst + i) =
                    make_float4(value[i], value[i + 1], value[i + 2], value[i + 3]);
            } else {
              for (uint32_t i = 0; i < kChunk && n_idx + c + i < n; ++i)
                dst[i] = value[i];
            }
          }
        }
      }
      }
      tcgen05_before_thread_sync();
      // ONE ARRIVAL PER WARP, NOT PER THREAD.
      __syncwarp();
      if (lane == 0)
        acc_empty(g)->arrive();
      ++gen;
      }
    }
  }

  __syncthreads();
  if (warp == 2)
    // Also ``.sync.aligned``, and by the warp that allocated. Every epilogue read is past
    // the barrier above, so the allocation's base address is in the slot by now.
    TmemAllocator().free(ld_shared_u32(tmem_slot), kTmemCols);
#else
  if (threadIdx.x == 0)
    __trap();
#endif
}

// ------------------------------------------------------- swap-AB raw projection ---
//
//: Swap A/B for sparse expert rows to keep token UMMA tiles small. Each
//: output retains one ascending K chain; padded token rows are never stored.
template <class T, uint32_t NTOK, uint32_t STAGES, bool F32OUT>
__global__ __launch_bounds__(32 * (kEpiWarp0 + kEpiWarps), 1) void swap_kernel(
    void *__restrict__ out, const int *__restrict__ tiles, const int *__restrict__ total, uint32_t n,
    uint32_t atoms, uint32_t w_blocks, const __grid_constant__ CUtensorMap xm,
    const __grid_constant__ CUtensorMap wm) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  constexpr uint32_t kSmemW = 2u * kBlockM * kAtomK * 2u;   // two 128-row weight sub-tiles
  constexpr uint32_t kSmemX = NTOK * kAtomK * 2u;
  constexpr uint32_t kPerStage = kSmemW + kSmemX;
  constexpr uint32_t kGens = 2u;
  constexpr uint32_t kTmemCols = kGens * 2u * NTOK < 32u ? 32u : kGens * 2u * NTOK;
  static_assert(NTOK % 16 == 0 && NTOK >= 16 && NTOK <= 128, "UMMA N for M = 128");
  static_assert(kPerStage % 1024 == 0, "stages keep the 1024-byte swizzle alignment");

  extern __shared__ __align__(1024) uint8_t smem[];
  auto smem_w = [&](uint32_t s) { return smem + s * kPerStage; };
  auto smem_x = [&](uint32_t s) { return smem + s * kPerStage + kSmemW; };
  auto bar_base = reinterpret_cast<Barrier *>(smem + STAGES * kPerStage);
  auto full = [&](uint32_t s) { return bar_base + s; };
  auto empty = [&](uint32_t s) { return bar_base + STAGES + s; };
  auto acc_full = [&](uint32_t g) { return bar_base + 2 * STAGES + g; };
  auto acc_empty = [&](uint32_t g) { return bar_base + 2 * STAGES + kGens + g; };
  auto tmem_slot = reinterpret_cast<uint32_t *>(bar_base + 2 * STAGES + 2 * kGens);

  const uint32_t warp = __shfl_sync(0xffffffffu, threadIdx.x / 32u, 0);
  const uint32_t lane = lane_id();
  if (warp == 1) {
    if (lane < 2 * STAGES + 2 * kGens)
      (bar_base + lane)->init(lane < 2 * STAGES + kGens ? 1u : kEpiWarps);
    cutlass::arch::fence_barrier_init();
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
  uint32_t tmem_base = 0;
  if (warp == 1 || (warp >= kEpiWarp0 && warp < kEpiWarp0 + kEpiWarps)) {
    while ((tmem_base = ld_shared_u32(tmem_slot)) == 0xffffffffu) {
    }
  }

  const uint32_t items = uint32_t(*total) * w_blocks;
  // item -> (expert, first row, end row, column block).
  auto decode = [&](uint32_t item, uint32_t &expert, uint32_t &first, uint32_t &end, uint32_t &wb) {
    wb = item % w_blocks;
    const uint32_t tile = item / w_blocks;
    expert = uint32_t(tiles[3 * tile]);
    first = uint32_t(tiles[3 * tile + 1]);
    end = min(uint32_t(tiles[3 * tile + 2]), first + kBlockM);
  };

  uint32_t slot = 0, phase = 0, issued = 0;
  if (warp == 0 && cute::elect_one_sync()) {
    for (uint32_t item = blockIdx.x; item < items; item += gridDim.x) {
      uint32_t expert, first, end, wb;
      decode(item, expert, first, end, wb);
      for (uint32_t r0 = first; r0 < end; r0 += NTOK)
      for (uint32_t it = 0; it < atoms; ++it) {
        if (issued >= STAGES)
          empty(slot)->wait(phase ^ 1u);
        const uint32_t k_idx = it * kAtomK;
        auto *bar = reinterpret_cast<uint64_t *>(full(slot));
        const uint64_t hint = uint64_t(cute::TMA::CacheHintSm100::EVICT_NORMAL);
        cute::SM90_TMA_LOAD_3D::copy(&wm, bar, hint, smem_w(slot), k_idx, wb * 256u, expert);
        cute::SM90_TMA_LOAD_3D::copy(&wm, bar, hint, smem_w(slot) + kBlockM * kAtomK * 2u, k_idx,
                                     wb * 256u + kBlockM, expert);
        cute::SM90_TMA_LOAD_2D::copy(&xm, bar, hint, smem_x(slot), k_idx, r0);
        full(slot)->arrive_and_expect_tx(kPerStage);
        ++issued;
        if (++slot == STAGES) {
          slot = 0;
          phase ^= 1u;
        }
      }
    }
  } else if (warp == 1) {
    using Elem = std::conditional_t<std::is_same_v<T, __nv_bfloat16>, cutlass::bfloat16_t,
                                    cutlass::half_t>;
    const uint64_t idesc = cute::UMMA::make_runtime_instr_desc(
        cute::UMMA::make_instr_desc<Elem, Elem, float, kBlockM, NTOK, cute::UMMA::Major::K,
                                    cute::UMMA::Major::K>());
    auto w_desc = make_desc_k_major<T>(smem_w(0));
    auto x_desc = make_desc_k_major<T>(smem_x(0));
    const uint32_t w_lo0 = w_desc.lo, x_lo0 = x_desc.lo;
    uint32_t gen = 0;
    for (uint32_t item = blockIdx.x; item < items; item += gridDim.x) {
      uint32_t expert, first, end, wb;
      decode(item, expert, first, end, wb);
      for (uint32_t r0 = first; r0 < end; r0 += NTOK) {
      const uint32_t g = gen & 1u, use = gen / kGens;
      const uint32_t acc0 = g * 2u * NTOK;
      if (use)
        acc_empty(g)->wait((use - 1u) & 1u);
      tcgen05_after_thread_sync();
      for (uint32_t it = 0; it < atoms; ++it) {
        full(slot)->wait(phase);
        tcgen05_after_thread_sync();
        const uint32_t w_base = w_lo0 + slot * (kPerStage >> 4);
        const uint32_t x_base = x_lo0 + slot * (kPerStage >> 4);
        if (cute::elect_one_sync()) {
#pragma unroll
          for (uint32_t kk = 0; kk < kAtomK / kUmmaK; ++kk) {
            const uint32_t in_atom = kk * kUmmaK;
            const uint32_t scale = (it > 0 || kk > 0) ? 1u : 0u;
            x_desc.lo = advance_lo<T>(x_base, in_atom);
#pragma unroll
            for (uint32_t sub = 0; sub < 2u; ++sub) {
              w_desc.lo = advance_lo<T>(w_base + sub * ((kBlockM * kAtomK * 2u) >> 4), in_atom);
              Umma1Sm::fma(w_desc, x_desc, tmem_base + acc0 + sub * NTOK, scale, idesc);
            }
          }
        }
        cutlass::arch::umma_arrive(reinterpret_cast<uint64_t *>(empty(slot)));
        if (++slot == STAGES) {
          slot = 0;
          phase ^= 1u;
        }
      }
      cutlass::arch::umma_arrive(reinterpret_cast<uint64_t *>(acc_full(g)));
      ++gen;
      }
    }
  } else if (warp >= kEpiWarp0 && warp < kEpiWarp0 + kEpiWarps) {
    // Warp (quarter, sub): TMEM lane = output column sub*128 + quarter*32 + lane of the block;
    // TMEM column j of the sub-tile = token row r0 + j. A store of one row by one warp is 32
    // consecutive columns.
    const uint32_t quarter = warp & 3u;
    const uint32_t sub = (warp - kEpiWarp0) >> 2;
    uint32_t gen = 0;
    for (uint32_t item = blockIdx.x; item < items; item += gridDim.x) {
      uint32_t expert, first, end, wb;
      decode(item, expert, first, end, wb);
      for (uint32_t r0 = first; r0 < end; r0 += NTOK) {
      const uint32_t cnt = min(end - r0, NTOK);
      const uint32_t g = gen & 1u;
      const uint32_t acc0 = g * 2u * NTOK + sub * NTOK;
      acc_full(g)->wait((gen / kGens) & 1u);
      tcgen05_after_thread_sync();
      const uint32_t col = wb * 256u + sub * kBlockM + quarter * 32u + lane;
#pragma unroll
      for (uint32_t c = 0; c < NTOK; c += 8u) {
        if (c >= cnt)
          break;   // warp-uniform: cnt is the item's
        uint32_t raw[8];
        TmemLd<8>::copy(tmem_base + acc0 + c, raw);
        cutlass::arch::fence_view_async_tmem_load();
        if (col < n) {
#pragma unroll
          for (uint32_t j = 0; j < 8u; ++j) {
            if (c + j < cnt) {
              const int64_t at = int64_t(r0 + c + j) * n + col;
              if constexpr (F32OUT)
                static_cast<float *>(out)[at] = rounded<T>(__uint_as_float(raw[j]));
              else
                static_cast<T *>(out)[at] = cast<T>(__uint_as_float(raw[j]));
            }
          }
        }
      }
      tcgen05_before_thread_sync();
      __syncwarp();
      if (lane == 0)
        acc_empty(g)->arrive();
      ++gen;
      }
    }
  }
  __syncthreads();
  if (warp == 2)
    TmemAllocator().free(ld_shared_u32(tmem_slot), kTmemCols);
#else
  if (threadIdx.x == 0)
    __trap();
#endif
}

// ---------------------------------------------------------------------- host ---

int sm_count() {
  static int count = 0;
  if (count == 0) {
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device);
  }
  return count;
}




// THE RAW PROJECTION'S COLUMN TILE (round seven).
bool raw_wide(int64_t n) {
  return n % 256 == 0;
}


//: an upper bound on the tile count from the row count and the expert count alone -- the same
//: bound ``ops.py::_tile_bound`` computes, and the reason the rule below needs no device read.
int64_t tile_bound(int64_t rows, int64_t experts) {
  if (!rows)
    return 1;
  return std::min<int64_t>(rows, (rows + kBlockM - 1) / kBlockM + experts - 1);
}






template <class T, uint32_t L, bool GLU, uint32_t BN, uint32_t STAGES, bool NARROW = false,
          uint32_t G = 1, uint32_t EW = kEpiWarps, bool PACK = false, bool PAIR = false>
void launch_one(void *out, const at::Tensor &x, const at::Tensor &w,
                const at::Tensor &tiles, const at::Tensor &total, const float *probs,
                int64_t n, int64_t seg_atoms, int64_t gate_up_half, const CUtensorMap (&am)[4],
                CUtensorMap bm, cudaStream_t stream) {
  using Sh = Shapes<GLU || PAIR, BN>;
  constexpr uint32_t kAccH = (GLU || PAIR) ? 2u : L / (GLU ? 1u : G);
  constexpr uint32_t kGensH = (2u * kAccH * BN <= 512u) ? 2u : 1u;
  constexpr uint32_t bytes = STAGES * Sh::kPerStage + 64 * (2 * STAGES + 2 * kGensH + 1);
  auto kernel = grouped_kernel<T, L, GLU, BN, STAGES, NARROW, G, EW, PACK, PAIR>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(bytes)));
  // PAIR: an item owns column blocks nb and half + nb (``gate_up_half`` = half, a BN multiple).
  const uint32_t n_blocks = PAIR ? uint32_t(gate_up_half / BN) : uint32_t((n + BN - 1) / BN);
  const uint32_t grid = uint32_t(sm_count());
  kernel<<<grid, 32 * (kEpiWarp0 + EW), bytes, stream>>>(
      out, tiles.data_ptr<int>(), total.data_ptr<int>(), probs, uint32_t(n),
      uint32_t(seg_atoms), n_blocks, uint32_t(gate_up_half), true, am[0], am[1],
      am[2], am[3], bm);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

//: the A operand at four box heights: 8 rows (one 128-byte swizzle atom), 32, 64, and the full
//: tile. Only the number of rows the TMA delivers differs -- the shared-memory map, the swizzle,
//: the stage stride and every UMMA descriptor are byte-for-byte the same.
struct AMaps {
  CUtensorMap map[4];
};

AMaps encode_a(const at::Tensor &x) {
  AMaps out{};
  const uint32_t heights[4] = {8u, 32u, 64u, kBlockM};
  for (int i = 0; i < 4; ++i) {
    int rc = make_desc2(&out.map[i], x.data_ptr(), uint64_t(x.size(1)), uint64_t(x.size(0)),
                        uint64_t(x.stride(0)), kAtomK, heights[i]);
    TORCH_CHECK(rc == 0, "sm100 MoE: activation TMA descriptor rejected (", rc, ")");
  }
  return out;
}

template <uint32_t BN> CUtensorMap encode_b(const at::Tensor &w) {
  CUtensorMap map{};
  int rc = make_desc3(&map, w.data_ptr(), uint64_t(w.size(2)), uint64_t(w.size(1)),
                      uint64_t(w.size(0)), uint64_t(w.stride(1)), uint64_t(w.stride(0)),
                      kAtomK, BN);
  TORCH_CHECK(rc == 0, "sm100 MoE: weight TMA descriptor rejected (", rc, ")");
  return map;
}


//: the token tile of the swap-AB items, 0 for the non-swap items.
int swap_ntok(int64_t rows, int64_t experts, int64_t n) {
  if (n % 256 != 0)
    return 0;
  const double mean = double(rows) / double(std::max<int64_t>(experts, 1));
  return mean <= 28.0 ? 64 : 128;
}

template <class T, uint32_t NTOK, bool F32OUT>
void launch_swap(void *out, const at::Tensor &x, const at::Tensor &w, const at::Tensor &tiles,
                 const at::Tensor &total, int64_t n, int64_t atoms, cudaStream_t stream) {
  constexpr uint32_t kPerStage = 2u * kBlockM * kAtomK * 2u + NTOK * kAtomK * 2u;
  constexpr uint32_t kStages = (220u * 1024u) / kPerStage;
  constexpr uint32_t bytes = kStages * kPerStage + 64 * (2 * kStages + 2 * 2 + 1);
  auto kernel = swap_kernel<T, NTOK, kStages, F32OUT>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(bytes)));
  CUtensorMap xm{};
  int rc = make_desc2(&xm, x.data_ptr(), uint64_t(x.size(1)), uint64_t(x.size(0)), uint64_t(x.stride(0)),
                      kAtomK, NTOK);
  TORCH_CHECK(rc == 0, "sm100 MoE: swap token TMA descriptor rejected (", rc, ")");
  const CUtensorMap wm = encode_b<kBlockM>(w);
  kernel<<<uint32_t(sm_count()), kThreads, bytes, stream>>>(
      out, tiles.data_ptr<int>(), total.data_ptr<int>(), uint32_t(n), uint32_t(atoms), uint32_t(n / 256),
      xm, wm);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

//: dispatch the swap items when the rule asks for them; false leaves the caller's path.
template <bool F32OUT>
bool try_swap(void *out, const at::Tensor &x, const at::Tensor &w, const at::Tensor &tiles,
              const at::Tensor &total, int64_t n, int64_t atoms, cudaStream_t stream) {
  const int ntok = swap_ntok(x.size(0), w.size(0), n);
  if (!ntok)
    return false;
#define SWAP_T(T)                                                                        \
  do {                                                                                   \
    if (ntok == 16)                                                                      \
      launch_swap<T, 16, F32OUT>(out, x, w, tiles, total, n, atoms, stream);             \
    else if (ntok == 32)                                                                 \
      launch_swap<T, 32, F32OUT>(out, x, w, tiles, total, n, atoms, stream);             \
    else if (ntok == 64)                                                                 \
      launch_swap<T, 64, F32OUT>(out, x, w, tiles, total, n, atoms, stream);             \
    else                                                                                 \
      launch_swap<T, 128, F32OUT>(out, x, w, tiles, total, n, atoms, stream);            \
  } while (0)
  if (x.scalar_type() == at::kBFloat16)
    SWAP_T(__nv_bfloat16);
  else
    SWAP_T(__half);
#undef SWAP_T
  return true;
}

} // namespace

// ``x[R,K] @ gate_up[e][:, :]`` with the fused GLU epilogue -> ``h[R,I]`` operand dtype.
void fc1_glu_cuda(const at::Tensor &x, const at::Tensor &gate_up, const at::Tensor &tiles,
                  const at::Tensor &total, const at::Tensor &probs, at::Tensor out) {
  const c10::cuda::CUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t n = out.size(1), k = x.size(1);
  TORCH_CHECK(k % kAtomK == 0, "sm100 MoE FC1 needs K divisible by 64");
  TORCH_CHECK(n % 8 == 0, "sm100 MoE FC1 needs an intermediate width divisible by 8");
  const float *p = probs.numel() ? probs.data_ptr<float>() : nullptr;
  const auto amaps = encode_a(x);
  const int64_t atoms = k / kAtomK;
  // The narrow column tile is a PARALLELISM choice, not an arithmetic one: with few rows per
  // expert a 128-wide tile leaves the machine idle, and splitting the intermediate width
  // duplicates no work because a column block owns its whole K chain.
  const int64_t items = tile_bound(x.size(0), gate_up.size(0)) * ((n + 127) / 128);
  const int forced = -1;
  const bool narrow =
      n % 32 == 0 && (forced >= 0 ? forced == 1 : items < 2 * int64_t(sm_count()));
#define FC1_CASE(T, BN, ST)                                                             \
  do {                                                                                  \
    auto bm = encode_b<BN>(gate_up);                                                     \
    launch_one<T, 1, true, BN, ST>(out.data_ptr(), x, gate_up, tiles, total, p, n, atoms, \
                                   n, amaps.map, bm, stream);                             \
  } while (0)
#define FC1_WIDTH(T)                                                                    \
  do {                                                                                  \
    if (narrow)                                                                          \
      FC1_CASE(T, 32, 6);                                                                \
    else                                                                                 \
      FC1_CASE(T, 128, 4);                                                               \
  } while (0)
  if (x.scalar_type() == at::kBFloat16)
    FC1_WIDTH(__nv_bfloat16);
  else
    FC1_WIDTH(__half);
#undef FC1_WIDTH
#undef FC1_CASE
}

// The raw grouped projection: one K chain, rounded once into the operand dtype. Used by
// the backward recompute and by the single-segment compact-root storage option.
void fc1_cuda(const at::Tensor &x, const at::Tensor &w, const at::Tensor &tiles,
              const at::Tensor &total, at::Tensor out) {
  const c10::cuda::CUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t n = out.size(1), k = x.size(1);
  TORCH_CHECK(k % kAtomK == 0, "sm100 MoE FC1 needs K divisible by 64");
  TORCH_CHECK(n % 8 == 0, "sm100 MoE narrow output needs a width divisible by 8");
  const auto amaps = encode_a(x);
  auto bm = encode_b<128>(w);
  const int64_t atoms = k / kAtomK;
  // PAIRED ITEMS when the width splits into two halves of whole 128-column blocks: two
  // independent accumulator chains per item (see the kernel's ``PAIR``). 4 stages: two B tiles.
  const bool pair = 0 != 0 && n % 256 == 0;
  if (try_swap<false>(out.data_ptr(), x, w, tiles, total, n, atoms, stream)) {
  } else if (raw_wide(n)) {
    auto bm256 = encode_b<256>(w);
    if (x.scalar_type() == at::kBFloat16)
      launch_one<__nv_bfloat16, 1, false, 256, 4, true>(out.data_ptr(), x, w, tiles, total, nullptr, n, atoms,
                                                        0, amaps.map, bm256, stream);
    else
      launch_one<__half, 1, false, 256, 4, true>(out.data_ptr(), x, w, tiles, total, nullptr, n, atoms, 0,
                                                 amaps.map, bm256, stream);
  } else if (pair) {
    if (x.scalar_type() == at::kBFloat16)
      launch_one<__nv_bfloat16, 1, false, 128, 4, true, 1, kEpiWarps, false, true>(
          out.data_ptr(), x, w, tiles, total, nullptr, n, atoms, n / 2, amaps.map, bm, stream);
    else
      launch_one<__half, 1, false, 128, 4, true, 1, kEpiWarps, false, true>(
          out.data_ptr(), x, w, tiles, total, nullptr, n, atoms, n / 2, amaps.map, bm, stream);
  } else if (x.scalar_type() == at::kBFloat16)
    launch_one<__nv_bfloat16, 1, false, 128, 6, true>(out.data_ptr(), x, w, tiles, total,
                                                      nullptr, n, atoms, 0,
                                                      amaps.map, bm, stream);
  else
    launch_one<__half, 1, false, 128, 6, true>(out.data_ptr(), x, w, tiles, total, nullptr,
                                               n, atoms, 0, amaps.map, bm, stream);
}

// ``h[R,I] @ down[e]`` with L fixed K segments and the balanced rounded tree -> fp32.
void fc2_cuda(const at::Tensor &h, const at::Tensor &down, const at::Tensor &tiles,
              const at::Tensor &total, at::Tensor out, int64_t leaves) {
  const c10::cuda::CUDAGuard guard(h.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t n = out.size(1), k = h.size(1), seg = k / leaves;
  TORCH_CHECK(seg % kAtomK == 0, "sm100 MoE FC2 needs every K segment divisible by 64");
  TORCH_CHECK(n % 4 == 0, "sm100 MoE FC2 needs a hidden size divisible by 4");
  const int64_t atoms = seg / kAtomK;
  const auto amaps = encode_a(h);
  const bool fc2_pack = 1 != 0;
#define FC2_CASE(T, L, BN, ST)                                                          \
  do {                                                                                  \
    auto bm = encode_b<BN>(down);                                                       \
    if (fc2_pack)                                                                        \
      launch_one<T, L, false, BN, ST, false, 1, kEpiWarps, true>(                        \
          out.data_ptr(), h, down, tiles, total, nullptr, n, atoms, 0, amaps.map, bm, stream); \
    else                                                                                 \
      launch_one<T, L, false, BN, ST>(out.data_ptr(), h, down, tiles, total, nullptr, n, \
                                      atoms, 0, amaps.map, bm, stream);                  \
  } while (0)
#define FC2_CASE_GW(T, L, G, BN, ST, EW)                                                \
  do {                                                                                  \
    auto bm = encode_b<BN>(down);                                                       \
    if (fc2_pack)                                                                       \
      launch_one<T, L, false, BN, ST, false, G, EW, true>(out.data_ptr(), h, down, tiles, \
                                                         total, nullptr, n, atoms, 0,    \
                                                         amaps.map, bm, stream);         \
    else                                                                                 \
      launch_one<T, L, false, BN, ST, false, G, EW, false>(out.data_ptr(), h, down, tiles,\
                                                          total, nullptr, n, atoms, 0,   \
                                                          amaps.map, bm, stream);        \
  } while (0)
#define FC2_CASE_G(T, L, G, BN, ST)                                                     \
  do {                                                                                  \
    auto bm = encode_b<BN>(down);                                                       \
    if (fc2_pack)                                                                        \
      launch_one<T, L, false, BN, ST, false, G, kEpiWarps, true>(                        \
          out.data_ptr(), h, down, tiles, total, nullptr, n, atoms, 0, amaps.map, bm, stream); \
    else                                                                                 \
      launch_one<T, L, false, BN, ST, false, G>(out.data_ptr(), h, down, tiles, total,   \
                                                nullptr, n, atoms, 0, amaps.map, bm,     \
                                                stream);                                 \
  } while (0)
  // SEGMENT-GROUP GENERATIONS at L = 8: see the kernel's ``G`` comment. ``G = 2`` is the
  // default because it buys the second accumulator generation at eight leaves.
  const int groups_forced = 2;
  const bool two_groups = groups_forced != 1;
  // THE WIDE EPILOGUE RULE, and it is about how long a group's mainloop is, not about rows.
  const int warps_forced = 0;
  const bool wide_epi = warps_forced ? warps_forced == 16
                                     : (atoms == 1 && h.size(0) <= kBlockM * down.size(0));
  const int bn_forced = 0;
  // THE GLM FC2 RULE (rounds five and six).
  const bool glm_g4 = bn_forced == 0 && leaves == 8 && atoms >= 2 && groups_forced == 2 &&
                      warps_forced == 0;
  const bool bn128 = bn_forced == 128;
#define FC2_LEAVES(T)                                                                   \
  do {                                                                                  \
    if (leaves == 8 && (groups_forced == 4 || glm_g4)) {                                  \
      FC2_CASE_GW(T, 8, 4, 128, 6, 16);                                                   \
      break;                                                                              \
    }                                                                                     \
    if (leaves == 8 && bn128) {                                                           \
      FC2_CASE_GW(T, 8, 2, 128, 6, 16);                                                   \
      break;                                                                              \
    }                                                                                     \
    if (leaves == 8) {                                                                   \
      if (two_groups && wide_epi)                                                       \
        FC2_CASE_GW(T, 8, 2, 64, 6, 16);                                                     \
      else if (two_groups)                                                                 \
        FC2_CASE_G(T, 8, 2, 64, 6);                                                        \
      else                                                                                \
        FC2_CASE(T, 8, 64, 6);                                                             \
      break;                                                                              \
    }                                                                                     \
    switch (leaves) {                                                                   \
    case 1:                                                                             \
      if (try_swap<true>(out.data_ptr(), h, down, tiles, total, n, atoms, stream)) {      \
      } else if (raw_wide(n)) {                                                          \
        auto bm256 = encode_b<256>(down);                                                \
        if (fc2_pack)                                                                    \
          launch_one<T, 1, false, 256, 4, false, 1, kEpiWarps, true>(                    \
              out.data_ptr(), h, down, tiles, total, nullptr, n, atoms, 0, amaps.map, bm256, stream); \
        else                                                                             \
          launch_one<T, 1, false, 256, 4>(                                               \
              out.data_ptr(), h, down, tiles, total, nullptr, n, atoms, 0, amaps.map, bm256, stream); \
      } else if (0 != 0 && n % 256 == 0) {                                 \
        auto bm = encode_b<128>(down);                                                   \
        if (fc2_pack)                                                                    \
          launch_one<T, 1, false, 128, 4, false, 1, kEpiWarps, true, true>(              \
              out.data_ptr(), h, down, tiles, total, nullptr, n, atoms, n / 2, amaps.map, bm, stream); \
        else                                                                             \
          launch_one<T, 1, false, 128, 4, false, 1, kEpiWarps, false, true>(             \
              out.data_ptr(), h, down, tiles, total, nullptr, n, atoms, n / 2, amaps.map, bm, stream); \
      } else {                                                                           \
        FC2_CASE(T, 1, 128, 6);                                                          \
      }                                                                                  \
      break;                                                                            \
    case 2:                                                                             \
      FC2_CASE(T, 2, 128, 6);                                                           \
      break;                                                                            \
    case 4:                                                                             \
      FC2_CASE(T, 4, 128, 6);                                                           \
      break;                                                                            \
    default:                                                                            \
      FC2_CASE(T, 8, 64, 6);                                                            \
      break;                                                                            \
    }                                                                                   \
  } while (0)
  if (h.scalar_type() == at::kBFloat16)
    FC2_LEAVES(__nv_bfloat16);
  else
    FC2_LEAVES(__half);
#undef FC2_LEAVES
#undef FC2_CASE_GW
#undef FC2_CASE_G
#undef FC2_CASE
}

} // namespace moe100
