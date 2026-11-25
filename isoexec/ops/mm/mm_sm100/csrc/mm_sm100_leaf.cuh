// SM100 tcgen05 BF16 GEMM leaf with fixed K segments.
// Each segment uses its own FP32 accumulator; the epilogue merges adjacent
// segments in the declared tree. TMA bounds suppress padded M/N accesses.

#pragma once

#include <cstdint>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/mma_sm100_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/copy_sm100_tma.hpp>
#include <cute/arch/copy_sm100.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/numeric_types.h>

#include "mm_sm100_common.cuh"
#include "mm_sm100_merge.cuh"

namespace isoexec_mm_sm100 {

using bf16_t = cutlass::bfloat16_t;
using Barrier = cutlass::arch::ClusterTransactionBarrier;

// ---------------------------------------------------------------- device utils

__device__ __forceinline__ uint32_t lane_id() {
  uint32_t v;
  asm("mov.u32 %0, %%laneid;" : "=r"(v));
  return v;
}

__device__ __forceinline__ void tcgen05_after_thread_sync() {
  asm volatile("tcgen05.fence::after_thread_sync;");
}

__device__ __forceinline__ void tcgen05_before_thread_sync() {
  asm volatile("tcgen05.fence::before_thread_sync;");
}

__device__ __forceinline__ uint32_t ld_shared_u32(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "l"(__cvta_generic_to_shared(p)));
  return v;
}

__device__ __forceinline__ void st_shared_v4(const void* p, uint32_t x, uint32_t y,
                                             uint32_t z, uint32_t w) {
  asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};"
               :: "l"(__cvta_generic_to_shared(p)), "r"(x), "r"(y), "r"(z), "r"(w));
}

// The ONE rounding point of the whole recipe: FP32 tree root -> BF16, RNE.
__device__ __forceinline__ uint32_t pack_bf16(const float& x, const float& y) {
  auto v = __float22bfloat162_rn({x, y});
  return *reinterpret_cast<uint32_t*>(&v);
}

// One tcgen05.ld of N 32-bit values per lane. Issuing every load of a subtile
// before a single tcgen05.wait::ld is the point: the wait is a full scoreboard
// drain, so one per load serialises the tensor-memory read.
template <uint32_t N> struct TmemLd;

template <> struct TmemLd<8> {
  __device__ __forceinline__ static void copy(uint32_t addr, uint32_t* v) {
    cute::SM100::TMEM::LOAD::SM100_TMEM_LOAD_32dp32b8x::copy(
        addr, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
  }
};

template <> struct TmemLd<16> {
  __device__ __forceinline__ static void copy(uint32_t addr, uint32_t* v) {
    cute::SM100::TMEM::LOAD::SM100_TMEM_LOAD_32dp32b16x::copy(
        addr, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7],
        v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15]);
  }
};

template <> struct TmemLd<32> {
  __device__ __forceinline__ static void copy(uint32_t addr, uint32_t* v) {
    cute::SM100::TMEM::LOAD::SM100_TMEM_LOAD_32dp32b32x::copy(
        addr, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7],
        v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15],
        v[16], v[17], v[18], v[19], v[20], v[21], v[22], v[23],
        v[24], v[25], v[26], v[27], v[28], v[29], v[30], v[31]);
  }
};

// The remaining levels of the DECLARED tree over G published FP32 roots, for NCOL consecutive
// output columns of one row.
template <uint32_t G, uint32_t NCOL>
__device__ __forceinline__ void fold_cols(const float* __restrict__ src, size_t group_stride,
                                          uint32_t lim, float (&out)[NCOL]) {
  // A column chunk at a time, sized so G * kChunk loads are in flight and no more: fully
  // unrolling NCOL columns over G groups asks for NCOL * G live registers, which at the wide
  // tiles put ptxas against the 255-register ceiling.
  constexpr uint32_t kChunk = G <= 2 ? 16u : (G <= 4 ? 8u : 4u);
  static_assert(NCOL % kChunk == 0 || NCOL < kChunk, "chunk must tile the subtile width");
#pragma unroll
  for (uint32_t c0 = 0; c0 < NCOL; c0 += kChunk) {
    float t[G][kChunk < NCOL ? kChunk : NCOL];
#pragma unroll
    for (uint32_t g = 0; g < G; ++g) {
#pragma unroll
      for (uint32_t i = 0; i < kChunk && c0 + i < NCOL; ++i) {
        t[g][i] = c0 + i < lim ? src[size_t(g) * group_stride + c0 + i] : 0.0f;
      }
    }
#pragma unroll
    for (uint32_t i = 0; i < kChunk && c0 + i < NCOL; ++i) {
      float col[G];
#pragma unroll
      for (uint32_t g = 0; g < G; ++g) col[g] = t[g][i];
      out[c0 + i] = merge_tree<G>(col);
    }
  }
}

//: Stride between consecutive 8-row groups of a K-major 128B-swizzled operand: eight rows of one
//: 64-element atom. This is the "8 rows" of the swizzle atom, and it is a property of the UMMA
//: OPERAND's addressing, not of what the TMA may write.
constexpr uint32_t kSboRowGroup = (8u * 64u * sizeof(bf16_t)) >> 4;

// K-major 128B-swizzled shared descriptor for one (MN x 64) tile sequence.
__device__ __forceinline__ cute::UMMA::SmemDescriptor make_desc_k_major(
    const void* base, uint32_t sbo = kSboRowGroup) {
  cute::UMMA::SmemDescriptor d;
  d.desc_ = 0;
  d.version_ = 1;
  d.lbo_mode_ = 0;
  d.layout_type_ = static_cast<uint8_t>(cute::UMMA::LayoutType::SWIZZLE_128B);
  d.start_address_ = static_cast<uint16_t>(cute::cast_smem_ptr_to_uint(base) >> 4);
  d.base_offset_ = 0;
  d.stride_byte_offset_ = static_cast<uint16_t>(sbo);
  d.leading_byte_offset_ = 0;
  return d;
}

__device__ __forceinline__ uint32_t advance_lo(uint32_t base, uint32_t elems) {
  return base + ((elems * static_cast<uint32_t>(sizeof(bf16_t))) >> 4);
}

struct Umma1Sm {
  __device__ __forceinline__ static void fma(uint64_t a, uint64_t b, uint32_t c,
                                             uint32_t scale, uint64_t idesc) {
    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
                 "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
                 :: "r"(c), "l"(a), "l"(b), "r"(static_cast<uint32_t>(idesc >> 32)), "r"(scale));
  }
};

struct Umma2Sm {
  __device__ __forceinline__ static void fma(uint64_t a, uint64_t b, uint32_t c,
                                             uint32_t scale, uint64_t idesc) {
    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
                 "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
                 :: "r"(c), "l"(a), "l"(b), "r"(static_cast<uint32_t>(idesc >> 32)), "r"(scale));
  }
};

// Exact below 2^24, which every tile index here is. The hardware 32-bit divide is
// ~25 instructions and the scheduler needs several before the first operand fetch.
__device__ __forceinline__ uint32_t fast_div(uint32_t a, uint32_t b) {
  uint32_t q = __float2uint_rz(__uint2float_rn(a) * __frcp_rn(__uint2float_rn(b)));
  uint32_t prod = q * b;
  if (prod > a) { --q; prod -= b; }
  if (a - prod >= b) ++q;
  return q;
}

// ------------------------------------------------------------------- config ---
//
// Cfg controls tile geometry, pipeline depth, segment ownership, and epilogue.
// PART 0 stores a whole tree; PART 1 publishes FP32 roots; PART 2 folds roots
// in segment order after the final CTA arrives.
template <int BM_, int BN_, int KATOMS_, int STAGES_, int TWO_CTA_, int GROUP_,
          int MCAST_, int SPC_, int PART_, int AROWS_ = 0,
          int EPI_ = 4, int PRO_ = 6, int FREE_ = 1, int FILL_ = 2, int MAINSW_ = 3,
          int SDIV_ = 1, int WAITSW_ = 0, int STOREST_ = 2, int TLOAD_ = 32,
          int THREADS_ = 224, int LAUNCH_ = 1, int SKIPA_ = 1, int ALIAS_ = 0>
struct Cfg {
  static constexpr uint32_t kAtomK = 64;
  static constexpr uint32_t kBlockM = BM_;
  // Fetch only live A rows; zero the rest before UMMA reads shared memory.
  static constexpr uint32_t kARows = AROWS_ ? uint32_t(AROWS_) : uint32_t(BM_);
  static constexpr uint32_t kBlockN = BN_;
  static constexpr uint32_t kKAtoms = KATOMS_;
  static constexpr uint32_t kBlockK = kAtomK * KATOMS_;
  static constexpr uint32_t kStages = STAGES_;
  static constexpr uint32_t kMulti = TWO_CTA_ ? 2u : 1u;
  static constexpr uint32_t kGroup = GROUP_;
  static constexpr uint32_t kMcast = MCAST_;
  static constexpr uint32_t kSpc = SPC_;
  //: Publishes an FP32 subtree root instead of owning the whole tree.
  static constexpr bool kPublish = PART_ != 0;
  //: Publishes and leaves the remaining levels to the separate combine kernel.
  static constexpr bool kPartial = PART_ == 1;
  //: Publishes and folds the remaining levels in kernel, last arriver wins.
  static constexpr bool kFold = PART_ == 2;
  //: 0 whole tree on chip, 1 publish + combine kernel, 2 publish + in-kernel fold.
  static constexpr uint32_t kPublishMode = uint32_t(PART_);
  //: Rounds and stores C itself -- either because it owns the whole tree, or
  //: because it may be the last arriver of one.
  static constexpr bool kStoresC = PART_ != 1;

  static constexpr uint32_t kEpiVariant = EPI_;
  static constexpr uint32_t kPrologue = PRO_;
  static constexpr uint32_t kFreeMode = FREE_;
  static constexpr uint32_t kFill = FILL_;
  static constexpr uint32_t kMainSw = MAINSW_;
  static constexpr uint32_t kSdiv = SDIV_;
  static constexpr uint32_t kWaitSw = WAITSW_;
  static constexpr uint32_t kStoreStages = STOREST_;
  static constexpr uint32_t kTLoad = TLOAD_;
  static constexpr uint32_t kLaunchAttr = LAUNCH_;
  static constexpr uint32_t kThreads = THREADS_;
  //: Whether a CTA whose whole A tile is past M may skip its A TMA entirely (section 2.8). A
  //: compile-time axis only so the saving can be measured against the same kernel without it.
  static constexpr bool kSkipPhantomA = SKIPA_ != 0;

  static constexpr uint32_t kClusterCTAs = kMcast ? 4u : kMulti;
  static constexpr uint32_t kClusterM = kMulti * (kMcast == 2 ? 2u : 1u);
  static constexpr uint32_t kClusterN = kMcast == 1 ? 2u : 1u;
  static constexpr uint32_t kPairsPerCluster = kClusterCTAs / kMulti;
  static constexpr uint16_t kClusterMask = uint16_t((1u << kClusterCTAs) - 1u);

  static constexpr uint32_t kLayoutM = 128;
  static constexpr uint32_t kWaveM = kBlockM < kLayoutM ? kBlockM : kLayoutM;
  static constexpr uint32_t kNumMWaves = kBlockM / kWaveM;
  static constexpr uint32_t kLoadBlockN = kBlockN / kMulti;
  static constexpr uint32_t kUmmaM = kWaveM * kMulti;
  static constexpr uint32_t kUmmaN = kBlockN;
  static constexpr uint32_t kUmmaK = 16;

  static constexpr bool kSplitTmemN = (kMulti == 2u && kWaveM == 64u);
  static constexpr uint32_t kTmemN = kSplitTmemN ? kBlockN / 2u : kBlockN;
  //: Tensor-memory columns between consecutive accumulators (segments, M waves, pipeline stages):
  //: kTmemN rounded up to 32 columns. Every tile this package compiled before round 5 has kTmemN a
  //: multiple of 32 already, so for them this IS kTmemN; a 144- or 176-wide tile keeps each
  //: accumulator 32-column aligned rather than rely on an alignment the ISA does not promise.
  static constexpr uint32_t kAccN = (kTmemN + 31u) / 32u * 32u;

  // ---- the epilogue's store geometry (README section 4.18) -----------------------------------
  // One epilogue warp's datapath quarter holds `kHalfN` output columns: the whole tile on the 4x1
  // atom (BM >= 128), one column half on the interleaved 2x2 atom (the BM = 64 pair). The epilogue
  // walks them in STEPS of W in {64, 32, 16} columns -- as many 64-wide steps as fit, then at most
  // one 32 and one 16 -- so any BN that is a multiple of 16 is expressible, and a step of W columns
  // is one TMA store box of W bf16 (128 / 64 / 32 bytes a row, SWIZZLE_128B / 64B / 32B). On the
  // 2x2 atom a step is two boxes, one per column half, except when the whole half is one step of at
  // most 32 columns: then both halves are one contiguous box (BN 64 -> the 64-wide box `narrow` has
  // always stored, BN 32 -> a 32-wide one).
  static constexpr uint32_t kHalfN = kTmemN;
  static constexpr uint32_t kN64 = kHalfN / 64u;
  static constexpr uint32_t kHas32 = (kHalfN % 64u) / 32u;
  static constexpr uint32_t kHas16 = (kHalfN % 32u) / 16u;
  //: Round 6 (README section 4.25): an 8-column step, for the BM-64 pair at BN % 32 == 16 (column
  //: halves 56 / 72 / 40 / 24: cuBLAS's own 112 / 144 / 80 / 48-wide pair tiles). A 16-byte box row,
  //: no swizzle. The UMMA N (BN) is still a multiple of 16.
  static constexpr uint32_t kHas8 = (kHalfN % 16u) / 8u;
  static constexpr uint32_t kSteps = kN64 + kHas32 + kHas16 + kHas8;
  static constexpr bool kMergedHalves = kSplitTmemN && kSteps == 1u && kHalfN <= 32u;
  static constexpr uint32_t kBoxesPerStep = (kSplitTmemN && !kMergedHalves) ? 2u : 1u;
  //: Box widths (elements) this tile stores, and so the C tensor maps it needs.
  static constexpr bool kNeedCd64 = kN64 > 0u || (kMergedHalves && kHas32);
  static constexpr bool kNeedCd32 = (!kMergedHalves && kHas32) || (kMergedHalves && kHas16);
  static constexpr bool kNeedCd16 = (!kMergedHalves && kHas16) || (kMergedHalves && kHas8);
  static constexpr bool kNeedCd8 = !kMergedHalves && kHas8;
  static constexpr uint32_t kNumCdMaps = uint32_t(kNeedCd64) + uint32_t(kNeedCd32) +
                                         uint32_t(kNeedCd16) + uint32_t(kNeedCd8);
  //: Which of the kernel's C tensor-map slots holds the box of width `w`: the widest needed box is
  //: slot 0 (`Params::tma_cd`), the next ones follow.
  static constexpr uint32_t cd_slot(uint32_t w) {
    return w == 64u ? 0u
         : w == 32u ? uint32_t(kNeedCd64)
         : w == 16u ? uint32_t(kNeedCd64) + uint32_t(kNeedCd32)
                    : uint32_t(kNeedCd64) + uint32_t(kNeedCd32) + uint32_t(kNeedCd16);
  }
  static constexpr uint32_t kSwizzleCD =
      kNeedCd64 ? 128u : (kNeedCd32 ? 64u : (kNeedCd16 ? 32u : 16u));
  static constexpr uint32_t kStoreM = kBlockM < kLayoutM ? kBlockM : kLayoutM;
  static constexpr uint32_t kEpiRowWarps = kSplitTmemN ? 2u : 4u;
  static constexpr uint32_t kEpiColWarps = kSplitTmemN ? 2u : 1u;

  static constexpr uint32_t kStoreRows = kStoreM;
  static constexpr uint32_t kEpiThreads = 128;
  static constexpr uint32_t kNonEpiThreads = THREADS_ - 128;
  static constexpr uint32_t kEpiWarps = kEpiThreads / 32;
  static constexpr uint32_t kEpiWarp0 = kNonEpiThreads / 32;
  static constexpr uint32_t kFreeWarp = kEpiWarp0 + ((3u - (kEpiWarp0 & 3u)) & 3u);

  // Accumulator pipeline. One accumulator per segment this CTA owns, times the
  // pipeline depth. Depth two overlaps tile i's epilogue with tile i+1's
  // mainloop; it is dropped when the segments already fill tensor memory.
  static constexpr uint32_t kAccPerStage = kSpc * kNumMWaves * kAccN;
  static constexpr uint32_t kEpiStages = (2u * kAccPerStage) <= 512u ? 2u : 1u;
  static constexpr uint32_t kAccumCols = kEpiStages * kAccPerStage;
  static constexpr uint32_t kTmemCols =
      kAccumCols <= 32 ? 32 : (kAccumCols <= 64 ? 64 : (kAccumCols <= 128 ? 128 :
      (kAccumCols <= 256 ? 256 : 512)));

  // The FP32 partial path stores straight from registers, so it needs no CD slab.
  // The in-kernel fold does store C, so it keeps the slab.
  static constexpr uint32_t kSmemCDPerStage = kBoxesPerStep * kStoreM * kSwizzleCD;
  static constexpr uint32_t kSmemCD = kStoresC ? kSmemCDPerStage * kStoreStages : 0u;
  //: A aliasing is exact for M <= 8 because only those rows can be stored.
  static constexpr bool kAAlias = ALIAS_ != 0;
  static constexpr uint32_t kSmemARows = kAAlias ? 8u : kBlockM;
  //: Stride between the operand's 8-row groups: zero aliases them onto the one resident group.
  static constexpr uint32_t kADescSbo = kAAlias ? 0u : kSboRowGroup;
  static constexpr uint32_t kSmemAPerStage = kSmemARows * kBlockK * sizeof(bf16_t);
  //: Bytes the TMA actually delivers for A per stage, which is what the transaction count must
  //: expect. The stage still reserves kSmemAPerStage so the UMMA's shared descriptor is the same.
  static constexpr uint32_t kFetchAPerStage = kARows * kBlockK * sizeof(bf16_t);
  static constexpr uint32_t kSmemBPerStage = kLoadBlockN * kBlockK * sizeof(bf16_t);
  static constexpr uint32_t kNumBarriers = 2 * kStages + 2 * kEpiStages + 1;
  static constexpr uint32_t kSmemBarriers = (kStages * 3 + kEpiStages * 2) * 8;
  static constexpr uint32_t kSmemTotal =
      kSmemCD + kStages * (kSmemAPerStage + kSmemBPerStage) + kSmemBarriers + 16;

  static constexpr bool kFillEarly = (kFill & (1 | 8 | 16)) != 0;

  using UmmaOp = cute::conditional_t<kMulti == 1, Umma1Sm, Umma2Sm>;
  using TmemAllocator = cute::conditional_t<kMulti == 1, cute::TMEM::Allocator1Sm,
                                            cute::TMEM::Allocator2Sm>;

  // Register width of one on-chip merge chunk: SPC live partials per column, so
  // the whole tree of one chunk stays in registers.
  static constexpr uint32_t kMergeChunk =
      kSpc <= 2 ? 32u : (kSpc <= 4 ? 16u : 8u);
  // A tcgen05.ld may not be wider than the chunk, or it would reach into the
  // next segment's accumulator columns.
  static constexpr uint32_t kMergeLd = kMergeChunk < kTLoad ? kMergeChunk : kTLoad;

  static_assert(kBlockM == 64 || kBlockM == 128 || kBlockM == 256, "BM must be 64, 128 or 256");
  // A TMA box SHORTER than the 8-row swizzle atom is legal, and the two 8s are different 8s.
  static_assert((kARows % 8 == 0 || kARows == 1 || kARows == 2 || kARows == 4) &&
                    kARows >= 1 && kARows <= kBlockM,
                "AROWS must be 1, 2, 4 or a multiple of 8, and at most BM");
  static_assert(!kAAlias || kARows <= 8,
                "ALIAS aliases the whole operand onto ONE 8-row group, so the fetch is <= 8 rows");
  static_assert(!kAAlias || kNumMWaves == 1, "ALIAS has no per-M-wave A offset to alias");
  static_assert(!kAAlias || kMcast == 0, "ALIAS is not derived for a multicast A box");
  static_assert(kBlockM != 64 || kMulti == 2, "BM 64 is only the cta_group::2 M=128 instruction");
  static_assert(!kSplitTmemN || kBlockN % 16 == 0,
                "the interleaved 2x2 accumulator splits BN into two 8-column-multiple halves");
  static_assert(kNumCdMaps <= 3, "at most three C store-box widths per tile");
  static_assert(!kSplitTmemN || (EPI_ == 1 || EPI_ == 4), "interleaved epilogue is EPI 1 and 4");
  // A pair's two CTAs each fetch BN / 2 rows of B, which must be whole 8-row swizzle atoms, and the
  // UMMA N must be a multiple of 16 (cta_group::1 and ::2 alike): BN % 16 covers both.
  static_assert(kBlockN % 16 == 0, "invalid BN");
  static_assert(kUmmaN >= 16 && kUmmaN <= 256, "invalid UMMA N");
  static_assert(kGroup % kMulti == 0, "raster group must be a multiple of the CTA pair");
  static_assert(kTLoad == 8 || kTLoad == 16 || kTLoad == 32, "invalid TLOAD");
  static_assert(kStoreStages == 1 || kStoreStages == 2 || kStoreStages == 4, "STOREST 1, 2 or 4");
  static_assert(THREADS_ == 256 || THREADS_ == 224, "THREADS must be 256 or 224");
  static_assert(PRO_ < 5 || TWO_CTA_, "PRO 5..7 hoist a cta_group::2 allocation");
  static_assert(PRO_ <= 7, "PRO must be 0..7");
  static_assert(PRO_ != 7 || FREE_ >= 1, "PRO 7 removes what the FREE 0 warp needs");
  static_assert(FREE_ <= 3, "FREE must be 0, 1, 2 or 3");
  static_assert(FREE_ != 2 || EPI_ == 4, "FREE 2 moves the EPI 4 rendezvous");
  static_assert(PRO_ != 3 || kNumBarriers <= 32, "one-lane-per-mbarrier init needs <= 32 barriers");
  static_assert(kAccumCols <= 512, "accumulators do not fit in tensor memory");
  static_assert(kSmemTotal <= 227 * 1024, "over the Blackwell dynamic shared-memory limit");
  static_assert((kFill & (1 | 8 | 16)) == 0 ||
                    (kFill & (1 | 8 | 16)) == 1 || (kFill & (1 | 8 | 16)) == 8 ||
                    (kFill & (1 | 8 | 16)) == 16,
                "FILL bits 0, 3 and 4 place the scheduler differently");
  static_assert(!(kFill & 8) || (PRO_ >= 2 && TWO_CTA_), "FILL bit 3 needs PRO >= 2 on a pair");
  static_assert(kSpc == 1 || kSpc == 2 || kSpc == 4 || kSpc == 8, "SPC must be 1, 2, 4 or 8");
  static_assert(PART_ == 0 || PART_ == 1 || PART_ == 2, "PART is 0, 1 (combine) or 2 (fold)");
  // The fold flag lives in the word after the tensor-memory pointer slot, inside the slack the
  // barrier region already reserves.
  static_assert((3 * kStages + 2 * kEpiStages) * 8 + 16 >=
                    (2 * kStages + 2 * kEpiStages + 1) * 8 + 8,
                "no room for the fold flag beside the tensor-memory pointer slot");
  static_assert(EPI_ == 4, "this package carries the EPI 4 epilogue only");
  static_assert(kSteps >= 1 && kHalfN % 8 == 0, "BN must be a whole number of 8-column steps");
  static_assert(kSmemCDPerStage % 1024 == 0, "every C box image must stay 1 KB aligned");
};

// ---------------------------------------------------------------- parameters --

struct Params {
  const void* a;
  const void* b;
  void* c;              // BF16 [M,N], written when the CTA owns every segment
  float* partial;       // FP32 [groups, M, N], written otherwise
  uint32_t m, n, k;
  uint32_t kb_per_seg;  // mainloop K blocks per segment, the tail block included
  // EXECUTION segments may end inside a K block (README section 4.17): a segment is `seg_k /
  // 16` UMMAs, `kb_full` whole K blocks and then one TAIL block of which only the first
  // `tail_ummas` UMMAs lie inside the segment.
  uint32_t seg_k;       // segment size in K elements, a multiple of 16
  uint32_t kb_full;     // whole K blocks per segment
  uint32_t tail_ummas;  // UMMAs in the tail block, 0 when the segment is whole blocks
  // PART 1 only: publish each root RNE-rounded to BF16 instead of FP32 (the pik bf16-leaf plan's
  // rounding point). Legal only at SPC == 1, where the published root IS a segment root and no
  // internal node of the tree is rounded; the host refuses anything else.
  uint32_t publish_bf16;
  uint32_t groups;      // S / SPC, the number of CTAs cooperating on one tile
  // Per-output-tile arrival counter for the in-kernel fold (PART 2). INTEGER atomics only: it
  // counts arrivals and never carries a partial sum, so it cannot perturb the reduction. It is
  // reset to zero by the CTA that folds, so the buffer is zero before and after every launch and
  // no per-call memset is needed -- which is what keeps the fold CUDA-graph capturable.
  uint32_t* counter;
  uint32_t tile_stride; // nb values per mb row, so tile = mb * tile_stride + nb
  uint32_t slots;       // cluster slots actually launched (gridDim.x / kClusterCTAs)
  uint32_t sched_magic, span_magic, group_magic;
  CUtensorMap tma_a, tma_b, tma_cd;
};

//: A tile whose epilogue stores boxes of more than one width (section 4.18) carries one C map per
//: width; every other configuration keeps the parameter block it always had.
struct ParamsCd3 : Params {
  CUtensorMap tma_cd_x[2];
};

template <class C>
using ParamsFor = cute::conditional_t<(C::kNumCdMaps > 1), ParamsCd3, Params>;

//: C tensor map in slot `kSlot` of a parameter block (slot 0 is `tma_cd`).
template <uint32_t kSlot, class P>
__device__ __forceinline__ const CUtensorMap* cd_map_slot(const P& p) {
  if constexpr (kSlot == 0) {
    return &p.tma_cd;
  } else {
    return &p.tma_cd_x[kSlot - 1];
  }
}

// ---------------------------------------------------------------- scheduler ---
// One linear work index per (iteration, cluster), rastered in groups of kGroup
// along M so two neighbouring CTAs always own two M-adjacent tiles of the same N
// column, which is what the 2-CTA UMMA requires. The segment-group index is the
// OUTERMOST axis, so the CTAs of one wave read one contiguous K slice of B.

template <class C>
struct Scheduler {
  uint32_t nm = 0, nn = 0, per_group_tiles = 0, num_clusters = 0, off_m = 0, off_n = 0;
  uint32_t pg_magic = 0, span_magic = 0, grp_magic = 0;
  uint32_t slots = 1;
  bool span_full = false;
  int iter = -1;
  uint32_t first_mb = 0, first_nb = 0, first_seg = 0;
  bool first_ok = false;

  Scheduler() = default;

  __device__ __forceinline__ void init(const Params& p, uint32_t rank) {
    pg_magic = p.sched_magic;
    span_magic = p.span_magic;
    grp_magic = p.group_magic;
    slots = p.slots;
    // Round up, do not require divisibility: the extra tiles of a partly-filled
    // cluster are entirely past M or N and contribute nothing.
    const uint32_t tiles_m = (p.m + C::kBlockM - 1) / C::kBlockM;
    const uint32_t tiles_n = (p.n + C::kBlockN - 1) / C::kBlockN;
    nm = (tiles_m + C::kClusterM - 1) / C::kClusterM;
    nn = (tiles_n + C::kClusterN - 1) / C::kClusterN;
    span_full = (nm % C::kGroup) == 0u;
    per_group_tiles = nm * nn;
    num_clusters = per_group_tiles * p.groups;
    if constexpr (C::kMcast == 1) {
      off_m = rank & 1u;
      off_n = rank >> 1;
    } else if constexpr (C::kMcast == 2) {
      off_m = (rank & 1u) + 2u * (rank >> 1);
      off_n = 0;
    } else {
      off_m = rank;
      off_n = 0;
    }
    if constexpr (C::kFillEarly) first_ok = tile(0u, first_mb, first_nb, first_seg);
  }

  __device__ __forceinline__ uint32_t divide(uint32_t a, uint32_t b, uint32_t magic) const {
    if constexpr (C::kSdiv) {
      if (magic) return __umulhi(a, magic);
    }
    if constexpr (C::kFill & 4) return fast_div(a, b);
    return a / b;
  }

  __device__ __forceinline__ bool tile(uint32_t it, uint32_t& mb, uint32_t& nb,
                                       uint32_t& seg_group) const {
    // The grid is sized to the work: `slots` is how many cluster slots were
    // launched, never a compile-time SM count, so a shape with few tiles does not
    // pay a launch ramp, a start barrier and a tcgen05.alloc for idle slots.
    const uint32_t ci = it * slots + blockIdx.x / C::kClusterCTAs;
    if (ci >= num_clusters) return false;
    // Segment group outermost.
    const uint32_t g = divide(ci, per_group_tiles, grp_magic);
    seg_group = g;
    const uint32_t cj = ci - g * per_group_tiles;
    const uint32_t per_group = nn * C::kGroup;
    const uint32_t group = divide(cj, per_group, pg_magic);
    const uint32_t first = group * C::kGroup;
    const uint32_t in_group = cj - group * per_group;
    const uint32_t span = min(C::kGroup, nm - first);
    uint32_t col;
    if (span_full) {
      col = in_group / C::kGroup;
    } else {
      col = divide(in_group, span, span_magic);
    }
    mb = (first + (in_group - col * span)) * C::kClusterM + off_m;
    nb = col * C::kClusterN + off_n;
    return true;
  }

  __device__ __forceinline__ uint32_t total_iters() const {
    const uint32_t base = blockIdx.x / C::kClusterCTAs;
    return base >= num_clusters ? 0u : (num_clusters - base + slots - 1) / slots;
  }

  __device__ __forceinline__ bool next(uint32_t& mb, uint32_t& nb, uint32_t& seg_group) {
    const uint32_t it = static_cast<uint32_t>(++iter);
    if constexpr (C::kFillEarly) {
      if (it == 0u) { mb = first_mb; nb = first_nb; seg_group = first_seg; return first_ok; }
    }
    return tile(it, mb, nb, seg_group);
  }
};

// ------------------------------------------------------------------- kernel ---

template <class C>
__global__ void __launch_bounds__(C::kNonEpiThreads + C::kEpiThreads, 1)
gemm_kernel(const __grid_constant__ ParamsFor<C> p) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  extern __shared__ __align__(1024) uint8_t smem[];
  const CUtensorMap* const desc_a = &p.tma_a;
  const CUtensorMap* const desc_b = &p.tma_b;
  const CUtensorMap* const desc_cd = &p.tma_cd;
  // C tensor map for a store box of `w` elements a row (slot 0 is `tma_cd` itself).
  auto cd_map = [&](auto kWT) -> const CUtensorMap* {
    return cd_map_slot<C::cd_slot(decltype(kWT)::value)>(p);
  };

  const uint32_t rank = cute::block_rank_in_cluster();
  const uint32_t pair_leader = C::kMulti > 1 ? (rank & ~1u) : rank;
  const bool leader = (rank & (C::kMulti - 1u)) == 0u;
  const uint16_t pair_mask = uint16_t(((1u << C::kMulti) - 1u) << pair_leader);
  const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
  const uint32_t lane = lane_id();

  // At decode, a wholly out-of-bounds A tile can skip TMA: prologue zeros
// provide the same UMMA operand, and its output rows are never stored.
const uint32_t tiles_m_all = (p.m + C::kBlockM - 1) / C::kBlockM;
  const uint32_t nm_all = (tiles_m_all + C::kClusterM - 1) / C::kClusterM;
  const bool a_decidable =
      C::kSkipPhantomA && C::kARows < C::kBlockM && C::kMcast == 0 && nm_all == 1;
  const bool a_all_oob = a_decidable && rank * C::kBlockM >= p.m;
  // Members of this pair that fetch A at all, which is what the leader must expect bytes for.
  uint32_t a_fetchers = C::kMulti;
  if (a_decidable) {
    a_fetchers = 0;
#pragma unroll
    for (uint32_t r = 0; r < C::kMulti; ++r) a_fetchers += (r * C::kBlockM < p.m) ? 1u : 0u;
  }

  Scheduler<C> sched;
  if constexpr ((C::kFill & 1) != 0) sched.init(p, rank);

  auto smem_cd = [&](uint32_t i) {
    return reinterpret_cast<bf16_t*>(smem + i * C::kSmemCDPerStage);
  };
  auto smem_a = [&](uint32_t i) {
    return reinterpret_cast<bf16_t*>(smem + C::kSmemCD + i * C::kSmemAPerStage);
  };
  auto smem_b = [&](uint32_t i) {
    return reinterpret_cast<bf16_t*>(smem + C::kSmemCD + C::kStages * C::kSmemAPerStage +
                                     i * C::kSmemBPerStage);
  };

  auto bar_base = reinterpret_cast<Barrier*>(
      smem + C::kSmemCD + C::kStages * (C::kSmemAPerStage + C::kSmemBPerStage));
  auto full = [&](uint32_t i) { return bar_base + i; };
  auto empty = [&](uint32_t i) { return bar_base + C::kStages + i; };
  auto tmem_full = [&](uint32_t i) { return bar_base + 2 * C::kStages + i; };
  auto tmem_empty = [&](uint32_t i) {
    return bar_base + 2 * C::kStages + C::kEpiStages + i;
  };
  auto dealloc_bar = bar_base + 2 * C::kStages + 2 * C::kEpiStages;
  auto tmem_ptr_slot =
      reinterpret_cast<uint32_t*>(bar_base + 2 * C::kStages + 2 * C::kEpiStages + 1);
  // One shared word the elected thread writes the last-arriver verdict into, so all 128 epilogue
  // threads take the same branch after one named barrier.
  auto fold_flag = tmem_ptr_slot + 1;

  if (warp == 0 && cute::elect_one_sync()) {
    cute::prefetch_tma_descriptor(reinterpret_cast<const cute::TmaDescriptor*>(desc_a));
    cute::prefetch_tma_descriptor(reinterpret_cast<const cute::TmaDescriptor*>(desc_b));
    if constexpr (C::kStoresC) {
      cute::prefetch_tma_descriptor(reinterpret_cast<const cute::TmaDescriptor*>(desc_cd));
      if constexpr (C::kNumCdMaps > 1) {
        cute::prefetch_tma_descriptor(reinterpret_cast<const cute::TmaDescriptor*>(cd_map_slot<1>(p)));
      }
      if constexpr (C::kNumCdMaps > 2) {
        cute::prefetch_tma_descriptor(reinterpret_cast<const cute::TmaDescriptor*>(cd_map_slot<2>(p)));
      }
    }
  }

  if constexpr ((C::kFill & 16) != 0) {
    // The tensormap prefetch is already issued, so the parameter fetch and the
    // tile-index divisions run underneath mbarrier init and the start barrier.
    sched.init(p, rank);
  }

  if (warp == 1 && (C::kPrologue == 3 || cute::elect_one_sync())) {
    if constexpr (C::kPrologue == 3) {
      if (lane < C::kNumBarriers) {
        const uint32_t count =
            lane < C::kStages ? C::kMulti
          : lane < 2 * C::kStages ? C::kPairsPerCluster
          : lane < 2 * C::kStages + C::kEpiStages ? 1u
          : lane < 2 * C::kStages + 2 * C::kEpiStages ? C::kMulti * C::kEpiThreads
          : C::kMulti;
        (bar_base + lane)->init(count);
      }
    } else {
#pragma unroll
      for (uint32_t i = 0; i < C::kStages; ++i) {
        full(i)->init(C::kMulti);
        empty(i)->init(C::kPairsPerCluster);
      }
#pragma unroll
      for (uint32_t i = 0; i < C::kEpiStages; ++i) {
        tmem_full(i)->init(1);
        tmem_empty(i)->init(C::kMulti * C::kEpiThreads);
      }
      dealloc_bar->init(C::kMulti);
    }
    cutlass::arch::fence_barrier_init();
  } else if (warp == 2) {
    if constexpr (C::kPrologue == 0) {
      typename C::TmemAllocator().allocate(C::kTmemCols, tmem_ptr_slot);
      typename C::TmemAllocator().release_allocation_lock();
    } else if constexpr (C::kPrologue == 5) {
      typename C::TmemAllocator().allocate(C::kTmemCols, tmem_ptr_slot);
      typename C::TmemAllocator().release_allocation_lock();
    }
    // No sentinel: the peer CTA can write its TMEM slot before this CTA reaches a store.
  }

  if constexpr (C::kARows < C::kBlockM) {
    // Zero unfetched A rows before the start barrier orders UMMA reads.
    if (warp != 0 && warp != 1) {
      const uint32_t tid = threadIdx.x - 64u;
      const uint32_t nthreads = C::kNonEpiThreads + C::kEpiThreads - 64u;
      // Two loops rather than one with a runtime row count: the divisor stays a compile-time
      // constant in both, so this stays a multiply-high and not a 32-bit hardware divide on the
      // prologue's critical path.
      auto zero_rows = [&](auto kFromT, auto kRowsT) {
        constexpr uint32_t kFrom = decltype(kFromT)::value;
        constexpr uint32_t kRows = decltype(kRowsT)::value;
        // Nothing to zero when the stage reserves exactly the rows the box fetches, which is what
        // ALIAS arranges for rank 0; the `if constexpr` also keeps `u / kPerAtom` out of the
        // compiler's hands when kPerAtom would be zero.
        if constexpr (kRows == 0) return;
        constexpr uint32_t kPerAtom = kRows * C::kAtomK * sizeof(bf16_t) / 16u;
        constexpr uint32_t kUnits = C::kStages * C::kKAtoms * kPerAtom;
        for (uint32_t u = tid; u < kUnits; u += nthreads) {
          const uint32_t atom = u / kPerAtom;
          const uint32_t within = u - atom * kPerAtom;
          const uint32_t st = atom / C::kKAtoms;
          const uint32_t ka = atom - st * C::kKAtoms;
          auto* base = reinterpret_cast<uint8_t*>(smem_a(st)) +
                       (ka * C::kSmemARows + kFrom) * C::kAtomK * sizeof(bf16_t);
          st_shared_v4(base + within * 16u, 0u, 0u, 0u, 0u);
        }
      };
      zero_rows(cute::Int<C::kARows>{}, cute::Int<C::kSmemARows - C::kARows>{});
      if (a_all_oob) zero_rows(cute::Int<0>{}, cute::Int<C::kARows>{});
      // tcgen05.mma reads shared memory through the async proxy, so ordinary shared stores need
      // a proxy fence before it may observe them. The start cluster barrier below then orders
      // every zeroing thread ahead of the first UMMA.
      cutlass::arch::fence_view_async_shared();
    }
  }

  if constexpr (C::kMulti > 1) {
    if constexpr (C::kPrologue >= 2) {
      if constexpr (C::kARows < C::kBlockM) {
        // A RELEASING arrive, not the relaxed one: the relaxed form carries no memory ordering,
        // and `fence.proxy.async.shared::cta` above is a proxy fence rather than a release, so
        // without this the UMMA issuer's first tcgen05.mma is not formally ordered after the
        // phantom-row zeroing. It works in practice only because the allocation latency dwarfs
        // the zeroing, which is not a guarantee. The barrier is already there, so the release
        // costs nothing; a configuration that fetches its tile's full A height has nothing to
        // publish and keeps the relaxed arrive.
        cute::cluster_arrive();
      } else {
        cute::cluster_arrive_relaxed();
      }
      if constexpr (C::kPrologue >= 6) {
        if (warp == 2) {
          typename C::TmemAllocator().allocate(C::kTmemCols, tmem_ptr_slot);
          typename C::TmemAllocator().release_allocation_lock();
          tcgen05_before_thread_sync();
        }
      }
      if constexpr ((C::kFill & 8) != 0) sched.init(p, rank);
      cute::cluster_wait();
    } else {
      cute::cluster_sync();
    }
  } else {
    __syncthreads();
  }

  if constexpr (C::kPrologue >= 1) {
    if (C::kPrologue < 5 && warp == 2) {
      typename C::TmemAllocator().allocate(C::kTmemCols, tmem_ptr_slot);
      typename C::TmemAllocator().release_allocation_lock();
      tcgen05_before_thread_sync();
    }
    // A CTA the scheduler hands no tile must still observe the allocation, or it
    // would deallocate tensor memory that was never allocated.
    if (C::kPrologue != 7 && (warp == 1 || warp == 2 || warp == C::kFreeWarp))
      cutlass::arch::NamedBarrier::sync(96, 1);
    // The allocating warp arrived at the named barrier after tcgen05.alloc returned with the slot
    // written, so the barrier orders the issuer's read after the write (section 4.16).
    if (warp == 1) tcgen05_after_thread_sync();
  }

  if constexpr (!C::kFillEarly) sched.init(p, rank);

  uint32_t mb = 0, nb = 0, seg_group = 0;
  uint32_t stage = 0, phase = 0;
  const uint32_t kb_per_seg = p.kb_per_seg;
  const uint32_t kb_per_cta = kb_per_seg * C::kSpc;
  const uint32_t kb_full = p.kb_full;
  const uint32_t tail_ummas = p.tail_ummas;

  auto advance = [&]() {
    if constexpr (C::kMainSw & 2) {
      const uint32_t next = stage + 1u;
      stage = next == C::kStages ? 0u : next;
    } else {
      stage = (stage + 1) % C::kStages;
    }
    phase ^= (stage == 0);
  };

  if (C::kFreeMode >= 1 && warp == 2) {
    // ---- tensor-memory release ---------------------------------------------
    if constexpr (C::kFreeMode == 3) {
      // FREE 3 (README section 4.19): park in a HARDWARE named barrier with the last epilogue warp
      // until it is done, instead of polling the dealloc mbarrier for the whole kernel. A warp
      // blocked in bar.sync issues nothing; a try_wait/nanosleep loop wakes on every mbarrier
      // update in the CTA and issued 12.8 M of vwide's 131 M warp-instructions at M=16384.
      cutlass::arch::NamedBarrier::sync(64, 3);
    }
    if constexpr (C::kWaitSw & 2) {
      while (!dealloc_bar->try_wait(0)) __nanosleep(256);
    } else {
      dealloc_bar->wait(0);
    }
    tcgen05_after_thread_sync();
    // The address tcgen05.alloc actually returned, not a hard-coded column 0. It IS column 0
    // today, because 213-230 KB of dynamic shared memory admits one CTA per SM and nothing else
    // on the SM allocates tensor memory -- but that is a property of the environment and not of
    // this kernel, and a kernel that reads the base is correct either way.
    typename C::TmemAllocator().free(ld_shared_u32(tmem_ptr_slot), C::kTmemCols);
  } else if (warp == 0 && cute::elect_one_sync()) {
    // ---- TMA producer ------------------------------------------------------
    while (sched.next(mb, nb, seg_group)) {
      // Segment by segment, from each segment's OWN first K element: with whole-block segments
      // this is the contiguous range (seg_group * kSpc + s) * seg_k + jj * kBlockK always streamed,
      // and with a tail it re-fetches the tail block's columns past the segment as the next
      // segment's first block.
      uint32_t seg_base = seg_group * C::kSpc * p.seg_k;
      uint32_t jj = 0;
      for (uint32_t j = 0; j < kb_per_cta; ++j) {
        if constexpr (C::kFill & 2) {
          // The first kStages buffers of the first tile have never been written.
          if (sched.iter != 0 || j >= C::kStages) empty(stage)->wait(phase ^ 1);
        } else {
          empty(stage)->wait(phase ^ 1);
        }
        const uint32_t m_idx = mb * C::kBlockM;
        const uint32_t n_idx =
            nb * C::kBlockN + (C::kMulti > 1 ? (rank & 1u) * C::kLoadBlockN : 0);
        const uint32_t k_idx = seg_base + jj * C::kBlockK;
        if (++jj == kb_per_seg) { jj = 0; seg_base += p.seg_k; }
        auto* bar = reinterpret_cast<uint64_t*>(full(stage));
        const uint16_t mc_mask = uint16_t(0x5u << (rank & 1u));
        const bool issues_shared = (rank >> 1) == 0u;
#pragma unroll
        for (uint32_t i = 0; i < C::kKAtoms; ++i) {
          const uint64_t hint = uint64_t(cute::TMA::CacheHintSm100::EVICT_NORMAL);
          auto* pa = smem_a(stage) + i * C::kSmemARows * C::kAtomK;
          auto* pb = smem_b(stage) + i * C::kLoadBlockN * C::kAtomK;
          const uint32_t ka = k_idx + i * C::kAtomK;
          if constexpr (C::kMulti == 1) {
            if (!a_all_oob) cute::SM90_TMA_LOAD_2D::copy(desc_a, bar, hint, pa, ka, m_idx);
            cute::SM90_TMA_LOAD_2D::copy(desc_b, bar, hint, pb, ka, n_idx);
          } else if constexpr (C::kMcast == 1) {
            if (issues_shared)
              cute::SM100_TMA_2SM_LOAD_MULTICAST_2D::copy(desc_a, bar, mc_mask, hint, pa, ka, m_idx);
            cute::SM100_TMA_2SM_LOAD_2D::copy(desc_b, bar, hint, pb, ka, n_idx);
          } else if constexpr (C::kMcast == 2) {
            cute::SM100_TMA_2SM_LOAD_2D::copy(desc_a, bar, hint, pa, ka, m_idx);
            if (issues_shared)
              cute::SM100_TMA_2SM_LOAD_MULTICAST_2D::copy(desc_b, bar, mc_mask, hint, pb, ka, n_idx);
          } else {
            if (!a_all_oob) cute::SM100_TMA_2SM_LOAD_2D::copy(desc_a, bar, hint, pa, ka, m_idx);
            cute::SM100_TMA_2SM_LOAD_2D::copy(desc_b, bar, hint, pb, ka, n_idx);
          }
        }
        if (leader) {
          full(stage)->arrive_and_expect_tx(
              C::kFetchAPerStage * a_fetchers + C::kSmemBPerStage * C::kMulti);
        } else {
          full(stage)->arrive(pair_leader);
        }
        advance();
      }
    }
  } else if (warp == 1 && leader) {
    // ---- UMMA issuer -------------------------------------------------------
    const uint64_t kIdesc = cute::UMMA::make_runtime_instr_desc(
        cute::UMMA::make_instr_desc<bf16_t, bf16_t, float, C::kUmmaM, C::kUmmaN,
                                    cute::UMMA::Major::K, cute::UMMA::Major::K>());

    auto a_desc = make_desc_k_major(smem_a(0), C::kADescSbo);
    auto b_desc = make_desc_k_major(smem_b(0));
    const uint32_t a_lo0 = a_desc.lo, b_lo0 = b_desc.lo;
    const uint32_t a_lo = lane < C::kStages ? a_lo0 + lane * (C::kSmemAPerStage >> 4) : 0u;
    const uint32_t b_lo = lane < C::kStages ? b_lo0 + lane * (C::kSmemBPerStage >> 4) : 0u;
    (void)a_lo; (void)b_lo;

    // Read once, after the spin above proved the allocation complete.
    const uint32_t tmem_origin = ld_shared_u32(tmem_ptr_slot);

    while (sched.next(mb, nb, seg_group)) {
      const uint32_t accum = sched.iter % C::kEpiStages;
      const uint32_t accum_phase = (sched.iter / C::kEpiStages) & 1;
      tmem_empty(accum)->wait(accum_phase ^ 1);
      tcgen05_after_thread_sync();
      const uint32_t acc_base = tmem_origin + accum * C::kAccPerStage;

      // One chain per segment, ascending K, starting from zero. SPC is a
      // compile-time constant so the segment switch costs no arithmetic.
#pragma unroll
      for (uint32_t s = 0; s < C::kSpc; ++s) {
        for (uint32_t j = 0; j < kb_per_seg; ++j) {
          full(stage)->wait(phase);
          tcgen05_after_thread_sync();
          uint32_t a_base, b_base;
          if constexpr (C::kMainSw & 1) {
            a_base = a_lo0 + stage * (C::kSmemAPerStage >> 4);
            b_base = b_lo0 + stage * (C::kSmemBPerStage >> 4);
          } else {
            a_base = __shfl_sync(0xffffffff, a_lo, static_cast<int>(stage));
            b_base = __shfl_sync(0xffffffff, b_lo, static_cast<int>(stage));
          }
          // One K block's UMMAs, the first `n_kk` of them. A whole block is the compile-time
          // count; only a segment's TAIL block (README section 4.17) issues fewer, and the
          // predicate is uniform, so the whole-block path is the unrolled loop it always was.
          auto issue_block = [&](auto kPartialT, uint32_t n_kk) {
            constexpr bool kPartial = decltype(kPartialT)::value;
#pragma unroll
            for (uint32_t kk = 0; kk < C::kBlockK / C::kUmmaK; ++kk) {
              if (kPartial && kk >= n_kk) break;
              const uint32_t atom = kk * C::kUmmaK / C::kAtomK;
              const uint32_t in_atom = kk * C::kUmmaK % C::kAtomK;
              b_desc.lo = advance_lo(b_base, atom * C::kLoadBlockN * C::kAtomK + in_atom);
#pragma unroll
              for (uint32_t w = 0; w < C::kNumMWaves; ++w) {
                a_desc.lo = advance_lo(
                    a_base,
                    atom * C::kSmemARows * C::kAtomK + w * C::kWaveM * C::kAtomK + in_atom);
                C::UmmaOp::fma(
                    a_desc, b_desc,
                    acc_base + (s * C::kNumMWaves + w) * C::kAccN,
                    (j > 0 || kk > 0) ? 1u : 0u, kIdesc);
              }
            }
          };
          if (cute::elect_one_sync()) {
            if (j < kb_full) {
              issue_block(cute::false_type{}, 0u);
            } else {
              issue_block(cute::true_type{}, tail_ummas);
            }
          }
          if constexpr (C::kMulti == 1) {
            cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(empty(stage)));
          } else {
            cutlass::arch::umma_arrive_multicast_2x1SM(
                reinterpret_cast<uint64_t*>(empty(stage)), C::kClusterMask);
          }
          if (s == C::kSpc - 1 && j == kb_per_seg - 1) {
            if constexpr (C::kMulti == 1) {
              cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(tmem_full(accum)));
            } else {
              cutlass::arch::umma_arrive_multicast_2x1SM(
                  reinterpret_cast<uint64_t*>(tmem_full(accum)), pair_mask);
            }
          }
          advance();
        }
      }
    }

    if constexpr (C::kMulti > 1) {
      const int last = sched.iter - 1;
      if (last >= 0)
        tmem_empty(last % C::kEpiStages)->wait((last / C::kEpiStages) & 1);
    }
  } else if (warp >= C::kEpiWarp0 && warp < C::kEpiWarp0 + C::kEpiWarps) {
    // ---- epilogue ----------------------------------------------------------
    const uint32_t ew = warp & 3u;
    const uint32_t rw = C::kEpiColWarps == 1 ? ew : (ew % C::kEpiRowWarps);
    const uint32_t cw = C::kEpiColWarps == 1 ? 0u : (ew / C::kEpiRowWarps);
    uint32_t store_stage = 0;
    uint32_t stores_issued = 0;
    const uint32_t iters = sched.total_iters();
    bool signalled = false;
    // The row of the store box this thread stages: a 128-row box on the 4x1 atom, 64 on the 2x2.
    const uint32_t box_row = rw * 32u + lane;
    // Output column of this warp's quarter relative to the step: its column half on the 2x2 atom.
    const uint32_t quarter_col = C::kSplitTmemN ? cw * C::kHalfN : 0u;

    auto signal_release = [&]() {
      if (cute::elect_one_sync()) {
#pragma unroll
        for (uint32_t c = 0; c < C::kMulti; ++c) dealloc_bar->arrive(pair_leader + c);
      }
      __syncwarp();
      if constexpr (C::kFreeMode == 3) cutlass::arch::NamedBarrier::arrive(64, 3);
      signalled = true;
    };

    // Stage one STEP -- W columns of this warp's quarter -- of BF16 through shared memory and issue
    // its TMA store(s). The whole-tree plan and the in-kernel fold both end here, so the single
    // rounding point and the store schedule are literally one piece of code.
    auto store_step = [&](auto kWT, uint32_t m_idx, uint32_t n_base, const float* v,
                          bool release_now) {
      constexpr uint32_t W = decltype(kWT)::value;
      constexpr uint32_t kBoxW = C::kMergedHalves ? 2u * W : W;          // elements a box row
      constexpr uint32_t kRowBytes = kBoxW * uint32_t(sizeof(bf16_t));   // == its swizzle span
      constexpr uint32_t kUnits = W / 8u;                                 // 16-byte units a row
      constexpr uint32_t kBoxBytes = C::kStoreM * kRowBytes;
      if (stores_issued >= C::kStoreStages) {
        if (ew == 0) cute::tma_store_wait<C::kStoreStages - 1>();
        cutlass::arch::NamedBarrier::sync(C::kEpiThreads, 0);
      }
      uint8_t* img = reinterpret_cast<uint8_t*>(smem_cd(store_stage)) +
                     (C::kBoxesPerStep > 1 ? cw * kBoxBytes : 0u);
      uint8_t* row = img + box_row * kRowBytes;
      // The TMA's SWIZZLE_{128,64,32}B: address bits [4, 4+b) XOR bits [7, 7+b), b = log2(units).
      const uint32_t phase = ((box_row * kRowBytes) >> 7) & (kRowBytes / 16u - 1u);
#pragma unroll
      for (uint32_t jj = 0; jj < kUnits; ++jj) {
        const uint32_t j = (C::kMergedHalves ? cw * kUnits : 0u) + jj;
        st_shared_v4(row + (j ^ phase) * 16u,
                     pack_bf16(v[8 * jj + 0], v[8 * jj + 1]),
                     pack_bf16(v[8 * jj + 2], v[8 * jj + 3]),
                     pack_bf16(v[8 * jj + 4], v[8 * jj + 5]),
                     pack_bf16(v[8 * jj + 6], v[8 * jj + 7]));
      }
      __syncwarp();
      cute::tma_store_fence();
      cutlass::arch::NamedBarrier::sync(C::kEpiThreads, 0);
      if (C::kFreeMode == 2 && release_now && ew == C::kEpiWarps - 1 && !signalled) {
        signal_release();
      }
      if (ew == 0 && cute::elect_one_sync()) {
#pragma unroll
        for (uint32_t bx = 0; bx < C::kBoxesPerStep; ++bx) {
          cute::SM90_TMA_STORE_2D::copy(cd_map(cute::Int<kBoxW>{}),
                                        reinterpret_cast<uint8_t*>(smem_cd(store_stage)) +
                                            bx * kBoxBytes,
                                        n_base + bx * C::kHalfN, m_idx);
        }
        cute::tma_store_arrive();
      }
      store_stage = (store_stage + 1) % C::kStoreStages;
      ++stores_issued;
    };

    // Read W columns of the SPC accumulators at `tmem_base` and evaluate the declared tree over
    // them, one chunk at a time so the live partials of the whole subtree stay in registers. With
    // one segment it is a plain read.
    auto load_step = [&](auto kWT, uint32_t tmem_base, float* out) {
      constexpr uint32_t W = decltype(kWT)::value;
      if constexpr (C::kSpc == 1) {
        constexpr uint32_t kTL = C::kTLoad < W ? C::kTLoad : W;
        uint32_t raw[W];
#pragma unroll
        for (uint32_t i = 0; i < W / kTL; ++i) {
          TmemLd<kTL>::copy(tmem_base + i * kTL, raw + i * kTL);
        }
        cutlass::arch::fence_view_async_tmem_load();
#pragma unroll
        for (uint32_t i = 0; i < W; ++i) out[i] = __uint_as_float(raw[i]);
      } else {
        constexpr uint32_t kChunk = C::kMergeChunk < W ? C::kMergeChunk : W;
        constexpr uint32_t kLd = kChunk < C::kTLoad ? kChunk : C::kTLoad;
#pragma unroll
        for (uint32_t c0 = 0; c0 < W; c0 += kChunk) {
          // Every tcgen05.ld of the chunk -- across all SPC accumulators -- is in flight before
          // the single tcgen05.wait::ld, because the wait is a full scoreboard drain and one per
          // accumulator would serialise the merge read SPC ways.
          uint32_t raw[C::kSpc][kChunk];
#pragma unroll
          for (uint32_t sg = 0; sg < C::kSpc; ++sg) {
#pragma unroll
            for (uint32_t i = 0; i < kChunk / kLd; ++i) {
              TmemLd<kLd>::copy(tmem_base + sg * C::kNumMWaves * C::kAccN + c0 + i * kLd,
                                raw[sg] + i * kLd);
            }
          }
          cutlass::arch::fence_view_async_tmem_load();
          float part[C::kSpc][kChunk];
#pragma unroll
          for (uint32_t sg = 0; sg < C::kSpc; ++sg) {
#pragma unroll
            for (uint32_t i = 0; i < kChunk; ++i) part[sg][i] = __uint_as_float(raw[sg][i]);
          }
          float root[kChunk];
          merge_tree_cols<C::kSpc, kChunk>(part, root);
#pragma unroll
          for (uint32_t i = 0; i < kChunk; ++i) out[c0 + i] = root[i];
        }
      }
    };

    // FP32 (or, for the bf16 leaf, RNE-rounded) subtree root -> workspace[group][row][col]. Exact
    // transport; the remaining levels of the tree are evaluated by the combine kernel (PART 1), by
    // the caller (publish-only) or by whichever CTA arrives last at this output tile (PART 2).
    auto publish_step = [&](auto kWT, uint32_t m_idx, uint32_t col, const float* v) {
      constexpr uint32_t W = decltype(kWT)::value;
      const uint32_t row = m_idx + rw * 32 + lane;
      if (C::kPartial && p.publish_bf16 && row < p.m && col < p.n) {
        // The bf16 leaf (README section 4.17): the segment root, rounded once, RNE -- the same
        // conversion as the recipe's own rounding point, applied to a leaf because the caller's
        // tree (pik, leaf_dtype = bf16) declares the leaf its rounding point.
        auto* dst = reinterpret_cast<uint16_t*>(p.partial) +
                    (size_t(seg_group) * p.m + row) * size_t(p.n) + col;
        const uint32_t lim = p.n - col;
        if ((p.n & 7u) == 0u) {
#pragma unroll
          for (uint32_t i = 0; i < W; i += 8) {
            if (i + 8 <= lim) {
              *reinterpret_cast<uint4*>(dst + i) =
                  make_uint4(pack_bf16(v[i], v[i + 1]), pack_bf16(v[i + 2], v[i + 3]),
                             pack_bf16(v[i + 4], v[i + 5]), pack_bf16(v[i + 6], v[i + 7]));
            }
          }
        } else {
#pragma unroll
          for (uint32_t i = 0; i < W; i += 2) {
            if (i < lim) {
              const uint32_t pr = pack_bf16(v[i], v[i + 1]);
              dst[i] = uint16_t(pr & 0xffffu);
              if (i + 1 < lim) dst[i + 1] = uint16_t(pr >> 16);
            }
          }
        }
      } else if (row < p.m && col < p.n) {
        float* dst = p.partial + (size_t(seg_group) * p.m + row) * size_t(p.n) + col;
        const uint32_t lim = p.n - col;
        if ((p.n & 3u) == 0u) {
#pragma unroll
          for (uint32_t i = 0; i < W; i += 4) {
            if (i + 4 <= lim) {
              *reinterpret_cast<float4*>(dst + i) = make_float4(v[i], v[i + 1], v[i + 2], v[i + 3]);
            }
          }
        } else {
#pragma unroll
          for (uint32_t i = 0; i < W; ++i) {
            if (i < lim) dst[i] = v[i];
          }
        }
      }
    };

    // The steps of one M wave, widest first: kN64 of 64 columns, then at most one 32 and one 16.
    auto for_each_step = [&](auto&& fn) {
      if constexpr (C::kN64 > 0) {  // a zero-trip loop would still instantiate the 64-wide step
#pragma unroll
        for (uint32_t s = 0; s < C::kN64; ++s) {
          fn(cute::Int<64>{}, s * 64u, !C::kHas32 && !C::kHas16 && !C::kHas8 && s + 1 == C::kN64);
        }
      }
      if constexpr (C::kHas32) fn(cute::Int<32>{}, C::kN64 * 64u, !C::kHas16 && !C::kHas8);
      if constexpr (C::kHas16) fn(cute::Int<16>{}, C::kN64 * 64u + C::kHas32 * 32u, !C::kHas8);
      if constexpr (C::kHas8)
        fn(cute::Int<8>{}, C::kN64 * 64u + C::kHas32 * 32u + C::kHas16 * 16u, true);
    };

    while (sched.next(mb, nb, seg_group)) {
      const uint32_t accum = sched.iter % C::kEpiStages;
      const uint32_t accum_phase = (sched.iter / C::kEpiStages) & 1;
      if constexpr (C::kWaitSw & 1) {
        if (ew == 0) tmem_full(accum)->wait(accum_phase);
        cutlass::arch::NamedBarrier::sync(C::kEpiThreads, 2);
      } else {
        tmem_full(accum)->wait(accum_phase);
      }
      tcgen05_after_thread_sync();
      const bool last_tile = uint32_t(sched.iter) + 1 == iters;
      // A multicast cluster spans kClusterM x kClusterN tiles and overhangs the tile grid
      // whenever the grid does not divide: this CTA's own tile may be wholly past M or N.
      const bool tile_live = mb * C::kBlockM < p.m && nb * C::kBlockN < p.n;
      (void)tile_live;
      // The accumulator barrier the UMMA arrived on orders this read after the allocation, which
      // the issuer's own read was ordered after by the named barrier.
      const uint32_t acc_base = ld_shared_u32(tmem_ptr_slot) + accum * C::kAccPerStage;

#pragma unroll
      for (uint32_t w = 0; w < C::kNumMWaves; ++w) {
        const uint32_t m_idx = mb * C::kBlockM + w * C::kWaveM;
        for_each_step([&](auto kWT, uint32_t col_off, bool last_step) {
          constexpr uint32_t W = decltype(kWT)::value;
          // The 2x2 tensor-memory atom encodes the column half in the datapath, not in the
          // column index, so `cw` belongs to the OUTPUT column and never to the TMEM address.
          float v[W];
          load_step(kWT, acc_base + w * C::kAccN + col_off, v);
          // Release the accumulator as soon as every thread of both CTAs has read its share,
          // before anything is staged or stored.
          if (w == C::kNumMWaves - 1 && last_step) {
            tcgen05_before_thread_sync();
            tmem_empty(accum)->arrive(pair_leader);
          }
          const uint32_t n_base = nb * C::kBlockN + col_off;
          if constexpr (C::kPublish) {
            publish_step(kWT, m_idx, n_base + quarter_col, v);
          } else if (C::kMcast == 0 || tile_live) {
            store_step(kWT, m_idx, n_base, v, last_tile && w == C::kNumMWaves - 1 && last_step);
          }
        });
      }
      if constexpr (C::kPartial) {
        // The partial path takes no named barrier per subtile, so the release
        // rendezvous needs one of its own: it must not be taken until every
        // epilogue warp of this CTA is past its tensor-memory reads.
        if (last_tile) {
          cutlass::arch::NamedBarrier::sync(C::kEpiThreads, 0);
          if (ew == C::kEpiWarps - 1 && !signalled) signal_release();
        }
      }
      if constexpr (C::kFold) {
        // Each thread fences its published roots before the CTA barrier; the final
// integer arrival then folds visible roots in segment order.
__threadfence();
        cutlass::arch::NamedBarrier::sync(C::kEpiThreads, 0);
        const uint32_t tile_id = mb * p.tile_stride + nb;
        if (ew == 0 && cute::elect_one_sync()) {
          const uint32_t prior = atomicAdd(p.counter + tile_id, 1u);
          // FAIL CLOSED. Exactly `groups` CTAs arrive at a tile, so `prior + 1 > groups` is
          // impossible unless the word was already nonzero -- an aborted launch, or a buffer
          // shared with another stream.
          if (prior + 1u > p.groups) __trap();
          asm volatile("st.shared.u32 [%0], %1;"
                       :: "l"(__cvta_generic_to_shared(fold_flag)),
                          "r"(prior + 1u == p.groups ? 1u : 0u));
        }
        cutlass::arch::NamedBarrier::sync(C::kEpiThreads, 0);
        if (ld_shared_u32(fold_flag) != 0u) {
          __threadfence();
          const size_t group_stride = size_t(p.m) * size_t(p.n);
#pragma unroll
          for (uint32_t w = 0; w < C::kNumMWaves; ++w) {
            const uint32_t m_idx = mb * C::kBlockM + w * C::kWaveM;
            const uint32_t row = m_idx + rw * 32 + lane;
            for_each_step([&](auto kWT, uint32_t col_off, bool) {
              constexpr uint32_t W = decltype(kWT)::value;
              const uint32_t n_base = nb * C::kBlockN + col_off;
              const uint32_t col = n_base + quarter_col;
              const uint32_t lim = (row < p.m && col < p.n) ? (p.n - col) : 0u;
              const float* src = p.partial + size_t(row) * size_t(p.n) + col;
              float v[W];
              if (p.groups == 2) {
                fold_cols<2, W>(src, group_stride, lim, v);
              } else if (p.groups == 4) {
                fold_cols<4, W>(src, group_stride, lim, v);
              } else {
                fold_cols<8, W>(src, group_stride, lim, v);
              }
              store_step(kWT, m_idx, n_base, v, false);
            });
          }
          // Self-resetting: this CTA is the last arrival at this tile in this launch, so nothing
          // else will touch the word until the next launch reads it as zero. No per-call memset,
          // which is what keeps the fold CUDA-graph capturable.
          if (ew == 0 && cute::elect_one_sync()) atomicExch(p.counter + tile_id, 0u);
        }
      }
    }

    if constexpr (C::kStoresC) {
      if (ew == 0) cute::tma_store_wait<0>();
    }

    if (ew == C::kEpiWarps - 1) {
      if constexpr (C::kFreeMode >= 1) {
        if (!signalled) signal_release();
      } else {
        if constexpr (C::kMulti > 1) {
          if (cute::elect_one_sync()) {
            dealloc_bar->arrive(pair_leader);
            dealloc_bar->arrive(pair_leader + 1);
          }
          dealloc_bar->wait(0);
        }
        typename C::TmemAllocator().free(ld_shared_u32(tmem_ptr_slot), C::kTmemCols);
      }
    }
  }

  if constexpr (C::kClusterCTAs > C::kMulti) {
    // A FOUR-CTA cluster (MCAST 1 / 2) is two CTA pairs that write into each OTHER's shared memory:
    // each pair's issuer multicasts its stage-release commit (tcgen05.commit ... multicast) onto the
    // empty barriers of all four CTAs, and one pair's producer multicasts the shared operand into
    // the other pair's stages. Nothing orders the LAST of those remote arrivals before the target
    // CTA's exit, so a pair that finished first could leave while the other pair's final commit was
    // still in flight towards its shared memory -- which the SM may already have handed to the next
    // CTA. That is the launch failure and the wrong bits README section 4.13 recorded (gap 13). The
    // pair case needs none of this: its only cross-CTA traffic is ordered by the dealloc
    // rendezvous. One releasing cluster barrier, after every role has finished, is the fix.
    cute::cluster_arrive();
    cute::cluster_wait();
  }
#else
  if (threadIdx.x == 0) __trap();
  (void)p;
#endif
}

// ------------------------------------------------------------------ combine ---
// Evaluates the remaining levels of the declared tree over the FP32 subtree
// roots the leaf published, and performs the recipe's single rounding. One live
// scalar per output element, no atomics.

template <int G>
__global__ void combine_kernel(const float* __restrict__ partial,
                               __nv_bfloat16* __restrict__ out, uint32_t m, uint32_t n) {
  const size_t total = size_t(m) * (size_t(n) >> 1);
  const size_t stride = size_t(m) * size_t(n);
  for (size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += size_t(gridDim.x) * blockDim.x) {
    const size_t base = idx << 1;
    float v0[G], v1[G];
#pragma unroll
    for (int g = 0; g < G; ++g) {
      const float2 pair =
          *reinterpret_cast<const float2*>(partial + size_t(g) * stride + base);
      v0[g] = pair.x;
      v1[g] = pair.y;
    }
    // The remaining levels of the same tree, then the recipe's one rounding.
    *reinterpret_cast<uint32_t*>(out + base) =
        pack_bf16(merge_tree<G>(v0), merge_tree<G>(v1));
  }
}

}  // namespace isoexec_mm_sm100
