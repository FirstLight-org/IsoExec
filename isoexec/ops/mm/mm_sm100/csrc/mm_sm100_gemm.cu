// Host side of the SM100 deterministic BF16 GEMM: tensor maps, the balanced
// persistent grid, the configuration table and the combine launch.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cstring>

#include <map>
#include <mutex>
#include <vector>

#include "mm_sm100_leaf.cuh"

#include <cstring>
#include <mutex>
#include <unordered_map>

namespace isoexec_mm_sm100 {
namespace {

PFN_cuTensorMapEncodeTiled_v12000 encode_fn() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = nullptr;
  if (fn == nullptr) {
    void* ptr = nullptr;
    cudaDriverEntryPointQueryResult status{};
    if (cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &ptr, cudaEnableDefault,
                                &status) != cudaSuccess || ptr == nullptr) {
      return nullptr;
    }
    fn = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(ptr);
  }
  return fn;
}

// Rank-2 K-major map. The global dimensions are the TRUE M/N/K, so rows past M and columns
// past N are out of bounds: a load of them returns zero without touching memory and a store
// to them is dropped.
struct DescKey {
  uint64_t v[6];
  bool operator==(const DescKey& o) const { return std::memcmp(v, o.v, sizeof v) == 0; }
};
struct DescKeyHash {
  size_t operator()(const DescKey& k) const {
    uint64_t h = 1469598103934665603ull;
    for (uint64_t x : k.v) h = (h ^ x) * 1099511628211ull;
    return size_t(h);
  }
};

int make_desc_uncached(CUtensorMap* map, const void* base, uint64_t gmem_inner,
                       uint64_t gmem_outer, uint32_t smem_inner, uint32_t smem_outer,
                       uint64_t outer_stride_bytes, uint32_t swizzle_bytes);

int make_desc(CUtensorMap* map, const void* base, uint64_t gmem_inner,
              uint64_t gmem_outer, uint32_t smem_inner, uint32_t smem_outer,
              uint64_t outer_stride_bytes, uint32_t swizzle_bytes) {
  static std::unordered_map<DescKey, CUtensorMap, DescKeyHash> cache;
  static std::mutex mu;
  const DescKey key{{uint64_t(reinterpret_cast<uintptr_t>(base)), gmem_inner, gmem_outer,
                     (uint64_t(smem_inner) << 32) | smem_outer, outer_stride_bytes, swizzle_bytes}};
  {
    std::lock_guard<std::mutex> lock(mu);
    auto it = cache.find(key);
    if (it != cache.end()) {
      *map = it->second;
      return 0;
    }
  }
  const int rc = make_desc_uncached(map, base, gmem_inner, gmem_outer, smem_inner, smem_outer,
                                    outer_stride_bytes, swizzle_bytes);
  if (rc == 0) {
    std::lock_guard<std::mutex> lock(mu);
    if (cache.size() >= 8192) cache.clear();
    cache.emplace(key, *map);
  }
  return rc;
}

int make_desc_uncached(CUtensorMap* map, const void* base, uint64_t gmem_inner,
                       uint64_t gmem_outer, uint32_t smem_inner, uint32_t smem_outer,
                       uint64_t outer_stride_bytes, uint32_t swizzle_bytes) {
  auto fn = encode_fn();
  if (fn == nullptr) return -10;
  const uint64_t dims[2] = {gmem_inner, gmem_outer};
  const uint64_t strides[1] = {outer_stride_bytes};
  const uint32_t box[2] = {smem_inner, smem_outer};
  const uint32_t elem_strides[2] = {1, 1};
  CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_NONE;
  if (swizzle_bytes == 32) swizzle = CU_TENSOR_MAP_SWIZZLE_32B;
  else if (swizzle_bytes == 64) swizzle = CU_TENSOR_MAP_SWIZZLE_64B;
  else if (swizzle_bytes == 128) swizzle = CU_TENSOR_MAP_SWIZZLE_128B;
  CUresult rc = fn(map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(base),
                   dims, strides, box, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
                   swizzle, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                   CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return rc == CUDA_SUCCESS ? 0 : -(20 + int(rc));
}

// __umulhi(x, floor(2^32/d)+1) == x/d, verified over the whole index range the
// scheduler can produce. Zero means "use the hardware divide".
uint32_t verified_magic_uncached(uint64_t d, uint32_t limit);

// Memoized (round 8): the verification walks the whole index range, which is host time on every call.
uint32_t verified_magic(uint64_t d, uint32_t limit) {
  static std::unordered_map<uint64_t, uint32_t> memo;
  static std::mutex mu;
  const uint64_t key = (d << 32) ^ limit;
  {
    std::lock_guard<std::mutex> lock(mu);
    auto it = memo.find(key);
    if (it != memo.end()) return it->second;
  }
  const uint32_t v = verified_magic_uncached(d, limit);
  std::lock_guard<std::mutex> lock(mu);
  if (memo.size() >= 65536) memo.clear();
  memo.emplace(key, v);
  return v;
}

uint32_t verified_magic_uncached(uint64_t d, uint32_t limit) {
  if (d < 2) return 0;
  const uint64_t magic = (uint64_t(1) << 32) / d + 1;
  if (magic > 0xffffffffull) return 0;
  for (uint32_t x = 0; x <= limit; ++x) {
    if (uint32_t((uint64_t(x) * magic) >> 32) != uint32_t(x / d)) return 0;
  }
  return uint32_t(magic);
}

int sm_count() {
  static int n = 0;
  if (n == 0) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
    if (n <= 0) n = 148;
  }
  return n;
}

struct ConfigInfo {
  const char* name;
  int block_m, block_n, katoms, stages, two_cta, group, mcast, spc, part;
  int threads, smem_bytes, epi_stages, tmem_cols, a_rows;
};

using RunFn = int (*)(const void*, const void*, void*, float*, uint32_t*, uint32_t, uint32_t,
                      uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, cudaStream_t);

//: `Runner::run` flag bits.
constexpr uint32_t kRunPublishBf16 = 1u;
using InfoFn = ConfigInfo (*)();
using CounterFn = int64_t (*)(uint32_t, uint32_t);

template <class C>
struct Runner {
  static ConfigInfo info() {
    return ConfigInfo{"", int(C::kBlockM), int(C::kBlockN), int(C::kKAtoms),
                      int(C::kStages), int(C::kMulti == 2), int(C::kGroup),
                      int(C::kMcast), int(C::kSpc), int(C::kPublishMode),
                      int(C::kThreads), int(C::kSmemTotal), int(C::kEpiStages),
                      int(C::kTmemCols), int(C::kARows)};
  }

  // Words of the per-output-tile arrival counter the in-kernel fold needs at this (m, n). The
  // padded tile grid, not the true one: a CTA whose whole tile is past M still arrives.
  static int64_t counter_words(uint32_t m, uint32_t n) {
    const uint32_t tiles_m = (m + C::kBlockM - 1) / C::kBlockM;
    const uint32_t tiles_n = (n + C::kBlockN - 1) / C::kBlockN;
    const uint32_t nm = (tiles_m + C::kClusterM - 1) / C::kClusterM;
    const uint32_t nn = (tiles_n + C::kClusterN - 1) / C::kClusterN;
    return int64_t(nm) * C::kClusterM * int64_t(nn) * C::kClusterN;
  }

  static int run(const void* a, const void* b, void* c, float* ws, uint32_t* counter, uint32_t m,
                 uint32_t n, uint32_t k, uint32_t seg_k, uint32_t groups,
                 uint32_t slots_override, uint32_t flags, cudaStream_t stream) {
    // An EXECUTION segment is any multiple of the UMMA K (16): a whole number of K blocks plus at
    // most one tail block the issuer only partly multiplies (README section 4.17). The published
    // table's SEGMENT_ALIGN (256) is policy and is enforced in segments.py, not here.
    if (seg_k == 0 || seg_k % C::kUmmaK != 0) return -2;
    if (seg_k * groups * C::kSpc != k) return -3;
    if ((flags & kRunPublishBf16) && !(C::kPartial && C::kSpc == 1)) return -11;

    ParamsFor<C> p{};
    p.a = a; p.b = b; p.c = c; p.partial = ws; p.counter = counter;
    p.m = m; p.n = n; p.k = k;
    p.seg_k = seg_k;
    p.kb_full = seg_k / C::kBlockK;
    p.tail_ummas = (seg_k % C::kBlockK) / C::kUmmaK;
    p.kb_per_seg = p.kb_full + (p.tail_ummas ? 1u : 0u);
    p.publish_bf16 = (flags & kRunPublishBf16) ? 1u : 0u;
    p.groups = groups;

    // The A box is kARows rows, not kBlockM: a box wider than M pays for out-of-bounds
    // zero-fill the UMMA's real rows never use.
    if constexpr (C::kARows < C::kBlockM) { if (m > C::kARows) return -8; }
    int rc = make_desc(&p.tma_a, a, uint64_t(k), uint64_t(m), C::kAtomK, C::kARows,
                       uint64_t(k) * 2, 128);
    if (rc != 0) return rc;
    rc = make_desc(&p.tma_b, b, uint64_t(k), uint64_t(n), C::kAtomK, C::kLoadBlockN,
                   uint64_t(k) * 2, 128);
    if (rc != 0) return rc;
    if constexpr (C::kStoresC) {
      // A configuration that only publishes never touches C, and in publish-only mode the caller
      // passes no C at all.
      // One C map per store-box width the tile uses (README section 4.18), slot 0 the widest.
      // A box of W bf16 is W * 2 bytes a row, which is exactly its swizzle span.
      CUtensorMap* slots[3] = {&p.tma_cd, nullptr, nullptr};
      if constexpr (C::kNumCdMaps > 1) { slots[1] = &p.tma_cd_x[0]; slots[2] = &p.tma_cd_x[1]; }
      // Round 6: an 8-wide box (16-byte rows, no swizzle) for the BM-64 pairs at BN % 32 == 16.
      const uint32_t widths[4] = {64u, 32u, 16u, 8u};
      const bool need[4] = {C::kNeedCd64, C::kNeedCd32, C::kNeedCd16, C::kNeedCd8};
      for (int i = 0; i < 4; ++i) {
        if (!need[i]) continue;
        rc = make_desc(slots[C::cd_slot(widths[i])], c, uint64_t(n), uint64_t(m), widths[i],
                       C::kStoreRows, uint64_t(n) * 2, widths[i] * 2);
        if (rc != 0) return rc;
      }
    }

    const uint32_t tiles_m = (m + C::kBlockM - 1) / C::kBlockM;
    const uint32_t tiles_n = (n + C::kBlockN - 1) / C::kBlockN;
    const uint32_t nm = (tiles_m + C::kClusterM - 1) / C::kClusterM;
    const uint32_t nn = (tiles_n + C::kClusterN - 1) / C::kClusterN;
    const uint32_t num_clusters = nm * nn * groups;
    if (num_clusters == 0) return -4;
    p.tile_stride = nn * C::kClusterN;
    if (C::kFold && counter == nullptr) return -9;

    // Balanced persistent grid: the fewest waves the machine allows, then the
    // fewest slots that still reach it, so no slot is idle. It is a scheduling
    // choice and nothing numerical reads it.
    uint32_t max_slots = uint32_t(sm_count()) / C::kClusterCTAs;
    if constexpr (C::kClusterCTAs > 2) {
      // A four-CTA cluster must be placed inside one GPC, so fewer than sm_count / 4 of them can be
      // resident at once. A persistent grid wider than that would run its last clusters as a
      // second, serial wave. Cap the slots at what the hardware co-schedules (queried once).
      static int resident_by_dev[64] = {};  // 0 = not queried yet, else 1 + resident clusters (per device)
      int res_dev = 0;
      cudaGetDevice(&res_dev);
      int& resident_slot = resident_by_dev[unsigned(res_dev) % 64u];
      if (resident_slot == 0) {
        cudaFuncSetAttribute(reinterpret_cast<const void*>(gemm_kernel<C>),
                             cudaFuncAttributeMaxDynamicSharedMemorySize, int(C::kSmemTotal));
        cudaLaunchConfig_t q{};
        q.gridDim = dim3(C::kClusterCTAs * max_slots, 1, 1);
        q.blockDim = dim3(C::kNonEpiThreads + C::kEpiThreads, 1, 1);
        q.dynamicSmemBytes = C::kSmemTotal;
        cudaLaunchAttribute qa[1];
        qa[0].id = cudaLaunchAttributeClusterDimension;
        qa[0].val.clusterDim.x = C::kClusterCTAs;
        qa[0].val.clusterDim.y = 1;
        qa[0].val.clusterDim.z = 1;
        q.attrs = qa;
        q.numAttrs = 1;
        int n = 0;
        if (cudaOccupancyMaxActiveClusters(&n, gemm_kernel<C>, &q) != cudaSuccess) {
          (void)cudaGetLastError();
          n = 0;
        }
        resident_slot = n + 1;
      }
      const int resident = resident_slot - 1;
      if (resident > 0 && uint32_t(resident) < max_slots) max_slots = uint32_t(resident);
    }
    if (max_slots == 0) return -5;
    const uint32_t waves = (num_clusters + max_slots - 1) / max_slots;
    uint32_t slots = (num_clusters + waves - 1) / waves;
    if (slots_override) {
      // The relation battery drives this: how many cluster slots run the work is scheduling,
      // and a kernel whose bits changed with it would not be deterministic.
      slots = slots_override < num_clusters ? slots_override : num_clusters;
    }
    p.slots = slots;

    const uint32_t bound = num_clusters + slots;
    p.sched_magic = verified_magic(uint64_t(nn) * C::kGroup, bound);
    p.span_magic = (nm < C::kGroup && nm % C::kGroup != 0)
                       ? verified_magic(uint64_t(nm), nn * C::kGroup + slots)
                       : 0u;
    p.group_magic = groups > 1 ? verified_magic(uint64_t(nm) * nn, bound) : 0u;

    // The attribute is per device: one once-flag and result per device index.
    static std::once_flag attr_once[64];
    static cudaError_t attr_rc[64];
    int attr_dev = 0;
    cudaGetDevice(&attr_dev);
    attr_dev = int(unsigned(attr_dev) % 64u);
    std::call_once(attr_once[attr_dev], [&] {
      attr_rc[attr_dev] = cudaFuncSetAttribute(
          reinterpret_cast<const void*>(gemm_kernel<C>),
          cudaFuncAttributeMaxDynamicSharedMemorySize, int(C::kSmemTotal));
    });
    if (attr_rc[attr_dev] != cudaSuccess) return -30;

    cudaLaunchConfig_t config{};
    config.gridDim = dim3(slots * C::kClusterCTAs, 1, 1);
    config.blockDim = dim3(C::kNonEpiThreads + C::kEpiThreads, 1, 1);
    config.dynamicSmemBytes = C::kSmemTotal;
    config.stream = stream;
    cudaLaunchAttribute attrs[2];
    int num_attrs = 0;
    if (C::kClusterCTAs > 1) {
      attrs[num_attrs].id = cudaLaunchAttributeClusterDimension;
      attrs[num_attrs].val.clusterDim.x = C::kClusterCTAs;
      attrs[num_attrs].val.clusterDim.y = 1;
      attrs[num_attrs].val.clusterDim.z = 1;
      ++num_attrs;
    }
    if (C::kLaunchAttr & 3u) {
      attrs[num_attrs].id = cudaLaunchAttributeClusterSchedulingPolicyPreference;
      attrs[num_attrs].val.clusterSchedulingPolicyPreference =
          (C::kLaunchAttr & 1u) ? cudaClusterSchedulingPolicySpread
                                : cudaClusterSchedulingPolicyLoadBalancing;
      ++num_attrs;
    }
    config.attrs = attrs;
    config.numAttrs = num_attrs;
    cudaError_t lrc = cudaLaunchKernelEx(&config, gemm_kernel<C>, p);
    return lrc == cudaSuccess ? 0 : -(100 + int(lrc));
  }
};

// ----------------------------------------------------------- configuration ---
// Tile shapes and pipeline knobs are pure ownership and scheduling: they change
// which worker computes what, never the arithmetic. The two families are the
// two hand-selected kernel configurations: a 128x64 pair tile
// with four K atoms and four stages, which gives narrow weights enough tiles,
// and a 256x128 pair tile with one K atom and eight stages for wide ones.

// 128x64 pair tile: BM 64 (the cta_group::2 M=128 instruction), BN 64.
template <int SPC, int PART, int AROWS = 0>
using Narrow = Cfg</*BM*/64, /*BN*/64, /*KATOMS*/4, /*STAGES*/4, /*TWO_CTA*/1,
                   /*GROUP*/2, /*MCAST*/0, SPC, PART, AROWS,
                   /*EPI*/4, /*PRO*/6, /*FREE*/1, /*FILL*/2, /*MAIN*/3, /*SDIV*/1,
                   /*WAIT*/0, /*STOREST*/2, /*TLOAD*/32, /*THREADS*/224, /*LAUNCH*/1>;

// 256x128 pair tile: BM 128, BN 128.
template <int SPC, int PART, int AROWS = 0>
using Wide = Cfg</*BM*/128, /*BN*/128, /*KATOMS*/1, /*STAGES*/8, /*TWO_CTA*/1,
                 /*GROUP*/16, /*MCAST*/0, SPC, PART, AROWS,
                 /*EPI*/4, /*PRO*/6, /*FREE*/3, /*FILL*/2, /*MAIN*/3, /*SDIV*/1,
                 /*WAIT*/1, /*STOREST*/2, /*TLOAD*/16, /*THREADS*/224, /*LAUNCH*/1>;

// 256x256 pair tile, for very wide N at prefill. Six stages is the deepest the shared-memory
// map expresses at 32768 bytes per stage plus the 32768-byte store slab.
template <int SPC, int PART, int AROWS = 0>
using Vwide = Cfg</*BM*/128, /*BN*/256, /*KATOMS*/1, /*STAGES*/6, /*TWO_CTA*/1,
                  /*GROUP*/16, /*MCAST*/0, SPC, PART, AROWS,
                  /*EPI*/4, /*PRO*/6, /*FREE*/3, /*FILL*/2, /*MAIN*/3, /*SDIV*/1,
                  /*WAIT*/1, /*STOREST*/2, /*TLOAD*/16, /*THREADS*/224, /*LAUNCH*/1>;

// 128x128 SINGLE-CTA tile: cta_group::1, so one SM owns one output tile instead of two SMs
// sharing one.
template <int SPC, int PART, int AROWS = 0>
using Solo = Cfg</*BM*/128, /*BN*/128, /*KATOMS*/1, /*STAGES*/6, /*TWO_CTA*/0,
                 /*GROUP*/16, /*MCAST*/0, SPC, PART, AROWS,
                 /*EPI*/4, /*PRO*/1, /*FREE*/1, /*FILL*/2, /*MAIN*/3, /*SDIV*/1,
                 /*WAIT*/0, /*STOREST*/2, /*TLOAD*/16, /*THREADS*/224, /*LAUNCH*/0>;

// ALIASED-A families: the same tiles with stride_byte_offset = 0 on the A descriptor, so the
// operand's eight 8-row groups alias the one 8-row group the stage holds.
template <int SPC, int PART, int STAGES = 4>
using NarrowAl = Cfg</*BM*/64, /*BN*/64, /*KATOMS*/4, STAGES, /*TWO_CTA*/1,
                     /*GROUP*/2, /*MCAST*/0, SPC, PART, /*AROWS*/8,
                     /*EPI*/4, /*PRO*/6, /*FREE*/1, /*FILL*/2, /*MAIN*/3, /*SDIV*/1,
                     /*WAIT*/0, /*STOREST*/2, /*TLOAD*/32, /*THREADS*/224, /*LAUNCH*/1,
                     /*SKIPA*/1, /*ALIAS*/1>;

template <int SPC, int PART, int STAGES = 6>
using SoloAl = Cfg</*BM*/128, /*BN*/128, /*KATOMS*/1, STAGES, /*TWO_CTA*/0,
                   /*GROUP*/16, /*MCAST*/0, SPC, PART, /*AROWS*/8,
                   /*EPI*/4, /*PRO*/1, /*FREE*/1, /*FILL*/2, /*MAIN*/3, /*SDIV*/1,
                   /*WAIT*/0, /*STOREST*/2, /*TLOAD*/16, /*THREADS*/224, /*LAUNCH*/0,
                   /*SKIPA*/1, /*ALIAS*/1>;

template <int SPC, int PART, int STAGES = 6>
using VwideAl = Cfg</*BM*/128, /*BN*/256, /*KATOMS*/1, STAGES, /*TWO_CTA*/1,
                    /*GROUP*/16, /*MCAST*/0, SPC, PART, /*AROWS*/8,
                    /*EPI*/4, /*PRO*/6, /*FREE*/1, /*FILL*/2, /*MAIN*/3, /*SDIV*/1,
                    /*WAIT*/0, /*STOREST*/2, /*TLOAD*/16, /*THREADS*/224, /*LAUNCH*/1,
                    /*SKIPA*/1, /*ALIAS*/1>;

// ---------------------------------------------------------------------------------------------
// 256x64 and 128x192 tiles reduce idle cluster slots on irregular N.
// These families use the same published segment tree; tile choice changes scheduling only.
// 256x64 pair: half `wide`'s N granularity, and nine stages fit in the bytes the narrower B stage
// frees. For a weight whose ceil(N/128) lands just above a wave boundary this is the tile that
// fills the machine.
template <int SPC, int PART, int AROWS = 0>
using P64 = Cfg</*BM*/128, /*BN*/64, /*KATOMS*/1, /*STAGES*/8, /*TWO_CTA*/1,
                /*GROUP*/16, /*MCAST*/0, SPC, PART, AROWS,
                /*EPI*/4, /*PRO*/6, /*FREE*/3, /*FILL*/2, /*MAIN*/3, /*SDIV*/1,
                /*WAIT*/1, /*STOREST*/2, /*TLOAD*/16, /*THREADS*/224, /*LAUNCH*/1>;

// 256x192 pair: three 128-byte store subtiles. 192 is not a power of two, which is exactly why it
// helps -- N = 2560, 5504, 3088 and 1536 all quantise better on 192 than on 128 or 256.
template <int SPC, int PART, int STAGES = 6>
using P192 = Cfg</*BM*/128, /*BN*/192, /*KATOMS*/1, STAGES, /*TWO_CTA*/1,
                 /*GROUP*/16, /*MCAST*/0, SPC, PART, /*AROWS*/0,
                 /*EPI*/4, /*PRO*/6, /*FREE*/3, /*FILL*/2, /*MAIN*/3, /*SDIV*/1,
                 /*WAIT*/1, /*STOREST*/2, /*TLOAD*/16, /*THREADS*/224, /*LAUNCH*/1>;

// 512x128 pair: BM 256 is the first configuration in this package with kNumMWaves = 2, so the
// issuer runs two accumulators per segment and the epilogue two M waves per tile. That code path
// exists and has never been instantiated here. BM 256 measured +1 % on the foundation matrix
// rows; the RL matrix has rows where M is large and N small, where it is the only tile that
// keeps the cluster count off a wave boundary.
template <int SPC, int PART, int STAGES = 4>
using Tall = Cfg</*BM*/256, /*BN*/128, /*KATOMS*/1, STAGES, /*TWO_CTA*/1,
                 /*GROUP*/16, /*MCAST*/0, SPC, PART, /*AROWS*/0,
                 /*EPI*/4, /*PRO*/6, /*FREE*/3, /*FILL*/2, /*MAIN*/3, /*SDIV*/1,
                 /*WAIT*/1, /*STOREST*/2, /*TLOAD*/16, /*THREADS*/224, /*LAUNCH*/1>;

// 128x256 single CTA: `vwide`'s tile on one SM. The decode rule already prefers a single CTA above
// WIDE_N; this is the wide-N member of that regime, which `solo` at BN = 128 is not.
template <int SPC, int PART, int STAGES = 4>
using S256 = Cfg</*BM*/128, /*BN*/256, /*KATOMS*/1, STAGES, /*TWO_CTA*/0,
                 /*GROUP*/16, /*MCAST*/0, SPC, PART, /*AROWS*/0,
                 /*EPI*/4, /*PRO*/1, /*FREE*/1, /*FILL*/2, /*MAIN*/3, /*SDIV*/1,
                 /*WAIT*/0, /*STOREST*/2, /*TLOAD*/16, /*THREADS*/224, /*LAUNCH*/0>;

// `wide` with a 128-deep K block instead of 64, and with four stages instead of eight. Both are
// for the short-K rows the RL matrix exposed -- GLM's q_up is K = 768, so `wide`'s eight-stage
// pipeline is 512 of the 768 elements and the fill is two thirds of the loop.
template <int SPC, int PART, int KATOMS = 2, int STAGES = 4>
using WideK = Cfg</*BM*/128, /*BN*/128, KATOMS, STAGES, /*TWO_CTA*/1,
                  /*GROUP*/16, /*MCAST*/0, SPC, PART, /*AROWS*/0,
                  /*EPI*/4, /*PRO*/6, /*FREE*/3, /*FILL*/2, /*MAIN*/3, /*SDIV*/1,
                  /*WAIT*/1, /*STOREST*/2, /*TLOAD*/16, /*THREADS*/224, /*LAUNCH*/1>;

// The RASTER-ORDER axis, per N. `kGroup` is how many cluster-M tiles the scheduler walks before
// stepping N; `narrow` uses 2 and every wide family 16. On a weight with few N tiles a group of 16
// is the whole N axis, so the raster degenerates to row-major and the B reuse in L2 is whatever
// the wave happens to give. These are the same tiles at GROUP 2 and 4.
template <int SPC, int PART, int GROUP>
using WideG = Cfg</*BM*/128, /*BN*/128, /*KATOMS*/1, /*STAGES*/8, /*TWO_CTA*/1,
                  GROUP, /*MCAST*/0, SPC, PART, /*AROWS*/0,
                  /*EPI*/4, /*PRO*/6, /*FREE*/3, /*FILL*/2, /*MAIN*/3, /*SDIV*/1,
                  /*WAIT*/1, /*STOREST*/2, /*TLOAD*/16, /*THREADS*/224, /*LAUNCH*/1>;

template <int SPC, int PART, int GROUP>
using VwideG = Cfg</*BM*/128, /*BN*/256, /*KATOMS*/1, /*STAGES*/6, /*TWO_CTA*/1,
                   GROUP, /*MCAST*/0, SPC, PART, /*AROWS*/0,
                   /*EPI*/4, /*PRO*/6, /*FREE*/3, /*FILL*/2, /*MAIN*/3, /*SDIV*/1,
                   /*WAIT*/1, /*STOREST*/2, /*TLOAD*/16, /*THREADS*/224, /*LAUNCH*/1>;

// Four-CTA multicast needs a cluster release barrier before any peer exits.
// These variants remain compiled for equivalence checks.
template <int SPC, int PART, int MCAST>
using WideMc = Cfg</*BM*/128, /*BN*/128, /*KATOMS*/1, /*STAGES*/8, /*TWO_CTA*/1,
                   /*GROUP*/16, MCAST, SPC, PART, /*AROWS*/0,
                   /*EPI*/4, /*PRO*/6, /*FREE*/3, /*FILL*/2, /*MAIN*/3, /*SDIV*/1,
                   /*WAIT*/1, /*STOREST*/2, /*TLOAD*/16, /*THREADS*/224, /*LAUNCH*/1>;

// ---------------------------------------------------------------------------------------------
// Narrower N tiles improve wave utilization; all retain the published segment tree.
// Prefill uses WAIT 1 / FREE 3 to avoid persistent polling.
template <int BN, int SPC, int PART, int STAGES>
using PairN = Cfg</*BM*/128, BN, /*KATOMS*/1, STAGES, /*TWO_CTA*/1,
                  /*GROUP*/16, /*MCAST*/0, SPC, PART, /*AROWS*/0,
                  /*EPI*/4, /*PRO*/6, /*FREE*/3, /*FILL*/2, /*MAIN*/3, /*SDIV*/1,
                  /*WAIT*/1, /*STOREST*/2, /*TLOAD*/16, /*THREADS*/224, /*LAUNCH*/1>;

// `qN`: the 128xN PAIR tile (BM 64 per CTA, the interleaved 2x2 atom) -- `narrow` generalised past
// BN = 64, with the prefill pipeline (a 64-deep K block and 8 stages) rather than narrow's decode
// one. A quarter of the old `wide` tile's rows at the same N, for rows whose M is what quantises.
template <int BN, int SPC, int PART, int STAGES>
using QuadN = Cfg</*BM*/64, BN, /*KATOMS*/1, STAGES, /*TWO_CTA*/1,
                  /*GROUP*/16, /*MCAST*/0, SPC, PART, /*AROWS*/0,
                  /*EPI*/4, /*PRO*/6, /*FREE*/3, /*FILL*/2, /*MAIN*/3, /*SDIV*/1,
                  /*WAIT*/1, /*STOREST*/2, /*TLOAD*/32, /*THREADS*/224, /*LAUNCH*/1>;

// Round 6 (section 4.28): `WideMc` at the 256x256 tile, 2x2 cluster, A multicast. MEASURED 5-11 % slower
// than `vwide_s1` on both devices; compiled and gated like `wide_mc*`, never selected.
template <int SPC, int PART, int MCAST>
using VwideMc = Cfg</*BM*/128, /*BN*/256, /*KATOMS*/1, /*STAGES*/6, /*TWO_CTA*/1,
                    /*GROUP*/16, MCAST, SPC, PART, /*AROWS*/0,
                    /*EPI*/4, /*PRO*/6, /*FREE*/3, /*FILL*/2, /*MAIN*/3, /*SDIV*/1,
                    /*WAIT*/1, /*STOREST*/2, /*TLOAD*/16, /*THREADS*/224, /*LAUNCH*/1>;

struct Entry {
  const char* name;
  RunFn run;
  InfoFn info;
  CounterFn counter_words;
};

#define ENTRY(NAME, ...) \
  Entry{NAME, &Runner<__VA_ARGS__>::run, &Runner<__VA_ARGS__>::info, \
        &Runner<__VA_ARGS__>::counter_words}

const std::vector<Entry>& table() {
  static const std::vector<Entry> t = {
      // Full-M-tile configurations: prefill and training, and any M above 32.
      ENTRY("narrow_s1", Narrow<1, 0>),
      // PART 2: publish the FP32 roots and fold them IN KERNEL -- the last CTA to arrive at an
      // output tile evaluates the remaining levels of the declared tree in segment-index order,
      // rounds once and stores C. Same bits as the combine kernel, one launch instead of two.
      ENTRY("narrow_s1_fold", Narrow<1, 2>),
      ENTRY("narrow_s2_fold", Narrow<2, 2>),
      ENTRY("narrow_s4_fold", Narrow<4, 2>),
      ENTRY("narrow_s1_part", Narrow<1, 1>),
      ENTRY("narrow_s2", Narrow<2, 0>),
      ENTRY("narrow_s2_part", Narrow<2, 1>),
      ENTRY("narrow_s4", Narrow<4, 0>),
      ENTRY("narrow_s4_part", Narrow<4, 1>),
      ENTRY("narrow_s8", Narrow<8, 0>),
      ENTRY("wide_s1", Wide<1, 0>),
      ENTRY("wide_s1_fold", Wide<1, 2>),
      ENTRY("wide_s2_fold", Wide<2, 2>),
      ENTRY("wide_s1_part", Wide<1, 1>),
      ENTRY("wide_s2", Wide<2, 0>),
      ENTRY("wide_s2_part", Wide<2, 1>),
      ENTRY("wide_s4", Wide<4, 0>),
      ENTRY("vwide_s1", Vwide<1, 0>),
      ENTRY("vwide_s1_fold", Vwide<1, 2>),
      ENTRY("vwide_s1_part", Vwide<1, 1>),
      ENTRY("vwide_s2", Vwide<2, 0>),
      // Short-A configurations: the TMA fetches 32 or 8 rows of A instead of the tile's full
      // height, which is the whole difference between a decode call and a prefill one on this
      // hardware. Admissible only while M does not exceed the fetched height.
      ENTRY("narrow_a32_s1", Narrow<1, 0, 32>),
      ENTRY("narrow_a32_s1_fold", Narrow<1, 2, 32>),
      ENTRY("narrow_a32_s2_fold", Narrow<2, 2, 32>),
      ENTRY("narrow_a32_s1_part", Narrow<1, 1, 32>),
      ENTRY("narrow_a32_s2", Narrow<2, 0, 32>),
      ENTRY("narrow_a32_s2_part", Narrow<2, 1, 32>),
      ENTRY("narrow_a32_s4", Narrow<4, 0, 32>),
      ENTRY("narrow_a32_s8", Narrow<8, 0, 32>),
      // A 16-row fetch: 16 is the next step below 32 (the sub-atom boxes below are the ones that
      // show the 8-row atom never bounded the TMA box at all), and on the
      // review's fitted TMA rate it is where M <= 16 stops being TMA-bound.
      ENTRY("narrow_a16_s1", Narrow<1, 0, 16>),
      ENTRY("narrow_a16_s1_fold", Narrow<1, 2, 16>),
      ENTRY("narrow_a16_s1_part", Narrow<1, 1, 16>),
      ENTRY("narrow_a16_s2", Narrow<2, 0, 16>),
      ENTRY("wide_a16_s1", Wide<1, 0, 16>),
      ENTRY("wide_a16_s1_part", Wide<1, 1, 16>),
      ENTRY("wide_a16_s2", Wide<2, 0, 16>),
      ENTRY("vwide_a16_s1", Vwide<1, 0, 16>),
      ENTRY("vwide_a16_s1_part", Vwide<1, 1, 16>),
      ENTRY("vwide_a16_s2", Vwide<2, 0, 16>),
      ENTRY("narrow_a8_s1", Narrow<1, 0, 8>),
      ENTRY("narrow_a8_s1_fold", Narrow<1, 2, 8>),
      // The same kernel WITHOUT the phantom-A skip of section 2.8, so the saving is measured
      // against itself rather than across a rebuild.
      ENTRY("narrow_a8_s1_noskip",
            Cfg<64, 64, 4, 4, 1, 2, 0, 1, 0, 8, 4, 6, 1, 2, 3, 1, 0, 2, 32, 224, 1, 0>),
      ENTRY("vwide_a8_s1_noskip",
            Cfg<128, 256, 1, 6, 1, 16, 0, 1, 0, 8, 4, 6, 1, 2, 3, 1, 0, 2, 16, 224, 1, 0>),
      // Sub-atom A boxes. The 8-row swizzle atom bounds the UMMA OPERAND's row groups, not the
      // TMA box: CU_TENSOR_MAP_SWIZZLE_128B constrains only the 128-byte inner extent, so at
      // M <= 4 the box can be exactly as tall as M and 7/8 of the out-of-bounds zero fill that
      // section 4.4 charges in full below 16 in-bounds rows simply never happens. The stage still
      // reserves kBlockM rows, every descriptor is unchanged, and the rows the box omits are
      // zeroed by the same prologue that already zeroes [kARows, kBlockM).
      ENTRY("narrow_a1_s1", Narrow<1, 0, 1>),
      ENTRY("narrow_a1_s2", Narrow<2, 0, 1>),
      ENTRY("narrow_a2_s1", Narrow<1, 0, 2>),
      ENTRY("narrow_a4_s1", Narrow<1, 0, 4>),
      ENTRY("narrow_a8_s2_fold", Narrow<2, 2, 8>),
      ENTRY("narrow_a8_s1_part", Narrow<1, 1, 8>),
      ENTRY("narrow_a8_s2", Narrow<2, 0, 8>),
      ENTRY("narrow_a8_s4", Narrow<4, 0, 8>),
      ENTRY("wide_a32_s1", Wide<1, 0, 32>),
      ENTRY("wide_a32_s1_fold", Wide<1, 2, 32>),
      ENTRY("wide_a32_s1_part", Wide<1, 1, 32>),
      ENTRY("wide_a32_s2", Wide<2, 0, 32>),
      ENTRY("wide_a32_s2_part", Wide<2, 1, 32>),
      ENTRY("vwide_a32_s1", Vwide<1, 0, 32>),
      ENTRY("vwide_a32_s1_fold", Vwide<1, 2, 32>),
      ENTRY("vwide_a32_s1_part", Vwide<1, 1, 32>),
      ENTRY("vwide_a32_s2", Vwide<2, 0, 32>),
      ENTRY("vwide_a32_s2_part", Vwide<2, 1, 32>),
      ENTRY("vwide_a8_s1", Vwide<1, 0, 8>),
      ENTRY("vwide_a8_s1_fold", Vwide<1, 2, 8>),
      ENTRY("vwide_a8_s1_part", Vwide<1, 1, 8>),
      ENTRY("vwide_a8_s2", Vwide<2, 0, 8>),
      ENTRY("vwide_a1_s1", Vwide<1, 0, 1>),
      // The cta_group::1 family.
      ENTRY("solo_s1", Solo<1, 0>),
      ENTRY("solo_s1_part", Solo<1, 1>),
      ENTRY("solo_s1_fold", Solo<1, 2>),
      ENTRY("solo_s2", Solo<2, 0>),
      ENTRY("solo_a32_s1", Solo<1, 0, 32>),
      ENTRY("solo_a32_s2", Solo<2, 0, 32>),
      ENTRY("solo_a16_s1", Solo<1, 0, 16>),
      ENTRY("solo_a16_s2", Solo<2, 0, 16>),
      ENTRY("solo_a8_s1", Solo<1, 0, 8>),
      ENTRY("solo_a8_s1_fold", Solo<1, 2, 8>),
      ENTRY("solo_a8_s2", Solo<2, 0, 8>),
      ENTRY("solo_a1_s1", Solo<1, 0, 1>),
      ENTRY("solo_a1_s2", Solo<2, 0, 1>),
      // The aliased-A families. `_al` keeps the family's own stage count, so it isolates what
      // deleting the phantom-row zeroing is worth; `_al8` / `_al11` spend the freed shared memory
      // on pipeline depth.
      ENTRY("narrow_al_s1", NarrowAl<1, 0>),
      ENTRY("narrow_al_s2", NarrowAl<2, 0>),
      ENTRY("narrow_al8_s1", NarrowAl<1, 0, 8>),
      ENTRY("narrow_al8_s2", NarrowAl<2, 0, 8>),
      ENTRY("narrow_al10_s1", NarrowAl<1, 0, 10>),
      ENTRY("solo_al_s1", SoloAl<1, 0>),
      ENTRY("solo_al11_s1", SoloAl<1, 0, 11>),
      ENTRY("vwide_al_s1", VwideAl<1, 0>),
      // `solo` with a 128-deep K block instead of 64: the same bytes in flight (3 stages of
      // 64 KB against 6 of 32 KB) delivered as 256-byte pieces rather than 128-byte ones.
      ENTRY("solo_k2_s1", Cfg<128, 128, 2, 3, 0, 16, 0, 1, 0, 0,
                              4, 1, 1, 2, 3, 1, 0, 1, 16, 224, 0>),
      ENTRY("solo_k2_a32_s1", Cfg<128, 128, 2, 3, 0, 16, 0, 1, 0, 32,
                                  4, 1, 1, 2, 3, 1, 0, 1, 16, 224, 0>),
      // The quantisation families of README section 4.12.
      ENTRY("p64_s1", P64<1, 0>),
      ENTRY("p64_s2", P64<2, 0>),
      ENTRY("p192_s1", P192<1, 0>),
      ENTRY("p192_s2", P192<2, 0>),
      ENTRY("p192_st4_s1", P192<1, 0, 4>),
      ENTRY("tall_s1", Tall<1, 0>),
      ENTRY("tall_s2", Tall<2, 0>),
      ENTRY("s256_s1", S256<1, 0>),
      ENTRY("s256_s2", S256<2, 0>),
      ENTRY("wide_k2_s1", WideK<1, 0>),
      ENTRY("wide_k2_s2", WideK<2, 0>),
      ENTRY("wide_st4_s1", WideK<1, 0, 1, 4>),
      ENTRY("wide_st4_s2", WideK<2, 0, 1, 4>),
      // The raster-order axis.
      ENTRY("wide_g2_s1", WideG<1, 0, 2>),
      ENTRY("wide_g4_s1", WideG<1, 0, 4>),
      ENTRY("vwide_g2_s1", VwideG<1, 0, 2>),
      ENTRY("vwide_g4_s1", VwideG<1, 0, 4>),
      // PUBLISHING TWINS (section 4.17): the same tile at PART 1 and SPC == S, so `run_plan(...,
      // root_dtype=float32)` can hand back the fp32 root of the whole tree at every family the
      // plan rule selects, and pik's leaf trees (S = m_leaves in {2, 4, 8}) have a one-kernel,
      // no-combine form on the families that fit that many accumulators.
      ENTRY("p64_s1_part", P64<1, 1>),
      ENTRY("p64_s2_part", P64<2, 1>),
      ENTRY("p192_s1_part", P192<1, 1>),
      ENTRY("p192_s2_part", P192<2, 1>),
      ENTRY("vwide_s2_part", Vwide<2, 1>),
      ENTRY("wide_s4_part", Wide<4, 1>),
      ENTRY("narrow_s8_part", Narrow<8, 1>),
      ENTRY("solo_s2_part", Solo<2, 1>),
      // Tile generality (section 4.18). Stage depth: the most that fits 227 KB beside the C slab.
      ENTRY("p144_s1", PairN<144, 1, 0, 7>),
      ENTRY("p144_s2", PairN<144, 2, 0, 7>),
      ENTRY("p176_s1", PairN<176, 1, 0, 7>),
      ENTRY("p176_s2", PairN<176, 2, 0, 7>),
      ENTRY("p208_s1", PairN<208, 1, 0, 6>),
      ENTRY("p224_s1", PairN<224, 1, 0, 6>),
      ENTRY("p240_s1", PairN<240, 1, 0, 6>),
      ENTRY("p240_s2", PairN<240, 2, 0, 6>),
      ENTRY("q96_s1", QuadN<96, 1, 0, 8>),
      ENTRY("q96_s2", QuadN<96, 2, 0, 8>),
      ENTRY("q192_s1", QuadN<192, 1, 0, 8>),
      // Round 6 (section 4.26): 9 stages, the most that fit beside the slab (MEASURED 0.962-0.991 of
      // the 8-stage build on its three RL rows). q96 at 15 and q192_s1 at 9 were not adopted.
      ENTRY("q192_s2", QuadN<192, 2, 0, 9>),
      // The cluster-shape axis (section 4.16): 2x2 with A multicast, 4x1 with B multicast.
      ENTRY("wide_mc1_s1", WideMc<1, 0, 1>),
      ENTRY("wide_mc2_s1", WideMc<1, 0, 2>),
      ENTRY("vwide_mc1_s1", VwideMc<1, 0, 1>),
      // Round 7 (section 4.30, gap 25): the deep-pipeline twins of q96_s1 / q192_s1 (15 / 9 stages, the
      // most that fit), which the rule substitutes only at K >= DEEP_STAGE_MIN_K (segments.py).
      ENTRY("q96d_s1", QuadN<96, 1, 0, 15>),
      ENTRY("q192d_s1", QuadN<192, 1, 0, 9>),
      // Round 6 (section 4.25): cuBLAS's own (a1) pair tiles, at the most stages that fit 227 KB:
      // 128x112 / 128x144 / 128x80 BM-64 pairs (64 tokens per CTA, the 2x2 atom,
      // the new 8-column store step) and the 256x112 BM-128 pair.
      ENTRY("q112_s1", QuadN<112, 1, 0, 14>),
      ENTRY("q144_s1", QuadN<144, 1, 0, 11>),
      ENTRY("q80_s1", QuadN<80, 1, 0, 16>),
      ENTRY("p112_s1", PairN<112, 1, 0, 8>),
  };
  return t;
}

#undef ENTRY

int launch_combine(const float* ws, __nv_bfloat16* out, uint32_t m, uint32_t n,
                   uint32_t groups, cudaStream_t stream) {
  const size_t total = size_t(m) * (size_t(n) >> 1);
  const uint32_t threads = 256;
  uint32_t blocks = uint32_t((total + threads - 1) / threads);
  const uint32_t cap = uint32_t(sm_count()) * 32u;
  if (blocks > cap) blocks = cap;
  if (blocks == 0) blocks = 1;
  switch (groups) {
    case 2: combine_kernel<2><<<blocks, threads, 0, stream>>>(ws, out, m, n); break;
    case 4: combine_kernel<4><<<blocks, threads, 0, stream>>>(ws, out, m, n); break;
    case 8: combine_kernel<8><<<blocks, threads, 0, stream>>>(ws, out, m, n); break;
    default: return -6;
  }
  return cudaGetLastError() == cudaSuccess ? 0 : -7;
}

}  // namespace

int64_t num_configs() { return int64_t(table().size()); }

py::dict config_info(int64_t idx) {
  TORCH_CHECK(idx >= 0 && idx < int64_t(table().size()), "config index out of range");
  const auto& e = table()[size_t(idx)];
  ConfigInfo i = e.info();
  py::dict d;
  d["name"] = std::string(e.name);
  d["block_m"] = i.block_m;
  d["block_n"] = i.block_n;
  d["katoms"] = i.katoms;
  d["stages"] = i.stages;
  d["two_cta"] = i.two_cta;
  d["group"] = i.group;
  d["mcast"] = i.mcast;
  d["segments_per_cta"] = i.spc;
  d["partial"] = i.part;            // 0 whole tree, 1 publish + combine, 2 publish + fold
  d["publishes"] = int(i.part != 0);
  d["folds"] = int(i.part == 2);
  d["threads"] = i.threads;
  d["smem_bytes"] = i.smem_bytes;
  d["epi_stages"] = i.epi_stages;
  d["tmem_cols"] = i.tmem_cols;
  d["block_k"] = i.katoms * 64;
  d["a_rows"] = i.a_rows;
  d["max_m"] = i.a_rows == i.block_m ? 0 : i.a_rows;  // 0 = any M
  return d;
}

int64_t sm_count_py() { return int64_t(sm_count()); }

//: Words of the arrival counter a fold configuration needs at this (M, N). Zero for every other
//: configuration. The caller keeps one zeroed buffer of at least this size alive; the kernel
//: leaves it zeroed, so it is never memset per call.
int64_t counter_words(int64_t config_id, int64_t m, int64_t n) {
  TORCH_CHECK(config_id >= 0 && config_id < int64_t(table().size()), "bad config id");
  const auto& e = table()[size_t(config_id)];
  if (e.info().part != 2) return 0;
  return e.counter_words(uint32_t(m), uint32_t(n));
}

// a: [M,K] bf16 row-major. b: [N,K] bf16 row-major. c: [M,N] bf16 row-major.
// workspace: float32 [groups*M*N] when the config publishes partials, else empty.
//
// combine=false (PUBLISH-ONLY, PART 1 only): no combine kernel runs and C is neither read nor
// written -- the caller wants the [groups, M, N] subtree roots in `workspace` and nothing else. It
// is the only mode in which groups == 1 is legal for a publishing configuration: the workspace
// [1, M, N] then IS the fp32 root of the whole tree.
// publish_bf16=true (PART 1, SPC 1, combine=false): each published root is a SEGMENT root and is
// stored RNE-rounded to bf16 in a bf16 workspace.
void run(int64_t config_id, const at::Tensor& a, const at::Tensor& b, at::Tensor& c,
         at::Tensor& workspace, at::Tensor& counter, int64_t segment_k, int64_t groups,
         int64_t slots_override, bool combine, bool publish_bf16) {
  TORCH_CHECK(config_id >= 0 && config_id < int64_t(table().size()), "bad config id");
  const auto& e = table()[size_t(config_id)];
  ConfigInfo info = e.info();
  // C is untouched exactly when a PART 1 configuration runs without its combine kernel.
  const bool uses_c = !(info.part == 1 && !combine);
  TORCH_CHECK(a.scalar_type() == at::kBFloat16 && b.scalar_type() == at::kBFloat16 &&
                  (!uses_c || c.scalar_type() == at::kBFloat16),
              "mm_sm100 is bf16 in and bf16 out");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && (!uses_c || c.dim() == 2), "2-D operands only");
  TORCH_CHECK(a.stride(1) == 1 && b.stride(1) == 1 && (!uses_c || c.stride(1) == 1),
              "A, B and C must be K-major / N-major with unit innermost stride");
  const int64_t m = a.size(0), k = a.size(1), n = b.size(0);
  TORCH_CHECK(b.size(1) == k, "A and B must share K");
  TORCH_CHECK(!uses_c || (c.size(0) == m && c.size(1) == n), "C shape mismatch");
  TORCH_CHECK(a.stride(0) == k && b.stride(0) == k && (!uses_c || c.stride(0) == n),
              "the TMA map is built for a contiguous row pitch");
  TORCH_CHECK(segment_k > 0 && groups > 0, "segment_k and groups must be positive");
  TORCH_CHECK((reinterpret_cast<uintptr_t>(a.data_ptr()) & 15) == 0 &&
                  (reinterpret_cast<uintptr_t>(b.data_ptr()) & 15) == 0 &&
                  (!uses_c || (reinterpret_cast<uintptr_t>(c.data_ptr()) & 15) == 0),
              "TMA needs 16-byte aligned base pointers");
  TORCH_CHECK(combine || info.part == 1,
              "combine=false (publish only) needs a PART 1 configuration; ", e.name,
              " is publish mode ", info.part);
  TORCH_CHECK(!publish_bf16 || (info.part == 1 && info.spc == 1 && !combine),
              "the bf16 leaf publish needs a PART 1, SPC 1 configuration in publish-only mode: "
              "at SPC > 1 the published root is an internal tree node and may not be rounded (",
              e.name, " spc=", info.spc, ")");
  float* ws = nullptr;
  uint32_t* cnt = nullptr;
  if (info.part == 2) {
    const int64_t want = e.counter_words(uint32_t(m), uint32_t(n));
    TORCH_CHECK(counter.scalar_type() == at::kInt && counter.is_contiguous(),
                "the fold arrival counter must be contiguous int32");
    TORCH_CHECK(counter.numel() >= want, "the fold arrival counter needs ", want,
                " words and has ", counter.numel());
    cnt = reinterpret_cast<uint32_t*>(counter.data_ptr<int32_t>());
  }
  if (info.part) {
    TORCH_CHECK(workspace.scalar_type() == (publish_bf16 ? at::kBFloat16 : at::kFloat) &&
                    workspace.is_contiguous(),
                "the partial workspace must be contiguous ", publish_bf16 ? "bfloat16" : "float32");
    TORCH_CHECK(workspace.numel() >= groups * m * n, "partial workspace too small");
    TORCH_CHECK((reinterpret_cast<uintptr_t>(workspace.data_ptr()) & 15) == 0,
                "the partial workspace must be 16-byte aligned");
    if (combine) {
      TORCH_CHECK(groups == 2 || groups == 4 || groups == 8,
                  "the combine kernel is built for 2, 4 or 8 subtree roots");
      TORCH_CHECK(n % 2 == 0, "the combine kernel pairs columns for the single BF16 round");
    } else {
      TORCH_CHECK(groups <= 8, "at most 8 published subtree roots");
    }
    ws = reinterpret_cast<float*>(workspace.data_ptr());
  } else {
    TORCH_CHECK(groups == 1, "a config that owns every segment has one root");
  }
  TORCH_CHECK(info.a_rows == info.block_m || m <= info.a_rows,
              "config ", e.name, " fetches only ", info.a_rows, " of its tile's ", info.block_m,
              " A rows, so it serves M up to ", info.a_rows, " and M is ", m);

  auto stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(slots_override >= 0, "slots_override must be non-negative");
  int rc = e.run(a.data_ptr(), b.data_ptr(), uses_c ? c.data_ptr() : nullptr, ws, cnt,
                 uint32_t(m), uint32_t(n), uint32_t(k), uint32_t(segment_k), uint32_t(groups),
                 uint32_t(slots_override), publish_bf16 ? kRunPublishBf16 : 0u, stream);
  TORCH_CHECK(rc == 0, "mm_sm100 leaf launch failed with code ", rc,
              " (config ", e.name, ", m=", m, " n=", n, " k=", k,
              " segment_k=", segment_k, " groups=", groups, ")");
  if (info.part == 1 && combine) {
    // PART 2 folded in kernel and already stored C; only the separate-combine form launches here.
    rc = launch_combine(ws, reinterpret_cast<__nv_bfloat16*>(c.data_ptr()), uint32_t(m),
                        uint32_t(n), uint32_t(groups), stream);
    TORCH_CHECK(rc == 0, "mm_sm100 combine launch failed with code ", rc);
  }
}

// The declared tree, evaluated on the host over FP32 partials, so a test can
// compare the device merge against the definition without a second kernel.
at::Tensor merge_reference(const at::Tensor& partials) {
  TORCH_CHECK(partials.dim() == 3 && partials.scalar_type() == at::kFloat,
              "expected a [groups, M, N] float32 tensor");
  auto p = partials.contiguous().cpu();
  const int64_t g = p.size(0), m = p.size(1), n = p.size(2);
  auto out = at::empty({m, n}, p.options().dtype(at::kBFloat16));
  const float* src = p.data_ptr<float>();
  auto* dst = reinterpret_cast<uint16_t*>(out.data_ptr());
  std::vector<float> v(static_cast<size_t>(g), 0.0f);
  for (int64_t i = 0; i < m * n; ++i) {
    for (int64_t j = 0; j < g; ++j) v[size_t(j)] = src[j * m * n + i];
    int cnt = int(g);
    while (cnt > 1) {
      int w = 0;
      for (int t = 0; t + 1 < cnt; t += 2) v[size_t(w++)] = v[size_t(t)] + v[size_t(t + 1)];
      if (cnt & 1) v[size_t(w++)] = v[size_t(cnt - 1)];
      cnt = w;
    }
    float r = v[0];
    uint32_t bits;
    std::memcpy(&bits, &r, 4);
    // Round-to-nearest-even to BF16, the recipe's single rounding point.
    const uint32_t rounded = (bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16;
    dst[i] = uint16_t(rounded & 0xffffu);
  }
  return out;
}

}  // namespace isoexec_mm_sm100

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod) {
  mod.def("num_configs", &isoexec_mm_sm100::num_configs);
  mod.def("config_info", &isoexec_mm_sm100::config_info, py::arg("index"));
  mod.def("sm_count", &isoexec_mm_sm100::sm_count_py);
  mod.def("counter_words", &isoexec_mm_sm100::counter_words, py::arg("config_id"), py::arg("m"),
          py::arg("n"));
  mod.def("run", &isoexec_mm_sm100::run, py::arg("config_id"), py::arg("a"), py::arg("b"),
          py::arg("c"), py::arg("workspace"), py::arg("counter"), py::arg("segment_k"),
          py::arg("groups"), py::arg("slots_override") = 0, py::arg("combine") = true,
          py::arg("publish_bf16") = false);
  mod.def("merge_reference", &isoexec_mm_sm100::merge_reference, py::arg("partials"));
}
