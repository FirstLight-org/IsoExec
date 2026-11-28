// SPDX-License-Identifier: Apache-2.0
// mm_sm100/general: the layouts and shapes the segtree leaf (``../../csrc``) declines.
#include <cstdint>
#include <cstring>
#include <mutex>
#include <type_traits>
#include <unordered_map>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/copy_sm100.hpp>
#include <cute/arch/copy_sm100_tma.hpp>
#include <cute/arch/mma_sm100_desc.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/atom/mma_traits_sm100.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/numeric_types.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAFunctions.h>
#include <torch/extension.h>

namespace isoexec_mm_sm100_general {
namespace {

using Barrier = cutlass::arch::ClusterTransactionBarrier;
constexpr uint32_t kBM = 128;     // UMMA M, cta_group::1
constexpr uint32_t kAtomK = 64;   // K per stage: one 128-byte swizzle row of bf16
constexpr uint32_t kUmmaK = 16;
constexpr uint32_t kEpiWarp0 = 4;
constexpr uint32_t kEpiWarps = 8;
constexpr uint32_t kThreads = 32 * (kEpiWarp0 + kEpiWarps);
constexpr uint32_t kMaxGens = 4;
//: accumulator generations a tile width affords (TMEM is 512 columns)
template <uint32_t BN> constexpr uint32_t gens_for() { return 2u; }
constexpr int kMaxSplit = 16;

__device__ __forceinline__ uint32_t lane_id() {
  uint32_t v;
  asm("mov.u32 %0, %%laneid;" : "=r"(v));
  return v;
}
__device__ __forceinline__ uint32_t ld_shared_u32(const uint32_t *p) {
  uint32_t v;
  asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "l"(__cvta_generic_to_shared(p)));
  return v;
}

//: 128-byte-swizzled shared descriptor. K-major: 8-row groups 1024 B apart (SBO), LBO unused.
//: MN-major: 64-element MN atoms ``lbo`` bytes apart (LBO), 8-K-row groups 1024 B apart (SBO).
__device__ __forceinline__ uint64_t make_desc(const void *base, bool mn, uint32_t lbo_bytes) {
  cute::UMMA::SmemDescriptor d;
  d.desc_ = 0;
  d.version_ = 1;
  d.lbo_mode_ = 0;
  d.layout_type_ = static_cast<uint8_t>(cute::UMMA::LayoutType::SWIZZLE_128B);
  d.start_address_ = static_cast<uint16_t>(cute::cast_smem_ptr_to_uint(base) >> 4);
  d.base_offset_ = 0;
  d.stride_byte_offset_ = static_cast<uint16_t>(1024u >> 4);
  d.leading_byte_offset_ = static_cast<uint16_t>(mn ? (lbo_bytes >> 4) : 0u);
  return d.desc_;
}

__device__ __forceinline__ void umma(uint64_t a, uint64_t b, uint32_t c, uint32_t scale, uint32_t idesc_hi) {
  asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
               "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}\n" ::"r"(c),
               "l"(a), "l"(b), "r"(idesc_hi), "r"(scale));
}

__device__ __forceinline__ void umma2(uint64_t a, uint64_t b, uint32_t c, uint32_t scale, uint32_t idesc_hi) {
  asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
               "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n\t}\n" ::"r"(c),
               "l"(a), "l"(b), "r"(idesc_hi), "r"(scale));
}

__device__ __forceinline__ void tmem_ld32(uint32_t addr, uint32_t (&v)[32]) {
  cute::SM100::TMEM::LOAD::SM100_TMEM_LOAD_32dp32b32x::copy(
      addr, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11], v[12], v[13], v[14],
      v[15], v[16], v[17], v[18], v[19], v[20], v[21], v[22], v[23], v[24], v[25], v[26], v[27], v[28], v[29],
      v[30], v[31]);
}

struct Problem {
  uint32_t M, N, batch;
  uint32_t atoms;      // K / 64
  uint32_t seg_atoms;  // atoms per segment (the last one may be shorter)
  uint32_t S;          // segments
  uint32_t m_tiles, n_tiles;
  uint64_t ldc;        // output row pitch (elements); batch pitch = M * ldc
  void *out;           // bf16 [batch, M, ldc] when S == 1, else fp32 workspace [S, batch, M, N]
  uint32_t tma_store;  // 1: the epilogue leaves through 32x32 TMA stores (``cm``)
  uint32_t f32out;     // 1: the root is returned in FP32 (``bmm(..., out_dtype=float32)``), never rounded
  uint32_t a_rows;     // K-major A: rows the TMA box delivers (M <= 128: the real rows rounded to 8)
};

constexpr uint32_t kEpiBuf = 4096;  // per epilogue warp: one 32-row x 32-column staging tile

//: The kernel. ``AMN``/``BMN``: operand is MN-major. ``BN``: UMMA N (a multiple of 16; of 64
//: when B is MN-major). Persistent; items are (segment, batch, m tile, n tile), n fastest, the
//: segment slowest so one wave works one K slice of a long reduction together (L2 reuse).
template <uint32_t BN, uint32_t STAGES, bool AMN, bool BMN, uint32_t CL>
__global__ __launch_bounds__(kThreads, 1) void gemm_kernel(Problem p, const __grid_constant__ CUtensorMap am,
                                                          const __grid_constant__ CUtensorMap bm,
                                                          const __grid_constant__ CUtensorMap cm) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  // CL = 2: a CTA pair (cta_group::2) takes a 256-row item; CTA r owns rows r*128.. of A and of the
  // accumulator and loads half of the B tile; the leader issues every UMMA (M = 256).
  constexpr uint32_t LBN = BN / CL;
  constexpr uint32_t kSmemA = kBM * kAtomK * 2u;
  constexpr uint32_t kSmemB = LBN * kAtomK * 2u;
  constexpr uint32_t kPerStage = kSmemA + kSmemB;
  constexpr uint32_t kGens = gens_for<BN>();
  constexpr uint32_t kTmemCols = kGens * BN < 32u ? 32u : kGens * BN;
  static_assert(CL == 1 || CL == 2, "cluster");
  static_assert(BN % 16 == 0 && BN >= 16 && BN <= 256, "UMMA N");
  static_assert(CL == 1 || BN % 32 == 0, "2-CTA N");
  static_assert(!BMN || LBN % 64 == 0, "an MN-major B tile is whole 64-element atoms");
  static_assert(kPerStage % 1024 == 0, "stage alignment");
  static_assert((kTmemCols & (kTmemCols - 1)) == 0, "TMEM allocation is a power of two");
  using Allocator = cute::conditional_t<CL == 1, cute::TMEM::Allocator1Sm, cute::TMEM::Allocator2Sm>;

  extern __shared__ __align__(1024) uint8_t smem[];
  auto smem_a = [&](uint32_t s) { return smem + s * kPerStage; };
  auto smem_b = [&](uint32_t s) { return smem + s * kPerStage + kSmemA; };
  auto bar = reinterpret_cast<Barrier *>(smem + STAGES * kPerStage);
  auto full = [&](uint32_t s) { return bar + s; };
  auto empty = [&](uint32_t s) { return bar + STAGES + s; };
  auto acc_full = [&](uint32_t g) { return bar + 2 * STAGES + g; };
  auto acc_empty = [&](uint32_t g) { return bar + 2 * STAGES + kGens + g; };
  auto tmem_slot = reinterpret_cast<uint32_t *>(bar + 2 * STAGES + 2 * kGens);
  uint8_t *epi_smem = smem + STAGES * kPerStage + 1024;   // 8 x kEpiBuf, 1024-aligned

  const uint32_t warp = __shfl_sync(0xffffffffu, threadIdx.x / 32u, 0);
  const uint32_t lane = lane_id();
  const uint32_t rank = CL == 1 ? 0u : cute::block_rank_in_cluster();
  const bool leader = rank == 0;
  if (warp == 0 && lane == 0) {
    cute::prefetch_tma_descriptor(&am);
    cute::prefetch_tma_descriptor(&bm);
    if (p.tma_store)
      cute::prefetch_tma_descriptor(&cm);
  }
  if (warp == 1) {
    if (lane < STAGES)
      full(lane)->init(CL);
    else if (lane < 2 * STAGES)
      empty(lane - STAGES)->init(1);
    else if (lane < 2 * STAGES + kGens)
      acc_full(lane - 2 * STAGES)->init(1);
    else if (lane < 2 * STAGES + 2 * kGens)
      acc_empty(lane - 2 * STAGES - kGens)->init(CL * kEpiWarps);
    cutlass::arch::fence_barrier_init();
  }
  // TMEM base handoff (mature reference: mm_sm100_leaf.cuh, "No sentinel"): the allocating warp writes the slot through
  // tcgen05.alloc and the readers take it after a RELEASING + ACQUIRING barrier. The former form stored a 0xffffffff
  // sentinel and then crossed a RELAXED cluster arrive, which orders no memory: a reader could load the slot before the
  // sentinel store was visible and take the stale shared-memory word a previous kernel left there as the TMEM base
  // (tcgen05.ld "misaligned address" in the epilogue), and under cta_group::2 the pair's allocation can land in this
  // CTA's slot before its own sentinel store, which would then overwrite the base. Synchronization only: no numerics.
  if (warp == 2) {
    Allocator().allocate(kTmemCols, tmem_slot);
    Allocator().release_allocation_lock();
    asm volatile("tcgen05.fence::before_thread_sync;");
  }
  if constexpr (CL == 2) {
    cute::cluster_sync();
  } else {
    __syncthreads();
  }
  uint32_t tmem_base = 0;
  if (warp == 1 || warp >= kEpiWarp0) {
    asm volatile("tcgen05.fence::after_thread_sync;");
    tmem_base = ld_shared_u32(tmem_slot);
  }

  const uint32_t tiles = p.m_tiles * p.n_tiles;
  const uint32_t per_seg = p.batch * tiles;
  const uint32_t items = p.S * per_seg;
  const uint32_t unit0 = blockIdx.x / CL, units = gridDim.x / CL;
  auto decode = [&](uint32_t item, uint32_t &s, uint32_t &b, uint32_t &m0, uint32_t &n0, uint32_t &k0,
                    uint32_t &k1) {
    s = item / per_seg;
    uint32_t r = item - s * per_seg;
    b = r / tiles;
    r -= b * tiles;
    const uint32_t mt = r / p.n_tiles;
    m0 = mt * kBM * CL;
    n0 = (r - mt * p.n_tiles) * BN;
    k0 = s * p.seg_atoms;
    k1 = min(p.atoms, k0 + p.seg_atoms);
  };

  if (warp == 0) {
    if (cute::elect_one_sync()) {
      uint32_t slot = 0, phase = 0, issued = 0;
      const uint64_t hint = uint64_t(cute::TMA::CacheHintSm100::EVICT_NORMAL);
      auto load = [&](const CUtensorMap *map, uint64_t *fb, uint8_t *dst, int32_t c0, int32_t c1, int32_t c2) {
        if constexpr (CL == 1)
          cute::SM90_TMA_LOAD_3D::copy(map, fb, hint, dst, c0, c1, c2);
        else
          cute::SM100_TMA_2SM_LOAD_3D::copy(map, fb, hint, dst, c0, c1, c2);
      };
      for (uint32_t item = unit0; item < items; item += units) {
        uint32_t s, b, m0, n0, k0, k1;
        decode(item, s, b, m0, n0, k0, k1);
        const uint32_t am0 = m0 + rank * kBM, bn0 = n0 + rank * LBN;
        for (uint32_t ka = k0; ka < k1; ++ka) {
          if (issued >= STAGES)
            empty(slot)->wait(phase ^ 1u);
          auto *fb = reinterpret_cast<uint64_t *>(full(slot));
          const int32_t k = int32_t(ka * kAtomK);
          if constexpr (AMN) {
#pragma unroll
            for (uint32_t j = 0; j < kBM / 64u; ++j)
              load(&am, fb, smem_a(slot) + j * 8192u, int32_t(am0 + j * 64u), k, int32_t(b));
          } else {
            load(&am, fb, smem_a(slot), k, int32_t(am0), int32_t(b));
          }
          if constexpr (BMN) {
#pragma unroll
            for (uint32_t j = 0; j < LBN / 64u; ++j)
              load(&bm, fb, smem_b(slot) + j * 8192u, int32_t(bn0 + j * 64u), k, int32_t(b));
          } else {
            load(&bm, fb, smem_b(slot), k, int32_t(bn0), int32_t(b));
          }
          if (leader)
            full(slot)->arrive_and_expect_tx((kSmemB + (AMN ? kSmemA : p.a_rows * kAtomK * 2u)) * CL);
          else
            full(slot)->arrive(0u);
          ++issued;
          if (++slot == STAGES) {
            slot = 0;
            phase ^= 1u;
          }
        }
      }
    }
  } else if (warp == 1) {
    if (leader) {
      constexpr auto kAMajor = AMN ? cute::UMMA::Major::MN : cute::UMMA::Major::K;
      constexpr auto kBMajor = BMN ? cute::UMMA::Major::MN : cute::UMMA::Major::K;
      const uint64_t idesc = cute::UMMA::make_runtime_instr_desc(
          cute::UMMA::make_instr_desc<cutlass::bfloat16_t, cutlass::bfloat16_t, float, kBM * CL, BN, kAMajor,
                                      kBMajor>());
      const uint32_t idesc_hi = uint32_t(idesc >> 32);
      const uint64_t a0 = make_desc(smem_a(0), AMN, 8192u);
      const uint64_t b0 = make_desc(smem_b(0), BMN, 8192u);
      // per-16-K step: K-major moves 32 B along the swizzled row, MN-major moves 16 K rows = 2048 B.
      constexpr uint32_t kStepA = AMN ? (2048u >> 4) : (32u >> 4);
      constexpr uint32_t kStepB = BMN ? (2048u >> 4) : (32u >> 4);
      auto commit = [&](Barrier *b) {
        if constexpr (CL == 1)
          cutlass::arch::umma_arrive(reinterpret_cast<uint64_t *>(b));
        else
          cutlass::arch::umma_arrive_multicast_2x1SM(reinterpret_cast<uint64_t *>(b), 3);
      };
      uint32_t slot = 0, phase = 0, gen = 0;
      for (uint32_t item = unit0; item < items; item += units) {
        uint32_t s, b, m0, n0, k0, k1;
        decode(item, s, b, m0, n0, k0, k1);
        const uint32_t g = gen % kGens, use = gen / kGens;
        if (use)
          acc_empty(g)->wait((use - 1u) & 1u);
        asm volatile("tcgen05.fence::after_thread_sync;");
        const uint32_t acc = tmem_base + g * BN;
        for (uint32_t ka = k0; ka < k1; ++ka) {
          full(slot)->wait(phase);
          asm volatile("tcgen05.fence::after_thread_sync;");
          if (cute::elect_one_sync()) {
            const uint64_t ad = a0 + uint64_t((slot * kPerStage) >> 4);
            const uint64_t bd = b0 + uint64_t((slot * kPerStage) >> 4);
#pragma unroll
            for (uint32_t kk = 0; kk < kAtomK / kUmmaK; ++kk) {
              if constexpr (CL == 1)
                umma(ad + kk * kStepA, bd + kk * kStepB, acc, (ka > k0 || kk > 0) ? 1u : 0u, idesc_hi);
              else
                umma2(ad + kk * kStepA, bd + kk * kStepB, acc, (ka > k0 || kk > 0) ? 1u : 0u, idesc_hi);
            }
          }
          __syncwarp();
          commit(empty(slot));
          if (++slot == STAGES) {
            slot = 0;
            phase ^= 1u;
          }
        }
        commit(acc_full(g));
        ++gen;
      }
    }
  } else if (warp >= kEpiWarp0) {
    const uint32_t quarter = warp & 3u;
    const uint32_t sub = (warp - kEpiWarp0) >> 2;
    uint32_t gen = 0;
    for (uint32_t item = unit0; item < items; item += units) {
      uint32_t s, b, m0, n0, k0, k1;
      decode(item, s, b, m0, n0, k0, k1);
      const uint32_t g = gen % kGens;
      acc_full(g)->wait((gen / kGens) & 1u);
      asm volatile("tcgen05.fence::after_thread_sync;");
      const uint32_t rbase = m0 + rank * kBM + quarter * 32u;
      const uint32_t row = rbase + lane;
      const uint32_t tmem_row = tmem_base + ((quarter * 32u) << 16) + g * BN;
#pragma unroll 1
      for (uint32_t c = sub * 32u; c < BN; c += 64u) {
        uint32_t raw[32];
        tmem_ld32(tmem_row + c, raw);
        cutlass::arch::fence_view_async_tmem_load();
        const uint32_t col = n0 + c;
        if (p.tma_store) {
          // Stage this warp's 32 x 32 sub-tile in shared memory (lane = row, swizzled to the map's
          // pattern) and let one TMA store write it; the map clips rows past M and columns past N.
          if (lane == 0)
            cute::tma_store_wait<0>();   // the previous store has finished reading the buffer
          __syncwarp();
          uint8_t *buf = epi_smem + (warp - kEpiWarp0) * kEpiBuf;
          if (p.S == 1 && !p.f32out) {
            __align__(16) __nv_bfloat162 v[16];
#pragma unroll
            for (uint32_t i = 0; i < 16; ++i)
              v[i] = __floats2bfloat162_rn(__uint_as_float(raw[2 * i]), __uint_as_float(raw[2 * i + 1]));
#pragma unroll
            for (uint32_t j = 0; j < 4; ++j)
              *reinterpret_cast<uint4 *>(buf + lane * 64u + ((j ^ ((lane >> 1) & 3u)) << 4)) =
                  reinterpret_cast<const uint4 *>(v)[j];
          } else {
#pragma unroll
            for (uint32_t j = 0; j < 8; ++j)
              *reinterpret_cast<uint4 *>(buf + lane * 128u + ((j ^ (lane & 7u)) << 4)) =
                  make_uint4(raw[4 * j], raw[4 * j + 1], raw[4 * j + 2], raw[4 * j + 3]);
          }
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
          __syncwarp();
          if (lane == 0) {
            const int32_t z = int32_t(s * p.batch + b);
            cute::SM90_TMA_STORE_3D::copy(&cm, buf, int32_t(col), int32_t(rbase), z);
            cute::tma_store_arrive();
          }
          continue;
        }
        if (row < p.M && col < p.N) {
          if (p.S == 1 && !p.f32out) {
            __nv_bfloat16 *dst = static_cast<__nv_bfloat16 *>(p.out) + (uint64_t(b) * p.M + row) * p.ldc + col;
            __align__(16) __nv_bfloat162 v[16];
#pragma unroll
            for (uint32_t i = 0; i < 16; ++i)
              v[i] = __floats2bfloat162_rn(__uint_as_float(raw[2 * i]), __uint_as_float(raw[2 * i + 1]));
            const __nv_bfloat16 *e = reinterpret_cast<const __nv_bfloat16 *>(v);
            for (uint32_t i = 0; i < 32u && col + i < p.N; ++i)
              dst[i] = e[i];
          } else {
            float *dst = static_cast<float *>(p.out) +
                         ((uint64_t(s) * p.batch + b) * p.M + row) * uint64_t(p.N) + col;
            for (uint32_t i = 0; i < 32u && col + i < p.N; ++i)
              dst[i] = __uint_as_float(raw[i]);
          }
        }
      }
      asm volatile("tcgen05.fence::before_thread_sync;");
      __syncwarp();
      if (lane == 0) {
        if constexpr (CL == 1)
          acc_empty(g)->arrive();
        else
          acc_empty(g)->arrive(0u);
      }
      ++gen;
    }
  }
  if (warp >= kEpiWarp0 && lane == 0)
    asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
  if constexpr (CL == 2) {
    cute::cluster_arrive_relaxed();
    cute::cluster_wait();
  } else {
    __syncthreads();
  }
  if (warp == 2)
    Allocator().free(ld_shared_u32(tmem_slot), kTmemCols);
#else
  if (threadIdx.x == 0)
    __trap();
#endif
}

//: The declared merge: balanced adjacent-pair tree over the segment index, an unpaired node carried
//: verbatim, FP32 throughout, one RNE rounding of the root.
template <int S> __device__ __forceinline__ float tree(float (&v)[S]) {
#pragma unroll
  for (int width = 1; width < S; width *= 2)
#pragma unroll
    for (int i = 0; i + width < S; i += 2 * width)
      v[i] = __fadd_rn(v[i], v[i + width]);
  return v[0];
}

template <int S, class OutT>
__global__ void combine_kernel(const float *__restrict__ ws, OutT *__restrict__ out, uint64_t rows,
                               uint32_t N, uint64_t ldc) {
  const uint64_t plane = rows * N;
  for (uint64_t i = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x; i < plane; i += uint64_t(gridDim.x) * blockDim.x) {
    float v[S];
#pragma unroll
    for (int s = 0; s < S; ++s)
      v[s] = ws[s * plane + i];
    const uint64_t r = i / N, c = i - r * N;
    if constexpr (std::is_same_v<OutT, float>)
      out[r * ldc + c] = tree<S>(v);
    else
      out[r * ldc + c] = __float2bfloat16_rn(tree<S>(v));
  }
}

// ------------------------------------------------------------------------------------- host ---

PFN_cuTensorMapEncodeTiled_v12000 encode_fn() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = nullptr;
  if (!fn) {
    void *ptr = nullptr;
    cudaDriverEntryPointQueryResult status{};
    TORCH_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &ptr, cudaEnableDefault, &status) ==
                        cudaSuccess && ptr,
                "mm_sm100/general: cuTensorMapEncodeTiled unavailable");
    fn = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(ptr);
  }
  return fn;
}

//: rank-3 map: dims {inner, outer, batch}, element strides {outer, batch}, box {box_inner, rows, 1}.
//: Host cache of encoded maps: a map is a pure function of its arguments, and the caching allocator
//: hands the same activation and weight addresses back call after call. Bounded; cleared when full.
struct MapKey {
  uint64_t v[9];
  bool operator==(const MapKey &o) const { return std::memcmp(v, o.v, sizeof v) == 0; }
};
struct MapKeyHash {
  size_t operator()(const MapKey &k) const {
    uint64_t h = 1469598103934665603ull;
    for (uint64_t x : k.v) h = (h ^ x) * 1099511628211ull;
    return size_t(h);
  }
};

CUtensorMap make_map(const void *base, uint64_t inner, uint64_t outer, uint64_t batch, uint64_t outer_stride,
                     uint64_t batch_stride, uint32_t box_rows, uint32_t box_inner = 64u, bool f32 = false,
                     CUtensorMapSwizzle swz = CU_TENSOR_MAP_SWIZZLE_128B) {
  static std::unordered_map<MapKey, CUtensorMap, MapKeyHash> cache;
  static std::mutex mu;
  const MapKey key{{uint64_t(reinterpret_cast<uintptr_t>(base)), inner, outer, batch, outer_stride, batch_stride,
                    (uint64_t(box_rows) << 32) | box_inner, uint64_t(f32), uint64_t(swz)}};
  {
    std::lock_guard<std::mutex> lock(mu);
    auto it = cache.find(key);
    if (it != cache.end())
      return it->second;
  }
  CUtensorMap map{};
  const uint64_t es_bytes = f32 ? 4u : 2u;
  const uint64_t dims[3] = {inner, outer, batch};
  const uint64_t strides[2] = {outer_stride * es_bytes, batch_stride * es_bytes};
  const uint32_t box[3] = {box_inner, box_rows, 1u};
  const uint32_t es[3] = {1, 1, 1};
  CUresult rc = encode_fn()(&map, f32 ? CU_TENSOR_MAP_DATA_TYPE_FLOAT32 : CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3,
                            const_cast<void *>(base), dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, swz,
                            CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(rc == CUDA_SUCCESS, "mm_sm100/general: tensor map rejected (", int(rc), ")");
  std::lock_guard<std::mutex> lock(mu);
  if (cache.size() >= 8192)
    cache.clear();
  cache.emplace(key, map);
  return map;
}

int sm_count(int device) {
  static int counts[64] = {0};
  if (!counts[device])
    cudaDeviceGetAttribute(&counts[device], cudaDevAttrMultiProcessorCount, device);
  return counts[device];
}

template <uint32_t BN, bool AMN, bool BMN, uint32_t CL>
void launch(const Problem &p, const CUtensorMap &am, const CUtensorMap &bm, const CUtensorMap &cm, int grid,
            cudaStream_t stream) {
  constexpr uint32_t kPerStage = kBM * kAtomK * 2u + (BN / CL) * kAtomK * 2u;
  constexpr uint32_t kBudget = 196u * 1024u;
  constexpr uint32_t kStages = kBudget / kPerStage > 8u ? 8u : kBudget / kPerStage;
  // stages | barriers (in the first 1024 B after them) | 8 epilogue staging tiles | alignment slack
  constexpr uint32_t bytes = kStages * kPerStage + 1024 + kEpiWarps * kEpiBuf + 1024;
  static_assert(8 * (2 * kStages + 2 * kMaxGens) + 16 <= 1024, "barrier block");
  auto kernel = gemm_kernel<BN, kStages, AMN, BMN, CL>;
  static bool attr_set[64] = {};
  int dev = 0;
  cudaGetDevice(&dev);
  if (!attr_set[dev % 64]) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(bytes)));
    attr_set[dev % 64] = true;
  }
  if constexpr (CL == 1) {
    kernel<<<grid, kThreads, bytes, stream>>>(p, am, bm, cm);
  } else {
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(grid);
    cfg.blockDim = dim3(kThreads);
    cfg.dynamicSmemBytes = bytes;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = 2;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, p, am, bm, cm));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <bool AMN, bool BMN>
void launch_bn(int bn, int cl, const Problem &p, const CUtensorMap &am, const CUtensorMap &bm,
               const CUtensorMap &cm, int grid, cudaStream_t s) {
  if (cl == 2) {
    switch (bn) {
    case 256: launch<256, AMN, BMN, 2>(p, am, bm, cm, grid, s); return;
    case 128: launch<128, AMN, BMN, 2>(p, am, bm, cm, grid, s); return;
    default: TORCH_CHECK(false, "mm_sm100/general: 2-CTA tile width ", bn);
    }
  }
  switch (bn) {
  case 256: launch<256, AMN, BMN, 1>(p, am, bm, cm, grid, s); break;
  case 128: launch<128, AMN, BMN, 1>(p, am, bm, cm, grid, s); break;
  case 64: launch<64, AMN, BMN, 1>(p, am, bm, cm, grid, s); break;
  case 32:
    if constexpr (!BMN) { launch<32, AMN, BMN, 1>(p, am, bm, cm, grid, s); break; }
    [[fallthrough]];
  default: TORCH_CHECK(false, "mm_sm100/general: unsupported tile width ", bn);
  }
}

//: a [batch, M, K] (any strides with one unit stride among the last two), b [batch, K, N] likewise,
//: out bf16 [batch, M, N] contiguous. ``split`` segments (1 = one chain); ``ws`` fp32, >= split*batch*M*N
//: when split > 1.
void gemm(const at::Tensor &a, const at::Tensor &b, at::Tensor out, at::Tensor ws, int64_t split, int64_t bn,
          int64_t grid_override, bool amn, bool bmn, int64_t cl) {
  TORCH_CHECK((a.dim() == 3 && b.dim() == 3 && out.dim() == 3) || (a.dim() == 2 && b.dim() == 2 && out.dim() == 2),
              "mm_sm100/general: 3-D (or 2-D) operands");
  const int o = a.dim() - 2;   // 2-D: batch 1, the batch axis absent
  TORCH_CHECK(a.scalar_type() == at::kBFloat16 && b.scalar_type() == at::kBFloat16 &&
                  (out.scalar_type() == at::kBFloat16 || out.scalar_type() == at::kFloat),
              "mm_sm100/general: bf16 operands, bf16 or fp32 out");
  const bool f32out = out.scalar_type() == at::kFloat;
  const c10::cuda::CUDAGuard guard(a.device());
  // An autograd worker may report the correct runtime device without a current driver
  // context.
  C10_CUDA_CHECK(c10::cuda::SetDevice(a.get_device(), /*force=*/true));
  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t B = o ? a.size(0) : 1, M = a.size(o), K = a.size(o + 1), N = b.size(o + 1);
  TORCH_CHECK((!o || (b.size(0) == B && out.size(0) == B)) && b.size(o) == K && out.size(o) == M &&
                  out.size(o + 1) == N, "mm_sm100/general: shape mismatch");
  TORCH_CHECK(K % 64 == 0 && K > 0, "mm_sm100/general: K % 64 == 0");
  TORCH_CHECK(out.stride(o + 1) == 1 && out.stride(o) >= N && (!o || out.stride(0) == M * out.stride(1)),
              "mm_sm100/general: out must be row-major [B, M, ldc]");
  TORCH_CHECK(amn ? a.stride(o) == 1 : a.stride(o + 1) == 1, "mm_sm100/general: A needs a unit stride");
  TORCH_CHECK(bmn ? b.stride(o + 1) == 1 : b.stride(o) == 1, "mm_sm100/general: B needs a unit stride");
  TORCH_CHECK(split >= 1 && split <= kMaxSplit, "mm_sm100/general: split in [1, 16]");
  TORCH_CHECK(!bmn || bn % 64 == 0, "mm_sm100/general: MN-major B needs a 64-multiple tile");
  const int64_t a_bs = B > 1 ? a.stride(0) : M * K, b_bs = B > 1 ? b.stride(0) : K * N;
  const CUtensorMap am = amn ? make_map(a.data_ptr(), M, K, B, a.stride(o + 1), a_bs, 64)
                             : make_map(a.data_ptr(), K, M, B, a.stride(o), a_bs,
                                        cl == 1 && M < int64_t(kBM) ? uint32_t((M + 7) / 8 * 8) : kBM);
  TORCH_CHECK(cl == 1 || (cl == 2 && (bn == 128 || bn == 256)), "mm_sm100/general: 2-CTA tile");
  const CUtensorMap bm = bmn ? make_map(b.data_ptr(), N, K, B, b.stride(o), b_bs, 64)
                             : make_map(b.data_ptr(), K, N, B, b.stride(o + 1), b_bs, uint32_t(bn / cl));
  Problem p{};
  p.M = uint32_t(M);
  p.N = uint32_t(N);
  p.batch = uint32_t(B);
  p.atoms = uint32_t(K / 64);
  p.seg_atoms = uint32_t((p.atoms + split - 1) / split);
  p.S = (p.atoms + p.seg_atoms - 1) / p.seg_atoms;
  TORCH_CHECK(int64_t(p.S) == split, "mm_sm100/general: split ", split, " leaves an empty segment at K=", K);
  p.m_tiles = uint32_t((M + kBM * cl - 1) / (kBM * cl));
  p.n_tiles = uint32_t((N + bn - 1) / bn);
  p.ldc = uint64_t(out.stride(o));
  CUtensorMap cm{};
  if (split > 1) {
    TORCH_CHECK(ws.scalar_type() == at::kFloat && ws.numel() >= split * B * M * N, "mm_sm100/general: workspace");
    p.out = ws.data_ptr();
    p.tma_store = (N % 4 == 0) ? 1u : 0u;
    if (p.tma_store)
      cm = make_map(ws.data_ptr(), N, M, uint64_t(split) * B, N, M * N, 32, 32, true, CU_TENSOR_MAP_SWIZZLE_128B);
  } else if (f32out) {
    TORCH_CHECK(p.ldc == uint64_t(N), "mm_sm100/general: fp32 out is contiguous");
    p.out = out.data_ptr();
    p.tma_store = (N % 4 == 0) ? 1u : 0u;
    if (p.tma_store)
      cm = make_map(out.data_ptr(), N, M, B, N, M * N, 32, 32, true, CU_TENSOR_MAP_SWIZZLE_128B);
  } else {
    p.out = out.data_ptr();
    p.tma_store = (p.ldc % 8 == 0) ? 1u : 0u;
    if (p.tma_store)
      cm = make_map(out.data_ptr(), N, M, B, p.ldc, M * p.ldc, 32, 32, false, CU_TENSOR_MAP_SWIZZLE_64B);
  }
  p.f32out = f32out ? 1u : 0u;
  // A short A box: rows past it are left as whatever the stage held (a UMMA row reads only its own
  // A row, and rows >= M are never stored), so only the real rows are fetched.
  p.a_rows = (!amn && cl == 1 && M < int64_t(kBM)) ? uint32_t((M + 7) / 8 * 8) : kBM;
  const int64_t items = int64_t(p.S) * B * p.m_tiles * p.n_tiles;
  const int sms = sm_count(a.device().index());
  int grid = int(std::min<int64_t>(items * cl, grid_override > 0 ? grid_override : sms));
  grid -= grid % int(cl);
  if (items > 0) {
    if (amn && bmn) launch_bn<true, true>(int(bn), int(cl), p, am, bm, cm, grid, stream);
    else if (amn) launch_bn<true, false>(int(bn), int(cl), p, am, bm, cm, grid, stream);
    else if (bmn) launch_bn<false, true>(int(bn), int(cl), p, am, bm, cm, grid, stream);
    else launch_bn<false, false>(int(bn), int(cl), p, am, bm, cm, grid, stream);
  }
  if (split > 1) {
    const uint64_t rows = uint64_t(B) * M;
    const int threads = 256;
    const int blocks = int(std::min<uint64_t>((rows * N + threads - 1) / threads, uint64_t(sms) * 16));
    auto *o = reinterpret_cast<__nv_bfloat16 *>(out.data_ptr());
    auto *of = reinterpret_cast<float *>(out.data_ptr());
    const float *w = ws.data_ptr<float>();
    switch (split) {
#define CASE(S_)                                                                                       \
  case S_:                                                                                             \
    if (f32out) combine_kernel<S_, float><<<blocks, threads, 0, stream>>>(w, of, rows, uint32_t(N), p.ldc); \
    else combine_kernel<S_, __nv_bfloat16><<<blocks, threads, 0, stream>>>(w, o, rows, uint32_t(N), p.ldc); \
    break;
      CASE(2) CASE(3) CASE(4) CASE(5) CASE(6) CASE(7) CASE(8) CASE(9) CASE(10) CASE(11) CASE(12) CASE(13)
      CASE(14) CASE(15) CASE(16)
#undef CASE
    default: TORCH_CHECK(false, "mm_sm100/general: split");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

} // namespace
} // namespace isoexec_mm_sm100_general

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemm", &isoexec_mm_sm100_general::gemm, "mm_sm100/general GEMM (see gen.cu)");
}
