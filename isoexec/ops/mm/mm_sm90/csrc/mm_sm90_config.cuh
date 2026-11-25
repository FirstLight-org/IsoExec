// One compiled configuration of the sm90 segment-tree GEMM: tile, cluster, kernel schedule, output
// type and tree depth. Every configuration computes the same program (../segments.py); they differ in
// ownership and scheduling only.
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"

#include "mm_sm90_mainloop.cuh"

namespace isoexec_mm_sm90 {

// Arguments of one launch. A is [M, *] with row stride lda, B (the weight) is [N, *] with row stride
// ldb, D is [M, N] with row stride ldd. The problem is batched over L: batch l reads A and B at
// K offset l * kseg (a_batch / b_batch elements) and writes D at l * d_batch elements.
struct LaunchArgs {
  const void* a;
  const void* b;
  void* d;
  int M, N, kseg, L;
  int64_t lda, ldb, ldd;
  int64_t a_batch, b_batch, d_batch;
  int seg_k_tiles;   // K blocks per segment inside one batch (0: one segment)
  int round_leaf;
  int sm_count;      // persistent schedulers: CTAs to launch (0: the device's)
  int swizzle;       // persistent schedulers: tile swizzle (1, 2, 4, 8) -- tile ORDER only
  int raster;        // persistent schedulers: 0 heuristic, 1 along M, 2 along N -- tile ORDER only
  cudaStream_t stream;
  int pdl;           // programmatic dependent launch: the kernel (built with CUTLASS_ENABLE_GDC_FOR_SM90)
                     // waits on its predecessor before touching global memory -- launch SCHEDULE only
};

struct ConfigInfo {
  const char* name;
  int block_m, block_n, block_k, cluster_m, cluster_n;
  const char* schedule;   // "ws", "pingpong", "cooperative"
  int out_fp32;           // 1: fp32 output (segment roots / subtree nodes); 0: bf16
  int levels;             // tree depth the onchip fold supports: max segments per CTA = 2^levels
  int stages;
  int swap;               // 1: tokens on the wgmma N side; block_m tiles the weight rows, block_n the tokens
};

using RunFn = int (*)(LaunchArgs const&);

// kSwap: the operands enter the tensor core swapped -- the weight on the wgmma M side, the tokens on
// its N side (8..256 wide), D^T = W x^T stored column-major, i.e. D row-major. Each output element is
// still the dot product of one x row and one W row over the same ascending k16 steps; what changes is
// only which tile extent holds the tokens. (Measured bit-equal; the battery gates it.) It is what lets
// a decode-sized M occupy 8..32 wgmma columns instead of 64 padded rows.
// kStages: pipeline depth; 0 takes every byte of shared memory (one CTA per SM). Fewer stages let
// several CTAs share an SM, which is what a one-tile-per-CTA decode launch needs.
template <int BM, int BN, int CM, int CN, class KernelSchedule, class EpilogueSchedule, class ElementD,
          int kLevels, bool kSwap = false, int kStages = 0>
struct Gemm {
  using ElementA = cutlass::bfloat16_t;
  using ElementB = cutlass::bfloat16_t;
  using ElementAcc = float;
  using TileShape = Shape<Int<BM>, Int<BN>, _64>;
  using ClusterShape = Shape<Int<CM>, Int<CN>, _1>;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;  // the weight [N, K] row-major: K-major B
  using LayoutD = cute::conditional_t<kSwap, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>;
  static constexpr int AlignAB = 8;
  static constexpr int AlignD = 16 / sizeof(ElementD);

  // Identity epilogue: the fp32 accumulator converted once (RN) to the output type. No alpha
  // multiply, no source operand.
  using Fusion = cutlass::epilogue::fusion::Sm90EVT<
      cutlass::epilogue::fusion::Sm90Compute<cutlass::epilogue::thread::Identity, ElementD, float,
                                             cutlass::FloatRoundStyle::round_to_nearest>,
      cutlass::epilogue::fusion::Sm90AccFetch>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAcc, float, void, LayoutD, AlignD, ElementD,
      LayoutD, AlignD, EpilogueSchedule, Fusion>::CollectiveOp;

  using StockMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, ElementA, LayoutA, AlignAB, ElementB, LayoutB, AlignAB,
      ElementAcc, TileShape, ClusterShape,
      cute::conditional_t<(kStages > 0), cutlass::gemm::collective::StageCount<(kStages > 0 ? kStages : 2)>,
                          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
                              sizeof(typename CollectiveEpilogue::SharedStorage))>>,
      KernelSchedule>::CollectiveOp;

  using Mainloop = SegTreeMainloop<StockMainloop, kLevels>;

  // Never StreamK: a K split would be a different program. The persistent scheduler only orders tiles.
  using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Mainloop, CollectiveEpilogue,
                                                      cutlass::gemm::PersistentScheduler>;
  using Device = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

  static constexpr int Stages = StockMainloop::DispatchPolicy::Stages;

  static int run(LaunchArgs const& a) {
    using StrideA = typename Kernel::StrideA;
    using StrideB = typename Kernel::StrideB;
    using StrideD = typename Kernel::StrideD;
    StrideD sd;
    typename Mainloop::Arguments ml{};
    int pm = a.M, pn = a.N;
    if constexpr (kSwap) {
      // problem (N, M): A = the weight, B = the tokens, D^T column-major == D row-major
      pm = a.N;
      pn = a.M;
      ml.ptr_A = static_cast<ElementA const*>(a.b);
      ml.dA = StrideA(a.ldb, Int<1>{}, a.b_batch);
      ml.ptr_B = static_cast<ElementB const*>(a.a);
      ml.dB = StrideB(a.lda, Int<1>{}, a.a_batch);
      sd = StrideD(Int<1>{}, a.ldd, a.d_batch);
    } else {
      ml.ptr_A = static_cast<ElementA const*>(a.a);
      ml.dA = StrideA(a.lda, Int<1>{}, a.a_batch);
      ml.ptr_B = static_cast<ElementB const*>(a.b);
      ml.dB = StrideB(a.ldb, Int<1>{}, a.b_batch);
      sd = StrideD(a.ldd, Int<1>{}, a.d_batch);
    }
    ml.seg_k_tiles = a.seg_k_tiles;
    ml.round_leaf = a.round_leaf;

    typename Device::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,
                                    {pm, pn, a.kseg, a.L},
                                    ml,
                                    {{}, nullptr, sd, static_cast<ElementD*>(a.d), sd}};
    cudaGetDevice(&args.hw_info.device_id);
    args.hw_info.sm_count = a.sm_count;
    // Raster order and swizzle only order the tiles (which CTA owns which output tile, when): no
    // element's arithmetic reads them.
    if constexpr (!cute::is_same_v<KernelSchedule, cutlass::gemm::KernelTmaWarpSpecialized>) {
      using Raster = cutlass::gemm::kernel::detail::RasterOrderOptions;
      args.scheduler.max_swizzle_size = a.swizzle > 0 ? a.swizzle : 1;
      args.scheduler.raster_order = a.raster == 1 ? Raster::AlongM : a.raster == 2 ? Raster::AlongN : Raster::Heuristic;
    }

    Device op;
    // The output's TMA store clips at the true extent; only its row pitch must be 16 bytes (ldd is
    // checked below). CUTLASS's host check also asks the contiguous EXTENT to be aligned, so it is
    // asked about the aligned extent -- the launch itself uses the true N.
    if (a.ldd % AlignD != 0 || a.d_batch % AlignD != 0) return 5;
    {
      auto check = args;
      constexpr int kAl = AlignD;
      if constexpr (kSwap) {
        get<0>(check.problem_shape) = (pm + kAl - 1) / kAl * kAl;
      } else {
        get<1>(check.problem_shape) = (pn + kAl - 1) / kAl * kAl;
      }
      if (op.can_implement(check) != cutlass::Status::kSuccess) return 1;
    }
    // Every configuration needs zero workspace (no StreamK, no split-K reduction): nothing to
    // allocate, so a launch is capture-safe.
    if (Device::get_workspace_size(args) != 0) return 2;
    if (op.initialize(args, nullptr, a.stream) != cutlass::Status::kSuccess) return 3;
    if (op.run(a.stream, nullptr, a.pdl != 0) != cutlass::Status::kSuccess) return 4;
    return 0;
  }
};

}  // namespace isoexec_mm_sm90
