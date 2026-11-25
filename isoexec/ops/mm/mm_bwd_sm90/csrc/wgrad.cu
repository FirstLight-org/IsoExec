// sm90 weight-gradient GEMM with an fp32 accumulate epilogue (gradient-accumulation fusion).
//
//   main_grad[N_out, K_in] (fp32) += dy[T, N_out]^T @ x[T, K_in]      (bf16 operands)
//
// This is the backward-only half of a linear layer: no forward bit depends on it. Program per
// output element: one fp32 WGMMA chain over the T axis in ascending 64-token K blocks (no split-K,
// no stream-K, one CTA owns a whole output tile), then ONE fp32 add of the existing main_grad value:
// d = fma(1, c, 1 * acc) == RN(c + acc). The unfused program it replaces rounded the chain to bf16
// first and added in a second pass; this one rounds once. The result is a function of the operands
// and of the compiled tile shape only (tile ORDER -- swizzle/raster -- never changes an element), so
// it is deterministic run to run and independent of what else shares the launch.
//
// Operand layouts (CUTLASS GEMM M = N_out, N = K_in, K = T):
//   A = dy^T  : element (m, k) at dy[k * ldy + m]  -> M-major (ColumnMajor), WGMMA transposed load
//   B = x     : element (n, k) at x[k * ldx + n]   -> N-major (RowMajor B)
//   C = D     : main_grad row-major with row pitch ldd (in place)
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"

using namespace cute;

namespace isoexec_wgrad_sm90 {

struct Args {
  const void *dy, *x;
  void *d;
  int M, N, K;  // M = N_out, N = K_in, K = T
  int64_t ldy, ldx, ldd;
  int sm_count, swizzle, raster;
  cudaStream_t stream;
};

template <int BM, int BN, int CM, int CN, class KernelSchedule, class EpilogueSchedule>
struct Wgrad {
  using ElementA = cutlass::bfloat16_t;
  using ElementB = cutlass::bfloat16_t;
  using ElementAcc = float;
  using ElementC = float;
  using ElementD = float;
  using LayoutA = cutlass::layout::ColumnMajor;  // dy^T: M (= N_out) contiguous
  using LayoutB = cutlass::layout::RowMajor;     // x: N (= K_in) contiguous
  using LayoutC = cutlass::layout::RowMajor;
  using TileShape = Shape<Int<BM>, Int<BN>, _64>;
  using ClusterShape = Shape<Int<CM>, Int<CN>, _1>;
  static constexpr int AlignAB = 8;
  static constexpr int AlignCD = 4;

  // D = alpha * acc + beta * C with alpha = beta = 1: fma(1, c, acc) -- one fp32 rounding of c + acc.
  using Fusion = cutlass::epilogue::fusion::LinearCombination<ElementD, float, ElementC, float,
                                                              cutlass::FloatRoundStyle::round_to_nearest>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAcc, float, ElementC, LayoutC, AlignCD, ElementD,
      LayoutC, AlignCD, EpilogueSchedule, Fusion>::CollectiveOp;

  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, ElementA, LayoutA, AlignAB, ElementB, LayoutB, AlignAB,
      ElementAcc, TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      KernelSchedule>::CollectiveOp;

  // Never StreamK / split-K: each output element is one chain. The persistent scheduler only orders tiles.
  using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Mainloop, CollectiveEpilogue,
                                                      cutlass::gemm::PersistentScheduler>;
  using Device = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

  static int run(Args const &a) {
    using StrideA = typename Kernel::StrideA;
    using StrideB = typename Kernel::StrideB;
    using StrideC = typename Kernel::StrideC;
    using StrideD = typename Kernel::StrideD;
    StrideA sa = make_stride(Int<1>{}, a.ldy, int64_t(0));
    StrideB sb = make_stride(Int<1>{}, a.ldx, int64_t(0));
    StrideC sc = make_stride(a.ldd, Int<1>{}, int64_t(0));
    StrideD sd = make_stride(a.ldd, Int<1>{}, int64_t(0));
    typename Device::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {a.M, a.N, a.K, 1},
        {static_cast<ElementA const *>(a.dy), sa, static_cast<ElementB const *>(a.x), sb},
        {{1.0f, 1.0f}, static_cast<ElementC const *>(a.d), sc, static_cast<ElementD *>(a.d), sd}};
    cudaGetDevice(&args.hw_info.device_id);
    args.hw_info.sm_count = a.sm_count;
    using Raster = cutlass::gemm::kernel::detail::RasterOrderOptions;
    args.scheduler.max_swizzle_size = a.swizzle > 0 ? a.swizzle : 1;
    args.scheduler.raster_order = a.raster == 1 ? Raster::AlongM : a.raster == 2 ? Raster::AlongN : Raster::Heuristic;

    Device op;
    // TMA needs 16-byte global strides and bases (checked by the caller); CUTLASS's host check also
    // asks the contiguous EXTENTS to be aligned, so it is asked about aligned extents. The launch
    // uses the true extents: TMA clips loads (zero fill) and stores at the tensor boundary.
    {
      auto check = args;
      get<0>(check.problem_shape) = (a.M + AlignAB - 1) / AlignAB * AlignAB;
      get<1>(check.problem_shape) = (a.N + AlignAB - 1) / AlignAB * AlignAB;
      if (op.can_implement(check) != cutlass::Status::kSuccess) return 1;
    }
    if (Device::get_workspace_size(args) != 0) return 2;
    if (op.initialize(args, nullptr, a.stream) != cutlass::Status::kSuccess) return 3;
    if (op.run(a.stream) != cutlass::Status::kSuccess) return 4;
    return 0;
  }
};

using CO = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
using COE = cutlass::epilogue::TmaWarpSpecializedCooperative;
using PP = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
using PPE = cutlass::epilogue::TmaWarpSpecialized;

// The compiled configurations. Speed only: every one computes the same element program.
#define WGRAD_CONFIGS(X)                     \
  X(co_128x256_c2, 128, 256, 2, 1, CO, COE)  \
  X(co_128x256, 128, 256, 1, 1, CO, COE)     \
  X(co_256x128_c2, 256, 128, 1, 2, CO, COE)  \
  X(co_128x128_c2, 128, 128, 2, 1, CO, COE)  \
  X(pp_128x128, 128, 128, 1, 1, PP, PPE)     \
  X(co_128x128, 128, 128, 1, 1, CO, COE)

#define WGRAD_RUN(name, BM, BN, CM, CN, KS, ES) \
  static int run_##name(Args const &a) { return Wgrad<BM, BN, CM, CN, KS, ES>::run(a); }
WGRAD_CONFIGS(WGRAD_RUN)
#undef WGRAD_RUN

struct Entry {
  const char *name;
  int (*run)(Args const &);
};
#define WGRAD_ENTRY(name, BM, BN, CM, CN, KS, ES) Entry{#name, &run_##name},
static const Entry kTable[] = {WGRAD_CONFIGS(WGRAD_ENTRY)};
#undef WGRAD_ENTRY
static constexpr int kNumConfigs = sizeof(kTable) / sizeof(kTable[0]);

static int device_sm_count() {
  int dev = 0, n = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
  return n;
}

}  // namespace isoexec_wgrad_sm90

using namespace isoexec_wgrad_sm90;

std::vector<std::string> wgrad_config_names() {
  std::vector<std::string> l;
  for (int i = 0; i < kNumConfigs; ++i) l.emplace_back(kTable[i].name);
  return l;
}

// main_grad[N_out, K_in] += dy[T, N_out]^T @ x[T, K_in], in place on the current stream.
void wgrad_accumulate(int64_t cid, torch::Tensor const &dy, torch::Tensor const &x, torch::Tensor &main_grad,
                       int64_t swizzle, int64_t raster) {
  TORCH_CHECK(cid >= 0 && cid < kNumConfigs, "wgrad_sm90: config index out of range");
  TORCH_CHECK(dy.scalar_type() == at::kBFloat16 && x.scalar_type() == at::kBFloat16, "wgrad_sm90: bf16 operands");
  TORCH_CHECK(main_grad.scalar_type() == at::kFloat, "wgrad_sm90: fp32 main_grad");
  TORCH_CHECK(dy.dim() == 2 && x.dim() == 2 && main_grad.dim() == 2, "wgrad_sm90: 2-D operands");
  TORCH_CHECK(dy.size(0) == x.size(0), "wgrad_sm90: dy and x must share the token axis");
  TORCH_CHECK(main_grad.size(0) == dy.size(1) && main_grad.size(1) == x.size(1), "wgrad_sm90: main_grad is [N_out, K_in]");
  TORCH_CHECK(dy.stride(1) == 1 && x.stride(1) == 1 && main_grad.stride(1) == 1, "wgrad_sm90: unit inner stride");
  TORCH_CHECK(dy.stride(0) % 8 == 0 && x.stride(0) % 8 == 0 && main_grad.stride(0) % 4 == 0,
              "wgrad_sm90: 16-byte row pitches required");
  auto al = [](void const *p) { return (reinterpret_cast<uintptr_t>(p) & 15u) == 0; };
  TORCH_CHECK(al(dy.data_ptr()) && al(x.data_ptr()) && al(main_grad.data_ptr()), "wgrad_sm90: 16-byte aligned bases");
  if (dy.size(0) == 0 || main_grad.numel() == 0) return;
  const at::cuda::CUDAGuard guard(dy.device());
  Args a{dy.data_ptr(),
         x.data_ptr(),
         main_grad.data_ptr(),
         static_cast<int>(main_grad.size(0)),
         static_cast<int>(main_grad.size(1)),
         static_cast<int>(dy.size(0)),
         dy.stride(0),
         x.stride(0),
         main_grad.stride(0),
         device_sm_count(),
         static_cast<int>(swizzle),
         static_cast<int>(raster),
         at::cuda::getCurrentCUDAStream().stream()};
  int rc = kTable[cid].run(a);
  TORCH_CHECK(rc == 0, "wgrad_sm90 config ", kTable[cid].name, " refused the launch (code ", rc, ") for N_out=", a.M,
              " K_in=", a.N, " T=", a.K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

