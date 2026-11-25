// Grouped (MoE) weight-gradient GEMM with the fp32 accumulate epilogue of wgrad.cu.
//
// Experts' rows are contiguous token ranges of the permuted operands: expert e owns tokens
// [end[e-1], end[e]). A group g names an expert e(g), a column block [c_g, c_g + M_g) of dy and the
// fp32 main_grad it accumulates into:
//
//   main_grad_g[M_g, N] += dy[k0:k1, c_g : c_g + M_g]^T @ b[k0:k1, :]           (k0, k1 = e(g)'s range)
//
// fc1 of a gated expert is two groups per expert (gate rows from di[:, :I], up rows from di[:, I:],
// each into its own parameter's main_grad); fc2 is one group per expert. The token counts live on the
// device (the routing plan), so a tiny prep kernel writes the per-group problem shapes and operand
// pointers and the CUTLASS ptr-array kernel reads them there: no host sync, no host shape list. An
// expert with no tokens gets M = 0 (no tiles; its main_grad is untouched).
//
// Per element the program is wgrad.cu's: one fp32 WGMMA chain over the expert's tokens in ascending
// 64-token blocks, then one fp32 add of the existing main_grad value. Deterministic run to run.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda_runtime.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

using namespace cute;

namespace isoexec_mm_bwd_sm90 {
namespace grouped {

using ElementA = cutlass::bfloat16_t;
using ElementB = cutlass::bfloat16_t;
using ElementC = float;
using LayoutA = cutlass::layout::ColumnMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;
using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
using TileShape = Shape<_128, _128, _64>;
using ClusterShape = Shape<_1, _1, _1>;

using Fusion = cutlass::epilogue::fusion::LinearCombination<float, float, float, float,
                                                            cutlass::FloatRoundStyle::round_to_nearest>;
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto, float, float, ElementC, LayoutC *, 4, ElementC, LayoutC *, 4,
    cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative, Fusion>::CollectiveOp;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, ElementA, LayoutA *, 8, ElementB, LayoutB *, 8, float,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
        sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative>::CollectiveOp;
using Kernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape, Mainloop, CollectiveEpilogue>;
using Device = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

using StrideA = typename Kernel::InternalStrideA;
using StrideB = typename Kernel::InternalStrideB;
using StrideC = typename Kernel::InternalStrideC;
using StrideD = typename Kernel::InternalStrideD;
using Shape3 = typename ProblemShape::UnderlyingProblemShape;

// meta[g] = {expert, column offset in dy, rows M_g, main_grad row pitch}; mg[g] = main_grad base address.
template <typename TEnd>
__global__ void prep(const TEnd *__restrict__ ends, const int64_t *__restrict__ meta, const int64_t *__restrict__ mg,
                     const ElementA *dy, int64_t ldy, const ElementB *b, int64_t ldb, int n, int groups,
                     Shape3 *shapes, const ElementA **pa, const ElementB **pb, const ElementC **pc, ElementC **pd,
                     StrideA *sa, StrideB *sb, StrideC *sc, StrideD *sd) {
  int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= groups) return;
  const int64_t e = meta[4 * g], col = meta[4 * g + 1], m = meta[4 * g + 2], ldd = meta[4 * g + 3];
  const int64_t k0 = e == 0 ? 0 : int64_t(ends[e - 1]);
  const int64_t k = int64_t(ends[e]) - k0;
  shapes[g] = Shape3(k > 0 ? int(m) : 0, n, int(k));
  pa[g] = dy + k0 * ldy + col;
  pb[g] = b + k0 * ldb;
  pc[g] = reinterpret_cast<const ElementC *>(mg[g]);
  pd[g] = reinterpret_cast<ElementC *>(mg[g]);
  sa[g] = make_stride(Int<1>{}, ldy, Int<0>{});
  sb[g] = make_stride(Int<1>{}, ldb, Int<0>{});
  sc[g] = make_stride(ldd, Int<1>{}, Int<0>{});
  sd[g] = make_stride(ldd, Int<1>{}, Int<0>{});
}

}  // namespace grouped
}  // namespace isoexec_mm_bwd_sm90

using namespace isoexec_mm_bwd_sm90::grouped;

// dy [T, *] bf16 (unit column stride), b [T, N] bf16, ends [E] int32/int64 (device, cumulative token
// ends), meta [G, 4] int64 (device), mg [G] int64 main_grad addresses (device).
void wgrad_grouped_accumulate(torch::Tensor const &dy, torch::Tensor const &b, torch::Tensor const &ends,
                              torch::Tensor const &meta, torch::Tensor const &mg) {
  TORCH_CHECK(dy.scalar_type() == at::kBFloat16 && b.scalar_type() == at::kBFloat16, "wgrad_grouped: bf16 operands");
  TORCH_CHECK(dy.dim() == 2 && b.dim() == 2 && dy.size(0) == b.size(0), "wgrad_grouped: [T, *] operands");
  TORCH_CHECK(dy.stride(1) == 1 && b.stride(1) == 1 && dy.stride(0) % 8 == 0 && b.stride(0) % 8 == 0,
              "wgrad_grouped: unit inner stride, 16-byte row pitches");
  TORCH_CHECK(meta.scalar_type() == at::kLong && meta.dim() == 2 && meta.size(1) == 4 && meta.is_contiguous(),
              "wgrad_grouped: meta is a contiguous [G, 4] int64 tensor");
  TORCH_CHECK(mg.scalar_type() == at::kLong && mg.numel() == meta.size(0) && mg.is_contiguous(),
              "wgrad_grouped: mg is [G] int64");
  TORCH_CHECK(ends.scalar_type() == at::kInt || ends.scalar_type() == at::kLong, "wgrad_grouped: int ends");
  const int groups = int(meta.size(0));
  if (groups == 0 || dy.size(0) == 0) return;
  const at::cuda::CUDAGuard guard(dy.device());
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  // One scratch block: shapes, 4 pointer arrays, 4 stride arrays (each padded to 16 bytes per entry).
  constexpr int64_t kSlot = 16;
  static_assert(sizeof(Shape3) <= kSlot && sizeof(StrideA) <= kSlot && sizeof(StrideC) <= kSlot, "slot size");
  auto scratch = at::empty({9 * groups * kSlot}, dy.options().dtype(at::kByte));
  uint8_t *base = scratch.data_ptr<uint8_t>();
  auto at_ = [&](int i) { return base + int64_t(i) * groups * kSlot; };
  auto shapes = reinterpret_cast<Shape3 *>(at_(0));
  auto pa = reinterpret_cast<const ElementA **>(at_(1));
  auto pb = reinterpret_cast<const ElementB **>(at_(2));
  auto pc = reinterpret_cast<const ElementC **>(at_(3));
  auto pd = reinterpret_cast<ElementC **>(at_(4));
  auto sa = reinterpret_cast<StrideA *>(at_(5));
  auto sb = reinterpret_cast<StrideB *>(at_(6));
  auto sc = reinterpret_cast<StrideC *>(at_(7));
  auto sd = reinterpret_cast<StrideD *>(at_(8));
  auto dyp = static_cast<const ElementA *>(dy.data_ptr());
  auto bp = static_cast<const ElementB *>(b.data_ptr());
  const int threads = 128, blocks = (groups + threads - 1) / threads;
  if (ends.scalar_type() == at::kInt)
    prep<<<blocks, threads, 0, stream>>>(ends.data_ptr<int>(), meta.data_ptr<int64_t>(), mg.data_ptr<int64_t>(), dyp,
                                         dy.stride(0), bp, b.stride(0), int(b.size(1)), groups, shapes, pa, pb, pc,
                                         pd, sa, sb, sc, sd);
  else
    prep<<<blocks, threads, 0, stream>>>(ends.data_ptr<int64_t>(), meta.data_ptr<int64_t>(), mg.data_ptr<int64_t>(),
                                         dyp, dy.stride(0), bp, b.stride(0), int(b.size(1)), groups, shapes, pa, pb,
                                         pc, pd, sa, sb, sc, sd);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  typename Device::Arguments args;
  args.mode = cutlass::gemm::GemmUniversalMode::kGrouped;
  args.problem_shape = {groups, shapes, nullptr};
  args.mainloop = {pa, sa, pb, sb};
  args.epilogue.ptr_C = pc;
  args.epilogue.dC = sc;
  args.epilogue.ptr_D = pd;
  args.epilogue.dD = sd;
  args.epilogue.thread.alpha = 1.0f;
  args.epilogue.thread.beta = 1.0f;
  cudaGetDevice(&args.hw_info.device_id);
  args.hw_info.sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

  Device op;
  TORCH_CHECK(op.can_implement(args) == cutlass::Status::kSuccess, "wgrad_grouped: CUTLASS refused the problem");
  const size_t ws_bytes = Device::get_workspace_size(args);
  auto ws = at::empty({int64_t(ws_bytes) + 1}, dy.options().dtype(at::kByte));
  TORCH_CHECK(op.initialize(args, ws.data_ptr(), stream) == cutlass::Status::kSuccess, "wgrad_grouped: initialize");
  TORCH_CHECK(op.run(stream) == cutlass::Status::kSuccess, "wgrad_grouped: run");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
