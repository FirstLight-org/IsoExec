// The sm90 segment-tree mainloop: CUTLASS's TMA warp-specialized GMMA collective with a segmented
// consumer. Producer side (TMA loads, pipeline, descriptors) is the stock collective, inherited.
//
// What this adds is the program of recipe sm90_wgmma_segtree_v1 (see ../segments.py):
//   * the K extent a CTA walks is cut into segments of `seg_k_tiles` K blocks;
//   * every segment starts from zero (wgmma scale-d = 0 on its first k16 step) and is accumulated
//     in ascending K, one k16 wgmma after the other, in fp32;
//   * when kLevels > 0, segment roots are pushed into a binary counter of fp32 register fragments,
//     which evaluates the balanced adjacent-pair tree over the segment index with the left operand
//     carrying the lower indices (pik.plan.combine_order). `round_leaf` rounds every segment root
//     to bf16 (and back) before it enters the tree: the pik bf16-leaf contract.
// With kLevels == 0 the segment structure is a single chain (S = 1 per batch) and the consumer's
// instruction stream is the stock collective's.
#pragma once

#include <cuda_bf16.h>

#include "cute/tensor.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"

namespace isoexec_mm_sm90 {

using namespace cute;

template <class Base, int kLevels>
struct SegTreeMainloop : Base {
  using typename Base::DispatchPolicy;
  using typename Base::TiledMma;
  using typename Base::SmemLayoutA;
  using typename Base::SmemLayoutB;
  using typename Base::MainloopPipeline;
  using typename Base::PipelineState;
  using typename Base::TensorStorage;
  using Base::K_PIPE_MMAS;

  static constexpr int Levels = kLevels;

  struct Arguments : Base::Arguments {
    int seg_k_tiles = 0;   // K blocks per segment; 0 = the whole K extent is one segment
    int round_leaf = 0;    // round each segment root to bf16 before the tree (pik bf16 leaves)
  };

  struct Params : Base::Params {
    int seg_k_tiles;
    int round_leaf;
  };

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args,
                                                  void* workspace) {
    Params p;
    static_cast<typename Base::Params&>(p) = Base::to_underlying_arguments(
        problem_shape, static_cast<typename Base::Arguments const&>(args), workspace);
    p.seg_k_tiles = args.seg_k_tiles;
    p.round_leaf = args.round_leaf;
    return p;
  }

  template <class Frg>
  CUTLASS_DEVICE static void round_bf16(Frg& f) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(f); ++i) {
      f(i) = __bfloat162float(__float2bfloat16_rn(f(i)));
    }
  }

  template <class Frg>
  CUTLASS_DEVICE static void add_into(Frg& dst, Frg const& lhs, Frg const& rhs) {
    // dst = lhs + rhs, one fp32 add per element (built with --fmad=false; no contraction possible)
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(dst); ++i) {
      dst(i) = __fadd_rn(lhs(i), rhs(i));
    }
  }

  template <class FrgTensorC>
  CUTLASS_DEVICE void mma(MainloopPipeline pipeline, PipelineState smem_pipe_read, FrgTensorC& accum,
                          int k_tile_count, int thread_idx, TensorStorage& shared_tensors,
                          Params const& mainloop_params) {
    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");

    Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.data()), SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.data()), SmemLayoutB{});

    constexpr int MmaWarpGroups = size(TiledMma{}) / cutlass::NumThreadsPerWarpGroup;
    Layout warp_group_thread_layout = make_layout(Int<MmaWarpGroups>{}, Int<cutlass::NumThreadsPerWarpGroup>{});
    int warp_group_idx = __shfl_sync(0xFFFFFFFF, thread_idx / cutlass::NumThreadsPerWarpGroup, 0);

    TiledMma tiled_mma;
    auto thread_mma = tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx));
    Tensor tCsA = thread_mma.partition_A(sA);
    Tensor tCsB = thread_mma.partition_B(sB);
    Tensor tCrA = thread_mma.make_fragment_A(tCsA);
    Tensor tCrB = thread_mma.make_fragment_B(tCsB);

    // Binary-counter stack of segment subtree roots: stack[l] holds a complete subtree of 2^l
    // segments while bit l of `filled` is set.
    using AccType = typename FrgTensorC::value_type;
    Tensor stack = make_tensor<AccType>(make_shape(Int<(kLevels > 0 ? kLevels : 1)>{}, size(accum)));
    uint32_t filled = 0;

    const int seg_tiles = (mainloop_params.seg_k_tiles > 0) ? mainloop_params.seg_k_tiles : k_tile_count;
    const int prologue = min(K_PIPE_MMAS, k_tile_count);
    PipelineState smem_pipe_release = smem_pipe_read;

    int in_seg = 0;  // K blocks of the current segment already issued
    CUTLASS_PRAGMA_NO_UNROLL
    for (int t = 0; t < k_tile_count; ++t) {
      auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      pipeline.consumer_wait(smem_pipe_read, barrier_token);
      int read_stage = smem_pipe_read.index();

      warpgroup_fence_operand(accum);
      warpgroup_arrive();
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
        // a segment starts from zero: scale-d = 0 on its first k16 step, 1 on every later one
        tiled_mma.accumulate_ = (in_seg == 0 && k_block == 0) ? GMMA::ScaleOut::Zero : GMMA::ScaleOut::One;
        cute::gemm(tiled_mma, tCrA(_, _, k_block, read_stage), tCrB(_, _, k_block, read_stage), accum);
      }
      warpgroup_commit_batch();
      ++in_seg;

      if constexpr (kLevels > 0) {
        if (in_seg == seg_tiles) {
          // segment root complete: drain the wgmma batch, then push it into the tree
          warpgroup_wait<0>();
          warpgroup_fence_operand(accum);
          in_seg = 0;
          if (seg_tiles != k_tile_count) {
            if (mainloop_params.round_leaf) {
              round_bf16(accum);
            }
            // carry: while level l is occupied, accum = stack[l] + accum (lower indices on the left)
            CUTLASS_PRAGMA_UNROLL
            for (int l = 0; l < kLevels; ++l) {
              Tensor slot = stack(l, _);
              if (filled & (1u << l)) {
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < size(accum); ++i) {
                  accum(i) = __fadd_rn(slot(i), accum(i));
                }
                filled &= ~(1u << l);
              } else {
                // first free level: park the subtree here unless it is the final root
                if (t + 1 < k_tile_count) {
                  CUTLASS_PRAGMA_UNROLL
                  for (int i = 0; i < size(accum); ++i) {
                    slot(i) = accum(i);
                  }
                  filled |= (1u << l);
                }
                break;
              }
            }
          }
          warpgroup_fence_operand(accum);
        } else {
          warpgroup_wait<K_PIPE_MMAS>();
          warpgroup_fence_operand(accum);
        }
      } else {
        warpgroup_wait<K_PIPE_MMAS>();
        warpgroup_fence_operand(accum);
      }

      if (t >= prologue) {
        pipeline.consumer_release(smem_pipe_release);
        ++smem_pipe_release;
      }
      ++smem_pipe_read;
    }
    warpgroup_fence_operand(accum);
  }
};

}  // namespace isoexec_mm_sm90
