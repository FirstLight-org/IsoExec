#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cstdint>

#include <cutlass/arch/barrier.h>

#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/epilogue/sm100_mega_gate.cuh>
#include <deep_gemm/layout/mega_gate.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/scheduler/mega_gate.cuh>

namespace deep_gemm {

namespace mega_gate_layout = layout::mega_gate;

// IsoExec router program (`router_sm100.route()` bit for bit): the logits are the 4-segment tree of ascending tcgen05
// chains (`layout::mega_gate::kNumSegments`), each segment in its own TMEM accumulator, and the gate is the router's
// selection (`kSelectMode`: 0 = sigmoid grouped, 1 = sqrt-softplus; `kNumExpertsPerLane` = the router's EPL), writing
// int32 ids and FP32 weights `[T, K + S]` in ascending expert id with the shared slots last.
template <uint32_t SHAPE_K, uint32_t kNumRoutedExperts,
          uint32_t kNumGateThreads,
          uint32_t kNumMmaCtas, uint32_t kNumSplitK, uint32_t kNumExpertGroups,
          uint32_t kNumTopk, uint32_t kSelectMode, uint32_t kNumExpertsPerLane>
CUTLASS_GLOBAL void __launch_bounds__(mega_gate_layout::kNumNonEpilogueThreads + kNumGateThreads, 1)
sm100_bf16_mega_gate_impl(const __grid_constant__ cute::TmaDescriptor tensor_map_x,
                          const __grid_constant__ cute::TmaDescriptor tensor_map_weight,
                          const float* bias, const uint8_t* mask,
                          int* out_idx, float* out_w,
                          uint32_t num_routed_experts, uint32_t num_shared_experts,
                          uint32_t num_groups, uint32_t num_topk_groups, float routed_scaling_factor,
                          void* gmem_scratch, void* gmem_score_barriers,
                          uint32_t num_tokens, uint32_t block_tokens, uint32_t num_stages,
                          uint32_t num_worker_groups, uint32_t num_token_blocks) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    using Allocator = cute::conditional_t<kNumMmaCtas == 1, cute::TMEM::Allocator1Sm, cute::TMEM::Allocator2Sm>;

    constexpr uint32_t kNumNonEpilogueWarps = mega_gate_layout::kNumNonEpilogueThreads / 32;
    constexpr uint32_t kNumGateWarps = kNumGateThreads / 32;
    DG_STATIC_ASSERT(kNumGateThreads % 128 == 0, "Gate warps must be whole warpgroups");
    constexpr uint32_t BLOCK_K = mega_gate_layout::BLOCK_K;
    constexpr uint32_t UMMA_K = 16;
    constexpr uint32_t kSwizzleMode = mega_gate_layout::kSwizzleMode;
    constexpr uint32_t kNumExpertsPerGroup = kNumRoutedExperts / kNumExpertGroups;

    // NOTES: the SMEM/TMEM capacity is `kNumMaxBlockTokens`, the actual UMMA N (`block_tokens`) is a runtime value
    constexpr uint32_t UMMA_M = kNumExpertsPerGroup;
    constexpr uint32_t UMMA_N = mega_gate_layout::kNumMaxBlockTokens;
    constexpr uint32_t kNumLogicalCtas = kNumMmaCtas * kNumExpertGroups * kNumSplitK;
    using ScoreBarrier = sched::mega_gate::ScoreBarrier<kNumLogicalCtas>;
    constexpr uint32_t LOAD_BLOCK_M = UMMA_N / kNumMmaCtas;
    constexpr uint32_t LOAD_BLOCK_N = UMMA_M / kNumMmaCtas;
    DG_STATIC_ASSERT((kNumMmaCtas == 1 and UMMA_M == 128) or (kNumMmaCtas == 2 and (UMMA_M == 128 or UMMA_M == 256)),
                     "Invalid MMA instruction shape");
    constexpr uint32_t kNumEpilogueStages = mega_gate_layout::kNumEpilogueStages;
    using SharedStorage = mega_gate_layout::SharedStorage;

    // IsoExec segments: a split-K slice owns `kNumSegmentsPerCta` whole segments, each ONE ascending chain from zero in
    // its own accumulator of `kAccumCols` columns; the block's tokens are bounded so all of them fit the 512 columns
    constexpr uint32_t kNumSegments = mega_gate_layout::kNumSegments;
    DG_STATIC_ASSERT(kNumSplitK >= 1 and kNumSplitK <= kNumSegments and kNumSegments % kNumSplitK == 0,
                     "A split-K slice must own whole segments");
    constexpr uint32_t kNumSegmentsPerCta = kNumSegments / kNumSplitK;
    DG_STATIC_ASSERT((SHAPE_K / BLOCK_K) % kNumSegments == 0, "A segment must be whole K blocks");
    constexpr uint32_t kNumKBlocksPerSegment = (SHAPE_K / BLOCK_K) / kNumSegments;
    constexpr uint32_t kAccumCols = mega_gate_layout::get_num_accum_cols(kNumSegmentsPerCta);
    constexpr uint32_t kMaxBlockTokens = mega_gate_layout::get_max_block_tokens(UMMA_M, kNumMmaCtas, kNumSegmentsPerCta);

    constexpr uint32_t kNumTmemCols = utils::get_num_aligned_tmem_cols<kNumEpilogueStages * kNumSegmentsPerCta * kAccumCols>();
    DG_STATIC_ASSERT(kNumTmemCols <= 512, "Segment accumulators exceed tensor memory");

    constexpr uint32_t kNumWeightStageBytes = LOAD_BLOCK_N * BLOCK_K * static_cast<uint32_t>(sizeof(cutlass::bfloat16_t));

    const auto mma_cta_rank = cute::block_rank_in_cluster();
    const auto mma_cluster_idx = blockIdx.x / kNumMmaCtas;
    const auto slice_idx = mma_cluster_idx % (kNumExpertGroups * kNumSplitK);
    const auto expert_group_idx = slice_idx % kNumExpertGroups;
    const auto split_k_idx = slice_idx / kNumExpertGroups;
    const auto logical_cta_rank = slice_idx * kNumMmaCtas + mma_cta_rank;
    constexpr uint32_t kNumKBlocksPerSlice = kNumKBlocksPerSegment * kNumSegmentsPerCta;
    const auto split_k_offset = split_k_idx * kNumKBlocksPerSlice * BLOCK_K;
    const auto is_leader_cta = mma_cta_rank == 0;
    const auto warp_idx = cutlass::canonical_warp_idx_sync();

    if (warp_idx == 0) {
        cute::prefetch_tma_descriptor(&tensor_map_x);
        cute::prefetch_tma_descriptor(&tensor_map_weight);
    }

    extern __shared__ __align__(mega_gate_layout::kSharedMemoryAlignment) uint8_t smem_buffer[];
    auto& smem = *reinterpret_cast<SharedStorage*>(smem_buffer);

    if constexpr (kNumMmaCtas > 1)
        comm::cluster_sync_with_relaxed_arrive();

    // Initialize barriers
    if (warp_idx < 4 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = warp_idx; i < mega_gate_layout::kNumMaxStages; i += 4) {
            smem.full_barriers[i].init(kNumMmaCtas);
            smem.empty_barriers[i].init(1);
        }
        if (warp_idx == 0) {
            #pragma unroll
            for (uint32_t i = 0; i < kNumEpilogueStages; ++ i) {
                smem.tmem_full_barriers[i].init(1);
                smem.tmem_empty_barriers[i].init(kNumMmaCtas);
            }
        }

        cutlass::arch::fence_barrier_init();
    }
    __syncwarp();

    if (warp_idx == 2)
        Allocator().allocate(kNumTmemCols, &smem.tmem_ptr);
    kNumMmaCtas > 1 ? comm::cluster_sync_with_relaxed_arrive() : __syncthreads();

    cudaGridDependencySynchronize();

    using Workspace = mega_gate_layout::Workspace<kNumSplitK, kNumRoutedExperts>;
    const Workspace workspace(gmem_scratch, gmem_score_barriers);
    const auto get_effective_umma_n = [&](const uint32_t& token_block_idx) {
        return math::align(cute::min(block_tokens, num_tokens - token_block_idx * block_tokens), 16u);
    };
    // Persistent schedule: the `kNumLogicalCtas` CTAs of a worker group stride over the token blocks together
    const auto worker_group_idx = blockIdx.x / kNumLogicalCtas;

    DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(&smem.tmem_ptr) == 0);
    DG_TRAP_ONLY_DEVICE_ASSERT(num_stages <= mega_gate_layout::kNumMaxStages);
    DG_TRAP_ONLY_DEVICE_ASSERT(block_tokens <= kMaxBlockTokens and block_tokens % 16 == 0);

    // Dispatch warps into different roles
    if (warp_idx == 0 and cute::elect_one_sync()) {
        // TMA load warp
        const auto num_x_stage_bytes = (block_tokens / kNumMmaCtas) * BLOCK_K * static_cast<uint32_t>(sizeof(cutlass::bfloat16_t));
        // A CTA pair splits both the experts (UMMA M) and the tokens (UMMA N) between its two CTAs
        const auto expert_base_idx = expert_group_idx * kNumExpertsPerGroup + mma_cta_rank * LOAD_BLOCK_N;
        uint32_t stage_idx = 0, phase = 0;
        for (auto token_block_idx = worker_group_idx; token_block_idx < num_token_blocks; token_block_idx += num_worker_groups) {
            const auto effective_umma_n = get_effective_umma_n(token_block_idx);
            const auto x_token_idx = token_block_idx * block_tokens + mma_cta_rank * (effective_umma_n / kNumMmaCtas);
            #pragma unroll 4
            for (uint32_t k_block_idx = 0; k_block_idx < kNumKBlocksPerSlice; ++ k_block_idx) {
                smem.empty_barriers[stage_idx].wait(phase ^ 1);

                // Issue TMAs
                const auto k_idx = split_k_offset + k_block_idx * BLOCK_K;
                tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleMode, cutlass::bfloat16_t, false,
                          static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_FIRST)>(
                    &tensor_map_x, &smem.full_barriers[stage_idx],
                    mega_gate_layout::get_x_stage_ptr(smem_buffer, stage_idx, num_x_stage_bytes),
                    k_idx, x_token_idx, kNumMmaCtas);
                tma::copy<BLOCK_K, LOAD_BLOCK_N, kSwizzleMode, cutlass::bfloat16_t>(
                    &tensor_map_weight, &smem.full_barriers[stage_idx],
                    mega_gate_layout::get_weight_stage_ptr(smem_buffer, stage_idx, num_stages, num_x_stage_bytes, kNumWeightStageBytes),
                    k_idx, expert_base_idx, kNumMmaCtas);

                if (is_leader_cta) {
                    smem.full_barriers[stage_idx].arrive_and_expect_tx((num_x_stage_bytes + kNumWeightStageBytes) * kNumMmaCtas);
                } else {
                    smem.full_barriers[stage_idx].arrive(0u);
                }
                if (++ stage_idx == num_stages)
                    stage_idx = 0, phase ^= 1;
            }
        }
    } else if (warp_idx == 1 and is_leader_cta) {
        // MMA issue warp
        auto instr_desc = cute::UMMA::make_instr_desc<cutlass::bfloat16_t, cutlass::bfloat16_t, float,
                                                      UMMA_M, UMMA_N, cute::UMMA::Major::K, cute::UMMA::Major::K>();
        const auto num_x_stage_bytes = (block_tokens / kNumMmaCtas) * BLOCK_K * static_cast<uint32_t>(sizeof(cutlass::bfloat16_t));
        auto x_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, LOAD_BLOCK_M, BLOCK_K, kSwizzleMode>(
            mega_gate_layout::get_x_stage_ptr(smem_buffer, 0, num_x_stage_bytes), 0, 0);
        auto weight_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, LOAD_BLOCK_N, BLOCK_K, kSwizzleMode>(
            mega_gate_layout::get_weight_stage_ptr(smem_buffer, 0, num_stages, num_x_stage_bytes, kNumWeightStageBytes), 0, 0);

        const auto x_desc_lo = x_desc.lo, weight_desc_lo = weight_desc.lo;
        auto x_desc_base_lo = x_desc_lo, weight_desc_base_lo = weight_desc_lo;

        uint32_t stage_idx = 0, phase = 0, iter_idx = 0;
        for (auto token_block_idx = worker_group_idx; token_block_idx < num_token_blocks;
             token_block_idx += num_worker_groups, ++ iter_idx) {
            mma::sm100::update_instr_desc_with_umma_n(instr_desc, get_effective_umma_n(token_block_idx));

            const auto accum_stage_idx = iter_idx % kNumEpilogueStages;
            const auto accum_phase_idx = iter_idx / kNumEpilogueStages & 1;
            smem.tmem_empty_barriers[accum_stage_idx].wait(accum_phase_idx ^ 1);
            ptx::tcgen05_after_thread_sync();

            auto umma_arrive = [](const uint64_t* barrier) {
                if constexpr (kNumMmaCtas == 1) {
                    cutlass::arch::umma_arrive(barrier);
                } else {
                    constexpr uint16_t kCTAMask = (1 << kNumMmaCtas) - 1;
                    cutlass::arch::umma_arrive_multicast_2x1SM(barrier, kCTAMask);
                }
            };
            auto empty_barrier_arrive = [&](const bool& do_tmem_full_arrive) {
                umma_arrive(reinterpret_cast<uint64_t*>(&smem.empty_barriers[stage_idx]));

                if (do_tmem_full_arrive)
                    umma_arrive(reinterpret_cast<uint64_t*>(&smem.tmem_full_barriers[accum_stage_idx]));
                __syncwarp();
            };

            for (uint32_t k_block_idx = 0; k_block_idx < kNumKBlocksPerSlice; ++ k_block_idx) {
                smem.full_barriers[stage_idx].wait(phase);
                ptx::tcgen05_after_thread_sync();
                const auto runtime_instr_desc = cute::UMMA::make_runtime_instr_desc(instr_desc);
                // IsoExec: the segment of this K block owns its accumulator; its chain restarts at the segment's first
                // block (accumulate flag off on its first UMMA) and runs in ascending K, never split or re-associated
                const auto segment_idx = k_block_idx / kNumKBlocksPerSegment;
                const auto segment_k_block_idx = k_block_idx - segment_idx * kNumKBlocksPerSegment;
                const auto accum_col = (accum_stage_idx * kNumSegmentsPerCta + segment_idx) * kAccumCols;

                // Issue UMMA
                if (cute::elect_one_sync()) {
                    #pragma unroll
                    for (uint32_t umma_k_idx = 0; umma_k_idx < BLOCK_K / UMMA_K; ++ umma_k_idx) {
                        using mma_t = cute::conditional_t<kNumMmaCtas == 1,
                                                          ptx::SM100_MMA_F16BF16_SS, ptx::SM100_MMA_F16BF16_2x1SM_SS>;
                        x_desc.lo = mma::sm100::advance_umma_desc_lo<cute::UMMA::Major::K, LOAD_BLOCK_M, kSwizzleMode, cutlass::bfloat16_t>(
                                        x_desc_base_lo, 0, umma_k_idx * UMMA_K);
                        weight_desc.lo = mma::sm100::advance_umma_desc_lo<cute::UMMA::Major::K, LOAD_BLOCK_N, kSwizzleMode, cutlass::bfloat16_t>(
                                            weight_desc_base_lo, 0, umma_k_idx * UMMA_K);
                        // NOTES: the weights are the A operand, so a TMEM lane holds one expert and a column one token
                        mma_t::fma(weight_desc, x_desc, accum_col,
                                   umma_k_idx > 0 or segment_k_block_idx > 0, runtime_instr_desc);
                    }
                }
                __syncwarp();

                empty_barrier_arrive(k_block_idx == kNumKBlocksPerSlice - 1);
                if (++ stage_idx == num_stages) {
                    stage_idx = 0, phase ^= 1;
                    x_desc_base_lo = x_desc_lo, weight_desc_base_lo = weight_desc_lo;
                } else {
                    x_desc_base_lo += num_x_stage_bytes / 16, weight_desc_base_lo += kNumWeightStageBytes / 16;
                }
            }
        }

        // To safely deconstruct barriers, we need another round of waits
        // The follower's remote arrivals are not ordered by the relaxed tail cluster sync
        if constexpr (kNumMmaCtas > 1) {
            const auto last_iter_idx = iter_idx - 1;
            smem.tmem_empty_barriers[last_iter_idx % kNumEpilogueStages].wait(last_iter_idx / kNumEpilogueStages & 1);
        }
    } else if (warp_idx >= kNumNonEpilogueWarps) {
        // Gate warps
        const auto gate_warp_idx = warp_idx - kNumNonEpilogueWarps;
        // IsoExec: the router's selection arguments (`rt100::SelectArgs`), the same for every token
        const rt100::SelectArgs select_args{bias, mask,
                                            static_cast<int>(num_routed_experts), static_cast<int>(num_groups),
                                            static_cast<int>(num_topk_groups), static_cast<int>(num_shared_experts),
                                            routed_scaling_factor, out_idx, out_w};
        for (auto token_block_idx = worker_group_idx, iter_idx = 0u; token_block_idx < num_token_blocks;
             token_block_idx += num_worker_groups, ++ iter_idx) {
            const auto token_base_idx = token_block_idx * block_tokens;
            const auto token_end_idx = cute::min(token_base_idx + block_tokens, num_tokens);
            const auto effective_umma_n = get_effective_umma_n(token_block_idx);
            const auto accum_stage_idx = iter_idx % kNumEpilogueStages;
            const auto accum_phase_idx = (iter_idx / kNumEpilogueStages) & 1;
            const auto score_barrier_ptr = workspace.get_score_barrier_ptr(token_block_idx);

            // The first logical CTA re-arms the score barrier of this block for the current grid
            if (gate_warp_idx == kNumGateWarps - 1 and kNumLogicalCtas > 1 and cute::elect_one_sync()) {
                if (logical_cta_rank == 0)
                    ScoreBarrier::init(score_barrier_ptr);
                else
                    ScoreBarrier::wait_init(score_barrier_ptr);
            }

            __syncwarp();
            smem.tmem_full_barriers[accum_stage_idx].wait(accum_phase_idx);
            ptx::tcgen05_after_thread_sync();

            // IsoExec: the FP32 partial logits of this split (its segments folded by the segment subtree), no scoring
            const auto expert_group_base_idx = expert_group_idx * kNumExpertsPerGroup;
            epilogue::mega_gate::sm100_store_tmem_partials_to_global<
                kNumSegmentsPerCta, kAccumCols, Workspace::kTokenStride, UMMA_M, kNumMmaCtas, kNumGateWarps>(
                    workspace.get_score_ptr(token_base_idx, split_k_idx),
                    accum_stage_idx * kNumSegmentsPerCta * kAccumCols,
                    effective_umma_n, expert_group_base_idx,
                    mma_cta_rank, gate_warp_idx
                );
            cutlass::arch::NamedBarrier::sync(kNumGateThreads, 0);
            if (gate_warp_idx == kNumGateWarps - 1 and cute::elect_one_sync()) {
                ptx::tcgen05_before_thread_sync();
                smem.tmem_empty_barriers[accum_stage_idx].arrive(0u);
                if constexpr (kNumLogicalCtas > 1)
                    ScoreBarrier::arrive(score_barrier_ptr);
            }
            if (gate_warp_idx == 0 and kNumLogicalCtas > 1 and cute::elect_one_sync())
                ScoreBarrier::wait(score_barrier_ptr);
            cutlass::arch::NamedBarrier::sync(kNumGateThreads, 0);

            // Every logical CTA of the block ranks a share of its tokens once all the splits are visible.
            // IsoExec: each token is the router's `select_tree` program over the split partials (one warp per token)
            for (uint32_t global_token_idx = token_base_idx + gate_warp_idx * kNumLogicalCtas + logical_cta_rank;
                 global_token_idx < token_end_idx;
                 global_token_idx += kNumGateWarps * kNumLogicalCtas) {
                epilogue::mega_gate::select_token<static_cast<int>(kSelectMode), static_cast<int>(kNumTopk),
                                                  static_cast<int>(kNumExpertsPerLane), kNumSplitK, kNumRoutedExperts>(
                    workspace.get_score_ptr(global_token_idx), static_cast<int64_t>(global_token_idx), select_args);
            }
        }
    }

    // TODO: Remove redundant synchronization
    kNumMmaCtas > 1 ? comm::cluster_sync_with_relaxed_arrive() : __syncthreads();

    if (warp_idx == 0)
        Allocator().free(0, kNumTmemCols);
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_100f");
#endif
}

} // namespace deep_gemm

#pragma clang diagnostic pop
