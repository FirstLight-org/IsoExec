#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>

#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/scheduler/gemm.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/epilogue/sm100_store_cd.cuh>
#include <deep_gemm/epilogue/sm100_store_cd_swap_ab.cuh>
#include <deep_gemm/layout/gemm.cuh>
#include <deep_gemm/epilogue/operators.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm {

template <cute::UMMA::Major kMajorA, cute::UMMA::Major kMajorB,
          uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K_,
          uint32_t kNumGroups,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode, uint32_t kSwizzleCDMode,
          uint32_t kNumStages_,
          uint32_t kNumNonEpilogueThreads, uint32_t kNumEpilogueThreads,
          uint32_t kNumMulticast, bool kIsMulticastOnA,
          uint32_t kNumSMs,
          uint32_t kKAlignment,
          bool kSwapAB, bool kEnsureZeroPadding,
          GemmType kGemmType, bool kWithAccumulation,
          typename cd_dtype_t, typename epilogue_operator_t,
          uint64_t kTensorCoreUtilControl,
          uint32_t kNumLeaves = 1, uint32_t kEarlyLaunch = 0>
CUTLASS_GLOBAL void __launch_bounds__(kNumNonEpilogueThreads + kNumEpilogueThreads, 1)
sm100_bf16_gemm_impl(int* grouped_layout,
                     uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                     const __grid_constant__ EpilogueOperatorArgs epilogue_args,
                     const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                     const __grid_constant__ cute::TmaDescriptor tensor_map_b,
                     const __grid_constant__ cute::TmaDescriptor tensor_map_cd) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    // Enlarge `BLOCK_K` for some cases
    // NOTES: this is for reducing the `umma_arrive()` overhead
    // IsoExec leaf tree (``kNumLeaves`` > 1): K is cut into kNumLeaves equal leaves, each ONE ascending chain from a
    // non-accumulating first UMMA into an FP32 accumulator; the leaves are folded by the balanced adjacent-pair tree
    // (one __fadd_rn a node, lower index on the left) before the store. Progressive fold: leaf l accumulates in slot
    // l % 2; when an odd leaf completes, the epilogue forms the pair node and carries it through a binary counter of
    // partials (only equal-level nodes merge, the lower-index node on the left: each value formed is a node of the
    // tree, so the root is bit for bit the all-at-once fold), freeing both slots for the next pair. A merged stage
    // must not straddle a leaf boundary.
    constexpr bool kDoMergeStages =
        kNumStages_ >= 8 and kGemmType == GemmType::Normal and
        kMajorA == cute::UMMA::Major::K and kMajorB == cute::UMMA::Major::K and
        (kNumLeaves == 1 or (SHAPE_K != 0 and (SHAPE_K / kNumLeaves) % (BLOCK_K_ * (kNumStages_ / 8)) == 0));
    // Ensure there are at least `kNumMinStages` stages after merge
    constexpr uint32_t kNumMinStages = 8;
    constexpr uint32_t kNumStagesPerMerge = kDoMergeStages ? kNumStages_ / kNumMinStages : 1;
    constexpr uint32_t BLOCK_K = BLOCK_K_ * kNumStagesPerMerge;
    constexpr uint32_t kNumStages = kNumStages_ / kNumStagesPerMerge;

    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using Allocator = cute::conditional_t<kNumMulticast == 1, cute::TMEM::Allocator1Sm, cute::TMEM::Allocator2Sm>;

    // C/D type: BF16 and FP32 are supported, with or without accumulation
    DG_STATIC_ASSERT(cute::is_same_v<cd_dtype_t, float> or cute::is_same_v<cd_dtype_t, cutlass::bfloat16_t>, "Invalid C/D data dtype");

    // MMA Configs
    constexpr uint32_t LAYOUT_AD_M = 128;
    constexpr uint32_t UMMA_M = LAYOUT_AD_M * kNumMulticast;
    constexpr uint32_t UMMA_N = kSwapAB ? BLOCK_M : BLOCK_N;
    constexpr uint32_t UMMA_K = 16;
    constexpr uint32_t LOAD_BLOCK_M = BLOCK_M / (kIsMulticastOnA ? kNumMulticast: 1);
    constexpr uint32_t LOAD_BLOCK_N = BLOCK_N / (kIsMulticastOnA ? 1 : kNumMulticast);
    DG_STATIC_ASSERT(BLOCK_K_ == 64, "Invalid block K");
    DG_STATIC_ASSERT(BLOCK_K % UMMA_K == 0, "Block K must be divisible by UMMA K");
    DG_STATIC_ASSERT(not is_k_grouped_contiguous(kGemmType) or kKAlignment % BLOCK_K == 0, "K alignment must be divisible by block K");
    DG_STATIC_ASSERT(kNumMulticast == 1 or kNumMulticast == 2, "Only support 1/2 multicast");
    // IsoExec padded swap-AB tile (kPadB): 64 weight rows per CTA, the MMA kept at UMMA M = 128 (cta_group::1). The B
    // stage is allocated for 128 rows and loaded with 64 (K-major, no multicast); MMA rows 64-127 read unused shared
    // memory into tensor-memory lanes the store never writes out. Each output element's chain is unchanged.
    constexpr bool kPadB = kSwapAB and BLOCK_N == 64;
    DG_STATIC_ASSERT(not kPadB or (kNumMulticast == 1 and kMajorB == cute::UMMA::Major::K), "Padded swap-AB tile: 1-CTA, K-major B");
    constexpr uint32_t PHYS_BLOCK_N = kPadB ? LAYOUT_AD_M : BLOCK_N;
    DG_STATIC_ASSERT((kSwapAB and (BLOCK_N == LAYOUT_AD_M or kPadB)) or
                     (not kSwapAB and (BLOCK_M == 32 or BLOCK_M == 64 or BLOCK_M == LAYOUT_AD_M)), "Invalid block size");

    // Epilogue configs
    // Always enable pipeline for better performance (one accumulator set when kNumLeaves sets do not fit twice)
    DG_STATIC_ASSERT(kNumLeaves == 1 or (not kSwapAB and kGemmType == GemmType::Normal and
                                         (kNumLeaves & (kNumLeaves - 1)) == 0 and kNumLeaves <= 8),
                     "Leaf trees: normal non-swapped GEMMs, a power-of-two leaf count <= 8");
    // Progressive fold (IsoExec): two leaf slots (leaf l writes slot l % 2) plus log2(L / 2) binary-counter partials
    constexpr uint32_t kNumLeafPairs = kNumLeaves == 1 ? 1 : kNumLeaves / 2;
    constexpr uint32_t kNumLeafPartials = (kNumLeaves >= 4) + (kNumLeaves >= 8);
    // IsoExec register partials (leaf trees launched with two epilogue warpgroups, 256 threads): the counter's
    // partials live in the epilogue threads' registers (sm100_fold_leaf_pair_regs / sm100_store_cd_leaf_regs), so
    // tensor memory holds the leaf ring only
    constexpr bool kRegPartials = kNumLeaves > 1 and not kSwapAB and kNumEpilogueThreads == 256;
    DG_STATIC_ASSERT(kNumEpilogueThreads == 128 or (kRegPartials and kGemmType == GemmType::Normal),
                     "Two epilogue warpgroups: leaf trees (register partials) only");
    constexpr uint32_t kNumTmemPartials = kRegPartials ? 0 : kNumLeafPartials;
    // A third ring slot (when it fits) lets the next leaf's chain run while a pair is folded: leaf l -> slot l % R
    constexpr uint32_t kLeafRing = kNumLeaves >= 4 and (3 + kNumTmemPartials) * (kSwapAB ? BLOCK_M : BLOCK_N) <= 512 ? 3 : 2;
    constexpr uint32_t kNumLeafSlots = kNumLeaves == 1 ? 1 : kLeafRing + kNumTmemPartials;
    constexpr uint32_t kNumEpilogueStages = 2 * kNumLeafSlots * (kSwapAB ? BLOCK_M : BLOCK_N) <= 512 ? 2 : 1;
    constexpr uint32_t kNumTMAStoreStages = 2;
    // NOTES: To maximize epilogue threads utilization, process an entire BLOCK_N
    //        per store stage for swap-AB cases, and an entire BLOCK_M for non-swap cases
    constexpr uint32_t STORE_BLOCK_M =        kSwapAB ? 16      : cute::min<uint32_t>(BLOCK_M, LAYOUT_AD_M);
    constexpr uint32_t STORE_BLOCK_N =        kSwapAB ? PHYS_BLOCK_N : kSwizzleCDMode / sizeof(cd_dtype_t);
    constexpr uint32_t kNumUMMAStoreThreads = (kSwapAB or kRegPartials) ? kNumEpilogueThreads: STORE_BLOCK_M;
    DG_STATIC_ASSERT(kNumUMMAStoreThreads % 32 == 0, "Invalid store block M");

    // NOTES: Make sure we have enough shared memory for UMMA padding
    static constexpr uint32_t UMMA_A_SIZE_PER_STAGE = math::constexpr_align(LOAD_BLOCK_M, LAYOUT_AD_M) * BLOCK_K * sizeof(nv_bfloat16);

    // Real tensor memory size and offsets
    constexpr uint32_t kNumAccumTmemCols = kNumEpilogueStages * UMMA_N * kNumLeafSlots;
    constexpr uint32_t kNumTmemCols = utils::get_num_aligned_tmem_cols<kNumAccumTmemCols>();
    DG_STATIC_ASSERT(32 <= kNumTmemCols and kNumTmemCols <= 512, "Invalid tensor memory columns");

    // Synchronize the cluster before 2-CTA TMEM allocation
    kNumMulticast > 1 ? comm::cluster_sync_with_relaxed_arrive() : void();

    // Utils
    bool is_leader_cta = cute::block_rank_in_cluster() == 0;
    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    const auto lane_idx = ptx::get_lane_idx();
    const auto epilogue_warp_idx = warp_idx - kNumNonEpilogueThreads / 32;

    // Prefetch TMA descriptors at the very beginning
    if (warp_idx == 0) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
        cute::prefetch_tma_descriptor(&tensor_map_cd);
    }

    // Overwrite shape constants if the compiler gives
    shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;

    using SharedStorage = layout::SM100BF16GemmSharedStorage<
        kNumStages, kNumEpilogueStages, kNumTMAStoreStages, LOAD_BLOCK_M, (kPadB ? PHYS_BLOCK_N : LOAD_BLOCK_N), BLOCK_K,
        STORE_BLOCK_M, STORE_BLOCK_N, cd_dtype_t>;
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    auto& smem = *reinterpret_cast<SharedStorage*>(smem_buffer);
    DG_STATIC_ASSERT(UMMA_A_SIZE_PER_STAGE <= sizeof(smem.a[0]) + sizeof(smem.b), "Memory out of bound for UMMA");

    // Initialize barriers
    if (warp_idx == 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            // Arrive only at the leader CTA
            smem.full_barriers[i].init(kNumMulticast);
            // Arrive at all CTAs
            smem.empty_barriers[i].init(1);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumEpilogueStages; ++ i) {
            // Arrive at all CTAs
            smem.tmem_full_barriers[i].init(1);
            // Arrive only at the leader CTA
            smem.tmem_empty_barriers[i].init(kNumMulticast * kNumUMMAStoreThreads);
        }
        if constexpr (kTensorCoreUtilControl < 100)
            smem.tensor_core_full_barrier.init(1);

        // Make initialized barrier visible in async proxy
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 2) {
        // Allocate tensor memory
        Allocator().allocate(kNumTmemCols, &smem.tmem_ptr);
    }
    kNumMulticast > 1 ? comm::cluster_sync_with_relaxed_arrive() : __syncthreads();

    // NOTES: BF16 has no SF, so `kSFKSpan` is unused here; pass `kKAlignment` explicitly to avoid relying on the default.
    using scheduler_t = sched::Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumGroups, kNumMulticast, kIsMulticastOnA, kNumSMs, kEnsureZeroPadding, kKAlignment, kKAlignment>;

    // B (weight) tile loads of one stage: the same TMA copies for the pre-wait loads and the main loop
    auto load_b = [&](scheduler_t& s, const uint32_t& stage, const uint32_t& k_block_idx,
                      const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
        uint32_t n_idx = s.template get_global_idx<(kMajorB == cute::UMMA::Major::K), sched::IndexType::MN> (
            shape_n, BLOCK_N, n_block_idx, m_block_idx);
        uint32_t k_b_idx = s.template get_global_idx<(kMajorB == cute::UMMA::Major::MN), sched::IndexType::K> (
            shape_k, BLOCK_K, k_block_idx, m_block_idx);
        if constexpr (kNumMulticast > 1)
            n_idx += kIsMulticastOnA ? 0 : (cute::block_rank_in_cluster() * LOAD_BLOCK_N);
        constexpr bool kIsBatchedMM = (kGemmType == GemmType::Batched);
        const uint32_t batch_idx = (kIsBatchedMM ? s.current_group_idx : 0);
        if constexpr (kMajorB == cute::UMMA::Major::K)
            tma::copy<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, cutlass::bfloat16_t, kIsBatchedMM>(
                &tensor_map_b, &smem.full_barriers[stage], smem.b[stage], k_b_idx, n_idx, kNumMulticast, batch_idx);
        if constexpr (kMajorB == cute::UMMA::Major::MN)
            tma::copy<LOAD_BLOCK_N, BLOCK_K, kSwizzleBMode, cutlass::bfloat16_t, kIsBatchedMM>(
                &tensor_map_b, &smem.full_barriers[stage], smem.b[stage], n_idx, k_b_idx, kNumMulticast, batch_idx);
    };

    // IsoExec early launch (kEarlyLaunch 1: an early dependent trigger only; k >= 2: also the B stages of the CTA's first
    // tile, at most k - 1 of them, before the dependency wait). B is a weight the caller guarantees no predecessor
    // kernel writes; A (the activation) and every store stay after the wait. Each stage's full barrier still completes
    // only with the A-side arrive, which carries the full A + B byte count as before, so the MMAs, their operands and
    // their order are unchanged.
    // kEarlyLaunch bit 6 (64): the lean post-wait path (see below); bit 7 (128): the lean tail (see the end);
    // bits 0-5: the early mode
    constexpr bool kLeanWait = (kEarlyLaunch & 64u) != 0;
    constexpr bool kLeanTail = (kEarlyLaunch & 128u) != 0;
    constexpr uint32_t kEarlyMode = kEarlyLaunch & 63u;
    DG_STATIC_ASSERT((kEarlyLaunch & ~255u) == 0 and (not kLeanWait or kEarlyMode > 0), "Invalid early launch mode");
    constexpr bool kEarlyB = kEarlyMode >= 2 and (kGemmType == GemmType::Normal or kGemmType == GemmType::Batched);
    // Pre-loaded stages of the first tile (Normal / Batched: every tile has the same K blocks)
    const uint32_t num_early_b_stages = cute::min(cute::min(kNumStages, kEarlyMode >= 2 ? kEarlyMode - 1 : 0u),
                                                  cute::max(1u, math::ceil_div(shape_k, BLOCK_K)));
    if constexpr (kEarlyB) {
        if (warp_idx == 0 and lane_idx == 0) {
            uint32_t m_block_idx, n_block_idx;
            auto pre = scheduler_t(shape_m, shape_n, shape_k, grouped_layout);
            if (pre.get_next_block(m_block_idx, n_block_idx)) {
                for (uint32_t k_block_idx = 0; k_block_idx < num_early_b_stages; ++ k_block_idx)
                    load_b(pre, k_block_idx, k_block_idx, m_block_idx, n_block_idx);
            }
        }
        __syncwarp();
    }

    // Wait for primary kernel completion
    // IsoExec: let the dependent grid launch now (only after this grid's own wait, so a dependent's pre-wait reads
    // never overlap a kernel before this one). A dependent waits for this grid's completion and memory flush before
    // it reads anything this grid writes; a co-resident dependent CTA may hold tensor memory, so every tensor-memory
    // address below is relative to this CTA's allocated base.
    // Lean post-wait path (kLeanWait, IsoExec, measured round 33: 0.35 us from the wait's release to the first TMA
    // issue on every call): the TMEM base, the scheduler and the epilogue operator (pure functions of shared memory
    // written before the cluster sync, blockIdx and the shape) are formed BEFORE the wait, and the TMA producer thread
    // issues its first stage before it triggers (every other thread triggers right after the wait). No global memory
    // is read before the wait; operands, MMAs and their order are unchanged.
    const bool is_tma_producer = warp_idx == 0 and lane_idx == 0;
    if constexpr (not kLeanWait) {
        cudaGridDependencySynchronize();
        if constexpr (kEarlyMode > 0)
            cudaTriggerProgrammaticLaunchCompletion();
    }
    const uint32_t tmem_base = ptx::ld_shared(&smem.tmem_ptr);

    // Block scheduler
    uint32_t m_block_idx, n_block_idx;
    auto scheduler = scheduler_t(shape_m, shape_n, shape_k, grouped_layout);
    epilogue_operator_t epilogue_operator(epilogue_args);
    if constexpr (kLeanWait) {
        cudaGridDependencySynchronize();
        if (not is_tma_producer)
            cudaTriggerProgrammaticLaunchCompletion();
    }

    // Pipeline and TMA phases
    uint32_t stage_idx = 0, phase = 0, tensor_core_phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;

        // Flip phases only if reach the next first stage
        stage_idx = (stage_idx + 1) % kNumStages;
        phase ^= stage_idx == 0;
    };

    // Lean tail: whether this thread already arrived on the final cluster barrier (an epilogue thread, last tile)
    bool tail_arrived = false;


    // Dispatch warps into different roles (lean path: the producer is lane 0, the thread that skipped the trigger)
    if (warp_idx == 0 and (kLeanWait ? lane_idx == 0 : cute::elect_one_sync())) {
        // TMA load warp
        // Persistently schedule over blocks
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            // Use dynamic load block M, when swap-AB is enabled
            const auto load_block_m = kSwapAB ? scheduler.get_aligned_effective_m_in_block(m_block_idx) / kNumMulticast : LOAD_BLOCK_M;

            // For k-grouped layout, the number of block K is variable
            const auto num_total_k_blocks = cute::max(1u, math::ceil_div(scheduler.current_shape_k, BLOCK_K));
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                // Wait consumer release
                smem.empty_barriers[stage_idx].wait(phase ^ 1);

                // Compute offsets
                // NOTES: the group is always concatenated with the outer dimension
                uint32_t m_idx = scheduler.template get_global_idx<(kGemmType == GemmType::MGroupedMasked), sched::IndexType::MN> (
                    shape_m, BLOCK_M, m_block_idx);

                // NOTES: `k_idx` is actually the k index default for K-major, while `k_b_idx` may be MN-major
                // And for all m-grouped GEMMs, A must be K-majored
                DG_STATIC_ASSERT(kGemmType == GemmType::Normal or is_k_grouped_contiguous(kGemmType) or kGemmType == GemmType::Batched or
                                 kMajorA == cute::UMMA::Major::K, "Invalid major");
                uint32_t k_idx = k_block_idx * BLOCK_K;
                uint32_t k_a_idx = scheduler.template get_global_idx<(kMajorA == cute::UMMA::Major::MN), sched::IndexType::K> (
                    shape_k, BLOCK_K, k_block_idx, m_block_idx);

                // Add 2 CTA offsets
                if constexpr (kNumMulticast > 1)
                    m_idx += kIsMulticastOnA ? (cute::block_rank_in_cluster() * load_block_m) : 0;

                // Issue TMAs
                constexpr bool kIsBatchedMM = (kGemmType == GemmType::Batched);
                const uint32_t batch_idx = (kIsBatchedMM ? scheduler.current_group_idx : 0);
                if constexpr (kMajorA == cute::UMMA::Major::K)
                    tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, cutlass::bfloat16_t, kIsBatchedMM>(
                        &tensor_map_a, &smem.full_barriers[stage_idx], smem.a[stage_idx], k_a_idx, m_idx, kNumMulticast, batch_idx);
                if constexpr (kMajorA == cute::UMMA::Major::MN)
                    tma::copy<LOAD_BLOCK_M, BLOCK_K, kSwizzleAMode, cutlass::bfloat16_t, kIsBatchedMM>(
                        &tensor_map_a, &smem.full_barriers[stage_idx], smem.a[stage_idx], m_idx, k_a_idx, kNumMulticast, batch_idx);
                // B: unless this stage of the first tile was loaded before the dependency wait
                if (not (kEarlyB and scheduler.current_iter == 0 and k_block_idx < num_early_b_stages))
                    load_b(scheduler, stage_idx, k_block_idx, m_block_idx, n_block_idx);

                // Arrive at full barriers
                // B bytes: the rows the TMA loads (a padded tile's stage is larger than its load)
                constexpr uint32_t kNumArrivalBytes = sizeof(smem.a[0]) + LOAD_BLOCK_N * BLOCK_K * sizeof(cutlass::bfloat16_t);
                DG_STATIC_ASSERT(kPadB or sizeof(smem.b[0]) == LOAD_BLOCK_N * BLOCK_K * sizeof(cutlass::bfloat16_t), "B stage size");
                if (is_leader_cta) {
                    smem.full_barriers[stage_idx].arrive_and_expect_tx(kNumArrivalBytes * kNumMulticast);
                } else {
                    smem.full_barriers[stage_idx].arrive(0u);
                }
                // Lean path: the producer triggers once its first stage is in flight
                if (kLeanWait and scheduler.current_iter == 0 and k_block_idx == 0)
                    cudaTriggerProgrammaticLaunchCompletion();
            }
        }
        if (kLeanWait and scheduler.current_iter <= 0)
            cudaTriggerProgrammaticLaunchCompletion();
    } else if (warp_idx == 1 and is_leader_cta) {
        // MMA issue warp
        // NOTES: only the leader CTA will do this
        // Make instruction descriptor
        auto instr_desc = kSwapAB ? cute::UMMA::make_instr_desc<cutlass::bfloat16_t, cutlass::bfloat16_t, float,
                                                                UMMA_M, UMMA_N, kMajorB, kMajorA>()
                                  : cute::UMMA::make_instr_desc<cutlass::bfloat16_t, cutlass::bfloat16_t, float,
                                                                UMMA_M, UMMA_N, kMajorA, kMajorB>();

        DG_STATIC_ASSERT(kNumStages <= 32, "Too many stages");
        // Merged stages only happens in NT normal GEMM cases
        constexpr uint32_t BLOCK_ATOM_K = BLOCK_K / kNumStagesPerMerge;
        auto a_desc = mma::sm100::make_umma_desc<kMajorA, LOAD_BLOCK_M, BLOCK_ATOM_K, kSwizzleAMode>(smem.a[0], 0, 0);
        auto b_desc = mma::sm100::make_umma_desc<kMajorB, LOAD_BLOCK_N, BLOCK_ATOM_K, kSwizzleBMode>(smem.b[0], 0, 0);
        uint32_t a_desc_lo = lane_idx < kNumStages ? a_desc.lo + lane_idx * sizeof(smem.a[0]) / 16 : 0u;
        uint32_t b_desc_lo = lane_idx < kNumStages ? b_desc.lo + lane_idx * sizeof(smem.b[0]) / 16 : 0u;

        // Checks for MMA instructions
        // NOTES: CUTLASS does not have such checks except the MMA traits, but we are not using these traits
        DG_STATIC_ASSERT((UMMA_M == 64  and UMMA_N %  8 == 0 and  8 <= UMMA_N and UMMA_N <= 256) or
                         (UMMA_M == 128 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256) or
                         (UMMA_M == 256 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256),
                         "Invalid MMA instruction shape");

        // Persistently schedule over blocks
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            // Wait tensor memory empty barrier arrival
            auto accum_stage_idx = scheduler.current_iter % kNumEpilogueStages;
            // Accumulator events of this stage: one per leaf pair (one per tile without leaves); event n's
            // completion is waited with parity n & 1. Before the tile: the previous tile's last event.
            const uint32_t first_event = (scheduler.current_iter / kNumEpilogueStages) * kNumLeafPairs;
            smem.tmem_empty_barriers[accum_stage_idx].wait((first_event & 1) ^ 1);
            ptx::tcgen05_after_thread_sync();

            // UMMA and empty barrier arrival alias
            auto umma_arrive = [](const uint64_t* barrier) {
                if constexpr (kNumMulticast == 1) {
                    cutlass::arch::umma_arrive(barrier);
                } else {
                    constexpr uint16_t kCTAMask = (1 << kNumMulticast) - 1;
                    cutlass::arch::umma_arrive_multicast_2x1SM(barrier, kCTAMask);
                }
            };
            auto empty_barrier_arrive = [&](const bool& do_tmem_full_arrive) {
                umma_arrive(reinterpret_cast<uint64_t*>(&smem.empty_barriers[stage_idx]));

                // NOTES: the tensor memory accumulator pipeline has nothing to do with multicasting
                if (do_tmem_full_arrive)
                    umma_arrive(reinterpret_cast<uint64_t*>(&smem.tmem_full_barriers[accum_stage_idx]));
                __syncwarp();
            };

            // Dynamic update of UMMA N based on effective M, when swap-AB is enabled
            if constexpr (kSwapAB) {
                uint32_t umma_n = scheduler.get_aligned_effective_m_in_block(m_block_idx);
                mma::sm100::update_instr_desc_with_umma_n(instr_desc, umma_n);
            }

            // Launch MMAs
            const auto num_total_k_blocks = cute::max(1u, math::ceil_div(scheduler.current_shape_k, BLOCK_K));
            const uint32_t leaf_k_blocks = kNumLeaves == 1 ? num_total_k_blocks : num_total_k_blocks / kNumLeaves;
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                // Wait TMA arrival
                smem.full_barriers[stage_idx].wait(phase);
                ptx::tcgen05_after_thread_sync();

                // The leaf this K block belongs to: its own accumulator, its chain restarts at the leaf's first block
                const uint32_t leaf_idx = kNumLeaves == 1 ? 0u : k_block_idx / leaf_k_blocks;
                const uint32_t leaf_k_block = k_block_idx - leaf_idx * leaf_k_blocks;
                // Ring slots are reused once their pair is folded. Leaf l writes slot l % R, last used by leaf l - R,
                // whose pair is (l - R) / 2: R = 2 -> even l >= 2 waits for pair l / 2 - 1; R = 3 -> odd l >= 3 waits for
                // pair (l - 3) / 2 (the other leaves' slots were freed by the same folds). Waiting for the latest pair
                // completed so far keeps the parity unambiguous.
                if (kNumLeaves > 1 and leaf_k_block == 0 and leaf_idx >= kLeafRing and
                    (leaf_idx & 1) == (kLeafRing == 2 ? 0u : 1u)) {
                    smem.tmem_empty_barriers[accum_stage_idx].wait((first_event + (leaf_idx - kLeafRing) / 2) & 1);
                    ptx::tcgen05_after_thread_sync();
                }
                const uint32_t accum_col = tmem_base + (accum_stage_idx * kNumLeafSlots + leaf_idx % kLeafRing) * UMMA_N;

                // Issue UMMA in the leader CTA
                using mma_t = cute::conditional_t<kNumMulticast == 1, ptx::SM100_MMA_F16BF16_SS, ptx::SM100_MMA_F16BF16_2x1SM_SS>;
                const auto runtime_instr_desc = cute::UMMA::make_runtime_instr_desc(instr_desc);
                const auto a_desc_base_lo = __shfl_sync(0xffffffff, a_desc_lo, static_cast<int>(stage_idx));
                const auto b_desc_base_lo = __shfl_sync(0xffffffff, b_desc_lo, static_cast<int>(stage_idx));
                if (cute::elect_one_sync()) {
                    #pragma unroll
                    for (uint32_t umma_k_idx = 0; umma_k_idx < BLOCK_K / UMMA_K; ++ umma_k_idx) {
                        const uint32_t atom_k_idx = umma_k_idx * UMMA_K / BLOCK_ATOM_K;
                        const uint32_t inner_k_idx = umma_k_idx * UMMA_K % BLOCK_ATOM_K;
                        a_desc.lo = mma::sm100::advance_umma_desc_lo<kMajorA, LOAD_BLOCK_M, kSwizzleAMode, cutlass::bfloat16_t>(
                                        a_desc_base_lo, atom_k_idx * LOAD_BLOCK_M * BLOCK_ATOM_K, inner_k_idx);
                        b_desc.lo = mma::sm100::advance_umma_desc_lo<kMajorB, LOAD_BLOCK_N, kSwizzleBMode, cutlass::bfloat16_t>(
                                        b_desc_base_lo, atom_k_idx * LOAD_BLOCK_N * BLOCK_ATOM_K, inner_k_idx);
                        if (kSwapAB) {
                            mma_t::fma(b_desc, a_desc, accum_col,
                                       umma_k_idx > 0 or leaf_k_block > 0, runtime_instr_desc);
                        } else {
                            mma_t::fma(a_desc, b_desc, accum_col,
                                       umma_k_idx > 0 or leaf_k_block > 0, runtime_instr_desc);
                        }
                    }
                }
                __syncwarp();

                // Commit to the mbarrier object
                // No explicit `tcgen05.fence::before_thread_sync` is needed, as this is implicitly performed by `tcgen05.commit`
                // Accumulator full: at the tile's end, or (leaf trees) when an odd leaf's chain completes a pair
                empty_barrier_arrive(kNumLeaves == 1 ? k_block_idx == num_total_k_blocks - 1
                                                     : (leaf_k_block == leaf_k_blocks - 1 and (leaf_idx & 1) == 1));

                // Let tensor cores relax for lower possibility of frequency drop
                DG_STATIC_ASSERT(kTensorCoreUtilControl > 0, "Invalid tensor utilization control");
                if constexpr (kTensorCoreUtilControl < 100) {
                    // For utilization control
                    umma_arrive(reinterpret_cast<uint64_t*>(&smem.tensor_core_full_barrier));
                    __syncwarp();

                    // Wait for last UMMA to be done
                    smem.tensor_core_full_barrier.wait(tensor_core_phase);
                    tensor_core_phase ^= 1;

                    // Sleep for certain cycles
                    constexpr static uint64_t kNumUMMACycles = (2ull * UMMA_M * UMMA_N * BLOCK_K) / 8192ull;
                    constexpr static uint64_t kNumDummyCycles = (100ull - kTensorCoreUtilControl) * kNumUMMACycles / kTensorCoreUtilControl;
                    const auto start_clock = clock64();
                    if (cute::elect_one_sync())
                        while (clock64() - start_clock < kNumDummyCycles) {}
                    __syncwarp();
                }
            }
        }

        // To safely deconstruct barriers, we need another round of waits
        const auto iter_idx = scheduler.current_iter - 1;
        if (kNumMulticast > 1 and iter_idx >= 0) {
            const auto last_event = (iter_idx / kNumEpilogueStages) * kNumLeafPairs + kNumLeafPairs - 1;
            smem.tmem_empty_barriers[iter_idx % kNumEpilogueStages].wait(last_event & 1);
        }
    } else if (epilogue_warp_idx < kNumUMMAStoreThreads / 32) {
        // Epilogue warp groups
        // Lean tail: the CTA's last tile arrives on the final cluster barrier from inside the store (after its last
        // tensor-memory read); a schedule is a pure function of blockIdx and the shape, so a copy tells the last tile
        auto is_last_tile = [&]() {
            if constexpr (not kLeanTail) {
                return false;
            } else {
                auto peek = scheduler;
                uint32_t m_peek, n_peek;
                return not peek.get_next_block(m_peek, n_peek);
            }
        };
        epilogue_operator.init(scheduler.num_blocks * kNumGroups, epilogue_warp_idx * 32 + lane_idx);

        // NOTES: tensor memory addresses are simplified, as the hardware will ignore the warp index bits,
        // i.e., no need for `tmem_ptr |= (epilogue_warp_idx * 32) << 16`.
        // NOTES (IsoExec): a co-resident CTA (a dependent grid launched early) may hold other tensor-memory columns;
        // every accumulator address is relative to this CTA's base `tmem_base`.

        // Share store pipeline between blocks
        uint32_t tma_stage_idx = 0;
        // Register partials of this thread's row and columns (kRegPartials only; see sm100_fold_leaf_pair_regs)
        constexpr uint32_t kRegCols = kRegPartials ? BLOCK_N / 2 : 1;
        uint32_t reg_part[kNumLeafPartials > 0 ? kNumLeafPartials : 1][kRegCols];

        // Persistently schedule over blocks
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            auto accum_stage_idx = scheduler.current_iter % kNumEpilogueStages;
            const uint32_t first_event = (scheduler.current_iter / kNumEpilogueStages) * kNumLeafPairs;
            const auto tmem_base_addr = tmem_base + accum_stage_idx * UMMA_N * kNumLeafSlots;

            // Leaf trees: fold every pair but the last as it completes (binary counter; the slots go back to the
            // MMA warp right after), the last pair in the store below
            if constexpr (kNumLeaves > 1) {
                #pragma unroll 1
                for (uint32_t pair = 0; pair + 1 < kNumLeafPairs; ++ pair) {
                    smem.tmem_full_barriers[accum_stage_idx].wait((first_event + pair) & 1);
                    ptx::tcgen05_after_thread_sync();
                    if constexpr (kRegPartials) {
                        epilogue::sm100_fold_leaf_pair_regs<kNumLeaves, UMMA_N, kLeafRing, STORE_BLOCK_N, kRegCols>(
                            tmem_base_addr, pair, epilogue_warp_idx / 4, reg_part);
                    } else {
                        epilogue::sm100_fold_leaf_pair<kNumLeaves, UMMA_N, kLeafRing>(tmem_base_addr, pair);
                    }
                    ptx::tcgen05_before_thread_sync();
                    smem.tmem_empty_barriers[accum_stage_idx].arrive(0u);
                }
            }

            // Wait UMMA arrival (the tile's last accumulator event)
            smem.tmem_full_barriers[accum_stage_idx].wait((first_event + kNumLeafPairs - 1) & 1);
            ptx::tcgen05_after_thread_sync();

            // Load from tensor memory into registers, and write shared memory with STSM
            // Whether the group offset is encoded in the flattened CD M coordinate
            constexpr bool kCDWithGroupOffset = not is_m_grouped_contiguous(kGemmType) and not is_k_grouped_contiguous(kGemmType);
            const auto base_m_idx = scheduler.template get_global_idx<
                kCDWithGroupOffset, sched::IndexType::MN>(shape_m, BLOCK_M, m_block_idx);
            const auto base_n_idx = n_block_idx * BLOCK_N;
            const bool is_empty_group = is_k_grouped_contiguous(kGemmType) and scheduler.current_shape_k == 0;

            const bool tail_arrive = is_last_tile();
            tail_arrived |= tail_arrive;
            if constexpr (kSwapAB) {
                const auto effective_m = scheduler.get_aligned_effective_m_in_block(m_block_idx);
                epilogue::sm100_store_cd_swap_ab<BLOCK_M, PHYS_BLOCK_N, STORE_BLOCK_M, STORE_BLOCK_N,
                                                 kSwizzleCDMode, kNumTMAStoreStages, kNumUMMAStoreThreads,
                                                 0,
                                                 kGemmType, kWithAccumulation,
                                                 epilogue_operator_t, SharedStorage, BLOCK_N>
                (smem, tma_stage_idx, tmem_base_addr,
                 base_m_idx, base_n_idx, scheduler.current_group_idx,
                 is_empty_group,
                 effective_m,
                 epilogue_warp_idx, lane_idx,
                 epilogue_operator,
                 false,
                 &smem.tmem_empty_barriers[accum_stage_idx],
                 &smem.tmem_empty_barriers[accum_stage_idx],
                 tensor_map_cd, tail_arrive);
            } else if constexpr (kRegPartials) {
                epilogue::sm100_store_cd_leaf_regs<BLOCK_N, STORE_BLOCK_M, STORE_BLOCK_N, kSwizzleCDMode,
                                                   kNumTMAStoreStages, kNumUMMAStoreThreads, kGemmType,
                                                   kWithAccumulation, epilogue_operator_t, SharedStorage,
                                                   kNumLeaves, UMMA_N, kLeafRing, kRegCols>
                (smem, tma_stage_idx, tmem_base_addr, base_m_idx, base_n_idx, is_empty_group,
                 epilogue_warp_idx, lane_idx, epilogue_operator,
                 &smem.tmem_empty_barriers[accum_stage_idx], tensor_map_cd, reg_part, tail_arrive);
            } else {
                epilogue::sm100_store_cd<BLOCK_M, BLOCK_N, STORE_BLOCK_M, STORE_BLOCK_N,
                                         kSwizzleCDMode, kNumTMAStoreStages, kNumUMMAStoreThreads,
                                         0,
                                         kGemmType, kWithAccumulation,
                                         epilogue_operator_t, SharedStorage, kNumLeaves, UMMA_N, kLeafRing>
                (smem, tma_stage_idx, tmem_base_addr,
                 base_m_idx, base_n_idx, scheduler.current_group_idx,
                 is_empty_group,
                 epilogue_warp_idx, lane_idx,
                 epilogue_operator,
                 false,
                 &smem.tmem_empty_barriers[accum_stage_idx],
                 &smem.tmem_empty_barriers[accum_stage_idx],
                 tensor_map_cd, tail_arrive);
            }
        }
    }

    // TODO: Remove redundant synchronization
    if constexpr (kLeanTail) {
        // IsoExec lean tail: one split-phase cluster barrier (an implicit one-CTA cluster when not clustered). The
        // epilogue threads arrived right after their last tensor-memory read (inside the store), every other thread
        // arrives here; tensor memory is freed as soon as all have arrived, overlapping the epilogue's last smem
        // writes and TMA store issue instead of following them. Remote barrier traffic (the MMA warp's final wait,
        // the epilogue's leader-CTA arrives) completes before each thread's arrive, as before the full sync.
        if (not tail_arrived)
            asm volatile("barrier.cluster.arrive;\n" : : : "memory");
        asm volatile("barrier.cluster.wait;\n" : : : "memory");
        if (warp_idx == 0) {
            __syncwarp();
            ptx::tcgen05_after_thread_sync();
            Allocator().free(tmem_base, kNumTmemCols);
        }
    } else {
        kNumMulticast > 1 ? comm::cluster_sync_with_relaxed_arrive() : __syncthreads();

        // Deallocate tensor memory
        if (warp_idx == 0)
            Allocator().free(tmem_base, kNumTmemCols);
    }

    // Finalize the epilogue while the other warp deallocates tensor memory.
    if (epilogue_warp_idx < kNumUMMAStoreThreads / 32)
        epilogue_operator.finalize();

#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_100f");
#endif
}

};  // namespace deep_gemm

#pragma clang diagnostic pop
