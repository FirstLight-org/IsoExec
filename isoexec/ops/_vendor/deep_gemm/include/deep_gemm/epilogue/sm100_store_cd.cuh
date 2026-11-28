#pragma once

#include <cute/atom/copy_traits_sm100.hpp>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/epilogue/operators.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>

namespace deep_gemm::epilogue {

template <uint32_t BLOCK_M, uint32_t BLOCK_N,
          uint32_t STORE_BLOCK_M, uint32_t STORE_BLOCK_N,
          uint32_t kSwizzleCDMode,
          uint32_t kNumTMAStoreStages,
          uint32_t kNumUMMAStoreThreads,
          uint32_t kNumOverlappedTmemCols,
          GemmType kGemmType, bool kWithAccumulation,
          typename epilogue_operator_t,
          typename smem_t,
          uint32_t kNumLeaves = 1, uint32_t kLeafTmemCols = 0, uint32_t kLeafRing = 2>
CUTLASS_DEVICE void
sm100_store_cd(smem_t& smem, uint32_t& tma_stage_idx,
               const uint32_t& tmem_base_addr,
               const uint32_t& base_m_idx, const uint32_t& base_n_idx, const uint32_t& batch_idx,
               const bool& is_empty_group,
               const uint32_t& epilogue_warp_idx, const uint32_t& lane_idx,
               epilogue_operator_t& epilogue_operator,
               const bool& reverse_store_order,
               const cutlass::arch::ClusterTransactionBarrier* tmem_overlap_barrier,
               const cutlass::arch::ClusterTransactionBarrier* tmem_empty_barrier,
               const cute::TmaDescriptor& tensor_map_cd,
               const bool& tail_arrive = false) {
    using cd_dtype_t = typename smem_t::cd_dtype;
    // Whether to cast D into FP8 with dynamic per-`kSFGranN` UE8M0 SFs (see `CastToFP8WithClamp`)
    constexpr bool kIsFP8CastWithClamp = epilogue_operator_t::kIsFP8CastWithClamp;
    constexpr uint32_t kSFGranN = operators::CastToFP8WithClamp::kSFGranN;
    // TMA checks
    constexpr uint32_t kNumBankGroupBytes = 16;
    constexpr uint32_t kNumElemsPerBankGroup = kNumBankGroupBytes / sizeof(cd_dtype_t);
    DG_STATIC_ASSERT(kSwizzleCDMode > 0, "TMA D must be swizzled");
    DG_STATIC_ASSERT(STORE_BLOCK_N % kNumElemsPerBankGroup == 0, "Invalid swizzling");
    DG_STATIC_ASSERT(BLOCK_M % STORE_BLOCK_M == 0, "Invalid block sizes");
    DG_STATIC_ASSERT(BLOCK_N % STORE_BLOCK_N == 0, "Invalid block sizes");
    DG_STATIC_ASSERT(not kIsFP8CastWithClamp or STORE_BLOCK_N % kSFGranN == 0, "A store must cover complete SF groups");
    DG_STATIC_ASSERT(not epilogue_operator_t::kWithStochasticRounding or
                     (cute::is_same_v<cd_dtype_t, cutlass::bfloat16_t> and not kWithAccumulation),
                     "Stochastic rounding only supports direct BF16 output");

    // Iterate over M waves
    constexpr auto kNumMWaves = BLOCK_M / STORE_BLOCK_M;
    DG_STATIC_ASSERT(kNumLeaves == 1 or kNumMWaves == 1, "Leaf trees: one M wave per accumulator");
    #pragma unroll
    for (uint32_t w = 0; w < kNumMWaves; ++ w) {
        // Issue every swizzled atom and pipeline STSM and TMA store
        constexpr uint32_t kNumStores = BLOCK_N / STORE_BLOCK_N;
        #pragma unroll
        for (uint32_t s = 0; s < kNumStores; ++ s) {
            const auto store_idx = reverse_store_order ? kNumStores - 1 - s : s;
            auto smem_base_ptr = reinterpret_cast<uint8_t*>(smem.cd[tma_stage_idx]);

            // Swizzled shared memory address of the `bank_group_idx`-th bank group in this
            // warp's atom, reshaping the atom in another view:
            //  - original: `(LAYOUT_AD_M, kSwizzleCDMode / kNumBankGroupBytes)`
            //  - new: `(LAYOUT_AD_M * kSwizzleCDMode / kNumBankGroupBytes / 8, 8)`
            // NOTES: "8" is the number of bank groups, "16" is the swizzling pattern
            const auto get_swizzled_smem_ptr = [&](const uint32_t& bank_group_idx) {
                constexpr bool kHasShortcut = (kSwizzleCDMode / kNumBankGroupBytes) == 8;
                const auto shifted_idx = bank_group_idx + lane_idx * (kSwizzleCDMode / kNumBankGroupBytes);
                auto row = kHasShortcut ? (bank_group_idx / 8 + lane_idx) : (shifted_idx / 8);
                auto col = kHasShortcut ? bank_group_idx : (shifted_idx % 8);
                col ^= row % (kSwizzleCDMode / 16);
                return smem_base_ptr +                                              // Base pointer
                       epilogue_warp_idx * 32 * kSwizzleCDMode +                    // Warp offset
                       row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes;   // In-atom offset
            };

            // Wait shared memory to be released
            if (epilogue_warp_idx == 0)
                cute::tma_store_wait<kNumTMAStoreStages - 1>();
            cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);

            // The pipeline stage
            const auto m_idx = base_m_idx + w * STORE_BLOCK_M;
            const auto n_idx = base_n_idx + store_idx * STORE_BLOCK_N;

            // Iterate over TMEM loads: each covers one swizzled bank group per lane, or one
            // complete SF group per lane (one row, `kSFGranN` columns) with FP8 output
            constexpr uint32_t kNumElemsPerLoad = kIsFP8CastWithClamp ? kSFGranN : kNumElemsPerBankGroup;
            constexpr uint32_t kNumLoads = STORE_BLOCK_N / kNumElemsPerLoad;
            constexpr auto kNumOverlapLoads = math::constexpr_ceil_div(kNumOverlappedTmemCols, kNumElemsPerLoad);
            DG_STATIC_ASSERT(kNumOverlapLoads <= kNumLoads, "The first store block must cover all overlapped columns");

            // FP8 output N is 128-aligned, so four SF groups always form a complete word.
            const auto row_idx = m_idx + epilogue_warp_idx * 32 + lane_idx;
            uint32_t sf_word = 0;

            #pragma unroll
            for (uint32_t i = 0; i < kNumLoads; ++ i) {
                const auto load_idx = reverse_store_order ? kNumLoads - 1 - i : i;
                const auto tmem_addr = tmem_base_addr +                                     // Accumulator offset
                                       w * BLOCK_N +                                        // Wave offset
                                       store_idx * STORE_BLOCK_N + load_idx * kNumElemsPerLoad; // In-block offset

                // Read from tensor memory into registers
                // NOTES: empty groups also read the accumulator (safe, as at least one UMMA is
                //        issued) and select zeros afterwards, keeping the loads branchless
                uint32_t values[kNumElemsPerLoad];
                DG_STATIC_ASSERT(kNumElemsPerLoad == 4 or kNumElemsPerLoad == 8 or kNumElemsPerLoad == 32, "Invalid load width");
                using tmem_load_t = cute::conditional_t<kNumElemsPerLoad == 32, cute::SM100_TMEM_LOAD_32dp32b32x,
                                                        cute::conditional_t<kNumElemsPerLoad ==  8, cute::SM100_TMEM_LOAD_32dp32b8x,
                                                                            cute::SM100_TMEM_LOAD_32dp32b4x>>;
                [&]<size_t... Is>(cute::index_sequence<Is...>) {
                    tmem_load_t::copy(tmem_addr, values[Is]...);
                }(cute::make_index_sequence<kNumElemsPerLoad>{});
                if constexpr (kNumLeaves > 1) {
                    // IsoExec leaf tree, LAST pair of the progressive fold (sm100_fold_leaf_pair): the pair node
                    // slot0 + slot1 (even leaf on the left), then the binary counter's pending partials, lowest level
                    // first, each on the LEFT (it holds the lower leaf indices): v = P[j] + v. Every add is a node of
                    // the balanced adjacent-pair tree with its operands in tree order: the root, one __fadd_rn a node.
                    DG_STATIC_ASSERT(kLeafTmemCols > 0, "Leaf accumulator stride");
                    constexpr uint32_t kNumPairs = kNumLeaves / 2;
                    // the last pair's leaves sit in ring slots (L - 2) % R and (L - 1) % R; `values` was loaded from
                    // slot 0, so reload the even leaf when the ring places it elsewhere
                    constexpr uint32_t kEvenSlot = (kNumLeaves - 2) % kLeafRing, kOddSlot = (kNumLeaves - 1) % kLeafRing;
                    uint32_t other[kNumElemsPerLoad];
                    if constexpr (kEvenSlot != 0) {
                        [&]<size_t... Is>(cute::index_sequence<Is...>) {
                            tmem_load_t::copy(tmem_addr + kEvenSlot * kLeafTmemCols, values[Is]...);
                        }(cute::make_index_sequence<kNumElemsPerLoad>{});
                    }
                    // every load of the last pair (odd leaf, each pending partial) before one wait
                    constexpr uint32_t kNumPartials = (kNumLeaves >= 4) + (kNumLeaves >= 8);
                    uint32_t part[kNumPartials > 0 ? kNumPartials : 1][kNumElemsPerLoad];
                    [&]<size_t... Is>(cute::index_sequence<Is...>) {
                        tmem_load_t::copy(tmem_addr + kOddSlot * kLeafTmemCols, other[Is]...);
                    }(cute::make_index_sequence<kNumElemsPerLoad>{});
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumPartials; ++ j) {
                        [&]<size_t... Is>(cute::index_sequence<Is...>) {
                            tmem_load_t::copy(tmem_addr + (kLeafRing + j) * kLeafTmemCols, part[j][Is]...);
                        }(cute::make_index_sequence<kNumElemsPerLoad>{});
                    }
                    cutlass::arch::fence_view_async_tmem_load();
                    #pragma unroll
                    for (uint32_t e = 0; e < kNumElemsPerLoad; ++ e)
                        values[e] = __float_as_uint(__fadd_rn(__uint_as_float(values[e]), __uint_as_float(other[e])));
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumPartials; ++ j) {
                        #pragma unroll
                        for (uint32_t e = 0; e < kNumElemsPerLoad; ++ e)
                            values[e] = __float_as_uint(__fadd_rn(__uint_as_float(part[j][e]), __uint_as_float(values[e])));
                    }
                } else {
                    cutlass::arch::fence_view_async_tmem_load();
                }
                epilogue_operator.apply_values(values);
                #pragma unroll
                for (uint32_t value_idx = 0; value_idx < kNumElemsPerLoad; ++ value_idx)
                    values[value_idx] = is_empty_group ? 0u : values[value_idx];

                // Notify the MMA warp once all overlapped TMEM columns have been read,
                // before any of the cast/store work
                if constexpr (kNumOverlapLoads > 0) {
                    if (w == 0 and s == 0 and i + 1 == kNumOverlapLoads) {
                        ptx::tcgen05_before_thread_sync();
                        tmem_overlap_barrier->arrive(0u);
                    }
                }
                // Notify tensor memory empty (only at the leader CTA) as soon as all loads are in registers
                if (w == kNumMWaves - 1 and s == kNumStores - 1 and i == kNumLoads - 1) {
                    ptx::tcgen05_before_thread_sync();
                    tmem_empty_barrier->arrive(0u);
                    // IsoExec lean tail: the CTA's last TMEM read is done -> arrive on the final cluster barrier now
                    // (release; tcgen05 fence above), so TMEM is freed while this thread still stores
                    if (tail_arrive)
                        asm volatile("barrier.cluster.arrive;\n" : : : "memory");
                }

                // Store into shared memory
                if constexpr (kIsFP8CastWithClamp) {
                    nv_bfloat162 values_bf16x2[kNumElemsPerLoad / 2];
                    uint32_t packed[kNumElemsPerLoad / 4];
                    uint2 sf_exp;
                    epilogue_operator.template prepare_fp8<false>(values, values_bf16x2, sf_exp, row_idx,
                                                                  n_idx + load_idx * kNumElemsPerLoad);
                    epilogue_operator.cast_fp8(values_bf16x2, packed, sf_exp);

                    constexpr uint32_t kNumBankGroupsPerLoad = kNumElemsPerLoad / kNumElemsPerBankGroup;
                    #pragma unroll
                    for (uint32_t bank_group_iter_idx = 0; bank_group_iter_idx < kNumBankGroupsPerLoad; ++ bank_group_iter_idx) {
                        const auto smem_ptr = get_swizzled_smem_ptr(load_idx * kNumBankGroupsPerLoad + bank_group_iter_idx);
                        const auto packed_idx = bank_group_iter_idx * 4;
                        ptx::st_shared(smem_ptr, packed[packed_idx], packed[packed_idx + 1],
                                                 packed[packed_idx + 2], packed[packed_idx + 3]);
                    }

                    // Accumulate the SF byte into the packed word, or store it directly
                    if constexpr (kNumLoads == 4)
                        sf_word |= sf_exp.x << (load_idx * 8);
                    else
                        epilogue_operator.store_sf(row_idx, n_idx + load_idx * kSFGranN, batch_idx, static_cast<uint8_t>(sf_exp.x));
                } else if constexpr (cute::is_same_v<cd_dtype_t, float>) {
                    ptx::st_shared(get_swizzled_smem_ptr(load_idx), values[0], values[1], values[2], values[3]);
                } else {
                    uint32_t packed[4];
                    epilogue_operator.template cast_bf16<false>(values, packed, n_idx + load_idx * kNumElemsPerLoad, lane_idx);
                    ptx::st_shared(get_swizzled_smem_ptr(load_idx), packed[0], packed[1], packed[2], packed[3]);
                }
            }
            if constexpr (kIsFP8CastWithClamp and kNumLoads == 4)
                epilogue_operator.store_sf(row_idx, n_idx, batch_idx, sf_word);

            // Synchronize all threads and issue TMA
            cute::tma_store_fence();
            cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);
            if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                if constexpr (kGemmType == GemmType::Batched or is_k_grouped_contiguous(kGemmType)) {
                    using cute_tma_t = cute::conditional_t<kWithAccumulation,
                                                           cute::SM90_TMA_REDUCE_ADD_3D, cute::SM90_TMA_STORE_3D>;
                    cute_tma_t::copy(&tensor_map_cd, smem_base_ptr, n_idx, m_idx, batch_idx);
                } else {
                    using cute_tma_t = cute::conditional_t<kWithAccumulation,
                                                           cute::SM90_TMA_REDUCE_ADD_2D, cute::SM90_TMA_STORE_2D>;
                    cute_tma_t::copy(&tensor_map_cd, smem_base_ptr, n_idx, m_idx);
                }
                cute::tma_store_arrive();
            }
            __syncwarp();
            tma_stage_idx = (tma_stage_idx + 1) % kNumTMAStoreStages;
        }
    }
}

// IsoExec progressive leaf fold, one NON-final pair k of a tile (k < kNumLeaves / 2 - 1): the pair node
// p = slot0 + slot1 (leaves 2k, 2k + 1; even on the left), then the binary counter over pairs: for level j = 0, 1, ..:
// if bit j of k is set, a partial of the same level waits in P[j] (it covers the lower leaves): p = P[j] + p and carry
// on; else P[j] = p and stop. Each epilogue thread owns one tensor-memory lane (row) of its warp's 32 rows and walks all
// UMMA_N columns, 16 at a time when the width allows, with every tensor-memory load of a step (both leaves and every
// partial the counter may read) issued before ONE wait. Partials live in tensor memory at slot R + j.
template <uint32_t kNumLeaves, uint32_t kLeafTmemCols, uint32_t kLeafRing = 2>
CUTLASS_DEVICE void sm100_fold_leaf_pair(const uint32_t& tmem_base_addr, const uint32_t& pair_idx) {
    // leaves of the pair: ring slots (2k) % R (left) and (2k + 1) % R; partials after the ring
    const uint32_t even_col = ((2 * pair_idx) % kLeafRing) * kLeafTmemCols;
    const uint32_t odd_col = ((2 * pair_idx + 1) % kLeafRing) * kLeafTmemCols;
    DG_STATIC_ASSERT(kLeafTmemCols % 8 == 0, "Leaf accumulator width");
    constexpr uint32_t kStep = kLeafTmemCols % 16 == 0 ? 16 : 8;
    constexpr uint32_t kNumPartials = (kNumLeaves >= 4) + (kNumLeaves >= 8);
    using ld_t = cute::conditional_t<kStep == 16, cute::SM100_TMEM_LOAD_32dp32b16x, cute::SM100_TMEM_LOAD_32dp32b8x>;
    using st_t = cute::conditional_t<kStep == 16, cute::SM100_TMEM_STORE_32dp32b16x, cute::SM100_TMEM_STORE_32dp32b8x>;
    #pragma unroll 1
    for (uint32_t c = 0; c < kLeafTmemCols; c += kStep) {
        uint32_t v[kStep], o[kStep], q[kNumPartials > 0 ? kNumPartials : 1][kStep];
        [&]<size_t... Is>(cute::index_sequence<Is...>) {
            ld_t::copy(tmem_base_addr + even_col + c, v[Is]...);
            ld_t::copy(tmem_base_addr + odd_col + c, o[Is]...);
        }(cute::make_index_sequence<kStep>{});
        #pragma unroll
        for (uint32_t j = 0; j < kNumPartials; ++ j) {
            if ((pair_idx >> j) & 1u) {
                [&]<size_t... Is>(cute::index_sequence<Is...>) {
                    ld_t::copy(tmem_base_addr + (kLeafRing + j) * kLeafTmemCols + c, q[j][Is]...);
                }(cute::make_index_sequence<kStep>{});
            }
        }
        cutlass::arch::fence_view_async_tmem_load();
        #pragma unroll
        for (uint32_t e = 0; e < kStep; ++ e)
            v[e] = __float_as_uint(__fadd_rn(__uint_as_float(v[e]), __uint_as_float(o[e])));
        #pragma unroll
        for (uint32_t j = 0; j <= kNumPartials; ++ j) {
            if (j < kNumPartials and ((pair_idx >> j) & 1u)) {
                #pragma unroll
                for (uint32_t e = 0; e < kStep; ++ e)
                    v[e] = __float_as_uint(__fadd_rn(__uint_as_float(q[j][e]), __uint_as_float(v[e])));
            } else {
                // j < kNumPartials here: a non-final pair index has a clear bit below log2(L / 2)
                [&]<size_t... Is>(cute::index_sequence<Is...>) {
                    st_t::copy(v[Is]..., tmem_base_addr + (kLeafRing + j) * kLeafTmemCols + c);
                }(cute::make_index_sequence<kStep>{});
                break;
            }
        }
    }
    cutlass::arch::fence_view_async_tmem_store();
}

// IsoExec register partials (two epilogue warpgroups, kNumUMMAStoreThreads = 256): warp w (epilogue index) reads
// tensor-memory lane quadrant w % 4; warpgroup h = w / 4 owns, inside every store block of STORE_BLOCK_N columns, the
// half [h STORE_BLOCK_N / 2, (h + 1) STORE_BLOCK_N / 2). The binary counter's partials of a thread's row and columns
// live in its registers: part[j][s STORE_BLOCK_N / 2 + c] for store block s, column c of the half (statically
// indexed). Every value formed is the same tree node, from the same two operands in the same order, as with the
// partials in tensor memory (sm100_fold_leaf_pair / the leaf branch of sm100_store_cd).
template <uint32_t kNumLeaves, uint32_t kLeafTmemCols, uint32_t kLeafRing, uint32_t STORE_BLOCK_N, uint32_t kRegCols>
CUTLASS_DEVICE void sm100_fold_leaf_pair_regs(const uint32_t& tmem_base_addr, const uint32_t& pair_idx,
                                              const uint32_t& half,
                                              uint32_t (&part)[(kNumLeaves >= 4) + (kNumLeaves >= 8) > 0 ?
                                                               (kNumLeaves >= 4) + (kNumLeaves >= 8) : 1][kRegCols]) {
    constexpr uint32_t kNumPartials = (kNumLeaves >= 4) + (kNumLeaves >= 8);
    constexpr uint32_t kHalfN = STORE_BLOCK_N / 2;
    constexpr uint32_t kNumStores = kLeafTmemCols / STORE_BLOCK_N;
    DG_STATIC_ASSERT(kLeafTmemCols % STORE_BLOCK_N == 0 and kHalfN % 8 == 0 and kRegCols == kNumStores * kHalfN,
                     "Register-partial column split");
    // 8-column steps: 16-column steps spill at 128 x 96 (two warpgroups at the 168-register launch bound)
    constexpr uint32_t kStep = 8;
    using ld_t = cute::conditional_t<kStep == 16, cute::SM100_TMEM_LOAD_32dp32b16x, cute::SM100_TMEM_LOAD_32dp32b8x>;
    const uint32_t even_col = ((2 * pair_idx) % kLeafRing) * kLeafTmemCols;
    const uint32_t odd_col = ((2 * pair_idx + 1) % kLeafRing) * kLeafTmemCols;
    #pragma unroll
    for (uint32_t s = 0; s < kNumStores; ++ s) {
        #pragma unroll
        for (uint32_t c = 0; c < kHalfN; c += kStep) {
            const uint32_t col = s * STORE_BLOCK_N + half * kHalfN + c;
            uint32_t v[kStep], o[kStep];
            [&]<size_t... Is>(cute::index_sequence<Is...>) {
                ld_t::copy(tmem_base_addr + even_col + col, v[Is]...);
                ld_t::copy(tmem_base_addr + odd_col + col, o[Is]...);
            }(cute::make_index_sequence<kStep>{});
            cutlass::arch::fence_view_async_tmem_load();
            #pragma unroll
            for (uint32_t e = 0; e < kStep; ++ e)
                v[e] = __float_as_uint(__fadd_rn(__uint_as_float(v[e]), __uint_as_float(o[e])));
            // binary counter without a loop exit (every register index stays static)
            bool carry = true;
            #pragma unroll
            for (uint32_t j = 0; j < kNumPartials; ++ j) {
                if (carry and ((pair_idx >> j) & 1u)) {
                    #pragma unroll
                    for (uint32_t e = 0; e < kStep; ++ e)
                        v[e] = __float_as_uint(__fadd_rn(__uint_as_float(part[j][s * kHalfN + c + e]),
                                                         __uint_as_float(v[e])));
                } else if (carry) {
                    #pragma unroll
                    for (uint32_t e = 0; e < kStep; ++ e)
                        part[j][s * kHalfN + c + e] = v[e];
                    carry = false;
                }
            }
        }
    }
}

// The last pair of a register-partial tile and its store (BF16 or FP32 C/D, one M wave): see above. Both warpgroups
// fill the same shared-memory stage (their halves of each row's bank groups), one thread issues the TMA store.
template <uint32_t BLOCK_N, uint32_t STORE_BLOCK_M, uint32_t STORE_BLOCK_N, uint32_t kSwizzleCDMode,
          uint32_t kNumTMAStoreStages, uint32_t kNumUMMAStoreThreads, GemmType kGemmType, bool kWithAccumulation,
          typename epilogue_operator_t, typename smem_t, uint32_t kNumLeaves, uint32_t kLeafTmemCols, uint32_t kLeafRing,
          uint32_t kRegCols>
CUTLASS_DEVICE void
sm100_store_cd_leaf_regs(smem_t& smem, uint32_t& tma_stage_idx, const uint32_t& tmem_base_addr,
                         const uint32_t& base_m_idx, const uint32_t& base_n_idx, const bool& is_empty_group,
                         const uint32_t& epilogue_warp_idx, const uint32_t& lane_idx,
                         epilogue_operator_t& epilogue_operator,
                         const cutlass::arch::ClusterTransactionBarrier* tmem_empty_barrier,
                         const cute::TmaDescriptor& tensor_map_cd,
                         uint32_t (&part)[(kNumLeaves >= 4) + (kNumLeaves >= 8) > 0 ?
                                          (kNumLeaves >= 4) + (kNumLeaves >= 8) : 1][kRegCols],
                         const bool& tail_arrive = false) {
    using cd_dtype_t = typename smem_t::cd_dtype;
    DG_STATIC_ASSERT(cute::is_same_v<cd_dtype_t, float> or cute::is_same_v<cd_dtype_t, cutlass::bfloat16_t>, "BF16 / FP32 C/D");
    DG_STATIC_ASSERT(not epilogue_operator_t::kIsFP8CastWithClamp and not epilogue_operator_t::kWithStochasticRounding,
                     "Register partials: plain BF16 / FP32 stores");
    DG_STATIC_ASSERT(not kWithAccumulation and kGemmType == GemmType::Normal, "Register partials: normal GEMMs");
    DG_STATIC_ASSERT(kNumUMMAStoreThreads == 256 and STORE_BLOCK_M == 128, "Two epilogue warpgroups, one M wave");
    constexpr uint32_t kNumBankGroupBytes = 16;
    constexpr uint32_t kNumElemsPerLoad = kNumBankGroupBytes / sizeof(cd_dtype_t);
    constexpr uint32_t kNumStores = BLOCK_N / STORE_BLOCK_N;
    constexpr uint32_t kNumLoads = STORE_BLOCK_N / kNumElemsPerLoad;
    constexpr uint32_t kHalfLoads = kNumLoads / 2;
    constexpr uint32_t kHalfN = STORE_BLOCK_N / 2;
    DG_STATIC_ASSERT(BLOCK_N % STORE_BLOCK_N == 0 and kNumLoads % 2 == 0 and kRegCols == kNumStores * kHalfN,
                     "Register-partial store split");
    constexpr uint32_t kNumPartials = (kNumLeaves >= 4) + (kNumLeaves >= 8);
    constexpr uint32_t kEvenSlot = (kNumLeaves - 2) % kLeafRing, kOddSlot = (kNumLeaves - 1) % kLeafRing;
    using tmem_load_t = cute::conditional_t<kNumElemsPerLoad == 8, cute::SM100_TMEM_LOAD_32dp32b8x,
                                            cute::SM100_TMEM_LOAD_32dp32b4x>;
    const uint32_t half = epilogue_warp_idx / 4, quad = epilogue_warp_idx % 4;

    #pragma unroll
    for (uint32_t s = 0; s < kNumStores; ++ s) {
        auto smem_base_ptr = reinterpret_cast<uint8_t*>(smem.cd[tma_stage_idx]);
        const auto get_swizzled_smem_ptr = [&](const uint32_t& bank_group_idx) {
            constexpr bool kHasShortcut = (kSwizzleCDMode / kNumBankGroupBytes) == 8;
            const auto shifted_idx = bank_group_idx + lane_idx * (kSwizzleCDMode / kNumBankGroupBytes);
            auto row = kHasShortcut ? (bank_group_idx / 8 + lane_idx) : (shifted_idx / 8);
            auto col = kHasShortcut ? bank_group_idx : (shifted_idx % 8);
            col ^= row % (kSwizzleCDMode / 16);
            return smem_base_ptr + quad * 32 * kSwizzleCDMode + row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes;
        };
        if (epilogue_warp_idx == 0)
            cute::tma_store_wait<kNumTMAStoreStages - 1>();
        cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);
        const auto n_idx = base_n_idx + s * STORE_BLOCK_N;

        #pragma unroll
        for (uint32_t il = 0; il < kHalfLoads; ++ il) {
            const uint32_t load_idx = half * kHalfLoads + il;
            const auto tmem_addr = tmem_base_addr + s * STORE_BLOCK_N + load_idx * kNumElemsPerLoad;
            uint32_t values[kNumElemsPerLoad], other[kNumElemsPerLoad];
            [&]<size_t... Is>(cute::index_sequence<Is...>) {
                tmem_load_t::copy(tmem_addr + kEvenSlot * kLeafTmemCols, values[Is]...);
                tmem_load_t::copy(tmem_addr + kOddSlot * kLeafTmemCols, other[Is]...);
            }(cute::make_index_sequence<kNumElemsPerLoad>{});
            cutlass::arch::fence_view_async_tmem_load();
            #pragma unroll
            for (uint32_t e = 0; e < kNumElemsPerLoad; ++ e)
                values[e] = __float_as_uint(__fadd_rn(__uint_as_float(values[e]), __uint_as_float(other[e])));
            #pragma unroll
            for (uint32_t j = 0; j < kNumPartials; ++ j) {
                #pragma unroll
                for (uint32_t e = 0; e < kNumElemsPerLoad; ++ e)
                    values[e] = __float_as_uint(__fadd_rn(__uint_as_float(part[j][s * kHalfN + il * kNumElemsPerLoad + e]),
                                                          __uint_as_float(values[e])));
            }
            epilogue_operator.apply_values(values);
            #pragma unroll
            for (uint32_t e = 0; e < kNumElemsPerLoad; ++ e)
                values[e] = is_empty_group ? 0u : values[e];
            if (s == kNumStores - 1 and il == kHalfLoads - 1) {
                ptx::tcgen05_before_thread_sync();
                tmem_empty_barrier->arrive(0u);
                if (tail_arrive)
                    asm volatile("barrier.cluster.arrive;\n" : : : "memory");
            }
            if constexpr (cute::is_same_v<cd_dtype_t, float>) {
                ptx::st_shared(get_swizzled_smem_ptr(load_idx), values[0], values[1], values[2], values[3]);
            } else {
                uint32_t packed[4];
                epilogue_operator.template cast_bf16<false>(values, packed, n_idx + load_idx * kNumElemsPerLoad, lane_idx);
                ptx::st_shared(get_swizzled_smem_ptr(load_idx), packed[0], packed[1], packed[2], packed[3]);
            }
        }
        cute::tma_store_fence();
        cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);
        if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
            cute::SM90_TMA_STORE_2D::copy(&tensor_map_cd, smem_base_ptr, n_idx, base_m_idx);
            cute::tma_store_arrive();
        }
        __syncwarp();
        tma_stage_idx = (tma_stage_idx + 1) % kNumTMAStoreStages;
    }
}

} // namespace deep_gemm::epilogue
