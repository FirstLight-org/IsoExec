#pragma once

#include <cutlass/arch/barrier.h>
#include <cutlass/numeric_types.h>

#include <deep_gemm/common/math.cuh>

namespace deep_gemm::layout::mega_gate {

static constexpr uint32_t BLOCK_K = 64;
static constexpr uint32_t kSwizzleMode = 128;
static constexpr uint32_t kNumNonEpilogueThreads = 128;
static constexpr uint32_t kNumEpilogueStages = 2;
static constexpr uint32_t kNumMaxBlockTokens = 256;
static constexpr uint32_t kExpertAlignment = 128;
static constexpr uint32_t kSharedMemoryAlignment = 1024;

static constexpr uint32_t kNumMinBlockTokens = 16;
static constexpr uint32_t kNumMaxTokens = 1u << 20;
static constexpr uint32_t kNumMaxTokenBlocks = math::constexpr_ceil_div(kNumMaxTokens, kNumMinBlockTokens);

static constexpr uint32_t kScoreBarrierLineBytes = 128;
static constexpr uint32_t kNumMaxLogicalCtas = 64;

// Expert count granularity of the API (the selection itself is router_sm100's, see `epilogue/sm100_mega_gate.cuh`)
static constexpr uint32_t kNumExpertsPerLaneVector = 4;

// IsoExec router logits program: the hidden axis is cut into `kNumSegments` fixed segments (H / 4 each, whole
// `BLOCK_K` blocks; `kHiddenAlignment = 256` makes it 4 for every admitted H). Segment g is ONE ascending tcgen05 chain
// from zero in its own FP32 TMEM accumulator; the logit is the balanced adjacent-pair tree of the 4 segment roots,
// ((s0 + s1) + (s2 + s3)), one __fadd_rn a node. A split-K slice owns `kNumSegments / num_split_k` whole segments
// (folded in its epilogue: a subtree), the gate folds the slices (the tree's upper levels): the same tree for every split.
static constexpr uint32_t kNumSegments = 4;

// TMEM budget: `kNumEpilogueStages` x segments-per-CTA accumulators share the 512 columns, `kNumMaxBlockTokens / SL`
// columns each. A 128-expert group on a CTA pair holds `N / 2` columns per CTA (the token halves on the two lane
// halves), so its token block may be twice as long.
CUTLASS_HOST_DEVICE constexpr uint32_t get_num_accum_cols(const uint32_t num_segments_per_cta) {
    return kNumMaxBlockTokens / num_segments_per_cta;
}

CUTLASS_HOST_DEVICE constexpr uint32_t get_max_block_tokens(const uint32_t num_experts_per_group, const uint32_t num_mma_ctas,
                                                            const uint32_t num_segments_per_cta) {
    const auto partitioned = num_experts_per_group == kExpertAlignment and num_mma_ctas == 2;
    const auto max_block_tokens = get_num_accum_cols(num_segments_per_cta) * (partitioned ? 2u : 1u);
    return max_block_tokens < kNumMaxBlockTokens ? max_block_tokens : kNumMaxBlockTokens;
}

static constexpr uint32_t kNumMaxStages = 32;

// Fixed SMEM header, followed by `num_stages` x stages then `num_stages` weight stages (runtime strides)
struct SharedStorage {
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    alignas(Barrier) Barrier full_barriers[kNumMaxStages];
    alignas(Barrier) Barrier empty_barriers[kNumMaxStages];
    alignas(Barrier) Barrier tmem_full_barriers[kNumEpilogueStages];
    alignas(Barrier) Barrier tmem_empty_barriers[kNumEpilogueStages];
    alignas(uint32_t) uint32_t tmem_ptr;
};

static constexpr uint32_t kStageDataOffset = math::constexpr_align(static_cast<uint32_t>(sizeof(SharedStorage)), kSharedMemoryAlignment);

CUTLASS_DEVICE cutlass::bfloat16_t* get_x_stage_ptr(uint8_t* smem_buffer, const uint32_t& stage_idx, const uint32_t& x_stage_bytes) {
    return reinterpret_cast<cutlass::bfloat16_t*>(smem_buffer + kStageDataOffset + stage_idx * x_stage_bytes);
}

CUTLASS_DEVICE cutlass::bfloat16_t* get_weight_stage_ptr(uint8_t* smem_buffer, const uint32_t& stage_idx, const uint32_t& num_stages,
                                                         const uint32_t& x_stage_bytes, const uint32_t& w_stage_bytes) {
    return reinterpret_cast<cutlass::bfloat16_t*>(smem_buffer + kStageDataOffset + num_stages * x_stage_bytes + stage_idx * w_stage_bytes);
}

// View over global score scratch: [num_token_blocks * block_tokens, kNumSplits, kNumExperts]
template <uint32_t kNumSplits = 0, uint32_t kNumExperts = 0>
struct Workspace {
    static constexpr uint32_t kTokenStride = kNumSplits * kNumExperts;

    void* gmem_scratch;
    void* gmem_score_barriers;

    CUTLASS_HOST_DEVICE
    Workspace(void* gmem_scratch, void* gmem_score_barriers):
        gmem_scratch(gmem_scratch), gmem_score_barriers(gmem_score_barriers) {}

    CUTLASS_HOST_DEVICE
    static constexpr uint64_t get_num_scratch_bytes(const uint32_t num_token_blocks,
                                                    const uint32_t num_splits,
                                                    const uint32_t block_tokens,
                                                    const uint32_t num_experts) {
        return static_cast<uint64_t>(num_token_blocks) * num_splits * block_tokens * num_experts * sizeof(float);
    }

    CUTLASS_HOST_DEVICE
    float* get_score_ptr(const uint32_t token_idx, const uint32_t split_idx = 0) const {
        return reinterpret_cast<float*>(gmem_scratch) +
               static_cast<uint64_t>(token_idx) * kTokenStride + split_idx * kNumExperts;
    }

    CUTLASS_HOST_DEVICE uint64_t* get_score_barrier_ptr(const uint32_t token_block_idx) const {
        return reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(gmem_score_barriers) +
                                           token_block_idx * kScoreBarrierLineBytes);
    }
};

} // namespace deep_gemm::layout::mega_gate
