/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

namespace flash {

template <class SeqlenInfo_t, int kBlockM, int kBlockN, bool Is_causal, bool Is_local, bool PackGQA=false, bool Split=false>
struct BlockMN {

    static
    CUTLASS_DEVICE
    cute::tuple<int, int> get_n_block_min_max(
            SeqlenInfo_t const& seqlen_info,
            int const m_block, int const bidb, int const split_idx, int const num_splits,
            int const window_size_left, int const window_size_right,
            cutlass::FastDivmod const& attention_chunk_divmod,
            cutlass::FastDivmod const& qhead_per_khead_divmod) {

        int const seqlen_k = seqlen_info.seqlen_k;
        int const seqlen_q = seqlen_info.seqlen_q;
        int n_block_max = cute::ceil_div(seqlen_k, kBlockN);
        if constexpr (Is_causal || Is_local) {
            int m_idx_max = (m_block + 1) * kBlockM;
            // TODO: check off-by-1 error
            if (PackGQA) { m_idx_max = qhead_per_khead_divmod.divide(m_idx_max - 1) + 1 ; }
            int const n_idx = m_idx_max + seqlen_info.seqlen_k - seqlen_info.seqlen_q;
            int n_idx_right = !Is_local ? n_idx : n_idx + window_size_right;
            if (Is_local && attention_chunk_divmod.divisor > 0) {
                n_idx_right = std::min(n_idx_right, flash::round_up(attention_chunk_divmod, n_idx));
            }
            n_block_max = std::min(n_block_max, cute::ceil_div(n_idx_right, kBlockN));
        }
        int n_block_min = 0;
        if constexpr (Is_local) {
            int m_idx_min = m_block * kBlockM;
            if (PackGQA) { m_idx_min = qhead_per_khead_divmod.divide(m_idx_min); }
            int const n_idx = m_idx_min + seqlen_k - seqlen_q;
            int n_idx_left = n_idx - window_size_left;
            if (attention_chunk_divmod.divisor > 0) {
                n_idx_left = std::max(n_idx_left, flash::round_down(attention_chunk_divmod, n_idx));
            }
            n_block_min = std::max(int(0), n_idx_left / kBlockN);
        }
        // if (threadIdx.x == 128) { printf("Inside, bid.x = %d, bid.y = %d, bid.z = %d, split_idx = %d, n_block_min: %d, n_block_max: %d\n", blockIdx.x, blockIdx.y, blockIdx.z, split_idx, n_block_min, n_block_max); }
        if constexpr (Split) {
            // IsoExec segment split (recipe sm90_fa3_segsplit): split s is the FIXED logical KV segment
            // [s * S, (s + 1) * S) in n-blocks, anchored at key 0, where S = `num_splits` here carries
            // the segment width in n-blocks (params.seg_n_blocks, see flash_fwd_launch_template.h).
            // Upstream divides the row's n-block range evenly by the (batch/occupancy dependent) split
            // count, which moves the boundaries; a fixed segment does not.
            // A sequence the scheduler gives one split (num_splits_dynamic == 1) is one work tile over its
            // whole range, which the mainloop walks segment by segment and folds (the fold path).
            uint32_t const num_splits_dynamic_u = reinterpret_cast<uint32_t const&>(split_idx) >> 16;
            int const num_splits_dynamic = reinterpret_cast<int const&>(num_splits_dynamic_u);
            if (num_splits_dynamic != 1) {
                int const split_idx_actual = split_idx & 0x0000FFFF;
                int const seg_n_blocks = num_splits;
                n_block_min = std::max(n_block_min, split_idx_actual * seg_n_blocks);
                n_block_max = std::min(n_block_max, (split_idx_actual + 1) * seg_n_blocks);
            }
        }
        // if (threadIdx.x == 128) { printf("After split, inside, bid.y = %d, bid.z = %d, split_idx = %d, n_block_min: %d, n_block_max: %d\n", blockIdx.y, blockIdx.z, split_idx, n_block_min, n_block_max); }
        return {n_block_min, n_block_max};
    }

    static
    CUTLASS_DEVICE
    cute::tuple<int, int> get_n_block_k_new_min_max(
            SeqlenInfo_t const& seqlen_info,
            int const m_block, int const bidb, int const split_idx, int const num_splits,
            int const window_size_left, int const window_size_right,
            cutlass::FastDivmod const& attention_chunk_divmod,
            cutlass::FastDivmod const& qhead_per_khead_divmod) {

        auto [n_block_min, n_block_max] = get_n_block_min_max(
            seqlen_info, m_block, bidb, split_idx, num_splits,
            window_size_left, window_size_right, attention_chunk_divmod, qhead_per_khead_divmod);
        int const idx_k_new_min = std::max(n_block_min * kBlockN - seqlen_info.seqlen_k_og, 0);
        int const idx_k_new_max = std::min(n_block_max * kBlockN - seqlen_info.seqlen_k_og, seqlen_info.seqlen_k_new);
        int const n_block_new_min = idx_k_new_min / kBlockN;
        int const n_block_new_max = idx_k_new_max > idx_k_new_min ? cute::ceil_div(idx_k_new_max, kBlockN) : n_block_new_min;
        // if (threadIdx.x == 128 && m_block == 0) { printf("bidb = %d, seqlen_k_new = %d, seqlen_k_og = %d, n_block_min = %d, n_block_max = %d, idx_k_new_min = %d, idx_k_new_max = %d, n_block_new_min = %d, n_block_new_max = %d\n", bidb, seqlen_k_new, seqlen_k_og, n_block_min, n_block_max, idx_k_new_min, idx_k_new_max, n_block_new_min, n_block_new_max);}
        return {n_block_new_min, n_block_new_max};
    }

    static
    CUTLASS_DEVICE
    cute::tuple<int, int> get_m_block_min_max(
            SeqlenInfo_t const& seqlen_info,
            int const n_block, int const bidb,
            int const window_size_left, int const window_size_right, int const sink_token_length) {
        // TODO: support attention_chunk
        int const seqlen_q = seqlen_info.seqlen_q;
        int const seqlen_k = seqlen_info.seqlen_k;
        int m_block_max = cute::ceil_div(seqlen_q, kBlockM);
        if constexpr (Is_local) {
            if (n_block >= cute::ceil_div(sink_token_length, kBlockN)) {
                m_block_max = std::min(m_block_max, cute::ceil_div((n_block + 1) * kBlockN + seqlen_q - seqlen_k + window_size_left, kBlockM));
            }
        }
        int m_block_min = 0;
        if constexpr (Is_causal || Is_local) {
            m_block_min = std::max(m_block_min, (n_block * kBlockN + seqlen_q - seqlen_k - window_size_right) / kBlockM);
        }
        return {m_block_min, m_block_max};
    }

    // If we have separate iterations with causal or local masking at the start, where do we stop
    static
    CUTLASS_DEVICE
    int get_n_block_min_causal_local_mask(
            SeqlenInfo_t const& seqlen_info,
            int const m_block, int const n_block_min, int const window_size_right,
            cutlass::FastDivmod const& attention_chunk_divmod,
            cutlass::FastDivmod const& qhead_per_khead_divmod) {
        int const m_idx_min = !PackGQA ? m_block * kBlockM : qhead_per_khead_divmod.divide(m_block * kBlockM);
        int const n_idx = m_idx_min + seqlen_info.seqlen_k - seqlen_info.seqlen_q;
        int n_idx_right = !Is_local ? n_idx : n_idx + window_size_right;
        if (Is_local && attention_chunk_divmod.divisor > 0) {
            n_idx_right = std::min(n_idx_right, flash::round_up(attention_chunk_divmod, n_idx));
        }
        return std::max(n_block_min, n_idx_right / kBlockN);
    }

    // If we have separate iterations with local masking at the end, where do we stop the non-masked iterations
    static
    CUTLASS_DEVICE
    int get_n_block_min_before_local_mask(
            SeqlenInfo_t const& seqlen_info,
            int const m_block, int const n_block_min, int const window_size_left,
            cutlass::FastDivmod const& attention_chunk_divmod,
            cutlass::FastDivmod const& qhead_per_khead_divmod) {
        int const m_idx_max = !PackGQA ? (m_block + 1) * kBlockM : qhead_per_khead_divmod.divide((m_block + 1) * kBlockM - 1) + 1;
        int const n_idx = m_idx_max + seqlen_info.seqlen_k - seqlen_info.seqlen_q;
        int n_idx_left = !Is_local ? n_idx : n_idx - window_size_left;
        if (Is_local && attention_chunk_divmod.divisor > 0) {
            n_idx_left = std::max(n_idx_left, flash::round_down(attention_chunk_divmod, n_idx));
        }
        return !Is_local ? n_block_min : std::max(n_block_min, cute::ceil_div(n_idx_left, kBlockN));
    }

};

} // namespace flash
