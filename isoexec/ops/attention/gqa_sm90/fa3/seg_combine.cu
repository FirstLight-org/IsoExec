// IsoExec segment-split combine for the vendored FA3 forward (recipe sm90_fa3_segsplit, gqa_sm90/spec.py).
// Not part of upstream FlashAttention-3; written for IsoExec against FA3's split-KV partial layout.
//
// The attention kernel, run with params.num_splits > 1 and params.seg_n_blocks > 0, attends each row's
// fixed logical KV segments [s * SEG, (s + 1) * SEG) (block.h) one per split and publishes, per segment,
// FA3's own finalized partial: O_s = acc_s * (1 / L_s) in fp32 and lse_s = M_s * scale * ln2 + log(L_s).
// A sequence the prepare kernel gave one split (its KV fits one segment, or HYBRID folded it in the
// attention kernel; params.seg_split_flags) is written directly by the attention kernel (FA3's epilogue
// for num_splits_dynamic == 1) and is skipped here.
//
// For every other row, with border B = (logical position of the row) + 1 and n = ceil(B / SEG) segments:
//
//   n == 1 :  out = cvt.rn.bf16(O_0), lse = lse_0
//             -- exactly the conversion FA3's unsplit epilogue applies to the same fp32 value, so the row's
//                bits equal the unsplit kernel's.
//   n >= 2 :  the right fold of seg_merge.h, state = seg[n-1]; state = merge(seg[s], state) for s = n-2..0,
//             then out = cvt.rn.bf16(state.O), lse = state.lse. The attention kernel's in-CTA fold (the
//             saturated path) runs the same merge on the same leaves in the same order, so the two paths
//             give identical bits.
//
// The function reads only the row's own n partials and n is a function of the row's logical position
// and SEG: the result cannot depend on the batch, the packing, the page layout, the grid or the SM count.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

#include "flash.h"
#include "cuda_check.h"
#include "seg_merge.h"

namespace isoexec_seg {

struct CombineParams {
    float const* o_partial;       // [num_splits, h, total_q, dv] (strides below, elements)
    float const* lse_partial;     // [num_splits, h, total_q]
    __nv_bfloat16* out;           // [total_q, h, dv]
    float* lse;                   // [h, total_q]
    int64_t op_split_stride, op_head_stride, op_row_stride;
    int64_t lp_split_stride, lp_head_stride;
    int64_t o_row_stride, o_head_stride;
    int64_t lse_head_stride;
    int const* cu_seqlens_q;
    int const* seqused_q;
    int const* cu_seqlens_k;
    int const* seqused_k;
    int seqlen_k_static;
    int seg_tokens;
    int num_splits;
    int const* split_flags;  // [b] each sequence's split count (prepare_varlen_num_blocks_kernel)
    int total_q;
    int num_batch;
};

// One warp per (row, head); kVec = dv / 32 contiguous fp32 per lane.
template <int kVec, int kWarps>
__global__ void __launch_bounds__(kWarps * 32)
seg_combine_kernel(CombineParams const p) {
    static_assert(kVec % 2 == 0, "pairs are converted with cvt.rn.bf16x2.f32");
    int const warp = threadIdx.x / 32;
    int const lane = threadIdx.x % 32;
    int const head = blockIdx.y;
    // Grid-stride over the packed rows of the whole launch (a bounded grid: the work is the split rows,
    // which only the prepare kernel knows). A row finds its sequence by binary search over cu_seqlens_q.
    for (int t = blockIdx.x * kWarps + warp; t < p.total_q; t += gridDim.x * kWarps) {
    int lo = 0, hi = p.num_batch;  // the b with cu_seqlens_q[b] <= t < cu_seqlens_q[b + 1]
    while (hi - lo > 1) {
        int const mid = (lo + hi) / 2;
        if (p.cu_seqlens_q[mid] <= t) { lo = mid; } else { hi = mid; }
    }
    int const b = lo;
    // One split for the sequence: the attention kernel wrote its rows itself -- one segment, or a folded
    // sequence. The prepare kernel recorded the choice; this reads it rather than re-deriving it.
    if (p.split_flags[b] <= 1) { continue; }
    int const q_start = p.cu_seqlens_q[b];
    int const seqlen_q = p.seqused_q ? p.seqused_q[b] : p.cu_seqlens_q[b + 1] - q_start;
    int const i = t - q_start;
    if (i >= seqlen_q) { continue; }  // padding between seqused_q and the next sequence
    int const seqlen_k = p.seqused_k ? p.seqused_k[b]
                       : (p.cu_seqlens_k ? p.cu_seqlens_k[b + 1] - p.cu_seqlens_k[b] : p.seqlen_k_static);
    // Causal, bottom-right aligned: the row attends keys [0, border).
    int const border = seqlen_k - seqlen_q + i + 1;
    int const n = border <= 0 ? 1 : (border + p.seg_tokens - 1) / p.seg_tokens;

    float const* lp = p.lse_partial + int64_t(head) * p.lp_head_stride + t;
    float const* op = p.o_partial + int64_t(head) * p.op_head_stride + int64_t(t) * p.op_row_stride + lane * kVec;
    __nv_bfloat16* out = p.out + int64_t(t) * p.o_row_stride + int64_t(head) * p.o_head_stride + lane * kVec;

    float acc[kVec];
    float lse_out;
    if (n > p.num_splits) {
        // max_seqlen_k understated this sequence's KV length: its later segments were never computed.
        #pragma unroll
        for (int j = 0; j < kVec; ++j) { acc[j] = __int_as_float(0x7fffffff); }
        lse_out = __int_as_float(0x7fffffff);
    } else if (n == 1) {
        #pragma unroll
        for (int j = 0; j < kVec; j += 2) {
            float2 v = *reinterpret_cast<float2 const*>(op + j);
            acc[j] = v.x; acc[j + 1] = v.y;
        }
        lse_out = lp[0];
    } else {
        // state = seg[n-1]
        float const* os = op + int64_t(n - 1) * p.op_split_stride;
        #pragma unroll
        for (int j = 0; j < kVec; j += 2) {
            float2 v = *reinterpret_cast<float2 const*>(os + j);
            acc[j] = v.x; acc[j + 1] = v.y;
        }
        lse_out = lp[int64_t(n - 1) * p.lp_split_stride];
        for (int s = n - 2; s >= 0; --s) {  // state = merge(seg[s], state)
            float const lse_a = lp[int64_t(s) * p.lp_split_stride];
            os = op + int64_t(s) * p.op_split_stride;
            if (lse_out == -INFINITY) {  // upper state empty: seg[s], verbatim
                #pragma unroll
                for (int j = 0; j < kVec; j += 2) {
                    float2 v = *reinterpret_cast<float2 const*>(os + j);
                    acc[j] = v.x; acc[j + 1] = v.y;
                }
                lse_out = lse_a;
                continue;
            }
            if (lse_a == -INFINITY) { continue; }  // seg[s] empty: the state, verbatim
            MergeCoef const c = merge_coef(lse_a, lse_out);
            #pragma unroll
            for (int j = 0; j < kVec; j += 2) {
                float2 v = *reinterpret_cast<float2 const*>(os + j);
                acc[j] = merge_elem(c, v.x, acc[j]);
                acc[j + 1] = merge_elem(c, v.y, acc[j + 1]);
            }
            lse_out = c.lse;
        }
    }
    #pragma unroll
    for (int j = 0; j < kVec; j += 2) {
        *reinterpret_cast<__nv_bfloat162*>(out + j) = __floats2bfloat162_rn(acc[j], acc[j + 1]);
    }
    if (lane == 0) { p.lse[int64_t(head) * p.lse_head_stride + t] = lse_out; }
    }
}

template <int kVec>
void launch(CombineParams const& p, int num_sm, int h, cudaStream_t stream) {
    constexpr int kWarps = 4;
    dim3 grid(std::max(1, std::min((p.total_q + kWarps - 1) / kWarps, 8 * num_sm)), h);
    seg_combine_kernel<kVec, kWarps><<<grid, kWarps * 32, 0, stream>>>(p);
}

}  // namespace isoexec_seg

void run_isoexec_seg_combine(Flash_fwd_params &params, cudaStream_t stream) {
    isoexec_seg::CombineParams p;
    p.o_partial = static_cast<float const*>(params.oaccum_ptr);
    p.lse_partial = static_cast<float const*>(params.softmax_lseaccum_ptr);
    p.out = static_cast<__nv_bfloat16*>(params.o_ptr);
    p.lse = static_cast<float*>(params.softmax_lse_ptr);
    p.op_split_stride = params.oaccum_split_stride;
    p.op_head_stride = params.oaccum_head_stride;
    p.op_row_stride = params.oaccum_row_stride;
    p.lp_split_stride = params.lseaccum_split_stride;
    p.lp_head_stride = params.lseaccum_head_stride;
    p.o_row_stride = params.o_row_stride;
    p.o_head_stride = params.o_head_stride;
    p.lse_head_stride = params.total_q;  // softmax_lse is [h, total_q] contiguous (varlen)
    p.cu_seqlens_q = params.cu_seqlens_q;
    p.seqused_q = params.seqused_q;
    p.cu_seqlens_k = params.cu_seqlens_k;
    p.seqused_k = params.seqused_k;
    p.seqlen_k_static = params.seqlen_k;
    p.seg_tokens = params.seg_tokens;
    p.num_splits = params.num_splits;
    p.split_flags = params.seg_split_flags;
    p.total_q = params.total_q;
    p.num_batch = params.b;
    switch (params.dv) {
        case 64: isoexec_seg::launch<2>(p, params.num_sm, params.h, stream); break;
        case 128: isoexec_seg::launch<4>(p, params.num_sm, params.h, stream); break;
        case 256: isoexec_seg::launch<8>(p, params.num_sm, params.h, stream); break;
        case 512: isoexec_seg::launch<16>(p, params.num_sm, params.h, stream); break;
        default: break;  // refused on the host (mha_fwd) before the attention kernel runs
    }
    CHECK_CUDA_KERNEL_LAUNCH();
}
