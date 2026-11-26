// SPDX-License-Identifier: MIT
// IsoExec sparse_sm100: host interface of the deterministic top-k (recipe sm100_dsa_topk_v1).
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace sparse_sm100::topk {

struct TopkParams {
    const void *scores;     // [rows, stride] bf16 (as uint16) or fp32 (as uint32)
    const int *ks;          // [rows] window start, or nullptr (= 0)
    const int *ke;          // [rows] window end (exclusive)
    int *out;               // [rows, k] int32, ascending, -1 padded
    int64_t stride;         // elements
    int rows;
    int k;
    int stash_keys;         // shared-memory key stash per CTA (a multiple of 8)
    int cmax_chunks;        // per-chunk max summary capacity per CTA (bf16 kernel): >= chunks of the widest range
    bool vec_ok;            // row starts 16-byte aligned (16-byte loads of 8-key chunks)
    // row band of this launch: only rows whose window length n = ke - ks satisfies n_lo <= n < n_hi are
    // written (the others belong to another launch of the same recipe with its own plan); 0 / INT_MAX: all
    int n_lo, n_hi;
    // bf16 cluster kernel: rows with n < solo_below run on the cluster's rank 0 alone (0: every row cooperates)
    int solo_below;
    // optional second output: each selected id j as the physical cache row rows_bt[rows_seq[row], j / 64] *
    // rows_per + j % 64 (-1 for a pad): the attention extra list, written by the emission itself
    int *rows_out; const int *rows_bt; int64_t rows_bt_stride; const int *rows_seq; int rows_per;
};

//: algo 0: the bf16-specialised kernel (fp32 always takes the generic one); 1: the generic kernel.
bool launch(const TopkParams &p, bool fp32, int cluster, int threads, int algo, cudaStream_t stream);

}  // namespace sparse_sm100::topk
