// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "params.h"

namespace gqa_sm100 {

cudaError_t launch_plan(const PlanParams& p, int grid_blocks, cudaStream_t stream);

//: `waves` and `head_ratio` select the compiled leaf; `item_grid` is the host-computed capacity the
//: grid is sized by, so the launch is CUDA-graph capturable with a device-side plan.
cudaError_t launch_leaf_d128(const LeafParams& p, const CUtensorMap& mq, const CUtensorMap& mk,
                            const CUtensorMap& mv, int waves, int head_ratio, int item_grid,
                            cudaStream_t stream);

//: The D256 leaf runs one CTA PAIR per work item, so its launch is a CLUSTER launch of dimension
//: two and `item_grid` is the ITEM capacity: the grid is twice that, and a multiple of two by
//: construction. There is no wave count -- each CTA of the pair owns 128 packed query rows, so the
//: tile is 256 rows wide and `tile_rows` is 256.
cudaError_t launch_leaf_d256(const LeafParams& p, const CUtensorMap& mq, const CUtensorMap& mk,
                            const CUtensorMap& mv, int head_ratio, int item_grid,
                            cudaStream_t stream);
//: The folding D256 leaf (`leaf_d256_fold.cu`), for a run longer than one segment.
cudaError_t launch_leaf_d256_fold(const LeafParams& p, const CUtensorMap& mq, const CUtensorMap& mk,
                            const CUtensorMap& mv, int head_ratio, int item_grid,
                            cudaStream_t stream);

//: `live_rows` is the most packed query rows any ONE tile can hold -- `min(tile_rows,
//: min(max_seqlen_q, tile_tokens) * ratio)` -- and it sizes `grid.y`. The combine runs one WARP per
//: (token, query head), so a decode tile of one token has `ratio` live rows out of `tile_rows`, and
//: sizing the grid by the tile instead launched 32 blocks per tile to run four.
cudaError_t launch_combine(const CombineParams& p, int dim, int max_n, int combine_grid,
                          int live_rows, int sms, int pub_tiles, bool group_loads,
                          cudaStream_t stream);

//: 5D bf16 tensor map over a [tokens, heads, dim] tensor with a (64, 1, head_box, tile_rows) box.
//: `token_stride` / `head_stride` are in elements; 0 means the dense value (heads * dim, dim).
CUresult encode_map(void* storage, const void* input, int tokens, int heads, int dim,
                   int tile_rows, int head_box, int64_t token_stride = 0, int64_t head_stride = 0);

}  // namespace gqa_sm100
