// SPDX-License-Identifier: MIT
// IsoExec mla_sm100: deployment constants of the deterministic absorbed-MLA forward on Blackwell.
// These numbers are part of the recipe identity (mla_sm90/README.md "The numerical programs"):
// the KV tile is the page and is anchored at key 0, the row block is 64 query rows, and the
// head dims are the GLM-4.7-Flash absorbed geometry.  Changing any of them is a version bump.
#pragma once

namespace mla_sm100::fwd {

namespace Config {

static constexpr int BLOCK_SIZE_M = 64;       // query rows per row block (recipe: masking window)
static constexpr int PAGE_BLOCK_SIZE = 64;    // KV tile == page, anchored at key 0 (recipe)

static constexpr int HEAD_DIM_K = 576;        // [c_kv 512 | k_pe 64]
static constexpr int HEAD_DIM_V = 512;        // V = c_kv

// Blackwell execution shape (free: not part of the recipe).
static constexpr int NUM_WARPS = 8;           // 0-3 softmax/correction, 4 mma, 5 load, 6-7 spare
                                              // (6-7 only reach the item-boundary __syncthreads;
                                              //  dropping them is free and untried)
static constexpr int NUM_THREADS = NUM_WARPS * 32;
static constexpr int KV_STAGES = 3;           // SMEM KV tiles in flight (round 5: Q moved to TMEM)
static constexpr int S_BUFFERS = 2;           // TMEM score/probability slots
static constexpr int PV_N = 256;              // UMMA N of one PV accumulator half
static constexpr int QK_CHUNK = 64;           // head-dim chunk of one TMA / one QK MMA group

}  // namespace Config

}  // namespace mla_sm100::fwd
