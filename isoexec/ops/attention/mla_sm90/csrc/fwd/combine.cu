// SPDX-License-Identifier: MIT
// Derived from DeepSeek FlashMLA (https://github.com/deepseek-ai/FlashMLA at ba89a34; csrc/kernels/smxx/decode/combine/combine.cu), Copyright (c) 2025 DeepSeek, MIT License.
//
// Merge fp32 leaf states with the canonical adjacent-pair tree, carrying unpaired nodes unchanged.
// A streaming binary counter merges equal-sized nodes, then folds the remaining stack from the top.
// Scalar (L, M) merges publish scales that the grouped O tree replays.

// For lower-index a: M=max(a.M,b.M), sa=exp2f(a.M-M), sb=exp2f(b.M-M);
// O=a.O*sa+b.O*sb and L=a.L*sa+b.L*sb, without FMA contraction.
// Root: L'=(L==0 || L!=L)?1:L; out=bf16(O/L'); lse=logf(L')+M/log2(e).
// Inputs may be single leaves or aligned runs folded by workers (params.h node_size_at).
#include <math_constants.h>
#include <cuda_runtime_api.h>
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/arch/barrier.h>

#include "params.h"
#include "utils.h"
#include "combine.h"

using namespace cute;

namespace mla_sm90::fwd {

static constexpr int MAX_DEPTH = 12;
static constexpr float MAX_INIT_VAL_SM = -1e30f;   // the leaf program's running-max seed (leaf.cuh)

// Merge (a lower index, b higher) on NV float4 of O; fmad is off, so the terms are never contracted.
template<int NV>
__forceinline__ __device__ void merge_o(float4 a[NV], const float4 b[NV], float sa, float sb) {
    CUTLASS_PRAGMA_UNROLL
    for (int v = 0; v < NV; ++v) {
        float4 r;
        r.x = a[v].x*sa; r.y = a[v].y*sa; r.z = a[v].z*sa; r.w = a[v].w*sa;
        const float bx = b[v].x*sb, by = b[v].y*sb, bz = b[v].z*sb, bw = b[v].w*sb;
        r.x = r.x + bx; r.y = r.y + by; r.z = r.z + bz; r.w = r.w + bw;
        a[v] = r;
    }
}

// Worker = R combine rows x (E element blocks of EPT elements) x G tree groups, NT = R*G*E threads.
// When R == 1 a row may be split over EBR = 512 / (E*EPT) workers. Execution only: every configuration evaluates
// the same canonical tree, because merge scales are indexed by the merge node, id = (first leaf of the right
// operand) - 1, unique per internal node, and groups are aligned nodes [g*S, (g+1)*S) of the tree.
//   stage A (row threads):   gather every leaf's (L, M) into shared memory (exact copies)
//   stage B (row thread 0):  evaluate the (L, M) tree, publish scales and the root (lse from element block 0)
//   stage C (thread r,e,g):  evaluate group g's subtree on block e, publish the node
//   stage D (thread r,e,0):  evaluate the tree over the group nodes, round once, store
template<typename ElementT, int MAX_N, int R, int G, int E, bool GATHER, int EPT>
__global__ void __launch_bounds__(R*G*E)
mla_sm90_combine_kernel(__grid_constant__ const CombineParams params) {
    constexpr int NT = R*G*E;
    constexpr int EBR = HEAD_DIM_V / (E*EPT);
    constexpr int NV = EPT / 4;
    const int t = threadIdx.x;
    const int r = t / (G*E);
    const int e = (t / G) % E;
    const int g = t % G;
    const int cta_row = blockIdx.x / EBR;
    const int entry = cta_row*R + r;
    const int num_entries = params.counts[NUM_COMBINE_ROWS];
    if (cta_row*R >= num_entries) {
        return;
    }
    const bool active = entry < num_entries;
    int64_t global_row = 0, slot0 = 0, leaf_stride = PARTIAL_WIDTH;
    int n = 1, node = 1;
    if (active) {
        global_row = params.combine_rows[entry*4];
        n = params.combine_rows[entry*4+1];
        slot0 = params.combine_rows[entry*4+2];
        leaf_stride = (int64_t)(params.combine_rows[entry*4+3] & 255)*PARTIAL_WIDTH;
        node = 1 << (params.combine_rows[entry*4+3] >> 8);
    }
    const float* leaf0 = params.partials + slot0*PARTIAL_WIDTH;
    const int64_t first = ((int64_t)(blockIdx.x % EBR)*E + e)*EPT;
    int S = 1;
    while (S*G < n || ((params.dbg & 4) && S < n) || S < node) S *= 2;   // dbg 4: one group (plain streaming tree)
    const int groups = (n + S - 1) / S;

    if (params.pdl) {
        cudaGridDependencySynchronize();
    }

    __shared__ float smem_scales[R][2*MAX_N];
    __shared__ float smem_leaf_L[R][GATHER ? MAX_N + NT : 1];
    __shared__ float smem_leaf_M[R][GATHER ? MAX_N + NT : 1];
    __shared__ float smem_root_L[R];
    __shared__ float4 smem_group[R][E][G == 1 ? 1 : G][NV];

    // Stage A
    if constexpr (GATHER) {
    if (active) {
        const int lane = t % (G*E);
        #pragma unroll 1
        for (int k = lane; k < n; k += G*E) {
            smem_leaf_L[r][k] = leaf0[k*leaf_stride + HEAD_DIM_V];
            smem_leaf_M[r][k] = leaf0[k*leaf_stride + HEAD_DIM_V + 1];
        }
        cutlass::arch::fence_view_async_shared();
    }
    __syncthreads();
    }

    // Stage B
    if (active && e == 0 && g == 0) {
        FLASH_DEVICE_ASSERT(n <= MAX_N);
        float stL[MAX_DEPTH], stM[MAX_DEPTH];
        int stS[MAX_DEPTH], stZ[MAX_DEPTH];
        int depth = 0;
        #pragma unroll 1
        for (int k = 0; k < n;) {
            const int z = node_size_at(k, n, node);
            if constexpr (GATHER) {
                stL[depth] = smem_leaf_L[r][k];
                stM[depth] = smem_leaf_M[r][k];
            } else {
                stL[depth] = leaf0[k*leaf_stride + HEAD_DIM_V];
                stM[depth] = leaf0[k*leaf_stride + HEAD_DIM_V + 1];
            }
            stS[depth] = k;
            stZ[depth] = z;
            ++depth;
            k += z;
            #pragma unroll 1
            while (depth > 1 && (stZ[depth-2] == stZ[depth-1] || k == n)) {
                const float Ma = stM[depth-2], Mb = stM[depth-1];
                const float Mx = max(Ma, Mb);
                const float sa = exp2f(Ma - Mx);
                const float sb = exp2f(Mb - Mx);
                const float La = stL[depth-2]*sa;
                const float Lb = stL[depth-1]*sb;
                const int id = stS[depth-1] - 1;
                stL[depth-2] = La + Lb;
                stM[depth-2] = Mx;
                stZ[depth-2] += stZ[depth-1];
                smem_scales[r][2*id] = sa;
                smem_scales[r][2*id+1] = sb;
                --depth;
            }
        }
        // n == 0 (a row with no keys): the empty state, L = 0 and M = the running-max seed of the leaf program
        const float L = n > 0 ? stL[0] : 0.0f;
        const float M = n > 0 ? stM[0] : MAX_INIT_VAL_SM;
        const float root_L = (L == 0.0f || L != L) ? 1.0f : L;
        smem_root_L[r] = root_L;
        if (first == 0) {
            params.lse[global_row] = logf(root_L) + M / (float)M_LOG2E;
        }
        cutlass::arch::fence_view_async_shared();
    }
    __syncthreads();

    // Empty row: out = 0
    if (active && n == 0 && g == 0) {
        const uint64_t zero = 0;
        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < NV; ++v) {
            *(uint64_t*)((char*)params.out + global_row*1024 + (first + 4*v)*2) = zero;
        }
    }

    // Stage C
    if (active && g < groups) {
        const int k0 = g*S;
        const int k1 = min(n, k0 + S);
        float4 st[MAX_DEPTH][NV];
        int stS[MAX_DEPTH], stZ[MAX_DEPTH];
        int depth = 0;
        #pragma unroll 1
        for (int k = k0; k < k1;) {
            const int z = node_size_at(k, n, node);
            CUTLASS_PRAGMA_UNROLL
            for (int v = 0; v < NV; ++v) {
                st[depth][v] = *(float4*)(leaf0 + k*leaf_stride + first + 4*v);
            }
            stS[depth] = k;
            stZ[depth] = z;
            ++depth;
            k += z;
            #pragma unroll 1
            while (depth > 1 && (stZ[depth-2] == stZ[depth-1] || k == k1)) {
                const int id = stS[depth-1] - 1;
                merge_o<NV>(st[depth-2], st[depth-1], smem_scales[r][2*id], smem_scales[r][2*id+1]);
                stZ[depth-2] += stZ[depth-1];
                --depth;
            }
        }
        if constexpr (G == 1) {
            const float root_L = smem_root_L[r];
            CUTLASS_PRAGMA_UNROLL
            for (int v = 0; v < NV; ++v) {
                const float4 o = st[0][v];
                ElementT converted[4];
                converted[0] = (ElementT)(o.x / root_L);
                converted[1] = (ElementT)(o.y / root_L);
                converted[2] = (ElementT)(o.z / root_L);
                converted[3] = (ElementT)(o.w / root_L);
                *(uint64_t*)((char*)params.out + global_row*1024 + (first + 4*v)*2) = *(uint64_t*)converted;
            }
        } else {
            CUTLASS_PRAGMA_UNROLL
            for (int v = 0; v < NV; ++v) {
                smem_group[r][e][g][v] = st[0][v];
            }
            cutlass::arch::fence_view_async_shared();
        }
    }
    if constexpr (G == 1) {
        return;
    }
    __syncthreads();

    // Stage D
    if (active && g == 0 && n > 0) {
        float4 st[MAX_DEPTH][NV];
        int stS[MAX_DEPTH];
        int depth = 0;
        #pragma unroll 1
        for (int h = 0; h < groups; ++h) {
            CUTLASS_PRAGMA_UNROLL
            for (int v = 0; v < NV; ++v) {
                st[depth][v] = smem_group[r][e][h][v];
            }
            stS[depth] = h*S;
            ++depth;
            #pragma unroll 1
            for (int c = h + 1; depth > 1 && ((c & 1) == 0 || h == groups - 1); c >>= 1) {
                const int id = stS[depth-1] - 1;
                merge_o<NV>(st[depth-2], st[depth-1], smem_scales[r][2*id], smem_scales[r][2*id+1]);
                --depth;
            }
        }
        const float root_L = smem_root_L[r];
        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < NV; ++v) {
            const float4 o = st[0][v];
            ElementT converted[4];
            converted[0] = (ElementT)(o.x / root_L);
            converted[1] = (ElementT)(o.y / root_L);
            converted[2] = (ElementT)(o.z / root_L);
            converted[3] = (ElementT)(o.w / root_L);
            static_assert(sizeof(ElementT) == 2);
            *(uint64_t*)((char*)params.out + global_row*1024 + (first + 4*v)*2) = *(uint64_t*)converted;
        }
    }
}

// Worker configurations (R rows, G groups, E element blocks, gather, EPT elements per thread):
//   0: (1, 32, 1, gather, 16)   32 workers per row, deep trees at small batch
//   1: (1, 16, 8, gather, 16)    4 workers per row
//   2: (2,  2, 32, gather, 16)   one worker per 2 rows
//   3: (8,  1, 32, serial, 16)   one worker per 8 rows (n <= 64, else configuration 2)
template<typename ElementT>
void run_mla_sm90_combine_kernel(CombineParams &params, int num_entries, int max_n, int config) {
    FLASH_ASSERT(max_n <= 1024);
    #define MLA_SM90_LAUNCH_COMBINE(R, G, E, GATHER, NMAX, EPT) do {                                                 \
        const int ctas = (num_entries + R - 1) / R * (HEAD_DIM_V / (E*EPT));                                          \
        mla_sm90_combine_kernel<ElementT, NMAX, R, G, E, GATHER, EPT><<<ctas, R*G*E, 0, params.stream>>>(params);   \
    } while (0)
    #define MLA_SM90_BUCKETS(R, G, E) do {                                                                           \
        if (max_n <= 64) { MLA_SM90_LAUNCH_COMBINE(R, G, E, true, 64, 16); }                                          \
        else if (max_n <= 256) { MLA_SM90_LAUNCH_COMBINE(R, G, E, true, 256, 16); }                                   \
        else { MLA_SM90_LAUNCH_COMBINE(R, G, E, true, 1024, 16); }                                                    \
    } while (0)
    if (config == 0) {
        MLA_SM90_BUCKETS(1, 32, 1);
    } else if (config == 1) {
        MLA_SM90_BUCKETS(1, 16, 8);
    } else if (config == 2 || max_n > 64) {
        MLA_SM90_BUCKETS(2, 2, 32);
    } else {
        MLA_SM90_LAUNCH_COMBINE(8, 1, 32, false, 64, 16);
    }
    #undef MLA_SM90_BUCKETS
    #undef MLA_SM90_LAUNCH_COMBINE
    CHECK_CUDA_KERNEL_LAUNCH();
}

template void run_mla_sm90_combine_kernel<cutlass::bfloat16_t>(CombineParams &params, int num_entries, int max_n, int config);

}  // namespace mla_sm90::fwd
