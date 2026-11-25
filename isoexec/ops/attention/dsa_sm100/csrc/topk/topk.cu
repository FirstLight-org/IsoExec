// SPDX-License-Identifier: MIT
// IsoExec dsa_sm100: the deterministic top-k of DeepSeek Sparse Attention (recipe sm100_dsa_topk_v1).
//
// Numerical program (README.md section 3) -- a DECISION, so it is exact by construction:
//   * key map kappa: bf16 bits x -> (x & 0x8000) ? ~x : x ^ 0x8000, the NaN class -> 0 (NaN ranks
//     below -inf, which ranks below every finite); fp32 the same on 32 bits.  Monotone and total.
//   * selected = the K largest under (kappa desc, j asc); output = the selected logical positions in
//     ASCENDING j, -1 padded to K.  Only j in the row's window [ks, ke) are candidates.
//   * algorithm class: radix select over kappa with integer histograms (shared-memory integer atomics
//     for COUNTS only), an exact threshold theta and the exact count c_gt of keys above it; then an
//     ascending scan over j that emits every key above theta and the first K - c_gt keys equal to it,
//     each at its PREFIX-COUNT slot.  No slot is ever an atomic's return value.
//   * several CTAs of a cluster may split one row's key range (rows longer than TOPK_ROW_SPLIT, or any
//     row under a test override): they sum the SAME integer histograms over DSMEM and derive their
//     slot bases from the lower ranks' exact counts, so the list is identical by construction (R-T8).
//
// Schedule (free): one CTA (or one cluster) per row; each pass walks 8-key chunks aligned to the
// row's absolute position so a chunk is one 16-byte load; the keys of the first STASH positions of
// the CTA's range are kept in shared memory for the later passes, positions past it are re-read from
// global (L2).  Digit widths: bf16 12 + 4 bits, fp32 12 + 12 + 8.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <cstdint>
#include <cstdlib>

#include "topk.h"

namespace cg = cooperative_groups;

namespace dsa_sm100::topk {

template<typename KeyT> struct KeyTraits;
template<> struct KeyTraits<uint16_t> {
    static constexpr int BITS = 16;
    static constexpr int NPASS = 2;
    __device__ static constexpr int shift(int p) { return p == 0 ? 4 : 0; }
    __device__ static constexpr int width(int p) { return p == 0 ? 12 : 4; }
    __device__ __forceinline__ static uint32_t key(uint16_t x) {
        uint32_t k = (x & 0x8000u) ? (uint32_t)(uint16_t)~x : (uint32_t)(x ^ 0x8000u);
        return ((x & 0x7FFFu) > 0x7F80u) ? 0u : k;
    }
};
template<> struct KeyTraits<uint32_t> {
    static constexpr int BITS = 32;
    static constexpr int NPASS = 3;
    __device__ static constexpr int shift(int p) { return p == 0 ? 20 : (p == 1 ? 8 : 0); }
    __device__ static constexpr int width(int p) { return p == 2 ? 8 : 12; }
    __device__ __forceinline__ static uint32_t key(uint32_t x) {
        uint32_t k = (x & 0x80000000u) ? ~x : (x ^ 0x80000000u);
        return ((x & 0x7FFFFFFFu) > 0x7F800000u) ? 0u : k;
    }
};

static constexpr int NB = 4096;          // bins of the widest digit

// Block-wide exclusive scan of `v` (and the block total), THREADS a multiple of 32.
template<int THREADS>
__device__ __forceinline__ uint32_t block_exclusive_scan(uint32_t v, uint32_t *warp_buf, uint32_t &total) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    uint32_t x = v;
    #pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const uint32_t y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) warp_buf[warp] = x;
    __syncthreads();
    if (warp == 0) {
        uint32_t w = lane < THREADS / 32 ? warp_buf[lane] : 0u;
        #pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const uint32_t y = __shfl_up_sync(0xffffffffu, w, o);
            if (lane >= o) w += y;
        }
        if (lane < THREADS / 32) warp_buf[lane] = w;       // inclusive warp prefixes
    }
    __syncthreads();
    total = warp_buf[THREADS / 32 - 1];
    const uint32_t base = warp ? warp_buf[warp - 1] : 0u;
    __syncthreads();                                         // warp_buf reusable after return
    return base + x - v;
}

template<typename KeyT, int THREADS, int CLUSTER>
__global__ void __launch_bounds__(THREADS, 1)
dsa_topk_kernel(const TopkParams p) {
    using KT = KeyTraits<KeyT>;
    constexpr int NWARPS = THREADS / 32;
    constexpr int V4 = sizeof(KeyT) * 8 / 16;               // uint4 per 8-key chunk (1 bf16, 2 fp32)
    extern __shared__ __align__(16) unsigned char smem_raw[];
    uint32_t *hist_local = reinterpret_cast<uint32_t *>(smem_raw);          // [NB]
    uint32_t *hist_total = hist_local + NB;                                    // [NB]
    uint32_t *scratch = hist_total + NB;                                       // [64]
    uint4 *stash = reinterpret_cast<uint4 *>(scratch + 64);                   // [stash_chunks * V4]

    const int tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;
    // PDL (launch structure only): nothing global is read before the previous grid completed
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    int rank = 0;
    int row = blockIdx.x;
    if constexpr (CLUSTER > 1) {
        rank = (int)cg::this_cluster().block_rank();
        row = blockIdx.x / CLUSTER;
    }
    const int K = p.k;
    const int ks = p.ks ? __ldg(p.ks + row) : 0;
    const int ke = __ldg(p.ke + row);
    const int n = max(ke - ks, 0);
    if (n < p.n_lo || n >= p.n_hi) return;      // another band's row (uniform over the cluster: same n)
    int *out = p.out + (int64_t)row * K;

    // n <= K: every candidate is selected, in ascending j; pads trail.  (Each rank writes its share.)
    if (n <= K) {
        for (int s = rank * THREADS + tid; s < K; s += THREADS * CLUSTER) out[s] = s < n ? ks + s : -1;
        if constexpr (CLUSTER > 1) cg::this_cluster().sync();   // keep the cluster's lifetime uniform
        return;
    }

    // This rank's key range: absolute 8-chunks [ca, cb) of [ks, ke).
    const int c0 = ks >> 3, c1 = (ke + 7) >> 3;
    const int per = (c1 - c0 + CLUSTER - 1) / CLUSTER;
    const int ca = min(c0 + rank * per, c1), cb = min(ca + per, c1);
    const int nchunks = cb - ca;
    const int stash_chunks = min(nchunks, p.stash_keys / 8);
    const KeyT *row_ptr = reinterpret_cast<const KeyT *>(p.scores) + (int64_t)row * p.stride;
    const bool vec_ok = p.vec_ok;

    // The window mask of absolute chunk c: bit e set iff 8c + e is in [ks, ke).
    auto valid_of = [&](int c) -> uint32_t {
        const int j0 = c * 8;
        const int lo = max(ks - j0, 0), hi = min(ke - j0, 8);
        return hi <= lo ? 0u : (((1u << hi) - 1u) & ~((1u << lo) - 1u));
    };
    // Keys of chunk c from global memory (positions outside the window read as key 0, masked anyway).
    auto load_chunk = [&](int c, uint32_t (&k)[8]) {
        const int j0 = c * 8;
        if (vec_ok && j0 >= ks && j0 + 8 <= ke) {
            if constexpr (sizeof(KeyT) == 2) {
                const uint4 v = __ldg(reinterpret_cast<const uint4 *>(row_ptr + j0));
                const uint32_t w[4] = {v.x, v.y, v.z, v.w};
                #pragma unroll
                for (int e = 0; e < 8; ++e) k[e] = KT::key((KeyT)((w[e >> 1] >> ((e & 1) * 16)) & 0xFFFFu));
            } else {
                const uint4 v0 = __ldg(reinterpret_cast<const uint4 *>(row_ptr + j0));
                const uint4 v1 = __ldg(reinterpret_cast<const uint4 *>(row_ptr + j0 + 4));
                const uint32_t w[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
                #pragma unroll
                for (int e = 0; e < 8; ++e) k[e] = KT::key((KeyT)w[e]);
            }
        } else {
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int j = j0 + e;
                k[e] = (j >= ks && j < ke) ? KT::key(__ldg(row_ptr + j)) : 0u;
            }
        }
    };
    auto stash_put = [&](int u, const uint32_t (&k)[8]) {
        if constexpr (sizeof(KeyT) == 2) {
            stash[u] = make_uint4(k[0] | (k[1] << 16), k[2] | (k[3] << 16), k[4] | (k[5] << 16), k[6] | (k[7] << 16));
        } else {
            stash[2 * u] = make_uint4(k[0], k[1], k[2], k[3]);
            stash[2 * u + 1] = make_uint4(k[4], k[5], k[6], k[7]);
        }
    };
    // Keys of chunk c for the passes after the first: the stash, else a re-read.
    auto keys_of = [&](int c, uint32_t (&k)[8]) {
        const int u = c - ca;
        if (u < stash_chunks) {
            if constexpr (sizeof(KeyT) == 2) {
                const uint4 v = stash[u];
                const uint32_t w[4] = {v.x, v.y, v.z, v.w};
                #pragma unroll
                for (int e = 0; e < 8; ++e) k[e] = (w[e >> 1] >> ((e & 1) * 16)) & 0xFFFFu;
            } else {
                const uint4 v0 = stash[2 * u], v1 = stash[2 * u + 1];
                k[0] = v0.x; k[1] = v0.y; k[2] = v0.z; k[3] = v0.w;
                k[4] = v1.x; k[5] = v1.y; k[6] = v1.z; k[7] = v1.w;
            }
        } else {
            load_chunk(c, k);
        }
    };

    __syncthreads();
    for (int i = tid; i < NB; i += THREADS) hist_local[i] = 0u;
    __syncthreads();

    // ---- pass 0: histogram of the top digit, and the stash ----------------------------------------
    {
        constexpr int SH = KT::shift(0);
        #pragma unroll 2
        for (int u = tid; u < nchunks; u += THREADS) {
            uint32_t k[8];
            load_chunk(ca + u, k);
            const uint32_t valid = valid_of(ca + u);
            if (u < stash_chunks) stash_put(u, k);
            #pragma unroll
            for (int e = 0; e < 8; ++e)
                if (valid >> e & 1u) atomicAdd(&hist_local[k[e] >> SH], 1u);
        }
    }

    uint32_t prefix = 0;          // the threshold's digits found so far (in place)
    uint32_t prefix_mask = 0;
    int need = K;                 // how many of the current prefix class are still to be selected
    uint32_t c_gt_local = 0;      // this rank's count of keys strictly above the final threshold
    uint32_t eq_local = 0;        // this rank's count of keys equal to the final threshold
    __shared__ int s_digit, s_above;

    #pragma unroll 1
    for (int pass = 0; pass < KT::NPASS; ++pass) {
        const int SH = KT::shift(pass);
        const int nbins = 1 << KT::width(pass);
        if (pass > 0) {
            // histogram of this digit over the keys whose higher digits equal the prefix
            for (int i = tid; i < nbins; i += THREADS) hist_local[i] = 0u;
            __syncthreads();
            #pragma unroll 2
            for (int u = tid; u < nchunks; u += THREADS) {
                uint32_t k[8];
                keys_of(ca + u, k);
                const uint32_t valid = valid_of(ca + u);
                #pragma unroll
                for (int e = 0; e < 8; ++e)
                    if ((valid >> e & 1u) && (k[e] & prefix_mask) == prefix)
                        atomicAdd(&hist_local[(k[e] >> SH) & (nbins - 1)], 1u);
            }
        }
        __syncthreads();
        // ---- the cluster's histogram: an exact integer sum over the ranks ----------------------
        if constexpr (CLUSTER > 1) {
            cg::cluster_group cl = cg::this_cluster();
            cl.sync();
            for (int i = tid; i < nbins; i += THREADS) {
                uint32_t s = 0;
                #pragma unroll
                for (int r = 0; r < CLUSTER; ++r) s += cl.map_shared_rank(hist_local, r)[i];
                hist_total[i] = s;
            }
            cl.sync();
        } else {
            for (int i = tid; i < nbins; i += THREADS) hist_total[i] = hist_local[i];
        }
        __syncthreads();
        // ---- the digit: the bin d with above(d) < need <= above(d) + hist[d], above(d) being the
        // count in bins > d.  Thread `tid` owns the bpt bins [nbins - (tid+1)*bpt, nbins - tid*bpt),
        // i.e. ascending tid walks DESCENDING bins, so an exclusive scan over tid is above(group).
        {
            const int bpt = (nbins + THREADS - 1) / THREADS;          // bins per thread (1 or 4)
            const int hi = nbins - tid * bpt;                          // exclusive upper bin
            uint32_t mine = 0;
            for (int q = 1; q <= bpt; ++q) if (hi - q >= 0) mine += hist_total[hi - q];
            uint32_t tot;
            uint32_t above = block_exclusive_scan<THREADS>(mine, scratch, tot);
            for (int q = 1; q <= bpt; ++q) {
                const int dd = hi - q;
                if (dd < 0) break;
                const uint32_t h = hist_total[dd];
                if (above < (uint32_t)need && (uint32_t)need <= above + h) { s_digit = dd; s_above = (int)above; }
                above += h;
            }
            __syncthreads();
        }
        const int d = s_digit;
        const int above = s_above;
        // this rank's keys of the prefix class strictly above the digit, and (last pass) equal to it
        {
            uint32_t mine = 0;
            for (int i = tid; i < nbins; i += THREADS) if (i > d) mine += hist_local[i];
            uint32_t tot;
            (void)block_exclusive_scan<THREADS>(mine, scratch, tot);
            c_gt_local += tot;
            if (pass + 1 == KT::NPASS) eq_local = hist_local[d];
        }
        __syncthreads();
        need -= above;
        prefix |= (uint32_t)d << SH;
        prefix_mask |= (uint32_t)(nbins - 1) << SH;
        __syncthreads();
    }
    const uint32_t theta = prefix;
    const uint32_t need_eq = (uint32_t)need;          // K - c_gt over the whole row

    // ---- the slot bases of this rank: exact counts of the lower ranks ----------------------------
    uint32_t rank_gt = 0, rank_eq = 0;
    if constexpr (CLUSTER > 1) {
        cg::cluster_group cl = cg::this_cluster();
        if (tid == 0) { scratch[41] = c_gt_local; scratch[42] = eq_local; }
        cl.sync();
        for (int r = 0; r < rank; ++r) {
            rank_gt += cl.map_shared_rank(scratch, r)[41];
            rank_eq += cl.map_shared_rank(scratch, r)[42];
        }
        cl.sync();
    }

    // ---- the ascending scan.  Warp w owns the contiguous chunk range [w*cpw, (w+1)*cpw) of this
    // rank; a count sweep gives each warp its exact (gt, eq) totals, ONE block scan turns them into
    // bases, and each warp then emits its range in ascending order with warp-level prefix counts.
    const int cpw = (nchunks + NWARPS - 1) / NWARPS;
    const int wa = min(warp * cpw, nchunks), wb = min(wa + cpw, nchunks);
    uint32_t cnt = 0;                                  // gt and eq counted apart: a warp range can exceed 16 bits
    uint32_t cnt_eq = 0;
    for (int u = wa + lane; u < wb; u += 32) {
        uint32_t k[8];
        keys_of(ca + u, k);
        const uint32_t valid = valid_of(ca + u);
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
            cnt += (valid >> e & 1u) & (uint32_t)(k[e] > theta);
            cnt_eq += (valid >> e & 1u) & (uint32_t)(k[e] == theta);
        }
    }
    #pragma unroll
    for (int o = 16; o; o >>= 1) {
        cnt += __shfl_xor_sync(0xffffffffu, cnt, o);
        cnt_eq += __shfl_xor_sync(0xffffffffu, cnt_eq, o);
    }
    __syncthreads();
    if (lane == 0) { hist_total[warp] = cnt; hist_total[32 + warp] = cnt_eq; }   // hist_total is free now
    __syncthreads();
    __syncthreads();
    uint32_t gt_base = rank_gt, eq_base = rank_eq;
    for (int w = 0; w < warp; ++w) { gt_base += hist_total[w]; eq_base += hist_total[32 + w]; }

    #pragma unroll 1
    for (int u0 = wa; u0 < wb; u0 += 32) {
        const int u = u0 + lane;
        uint32_t gt = 0, eq = 0;
        if (u < wb) {
            uint32_t k[8];
            keys_of(ca + u, k);
            const uint32_t valid = valid_of(ca + u);
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                gt |= ((valid >> e) & (uint32_t)(k[e] > theta) & 1u) << e;
                eq |= ((valid >> e) & (uint32_t)(k[e] == theta) & 1u) << e;
            }
        }
        const uint32_t packed = ((uint32_t)__popc(eq) << 16) | (uint32_t)__popc(gt);   // <= 8 each
        uint32_t incl = packed;
        #pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const uint32_t y = __shfl_up_sync(0xffffffffu, incl, o);
            if (lane >= o) incl += y;
        }
        const uint32_t excl = incl - packed;             // per-warp step totals <= 256 per field
        const uint32_t tot = __shfl_sync(0xffffffffu, incl, 31);
        const uint32_t eq_before = eq_base + (excl >> 16);
        uint32_t taken_eq = min(eq_before, need_eq);
        uint32_t slot = gt_base + (excl & 0xFFFFu) + taken_eq;
        if (gt | eq) {
            const int j0 = (ca + u) * 8;
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                bool take = (gt >> e) & 1u;
                if (!take && ((eq >> e) & 1u) && taken_eq < need_eq) { take = true; ++taken_eq; }
                if (take) { if (slot < (uint32_t)K) out[slot] = j0 + e; ++slot; }   // bound: never outside the row
            }
        }
        gt_base += tot & 0xFFFFu;
        eq_base += tot >> 16;
    }
    __syncthreads();
    if constexpr (CLUSTER > 1) cg::this_cluster().sync();
}

// ------------------------------------------------------------------------------------------------
// The bf16 kernel (SCORE_DTYPE).  Same program, fewer instructions: two 12 + 4-bit digits, the
// packed key map on 2 x bf16 words, one 16-byte stash word per 8-key chunk, and no separate count
// sweep -- pass 1 counts, per warp, the keys above the top digit and histograms the boundary class
// (top digit == d0) by its low 4 bits, which is exactly what each warp's slot base needs.
// ------------------------------------------------------------------------------------------------

//: kappa on two packed bf16 (SPEC 3 item 1), per 16-bit half: x ^ (neg ? 0xFFFF : 0x8000), NaN -> 0.
//: `prmt` with selector 0xBB99 replicates the sign bit of byte 1 into bytes 0-1 and of byte 3 into
//: bytes 2-3: the per-half sign mask in one instruction.
__device__ __forceinline__ uint32_t sign_mask2(uint32_t x) {
    uint32_t r;
    asm("prmt.b32 %0, %1, 0, 0xBB99;" : "=r"(r) : "r"(x));
    return r;
}
__device__ __forceinline__ uint32_t key2(uint32_t x) {
    const uint32_t k = x ^ (sign_mask2(x) | 0x80008000u);
    const uint32_t nanb = ((x & 0x7FFF7FFFu) + 0x007F007Fu) & 0x80008000u;   // (x & 0x7FFF) > 0x7F80
    return k & ~sign_mask2(nanb);
}
//: the largest of the 8 keys of a chunk (unsigned 16-bit), for the skip tests.
__device__ __forceinline__ uint32_t vmax2(uint32_t a, uint32_t b) {
    uint32_t r;
    asm("max.u16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    return r;
}
__device__ __forceinline__ uint32_t key_at(const uint4 &k, int e) {
    const uint32_t w = e < 4 ? (e < 2 ? k.x : k.y) : (e < 6 ? k.z : k.w);
    return (e & 1) ? (w >> 16) : (w & 0xFFFFu);
}
//: bit e of the result: key e of the chunk >= t (16-bit unsigned compares on the halves).
__device__ __forceinline__ uint32_t ge_mask(const uint4 &k, uint32_t t) {
    const uint32_t w[4] = {k.x, k.y, k.z, k.w};
    uint32_t m = 0;
    #pragma unroll
    for (int h = 0; h < 4; ++h) {
        m |= (uint32_t)((w[h] & 0xFFFFu) >= t) << (2 * h);
        m |= (uint32_t)((w[h] >> 16) >= t) << (2 * h + 1);
    }
    return m;
}
__device__ __forceinline__ uint32_t chunk_max(const uint4 &k) {
    const uint32_t m = vmax2(vmax2(k.x, k.y), vmax2(k.z, k.w));
    return max(m & 0xFFFFu, m >> 16);
}

template<int THREADS, int CLUSTER>
__global__ void __launch_bounds__(THREADS, THREADS <= 256 ? 4 : (THREADS <= 512 ? 2 : 1))
dsa_topk_bf16_kernel(const TopkParams p) {
    constexpr int NWARPS = THREADS / 32;
    constexpr int QCAP = 64;                                                  // per-warp chunk queue
    extern __shared__ __align__(16) unsigned char smem_raw[];
    uint32_t *hist = reinterpret_cast<uint32_t *>(smem_raw);                 // [NB]  local, top digit
    uint32_t *hist_tot = hist + NB;                                            // [NB]  cluster sum
    uint32_t *whist = hist_tot + (CLUSTER > 1 ? NB : 0);                       // [NWARPS][16]
    uint32_t *wgt0 = whist + NWARPS * 16;                                      // [32]
    uint32_t *scratch = wgt0 + 32;                                             // [64]
    int *queue_all = reinterpret_cast<int *>(scratch + 64);                    // [NWARPS][QCAP]
    uint4 *stash = reinterpret_cast<uint4 *>(queue_all + NWARPS * QCAP);      // [stash_chunks]
    uint16_t *cmax = reinterpret_cast<uint16_t *>(stash + p.stash_keys / 8);  // [cmax_chunks]

    const int tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;
    // PDL (launch structure only): nothing global is read before the previous grid completed
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    int *queue = queue_all + warp * QCAP;
    int rank = 0;
    int row = blockIdx.x;
    if constexpr (CLUSTER > 1) {
        rank = (int)cg::this_cluster().block_rank();
        row = blockIdx.x / CLUSTER;
    }
    const int K = p.k;
    const int ks = p.ks ? __ldg(p.ks + row) : 0;
    const int ke = __ldg(p.ke + row);
    const int n = max(ke - ks, 0);
    if (n < p.n_lo || n >= p.n_hi) return;      // another band's row (uniform over the cluster: same n)
    // solo rows (window shorter than solo_below): rank 0 runs the row as the 1-CTA plan, the other ranks leave at
    // once and no cluster barrier or DSMEM access happens for the row (the decision is uniform over the cluster:
    // same row, same n). Execution only: the 1-CTA plan's list is the cluster plan's (R-T8).
    const bool coop = CLUSTER > 1 && n >= p.solo_below;
    if (CLUSTER > 1 && !coop && rank != 0) return;
    const int CL = coop ? CLUSTER : 1;
    int *out = p.out + (int64_t)row * K;
    int *rout = p.rows_out ? p.rows_out + (int64_t)row * K : nullptr;
    const int *rbt = p.rows_out ? p.rows_bt + (int64_t)(p.rows_seq ? __ldg(p.rows_seq + row) : row) * p.rows_bt_stride
                                : nullptr;
    // the selected id j as its physical cache row (page table entry * per + j % 64), or -1
    auto emit = [&](int slot, int j) {
        out[slot] = j;
        if (rout) rout[slot] = j < 0 ? -1 : __ldg(rbt + j / 64) * p.rows_per + j % 64;
    };
    if (n <= K) {
        for (int s = rank * THREADS + tid; s < K; s += THREADS * CL) emit(s, s < n ? ks + s : -1);
        if (coop) cg::this_cluster().sync();
        return;
    }
    const int c0 = ks >> 3, c1 = (ke + 7) >> 3;
    const int per = (c1 - c0 + CL - 1) / CL;
    const int ca = min(c0 + rank * per, c1), cb = min(ca + per, c1);
    const int nchunks = cb - ca;
    const int stash_chunks = min(nchunks, p.stash_keys / 8);
    const uint16_t *row_ptr = reinterpret_cast<const uint16_t *>(p.scores) + (int64_t)row * p.stride;
    const bool vec_ok = p.vec_ok;

    // 8 keys of absolute chunk c as 4 packed words.  A position outside the window reads as the raw
    // pattern 0xFFFF (a NaN), whose key is 0: it can never raise a chunk's max, and every count below
    // masks it out by the window anyway.
    auto load_keys = [&](int c) -> uint4 {
        const int j0 = c * 8;
        uint32_t w[4];
        if (vec_ok && j0 >= ks && j0 + 8 <= ke) {
            const uint4 v = __ldg(reinterpret_cast<const uint4 *>(row_ptr + j0));
            w[0] = v.x; w[1] = v.y; w[2] = v.z; w[3] = v.w;
        } else {
            #pragma unroll
            for (int h = 0; h < 4; ++h) {
                const int j = j0 + 2 * h;
                const uint32_t a = (j >= ks && j < ke) ? __ldg(row_ptr + j) : 0xFFFFu;
                const uint32_t b = (j + 1 >= ks && j + 1 < ke) ? __ldg(row_ptr + j + 1) : 0xFFFFu;
                w[h] = a | (b << 16);
            }
        }
        return make_uint4(key2(w[0]), key2(w[1]), key2(w[2]), key2(w[3]));
    };
    auto valid_of = [&](int c) -> uint32_t {
        const int j0 = c * 8;
        const int lo = max(ks - j0, 0), hi = min(ke - j0, 8);
        return hi <= lo ? 0u : (((1u << hi) - 1u) & ~((1u << lo) - 1u));
    };
    auto keys_of = [&](int u) -> uint4 { return u < stash_chunks ? stash[u] : load_keys(ca + u); };

    for (int i = tid; i < NB; i += THREADS) hist[i] = 0u;
    for (int i = tid; i < NWARPS * 16; i += THREADS) whist[i] = 0u;
    __syncthreads();

    // ---- pass 0: the stash, the per-chunk max, and the histogram of the top 12 bits ----------------
    // (MEASURED AND REJECTED: a sampled candidate bound -- histogram a 1/16 sample, then only keys
    // above the sample's bin, with the full histogram as the identical fallback.  It removes ~90% of
    // the atomics and is SLOWER: the phase went 9.3K -> 21.9K cycles at 32K keys, because the cost of a
    // pass is its per-chunk instruction issue, not the atomics.  README "Rejected variants".)
    #pragma unroll 4
    for (int u = tid; u < nchunks; u += THREADS) {
        const uint4 kk = load_keys(ca + u);
        if (u < stash_chunks) stash[u] = kk;
        cmax[u] = (uint16_t)chunk_max(kk);
        const uint32_t w[4] = {kk.x, kk.y, kk.z, kk.w};
        const int j0 = (ca + u) * 8;
        if (j0 >= ks && j0 + 8 <= ke) {
            #pragma unroll
            for (int h = 0; h < 4; ++h) {
                atomicAdd(&hist[(w[h] & 0xFFFFu) >> 4], 1u);
                atomicAdd(&hist[w[h] >> 20], 1u);
            }
        } else {
            const uint32_t valid = valid_of(ca + u);
            #pragma unroll
            for (int h = 0; h < 4; ++h) {
                if (valid >> (2 * h) & 1u) atomicAdd(&hist[(w[h] & 0xFFFFu) >> 4], 1u);
                if (valid >> (2 * h + 1) & 1u) atomicAdd(&hist[w[h] >> 20], 1u);
            }
        }
    }
    __syncthreads();

    uint32_t *htot = hist;
    if (CLUSTER > 1 && coop) {
        cg::cluster_group cl = cg::this_cluster();
        cl.sync();
        for (int i = tid; i < NB; i += THREADS) {
            uint32_t s = 0;
            #pragma unroll
            for (int r = 0; r < CLUSTER; ++r) s += cl.map_shared_rank(hist, r)[i];
            hist_tot[i] = s;
        }
        cl.sync();
        htot = hist_tot;
    }
    __shared__ int s_digit, s_above;
    // the top digit d0: above(d0) < K <= above(d0) + hist[d0]; thread tid owns 4096/THREADS bins,
    // descending with tid, so an exclusive scan over tid is above(its group)
    {
        // Each thread's BPT contiguous bins come in as BPT/4 16-byte loads (a scalar walk at a stride of
        // BPT words between threads is a BPT-way bank conflict: 7.5K cycles at 256 threads, measured).
        constexpr int BPT = NB / THREADS;
        static_assert(BPT % 4 == 0, "bins per thread must be whole 16-byte groups");
        const int lo = NB - (tid + 1) * BPT;                       // this thread's bins [lo, lo + BPT)
        uint32_t h[BPT];
        #pragma unroll
        for (int v = 0; v < BPT / 4; ++v) {
            const uint4 x = *reinterpret_cast<const uint4 *>(htot + lo + 4 * v);
            h[4 * v] = x.x; h[4 * v + 1] = x.y; h[4 * v + 2] = x.z; h[4 * v + 3] = x.w;
        }
        uint32_t mine = 0;
        #pragma unroll
        for (int q = 0; q < BPT; ++q) mine += h[q];
        uint32_t tot;
        uint32_t above = block_exclusive_scan<THREADS>(mine, scratch, tot);
        #pragma unroll
        for (int q = BPT - 1; q >= 0; --q) {                       // descending bins
            if (above < (uint32_t)K && (uint32_t)K <= above + h[q]) { s_digit = lo + q; s_above = (int)above; }
            above += h[q];
        }
        __syncthreads();
    }

    const uint32_t d0 = (uint32_t)s_digit;
    const int need1 = K - s_above;          // to select inside the boundary class d0
    const uint32_t lo_b = d0 << 4;           // the boundary class is [lo_b, lo_b + 16)

    // Warp w owns the contiguous chunk range [wa, wb) in pass 1 and in the emission.  Both passes
    // visit only the chunks whose max can reach their threshold, compacted (in ascending order) into
    // dense 32-chunk batches through a per-warp queue; a skipped chunk contributes nothing to either.
    const int cpw = (nchunks + NWARPS - 1) / NWARPS;
    const int wa = min(warp * cpw, nchunks), wb = min(wa + cpw, nchunks);
    const uint32_t lanemask_lt = (1u << lane) - 1u;
    // for each batch of up to 32 queued chunks, in ascending order: fn(u or -1)
    auto sweep = [&](uint32_t thr, auto &&fn) {
        int qn = 0;
        #pragma unroll 1
        for (int base = wa; base < wb; base += 32) {
            const int u = base + lane;
            const bool act = u < wb && (uint32_t)cmax[u] >= thr;
            const uint32_t bal = __ballot_sync(0xffffffffu, act);
            if (act) queue[qn + __popc(bal & lanemask_lt)] = u;
            qn += __popc(bal);
            __syncwarp();
            if (qn >= 32) {
                fn(queue[lane]);
                __syncwarp();
                const int rest = qn - 32;
                const int mv = lane < rest ? queue[32 + lane] : 0;
                __syncwarp();
                if (lane < rest) queue[lane] = mv;
                qn = rest;
                __syncwarp();
            }
        }
        if (qn > 0) fn(lane < qn ? queue[lane] : -1);
        __syncwarp();
    };

    // ---- pass 1: keys above d0 counted, the boundary class histogrammed by its low 4 bits ----------
    uint32_t gt0 = 0;
    sweep(lo_b, [&](int u) {
        if (u < 0) return;
        const uint4 kk = keys_of(u);
        const uint32_t valid = valid_of(ca + u);
        const uint32_t ge_lo = ge_mask(kk, lo_b) & valid;
        const uint32_t ge_hi = ge_mask(kk, lo_b + 16u) & valid;
        gt0 += (uint32_t)__popc(ge_hi);
        uint32_t bd = ge_lo & ~ge_hi;                // the boundary class: top digit == d0
        while (bd) {
            const int e = __ffs(bd) - 1;
            atomicAdd(&whist[warp * 16 + (key_at(kk, e) & 15u)], 1u);
            bd &= bd - 1u;
        }
    });
    #pragma unroll
    for (int o = 16; o; o >>= 1) gt0 += __shfl_xor_sync(0xffffffffu, gt0, o);
    if (lane == 0) wgt0[warp] = gt0;
    __syncthreads();

    // the boundary histogram of this rank (and of the cluster)
    if (tid < 16) {
        uint32_t s = 0;
        for (int w = 0; w < NWARPS; ++w) s += whist[w * 16 + tid];
        scratch[32 + tid] = s;                                // local
    }
    __syncthreads();
    if (CLUSTER > 1 && coop) {
        cg::cluster_group cl = cg::this_cluster();
        cl.sync();
        if (tid < 16) {
            uint32_t s = 0;
            #pragma unroll
            for (int r = 0; r < CLUSTER; ++r) s += cl.map_shared_rank(scratch, r)[32 + tid];
            scratch[48 + tid] = s;
        }
        cl.sync();
    } else {
        if (tid < 16) scratch[48 + tid] = scratch[32 + tid];
    }
    __syncthreads();
    if (tid == 0) {
        uint32_t above = 0;
        for (int b = 15; b >= 0; --b) {
            const uint32_t h = scratch[48 + b];
            if (above < (uint32_t)need1 && (uint32_t)need1 <= above + h) { s_digit = b; s_above = (int)above; }
            above += h;
        }
    }
    __syncthreads();

    const uint32_t d1 = (uint32_t)s_digit;
    const uint32_t theta = lo_b | d1;
    const uint32_t need_eq = (uint32_t)(need1 - s_above);

    // ---- slot bases: this warp's exact (gt, eq), the lower warps', the lower ranks' ---------------
    if (tid < NWARPS) {
        uint32_t g = wgt0[tid];
        for (int b = (int)d1 + 1; b < 16; ++b) g += whist[tid * 16 + b];
        scratch[tid] = g;                                  // per-warp gt (scratch[0, 32))
        hist[tid] = whist[tid * 16 + d1];                  // per-warp eq (hist is free: d0 is known)
    }
    __syncthreads();
    uint32_t gt_base = 0, eq_base = 0;
    for (int w = 0; w < warp; ++w) { gt_base += scratch[w]; eq_base += hist[w]; }
    if (CLUSTER > 1 && coop) {
        cg::cluster_group cl = cg::this_cluster();
        if (tid == 0) {
            uint32_t g = 0, q = 0;
            for (int w = 0; w < NWARPS; ++w) { g += scratch[w]; q += hist[w]; }
            scratch[40] = g; scratch[41] = q;
        }
        cl.sync();
        for (int r = 0; r < rank; ++r) {
            gt_base += cl.map_shared_rank(scratch, r)[40];
            eq_base += cl.map_shared_rank(scratch, r)[41];
        }
        cl.sync();
    }

    // ---- emission: ascending, prefix-count slots ---------------------------------------------------
    // Each queued chunk contributes `gt` (keys > theta) and `eq` (keys == theta) as bit masks.  A tie
    // is taken iff fewer than need_eq ties precede it in ascending j; a key's slot is the number of
    // taken keys before it.  Both counts are exact prefix sums, never an atomic's return.
    sweep(theta, [&](int u) {
        uint32_t gt = 0, eq = 0;
        if (u >= 0) {
            const uint4 kk = keys_of(u);
            const uint32_t valid = valid_of(ca + u);
            const uint32_t ge = ge_mask(kk, theta) & valid;
            gt = ge_mask(kk, theta + 1u) & valid;
            eq = ge & ~gt;
        }
        const uint32_t packed = ((uint32_t)__popc(eq) << 16) | (uint32_t)__popc(gt);
        uint32_t incl = packed;
        #pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const uint32_t y = __shfl_up_sync(0xffffffffu, incl, o);
            if (lane >= o) incl += y;
        }
        const uint32_t excl = incl - packed;
        const uint32_t tot = __shfl_sync(0xffffffffu, incl, 31);
        if (gt | eq) {
            const uint32_t eq_before = eq_base + (excl >> 16);
            uint32_t take = gt;
            if (eq) {
                // the lowest min(need_eq - eq_before, popc(eq)) set bits of eq are taken
                uint32_t left = eq_before < need_eq ? need_eq - eq_before : 0u;
                uint32_t t = eq;
                while (t && left) { take |= t & (0u - t); t &= t - 1u; --left; }
            }
            uint32_t slot = gt_base + (excl & 0xFFFFu) + min(eq_before, need_eq);
            const int j0 = (ca + u) * 8;
            while (take) {                                // ascending e: lowest set bit first
                const int e = __ffs(take) - 1;
                if (slot < (uint32_t)K) emit((int)slot, j0 + e);   // bound: never outside the row
                ++slot;
                take &= take - 1u;
            }
        }
        gt_base += tot & 0xFFFFu;
        eq_base += tot >> 16;
    });

    if (coop) cg::this_cluster().sync();
}

//: ISOEXEC_SMALL_PDL=0 turns the programmatic-dependent launch off (as for the other small kernels).
static bool pdl_enabled() {
    const char *e = std::getenv("ISOEXEC_SMALL_PDL");
    return !(e && e[0] == '0');
}

template<int THREADS, int CLUSTER>
static void launch_bf16(const TopkParams &p, cudaStream_t stream) {
    auto kernel = dsa_topk_bf16_kernel<THREADS, CLUSTER>;
    const size_t smem = (size_t)((CLUSTER > 1 ? 2 : 1) * NB + (THREADS / 32) * 16 + 32 + 64 +
                                 (THREADS / 32) * 64) * 4 +
                        (size_t)(p.stash_keys / 8) * 16 + (size_t)p.cmax_chunks * 2;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    if constexpr (CLUSTER > 1) {
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = dim3(p.rows * CLUSTER, 1, 1);
        cfg.blockDim = dim3(THREADS, 1, 1);
        cfg.dynamicSmemBytes = smem;
        cfg.stream = stream;
        cudaLaunchAttribute attr[2];
        attr[0].id = cudaLaunchAttributeClusterDimension;
        attr[0].val.clusterDim.x = CLUSTER;
        attr[0].val.clusterDim.y = 1;
        attr[0].val.clusterDim.z = 1;
        attr[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr[1].val.programmaticStreamSerializationAllowed = 1;
        cfg.attrs = attr;
        cfg.numAttrs = pdl_enabled() ? 2 : 1;
        cudaLaunchKernelEx(&cfg, kernel, p);
    } else {
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = dim3(p.rows, 1, 1);
        cfg.blockDim = dim3(THREADS, 1, 1);
        cfg.dynamicSmemBytes = smem;
        cfg.stream = stream;
        cudaLaunchAttribute attr[1];
        attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr[0].val.programmaticStreamSerializationAllowed = 1;
        cfg.attrs = attr;
        cfg.numAttrs = pdl_enabled() ? 1 : 0;
        cudaLaunchKernelEx(&cfg, kernel, p);
    }
}

template<typename KeyT, int THREADS, int CLUSTER>
static void launch_one(const TopkParams &p, cudaStream_t stream) {
    auto kernel = dsa_topk_kernel<KeyT, THREADS, CLUSTER>;
    const size_t smem = (size_t)(2 * NB + 64) * 4 + (size_t)(p.stash_keys / 8) * 8 * sizeof(KeyT);
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    if constexpr (CLUSTER > 1) {
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = dim3(p.rows * CLUSTER, 1, 1);
        cfg.blockDim = dim3(THREADS, 1, 1);
        cfg.dynamicSmemBytes = smem;
        cfg.stream = stream;
        cudaLaunchAttribute attr[2];
        attr[0].id = cudaLaunchAttributeClusterDimension;
        attr[0].val.clusterDim.x = CLUSTER;
        attr[0].val.clusterDim.y = 1;
        attr[0].val.clusterDim.z = 1;
        attr[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr[1].val.programmaticStreamSerializationAllowed = 1;
        cfg.attrs = attr;
        cfg.numAttrs = pdl_enabled() ? 2 : 1;
        cudaLaunchKernelEx(&cfg, kernel, p);
    } else {
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = dim3(p.rows, 1, 1);
        cfg.blockDim = dim3(THREADS, 1, 1);
        cfg.dynamicSmemBytes = smem;
        cfg.stream = stream;
        cudaLaunchAttribute attr[1];
        attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr[0].val.programmaticStreamSerializationAllowed = 1;
        cfg.attrs = attr;
        cfg.numAttrs = pdl_enabled() ? 1 : 0;
        cudaLaunchKernelEx(&cfg, kernel, p);
    }
}

bool launch(const TopkParams &p, bool fp32, int cluster, int threads, int algo, cudaStream_t stream) {
    if (!fp32 && algo == 0) {
#define DSA_TOPK_BF16(TH, CL) if (threads == TH && cluster == CL) { launch_bf16<TH, CL>(p, stream); return true; }
        DSA_TOPK_BF16(256, 1) DSA_TOPK_BF16(512, 1) DSA_TOPK_BF16(1024, 1)
        DSA_TOPK_BF16(512, 2) DSA_TOPK_BF16(1024, 2) DSA_TOPK_BF16(512, 4) DSA_TOPK_BF16(1024, 4)
#undef DSA_TOPK_BF16
    }
#define DSA_TOPK_CASE(KT, TH, CL) \
    if (threads == TH && cluster == CL) { launch_one<KT, TH, CL>(p, stream); return true; }
    if (!fp32) {
        DSA_TOPK_CASE(uint16_t, 512, 1) DSA_TOPK_CASE(uint16_t, 1024, 1)
        DSA_TOPK_CASE(uint16_t, 512, 2) DSA_TOPK_CASE(uint16_t, 1024, 2)
        DSA_TOPK_CASE(uint16_t, 512, 4) DSA_TOPK_CASE(uint16_t, 1024, 4)
    } else {
        DSA_TOPK_CASE(uint32_t, 512, 1) DSA_TOPK_CASE(uint32_t, 1024, 1)
        DSA_TOPK_CASE(uint32_t, 512, 2) DSA_TOPK_CASE(uint32_t, 1024, 2)
        DSA_TOPK_CASE(uint32_t, 512, 4) DSA_TOPK_CASE(uint32_t, 1024, 4)
    }
#undef DSA_TOPK_CASE
    return false;
}

}  // namespace dsa_sm100::topk
