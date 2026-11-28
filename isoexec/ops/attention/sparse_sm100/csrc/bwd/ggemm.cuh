// SPDX-License-Identifier: MIT
// IsoExec sparse_sm100: gather-GEMMs of the sparse-attention backward (recipe sm100_sparse_bwd_v1 is unchanged in
// meaning: the backward is fp64-accurate and run-to-run deterministic, not bitwise-pinned; README "Step 10").
//
// The key rows of a token are never materialised: the B operand is TMA-gathered (gather4) straight from kv by the
// token's list, as the forward leaf does.
//   SD:  [S ; dP][t] (2h x K, fp32)   = [q ; dO][t] (2h x D, two 64-row boxes, dO zero past its width) . kv[list[t]]^T
//   DQ:  dq[t]       (h x D, bf16)    = dS[t] (from the grad kernel's [dS ; P] rows) . kv[list[t]]
//   ROWS: rows[t]    (K x D, fp32)    = [dS ; P][t]^T (MN-major A, read in place) . [q ; dO|0][t] (64-head boxes)
// One tile = 128 rows x NC columns, K in 64-wide chunks through a TMA ring; persistent CTAs (one per SM) walk the
// (token, row tile, column tile) items; two TMEM accumulators let a tile's epilogue overlap the next tile's MMAs.
// Every output element is one ascending K16 chain from a zeroed fp32 accumulator (no atomics, no split).
#pragma once

#include <cuda.h>
#include <cute/tensor.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cutlass/arch/barrier.h>

namespace sparse_sm100::ggemm {

using namespace cute;
using bfx = cutlass::bfloat16_t;
using Bar = cutlass::arch::ClusterBarrier;

constexpr int MR = 128, KC = 64;
constexpr int EPI_WARPS = 8, WP_TMA = 8, WP_MMA = 9, NT = 10 * 32;
enum Mode { SD = 0, DQ = 1, ROWS = 2 };

template <int NC_>
struct Cfg {
    static constexpr int NC = NC_;
    static constexpr int NSP = NC == 256 ? 4 : (NC == 128 ? 6 : 8);   // ring: 192 KB
    using LayA = decltype(tile_to_shape(UMMA::Layout_K_SW128_Atom<bfx>{}, Shape<Int<MR>, Int<KC>>{}));
    using LayBK = decltype(tile_to_shape(UMMA::Layout_K_SW128_Atom<bfx>{}, Shape<Int<NC>, Int<KC>>{}));
    using LayBMb = decltype(tile_to_shape(UMMA::Layout_K_SW128_Atom<bfx>{}, Shape<Int<KC>, Int<NC>>{}));
    using LayBM = decltype(composition(LayBMb{}, make_layout(Shape<Int<NC>, Int<KC>>{}, GenRowMajor{})));
    using MmaK = decltype(make_tiled_mma(SM100_MMA_F16BF16_SS<bfx, bfx, float, MR, NC, UMMA::Major::K, UMMA::Major::K>{}));
    using MmaM = decltype(make_tiled_mma(SM100_MMA_F16BF16_SS<bfx, bfx, float, MR, NC, UMMA::Major::K, UMMA::Major::MN>{}));
    // ROWS: A MN-major (64 k-rows x 128 m, two 64-m atom columns), B MN-major
    using LayAMb = decltype(tile_to_shape(UMMA::Layout_K_SW128_Atom<bfx>{}, Shape<Int<KC>, Int<MR>>{}));
    using LayAM = decltype(composition(LayAMb{}, make_layout(Shape<Int<MR>, Int<KC>>{}, GenRowMajor{})));
    using MmaMM = decltype(make_tiled_mma(SM100_MMA_F16BF16_SS<bfx, bfx, float, MR, NC, UMMA::Major::MN, UMMA::Major::MN>{}));
    struct alignas(1024) Smem {
        cute::array_aligned<bfx, NSP * MR * KC, 1024> a;
        cute::array_aligned<bfx, NSP * NC * KC, 1024> b;
        alignas(1024) uint8_t ystage[EPI_WARPS][2][32 * 64];
        uint64_t full[NSP], empty[NSP], acc_full[2], acc_empty[2];
        uint32_t tmem;
    };
    static constexpr int SMEM_BYTES = sizeof(Smem) + 1024;
};

struct Params {
    int mode, h, dqk, K, T;         // tokens T, heads h, kv width dqk, list width K
    const int *lists;               // [T, K] (rows of kv, -1 pads)
    int64_t lists_stride;
    float *sd;                      // SD: [T, 2h, K] fp32; ROWS: [T, K, D] fp32
    int ntiles, nmt, nnt, nk;       // items, row tiles per token, column tiles per token, K chunks
};

struct Maps {
    CUtensorMap a0;   // SD: q  (D, T*h) box (64, 64);  DQ: the grad rows (K, T*2h) box (64, 128)
    CUtensorMap a1;   // SD: dO (Dv, T*h) box (64, 64)
    CUtensorMap kv;   // (D, S) box (64, 1), SW128: the gather4 source
    CUtensorMap y;    // DQ: dq (D, h, T) box (32, 32, 1), SW64 store
    CUtensorMap bq;   // ROWS: q  (D, T*h) box (64, 64)
    CUtensorMap bdo;  // ROWS: dO (Dv, T*h) box (64, 64)
};

__device__ __forceinline__ uint32_t smem_u32(const void *p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void expect_tx(uint64_t *bar, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void tma2(const CUtensorMap *map, uint64_t *bar, uint32_t dst, int c0, int c1) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
                 ::"r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void gather4(const CUtensorMap *map, uint64_t *bar, uint32_t dst, int col, int4 r) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4, %5, %6}], [%7];\n"
                 ::"r"(dst), "l"(map), "r"(col), "r"(r.x), "r"(r.y), "r"(r.z), "r"(r.w), "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void umma(uint32_t d, uint64_t da, uint64_t db, uint32_t idesc, uint32_t acc) {
    asm volatile("{\n\t.reg .pred p, lead;\n\telect.sync _|lead, -1;\n\tsetp.ne.b32 p, %4, 0;\n\t"
                 "@lead tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
                 :: "r"(d), "l"(da), "l"(db), "r"(idesc), "r"(acc) : "memory");
}
__device__ __forceinline__ uint32_t pack_bf16(float a, float b) {
    uint32_t r;
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(r) : "f"(a), "f"(b));
    return r;
}
__device__ __forceinline__ void tmem_ld32(uint32_t ta, uint32_t (&v)[32]) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];\n"
        : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]), "=r"(v[7]),
          "=r"(v[8]), "=r"(v[9]), "=r"(v[10]), "=r"(v[11]), "=r"(v[12]), "=r"(v[13]), "=r"(v[14]), "=r"(v[15]),
          "=r"(v[16]), "=r"(v[17]), "=r"(v[18]), "=r"(v[19]), "=r"(v[20]), "=r"(v[21]), "=r"(v[22]), "=r"(v[23]),
          "=r"(v[24]), "=r"(v[25]), "=r"(v[26]), "=r"(v[27]), "=r"(v[28]), "=r"(v[29]), "=r"(v[30]), "=r"(v[31])
        : "r"(ta));
}

// item -> (token, row tile, column tile): column tile fastest (the token's list rows stay hot in L1 / L2)
template <int MODE, int NC_T>
__global__ void __launch_bounds__(NT, 1) ggemm_kernel(const __grid_constant__ Maps maps, const __grid_constant__ Params p) {
    using C = Cfg<NC_T>;
    constexpr int NC = C::NC, NSP = C::NSP;
    extern __shared__ uint8_t raw[];
    typename C::Smem &s = *reinterpret_cast<typename C::Smem *>((reinterpret_cast<uintptr_t>(raw) + 1023) & ~uintptr_t(1023));
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    if (tid == 0) {
        for (int i = 0; i < NSP; ++i) {
            Bar::init(&s.full[i], 1);
            Bar::init(&s.empty[i], 1);
        }
        for (int i = 0; i < 2; ++i) {
            Bar::init(&s.acc_full[i], 1);
            Bar::init(&s.acc_empty[i], EPI_WARPS * 32);
        }
        cutlass::arch::fence_barrier_init();
    }
    cute::TMEM::Allocator1Sm alloc;
    if (warp == 0) alloc.allocate(2 * NC, &s.tmem);
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    const uint32_t tmem = s.tmem;
    const uint32_t ab = smem_u32(s.a.data()), bb = smem_u32(s.b.data());
    const int nk = p.nk;

    if (warp == WP_TMA) {
        if (cute::elect_one_sync()) {
            int c = 0;
            for (int it = blockIdx.x; it < p.ntiles; it += gridDim.x) {
                const int nt = it % p.nnt, mt = (it / p.nnt) % p.nmt, t = it / (p.nnt * p.nmt);
                const int *lst = p.lists + (int64_t)t * p.lists_stride;
                for (int kt = 0; kt < nk; ++kt, ++c) {
                    const int st = c % NSP, k0 = kt * KC;
                    if (c >= NSP) Bar::wait(&s.empty[st], ((c / NSP) - 1) & 1);
                    expect_tx(&s.full[st], (MR + NC) * KC * 2);
                    const uint32_t ad = ab + st * (MR * KC * 2), bd = bb + st * (NC * KC * 2);
                    if (MODE == SD) {
                        // A rows = [q heads ; dO heads] of token t, 64-row boxes (kmaj<128>: rows 64.. at +64*128)
#pragma unroll
                        for (int half = 0; half < 2; ++half) {
                            const int r = mt * MR + half * 64;              // row of the virtual [q ; dO]
                            const bool isq = r < p.h;
                            tma2(isq ? &maps.a0 : &maps.a1, &s.full[st], ad + half * 64 * 128, k0, t * p.h + (isq ? r : r - p.h));
                        }
                        // B rows = kv[list[t, n0 + i]], the 64 dims of chunk kt (K-major, NC rows)
                        const int n0 = nt * NC;
                        for (int g = 0; g < NC / 4; ++g) {
                            const int4 rr = __ldg(reinterpret_cast<const int4 *>(lst + n0 + 4 * g));
                            gather4(&maps.kv, &s.full[st], bd + g * 4 * 128, k0, rr);
                        }
                    } else if (MODE == ROWS) {
                        // A = rows k0.. of the token's [dS ; P] (2h x K) over keys m0..m0+128: two 64-key boxes
                        const int m0 = mt * MR;
#pragma unroll
                        for (int j = 0; j < 2; ++j) tma2(&maps.a0, &s.full[st], ad + j * (64 * 128), m0 + 64 * j, t * 2 * p.h + k0);
                        // B = rows k0.. of [q ; dO] (64 heads of one tensor) x the NC dims of this column tile
                        const bool isq = k0 < p.h;
                        const int hr = t * p.h + (isq ? k0 : k0 - p.h);
#pragma unroll
                        for (int j = 0; j < NC / 64; ++j) tma2(isq ? &maps.bq : &maps.bdo, &s.full[st], bd + j * (64 * 128), nt * NC + 64 * j, hr);
                    } else {
                        // A rows = the grad kernel's rows of token t (dS then P), K = list slots
                        tma2(&maps.a0, &s.full[st], ad, k0, t * 2 * p.h + mt * MR);
                        // B = kv[list[t, k0 + i]] (64 rows = the chunk's slots) x the NC dims of this column tile (MN-major)
                        const int n0 = nt * NC;
                        for (int g = 0; g < KC / 4; ++g) {
                            const int4 rr = __ldg(reinterpret_cast<const int4 *>(lst + k0 + 4 * g));
#pragma unroll
                            for (int j = 0; j < NC / 64; ++j)
                                gather4(&maps.kv, &s.full[st], bd + j * (64 * 128) + g * 4 * 128, n0 + 64 * j, rr);
                        }
                    }
                }
            }
        }
        __syncwarp();
    } else if (warp == WP_MMA) {
        uint64_t dA[NSP], dB[NSP];
        {
            typename C::MmaK mk;
            typename C::MmaM mm;
            auto tk = mk.get_slice(0);
            auto tm = mm.get_slice(0);
#pragma unroll
            for (int st = 0; st < NSP; ++st) {
                if (MODE == ROWS) {
                    typename C::MmaMM mmm;
                    auto tmm = mmm.get_slice(0);
                    Tensor sab = make_tensor(make_smem_ptr(s.a.data() + st * MR * KC), typename C::LayAMb{});
                    Tensor sa = make_tensor(sab.data(), typename C::LayAM{});
                    dA[st] = uint64_t(tmm.partition_fragment_A(sa).data().desc_);
                } else {
                    Tensor sa = make_tensor(make_smem_ptr(s.a.data() + st * MR * KC), typename C::LayA{});
                    dA[st] = uint64_t(tk.partition_fragment_A(sa).data().desc_);
                }
                if (MODE == SD) {
                    Tensor sb = make_tensor(make_smem_ptr(s.b.data() + st * NC * KC), typename C::LayBK{});
                    dB[st] = uint64_t(tk.partition_fragment_B(sb).data().desc_);
                } else {
                    Tensor sbb = make_tensor(make_smem_ptr(s.b.data() + st * NC * KC), typename C::LayBMb{});
                    Tensor sb = make_tensor(sbb.data(), typename C::LayBM{});
                    dB[st] = uint64_t(tm.partition_fragment_B(sb).data().desc_);
                }
            }
        }
        const uint32_t idesc = MODE == SD
            ? uint32_t(UMMA::make_instr_desc<bfx, bfx, float, MR, NC, UMMA::Major::K, UMMA::Major::K>())
            : MODE == DQ ? uint32_t(UMMA::make_instr_desc<bfx, bfx, float, MR, NC, UMMA::Major::K, UMMA::Major::MN>())
                         : uint32_t(UMMA::make_instr_desc<bfx, bfx, float, MR, NC, UMMA::Major::MN, UMMA::Major::MN>());
        int c = 0, j = 0;
        for (int it = blockIdx.x; it < p.ntiles; it += gridDim.x, ++j) {
            const int buf = j & 1;
            Bar::wait(&s.acc_empty[buf], (j >> 1) & 1);
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
            for (int kt = 0; kt < nk; ++kt, ++c) {
                const int st = c % NSP;
                Bar::wait(&s.full[st], (c / NSP) & 1);
                asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                uint64_t da = dA[0], db = dB[0];
#pragma unroll
                for (int i = 1; i < NSP; ++i)
                    if (st == i) {
                        da = dA[i];
                        db = dB[i];
                    }
#pragma unroll
                for (int kk = 0; kk < KC / 16; ++kk)
                    umma(tmem + buf * NC, da + uint64_t(MODE == ROWS ? kk * 128 : kk * 2), db + uint64_t(MODE == SD ? kk * 2 : kk * 128), idesc,
                         (kt | kk) ? 1u : 0u);
                cutlass::arch::umma_arrive(&s.empty[st]);
            }
            cutlass::arch::umma_arrive(&s.acc_full[buf]);
        }
    } else if (warp < EPI_WARPS) {
        const int quad = warp & 3, half = warp >> 2;
        constexpr int CH = NC / 64;
        const uint32_t lanebase = tmem + (static_cast<uint32_t>(quad * 32) << 16);
        int slot = 0;
        asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
        Bar::arrive(&s.acc_empty[0]);
        Bar::arrive(&s.acc_empty[1]);
        int j = 0;
        for (int it = blockIdx.x; it < p.ntiles; it += gridDim.x, ++j) {
            const int buf = j & 1;
            const int nt = it % p.nnt, mt = (it / p.nnt) % p.nmt, t = it / (p.nnt * p.nmt);
            Bar::wait(&s.acc_full[buf], (j >> 1) & 1);
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
            const int row = mt * MR + quad * 32 + lane;          // row of the token's output
#pragma unroll 1
            for (int cc = 0; cc < CH; ++cc) {
                const int col = nt * NC + (half * CH + cc) * 32;
                uint32_t v[32];
                tmem_ld32(lanebase + buf * NC + (half * CH + cc) * 32, v);
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                if (MODE != DQ) {
                    // fp32 rows: transpose the warp's 32 x 32 chunk through SMEM (XOR-swizzled, conflict-free) so each
                    // store instruction writes one row's 128 contiguous bytes with all 32 lanes
                    float *stg = reinterpret_cast<float *>(s.ystage[warp][0]);
#pragma unroll
                    for (int c = 0; c < 32; ++c) stg[lane * 32 + (c ^ lane)] = __uint_as_float(v[c]);
                    __syncwarp();
                    const int rbase = mt * MR + quad * 32;
                    const int nrows = MODE == ROWS ? p.K : 2 * p.h;
                    float *gbase = MODE == ROWS ? p.sd + (int64_t)t * p.K * p.dqk + col + lane
                                                : p.sd + (int64_t)t * 2 * p.h * p.K + col + lane;
                    const int64_t ld = MODE == ROWS ? p.dqk : p.K;
#pragma unroll 4
                    for (int r = 0; r < 32; ++r)
                        if (rbase + r < nrows) gbase[(int64_t)(rbase + r) * ld] = stg[r * 32 + (lane ^ r)];
                    __syncwarp();
                } else {
                    // bf16 dq rows < h: stage 32 x 32 (SW64) and TMA-store into dq[t] (rows >= h clipped)
                    const int sb = slot;
                    slot ^= 1;
                    if (lane == 0) asm volatile("cp.async.bulk.wait_group.read 1;" ::: "memory");
                    __syncwarp();
                    const uint32_t yb = smem_u32(s.ystage[warp][sb]);
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        const uint32_t a = yb + lane * 64 + ((q ^ ((lane >> 1) & 3)) << 4);
                        asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" :: "r"(a),
                                     "r"(pack_bf16(__uint_as_float(v[8 * q + 0]), __uint_as_float(v[8 * q + 1]))),
                                     "r"(pack_bf16(__uint_as_float(v[8 * q + 2]), __uint_as_float(v[8 * q + 3]))),
                                     "r"(pack_bf16(__uint_as_float(v[8 * q + 4]), __uint_as_float(v[8 * q + 5]))),
                                     "r"(pack_bf16(__uint_as_float(v[8 * q + 6]), __uint_as_float(v[8 * q + 7]))) : "memory");
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                    if (lane == 0 && mt * MR + quad * 32 < p.h) {
                        asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%0, {%1, %2, %3}], [%4];"
                                     :: "l"(&maps.y), "r"(col), "r"(mt * MR + quad * 32), "r"(t), "r"(yb) : "memory");
                    }
                    if (lane == 0) asm volatile("cp.async.bulk.commit_group;" ::: "memory");
                }
            }
            asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
            Bar::arrive(&s.acc_empty[buf]);
        }
        if (MODE == DQ && lane == 0) asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
    }
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();
    if (warp == 0) {
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        alloc.release_allocation_lock();
        alloc.free(tmem, 2 * NC);
    }
}

}  // namespace sparse_sm100::ggemm
