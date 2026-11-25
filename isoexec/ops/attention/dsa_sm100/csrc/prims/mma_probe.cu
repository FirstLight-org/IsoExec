// SPDX-License-Identifier: MIT
// IsoExec dsa_sm100 (tests only): a one-k-step tcgen05 kind::f16 oracle, to fit the accumulation model
// of reference.py.  D = C + A.B^T with A [64 x 16] bf16, B [64 x 16] bf16 (both K-major), M = N = 64:
// step 1 issues A0.B0^T with ScaleOut::Zero (C := A0.B0^T), step 2 accumulates A1.B1^T onto it.  The
// caller chooses A0/B0 so that C is any value it wants (one nonzero product per element).
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cute/tensor.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cutlass/arch/barrier.h>

using namespace cute;
using bf16 = cutlass::bfloat16_t;

using SL = decltype(tile_to_shape(UMMA::Layout_K_SW32_Atom<bf16>{}, Shape<_64, _16>{}));
using MMA = decltype(make_tiled_mma(SM100_MMA_F16BF16_SS<bf16, bf16, float, 64, 64, UMMA::Major::K, UMMA::Major::K>{}));

struct __align__(1024) ProbeSmem {
    cute::array_aligned<bf16, cosize_v<SL>, 1024> a0, b0, a1, b1;
    cutlass::arch::ClusterTransactionBarrier bar;
    uint32_t tmem;
};

__global__ void probe_kernel(const bf16 *a0, const bf16 *b0, const bf16 *a1, const bf16 *b1, float *d, int steps) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    extern __shared__ __align__(1024) char sm[];
    ProbeSmem &s = *reinterpret_cast<ProbeSmem *>((reinterpret_cast<uintptr_t>(sm) + 1023) & ~uintptr_t(1023));
    const int tid = threadIdx.x;
    Tensor sA0 = make_tensor(make_smem_ptr(s.a0.data()), SL{});
    Tensor sB0 = make_tensor(make_smem_ptr(s.b0.data()), SL{});
    Tensor sA1 = make_tensor(make_smem_ptr(s.a1.data()), SL{});
    Tensor sB1 = make_tensor(make_smem_ptr(s.b1.data()), SL{});
    for (int i = tid; i < 64 * 16; i += blockDim.x) {
        const int r = i / 16, c = i % 16;
        sA0(r, c) = a0[i]; sB0(r, c) = b0[i]; sA1(r, c) = a1[i]; sB1(r, c) = b1[i];
    }
    if (tid == 0) { s.bar.init(1); cutlass::arch::fence_barrier_init(); }
    cutlass::arch::fence_view_async_shared();
    __syncthreads();
    cute::TMEM::Allocator1Sm alloc;
    if (tid / 32 == 0) { alloc.allocate(64, &s.tmem); __syncwarp(); }
    asm volatile("tcgen05.fence::before_thread_sync;");
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");
    const uint32_t tm = s.tmem;
    if (tid / 32 == 0) {
        MMA mma;
        auto thr = mma.get_slice(0);
        Tensor tA0 = thr.partition_fragment_A(sA0), tB0 = thr.partition_fragment_B(sB0);
        Tensor tA1 = thr.partition_fragment_A(sA1), tB1 = thr.partition_fragment_B(sB1);
        Tensor tC = thr.make_fragment_C(partition_shape_C(mma, Shape<_64, _64>{}));
        tC.data() = tm;
        mma.accumulate_ = UMMA::ScaleOut::Zero;
        cute::gemm(mma, tA0, tB0, tC);
        if (steps > 1) {
            mma.accumulate_ = UMMA::ScaleOut::One;
            cute::gemm(mma, tA1, tB1, tC);
        }
        cutlass::arch::umma_arrive(reinterpret_cast<uint64_t *>(&s.bar));
    }
    s.bar.wait(0);
    asm volatile("tcgen05.fence::after_thread_sync;");
    // read D: warp w, lane l -> data path 32w + l; rows 16w + l for l < 16 (the M = 64 half layout)
    if (tid < 128) {
        const int w = tid / 32, l = tid % 32;
        uint32_t v[16];
        for (int c0 = 0; c0 < 64; c0 += 16) {
            SM100_TMEM_LOAD_32dp32b16x::copy(tm + ((uint32_t)(w * 32) << 16) + c0, v[0], v[1], v[2], v[3], v[4], v[5],
                                             v[6], v[7], v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15]);
            cutlass::arch::fence_view_async_tmem_load();
            if (l < 16) for (int j = 0; j < 16; ++j) d[(w * 16 + l) * 64 + c0 + j] = __uint_as_float(v[j]);
        }
    }
    asm volatile("tcgen05.fence::before_thread_sync;");
    __syncthreads();
    if (tid / 32 == 0) { alloc.release_allocation_lock(); alloc.free(tm, 64); }
#endif
}

// round 4: the same two steps with tcgen05.mma.ws (weight-stationary, full rate at M = 64), dumping the
// RAW accumulator (128 lanes x 64 columns) so the test can recover the D layout and compare bits.
__global__ void probe_ws_kernel(const bf16 *a0, const bf16 *b0, const bf16 *a1, const bf16 *b1, float *d, int steps) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    extern __shared__ __align__(1024) char sm[];
    const uint32_t mis = static_cast<uint32_t>(__cvta_generic_to_shared(sm)) & 1023u;
    ProbeSmem &s = *reinterpret_cast<ProbeSmem *>(sm + ((1024u - mis) & 1023u));
    const int tid = threadIdx.x;
    Tensor sA0 = make_tensor(make_smem_ptr(s.a0.data()), SL{});
    Tensor sB0 = make_tensor(make_smem_ptr(s.b0.data()), SL{});
    Tensor sA1 = make_tensor(make_smem_ptr(s.a1.data()), SL{});
    Tensor sB1 = make_tensor(make_smem_ptr(s.b1.data()), SL{});
    for (int i = tid; i < 64 * 16; i += blockDim.x) {
        const int r = i / 16, c = i % 16;
        sA0(r, c) = a0[i]; sB0(r, c) = b0[i]; sA1(r, c) = a1[i]; sB1(r, c) = b1[i];
    }
    if (tid == 0) { s.bar.init(1); cutlass::arch::fence_barrier_init(); }
    cutlass::arch::fence_view_async_shared();
    __syncthreads();
    cute::TMEM::Allocator1Sm alloc;
    if (tid / 32 == 0) { alloc.allocate(64, &s.tmem); __syncwarp(); }
    asm volatile("tcgen05.fence::before_thread_sync;");
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");
    const uint32_t tm = s.tmem;
    // zero the whole 128 x 64 region first (lanes the ws layout may not write stay 0)
    if (tid < 128) {
        const int w = tid / 32;
        for (int c0 = 0; c0 < 64; c0 += 16)
            SM100_TMEM_STORE_32dp32b16x::copy(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                                              tm + ((uint32_t)(w * 32) << 16) + c0);
        cutlass::arch::fence_view_async_tmem_store();
    }
    asm volatile("tcgen05.fence::before_thread_sync;");
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");
    if (tid / 32 == 0) {
        MMA mma;
        auto thr = mma.get_slice(0);
        const uint64_t da0 = uint64_t(thr.partition_fragment_A(sA0).data().desc_);
        const uint64_t db0 = uint64_t(thr.partition_fragment_B(sB0).data().desc_);
        const uint64_t da1 = uint64_t(thr.partition_fragment_A(sA1).data().desc_);
        const uint64_t db1 = uint64_t(thr.partition_fragment_B(sB1).data().desc_);
        const uint32_t idesc = uint32_t(UMMA::make_instr_desc<bf16, bf16, float, 64, 64, UMMA::Major::K, UMMA::Major::K>());
        if (cute::elect_one_sync()) {
            asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
                         "tcgen05.mma.ws.cta_group::1.kind::f16 [%0], %1, %2, %3, p, 0;\n\t}\n"
                         :: "r"(tm), "l"(da0), "l"(db0), "r"(idesc), "r"(0u));
            if (steps > 1)
                asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
                             "tcgen05.mma.ws.cta_group::1.kind::f16 [%0], %1, %2, %3, p, 0;\n\t}\n"
                             :: "r"(tm), "l"(da1), "l"(db1), "r"(idesc), "r"(1u));
        }
        __syncwarp();
        cutlass::arch::umma_arrive(reinterpret_cast<uint64_t *>(&s.bar));
    }
    s.bar.wait(0);
    asm volatile("tcgen05.fence::after_thread_sync;");
    if (tid < 128) {
        const int w = tid / 32, l = tid % 32;
        uint32_t v[16];
        for (int c0 = 0; c0 < 64; c0 += 16) {
            SM100_TMEM_LOAD_32dp32b16x::copy(tm + ((uint32_t)(w * 32) << 16) + c0, v[0], v[1], v[2], v[3], v[4], v[5],
                                             v[6], v[7], v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15]);
            cutlass::arch::fence_view_async_tmem_load();
            for (int j = 0; j < 16; ++j) d[(w * 32 + l) * 64 + c0 + j] = __uint_as_float(v[j]);
        }
    }
    asm volatile("tcgen05.fence::before_thread_sync;");
    __syncthreads();
    if (tid / 32 == 0) { alloc.release_allocation_lock(); alloc.free(tm, 64); }
#endif
}

torch::Tensor mma_probe_ws(const torch::Tensor &a0, const torch::Tensor &b0, const torch::Tensor &a1,
                           const torch::Tensor &b1, int64_t steps) {
    auto d = torch::empty({128, 64}, a0.options().dtype(torch::kFloat));
    const size_t smem = sizeof(ProbeSmem) + 1024;
    cudaFuncSetAttribute(probe_ws_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    probe_ws_kernel<<<1, 128, smem, c10::cuda::getCurrentCUDAStream()>>>(
        (const bf16 *)a0.data_ptr(), (const bf16 *)b0.data_ptr(), (const bf16 *)a1.data_ptr(),
        (const bf16 *)b1.data_ptr(), d.data_ptr<float>(), (int)steps);
    return d;
}

torch::Tensor mma_probe(const torch::Tensor &a0, const torch::Tensor &b0, const torch::Tensor &a1,
                        const torch::Tensor &b1, int64_t steps) {
    auto d = torch::empty({64, 64}, a0.options().dtype(torch::kFloat));
    const size_t smem = sizeof(ProbeSmem) + 1024;
    cudaFuncSetAttribute(probe_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    probe_kernel<<<1, 128, smem, c10::cuda::getCurrentCUDAStream()>>>(
        (const bf16 *)a0.data_ptr(), (const bf16 *)b0.data_ptr(), (const bf16 *)a1.data_ptr(),
        (const bf16 *)b1.data_ptr(), d.data_ptr<float>(), (int)steps);
    return d;
}

// ---------------------------------------------------------------------------------------------------
// round 5: a K = 16 n chain (n k-steps, n % 4 == 0) of S = A.B^T, M = N = 64, into ONE fp32 accumulator,
// issued in one of several forms; the raw accumulator is dumped (128 lanes x 64 columns) and the test
// compares bits after undoing the layout:
//   mode 0: non-ws SS, one chain                     mode 1: non-ws SS, split n/2 + n/2 (second part accumulates)
//   mode 2: non-ws SS, split (n-4) + 4 (latent, rope)  mode 3: ws SS, one chain (fold-2 D)
//   mode 4: non-ws TS, A from TMEM (DP+0 half, k-step k at column 64 + 8k), one chain
//   mode 5: non-ws, k-steps [0, n/2) TS + [n/2, n) SS into the same accumulator
//   mode 6: ws SS split n/2 + n/2
//   mode 7: ws TS first half + SS second half; mode 8: ws TS whole chain
//   modes 9/10: non-ws/ws mixed chain, first 256 Q dims at col256, D at384; O reserves cols0..255
using SLC = decltype(tile_to_shape(UMMA::Layout_K_SW128_Atom<bf16>{}, Shape<_64, Int<576>>{}));

struct __align__(1024) ChainSmem {
    cute::array_aligned<bf16, cosize_v<SLC>, 1024> a, b;
    cutlass::arch::ClusterTransactionBarrier bar;
    uint32_t tmem;
};

__device__ __forceinline__ void issue(bool ws, uint32_t d, uint64_t da, uint64_t db, uint32_t idesc, uint32_t acc) {
    if (ws)
        asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
                     "tcgen05.mma.ws.cta_group::1.kind::f16 [%0], %1, %2, %3, p, 0;\n\t}\n"
                     :: "r"(d), "l"(da), "l"(db), "r"(idesc), "r"(acc));
    else
        asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
                     "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
                     :: "r"(d), "l"(da), "l"(db), "r"(idesc), "r"(acc));
}
__device__ __forceinline__ void issue_ts(bool ws, uint32_t d, uint32_t ta, uint64_t db, uint32_t idesc, uint32_t acc) {
    if (ws)
        asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
                     "tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [%1], %2, %3, p, 0;\n\t}\n"
                     :: "r"(d), "r"(ta), "l"(db), "r"(idesc), "r"(acc));
    else asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
                 "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n\t}\n"
                 :: "r"(d), "r"(ta), "l"(db), "r"(idesc), "r"(acc));
}

__global__ void chain_kernel(const bf16 *a, const bf16 *b, float *out, int n, int mode) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    extern __shared__ __align__(1024) char sm[];
    const uint32_t mis = static_cast<uint32_t>(__cvta_generic_to_shared(sm)) & 1023u;
    ChainSmem &s = *reinterpret_cast<ChainSmem *>(sm + ((1024u - mis) & 1023u));
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int K = 16 * n;
    Tensor sA = make_tensor(make_smem_ptr(s.a.data()), SLC{});
    Tensor sB = make_tensor(make_smem_ptr(s.b.data()), SLC{});
    for (int i = tid; i < 64 * 576; i += blockDim.x) {
        const int r = i / 576, c = i % 576;
        sA(r, c) = c < K ? a[r * K + c] : bf16(0.f);
        sB(r, c) = c < K ? b[r * K + c] : bf16(0.f);
    }
    if (tid == 0) { s.bar.init(1); cutlass::arch::fence_barrier_init(); }
    cutlass::arch::fence_view_async_shared();
    __syncthreads();
    cute::TMEM::Allocator1Sm alloc;
    if (warp == 0) { alloc.allocate(512, &s.tmem); __syncwarp(); }
    asm volatile("tcgen05.fence::before_thread_sync;");
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");
    const uint32_t tm = s.tmem;
    const uint32_t dest = tm + (mode >= 9 ? 384 : 0);
    const uint32_t qbase = tm + (mode >= 9 ? 256 : 64);
    const bool ws_a = mode == 7 || mode == 8 || mode == 10;
    // zero the D region (columns 0..63, all lanes); TS modes: A (Q) into DP+0 lanes, columns 64 + 8k
    if (tid < 128) {
        for (int c0 = 0; c0 < 64; c0 += 16)
            SM100_TMEM_STORE_32dp32b16x::copy(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                                              dest + ((uint32_t)(warp * 32) << 16) + c0);
        if (mode == 4 || mode == 5 || mode == 7 || mode == 8 || mode >= 9) {
            const int m = ws_a ? (warp % 2) * 32 + lane : warp * 16 + (lane % 16);
            const int resident = mode >= 9 ? 16 : ((mode == 5 || mode == 7) ? n / 2 : n);
            for (int k = 0; k < resident; ++k) {
                uint32_t r[8];
                for (int j = 0; j < 8; ++j) {
                    bf16 two[2] = {(ws_a || lane < 16) ? a[m * K + 16 * k + 2 * j] : bf16(0.f), (ws_a || lane < 16) ? a[m * K + 16 * k + 2 * j + 1] : bf16(0.f)};
                    r[j] = *reinterpret_cast<uint32_t *>(two);
                }
                SM100_TMEM_STORE_32dp32b8x::copy(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7],
                                                 qbase + ((uint32_t)(warp * 32) << 16) + 8 * k);
            }
        }
        cutlass::arch::fence_view_async_tmem_store();
    }
    asm volatile("tcgen05.fence::before_thread_sync;");
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");
    if (warp == 0) {
        using MMA = decltype(make_tiled_mma(SM100_MMA_F16BF16_SS<bf16, bf16, float, 64, 64, UMMA::Major::K, UMMA::Major::K>{}));
        MMA mma;
        auto thr = mma.get_slice(0);
        const uint64_t da0 = uint64_t(thr.partition_fragment_A(sA).data().desc_);
        const uint64_t db0 = uint64_t(thr.partition_fragment_B(sB).data().desc_);
        const uint32_t idesc = uint32_t(UMMA::make_instr_desc<bf16, bf16, float, 64, 64, UMMA::Major::K, UMMA::Major::K>());
        if (cute::elect_one_sync()) {
            for (int k = 0; k < n; ++k) {
                // K-major SW128: k-step k = chunk k / 4 (8 KB = 512 desc units), +2 per 16 elements inside it
                const uint64_t off = (uint64_t)((k / 4) * 512 + (k % 4) * 2);
                const uint32_t acc = k == 0 ? 0u : 1u;
                const bool ws = mode == 3 || mode == 6 || mode == 7 || mode == 8 || mode == 10;
                if (mode == 4 || mode == 8 || ((mode == 5 || mode == 7) && k < n / 2) || (mode >= 9 && k < 16)) issue_ts(ws, dest, qbase + 8 * k, db0 + off, idesc, acc);
                else issue(ws, dest, da0 + off, db0 + off, idesc, acc);
                // the "split" modes are the same instructions; the split is only where a second issue loop
                // would start (accumulate = 1, never zero) -- reproduced here by a fence + commit-wait
                // between the parts so the two parts are separately committed chains
                if ((mode == 1 || mode == 6) && k == n / 2 - 1) {
                    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                                 :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&s.bar))) : "memory");
                    s.bar.wait(0);
                }
                if (mode == 2 && k == n - 5) {
                    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                                 :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&s.bar))) : "memory");
                    s.bar.wait(0);
                }
            }
        }
        __syncwarp();
    }
    if (warp == 0)
        cutlass::arch::umma_arrive(reinterpret_cast<uint64_t *>(&s.bar));
    __syncthreads();
    // Split modes wait before issuing the next half; do not let two barrier phases
    // complete before a waiter observes phase zero (an ABA parity deadlock).
    s.bar.wait((mode == 1 || mode == 2 || mode == 6) ? 1 : 0);
    asm volatile("tcgen05.fence::after_thread_sync;");
    if (tid < 128) {
        uint32_t v[16];
        for (int c0 = 0; c0 < 64; c0 += 16) {
            SM100_TMEM_LOAD_32dp32b16x::copy(dest + ((uint32_t)(warp * 32) << 16) + c0, v[0], v[1], v[2], v[3], v[4], v[5],
                                             v[6], v[7], v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15]);
            cutlass::arch::fence_view_async_tmem_load();
            for (int j = 0; j < 16; ++j) out[(warp * 32 + lane) * 64 + c0 + j] = __uint_as_float(v[j]);
        }
    }
    asm volatile("tcgen05.fence::before_thread_sync;");
    __syncthreads();
    if (warp == 0) { alloc.release_allocation_lock(); alloc.free(tm, 512); }
#endif
}

torch::Tensor mma_chain(const torch::Tensor &a, const torch::Tensor &b, int64_t n, int64_t mode) {
    TORCH_CHECK(n >= 8 && n % 4 == 0 && n <= 36, "n must be a multiple of four in [8,36]");
    TORCH_CHECK(mode >= 0 && mode <= 10, "invalid chain mode");
    TORCH_CHECK(mode < 9 || n >= 16, "compact placement requires at least 256 Q dimensions");
    TORCH_CHECK(a.is_cuda() && b.is_cuda() && a.device() == b.device(), "same CUDA device required");
    TORCH_CHECK(a.scalar_type() == torch::kBFloat16 && b.scalar_type() == torch::kBFloat16,
                "BF16 operands required");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "contiguous operands required");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && a.size(0) == 64 && a.size(1) == 16 * n &&
                a.sizes() == b.sizes(), "operands must both have shape [64,16*n]");
    auto d = torch::empty({128, 64}, a.options().dtype(torch::kFloat));
    const size_t smem = sizeof(ChainSmem) + 1024;
    cudaFuncSetAttribute(chain_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    chain_kernel<<<1, 128, smem, c10::cuda::getCurrentCUDAStream()>>>((const bf16 *)a.data_ptr(), (const bf16 *)b.data_ptr(),
                                                                     d.data_ptr<float>(), (int)n, (int)mode);
    return d;
}
