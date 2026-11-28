// SPDX-License-Identifier: Apache-2.0
//
// Data movement around the canonical DeepGEMM MegaMoE EP launch (``mega_ep.py``). No arithmetic other than the
// 1x32 UE8M0 E4M3 activation encoding, which is ``clamped_swiglu_sm100/dense/csrc/swiglu_quant.cu``'s
// ``quant_rows`` expression for expression (same amax, same 1e-4 floor, same scale exponent, same __fdiv_rn,
// clamp and SATFINITE conversion; amax is a max, exact in any order).
//
//   prep:   one launch that fills MegaMoE's per-rank input buffer for an n-row call: E4M3 codes (x), UE8M0
//           scales (x_sf, and the shared expert's MN-major copy at the block_m-dependent row map), int64 top-k
//           ids and FP32 weights from the router's [T, k + S] output; padding rows (real <= r < n) get zero
//           codes / scales / weights and id -1 (never dispatched). It also advances the per-rank call counter
//           that tags the gather below.
//   prep_bf16: the BF16 MegaMoE's input (``moe/deepseek_v41_sm100/mega.py``): the rows copied as they are (no
//           encoding; the BF16 kernel's shared expert reads the same buffer), int64 top-k ids and FP32 weights,
//           padding rows (zeros, id -1), and the same call-counter advance. Byte copies only.
//   gather: reassembles the replicated output from every rank's MegaMoE output rows, which MegaMoE wrote into
//           the rank's symmetric output region: each rank publishes "my output of call t is complete"
//           (st.release.sys of t into every peer's flag word for this rank), waits until every peer's flag
//           for it equals t (polling only LOCAL memory), then reads the peers' rows over NVLink into the output
//           tensor. Plain loads and stores; every output row has exactly one source row.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/extension.h>

#include <cstdint>

namespace isoexec_mega_ep {
namespace {

__device__ __forceinline__ uint64_t ld_acquire_sys(const uint64_t* p) {
  uint64_t v;
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ void st_release_sys(uint64_t* p, uint64_t v) {
  asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}

// One thread per (row, 32-column group). Row r < real: encode; real <= r < n: padding.
__global__ __launch_bounds__(256) void prep_kernel(
    const __nv_bfloat16* __restrict__ x, int64_t x_stride, const int* __restrict__ ids, const float* __restrict__ w,
    int ks, int k, int real, int n, int H, int block_m, uint8_t* __restrict__ codes, uint8_t* __restrict__ x_sf,
    int64_t* __restrict__ topk_idx, float* __restrict__ topk_w, uint8_t* __restrict__ shared_sf, int64_t shared_ld,
    uint64_t* __restrict__ counter) {
  cudaGridDependencySynchronize();     // no-op unless launched as a PDL secondary (inputs come from the router)
  const int groups = H / 32;
  const int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t == 0) *counter += 1;           // stream-ordered: the gather of this call reads it after MegaMoE
  if (t >= static_cast<int64_t>(n) * groups) return;
  const int r = static_cast<int>(t / groups);
  const int g = static_cast<int>(t % groups);
  const bool live = r < real;
  uint32_t exponent = 0;
  uint8_t q[32];
  if (live) {
    float v[32];
    float amax = 0.0f;
    const uint4* src = reinterpret_cast<const uint4*>(x + r * x_stride + g * 32);
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const uint4 xq = src[c];
      const __nv_bfloat16* xv = reinterpret_cast<const __nv_bfloat16*>(&xq);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        v[c * 8 + j] = __bfloat162float(xv[j]);
        amax = fmaxf(amax, fabsf(v[c * 8 + j]));
      }
    }
    amax = fmaxf(amax, 1.0e-4f);
    float scale = __fmul_rn(amax, 1.0f / 448.0f);
    const uint32_t b = __float_as_uint(scale);
    exponent = ((b >> 23) & 255) + ((b & 0x7fffff) != 0);
    scale = __uint_as_float(exponent << 23);
#pragma unroll
    for (int j = 0; j < 32; ++j)
      q[j] = __nv_cvt_float_to_fp8(fminf(448.0f, fmaxf(-448.0f, __fdiv_rn(v[j], scale))), __NV_SATFINITE, __NV_E4M3);
  } else {
#pragma unroll
    for (int j = 0; j < 32; ++j) q[j] = 0;
  }
  uint4* dst = reinterpret_cast<uint4*>(codes + static_cast<int64_t>(r) * H + g * 32);
  dst[0] = *reinterpret_cast<const uint4*>(q);
  dst[1] = *reinterpret_cast<const uint4*>(q + 16);
  x_sf[static_cast<int64_t>(r) * groups + g] = static_cast<uint8_t>(exponent);
  // Shared expert's activation scales: int32 word (row map(r), g / 4) of an MN-major [S, H/128] tensor,
  // byte g % 4 of the word (little endian, = x_sf's byte order). Row map: ``Workspace.sf_rows``.
  const int within = r % block_m;
  const int64_t srow = static_cast<int64_t>(r / block_m) * ((block_m + 127) / 128 * 128) + (within / 128) * 128 +
                       (within % 32) * 4 + (within % 128) / 32;
  shared_sf[(srow + static_cast<int64_t>(g / 4) * shared_ld) * 4 + (g % 4)] = static_cast<uint8_t>(exponent);
  if (g < k) {
    topk_idx[static_cast<int64_t>(r) * k + g] = live ? static_cast<int64_t>(ids[static_cast<int64_t>(r) * ks + g]) : -1;
    topk_w[static_cast<int64_t>(r) * k + g] = live ? w[static_cast<int64_t>(r) * ks + g] : 0.0f;
  }
}

// One thread per (row, 16-byte chunk). Row r < real: copy; real <= r < n: padding.
__global__ __launch_bounds__(256) void prep_bf16_kernel(
    const uint4* __restrict__ x, int64_t x_stride_vec, const int* __restrict__ ids, const float* __restrict__ w,
    int ks, int k, int real, int n, int vec, uint4* __restrict__ rows, int64_t* __restrict__ topk_idx,
    float* __restrict__ topk_w, uint64_t* __restrict__ counter) {
  cudaGridDependencySynchronize();     // no-op unless launched as a PDL secondary (inputs come from the router)
  const int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t == 0) *counter += 1;           // stream-ordered: the gather of this call reads it after MegaMoE
  if (t >= static_cast<int64_t>(n) * vec) return;
  const int r = static_cast<int>(t / vec);
  const int c = static_cast<int>(t % vec);
  const bool live = r < real;
  rows[static_cast<int64_t>(r) * vec + c] = live ? x[r * x_stride_vec + c] : make_uint4(0, 0, 0, 0);
  if (c < k) {
    topk_idx[static_cast<int64_t>(r) * k + c] = live ? static_cast<int64_t>(ids[static_cast<int64_t>(r) * ks + c]) : -1;
    topk_w[static_cast<int64_t>(r) * k + c] = live ? w[static_cast<int64_t>(r) * ks + c] : 0.0f;
  }
}

// flags: this rank's [world] u64 words (peer q writes word q); ptrs: [world] base addresses of every rank's
// symmetric region (same layout); y rows live at byte offset y_off. grid.x covers output rows of this chunk.
__global__ __launch_bounds__(256) void gather_kernel(const int64_t* __restrict__ ptrs, int world, int rank,
                                                     int64_t flags_off, int64_t y_off, const uint64_t* __restrict__ counter,
                                                     int64_t M, int64_t start, int64_t per, int H,
                                                     __nv_bfloat16* __restrict__ out) {
  cudaGridDependencySynchronize();     // no-op unless launched as a PDL secondary (MegaMoE's output)
  const uint64_t tag = *counter;
  uint64_t* const my_flags = reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(ptrs[rank]) + flags_off);
  if (blockIdx.x == 0) {
    __threadfence_system();
    for (int q = threadIdx.x; q < world; q += blockDim.x)
      st_release_sys(reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(ptrs[q]) + flags_off) + rank, tag);
  }
  for (int q = threadIdx.x; q < world; q += blockDim.x) {
    const long long t0 = clock64();
    uint64_t seen;
    while ((seen = ld_acquire_sys(my_flags + q)) < tag) {
      if (clock64() - t0 > (1ll << 37)) {      // ~60 s: a peer never published this call
        printf("isoexec mega_ep gather timeout: rank=%d peer=%d tag=%llu seen=%llu\n", rank, q,
               static_cast<unsigned long long>(tag), static_cast<unsigned long long>(seen));
        __trap();
      }
    }
    if (seen != tag) {                           // a peer is a call ahead: the protocol is broken, fail closed
      printf("isoexec mega_ep gather protocol violation: rank=%d peer=%d tag=%llu seen=%llu\n", rank, q,
             static_cast<unsigned long long>(tag), static_cast<unsigned long long>(seen));
      __trap();
    }
  }
  __syncthreads();
  const int vec = H / 8;                         // uint4 per row
  // Output row index i in [0, M): owner q = the slice holding it, its row j in the slice; the row is in this
  // chunk iff start <= j < start + per.
  for (int64_t i = blockIdx.x; i < M; i += gridDim.x) {
    // home slice of row i: largest q with M*q/W <= i
    int q = static_cast<int>(((i + 1) * world - 1) / M);
    while (q > 0 && M * q / world > i) --q;
    while (q + 1 < world && M * (q + 1) / world <= i) ++q;
    const int64_t j = i - M * q / world;
    if (j < start || j >= start + per) continue;
    const uint4* src = reinterpret_cast<const uint4*>(reinterpret_cast<const uint8_t*>(ptrs[q]) + y_off) +
                       (j - start) * vec;
    uint4* dst = reinterpret_cast<uint4*>(out) + i * vec;
    // Issue every load of the row before any store (a peer row is one NVLink round trip, not one per vector:
    // with interleaved load/store the compiler must assume dst may alias src and serializes them).
    constexpr int kPer = 8;
    for (int base = 0; base < vec; base += kPer * static_cast<int>(blockDim.x)) {
      uint4 v[kPer];
#pragma unroll
      for (int u = 0; u < kPer; ++u) {
        const int c = base + u * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
        if (c < vec) v[u] = __ldcg(src + c);
      }
#pragma unroll
      for (int u = 0; u < kPer; ++u) {
        const int c = base + u * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
        if (c < vec) dst[c] = v[u];
      }
    }
  }
}

template <typename... KArgs, typename... Args>
void launch(void (*kernel)(KArgs...), unsigned blocks, unsigned threads, bool pdl, Args... args) {
  cudaLaunchConfig_t config = {};
  config.gridDim = dim3(blocks);
  config.blockDim = dim3(threads);
  config.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  config.attrs = attr;
  config.numAttrs = pdl ? 1 : 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, static_cast<KArgs>(args)...));
}

}  // namespace

void prep(const at::Tensor& x, const at::Tensor& ids, const at::Tensor& w, int64_t k, int64_t n, int64_t block_m,
          const at::Tensor& codes, const at::Tensor& x_sf, const at::Tensor& topk_idx, const at::Tensor& topk_w,
          const at::Tensor& shared_sf, const at::Tensor& counter, bool pdl) {
  const int64_t real = x.size(0);
  const int H = static_cast<int>(codes.size(1));
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.scalar_type() == at::kBFloat16 && x.stride(1) == 1 &&
                  (real == 0 || x.size(1) == H) && x.stride(0) % 8 == 0 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
              "x: BF16 [real, H], unit column stride, 16-byte aligned rows");
  TORCH_CHECK(H % 128 == 0 && n >= real && n <= codes.size(0) && k >= 1 && block_m >= 1, "prep geometry");
  TORCH_CHECK(codes.is_contiguous() && codes.element_size() == 1, "codes: [cap, H] bytes");
  TORCH_CHECK(x_sf.is_contiguous() && x_sf.scalar_type() == at::kInt && x_sf.size(1) == H / 128, "x_sf: int32 [cap, H/128]");
  TORCH_CHECK(topk_idx.is_contiguous() && topk_idx.scalar_type() == at::kLong && topk_idx.size(1) == k, "topk_idx");
  TORCH_CHECK(topk_w.is_contiguous() && topk_w.scalar_type() == at::kFloat && topk_w.size(1) == k, "topk_w");
  TORCH_CHECK(shared_sf.scalar_type() == at::kInt && shared_sf.dim() == 2 && shared_sf.stride(0) == 1 &&
                  shared_sf.size(1) == H / 128, "shared_sf: MN-major int32 [S, H/128]");
  TORCH_CHECK(counter.scalar_type() == at::kLong && counter.numel() == 1, "counter");
  int ks = static_cast<int>(k);
  const int* ids_p = nullptr;
  const float* w_p = nullptr;
  if (real) {
    TORCH_CHECK(ids.scalar_type() == at::kInt && ids.dim() == 2 && ids.stride(1) == 1 && ids.size(0) == real &&
                    ids.size(1) >= k && w.scalar_type() == at::kFloat && w.dim() == 2 && w.stride(1) == 1 &&
                    w.size(0) == real && w.stride(0) == ids.stride(0),
                "ids int32 / w fp32 [real, >= k] with equal row strides");
    ks = static_cast<int>(ids.stride(0));
    ids_p = ids.data_ptr<int>();
    w_p = w.data_ptr<float>();
  }
  // a block_m larger than the shared SF tensor's rows cannot occur (Workspace.sf_rows uses the same map)
  const at::cuda::CUDAGuard guard(codes.device());
  const int64_t threads = std::max<int64_t>(n * (H / 32), 1);
  launch(prep_kernel, static_cast<unsigned>((threads + 255) / 256), 256, pdl,
      real ? reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()) : nullptr, real ? x.stride(0) : 0, ids_p, w_p, ks,
      static_cast<int>(k), static_cast<int>(real), static_cast<int>(n), H, static_cast<int>(block_m),
      reinterpret_cast<uint8_t*>(codes.data_ptr()), reinterpret_cast<uint8_t*>(x_sf.data_ptr()),
      topk_idx.data_ptr<int64_t>(), topk_w.data_ptr<float>(), reinterpret_cast<uint8_t*>(shared_sf.data_ptr()),
      shared_sf.stride(1), reinterpret_cast<uint64_t*>(counter.data_ptr<int64_t>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void prep_bf16(const at::Tensor& x, const at::Tensor& ids, const at::Tensor& w, int64_t k, int64_t n,
               const at::Tensor& rows, const at::Tensor& topk_idx, const at::Tensor& topk_w, const at::Tensor& counter,
               bool pdl) {
  const int64_t real = x.size(0);
  TORCH_CHECK(rows.is_contiguous() && rows.scalar_type() == at::kBFloat16 && rows.dim() == 2 && rows.size(1) % 8 == 0,
              "rows: BF16 [cap, H]");
  const int64_t H = rows.size(1);
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.scalar_type() == at::kBFloat16 && x.stride(1) == 1 &&
                  (real == 0 || x.size(1) == H) && x.stride(0) % 8 == 0 &&
                  reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
              "x: BF16 [real, H], unit column stride, 16-byte aligned rows");
  TORCH_CHECK(n >= real && n <= rows.size(0) && k >= 1 && k <= H / 8, "prep_bf16 geometry");
  TORCH_CHECK(topk_idx.is_contiguous() && topk_idx.scalar_type() == at::kLong && topk_idx.size(1) == k, "topk_idx");
  TORCH_CHECK(topk_w.is_contiguous() && topk_w.scalar_type() == at::kFloat && topk_w.size(1) == k, "topk_w");
  TORCH_CHECK(counter.scalar_type() == at::kLong && counter.numel() == 1, "counter");
  int ks = static_cast<int>(k);
  const int* ids_p = nullptr;
  const float* w_p = nullptr;
  if (real) {
    TORCH_CHECK(ids.scalar_type() == at::kInt && ids.dim() == 2 && ids.stride(1) == 1 && ids.size(0) == real &&
                    ids.size(1) >= k && w.scalar_type() == at::kFloat && w.dim() == 2 && w.stride(1) == 1 &&
                    w.size(0) == real && w.stride(0) == ids.stride(0),
                "ids int32 / w fp32 [real, >= k] with equal row strides");
    ks = static_cast<int>(ids.stride(0));
    ids_p = ids.data_ptr<int>();
    w_p = w.data_ptr<float>();
  }
  const at::cuda::CUDAGuard guard(rows.device());
  const int vec = static_cast<int>(H / 8);
  const int64_t threads = std::max<int64_t>(n * vec, 1);
  launch(prep_bf16_kernel, static_cast<unsigned>((threads + 255) / 256), 256, pdl,
      real ? reinterpret_cast<const uint4*>(x.data_ptr()) : nullptr, real ? x.stride(0) / 8 : 0, ids_p, w_p, ks,
      static_cast<int>(k), static_cast<int>(real), static_cast<int>(n), vec, reinterpret_cast<uint4*>(rows.data_ptr()),
      topk_idx.data_ptr<int64_t>(), topk_w.data_ptr<float>(), reinterpret_cast<uint64_t*>(counter.data_ptr<int64_t>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gather(const at::Tensor& ptrs, int64_t rank, int64_t flags_off, int64_t y_off, const at::Tensor& counter,
            int64_t start, int64_t per, const at::Tensor& out, bool pdl) {
  TORCH_CHECK(ptrs.is_cuda() && ptrs.scalar_type() == at::kLong && ptrs.dim() == 1, "ptrs: int64 [world] on device");
  TORCH_CHECK(out.is_contiguous() && out.scalar_type() == at::kBFloat16 && out.dim() == 2 && out.size(1) % 8 == 0, "out");
  TORCH_CHECK(counter.scalar_type() == at::kLong && counter.numel() == 1, "counter");
  const int world = static_cast<int>(ptrs.numel());
  const int64_t M = out.size(0);
  TORCH_CHECK(0 <= rank && rank < world && flags_off % 8 == 0 && y_off % 16 == 0, "gather geometry");
  const at::cuda::CUDAGuard guard(out.device());
  const int64_t rows = std::max<int64_t>(std::min<int64_t>(M, static_cast<int64_t>(per) * world), 1);
  const unsigned blocks = static_cast<unsigned>(std::min<int64_t>(rows, 4096));
  launch(gather_kernel, blocks, 128, pdl, ptrs.data_ptr<int64_t>(), world, static_cast<int>(rank), flags_off, y_off,
      reinterpret_cast<const uint64_t*>(counter.data_ptr<int64_t>()), M, start, per, static_cast<int>(out.size(1)),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace isoexec_mega_ep

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prep", &isoexec_mega_ep::prep);
  m.def("prep_bf16", &isoexec_mega_ep::prep_bf16);
  m.def("gather", &isoexec_mega_ep::gather);
}
