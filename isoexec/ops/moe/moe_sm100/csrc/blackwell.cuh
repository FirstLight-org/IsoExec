// SPDX-License-Identifier: Apache-2.0
// Blackwell instruction vocabulary: tcgen05 UMMA over a shared-memory operand pair, TMEM
// allocation and readout, TMA tile loads and mbarrier phases.
#pragma once
#include <cstdint>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/copy_sm100.hpp>
#include <cute/arch/copy_sm100_tma.hpp>
#include <cute/arch/mma_sm100_desc.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/numeric_types.h>

namespace moe100 {

using Barrier = cutlass::arch::ClusterTransactionBarrier;

//: bf16 elements in one 128-byte swizzle atom; the TMA box width and the descriptor's
//: atom stride are both derived from it.
constexpr uint32_t kAtomK = 64;
//: tcgen05's K step for 16-bit operands.
constexpr uint32_t kUmmaK = 16;
//: TMEM/UMMA row granularity for cta_group::1.
constexpr uint32_t kLayoutM = 128;

__device__ __forceinline__ uint32_t lane_id() {
  uint32_t v;
  asm("mov.u32 %0, %%laneid;" : "=r"(v));
  return v;
}

__device__ __forceinline__ void tcgen05_after_thread_sync() {
  asm volatile("tcgen05.fence::after_thread_sync;");
}

__device__ __forceinline__ void tcgen05_before_thread_sync() {
  asm volatile("tcgen05.fence::before_thread_sync;");
}

__device__ __forceinline__ uint32_t ld_shared_u32(const uint32_t *p) {
  uint32_t v;
  asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "l"(__cvta_generic_to_shared(p)));
  return v;
}

// ------------------------------------------------------------------ operands ---
//
// K-major 128-byte-swizzled shared descriptor for a sequence of (MN x kAtomK) atoms.
// One atom on K, so the leading offset is unused and the atom stride is 8 rows.
template <class Element>
__device__ __forceinline__ cute::UMMA::SmemDescriptor make_desc_k_major(const void *base) {
  cute::UMMA::SmemDescriptor d;
  d.desc_ = 0;
  d.version_ = 1;
  d.lbo_mode_ = 0;
  d.layout_type_ = static_cast<uint8_t>(cute::UMMA::LayoutType::SWIZZLE_128B);
  d.start_address_ = static_cast<uint16_t>(cute::cast_smem_ptr_to_uint(base) >> 4);
  d.base_offset_ = 0;
  d.stride_byte_offset_ = static_cast<uint16_t>((8u * kAtomK * sizeof(Element)) >> 4);
  d.leading_byte_offset_ = 0;
  return d;
}

//: element offset -> descriptor low-word delta (K-major: the stride on K is one).
template <class Element>
__device__ __forceinline__ uint32_t advance_lo(uint32_t base, uint32_t elems) {
  return base + ((elems * static_cast<uint32_t>(sizeof(Element))) >> 4);
}

// ---------------------------------------------------------------------- UMMA ---

struct Umma1Sm {
  __device__ __forceinline__ static void fma(uint64_t a, uint64_t b, uint32_t c,
                                            uint32_t scale, uint64_t idesc) {
    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
                 "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}\n" ::"r"(c),
                 "l"(a), "l"(b), "r"(static_cast<uint32_t>(idesc >> 32)), "r"(scale));
  }
};

using TmemAllocator = cute::TMEM::Allocator1Sm;

// ---------------------------------------------------------------- TMEM reads ---
//
// One tcgen05.ld per lane of N consecutive 32-bit accumulator columns. The datapath
// quarter a warp reads is its warp index modulo four, so four epilogue warps cover the
// 128 rows of one accumulator and lane `l` of quarter `q` owns row 32*q + l.
template <uint32_t N> struct TmemLd;

template <> struct TmemLd<8> {
  __device__ __forceinline__ static void copy(uint32_t addr, uint32_t *v) {
    cute::SM100::TMEM::LOAD::SM100_TMEM_LOAD_32dp32b8x::copy(addr, v[0], v[1], v[2], v[3],
                                                             v[4], v[5], v[6], v[7]);
  }
};

template <> struct TmemLd<16> {
  __device__ __forceinline__ static void copy(uint32_t addr, uint32_t *v) {
    cute::SM100::TMEM::LOAD::SM100_TMEM_LOAD_32dp32b16x::copy(
        addr, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11],
        v[12], v[13], v[14], v[15]);
  }
};

template <> struct TmemLd<32> {
  __device__ __forceinline__ static void copy(uint32_t addr, uint32_t *v) {
    cute::SM100::TMEM::LOAD::SM100_TMEM_LOAD_32dp32b32x::copy(
        addr, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11],
        v[12], v[13], v[14], v[15], v[16], v[17], v[18], v[19], v[20], v[21], v[22],
        v[23], v[24], v[25], v[26], v[27], v[28], v[29], v[30], v[31]);
  }
};

// ----------------------------------------------------------------- host: TMA ---

#ifndef __CUDACC_RTC__
inline PFN_cuTensorMapEncodeTiled_v12000 tma_encode_fn() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = nullptr;
  if (fn == nullptr) {
    void *ptr = nullptr;
    cudaDriverEntryPointQueryResult status{};
    if (cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &ptr, cudaEnableDefault,
                                &status) != cudaSuccess ||
        ptr == nullptr)
      return nullptr;
    fn = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(ptr);
  }
  return fn;
}

//: rank-2 128B-swizzled map over a [outer, inner] row-major 16-bit tensor.
inline int make_desc2(CUtensorMap *map, const void *base, uint64_t inner, uint64_t outer,
                      uint64_t outer_stride_elems, uint32_t box_inner,
                      uint32_t box_outer) {
  auto fn = tma_encode_fn();
  if (fn == nullptr)
    return -10;
  const uint64_t dims[2] = {inner, outer};
  const uint64_t strides[1] = {outer_stride_elems * 2u};
  const uint32_t box[2] = {box_inner, box_outer};
  const uint32_t elem_strides[2] = {1, 1};
  CUresult rc = fn(map, CU_TENSOR_MAP_DATA_TYPE_UINT16, 2, const_cast<void *>(base), dims,
                   strides, box, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
                   CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                   CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return rc == CUDA_SUCCESS ? 0 : -(20 + int(rc));
}

//: rank-3 map over a [pages, outer, inner] 16-bit tensor: the expert axis is the page.
inline int make_desc3(CUtensorMap *map, const void *base, uint64_t inner, uint64_t outer,
                      uint64_t pages, uint64_t outer_stride_elems,
                      uint64_t page_stride_elems, uint32_t box_inner,
                      uint32_t box_outer) {
  auto fn = tma_encode_fn();
  if (fn == nullptr)
    return -10;
  const uint64_t dims[3] = {inner, outer, pages};
  const uint64_t strides[2] = {outer_stride_elems * 2u, page_stride_elems * 2u};
  const uint32_t box[3] = {box_inner, box_outer, 1u};
  const uint32_t elem_strides[3] = {1, 1, 1};
  CUresult rc = fn(map, CU_TENSOR_MAP_DATA_TYPE_UINT16, 3, const_cast<void *>(base), dims,
                   strides, box, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
                   CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                   CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return rc == CUDA_SUCCESS ? 0 : -(40 + int(rc));
}
#endif

} // namespace moe100
