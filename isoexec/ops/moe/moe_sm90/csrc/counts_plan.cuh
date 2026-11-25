// Build offsets and compact tile descriptors from device-side expert counts.
// The host allocates an upper bound; the GEMM reads the actual tile count.
#pragma once
#include <cub/block/block_scan.cuh>

template <class C>
__global__ void counts_plan_kernel(const C *counts, int *offsets, int *tiles,
                                   int *total, int experts, int rows) {
  using Scan = cub::BlockScan<int64_t, 256>;
  __shared__ Scan::TempStorage tmp;
  __shared__ int64_t row_carry, tile_carry;
  if (!threadIdx.x) {
    row_carry = 0;
    tile_carry = 0;
  }
  __syncthreads();
  int bm = rows <= int64_t(experts) * 16   ? 16
           : rows <= int64_t(experts) * 64 ? 64
                                           : 128;
  for (int base = 0; base < experts; base += 256) {
    int e = base + threadIdx.x;
    int64_t count = e < experts ? counts[e] : 0, prefix, aggregate;
    assert(count >= 0 && count <= rows);
    Scan(tmp).ExclusiveSum(count, prefix, aggregate);
    int64_t start = row_carry + prefix, end = start + count;
    assert(end <= rows);
    if (e < experts)
      offsets[e] = start;
    __syncthreads();
    if (!threadIdx.x)
      row_carry += aggregate;
    int64_t nt = (count + bm - 1) / bm, tp, ta;
    Scan(tmp).ExclusiveSum(nt, tp, ta);
    int tile = tile_carry + tp;
    for (int j = 0; j < nt; ++j) {
      tiles[3 * (tile + j)] = e;
      tiles[3 * (tile + j) + 1] = start + j * bm;
      tiles[3 * (tile + j) + 2] = end;
    }
    __syncthreads();
    if (!threadIdx.x)
      tile_carry += ta;
    __syncthreads();
  }
  if (!threadIdx.x) {
    assert(row_carry == rows);
    offsets[experts] = rows;
    *total = tile_carry;
  }
}
