#pragma once
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_scan.cuh>

// Integer-only metadata fusion for at most 256 routes and 256 experts.
// Sorting (expert, original route) retains token order within each expert.
template <class I, bool RADIX>
__global__ void
small_routes_kernel(const I *ids, const float *probs, int routes, int experts,
                    int topk, int64_t *sorted, float *permuted_probs,
                    int *combine_rows, int *offsets, int *tiles, int *total) {
  using Sort = cub::BlockRadixSort<unsigned, 256, 1, unsigned>;
  using Scan = cub::BlockScan<int, 256>;
  __shared__ union {
    typename Sort::TempStorage sort;
    typename Scan::TempStorage scan;
  } tmp;
  __shared__ int counts[256];
  __shared__ unsigned keys[64];
  int lane = threadIdx.x;
  counts[lane] = 0;
  __syncthreads();
  unsigned key[1] = {0xffffffffu}, value[1] = {unsigned(lane)};
  if (lane < routes) {
    int expert = ids[lane];
    assert(expert >= 0 && expert < experts);
    key[0] = expert * 256 + lane;
    atomicAdd(counts + expert, 1);
  }
  int position = lane;
  if constexpr (RADIX)
    Sort(tmp.sort).Sort(key, value, 0, 17);
  else {
    if (lane < routes)
      keys[lane] = key[0];
    __syncthreads();
    if (lane < routes) {
      position = 0;
      for (int j = 0; j < routes; ++j)
        position += keys[j] < key[0];
    }
  }
  if (lane < routes) {
    int route = value[0], token = route / topk, slot = route % topk,
        expert = key[0] / 256;
    sorted[position] = token;
    reinterpret_cast<unsigned *>(permuted_probs)[position] =
        reinterpret_cast<const unsigned *>(probs)[route];
    int canonical = 0;
    for (int j = 0; j < topk; ++j) {
      int other = ids[token * topk + j];
      canonical += other < expert || (other == expert && j < slot);
    }
    combine_rows[token * topk + canonical] = position;
  }
  __syncthreads();
  int count = lane < experts ? counts[lane] : 0, prefix, aggregate;
  Scan(tmp.scan).ExclusiveSum(count, prefix, aggregate);
  if (lane < experts)
    offsets[lane] = prefix;
  if (!lane)
    offsets[experts] = aggregate;
  __syncthreads();
  int bm = routes <= experts * 16 ? 16 : routes <= experts * 64 ? 64 : 128;
  int nt = (count + bm - 1) / bm, tp, ta;
  Scan(tmp.scan).ExclusiveSum(nt, tp, ta);
  for (int j = 0; j < nt; ++j) {
    tiles[3 * (tp + j)] = lane;
    tiles[3 * (tp + j) + 1] = prefix + j * bm;
    tiles[3 * (tp + j) + 2] = prefix + count;
  }
  if (!lane)
    *total = ta;
}

__global__ void route_metadata_kernel(const int *ids, const float *probs,
                                      const int *inverse, int64_t *sorted,
                                      float *permuted_probs, int *rows,
                                      int routes, int topk) {
  int row = blockIdx.x * 256 + threadIdx.x;
  if (row >= routes)
    return;
  int route = inverse[row], token = route / topk, slot = route % topk,
      expert = ids[route], canonical = 0;
  sorted[row] = token;
  reinterpret_cast<unsigned *>(permuted_probs)[row] =
      reinterpret_cast<const unsigned *>(probs)[route];
  for (int j = 0; j < topk; ++j) {
    int other = ids[token * topk + j];
    canonical += other < expert || (other == expert && j < slot);
  }
  rows[token * topk + canonical] = row;
}
