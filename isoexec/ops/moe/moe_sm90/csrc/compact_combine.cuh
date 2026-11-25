#pragma once
#include "wgmma.cuh"

// Eight columns per thread reduce CTA and memory-instruction overhead. Each
// column still starts with the first root and uses ascending FP32 add.rn.
template <class T>
__global__ void compact_combine_vector(const T *input, const int *rows,
                                       T *output, int hidden, int topk) {
  int token = blockIdx.x, vector = blockIdx.y * 128 + threadIdx.x;
  if (vector >= hidden / 8)
    return;
  auto load = [&](int j) {
    return reinterpret_cast<const uint4 *>(
        input + int64_t(rows[token * topk + j]) * hidden)[vector];
  };
  uint4 first = load(0);
  float acc[8] = {unpack_leaf<T>(first.x, 0), unpack_leaf<T>(first.x, 1),
                  unpack_leaf<T>(first.y, 0), unpack_leaf<T>(first.y, 1),
                  unpack_leaf<T>(first.z, 0), unpack_leaf<T>(first.z, 1),
                  unpack_leaf<T>(first.w, 0), unpack_leaf<T>(first.w, 1)};
  for (int j = 1; j < topk; ++j) {
    uint4 v = load(j);
    unsigned pair[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int c = 0; c < 8; ++c)
      acc[c] = __fadd_rn(acc[c], unpack_leaf<T>(pair[c / 2], c % 2));
  }
  reinterpret_cast<uint4 *>(output + int64_t(token) * hidden)[vector] =
      make_uint4(pack_leaf<T>(acc[0], acc[1]), pack_leaf<T>(acc[2], acc[3]),
                 pack_leaf<T>(acc[4], acc[5]), pack_leaf<T>(acc[6], acc[7]));
}
