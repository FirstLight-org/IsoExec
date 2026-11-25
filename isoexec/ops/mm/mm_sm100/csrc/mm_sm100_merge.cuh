// The one declared merge of the sm100 mm recipe.

#pragma once

namespace isoexec_mm_sm100 {

// N FP32 partials in v[0..N), returns the tree root.
template <int N>
__device__ __host__ __forceinline__ float merge_tree(const float (&v)[N]) {
  if constexpr (N == 1) {
    return v[0];
  } else {
    constexpr int W = (N + 1) / 2;
    float out[W];
#pragma unroll
    for (int i = 0; i + 1 < N; i += 2) out[i / 2] = v[i] + v[i + 1];
    if constexpr (N & 1) out[W - 1] = v[N - 1];
    return merge_tree<W>(out);
  }
}

// The same tree applied independently to NCOL columns. Each column is handed to
// the scalar form above, so the two cannot disagree.
template <int NSEG, int NCOL>
__device__ __forceinline__ void merge_tree_cols(const float src[NSEG][NCOL],
                                                float (&dst)[NCOL]) {
#pragma unroll
  for (int c = 0; c < NCOL; ++c) {
    float t[NSEG];
#pragma unroll
    for (int i = 0; i < NSEG; ++i) t[i] = src[i][c];
    dst[c] = merge_tree<NSEG>(t);
  }
}

}  // namespace isoexec_mm_sm100
