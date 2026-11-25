// C[M,N] = A[M,K] @ B[N,K]^T with FP32 accumulation. Fixed K segments start
// from zero and merge with a balanced adjacent-pair FP32 tree before one BF16
// round. Segment size depends only on K, N, and dtype; CTA grouping cannot
// change the result. TMA bounds exclude padded M/N rows from memory access.
// The build disables contraction and flushing to preserve the declared tree.

#pragma once

#include <cstdint>

namespace isoexec_mm_sm100 {

// K granularity of every segment boundary. Every mainloop K block used by any
// configuration divides this, so a segment is always a whole number of mainloop
// K blocks and the ascending 16-wide UMMA chain inside a segment is the same
// chain no matter which configuration runs it.
constexpr int kSegmentAlign = 256;

// Largest split this build admits. The tree is balanced, so S is a power of two.
constexpr int kMaxSplit = 8;

}  // namespace isoexec_mm_sm100
