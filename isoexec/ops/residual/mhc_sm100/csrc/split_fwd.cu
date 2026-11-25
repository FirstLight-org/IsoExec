// SPDX-License-Identifier: Apache-2.0
// mHC split forward (GLM / DeepSeek recipe): RMS statistic + 24-wide projection + pre/post/comb + collapse.
//
// Three kernels (no atomics anywhere; the only cross-CTA exchange is a plain global workspace written once):
//   proj_kernel     K1: per (row tile, h-slice) CTA, partial 24 projections + sum of squares -> ws[row][slice][25]
//   tail_kernel     K2: per row, fixed-order tree over the slices, rstd, scalar tail (pre/post/comb)
//   collapse_kernel K3: out[row][h] = bf16(sum_s pre[s] * x[s][h])
//
// NUMERICAL PROGRAM (README "Numerical program"; everything here is a function of H only, never of T):
//  * Flat reduction index k = stream*H + h over n*H = 4H elements. A K1 CTA owns h-slice [c*Hs, (c+1)*Hs) of ALL four
//    streams (Hs = H/32, NSL = 32 slices) for RT rows.
//  * Leaf = one warp = one stream over the CTA slice. A leaf accumulates, in ascending h order, 16 elements per
//    tensor-core step (thread t of a quad supplies h = 16i + 4t..4t+3; the permutation of k inside one mma is
//    consistent between the x and fn operands). The mma products are exact (bf16 x bf16 for a bf16 fn; bf16 x three exact
//    bf16 pieces of an fp32 fn, pieces consumed hi, mid, lo each step) and accumulate in fp32. The sum of squares
//    (x^2 is exact in fp32) of the same elements is a per-thread sequential fp32 chain over the same steps, then
//    (t0+t1)+(t2+t3).
//  * Slice partial = (l0+l1)+(l2+l3) over the four stream leaves; row total = balanced adjacent-pair tree over the 32
//    slices ((s0+s1)+(s2+s3))+... . The tree depends only on H.
//  * rstd = 1 / sqrt(sumsq / (4H) + rms_eps) (IEEE rn), mix = proj * rstd, then the scalar tail (tail.cuh).
//  * Collapse: ((p0*x0 + p1*x1) + p2*x2) + p3*x3 in fp32, every product and sum rounded, one rounding to bf16.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "pdl.cuh"

#include <type_traits>

#include "common.cuh"
#include "norm_segments.cuh"
#include "tail.cuh"


namespace mhc {

constexpr int kSlices = 32;   // NSL: h-slices of the reduction tree (deployment constant)
constexpr int kWsCols = 32;   // slice-partial record: 25 values (24 projections + sum of squares) padded to one 128-byte line
constexpr int kRedStr = 28;   // smem stride (floats) of a leaf record: 16-byte aligned rows

struct SplitParams {
  const __nv_bfloat16 *x;   // [T][4][H]
  const void *fn;           // [24][4H], bf16 or fp32
  const float *scale;       // [3]
  const float *base;        // [24]
  float *ws;                // [kSlices][T][kWsCols]  (slice-major: a tile's records of one slice are contiguous)
  int T, H;
  // MERGE form (merge_split_forward): x holds the PREVIOUS streams; the staged tile is the GLM merge
  // res = merge(x, m_out, m_post, m_comb) (merge.cu merge_fwd_kernel<CAST = true>'s element program), which is
  // also stored to m_res [T][4][H] (the new streams) by the CTA that owns the (tile, slice) unit.
  const __nv_bfloat16 *m_out;  // [T][H]
  const float *m_post;         // [T][4]
  const float *m_comb;         // [T][16]
  __nv_bfloat16 *m_res;        // [T][4][H]
};

// merge.cu's GLM element program (CAST = true): post/comb rounded to bf16 at the merge, t1 = bf16(post_b * out),
// t2 = bf16(((comb_b0 * x0 + comb_b1 * x1) + comb_b2 * x2) + comb_b3 * x3) with the exact-product fma chain,
// res = bf16(t1 + t2). Eight h of all four destination streams; the caller packs the rounded values.
__device__ __forceinline__ float m_bfr(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
__device__ __forceinline__ void merge8(const SplitParams &p, int64_t row, int h, uint4 (&res)[4]) {
  const int H = p.H;
  float pb[4], cb[16];
  {
    const float4 q = *reinterpret_cast<const float4 *>(p.m_post + row * 4);
    pb[0] = m_bfr(q.x); pb[1] = m_bfr(q.y); pb[2] = m_bfr(q.z); pb[3] = m_bfr(q.w);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const float4 c = *reinterpret_cast<const float4 *>(p.m_comb + row * 16 + 4 * k);
      cb[4 * k] = m_bfr(c.x); cb[4 * k + 1] = m_bfr(c.y); cb[4 * k + 2] = m_bfr(c.z); cb[4 * k + 3] = m_bfr(c.w);
    }
  }
  float xv[4][8], ov[8];
#pragma unroll
  for (int s = 0; s < 4; ++s) {
    const uint4 q = *reinterpret_cast<const uint4 *>(p.x + (row * 4 + s) * H + h);
    const uint32_t w[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) { xv[s][2 * j] = bf_lo(w[j]); xv[s][2 * j + 1] = bf_hi(w[j]); }
  }
  {
    const uint4 q = *reinterpret_cast<const uint4 *>(p.m_out + row * H + h);
    const uint32_t w[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) { ov[2 * j] = bf_lo(w[j]); ov[2 * j + 1] = bf_hi(w[j]); }
  }
#pragma unroll
  for (int d = 0; d < 4; ++d) {
    float r[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const float t1 = m_bfr(__fmul_rn(pb[d], ov[e]));
      float acc = __fmul_rn(cb[0 * 4 + d], xv[0][e]);
      acc = __fmaf_rn(cb[1 * 4 + d], xv[1][e], acc);
      acc = __fmaf_rn(cb[2 * 4 + d], xv[2][e], acc);
      acc = __fmaf_rn(cb[3 * 4 + d], xv[3][e], acc);
      r[e] = __fadd_rn(t1, m_bfr(acc));
    }
    res[d].x = pack_f2_bf16(r[0], r[1]); res[d].y = pack_f2_bf16(r[2], r[3]);
    res[d].z = pack_f2_bf16(r[4], r[5]); res[d].w = pack_f2_bf16(r[6], r[7]);
  }
}

template <int HS, int MT, bool FN32, int NS>
static constexpr size_t proj_smem_bytes() {
  return (size_t)NS * 16 * MT * (4 * HS * 2 + 32) + (size_t)4 * 16 * MT * kRedStr * 4;
}

// Persistent and slice-resident: CTA b owns h-slice (b % kSlices) for its whole life, keeps that slice's fn operand
// (24 x 4 x HS) in registers, and processes row tiles tg, tg + G, tg + 2G, ... (tg = b / kSlices). The x tiles stream through an
// NS-deep cp.async ring (NS - 1 tiles in flight ahead of the one being computed). The tile -> CTA map only decides which CTA
// computes a (tile, slice) unit; every unit's numerics are identical.
template <int HS, int MT, bool FN32, int NS, bool MERGE = false, int NP = 3>
__global__ void __launch_bounds__(128) proj_kernel(const SplitParams p, int ntiles, int tgroups) {
  pdl::begin();   // launch structure only (pdl.cuh): nothing is read before the predecessor completes
  constexpr int NTH = 128;
  constexpr int RT = 16 * MT, Hs = HS, H = kSlices * HS;
  constexpr int XSTR = 4 * Hs * 2 + 32;  // bytes per smem row
  constexpr int ksteps = Hs / 16;
  extern __shared__ __align__(16) unsigned char smem[];
  float *red = reinterpret_cast<float *>(smem + (size_t)NS * RT * XSTR);   // [4][RT][kRedStr]

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int stream = warp;                          // leaf stream
  const int g = lane >> 2, t = lane & 3;
  const int slice = blockIdx.x % kSlices, tg = blockIdx.x / kSlices;
  const int hs0 = slice * Hs;
  constexpr size_t fn_ld = (size_t)4 * H;
  using Raw = typename std::conditional<FN32, float4, uint2>::type;

  auto stage = [&](int tile, unsigned char *dst) {
    const int row0 = tile * RT;
    constexpr int chunks = Hs / 8;  // 16-byte chunks per (row, stream)
    if constexpr (MERGE) {
      // Synchronous: merge the unit's 8-h chunks of all four streams, store the new streams, stage the same bits.
      for (int i = tid; i < RT * chunks; i += NTH) {
        const int r = i / chunks, c = i - r * chunks;
        uint4 res[4];
        if (row0 + r < p.T) {
          const int h = hs0 + c * 8;
          merge8(p, row0 + r, h, res);
#pragma unroll
          for (int s = 0; s < 4; ++s)
            *reinterpret_cast<uint4 *>(p.m_res + ((size_t)(row0 + r) * 4 + s) * H + h) = res[s];
        } else {
#pragma unroll
          for (int s = 0; s < 4; ++s) res[s] = make_uint4(0u, 0u, 0u, 0u);   // rows beyond T: zero, as cp.async fills
        }
#pragma unroll
        for (int s = 0; s < 4; ++s) *reinterpret_cast<uint4 *>(dst + (size_t)r * XSTR + (s * Hs + c * 8) * 2) = res[s];
      }
      return;
    }
#pragma unroll
    for (int i = tid; i < RT * 4 * chunks; i += NTH) {
      const int r = i / (4 * chunks), rem = i - r * 4 * chunks, s = rem / chunks, c = rem - s * chunks;
      const bool valid = row0 + r < p.T;
      const __nv_bfloat16 *src = p.x + ((size_t)(valid ? row0 + r : 0) * 4 + s) * H + hs0 + c * 8;
      cp_async16(dst + (size_t)r * XSTR + (s * Hs + c * 8) * 2, src, valid);
    }
  };

  // prologue: NS - 1 tiles in flight (one commit group each, empty past the end)
#pragma unroll
  for (int s = 0; s < NS - 1; ++s) {
    if (tg + s * tgroups < ntiles) stage(tg + s * tgroups, smem + (size_t)s * RT * XSTR);
    cp_async_commit();
  }

  // fn operand of this slice, resident for the whole kernel: B fragment (k = 4t..4t+3 of every 16-step, n = g)
  Raw Bf[ksteps][3];
  {
    const size_t fk0 = (size_t)stream * H + hs0 + 4 * t;
#pragma unroll
    for (int ks = 0; ks < ksteps; ++ks)
#pragma unroll
      for (int n = 0; n < 3; ++n) {
        if constexpr (FN32)
          Bf[ks][n] = *reinterpret_cast<const float4 *>(reinterpret_cast<const float *>(p.fn) +
                                                        (size_t)(n * 8 + g) * fn_ld + fk0 + ks * 16);
        else
          Bf[ks][n] = *reinterpret_cast<const uint2 *>(reinterpret_cast<const __nv_bfloat16 *>(p.fn) +
                                                       (size_t)(n * 8 + g) * fn_ld + fk0 + ks * 16);
      }
  }

  for (int e = tid; e < 4 * RT * (kRedStr - kCols); e += NTH) {      // pad columns of the leaf records (never written afterwards)
    const int rec = e / (kRedStr - kCols), c = e - rec * (kRedStr - kCols);
    red[(size_t)rec * kRedStr + kCols + c] = 0.f;
  }
  int buf = 0;
  for (int tile = tg; tile < ntiles; tile += tgroups) {
    const int row0 = tile * RT;
    {
      const int nxt = tile + (NS - 1) * tgroups;
      if (nxt < ntiles) stage(nxt, smem + (size_t)((buf + NS - 1) % NS) * RT * XSTR);
      cp_async_commit();
    }
    cp_async_wait_group_n<NS - 1>();
    __syncthreads();

    float acc[MT][3][4];
    float ss[MT][2];
#pragma unroll
    for (int m = 0; m < MT; ++m) {
      ss[m][0] = ss[m][1] = 0.f;
#pragma unroll
      for (int n = 0; n < 3; ++n) acc[m][n][0] = acc[m][n][1] = acc[m][n][2] = acc[m][n][3] = 0.f;
    }
    const unsigned char *xrow = smem + (size_t)buf * RT * XSTR + (size_t)g * XSTR + (size_t)(stream * Hs + 4 * t) * 2;
#pragma unroll
    for (int ks = 0; ks < ksteps; ++ks) {
      uint32_t A[MT][4];
#pragma unroll
      for (int m = 0; m < MT; ++m) {
        const uint2 lo = *reinterpret_cast<const uint2 *>(xrow + (size_t)(m * 16) * XSTR + ks * 32);
        const uint2 hi = *reinterpret_cast<const uint2 *>(xrow + (size_t)(m * 16 + 8) * XSTR + ks * 32);
        A[m][0] = lo.x; A[m][1] = hi.x; A[m][2] = lo.y; A[m][3] = hi.y;
        {
          // sum of squares: x^2 is exact in fp32, so fma(x, x, q) == q + x*x bit for bit; ascending-k chain
          float q = ss[m][0];
          q = __fmaf_rn(bf_lo(lo.x), bf_lo(lo.x), q);
          q = __fmaf_rn(bf_hi(lo.x), bf_hi(lo.x), q);
          q = __fmaf_rn(bf_lo(lo.y), bf_lo(lo.y), q);
          q = __fmaf_rn(bf_hi(lo.y), bf_hi(lo.y), q);
          ss[m][0] = q;
          q = ss[m][1];
          q = __fmaf_rn(bf_lo(hi.x), bf_lo(hi.x), q);
          q = __fmaf_rn(bf_hi(hi.x), bf_hi(hi.x), q);
          q = __fmaf_rn(bf_lo(hi.y), bf_lo(hi.y), q);
          q = __fmaf_rn(bf_hi(hi.y), bf_hi(hi.y), q);
          ss[m][1] = q;
        }
      }
      if constexpr (!FN32) {
#pragma unroll
        for (int n = 0; n < 3; ++n)
#pragma unroll
          for (int m = 0; m < MT; ++m) mma_bf16(acc[m][n], A[m], Bf[ks][n].x, Bf[ks][n].y);
      } else {
        uint32_t B[3][3][2];  // [n][piece][b0/b1]
#pragma unroll
        for (int n = 0; n < 3; ++n) {
          float4 fv = Bf[ks][n];
          // opaque: re-split every tile instead of hoisting 180 registers of pieces out of the tile loop
          asm volatile("" : "+f"(fv.x), "+f"(fv.y), "+f"(fv.z), "+f"(fv.w));
          uint32_t lo[3], hi[3];
          split3_trunc_pair(fv.x, fv.y, lo);
          split3_trunc_pair(fv.z, fv.w, hi);
#pragma unroll
          for (int pc = 0; pc < 3; ++pc) { B[n][pc][0] = lo[pc]; B[n][pc][1] = hi[pc]; }
        }
        // NP = 3: the exact FP32 fn (hi, mid, lo); NP = 2 (the TF32-fn variant): the fn was rounded to TF32 by the
        // caller, so its lo piece is exactly zero and the two pieces are its exact value
#pragma unroll
        for (int pc = 0; pc < NP; ++pc)
#pragma unroll
          for (int n = 0; n < 3; ++n)
#pragma unroll
            for (int m = 0; m < MT; ++m) mma_bf16(acc[m][n], A[m], B[n][pc][0], B[n][pc][1]);
      }
    }
    // leaf results -> red[stream][row][col]
#pragma unroll
    for (int m = 0; m < MT; ++m) {
      float *r0 = red + ((size_t)stream * RT + m * 16 + g) * kRedStr;
      float *r1 = red + ((size_t)stream * RT + m * 16 + g + 8) * kRedStr;
      {
        float s0 = __fadd_rn(ss[m][0], __shfl_xor_sync(FULL, ss[m][0], 1));
        s0 = __fadd_rn(s0, __shfl_xor_sync(FULL, s0, 2));
        float s1 = __fadd_rn(ss[m][1], __shfl_xor_sync(FULL, ss[m][1], 1));
        s1 = __fadd_rn(s1, __shfl_xor_sync(FULL, s1, 2));
        if (t == 0) { r0[24] = s0; r1[24] = s1; }
      }
#pragma unroll
      for (int n = 0; n < 3; ++n) {
        const int col = n * 8 + 2 * t;
        r0[col] = acc[m][n][0];
        r0[col + 1] = acc[m][n][1];
        r1[col] = acc[m][n][2];
        r1[col + 1] = acc[m][n][3];
      }
    }
    __syncthreads();   // red complete; all reads of this x buffer done (it is refilled NS - 1 tiles from now)
    // slice partial = (l0 + l1) + (l2 + l3) -> ws[slice][row] (one 128-byte record per row; 16-byte stores)
    for (int e = tid; e < RT * (kWsCols / 4); e += NTH) {
      const int r = e / (kWsCols / 4), c4 = e - r * (kWsCols / 4);
      if (row0 + r >= p.T) continue;
      float4 o = make_float4(0.f, 0.f, 0.f, 0.f);
      if (c4 < kRedStr / 4) {
        const float4 v0 = *reinterpret_cast<const float4 *>(red + ((size_t)0 * RT + r) * kRedStr + 4 * c4);
        const float4 v1 = *reinterpret_cast<const float4 *>(red + ((size_t)1 * RT + r) * kRedStr + 4 * c4);
        const float4 v2 = *reinterpret_cast<const float4 *>(red + ((size_t)2 * RT + r) * kRedStr + 4 * c4);
        const float4 v3 = *reinterpret_cast<const float4 *>(red + ((size_t)3 * RT + r) * kRedStr + 4 * c4);
        o.x = __fadd_rn(__fadd_rn(v0.x, v1.x), __fadd_rn(v2.x, v3.x));
        o.y = __fadd_rn(__fadd_rn(v0.y, v1.y), __fadd_rn(v2.y, v3.y));
        o.z = __fadd_rn(__fadd_rn(v0.z, v1.z), __fadd_rn(v2.z, v3.z));
        o.w = __fadd_rn(__fadd_rn(v0.w, v1.w), __fadd_rn(v2.w, v3.w));
      }
      *reinterpret_cast<float4 *>(p.ws + ((size_t)slice * p.T + row0 + r) * kWsCols + 4 * c4) = o;
    }
    buf = buf + 1 == NS ? 0 : buf + 1;
  }
}

struct TailParams {
  const float *ws; const float *scale, *base;
  float *pre, *post, *comb, *proj, *rstd;
  int T, H, iters;
  float rms_eps, eps;
};

// One warp per TWO rows (2w, 2w + 1): fixed-order tree over the slice partials, rstd, scalar tail. The Sinkhorn chain only needs 16
// lanes per row, so row A lives in lanes 16..31 and row B in lanes 0..15 (every xor-shuffle of the chain stays inside a 16-lane half):
// each lane performs exactly the operations of tail_forward on its own row's data, two rows for the instruction cost of one.
__global__ void __launch_bounds__(128) tail_kernel(const TailParams p) {
  pdl::begin();   // launch structure only (pdl.cuh): nothing is read before the predecessor completes
  const int lane = threadIdx.x & 31;
  const int rowA = (blockIdx.x * 4 + (threadIdx.x >> 5)) * 2, rowB = rowA + 1;
  if (rowA >= p.T) return;
  const bool hasB = rowB < p.T;
  float va[kSlices], vb[kSlices];
#pragma unroll
  for (int c = 0; c < kSlices; ++c) {
    va[c] = lane < kCols ? p.ws[((size_t)c * p.T + rowA) * kWsCols + lane] : 0.f;
    vb[c] = lane < kCols && hasB ? p.ws[((size_t)c * p.T + rowB) * kWsCols + lane] : 0.f;
  }
#pragma unroll
  for (int stride = 1; stride < kSlices; stride *= 2)
#pragma unroll
    for (int i = 0; i < kSlices; i += 2 * stride) {
      va[i] = __fadd_rn(va[i], va[i + stride]);
      vb[i] = __fadd_rn(vb[i], vb[i + stride]);
    }
  const float valA = va[0], valB = vb[0];
  const float rstdA = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(__shfl_sync(FULL, valA, 24), (float)(4 * p.H)), p.rms_eps)));
  const float rstdB = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(__shfl_sync(FULL, valB, 24), (float)(4 * p.H)), p.rms_eps)));
  float ppA, ppB, comb;
  tail_forward_pair(lane, __fmul_rn(valA, rstdA), __fmul_rn(valB, rstdB), p.scale, p.base, p.eps, p.iters, ppA, ppB, comb);
  if (lane < 4) p.pre[(size_t)rowA * 4 + lane] = ppA;
  else if (lane < 8) p.post[(size_t)rowA * 4 + lane - 4] = ppA;
  if (lane >= 16) p.comb[(size_t)rowA * 16 + lane - 16] = comb;
  if (hasB) {
    if (lane < 4) p.pre[(size_t)rowB * 4 + lane] = ppB;
    else if (lane < 8) p.post[(size_t)rowB * 4 + lane - 4] = ppB;
    if (lane < 16) p.comb[(size_t)rowB * 16 + lane] = comb;
  }
  if (p.proj && lane < 24) {
    p.proj[(size_t)rowA * 24 + lane] = valA;
    if (hasB) p.proj[(size_t)rowB * 24 + lane] = valB;
  }
  if (p.rstd && lane == 0) {
    p.rstd[rowA] = rstdA;
    if (hasB) p.rstd[rowB] = rstdB;
  }
}

// out[row][h] = bf16(((p0*x0 + p1*x1) + p2*x2) + p3*x3); 8 h per thread; pre is [T][4] fp32 (own or shifted).
__global__ void __launch_bounds__(256) collapse_kernel(const __nv_bfloat16 *__restrict__ x, const float *__restrict__ pre,
                                                      __nv_bfloat16 *__restrict__ out, int T, int H) {
  pdl::begin();   // launch structure only (pdl.cuh): nothing is read before the predecessor completes
  const int hv = H / 8;
  const int64_t total = (int64_t)T * hv;
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (int64_t)gridDim.x * blockDim.x) {
    const int64_t row = i / hv;
    const int h = (int)(i - row * hv) * 8;
    const float4 pr = *reinterpret_cast<const float4 *>(pre + row * 4);
    const float pv[4] = {pr.x, pr.y, pr.z, pr.w};
    float o[8];
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const uint4 q = *reinterpret_cast<const uint4 *>(x + (row * 4 + s) * H + h);
      const uint32_t w[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float lo = __fmul_rn(pv[s], bf_lo(w[j])), hi = __fmul_rn(pv[s], bf_hi(w[j]));
        o[2 * j] = s == 0 ? lo : __fadd_rn(o[2 * j], lo);
        o[2 * j + 1] = s == 0 ? hi : __fadd_rn(o[2 * j + 1], hi);
      }
    }
    uint4 ov;
    ov.x = pack_f2_bf16(o[0], o[1]); ov.y = pack_f2_bf16(o[2], o[3]);
    ov.z = pack_f2_bf16(o[4], o[5]); ov.w = pack_f2_bf16(o[6], o[7]);
    *reinterpret_cast<uint4 *>(out + row * H + h) = ov;
  }
}


// ---- GLM decode/prefill fusion: tail + own collapse + the block's RMSNorm, one CTA (128 threads) per row ----------------
// Replaces tail_kernel + collapse_kernel + norms_sm100's round-then-scale RMSNorm (rms_fwd at N = 4096) of the collapsed
// row. Every value is computed by the same operations in the same order as those three kernels:
//  * warp 0: tail_kernel's row program (fixed slice tree, rstd, tail_forward == tail_forward_pair's per-row bits) and its
//    pre / post / comb stores;
//  * the collapse of column h: collapse_kernel's ((p0*x0 + p1*x1) + p2*x2) + p3*x3, one rounding to bf16;
//  * the norm, norms_sm100's schedule at N = 4096 (norm_segments.cuh: segments of S consecutive columns, chains from
//    +0, adjacent tree; S from schedule_policy through norms_sm100/schedule.py), realised with L = 128 lanes and K = 4
//    loads of 8 per lane (lane l loads columns k*1024 + 8l + j), rstd = frsqrt_rn(add.rn(div.rn(acc, N), eps)),
//    y = rn_bf16(mul.rn(rn_bf16(mul.rn(v, rstd)), w)).
// The battery holds the fused outputs to split_forward + rms_norm_round_scale bit for bit.
namespace fusednorm {
constexpr int L = 128, K = 4, N = 8 * L * K;
__device__ __forceinline__ float bf16r(float a) { return __bfloat162float(__float2bfloat16_rn(a)); }
__device__ __forceinline__ uint32_t pack2rn(float a, float b) {
  uint32_t r;
  asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(r) : "f"(a), "f"(b));
  return r;
}
}  // namespace fusednorm

__global__ void __launch_bounds__(160) tail_collapse_norm_kernel(const TailParams p, const __nv_bfloat16 *__restrict__ x,
                                                                const __nv_bfloat16 *__restrict__ w,
                                                                __nv_bfloat16 *__restrict__ out, float norm_eps) {
  pdl::begin();   // launch structure only (pdl.cuh): nothing is read before the predecessor completes
  // Warps 0..3 (tid 0..127) are the norm's L = 128 lanes; warp 4 runs the row's tail. The pre values are published as soon
  // as tail_forward's first stage has them, so the collapse + norm overlap the Sinkhorn chain of comb (each value is still
  // produced by exactly tail_forward's operations: the function is split, not changed).
  using namespace fusednorm;
  const int row = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  __shared__ float s_pre[4];
  __shared__ float s_fold[K * L];   // one partial per segment of the collapsed row
  uint4 xq[K][4];
  if (warp < 4) {
#pragma unroll
    for (int k = 0; k < K; ++k)
#pragma unroll
      for (int s = 0; s < 4; ++s)
        xq[k][s] = *reinterpret_cast<const uint4 *>(x + ((size_t)row * 4 + s) * N + k * 8 * L + 8 * tid);
  }
  float mix_lane = 0.f;
  if (warp == 4) {
    float va[kSlices];
#pragma unroll
    for (int c = 0; c < kSlices; ++c) va[c] = lane < kCols ? p.ws[((size_t)c * p.T + row) * kWsCols + lane] : 0.f;
#pragma unroll
    for (int stride = 1; stride < kSlices; stride *= 2)
#pragma unroll
      for (int i = 0; i < kSlices; i += 2 * stride) va[i] = __fadd_rn(va[i], va[i + stride]);
    const float val = va[0];
    const float rstd = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(__shfl_sync(FULL, val, 24), (float)(4 * p.H)), p.rms_eps)));
    mix_lane = __fmul_rn(val, rstd);
    // tail_forward, first stage: pp (pre in lanes 0..3, post in lanes 4..7)
    const float a = __fadd_rn(__fmul_rn(mix_lane, p.scale[lane < 4 ? 0 : 1]), p.base[lane & 7]);
    const float sg = sigmoid_rn(a);
    const float pp = lane < 4 ? __fadd_rn(sg, p.eps) : __fmul_rn(2.0f, sg);
    if (lane < 4) { p.pre[(size_t)row * 4 + lane] = pp; s_pre[lane] = pp; }
    else if (lane < 8) p.post[(size_t)row * 4 + lane - 4] = pp;
  }
  __syncthreads();
  if (warp == 4) {
    // tail_forward, second stage: comb (Sinkhorn), lanes 16..31
    const float mix_c = __shfl_sync(FULL, mix_lane, 8 + (lane & 15));
    const float a = __fadd_rn(__fmul_rn(mix_c, p.scale[2]), p.base[8 + (lane & 15)]);
    float m = fmaxf(a, __shfl_xor_sync(FULL, a, 1));
    m = fmaxf(m, __shfl_xor_sync(FULL, m, 2));
    const float e = expf(__fsub_rn(a, m));
    float q = __fadd_rn(__fdiv_rn(e, sum4_row(e)), p.eps);
    q = __fdiv_rn(q, __fadd_rn(sum4_col(q), p.eps));
    for (int it = 1; it < p.iters; ++it) {
      q = __fdiv_rn(q, __fadd_rn(sum4_row(q), p.eps));
      q = __fdiv_rn(q, __fadd_rn(sum4_col(q), p.eps));
    }
    if (lane >= 16) p.comb[(size_t)row * 16 + lane - 16] = q;
    return;
  }
  const float pv[4] = {s_pre[0], s_pre[1], s_pre[2], s_pre[3]};
  float v[8 * K];
#pragma unroll
  for (int k = 0; k < K; ++k) {
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const uint32_t wv[4] = {xq[k][s].x, xq[k][s].y, xq[k][s].z, xq[k][s].w};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float lo = __fmul_rn(pv[s], bf_lo(wv[j])), hi = __fmul_rn(pv[s], bf_hi(wv[j]));
        v[8 * k + 2 * j] = s == 0 ? lo : __fadd_rn(v[8 * k + 2 * j], lo);
        v[8 * k + 2 * j + 1] = s == 0 ? hi : __fadd_rn(v[8 * k + 2 * j + 1], hi);
      }
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) v[8 * k + j] = bf16r(v[8 * k + j]);   // the collapsed bf16 row
  }
#pragma unroll
  for (int k = 0; k < K; ++k) s_fold[k * L + tid] = norm_segments::segment_sq(v + 8 * k);   // segment k L + tid
  asm volatile("bar.sync 1, 128;" ::: "memory");   // the four norm warps only
  const float root = norm_segments::segment_tree_warp<K * L>(s_fold);
  const float rs = __frsqrt_rn(__fadd_rn(__fdiv_rn(root, float(N)), norm_eps));
  __nv_bfloat16 *orow = out + (size_t)row * N;
#pragma unroll
  for (int k = 0; k < K; ++k) {
    const uint4 gq = *reinterpret_cast<const uint4 *>(w + k * 8 * L + 8 * tid);
    const uint32_t gw[4] = {gq.x, gq.y, gq.z, gq.w};
    float y[8];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      y[2 * j] = __fmul_rn(bf16r(__fmul_rn(v[8 * k + 2 * j], rs)), bf_lo(gw[j]));
      y[2 * j + 1] = __fmul_rn(bf16r(__fmul_rn(v[8 * k + 2 * j + 1], rs)), bf_hi(gw[j]));
    }
    reinterpret_cast<uint4 *>(orow + k * 8 * L + 8 * tid)[0] =
        make_uint4(pack2rn(y[0], y[1]), pack2rn(y[2], y[3]), pack2rn(y[4], y[5]), pack2rn(y[6], y[7]));
  }
}

template <int HS, int MT, bool FN32, int NS, bool MERGE = false, int NP = 3>
static void launch_proj(const SplitParams &p, int ntiles, cudaStream_t st, int groups_override) {
  auto kern = proj_kernel<HS, MT, FN32, NS, MERGE, NP>;
  constexpr size_t smem = proj_smem_bytes<HS, MT, FN32, NS>();
  constexpr int nth = 128;
  cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  int per_sm = 0, sms = 0, dev = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, nth, smem);
  int tgroups = std::max(1, std::min(ntiles, sms * std::max(per_sm, 1) / kSlices));
  if (groups_override > 0) tgroups = std::min(ntiles, groups_override);   // ownership only (tests); never changes a value
  pdl::launch(kern, dim3(tgroups * kSlices), dim3(nth), size_t(smem), st, p, ntiles, tgroups);
}

}  // namespace mhc

namespace mhc_host {

using namespace mhc;

std::vector<at::Tensor> split_forward_cuda(const at::Tensor &x, const at::Tensor &fn, const at::Tensor &scale,
                                           const at::Tensor &base, const c10::optional<at::Tensor> &ext_pre,
                                           int64_t collapse_mode, double rms_eps, double eps, int64_t iters,
                                           bool want_stats, int64_t mt, int64_t groups) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() == 3 && x.size(1) == 4 &&
                  x.scalar_type() == at::kBFloat16, "x must be contiguous CUDA bf16 [T, 4, H]");
  const int64_t T = x.size(0), H = x.size(2);
  TORCH_CHECK(fn.is_cuda() && fn.is_contiguous() && fn.sizes() == at::IntArrayRef({24, 4 * H}), "fn must be [24, 4H]");
  const bool fn32 = fn.scalar_type() == at::kFloat;
  TORCH_CHECK(fn32 || fn.scalar_type() == at::kBFloat16, "fn must be fp32 or bf16");
  TORCH_CHECK(scale.scalar_type() == at::kFloat && scale.numel() == 3 && base.scalar_type() == at::kFloat &&
                  base.numel() == 24, "scale [3] and base [24] must be fp32");
  TORCH_CHECK(iters >= 1 && collapse_mode >= 0 && collapse_mode <= 2, "bad iters / collapse mode");
  TORCH_CHECK(H == 4096 || H == 5120, "mHC split serves the published widths H = 4096 (GLM) and 5120 (DeepSeek)");
  const c10::cuda::CUDAGuard guard(x.device());
  auto f32 = x.options().dtype(at::kFloat);
  at::Tensor pre = at::empty({T, 4}, f32), post = at::empty({T, 4}, f32), comb = at::empty({T, 4, 4}, f32);
  at::Tensor proj = want_stats ? at::empty({T, 24}, f32) : at::empty({0}, f32);
  at::Tensor rstd = want_stats ? at::empty({T}, f32) : at::empty({0}, f32);
  at::Tensor col = collapse_mode ? at::empty({T, H}, x.options()) : at::empty({0}, x.options());
  if (T == 0) return {pre, post, comb, proj, rstd, col};
  at::Tensor ws = at::empty({kSlices, T, kWsCols}, f32);
  auto st = at::cuda::getCurrentCUDAStream().stream();
  SplitParams p{};
  p.x = reinterpret_cast<const __nv_bfloat16 *>(x.data_ptr());
  p.fn = fn.data_ptr();
  p.scale = scale.data_ptr<float>();
  p.base = base.data_ptr<float>();
  p.ws = ws.data_ptr<float>();
  p.T = (int)T; p.H = (int)H;
  if (mt == 0) mt = 1;
#define LAUNCH(HSV, MTV)                                                                                    \
  if (H == kSlices * HSV && mt == MTV) {                                                                    \
    const int ntiles = (int)((T + 16 * MTV - 1) / (16 * MTV));                                              \
    fn32 ? launch_proj<HSV, MTV, true, 2>(p, ntiles, st, (int)groups) : launch_proj<HSV, MTV, false, 2>(p, ntiles, st, (int)groups); \
  } else
  LAUNCH(128, 1) LAUNCH(128, 2) LAUNCH(160, 1) { TORCH_CHECK(false, "unsupported H / mt"); }
#undef LAUNCH
  TailParams tp{};
  tp.ws = p.ws; tp.scale = p.scale; tp.base = p.base;
  tp.pre = pre.data_ptr<float>(); tp.post = post.data_ptr<float>(); tp.comb = comb.data_ptr<float>();
  tp.proj = want_stats ? proj.data_ptr<float>() : nullptr;
  tp.rstd = want_stats ? rstd.data_ptr<float>() : nullptr;
  tp.T = (int)T; tp.H = (int)H; tp.iters = (int)iters; tp.rms_eps = (float)rms_eps; tp.eps = (float)eps;
  pdl::launch(tail_kernel, dim3((unsigned)((T + 7) / 8)), dim3(128), size_t(0), st, tp);
  if (collapse_mode) {
    const float *cp = tp.pre;
    if (collapse_mode == 2) {
      TORCH_CHECK(ext_pre.has_value() && ext_pre->is_contiguous() && ext_pre->scalar_type() == at::kFloat &&
                      ext_pre->numel() == T * 4, "collapse mode 2 needs fp32 pre [T, 4]");
      cp = ext_pre->data_ptr<float>();
    }
    pdl::launch(collapse_kernel, dim3((unsigned)std::min<int64_t>((T * (H / 8) + 255) / 256, 1 << 20)), dim3(256), size_t(0), st, 
        p.x, cp, reinterpret_cast<__nv_bfloat16 *>(col.data_ptr()), (int)T, (int)H);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {pre, post, comb, proj, rstd, col};
}


// The DeepSeek coefficients in two launches the caller may put on different streams (inference): split_proj = exactly
// the projection launch of split_forward (proj_kernel<160, 1, true, 2>) writing the slice-partial workspace; split_tail =
// split_forward's tail_kernel on that workspace. Bit for bit split_forward(collapse = None, stats = False).
at::Tensor split_proj_cuda(const at::Tensor &x, const at::Tensor &fn, int64_t groups, int64_t pieces) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() == 3 && x.size(1) == 4 && x.size(2) == 5120 &&
                  x.scalar_type() == at::kBFloat16, "split_proj: x must be contiguous CUDA bf16 [T, 4, 5120]");
  const int64_t T = x.size(0), H = 5120;
  TORCH_CHECK(fn.is_cuda() && fn.is_contiguous() && fn.scalar_type() == at::kFloat &&
                  fn.sizes() == at::IntArrayRef({24, 4 * H}), "split_proj: fp32 fn [24, 4H]");
  const c10::cuda::CUDAGuard guard(x.device());
  at::Tensor ws = at::empty({kSlices, T, kWsCols}, x.options().dtype(at::kFloat));
  if (T == 0) return ws;
  auto st = at::cuda::getCurrentCUDAStream().stream();
  SplitParams p{};
  p.x = reinterpret_cast<const __nv_bfloat16 *>(x.data_ptr());
  p.fn = fn.data_ptr();
  p.ws = ws.data_ptr<float>();
  p.T = (int)T; p.H = (int)H;
  TORCH_CHECK(pieces == 2 || pieces == 3, "split_proj: pieces 3 (exact FP32 fn) or 2 (TF32-rounded fn)");
  if (pieces == 3) launch_proj<160, 1, true, 2>(p, (int)((T + 15) / 16), st, (int)groups);
  else launch_proj<160, 1, true, 2, false, 2>(p, (int)((T + 15) / 16), st, (int)groups);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return ws;
}

std::vector<at::Tensor> split_tail_cuda(const at::Tensor &ws, const at::Tensor &scale, const at::Tensor &base,
                                        double rms_eps, double eps, int64_t iters) {
  TORCH_CHECK(ws.is_cuda() && ws.is_contiguous() && ws.scalar_type() == at::kFloat && ws.dim() == 3 &&
                  ws.size(0) == kSlices && ws.size(2) == kWsCols, "split_tail: workspace [32, T, 32] fp32");
  TORCH_CHECK(scale.scalar_type() == at::kFloat && scale.numel() == 3 && base.scalar_type() == at::kFloat &&
                  base.numel() == 24 && iters >= 1, "split_tail: scale [3], base [24] fp32");
  const int64_t T = ws.size(1), H = 5120;
  const c10::cuda::CUDAGuard guard(ws.device());
  auto f32 = ws.options();
  at::Tensor pre = at::empty({T, 4}, f32), post = at::empty({T, 4}, f32), comb = at::empty({T, 4, 4}, f32);
  if (T == 0) return {pre, post, comb};
  auto st = at::cuda::getCurrentCUDAStream().stream();
  TailParams tp{};
  tp.ws = ws.data_ptr<float>(); tp.scale = scale.data_ptr<float>(); tp.base = base.data_ptr<float>();
  tp.pre = pre.data_ptr<float>(); tp.post = post.data_ptr<float>(); tp.comb = comb.data_ptr<float>();
  tp.proj = nullptr; tp.rstd = nullptr;
  tp.T = (int)T; tp.H = (int)H; tp.iters = (int)iters; tp.rms_eps = (float)rms_eps; tp.eps = (float)eps;
  pdl::launch(tail_kernel, dim3((unsigned)((T + 7) / 8)), dim3(128), size_t(0), st, tp);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {pre, post, comb};
}


// GLM split + the block RMSNorm of the collapsed row (round-then-scale, bf16 weight [4096]): (pre, post, comb, normed).
// With m_out/m_post/m_comb: x is the previous streams and the split runs on merge(x, m_out, m_post, m_comb), the
// new streams being returned as a fifth tensor (merge_forward(cast=True) + split_norm in two launches).
std::vector<at::Tensor> split_norm_forward_cuda(const at::Tensor &x, const at::Tensor &fn, const at::Tensor &scale,
                                                const at::Tensor &base, const at::Tensor &norm_w, double rms_eps,
                                                double eps, int64_t iters, double norm_eps,
                                                const c10::optional<at::Tensor> &m_out,
                                                const c10::optional<at::Tensor> &m_post,
                                                const c10::optional<at::Tensor> &m_comb) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() == 3 && x.size(1) == 4 && x.size(2) == 4096 &&
                  x.scalar_type() == at::kBFloat16, "split_norm: x must be contiguous CUDA bf16 [T, 4, 4096]");
  TORCH_CHECK(norm_w.is_cuda() && norm_w.is_contiguous() && norm_w.scalar_type() == at::kBFloat16 && norm_w.numel() == 4096,
              "split_norm: bf16 norm weight [4096]");
  const int64_t T = x.size(0), H = 4096;
  TORCH_CHECK(fn.is_cuda() && fn.is_contiguous() && fn.sizes() == at::IntArrayRef({24, 4 * H}), "fn must be [24, 4H]");
  const bool fn32 = fn.scalar_type() == at::kFloat;
  TORCH_CHECK(fn32 || fn.scalar_type() == at::kBFloat16, "fn must be fp32 or bf16");
  TORCH_CHECK(scale.scalar_type() == at::kFloat && scale.numel() == 3 && base.scalar_type() == at::kFloat &&
                  base.numel() == 24, "scale [3] and base [24] must be fp32");
  TORCH_CHECK(iters >= 1, "bad iters");
  const c10::cuda::CUDAGuard guard(x.device());
  auto f32 = x.options().dtype(at::kFloat);
  at::Tensor pre = at::empty({T, 4}, f32), post = at::empty({T, 4}, f32), comb = at::empty({T, 4, 4}, f32);
  at::Tensor normed = at::empty({T, H}, x.options());
  const bool merge = m_out.has_value();
  TORCH_CHECK(merge == m_post.has_value() && merge == m_comb.has_value(), "split_norm: merge needs out, post and comb");
  if (merge) {
    TORCH_CHECK(m_out->is_contiguous() && m_out->scalar_type() == at::kBFloat16 && m_out->numel() == T * H &&
                    m_post->is_contiguous() && m_post->scalar_type() == at::kFloat && m_post->numel() == T * 4 &&
                    m_comb->is_contiguous() && m_comb->scalar_type() == at::kFloat && m_comb->numel() == T * 16 &&
                    m_out->device() == x.device() && m_post->device() == x.device() && m_comb->device() == x.device(),
                "split_norm merge: out bf16 [T, H], post fp32 [T, 4], comb fp32 [T, 4, 4], contiguous");
  }
  at::Tensor streams = merge ? at::empty_like(x) : at::empty({0}, x.options());
  if (T == 0) return {pre, post, comb, normed, streams};
  at::Tensor ws = at::empty({kSlices, T, kWsCols}, f32);
  auto st = at::cuda::getCurrentCUDAStream().stream();
  SplitParams p{};
  p.x = reinterpret_cast<const __nv_bfloat16 *>(x.data_ptr());
  p.fn = fn.data_ptr();
  p.scale = scale.data_ptr<float>();
  p.base = base.data_ptr<float>();
  p.ws = ws.data_ptr<float>();
  p.T = (int)T; p.H = (int)H;
  const int ntiles = (int)((T + 15) / 16);
  if (merge) {
    p.m_out = reinterpret_cast<const __nv_bfloat16 *>(m_out->data_ptr());
    p.m_post = m_post->data_ptr<float>();
    p.m_comb = m_comb->data_ptr<float>();
    p.m_res = reinterpret_cast<__nv_bfloat16 *>(streams.data_ptr());
    fn32 ? launch_proj<128, 1, true, 2, true>(p, ntiles, st, 0) : launch_proj<128, 1, false, 2, true>(p, ntiles, st, 0);
  } else {
    fn32 ? launch_proj<128, 1, true, 2>(p, ntiles, st, 0) : launch_proj<128, 1, false, 2>(p, ntiles, st, 0);
  }
  const __nv_bfloat16 *cx = merge ? p.m_res : p.x;   // the tail/collapse/norm read the (new) streams
  TailParams tp{};
  tp.ws = p.ws; tp.scale = p.scale; tp.base = p.base;
  tp.pre = pre.data_ptr<float>(); tp.post = post.data_ptr<float>(); tp.comb = comb.data_ptr<float>();
  tp.proj = nullptr; tp.rstd = nullptr;
  tp.T = (int)T; tp.H = (int)H; tp.iters = (int)iters; tp.rms_eps = (float)rms_eps; tp.eps = (float)eps;
  pdl::launch(tail_collapse_norm_kernel, dim3((unsigned)T), dim3(160), size_t(0), st, tp, cx, reinterpret_cast<const __nv_bfloat16 *>(norm_w.data_ptr()),
                                                        reinterpret_cast<__nv_bfloat16 *>(normed.data_ptr()), (float)norm_eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {pre, post, comb, normed, streams};
}

// Standalone shifted collapse: out[row] = bf16(sum_s pre[s] * x[s]) with a caller-supplied fp32 pre [T, 4].
at::Tensor collapse_forward_cuda(const at::Tensor &x, const at::Tensor &pre) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() == 3 && x.size(1) == 4 && x.scalar_type() == at::kBFloat16, "bad x");
  const int64_t T = x.size(0), H = x.size(2);
  TORCH_CHECK(H % 8 == 0 && pre.is_contiguous() && pre.scalar_type() == at::kFloat && pre.numel() == T * 4, "bad pre");
  const c10::cuda::CUDAGuard guard(x.device());
  at::Tensor out = at::empty({T, H}, x.options());
  if (T == 0) return out;
  auto st = at::cuda::getCurrentCUDAStream().stream();
  pdl::launch(collapse_kernel, dim3((unsigned)std::min<int64_t>((T * (H / 8) + 255) / 256, 1 << 20)), dim3(256), size_t(0), st, 
      reinterpret_cast<const __nv_bfloat16 *>(x.data_ptr()), pre.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16 *>(out.data_ptr()), (int)T, (int)H);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace mhc_host
