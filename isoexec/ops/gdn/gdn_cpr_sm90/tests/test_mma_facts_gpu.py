"""The hardware facts the one-identity claim rests on, checked on this GPU bit for bit.

Decode evaluates a chunk row with mma.sync on zero-filled operands; prefill evaluates the whole chunk with
mma.sync / WGMMA on real neighbouring rows. They agree only if:
  (1) an mma.sync m16n8k16 bf16 result does not depend on where its row / column sits in the tile, nor on
      the other rows and columns (placement invariance);
  (2) a k-term whose coefficient is +0 contributes nothing, whatever finite value the other operand holds
      (masked-operand invariance), for ffma (partial sums other than -0) and for mma.sync;
  (3) skipping a k16 step whose B column is all +0 leaves the accumulator bit-identical (decode skips them);
  (4) each WGMMA form production issues equals the chain of mma.sync over the same k16 steps, with the
      wrappers, descriptors and tile layouts of ``csrc/tile.cuh`` itself: ``wgmma_n64_ss`` with both
      operands K-major (the output units: Wide<64> operands unseeded, Tile<64> operands seeded) and
      ``wgmma_n128_ss<1>`` with B read MN-major (pass 1's u and w);
  (5) the prefill sweep's transposed products, ``wgmma_ss`` at the sweep/fused widths (m64n16, m64n32 and m64n64) with
      both operands in shared memory and the operand roles swapped (A = W or the k^ tile read MN-major,
      B = S^ or d~), give element by element the transpose of the recipe's mma.sync chain -- unseeded over
      8 steps (the deltas) and seeded over 4 steps (the boundary state).
The kernels are built with the family's numerics flags (no FTZ, no FMA contraction). Claim map:
tests/README.md.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import pytest
import torch

from isoexec.ops._testing import bitwise_equal
from isoexec.ops.gdn.gdn_cpr_sm90.tests._fixtures import BF, DEV, SM90

pytestmark = SM90

CSRC = Path(__file__).resolve().parents[1] / "csrc"
def _build_dir() -> str:
    """Beside the family's extension build (``_extension.build_dir``), in a subdirectory."""
    from isoexec.ops.gdn.gdn_cpr_sm90._extension import build_dir

    return str(build_dir() / "mma_facts")

CUDA = r"""
#include <torch/extension.h>
#include "tile.cuh"
using namespace gdn_cpr;

__device__ __forceinline__ void mma_c(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ uint32_t pk(uint16_t lo, uint16_t hi) { return uint32_t(lo) | (uint32_t(hi) << 16); }

// one m16n8k16 per tile: D = A[16][16] B[16 k][8 n] + C[16][8]
__global__ void tiles_k(const uint16_t *A, const uint16_t *B, const float *C, float *D, int nt) {
  int t = blockIdx.x * 4 + threadIdx.x / 32;
  if (t >= nt) return;
  int lane = threadIdx.x % 32, g = lane / 4, q = lane % 4;
  const uint16_t *a = A + t * 256, *b = B + t * 128;
  uint32_t ar[4] = {pk(a[g * 16 + 2 * q], a[g * 16 + 2 * q + 1]), pk(a[(g + 8) * 16 + 2 * q], a[(g + 8) * 16 + 2 * q + 1]),
                    pk(a[g * 16 + 2 * q + 8], a[g * 16 + 2 * q + 9]), pk(a[(g + 8) * 16 + 2 * q + 8], a[(g + 8) * 16 + 2 * q + 9])};
  uint32_t br[2] = {pk(b[(2 * q) * 8 + g], b[(2 * q + 1) * 8 + g]), pk(b[(2 * q + 8) * 8 + g], b[(2 * q + 9) * 8 + g])};
  const float *c = C + t * 128;
  float d[4] = {c[g * 8 + 2 * q], c[g * 8 + 2 * q + 1], c[(g + 8) * 8 + 2 * q], c[(g + 8) * 8 + 2 * q + 1]};
  mma_c(d, ar, br);
  float *o = D + t * 128;
  o[g * 8 + 2 * q] = d[0]; o[g * 8 + 2 * q + 1] = d[1]; o[(g + 8) * 8 + 2 * q] = d[2]; o[(g + 8) * 8 + 2 * q + 1] = d[3];
}
torch::Tensor tiles(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
  auto D = torch::empty_like(C);
  int nt = A.size(0);
  tiles_k<<<(nt + 3) / 4, 128>>>((const uint16_t *)A.data_ptr(), (const uint16_t *)B.data_ptr(), C.data_ptr<float>(),
                                 D.data_ptr<float>(), nt);
  return D;
}

// ffma(x, y, p) elementwise, no contraction games: the recipe's ffma
__global__ void fma_k(const float *x, const float *y, const float *p, float *o, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) o[i] = __fmaf_rn(x[i], y[i], p[i]);
}
torch::Tensor fma_rn(torch::Tensor x, torch::Tensor y, torch::Tensor p) {
  auto o = torch::empty_like(p);
  int n = p.numel();
  fma_k<<<(n + 255) / 256, 256>>>(x.data_ptr<float>(), y.data_ptr<float>(), p.data_ptr<float>(), o.data_ptr<float>(), n);
  return o;
}

// CHAIN64 of 4 k16 steps against one B column (column 0 of the n8 tile, entries k >= kz zero), seeded;
// full = every step, skip = only the steps with 16 s < kz.
__global__ void skip_k(const uint16_t *A, const uint16_t *Bc, const float *seed, const int *kz, float *full, float *skip) {
  const int t = blockIdx.x, lane = threadIdx.x, g = lane >> 2, tq = lane & 3;
  const uint16_t *a = A + t * 16 * 64, *bc = Bc + t * 64;
  float df[4] = {tq == 0 ? seed[t * 16 + g] : 0.f, 0.f, tq == 0 ? seed[t * 16 + g + 8] : 0.f, 0.f};
  float ds[4] = {df[0], df[1], df[2], df[3]};
  const int z = kz[t];
  for (int s = 0; s < 4; ++s) {
    const int k0 = 16 * s;
    uint32_t af[4] = {pk(a[g * 64 + k0 + 2 * tq], a[g * 64 + k0 + 2 * tq + 1]),
                      pk(a[(g + 8) * 64 + k0 + 2 * tq], a[(g + 8) * 64 + k0 + 2 * tq + 1]),
                      pk(a[g * 64 + k0 + 2 * tq + 8], a[g * 64 + k0 + 2 * tq + 9]),
                      pk(a[(g + 8) * 64 + k0 + 2 * tq + 8], a[(g + 8) * 64 + k0 + 2 * tq + 9])};
    auto bv = [&](int k) -> uint16_t { return (g == 0 && k < z) ? bc[k] : uint16_t(0); };
    uint32_t bf[2] = {pk(bv(k0 + 2 * tq), bv(k0 + 2 * tq + 1)), pk(bv(k0 + 2 * tq + 8), bv(k0 + 2 * tq + 9))};
    mma_c(df, af, bf);
    if (k0 < z) mma_c(ds, af, bf);
  }
  if (tq == 0) {
    full[t * 16 + g] = df[0]; full[t * 16 + g + 8] = df[2];
    skip[t * 16 + g] = ds[0]; skip[t * 16 + g + 8] = ds[2];
  }
}
std::vector<torch::Tensor> skip(torch::Tensor A, torch::Tensor Bc, torch::Tensor seed, torch::Tensor kz) {
  auto f = torch::empty_like(seed), s = torch::empty_like(seed);
  skip_k<<<A.size(0), 32>>>((const uint16_t *)A.data_ptr(), (const uint16_t *)Bc.data_ptr(), seed.data_ptr<float>(),
                            kz.data_ptr<int>(), f.data_ptr<float>(), s.data_ptr<float>());
  return {f, s};
}

// A fragments of rows 16w.. of a [64][K] row-major bf16 matrix, k16 step ks
__device__ __forceinline__ void afrag(uint32_t (&af)[4], const uint16_t *a, int K, int r0, int ks, int q) {
  const int c = 16 * ks + 2 * q;
  af[0] = pk(a[r0 * K + c], a[r0 * K + c + 1]); af[1] = pk(a[(r0 + 8) * K + c], a[(r0 + 8) * K + c + 1]);
  af[2] = pk(a[r0 * K + c + 8], a[r0 * K + c + 9]); af[3] = pk(a[(r0 + 8) * K + c + 8], a[(r0 + 8) * K + c + 9]);
}

// (4a) wgmma_n64_ss, both operands K-major in shared memory (the output units' two chains):
//   KW = 128: D[64 v][64 j] = CHAIN128_d(A[v][d], B[j][d]; +0), A and B Wide<64> tiles (S^, q^), 8 steps
//   KW = 64:  D[64 v][64 j] = CHAIN64_i(A[v][i], B[j][i]; C), A and B Tile<64> tiles (d, M), 4 steps, seeded
template <int KW>
__global__ void kmajor_k(const uint16_t *A, const uint16_t *B, const float *Cin, float *Dw, float *Dm) {
  __shared__ __align__(1024) uint16_t sa[64 * KW], sb[64 * KW];
  const uint16_t *a = A + blockIdx.x * 64 * KW, *b = B + blockIdx.x * 64 * KW;
  const float *cin = Cin + blockIdx.x * 4096;
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32, g = lane / 4, q = lane % 4, r0 = 16 * warp + g;
  for (int x = tid; x < 64 * KW; x += 128) {
    if constexpr (KW == 128) {
      Wide<64>{sa}.at(x / KW, x % KW) = a[x];
      Wide<64>{sb}.at(x / KW, x % KW) = b[x];
    } else {
      Tile<64>{sa}.at(x / KW, x % KW) = a[x];
      Tile<64>{sb}.at(x / KW, x % KW) = b[x];
    }
  }
  fence_async_smem();
  __syncthreads();
  constexpr int KS = KW / 16;
  float d[32], m[8][4];
  for (int nt = 0; nt < 8; ++nt)
    for (int e = 0; e < 4; ++e) d[4 * nt + e] = m[nt][e] = KW == 64 ? cin[(r0 + (e >> 1) * 8) * 64 + 8 * nt + 2 * q + (e & 1)] : 0.f;
  wg_fence();
  for (int ks = 0; ks < KS; ++ks) {
    if constexpr (KW == 128)
      wgmma_n64_ss(d, desc_kmajor(Wide<64>{sa}.half(ks >> 2).base, 16 * (ks & 3)),
                   desc_kmajor(Wide<64>{sb}.half(ks >> 2).base, 16 * (ks & 3)), ks > 0);
    else
      wgmma_n64_ss(d, desc_kmajor(sa, 16 * ks), desc_kmajor(sb, 16 * ks), true);
  }
  wg_commit();
  wg_wait<0>();
  wg_pin(d);
  for (int nt = 0; nt < 8; ++nt) {
    for (int ks = 0; ks < KS; ++ks) {
      uint32_t af[4];
      afrag(af, a, KW, r0, ks, q);
      const int j = 8 * nt + g, k0 = 16 * ks + 2 * q;
      uint32_t bb[2] = {pk(b[j * KW + k0], b[j * KW + k0 + 1]), pk(b[j * KW + k0 + 8], b[j * KW + k0 + 9])};
      mma_c(m[nt], af, bb);
    }
    for (int e = 0; e < 4; ++e) {
      const int v = r0 + (e >> 1) * 8, j = 8 * nt + 2 * q + (e & 1);
      Dw[blockIdx.x * 4096 + v * 64 + j] = d[4 * nt + e];
      Dm[blockIdx.x * 4096 + v * 64 + j] = m[nt][e];
    }
  }
}
std::vector<torch::Tensor> kmajor(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
  auto o = torch::empty({A.size(0), 64, 64}, A.options().dtype(torch::kFloat32)), m = torch::empty_like(o);
  auto launch = A.size(2) == 128 ? kmajor_k<128> : kmajor_k<64>;
  launch<<<A.size(0), 128>>>((const uint16_t *)A.data_ptr(), (const uint16_t *)B.data_ptr(), C.data_ptr<float>(),
                             o.data_ptr<float>(), m.data_ptr<float>());
  return {o, m};
}

// (4b) wgmma_n128_ss<1> (pass 1's u/w chains): W[64 j][128 d] = CHAIN64_i(X[j][i], K[i][d]; +0), A = X a Tile<64>
// K-major, B = K a Wide<64> read MN-major (desc_mnmajor), 4 steps
__global__ void mnmajor_k(const uint16_t *X, const uint16_t *K, float *Dw, float *Dm) {
  __shared__ __align__(1024) uint16_t sx[64 * 64], sk[64 * 128];
  const Tile<64> xt{sx};
  const Wide<64> kt{sk};
  const uint16_t *x_ = X + blockIdx.x * 64 * 64, *k = K + blockIdx.x * 64 * 128;
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32, g = lane / 4, q = lane % 4, r0 = 16 * warp + g;
  for (int x = tid; x < 64 * 128; x += 128) kt.at(x / 128, x % 128) = k[x];
  for (int x = tid; x < 64 * 64; x += 128) xt.at(x / 64, x % 64) = x_[x];
  fence_async_smem();
  __syncthreads();
  float S[64], M[16][4];
  wg_fence();
  for (int ks = 0; ks < 4; ++ks) wgmma_n128_ss<1>(S, desc_kmajor(xt.base, 16 * ks), desc_mnmajor(kt, 16 * ks), ks > 0);
  wg_commit();
  wg_wait<0>();
  wg_pin(S);
  for (int nt = 0; nt < 16; ++nt) {
    for (int e = 0; e < 4; ++e) M[nt][e] = 0.f;
    for (int ks = 0; ks < 4; ++ks) {
      uint32_t af[4];
      afrag(af, x_, 64, r0, ks, q);
      const int dd = 8 * nt + g, i = 16 * ks + 2 * q;
      uint32_t b[2] = {pk(k[i * 128 + dd], k[(i + 1) * 128 + dd]), pk(k[(i + 8) * 128 + dd], k[(i + 9) * 128 + dd])};
      mma_c(M[nt], af, b);
    }
    for (int e = 0; e < 4; ++e) {
      const int j = r0 + (e >> 1) * 8, dd = 8 * nt + 2 * q + (e & 1);
      Dw[blockIdx.x * 8192 + j * 128 + dd] = S[4 * nt + e];
      Dm[blockIdx.x * 8192 + j * 128 + dd] = M[nt][e];
    }
  }
}
std::vector<torch::Tensor> mnmajor(torch::Tensor X, torch::Tensor K) {
  auto o = torch::empty({X.size(0), 64, 128}, X.options().dtype(torch::kFloat32)), m = torch::empty_like(o);
  mnmajor_k<<<X.size(0), 128>>>((const uint16_t *)X.data_ptr(), (const uint16_t *)K.data_ptr(), o.data_ptr<float>(),
                                m.data_ptr<float>());
  return {o, m};
}

// (5a) D^T[i][v] = CHAIN128_d(W[i][d], S^[v][d]) by wgmma m64n32 SS vs the recipe's chain D[v][i] (A = S^ rows)
template <int VR>
__global__ void swap_delta_k(const uint16_t *S, const uint16_t *W, float *Dw, float *Dm) {
  __shared__ __align__(1024) uint16_t sw[64 * 128];
  __shared__ __align__(1024) uint16_t ss[VR * 128];
  const Wide<64> wt{sw};
  const Wide<VR> st{ss};
  const uint16_t *s = S + blockIdx.x * VR * 128, *w = W + blockIdx.x * 64 * 128;
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32, g = lane / 4, q = lane % 4;
  for (int x = tid; x < 64 * 128; x += 128) wt.at(x / 128, x % 128) = w[x];
  for (int x = tid; x < VR * 128; x += 128) st.at(x / 128, x % 128) = s[x];
  fence_async_smem();
  __syncthreads();
  float acc[VR / 2];
  wg_fence();
  for (int k = 0; k < 8; ++k)
    wgmma_ss<VR, 0>(acc, desc_kmajor(wt.half(k >> 2).base, 16 * (k & 3)), desc_kmajor(st.half(k >> 2).base, 16 * (k & 3)), k > 0);
  wg_commit();
  wg_wait<0>();
  wg_pin(acc);
  for (int e = 0; e < VR / 2; ++e) {   // acc element (i, v)
    const int i = 16 * warp + g + 8 * ((e >> 1) & 1), v = 8 * (e >> 2) + 2 * q + (e & 1);
    Dw[blockIdx.x * VR * 64 + v * 64 + i] = acc[e];
  }
  if (warp < VR / 16) {   // reference: A = S^ rows v (16 per warp), B columns = W rows i, 8 n8 tiles
    for (int nt = 0; nt < 8; ++nt) {
      float d[4] = {0.f, 0.f, 0.f, 0.f};
      const int v0 = 16 * warp;
      for (int ks = 0; ks < 8; ++ks) {
        uint32_t a[4];
        afrag(a, s, 128, v0 + g, ks, q);
        const int i = 8 * nt + g, k0 = 16 * ks + 2 * q;
        uint32_t b[2] = {pk(w[i * 128 + k0], w[i * 128 + k0 + 1]), pk(w[i * 128 + k0 + 8], w[i * 128 + k0 + 9])};
        mma_c(d, a, b);
      }
      for (int e = 0; e < 4; ++e) Dm[blockIdx.x * VR * 64 + (v0 + g + 8 * (e >> 1)) * 64 + 8 * nt + 2 * q + (e & 1)] = d[e];
    }
  }
}
std::vector<torch::Tensor> swap_delta(torch::Tensor S, torch::Tensor W) {
  auto o = torch::empty({S.size(0), S.size(1), 64}, S.options().dtype(torch::kFloat32)), m = torch::empty_like(o);
  auto launch = S.size(1) == 64 ? swap_delta_k<64> : (S.size(1) == 32 ? swap_delta_k<32> : swap_delta_k<16>);
  launch<<<S.size(0), 128>>>((const uint16_t *)S.data_ptr(), (const uint16_t *)W.data_ptr(), o.data_ptr<float>(),
                                   m.data_ptr<float>());
  return {o, m};
}

// (5b) S'^T[d][v] = CHAIN64_i(k^[i][d] (MN-major A), d~[v][i]; S^T) vs the recipe's chain S'[v][d] (A = d~ rows)
template <int VR>
__global__ void swap_state_k(const uint16_t *Dt, const uint16_t *K, const float *Cin, float *Dw, float *Dm) {
  __shared__ __align__(1024) uint16_t sk[64 * 128];
  __shared__ __align__(1024) uint16_t sd[VR * 64];
  const Wide<64> kt{sk};
  const Tile<64> dt{sd};
  const uint16_t *dd = Dt + blockIdx.x * VR * 64, *k = K + blockIdx.x * 64 * 128;
  const float *cin = Cin + blockIdx.x * VR * 128;
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32, g = lane / 4, q = lane % 4;
  for (int x = tid; x < 64 * 128; x += 128) kt.at(x / 128, x % 128) = k[x];
  for (int x = tid; x < VR * 64; x += 128) dt.at(x / 64, x % 64) = dd[x];
  fence_async_smem();
  __syncthreads();
  float acc[2][VR / 2];
  for (int db = 0; db < 2; ++db)
    for (int e = 0; e < VR / 2; ++e) {
      const int d = 64 * db + 16 * warp + g + 8 * ((e >> 1) & 1), v = 8 * (e >> 2) + 2 * q + (e & 1);
      acc[db][e] = cin[v * 128 + d];
    }
  wg_fence();
  for (int s = 0; s < 4; ++s)
    for (int db = 0; db < 2; ++db) wgmma_ss<VR, 1>(acc[db], desc_mnmajor_a(kt.half(db).base, 16 * s), desc_kmajor(dt.base, 16 * s), true);
  wg_commit();
  wg_wait<0>();
  wg_pin(acc[0]);
  wg_pin(acc[1]);
  for (int db = 0; db < 2; ++db)
    for (int e = 0; e < VR / 2; ++e) {
      const int d = 64 * db + 16 * warp + g + 8 * ((e >> 1) & 1), v = 8 * (e >> 2) + 2 * q + (e & 1);
      Dw[blockIdx.x * VR * 128 + v * 128 + d] = acc[db][e];
    }
  if (warp < VR / 16) {   // reference: A = d~ rows v, B columns = k^ columns d, seeded
    const int v0 = 16 * warp;
    for (int nt = 0; nt < 16; ++nt) {
      float d4[4];
      for (int e = 0; e < 4; ++e) d4[e] = cin[(v0 + g + 8 * (e >> 1)) * 128 + 8 * nt + 2 * q + (e & 1)];
      for (int ks = 0; ks < 4; ++ks) {
        uint32_t a[4];
        afrag(a, dd, 64, v0 + g, ks, q);
        const int d = 8 * nt + g, i = 16 * ks + 2 * q;
        uint32_t b[2] = {pk(k[i * 128 + d], k[(i + 1) * 128 + d]), pk(k[(i + 8) * 128 + d], k[(i + 9) * 128 + d])};
        mma_c(d4, a, b);
      }
      for (int e = 0; e < 4; ++e) Dm[blockIdx.x * VR * 128 + (v0 + g + 8 * (e >> 1)) * 128 + 8 * nt + 2 * q + (e & 1)] = d4[e];
    }
  }
}
std::vector<torch::Tensor> swap_state(torch::Tensor Dt, torch::Tensor K, torch::Tensor C) {
  auto o = torch::empty_like(C), m = torch::empty_like(C);
  auto launch = Dt.size(1) == 64 ? swap_state_k<64> : (Dt.size(1) == 32 ? swap_state_k<32> : swap_state_k<16>);
  launch<<<Dt.size(0), 128>>>((const uint16_t *)Dt.data_ptr(), (const uint16_t *)K.data_ptr(), C.data_ptr<float>(),
                                    o.data_ptr<float>(), m.data_ptr<float>());
  return {o, m};
}

template <int VR>
__global__ void swap_out_k(const uint16_t *Dv, const uint16_t *M, const float *Cin, float *Dw, float *Dm) {
  __shared__ __align__(1024) uint16_t sm_[64 * 64];
  __shared__ __align__(1024) uint16_t sd[VR * 64];
  const Tile<64> mt{sm_}, dt{sd};
  const uint16_t *dd = Dv + blockIdx.x * VR * 64, *m = M + blockIdx.x * 64 * 64;
  const float *cin = Cin + blockIdx.x * VR * 64;
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32, g = lane / 4, q = lane % 4;
  for (int x = tid; x < 64 * 64; x += 128) mt.at(x / 64, x % 64) = m[x];
  for (int x = tid; x < VR * 64; x += 128) dt.at(x / 64, x % 64) = dd[x];
  fence_async_smem();
  __syncthreads();
  float acc[VR / 2];
  for (int e = 0; e < VR / 2; ++e) {   // acc element (j, v), seeded
    const int j = 16 * warp + g + 8 * ((e >> 1) & 1), v = 8 * (e >> 2) + 2 * q + (e & 1);
    acc[e] = cin[v * 64 + j];
  }
  wg_fence();
  for (int k = 0; k < 4; ++k) wgmma_ss<VR, 0>(acc, desc_kmajor(mt.base, 16 * k), desc_kmajor(dt.base, 16 * k), true);
  wg_commit();
  wg_wait<0>();
  wg_pin(acc);
  for (int e = 0; e < VR / 2; ++e) {
    const int j = 16 * warp + g + 8 * ((e >> 1) & 1), v = 8 * (e >> 2) + 2 * q + (e & 1);
    Dw[blockIdx.x * VR * 64 + v * 64 + j] = acc[e];
  }
  if (warp < VR / 16) {   // reference: A = delta rows v (16 per warp), B columns = M rows j, seeded
    const int v0 = 16 * warp;
    for (int nt = 0; nt < 8; ++nt) {
      float d4[4];
      for (int e = 0; e < 4; ++e) d4[e] = cin[(v0 + g + 8 * (e >> 1)) * 64 + 8 * nt + 2 * q + (e & 1)];
      for (int ks = 0; ks < 4; ++ks) {
        uint32_t a[4];
        afrag(a, dd, 64, v0 + g, ks, q);
        const int j = 8 * nt + g, i = 16 * ks + 2 * q;
        uint32_t b[2] = {pk(m[j * 64 + i], m[j * 64 + i + 1]), pk(m[j * 64 + i + 8], m[j * 64 + i + 9])};
        mma_c(d4, a, b);
      }
      for (int e = 0; e < 4; ++e) Dm[blockIdx.x * VR * 64 + (v0 + g + 8 * (e >> 1)) * 64 + 8 * nt + 2 * q + (e & 1)] = d4[e];
    }
  }
}
std::vector<torch::Tensor> swap_out(torch::Tensor Dv, torch::Tensor M, torch::Tensor C) {
  auto o = torch::empty_like(C), m = torch::empty_like(C);
  auto launch = Dv.size(1) == 64 ? swap_out_k<64> : (Dv.size(1) == 32 ? swap_out_k<32> : swap_out_k<16>);
  launch<<<Dv.size(0), 128>>>((const uint16_t *)Dv.data_ptr(), (const uint16_t *)M.data_ptr(), C.data_ptr<float>(),
                              o.data_ptr<float>(), m.data_ptr<float>());
  return {o, m};
}

"""

CPP = """
#include <torch/extension.h>
torch::Tensor tiles(torch::Tensor A, torch::Tensor B, torch::Tensor C);
torch::Tensor fma_rn(torch::Tensor x, torch::Tensor y, torch::Tensor p);
std::vector<torch::Tensor> skip(torch::Tensor A, torch::Tensor Bc, torch::Tensor seed, torch::Tensor kz);
std::vector<torch::Tensor> kmajor(torch::Tensor A, torch::Tensor B, torch::Tensor C);
std::vector<torch::Tensor> mnmajor(torch::Tensor X, torch::Tensor K);
std::vector<torch::Tensor> swap_delta(torch::Tensor S, torch::Tensor W);
std::vector<torch::Tensor> swap_state(torch::Tensor Dt, torch::Tensor K, torch::Tensor C);
std::vector<torch::Tensor> swap_out(torch::Tensor Dv, torch::Tensor M, torch::Tensor C);
"""


@lru_cache(None)
def ext():
    from torch.utils.cpp_extension import load_inline

    build = _build_dir()
    os.makedirs(build, exist_ok=True)
    return load_inline(
        "gdn_cpr_sm90_mma_facts", CPP, cuda_sources=CUDA,
        functions=["tiles", "fma_rn", "skip", "kmajor", "mnmajor", "swap_delta", "swap_state", "swap_out"],
        extra_cuda_cflags=["-O3", "-std=c++20", "-gencode=arch=compute_90a,code=sm_90a", "--ftz=false",
                           "--fmad=false"],
        extra_include_paths=[str(CSRC)], build_directory=build, verbose=False)


def _bf16(shape, gen, *, wide=False):
    """bf16 values, optionally spread over a wide exponent range (products near over/underflow)."""
    x = torch.randn(*shape, device=DEV, generator=gen)
    if wide:
        x = torch.ldexp(x, torch.randint(-30, 31, shape, device=DEV, generator=gen))
    return x.to(BF)


def test_mma_placement_invariance():
    """Permuting the tile's rows and columns permutes the result, bit for bit; a row/column alone among
    zeros gives the same bits as among real neighbours."""
    g = torch.Generator(device=DEV).manual_seed(0)
    nt = 1 << 14
    A = _bf16((nt, 16, 16), g, wide=True)
    B = _bf16((nt, 16, 8), g, wide=True)
    C = torch.where(torch.rand(nt, 16, 8, device=DEV, generator=g) < 0.5, 0.0,
                    torch.randn(nt, 16, 8, device=DEV, generator=g))
    D = ext().tiles(A.view(torch.int16), B.view(torch.int16), C)
    pr = torch.argsort(torch.rand(nt, 16, device=DEV, generator=g), 1)
    pc = torch.argsort(torch.rand(nt, 8, device=DEV, generator=g), 1)
    A2 = torch.empty_like(A).scatter_(1, pr[:, :, None].expand(-1, -1, 16), A)
    B2 = torch.empty_like(B).scatter_(2, pc[:, None, :].expand(-1, 16, -1), B)
    C2 = torch.empty_like(C).scatter_(1, pr[:, :, None].expand(-1, -1, 8), C)
    C2 = torch.empty_like(C2).scatter_(2, pc[:, None, :].expand(-1, 16, -1), C2)
    D2 = ext().tiles(A2.view(torch.int16), B2.view(torch.int16), C2)
    D2 = torch.gather(D2, 1, pr[:, :, None].expand(-1, -1, 8))
    D2 = torch.gather(D2, 2, pc[:, None, :].expand(-1, 16, -1))
    assert bitwise_equal(D, D2)
    # isolation: keep row 0 and columns 0..2 only
    A3 = torch.zeros_like(A)
    A3[:, 0] = A[:, 0]
    B3 = torch.zeros_like(B)
    B3[:, :, :3] = B[:, :, :3]
    D3 = ext().tiles(A3.view(torch.int16), B3.view(torch.int16), C)
    assert bitwise_equal(D3[:, 0, :3], D[:, 0, :3])


def test_masked_operand_invariance():
    """k-terms with a +0 coefficient contribute nothing whatever the finite other operand is: mma.sync with
    the masked B rows zero vs arbitrary, and ffma(+0, x, p) == p."""
    g = torch.Generator(device=DEV).manual_seed(1)
    nt = 1 << 14
    A = _bf16((nt, 16, 16), g, wide=True)
    kcut = torch.randint(0, 17, (nt, 1, 1), device=DEV, generator=g)
    k = torch.arange(16, device=DEV)
    A = torch.where(k.view(1, 1, 16) < kcut, A, torch.zeros((), dtype=BF, device=DEV))   # +0 columns
    B = _bf16((nt, 16, 8), g, wide=True)
    masked = (k.view(1, 16, 1) >= kcut)
    B_zero = torch.where(masked, torch.zeros((), dtype=BF, device=DEV), B)
    B_junk = torch.where(masked, _bf16((nt, 16, 8), g, wide=True), B)
    C = torch.where(torch.rand(nt, 16, 8, device=DEV, generator=g) < 0.3, 0.0,
                    torch.ldexp(torch.randn(nt, 16, 8, device=DEV, generator=g),
                                torch.randint(-30, 31, (nt, 16, 8), device=DEV, generator=g)))
    assert bitwise_equal(ext().tiles(A.view(torch.int16), B_zero.view(torch.int16), C),
                         ext().tiles(A.view(torch.int16), B_junk.view(torch.int16), C))
    n = 1 << 20
    x = torch.ldexp(torch.randn(n, device=DEV, generator=g), torch.randint(-140, 128, (n,), device=DEV, generator=g))
    x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
    p = torch.ldexp(torch.randn(n, device=DEV, generator=g), torch.randint(-140, 128, (n,), device=DEV, generator=g))
    p = torch.where(torch.isfinite(p), p, torch.zeros_like(p))
    p[: n // 8] = 0.0
    # The one exception is IEEE's: (+0 * x) + (-0) is +0 for x > 0. The recipe's chains are seeded +0
    # and never carry -0 into an fma with a masked term, so p = -0 is excluded here, and shown apart.
    p = torch.where(p == 0, torch.zeros_like(p), p)
    assert bitwise_equal(ext().fma_rn(torch.zeros_like(x), x, p), p)
    negz = torch.full((4,), -0.0, device=DEV)
    assert bitwise_equal(ext().fma_rn(torch.zeros(4, device=DEV), torch.ones(4, device=DEV), negz), torch.zeros(4, device=DEV))


@pytest.mark.parametrize("a_mode", ["normal", "huge", "tiny", "denormal", "negzero"])
@pytest.mark.parametrize("seed_mode", ["random", "+0", "-0"])
def test_zero_column_k16_step_can_be_skipped(a_mode, seed_mode):
    """A CHAIN64 step whose B column is all +0 leaves the accumulator bit-identical to skipping it, for any
    finite A (decode skips the k16 steps past the open rows)."""
    g = torch.Generator(device=DEV).manual_seed(2)
    T = 1 << 13
    scale = {"normal": 4.0, "huge": 1e30, "tiny": 1e-38, "denormal": 3e-41, "negzero": 4.0}[a_mode]
    A = (torch.rand(T, 16, 64, device=DEV, generator=g) * 2 - 1) * scale
    if a_mode == "negzero":
        A[:, :, ::3] = -0.0
    A = A.to(BF)
    assert torch.isfinite(A.float()).all()
    Bc = ((torch.rand(T, 64, device=DEV, generator=g) * 2 - 1) * 2).to(BF)
    seed = {"random": (torch.rand(T, 16, device=DEV, generator=g) * 2 - 1) * 3,
            "+0": torch.zeros(T, 16, device=DEV), "-0": torch.full((T, 16), -0.0, device=DEV)}[seed_mode]
    kz = torch.randint(1, 64, (T,), device=DEV, generator=g, dtype=torch.int32)
    full, skipped = ext().skip(A.view(torch.int16), Bc.view(torch.int16), seed, kz)
    assert bitwise_equal(full, skipped)


@pytest.mark.parametrize("width", [128, 64])
def test_wgmma_ss_kmajor_equals_mma_chain(width):
    """wgmma_n64_ss with both operands K-major in shared memory (the output units: S^ and q^ in Wide<64>
    tiles over 8 k16 steps from +0; the deltas and M in Tile<64> tiles over 4 steps from an fp32 seed)
    == the mma.sync CHAIN128 / seeded CHAIN64."""
    g = torch.Generator(device=DEV).manual_seed(3 + width)
    nb = 256
    A = _bf16((nb, 64, width), g, wide=True)
    B = _bf16((nb, 64, width), g, wide=True)
    C = torch.where(torch.rand(nb, 64, 64, device=DEV, generator=g) < 0.5, 0.0,
                    torch.ldexp(torch.randn(nb, 64, 64, device=DEV, generator=g),
                                torch.randint(-20, 21, (nb, 64, 64), device=DEV, generator=g)))
    wg, mm = ext().kmajor(A.view(torch.int16), B.view(torch.int16), C)
    torch.cuda.synchronize()
    assert bitwise_equal(wg, mm)


def test_wgmma_ss_mnmajor_equals_mma_chain():
    """wgmma_n128_ss<1> (pass 1's u and w: A = xu / xw a Tile<64> K-major, B = v / k^ a Wide<64> read MN-major
    via desc_mnmajor) over 4 k16 steps from +0 == the mma.sync CHAIN64."""
    g = torch.Generator(device=DEV).manual_seed(4)
    nb = 256
    X = _bf16((nb, 64, 64), g, wide=True)
    K = _bf16((nb, 64, 128), g, wide=True)
    wg, mm = ext().mnmajor(X.view(torch.int16), K.view(torch.int16))
    torch.cuda.synchronize()
    assert bitwise_equal(wg, mm)


@pytest.mark.parametrize("rows", [16, 32, 64])
def test_swapped_wgmma_delta_chain_is_the_transpose(rows):
    """wgmma m64n{16,32,64}k16 SS (wgmma_ss, the sweep's widths), A = W [i][d] K-major, B = S^ [v][d] K-major,
    8 steps from +0 == the recipe's CHAIN128(S^, w_i) (A = S^ rows), transposed (the prefill sweep's deltas)."""
    g = torch.Generator(device=DEV).manual_seed(5 + rows)
    nb = 512
    S = _bf16((nb, rows, 128), g, wide=True)
    W = _bf16((nb, 64, 128), g, wide=True)
    wg, mm = ext().swap_delta(S.view(torch.int16), W.view(torch.int16))
    torch.cuda.synchronize()
    assert bitwise_equal(wg, mm)


@pytest.mark.parametrize("rows", [16, 32, 64])
def test_delta_chain_from_the_zero_state_is_plus_zero(rows):
    """From the zero state (S^ = +0) the sweep's delta chain is +0 in every element, whatever the signs of
    the finite w -- so d = rn(bf(u) (-) +0) = u, which the zero-start chunks use without evaluating it
    (prefill.cuh): a zero-start chunk and the same chunk continued from a fresh record agree bitwise."""
    g = torch.Generator(device=DEV).manual_seed(7 + rows)
    nb = 512
    S = torch.zeros(nb, rows, 128, device=DEV, dtype=BF)
    W = _bf16((nb, 64, 128), g, wide=True)
    W[: nb // 2] = -W[: nb // 2].abs()   # all-negative rows: every product is -0
    wg, mm = ext().swap_delta(S.view(torch.int16), W.view(torch.int16))
    torch.cuda.synchronize()
    zero = torch.zeros_like(wg)
    assert bitwise_equal(wg, zero) and bitwise_equal(mm, zero)


@pytest.mark.parametrize("rows", [16, 32, 64])
def test_swapped_wgmma_state_chain_is_the_transpose(rows):
    """wgmma m64n{16,32,64}k16 SS (wgmma_ss), A = the k^ tile [i][d] read MN-major (tnspA, desc_mnmajor_a),
    B = d~ [v][i] K-major, 4 steps seeded with S^T == the recipe's seeded CHAIN64(d~, k^; S) (A = d~ rows),
    transposed (the prefill sweep's boundary update)."""
    g = torch.Generator(device=DEV).manual_seed(6 + rows)
    nb = 512
    Dt = _bf16((nb, rows, 64), g, wide=True)
    K = _bf16((nb, 64, 128), g, wide=True)
    C = torch.where(torch.rand(nb, rows, 128, device=DEV, generator=g) < 0.3, 0.0,
                    torch.ldexp(torch.randn(nb, rows, 128, device=DEV, generator=g),
                                torch.randint(-20, 21, (nb, rows, 128), device=DEV, generator=g)))
    wg, mm = ext().swap_state(Dt.view(torch.int16), K.view(torch.int16), C)
    torch.cuda.synchronize()
    assert bitwise_equal(wg, mm)


@pytest.mark.parametrize("rows", [16, 32, 64])
def test_swapped_wgmma_output_chain_is_the_transpose(rows):
    """wgmma m64n{16,32,64}k16 SS, A = M [j][i] K-major, B = the deltas [v][i] K-major, 4 steps seeded with
    the output seed^T == the recipe's seeded CHAIN64(d, M) (A = delta rows), transposed (the fused mode's
    outputs, value rows as N)."""
    g = torch.Generator(device=DEV).manual_seed(8 + rows)
    nb = 512
    Dv = _bf16((nb, rows, 64), g, wide=True)
    M = _bf16((nb, 64, 64), g, wide=True)
    C = torch.where(torch.rand(nb, rows, 64, device=DEV, generator=g) < 0.3, 0.0,
                    torch.ldexp(torch.randn(nb, rows, 64, device=DEV, generator=g),
                                torch.randint(-20, 21, (nb, rows, 64), device=DEV, generator=g)))
    wg, mm = ext().swap_out(Dv.view(torch.int16), M.view(torch.int16), C)
    torch.cuda.synchronize()
    assert bitwise_equal(wg, mm)
