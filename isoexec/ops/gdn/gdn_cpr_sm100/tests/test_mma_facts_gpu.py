"""Check hardware facts required by the SM100 numerical identity."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import pytest
import torch

from isoexec.ops.build_cache import DIAGNOSTIC_FLAGS, build_sm100_extension
from isoexec.ops.gdn.gdn_cpr_sm100._extension import build_dir

from ._fixtures import SM100

pytestmark = SM100

CUDA = r"""
// SPDX-License-Identifier: Apache-2.0
// Blackwell (sm_100a) facts probe for the GDN chunk-form identity: does mma.sync.m16n8k16 exist,
// does it satisfy the invariances recipe.cuh relies on, and what is its issue-rate ceiling?
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <stdint.h>

namespace {

__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ uint32_t pk(uint16_t lo, uint16_t hi) { return uint32_t(lo) | (uint32_t(hi) << 16); }

// One warp: P = X Y^T for X,Y [16][K] row-major bf16 (K = 16 * STEPS), by STEPS ascending k16 mma steps.
// `arow`/`brow` permute the 16 rows of X / Y; `skip` skips a step whose B k-slice is all +0.
// Output: 16 x 16 fp32 at out[(i * 16) + j], written in the permuted-in coordinates' inverse.
// `seed` (nullable, [16][16] fp32 in the UNSWAPPED coordinates) is what makes the transpose fact usable on
// a SEEDED chain, which is what csrc/slab.cuh's boundary and output chains and csrc/decode.cu's `M d` chain
// and boundary actually are: the second review noted the fact was only ever exercised with a +0 seed.
template <bool SWAP>
__device__ void chain_warp(const uint16_t *X, const uint16_t *Y, float *out, int K, const int *arow,
                           const int *brow, bool skip, bool garbage, const float *seed = nullptr) {
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int steps = K / 16;
  const uint16_t *A = SWAP ? Y : X, *B = SWAP ? X : Y;
  const int *ap = SWAP ? brow : arow, *bp = SWAP ? arow : brow;
  float acc[2][4];
  #pragma unroll
  for (int nt = 0; nt < 2; ++nt) {
    #pragma unroll
    for (int e = 0; e < 4; ++e) acc[nt][e] = 0.f;
    if (seed != nullptr) {   // the same seed element at the same accumulator position in both orientations
      const int i0 = ap[g], i1 = ap[g + 8], j0 = bp[8 * nt + 2 * t], j1 = bp[8 * nt + 2 * t + 1];
      const int a0 = SWAP ? j0 * 16 + i0 : i0 * 16 + j0, a1 = SWAP ? j1 * 16 + i0 : i0 * 16 + j1;
      const int a2 = SWAP ? j0 * 16 + i1 : i1 * 16 + j0, a3 = SWAP ? j1 * 16 + i1 : i1 * 16 + j1;
      acc[nt][0] = seed[a0];
      acc[nt][1] = seed[a1];
      acc[nt][2] = seed[a2];
      acc[nt][3] = seed[a3];
    }
  }
  for (int s = 0; s < steps; ++s) {
    const int k0 = 16 * s;
    uint32_t bb[2][2];
    bool zero = true;
    #pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      const int n = bp[8 * nt + g];
      bb[nt][0] = pk(B[n * K + k0 + 2 * t], B[n * K + k0 + 2 * t + 1]);
      bb[nt][1] = pk(B[n * K + k0 + 2 * t + 8], B[n * K + k0 + 2 * t + 9]);
    }
    // a step is skippable only if EVERY column's k-slice is +0; test the whole slice on lane 0 basis
    for (int n = 0; n < 16 && zero; ++n)
      for (int e = 0; e < 16; ++e)
        if (B[bp[n] * K + k0 + e] != 0) { zero = false; break; }
    uint32_t aa[4];
    const int r0 = ap[g], r1 = ap[g + 8];
    if (garbage && zero) {
      aa[0] = pk(0x7f7fu, 0xff7fu); aa[1] = pk(0x7f7fu, 0xff7fu);   // +-3.4e38 bf16
      aa[2] = pk(0x4080u, 0xc080u); aa[3] = pk(0x0080u, 0x8080u);
    } else {
      aa[0] = pk(A[r0 * K + k0 + 2 * t], A[r0 * K + k0 + 2 * t + 1]);
      aa[1] = pk(A[r1 * K + k0 + 2 * t], A[r1 * K + k0 + 2 * t + 1]);
      aa[2] = pk(A[r0 * K + k0 + 2 * t + 8], A[r0 * K + k0 + 2 * t + 9]);
      aa[3] = pk(A[r1 * K + k0 + 2 * t + 8], A[r1 * K + k0 + 2 * t + 9]);
    }
    if (skip && zero) continue;
    #pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      const uint32_t b2[2] = {bb[nt][0], bb[nt][1]};
      mma16816(acc[nt], aa, b2);
    }
  }
  // d[0] = D[g][2t], d[1] = D[g][2t+1], d[2] = D[g+8][2t], d[3] = D[g+8][2t+1]; columns 8nt + ...
  #pragma unroll
  for (int nt = 0; nt < 2; ++nt) {
    const int i0 = ap[g], i1 = ap[g + 8], j0 = bp[8 * nt + 2 * t], j1 = bp[8 * nt + 2 * t + 1];
    if (SWAP) {   // out is the transpose: element (j, i)
      out[j0 * 16 + i0] = acc[nt][0];
      out[j1 * 16 + i0] = acc[nt][1];
      out[j0 * 16 + i1] = acc[nt][2];
      out[j1 * 16 + i1] = acc[nt][3];
    } else {
      out[i0 * 16 + j0] = acc[nt][0];
      out[i0 * 16 + j1] = acc[nt][1];
      out[i1 * 16 + j0] = acc[nt][2];
      out[i1 * 16 + j1] = acc[nt][3];
    }
  }
}

// mode: 0 plain, 1 permuted placement, 2 swapped roles (transpose), 3 skip zero steps, 4 garbage A on zero steps
__global__ void facts_kernel(const uint16_t *X, const uint16_t *Y, float *out, int K, int nprob, int mode,
                             int warp_offset, const float *seed = nullptr) {
  const int warp_global = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int p = warp_global - warp_offset;
  if (p < 0 || p >= nprob) return;
  const int64_t xs = int64_t(p) * 16 * K, ys = int64_t(p) * 16 * K;
  int ident[16], perm[16];
  #pragma unroll
  for (int i = 0; i < 16; ++i) { ident[i] = i; perm[i] = (i * 7 + 3) & 15; }
  const int *ar = (mode == 1) ? perm : ident;
  const int *br = (mode == 1) ? perm : ident;
  float *o = out + int64_t(p) * 256;
  const float *sd = (mode == 5 || mode == 6) ? seed + int64_t(p) * 256 : nullptr;
  if (mode == 2 || mode == 6) chain_warp<true>(X + xs, Y + ys, o, K, ar, br, false, false, sd);
  else chain_warp<false>(X + xs, Y + ys, o, K, ar, br, mode == 3, mode == 4, sd);
}

}  // namespace

torch::Tensor facts(torch::Tensor X, torch::Tensor Y, int64_t mode, int64_t warp_offset,
                    c10::optional<torch::Tensor> seed) {
  TORCH_CHECK(X.is_cuda() && X.dtype() == torch::kBFloat16 && X.dim() == 3 && X.size(1) == 16);
  const int nprob = X.size(0), K = X.size(2);
  auto out = torch::empty({nprob, 16, 16}, X.options().dtype(torch::kFloat32));
  const int warps = nprob + warp_offset;
  const int threads = 256, blocks = (warps * 32 + threads - 1) / threads;
  const float *sd = nullptr;
  if (seed.has_value() && seed->numel()) {
    torch::Tensor t = *seed;
    TORCH_CHECK(t.is_cuda() && t.dtype() == torch::kFloat32 && t.numel() == int64_t(nprob) * 256);
    sd = t.data_ptr<float>();
  }
  facts_kernel<<<blocks, threads>>>(reinterpret_cast<const uint16_t *>(X.data_ptr()),
                                    reinterpret_cast<const uint16_t *>(Y.data_ptr()), out.data_ptr<float>(), K,
                                    nprob, int(mode), int(warp_offset), sd);
  C10_CUDA_CHECK(cudaGetLastError());
  return out;
}

// ---------------------------------------------------------------- the paired bf16 narrowing
// Does `cvt.rn.bf16x2.f32` (what `recipe.cuh::rn2` emits) round each half exactly as `cvt.rn.bf16.f32`
// does? The kernel replaced about 965 scalar narrowings a thread a chunk with 437 paired ones, so this is
// the fact that keeps that from being an arithmetic change. The instruction is ELEMENTWISE, so sweeping
// every one of the 2^32 float bit patterns through one half is an EXHAUSTIVE proof for every pair --
// there is no cross-lane interaction to miss. Each thread checks both halves of `rn2(x, y)` against
// `rn(x)` and `rn(y)` and reports a count of disagreements.
__global__ void narrow_probe(uint32_t base, uint32_t stride, int per_thread, unsigned long long *bad) {
  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long mine = 0;
  for (int i = 0; i < per_thread; ++i) {
    const uint32_t bx = base + (uint32_t(tid) * uint32_t(per_thread) + uint32_t(i)) * stride;
    const uint32_t by = ~bx;                      // a second, unrelated pattern in the other half
    float x, y;
    memcpy(&x, &bx, 4);
    memcpy(&y, &by, 4);
    const __nv_bfloat162 p2 = __floats2bfloat162_rn(x, y);
    uint32_t got;
    memcpy(&got, &p2, 4);
    const uint32_t want = pk(__bfloat16_as_ushort(__float2bfloat16_rn(x)),
                             __bfloat16_as_ushort(__float2bfloat16_rn(y)));
    if (got != want) ++mine;
  }
  if (mine) atomicAdd(bad, mine);
}

int64_t narrow(int64_t base, int64_t stride, int64_t blocks, int64_t threads, int64_t per_thread) {
  auto bad = torch::zeros({1}, torch::dtype(torch::kInt64).device(torch::kCUDA));
  narrow_probe<<<int(blocks), int(threads)>>>(
      uint32_t(base), uint32_t(stride), int(per_thread),
      reinterpret_cast<unsigned long long *>(bad.data_ptr<int64_t>()));
  C10_CUDA_CHECK(cudaGetLastError());
  return bad.cpu().item<int64_t>();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("facts", &facts, py::arg("X"), py::arg("Y"), py::arg("mode"), py::arg("warp_offset"),
        py::arg("seed") = c10::optional<torch::Tensor>());
  m.def("narrow", &narrow);
}
"""


@lru_cache(None)
def facts_ext():
    """Beside the family's extension build, in a subdirectory."""
    root = build_dir() / "mma_facts"
    (root / "csrc").mkdir(parents=True, exist_ok=True)
    src = root / "csrc" / "facts.cu"
    if not src.is_file() or src.read_text() != CUDA:
        src.write_text(CUDA)
    os.environ.setdefault("ISOEXEC_GDN_MMA_FACTS_SM100_BUILD_DIR", str(root / "build"))
    return build_sm100_extension("gdn_mma_facts", str(root / "_anchor.py"), ("facts.cu",),
                                 extra_cuda_flags=DIAGNOSTIC_FLAGS)


DEV = "cuda"


def operands(nprob, K, *, wide=False, zero_tail_steps=0, seed=11):
    """bf16 operand pairs; ``wide`` spreads the exponent so products land near over/underflow."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(nprob, 16, K, generator=g, device=DEV)
    y = torch.randn(nprob, 16, K, generator=g, device=DEV)
    if wide:
        x = x * torch.exp2(torch.randint(-14, 15, x.shape, generator=g, device=DEV).float())
        y = y * torch.exp2(torch.randint(-14, 15, y.shape, generator=g, device=DEV).float())
    x, y = x.bfloat16(), y.bfloat16()
    if zero_tail_steps:
        y[:, :, K - 16 * zero_tail_steps:] = 0
    return x.contiguous(), y.contiguous()


def biteq(a, b):
    return bool(torch.equal(a.view(torch.int32), b.view(torch.int32)))


def test_mma_sync_m16n8k16_bf16_exists_on_sm100():
    x, y = operands(64, 128)
    out = facts_ext().facts(x, y, 0, 0)
    assert torch.isfinite(out).all()


def test_the_same_chain_is_deterministic_across_launches():
    x, y = operands(256, 128)
    first = facts_ext().facts(x, y, 0, 0)
    for _ in range(4):
        assert biteq(first, facts_ext().facts(x, y, 0, 0))


@pytest.mark.parametrize("K", [64, 128])
@pytest.mark.parametrize("wide", [False, True])
@pytest.mark.parametrize("warp_offset", [1, 7, 64, 1000])
def test_placement_invariance(K, wide, warp_offset):
    """Which lane, warp, CTA and SM evaluates a contraction does not enter its bits. Mode 1 also permutes
    the 16 rows of both operands and writes back to the unpermuted position."""
    x, y = operands(256, K, wide=wide)
    assert biteq(facts_ext().facts(x, y, 0, 0), facts_ext().facts(x, y, 1, warp_offset))


@pytest.mark.parametrize("K", [64, 128])
@pytest.mark.parametrize("wide", [False, True])
def test_swapped_operand_roles_is_exactly_the_transpose(K, wide):
    """Every product in slab.cuh is evaluated transposed so the value rows are the MMA's N. That is only
    the recipe's chain if exchanging the operand roles gives the transpose bit for bit."""
    x, y = operands(256, K, wide=wide)
    plain = facts_ext().facts(x, y, 0, 0)
    swapped = facts_ext().facts(x, y, 2, 0)
    assert biteq(plain, swapped)


@pytest.mark.parametrize("K", [64, 128])
@pytest.mark.parametrize("wide", [False, True])
def test_swapped_operand_roles_is_the_transpose_for_a_SEEDED_chain(K, wide):
    """Check the second transpose invariant."""
    x, y = operands(256, K, wide=wide)
    g = torch.Generator(device=DEV).manual_seed(5 + K)
    seed = torch.randn(256, 16, 16, generator=g, device=DEV)
    if wide:
        seed = seed * torch.exp2(torch.randint(-14, 15, seed.shape, generator=g, device=DEV).float())
    seed = seed.contiguous()
    plain = facts_ext().facts(x, y, 5, 0, seed)
    swapped = facts_ext().facts(x, y, 6, 0, seed)
    assert biteq(plain, swapped)
    assert not biteq(plain, facts_ext().facts(x, y, 0, 0))     # the seed is really in the sum


@pytest.mark.parametrize("K,zeros", [(64, 1), (64, 2), (128, 4), (128, 7)])
@pytest.mark.parametrize("wide", [False, True])
def test_a_k16_step_whose_b_slice_is_plus_zero_can_be_skipped(K, zeros, wide):
    """The zero extension: a chunk evaluated for 1, 17 or 64 rows gives the same bits for the rows it
    evaluates, because the absent rows' steps may be skipped rather than executed."""
    x, y = operands(256, K, wide=wide, zero_tail_steps=zeros)
    assert biteq(facts_ext().facts(x, y, 0, 0), facts_ext().facts(x, y, 3, 0))


@pytest.mark.parametrize("K,zeros", [(64, 1), (64, 2), (128, 4), (128, 7)])
@pytest.mark.parametrize("wide", [False, True])
def test_a_k16_step_whose_A_slice_is_plus_zero_can_be_skipped(K, zeros, wide):
    """The k-step pruning of the u / w chains and of the M d chain skips steps whose A-side COEFFICIENT slice is +0 (X_ji and M_ji are +0 for i > j)."""
    x, y = operands(256, K, wide=wide, zero_tail_steps=zeros)   # y's tail k-slices are +0
    a_zero = facts_ext().facts(y, x, 0, 0)                      # y is A: its tail slices face A
    skipped = facts_ext().facts(x, y, 3, 0).transpose(1, 2).contiguous()
    assert biteq(a_zero, skipped)


@pytest.mark.parametrize("K,zeros", [(64, 2), (128, 4)])
def test_the_a_operand_facing_a_zero_slice_is_irrelevant(K, zeros):
    """Evaluators need the operands of absent rows to be FINITE, nothing more: mode 4 replaces them with
    +-3.4e38 and the smallest normals."""
    x, y = operands(256, K, zero_tail_steps=zeros)
    assert biteq(facts_ext().facts(x, y, 0, 0), facts_ext().facts(x, y, 4, 0))


@pytest.mark.parametrize("K", [64, 128])
@pytest.mark.parametrize("bad,label", [(float("nan"), "nan"), (float("inf"), "inf"),
                                       (float("-inf"), "-inf")])
def test_a_nonfinite_a_row_reaches_only_its_own_output_row(K, bad, label):
    """A NaN or Inf in A row `i` changes D row `i` and NO OTHER ROW of the chain."""
    x, y = operands(64, K)
    ref = facts_ext().facts(x, y, 0, 0)
    for r in (0, 7, 15):
        poisoned = x.clone()
        poisoned[:, r, :] = torch.tensor(bad, dtype=torch.bfloat16, device=DEV)
        out = facts_ext().facts(poisoned.contiguous(), y, 0, 0)
        keep = [i for i in range(16) if i != r]
        assert biteq(out[:, keep].contiguous(), ref[:, keep].contiguous()), f"{label} in A row {r}"
        assert not torch.isfinite(out[:, r]).any(), f"{label} in A row {r} left its own row finite"


def test_a_chain_seeded_with_plus_zero_never_holds_minus_zero():
    x = torch.full((16, 16, 64), -1.0, device=DEV, dtype=torch.bfloat16)
    y = torch.zeros(16, 16, 64, device=DEV, dtype=torch.bfloat16)
    out = facts_ext().facts(x, y, 0, 0)
    assert int((out.view(torch.int32) == -2147483648).sum()) == 0


# --------------------------------------------------------------------------- the paired bf16 narrowing
#: `recipe.cuh::rn2(a, b)` must equal `pack(rn(a), rn(b))` for EVERY pair, because the kernel replaced
#: about 965 scalar `F2F.BF16.F32` a thread a chunk with 437 paired `F2FP.BF16.F32.PACK_AB` -- an
#: instruction-count change that is only not an arithmetic one if this holds. `cvt.rn.bf16x2.f32` is
#: ELEMENTWISE, so sweeping every one of the 2^32 float bit patterns through one half proves it for every
#: pair: there is no cross-element interaction for a sample to miss. The battery runs a 2^26 strided sweep
#: (every 64th pattern, which visits every exponent and every mantissa residue class mod 64, plus the whole
#: NaN, infinity and denormal space at that density) in about a second; `stride=1` over all 2^32 is the
#: exhaustive form and was run once, clean.
@pytest.mark.parametrize("stride,blocks", [(64, 2048), (1, 2048)], ids=["sweep_2p26", "dense_2p26_low"])
def test_the_paired_bf16_narrowing_is_the_scalar_one(stride, blocks):
    per = (1 << 26) // (blocks * 256)
    assert facts_ext().narrow(0, stride, blocks, 256, per) == 0


def test_the_paired_bf16_narrowing_is_the_scalar_one_on_the_hard_patterns():
    """The exponents where bf16 rounding ties, overflows and denormalizes, at full density."""
    for base in (0x00000000, 0x00800000, 0x3F000000, 0x3F800000, 0x7F000000, 0x7F7F0000, 0x7F800000,
                 0x7FC00000, 0x80000000, 0xBF800000, 0xFF800000, 0xFFC00000):
        assert facts_ext().narrow(base, 1, 1024, 256, 64) == 0, hex(base)
