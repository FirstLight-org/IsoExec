"""BF16 dense projections of the actor recipe on vendored DeepGEMM (``bf16_gemm_nt``, deterministic mode).

Program (per output element): ONE ascending K chain of 16-wide ``tcgen05.mma kind::f16`` steps (BLOCK_K = 64,
UMMA_K = 16) from a non-accumulating first step, FP32 accumulator in tensor memory, one BF16 RNE of the root; no
split-K (deterministic mode never splits K and never dispatches to cuBLASLt). That is ``mm_sm100``'s
``sm100_tcgen05_segtree_v2`` program at a segment count S = 1, so this provider serves ONLY weights whose
schedule is one segment (``mm_sm100.segments.segment_count``: the schedule policy's G, 1 for every K) and the bits are
those of the incumbent provider (the GPU gate compares them at every M). Tile shape, swap-AB, cluster, persistent schedule and PDL are ownership / launch
timing only. Weights with S > 1 keep the incumbent provider (``aten::mm`` on the folded rows, through the installed
``mm`` chain: exactly ``F.linear``'s decomposition, but a runtime that overrides the composite ``aten::linear`` /
``aten::matmul``, such as vLLM's batch-invariant mode, cannot move it off the chain).

Backward (``DGLinearFn``): exactly the GEMMs autograd runs for ``F.linear``'s ``mm`` (``dX = dY W``,
``dW = dY^T X``, through the installed provider chain), so gradients are bitwise those of ``F.linear``.
"""
from __future__ import annotations

import torch

IMPLEMENTATION = "vendored_deepgemm_sm100_bf16_gemm_v1"


def serves(n: int, k: int) -> bool:
    """True when this provider's program equals the incumbent's for an ``[n, k]`` BF16 weight: the schedule
    policy's segment count (``mm_sm100.segments``) is 1 (one chain) and the shape is a legal DeepGEMM operand."""
    from isoexec.ops.mm.mm_sm100 import segments as seg
    if k % 64 or n % 16 or not seg.supported(k, n):
        return False
    return seg.segment_count(k, torch.bfloat16) == 1


#: Measured per-shape launch table (dense lane r35 / r36 / r37: paired per-group ratios vs native, two runs for the
#: early mode, bits equal in every arm): weight shape (N, K) -> [(M lo, M hi, early mode or None, block-size multiple
#: or None)]. Keyed on the weight shape and M only; launch timing / tile ownership, never bits. The fused
#: wq_b | idx wq_b weight: M 1-32 the lean post-wait path (0.98x of the default, both runs); M 64-128 the non-swapped
#: layout (r36: 1.04 / 0.97 / 1.03 vs native, default 1.11 / 1.05 / 1.17). Rows outside the bands keep the default.
SHAPE_TABLE = {(12288, 1280): ((1, 32, 65, None), (64, 128, None, (512, 1)))}


def _policy(m, n, k):
    for lo, hi, early, multiple in SHAPE_TABLE.get((n, k), ()):
        if lo <= m <= hi:
            return early, multiple
    return None, None


def _gemm(x2, w):
    from . import jit_warm
    from . import ops as dgo
    jit_warm.note("gemm", n=int(w.shape[0]), k=int(w.shape[1]))
    out = torch.empty((x2.shape[0], w.shape[0]), dtype=torch.bfloat16, device=x2.device)
    if x2.shape[0]:
        early, multiple = _policy(x2.shape[0], w.shape[0], w.shape[1])
        if multiple is None:
            dgo._launch(lambda dg: dg.bf16_gemm_nt(x2, w, out), weight_b=True, early=early)
        else:
            dg = dgo._dg()
            dg.set_block_size_multiple_of(multiple)
            try:
                dgo._launch(lambda g: g.bf16_gemm_nt(x2, w, out), weight_b=True, early=early)
            finally:
                dg.set_block_size_multiple_of((1, 1))
    return out


def _check(x, w):
    if x.dtype != torch.bfloat16 or w.dtype != torch.bfloat16 or w.dim() != 2 or x.shape[-1] != w.shape[1]:
        raise ValueError("DeepGEMM BF16 linear needs BF16 x [..., K] and weight [N, K]")
    if not w.is_contiguous():
        raise ValueError("DeepGEMM BF16 linear reads K-major weights in place")


class DGLinearFn(torch.autograd.Function):
    """``F.linear(x, w)`` with the DeepGEMM forward; backward = ``F.linear``'s (``mm`` through the provider chain)."""

    @staticmethod
    def forward(ctx, x, w):
        _check(x, w)
        x2 = x.reshape(-1, x.shape[-1])
        if x2.stride(-1) != 1 or x2.stride(0) != x2.shape[1]:
            x2 = x2.contiguous()
        ctx.save_for_backward(x, w)
        return _gemm(x2, w).view(*x.shape[:-1], w.shape[0])

    @staticmethod
    def backward(ctx, dy):
        x, w = ctx.saved_tensors
        dy2 = dy.reshape(-1, w.shape[0])
        dx = dw = None
        if ctx.needs_input_grad[0]:
            dx = torch.mm(dy2, w).view(x.shape)
        if ctx.needs_input_grad[1]:
            dw = torch.mm(dy2.t(), x.reshape(-1, x.shape[-1]))
        return dx, dw


#: Rows up to which a weight at least this wide stays on the incumbent kernel (measured faster there: the same
#: one-chain program, so the choice is per call and bit-neutral).
INCUMBENT_ROWS, INCUMBENT_MIN_N = 8, 32768


def incumbent(x, w):
    """``F.linear(x, w)`` (no bias) as it decomposes: ``aten::mm`` of the folded rows with ``w^T``."""
    return torch.mm(x.reshape(-1, x.shape[-1]), w.t()).view(*x.shape[:-1], w.shape[0])


def linear(x, w):
    """``F.linear(x, w)`` (no bias) on DeepGEMM for S = 1 weights, the incumbent provider otherwise."""
    if not serves(*w.shape) or x.dtype != torch.bfloat16 or w.dtype != torch.bfloat16 \
            or not w.is_contiguous() or not x.is_cuda:
        return incumbent(x, w)
    if w.shape[0] >= INCUMBENT_MIN_N and x.numel() // x.shape[-1] <= INCUMBENT_ROWS:
        return incumbent(x, w)
    if torch.is_grad_enabled() and (x.requires_grad or w.requires_grad):
        return DGLinearFn.apply(x, w)
    _check(x, w)
    x2 = x.reshape(-1, x.shape[-1])
    if x2.stride(-1) != 1 or x2.stride(0) != x2.shape[1]:
        x2 = x2.contiguous()
    return _gemm(x2, w).view(*x.shape[:-1], w.shape[0])


__all__ = ["DGLinearFn", "IMPLEMENTATION", "INCUMBENT_MIN_N", "INCUMBENT_ROWS", "incumbent", "linear",
           "serves"]
