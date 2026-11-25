"""A plain reference of the sm90 segment-tree program, independent of the CUTLASS kernel.

The program (``spec.RECIPE_VERSION``) has three parts, and the reference keeps them apart:

1. the LEAF: one segment's ascending fp32 chain. It is a tensor-core fact (wgmma's k16 step), so no CPU
   code can reproduce its bits. :func:`chain_leaf_triton` computes it with an independent kernel -- a
   Triton ``tl.dot`` chain, which pik's arch self-check (``pik/arch.py --verify``) establishes is
   the canonical Hopper chain -- and :func:`fp64_leaf` gives a numerically faithful CPU stand-in for
   tests of the structure.
2. the TREE: the balanced adjacent-pair tree over the segment index, every add fp32, left operand the
   lower indices (``pik.plan.combine_order``). Exact in torch.
3. the ROUND: one bf16 RN rounding at the root (optionally each segment root first: pik bf16 leaves).

NaN: the device rounds with ``cvt.rn.bf16.f32`` (canonical NaN), torch's cast keeps the payload; compare
NaN positions with :func:`same_bits_mod_nan`.
"""

from __future__ import annotations

from typing import Callable

import torch

Leaf = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def fp64_leaf(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """CPU stand-in for a leaf: the exact dot product, rounded once to fp32. NOT the device's bits."""
    return (x.double() @ w.double().t()).float()


def combine_order(num: int) -> list[tuple[int, int, int]]:
    """The fixed balanced tree, as (dst, lhs, rhs) slot triples (the same as ``pik.plan.combine_order``)."""
    if num < 1 or num & (num - 1):
        raise ValueError(f"segment count must be a power of two, got {num}")
    sched = []
    stride = 1
    while stride < num:
        for i in range(0, num, 2 * stride):
            sched.append((i, i, i + stride))
        stride *= 2
    return sched


def tree(values: list[torch.Tensor]) -> torch.Tensor:
    """Evaluate the balanced tree over fp32 tensors."""
    slots = [v.float() for v in values]
    for dst, lhs, rhs in combine_order(len(slots)):
        slots[dst] = slots[lhs] + slots[rhs]
    return slots[0]


def segtree(x: torch.Tensor, w: torch.Tensor, S: int, *, leaf: Leaf = fp64_leaf, round_leaf: bool = False,
            out_dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """``x[M, K] @ w[N, K]^T`` under the program with S segments: leaves, tree, one round."""
    K = x.shape[1]
    if K % S:
        raise ValueError(f"S={S} does not divide K={K}")
    sk = K // S
    roots = [leaf(x[:, s * sk:(s + 1) * sk], w[:, s * sk:(s + 1) * sk]).float() for s in range(S)]
    if round_leaf:
        roots = [r.to(torch.bfloat16).float() for r in roots]
    root = tree(roots)
    return root if out_dtype is torch.float32 else root.to(out_dtype)


_CHAIN = None


def _chain_kernel():
    global _CHAIN
    if _CHAIN is not None:
        return _CHAIN
    import triton
    import triton.language as tl

    @triton.jit
    def _chain(A, B, C, M, N, K, sam, sbn, scm, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pid = tl.program_id(0)
        npn = tl.cdiv(N, BN)
        pm, pn = pid // npn, pid % npn
        om = (pm * BM + tl.arange(0, BM)) % M
        on = (pn * BN + tl.arange(0, BN)) % N
        ok = tl.arange(0, BK)
        ap = A + om[:, None] * sam + ok[None, :]
        bp = B + on[:, None] * sbn + ok[None, :]
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            # the program's tail: elements past K are zeros (TMA out-of-bounds fill on the device)
            km = (k0 + ok) < K
            acc = tl.dot(tl.load(ap, mask=km[None, :], other=0.0), tl.trans(tl.load(bp, mask=km[None, :], other=0.0)),
                         acc)
            ap += BK
            bp += BK
        rm = pm * BM + tl.arange(0, BM)
        rn = pn * BN + tl.arange(0, BN)
        tl.store(C + rm[:, None] * scm + rn[None, :], acc, mask=(rm[:, None] < M) & (rn[None, :] < N))

    _CHAIN = _chain
    return _CHAIN


def chain_leaf_triton(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """One ascending fp32 ``tl.dot`` chain over all of K zero-padded to whole 64-wide blocks (64-row tiles:
    wgmma on Hopper), fp32 out."""
    import triton

    _chain = _chain_kernel()
    M, K = x.shape
    N = w.shape[0]
    x = x.contiguous()
    w = w.contiguous()
    c = torch.empty((M, N), dtype=torch.float32, device=x.device)
    BM, BN = 64, 64
    _chain[(triton.cdiv(M, BM) * triton.cdiv(N, BN),)](x, w, c, M, N, K, x.stride(0), w.stride(0), c.stride(0),
                                                       BM=BM, BN=BN, BK=64, num_warps=4, num_stages=3)
    return c


def same_bits_mod_nan(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bitwise equality, with every NaN equal to every NaN (the device canonicalises NaN payloads)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    it = {torch.bfloat16: torch.int16, torch.float32: torch.int32, torch.float16: torch.int16}[a.dtype]
    na, nb = torch.isnan(a), torch.isnan(b)
    if not torch.equal(na, nb):
        return False
    return torch.equal(a.view(it)[~na], b.view(it)[~nb])


__all__ = ["chain_leaf_triton", "combine_order", "fp64_leaf", "same_bits_mod_nan", "segtree", "tree"]
