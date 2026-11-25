"""The ``mm_sm100`` segment-tree program (``sm100_tcgen05_segtree_v2``) with the segments on vendored DeepGEMM.

``out = tree_S(P_0, ..., P_{S-1})`` with ``P_s = x[:, s*Ks:(s+1)*Ks] @ W[:, s*Ks:(s+1)*Ks]^T`` (``Ks = K / S``):

* every ``P_s`` element is ONE ascending K chain of 16-wide ``tcgen05.mma kind::f16`` steps from a non-accumulating
  first step, FP32 accumulator (DeepGEMM's batched BF16 GEMM, einsum ``bhr,hdr->bhd`` with h = segment, b = token,
  r = the segment's K; deterministic mode: no split-K, no cuBLASLt). That is ``mm_sm100``'s segment chain.
* the S FP32 segment roots ``[S, M, N]`` are folded by the balanced adjacent-pair tree (``__fadd_rn``) and rounded
  once to BF16 (RNE), or returned as the FP32 root: the attention extension's ``leaf_tree`` kernel, which
  ``wo_b_tree``'s publishing plans already use to finish ``mm_sm100``'s tree (same tree, same bits).

So this is ``mm_sm100``'s cut-tree plan (one CTA per segment, ``COMBINE`` mode) executed by other kernels: the bits
are those of ``mm_sm100`` at the same S (``segments.segment_count``), which the GPU gate compares at every M. S is the
caller's: a fixed leaf count (``wo_b``'s TP leaves, the router's logit segments) or the weight's published
segment count (1 for every K under mm_sm100 policy v4). Tile shape,
swap-AB and the CTA count are ownership only. Forward only (no autograd); the callers keep their incumbent
paths under autograd.

Why: at decode sizes the incumbent plans of the skinny-N, long-K weights (``wq_a``, ``wkv``, the compressor, the
indexer's ``weights_proj``) and of ``wo_b``'s leaf tree keep a whole segment chain set (all of K) on one CTA pair, so
a launch busies few SMs; one CTA per segment spreads the same chains over S times the SMs.
"""
from __future__ import annotations

import torch

#: Block-size policy for the vendored heuristic per launch: ``(512, 1)`` excludes the swap-AB layouts (no swap-AB
#: block M is a multiple of 512), so it picks the non-swapped layout with a narrow weight tile (BLOCK_N 16 / 32):
#: more CTAs for these skinny weights. Ownership only.
NOSWAP = (512, 1)

#: Leaf trees with the fold's counter partials in the registers of two epilogue warpgroups (vendored kernel
#: ``sm100_fold_leaf_pair_regs`` / ``sm100_store_cd_leaf_regs``): tensor memory then holds the leaf ring only.
#: Storage location only, the same tree nodes; opt-in (ISOEXEC_DEEPGEMM_LEAF_REGS=1) until measured.
LEAF_REGS = __import__("os").environ.get("ISOEXEC_DEEPGEMM_LEAF_REGS", "0") == "1"


def serves(n: int, k: int, segments: int) -> bool:
    """True when the segment GEMM is a legal DeepGEMM operand set: whole 64-wide K blocks per segment, 16-byte
    aligned segment strides, N a multiple of 16."""
    return segments >= 1 and k % segments == 0 and (k // segments) % 64 == 0 and n % 16 == 0 and n > 0


def _fold():
    # the attention extension's balanced leaf tree (rows.cu ``leaf_tree_kernel``): the tree mm_sm100's
    # merge_tree defines, one ``__fadd_rn`` a node, one RNE of the root (the FP32 root when ``out`` is FP32)
    from isoexec.ops.attention.deepseek_v41_cache_sm100._extension import extension
    return extension().leaf_tree


class _Policy:
    """The NOSWAP block-size policy of the vendored heuristic for one launch (restored after it). Ownership /
    launch shape only."""

    def __init__(self, dg, noswap):
        self.dg, self.noswap = dg, noswap

    def __enter__(self):
        if self.noswap:
            self.dg.set_block_size_multiple_of(NOSWAP)

    def __exit__(self, *exc):
        if self.noswap:
            self.dg.set_block_size_multiple_of((1, 1))


def partials(x: torch.Tensor, w: torch.Tensor, segments: int, *, noswap: bool = True,
             out: torch.Tensor | None = None) -> torch.Tensor:
    """The ``[S, M, N]`` FP32 segment roots of ``x [M, K] @ w [N, K]^T`` (contiguous BF16 operands)."""
    from . import jit_warm
    from . import ops as dgo
    M, K = x.shape
    N = w.shape[0]
    jit_warm.note("seg_partials", n=int(N), k=int(K), segments=int(segments), noswap=bool(noswap))
    ks = K // segments
    if out is None:
        out = torch.empty((segments, M, N), dtype=torch.float32, device=x.device)
    if M == 0:
        return out
    a = x.view(M, segments, ks)                       # b, h, r
    b = w.view(N, segments, ks).permute(1, 0, 2)      # h, d, r  (strides Ks, K, 1)
    d = out.permute(1, 0, 2)                          # b, h, d  over the segment-major [S, M, N] buffer
    with _Policy(dgo._dg(), noswap):
        dgo._launch(lambda g: g.einsum("bhr,hdr->bhd", a, b, d), weight_b=True)
    return out


def linear(x: torch.Tensor, w: torch.Tensor, segments: int, *, out_dtype=torch.bfloat16,
           noswap: bool = True) -> torch.Tensor:
    """``x [M, K] @ w [N, K]^T`` by the S-segment tree: BF16 (one RNE of the root) or the FP32 root."""
    if x.dtype != torch.bfloat16 or w.dtype != torch.bfloat16 or x.dim() != 2 or w.dim() != 2:
        raise ValueError("segment-tree GEMM needs BF16 x [M, K] and weight [N, K]")
    if out_dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("segment-tree GEMM writes BF16 or the FP32 root")
    x = x.contiguous()
    w = w.contiguous()
    M, N = x.shape[0], w.shape[0]
    from . import jit_warm
    jit_warm.note("seg_linear", n=int(N), k=int(x.shape[1]), segments=int(segments),
                  out_dtype=str(out_dtype).split(".")[-1], noswap=bool(noswap))
    out = torch.empty((M, N), dtype=out_dtype, device=x.device)
    if M == 0:
        return out
    if segments == 1:
        # one chain: BF16 is its RNE, the FP32 root is the accumulator itself (the one-leaf tree is a copy), so the
        # FP32 root is written by the same launch (the leaf kernel at L = 1) instead of partials + a fold launch
        from . import ops as dgo
        if out_dtype is torch.bfloat16:
            with _Policy(dgo._dg(), noswap):
                dgo._launch(lambda g: g.bf16_gemm_nt(x, w, out), weight_b=True)
        else:
            dgo._launch(lambda g: g.bf16_gemm_nt_leaves(x, w, out, 1), weight_b=True)
        return out
    roots = partials(x, w, segments, noswap=noswap)
    _fold()(roots, out)
    return out


def leaf_linear(x: torch.Tensor, w: torch.Tensor, leaves: int, *, out_dtype=torch.bfloat16) -> torch.Tensor:
    """The same S-leaf tree in ONE launch with the tree on chip: DeepGEMM's BF16 GEMM with an IsoExec leaf
    epilogue (``bf16_gemm_nt_leaves``: each leaf one ascending chain from +0 in its own tensor-memory FP32
    accumulator, the leaves folded by the balanced adjacent-pair tree with ``__fadd_rn`` before the store, one RNE
    or the FP32 root) -- ``mm_sm100``'s whole-tree plan, the same bits as ``linear`` at ``segments = leaves``. For
    larger M, where the cut form's FP32 leaf traffic dominates (``wo_b`` above its decode rows)."""
    from . import jit_warm
    from . import ops as dgo
    x = x.contiguous()
    w = w.contiguous()
    jit_warm.note("seg_leaf", n=int(w.shape[0]), k=int(w.shape[1]), leaves=int(leaves),
                  out_dtype=str(out_dtype).split(".")[-1])
    out = torch.empty((x.shape[0], w.shape[0]), dtype=out_dtype, device=x.device)
    if x.shape[0]:
        dg = dgo._dg()
        regs = LEAF_REGS and leaves > 1
        if regs:
            dg.set_leaf_register_partials(True)
        try:
            dgo._launch(lambda g: g.bf16_gemm_nt_leaves(x, w, out, leaves), weight_b=True)
        finally:
            if regs:
                dg.set_leaf_register_partials(False)
    return out


__all__ = ["NOSWAP", "leaf_linear", "linear", "partials", "serves"]
