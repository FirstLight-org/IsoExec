"""Autograd absorbed MLA: batch-invariant BMM query absorption, zero-padded per-sequence KV pages,
segmented SM90 attention with one bf16 root round, then batch-invariant BMM output projection.
Backward uses FA3 on decompressed per-head Q/K/V with the forward's y/lse and cuBLAS projection gradients;
only forward is bitwise deterministic."""

from __future__ import annotations

import math
import operator
from collections.abc import Sequence

import torch

from isoexec.ops.attention.mla_sm90.spec import PAGE

LATENT = 512
ROPE = 64


def _lengths(seq_lens: Sequence[int], total: int) -> tuple[int, ...]:
    if isinstance(seq_lens, torch.Tensor):
        raise TypeError("seq_lens must contain host integers, not a device tensor")
    try:
        lens = tuple(operator.index(n) for n in seq_lens)
    except TypeError as exc:
        raise TypeError("seq_lens must contain host integers") from exc
    if not lens or any(n <= 0 for n in lens) or sum(lens) != total:
        raise ValueError(f"positive sequence lengths must sum to {total}, got {lens}")
    return lens


def pack_latent(c_kv, k_pe, seq_lens: Sequence[int], page: int = 64):
    """Pack each sequence into its own page-aligned [latent | rotary] range.

    The returned block table cannot address another sequence through padding.
    Padding is initialized to zero: FA3 masks its probability but may still read
    its value, and zero times NaN would contaminate output. Engine hosts must
    preserve finite padding as well.
"""
    if type(page) is not int or page <= 0 or page % 16:
        raise ValueError("FA3 MLA page size must be a positive multiple of 16")
    if c_kv.ndim != 2 or k_pe.ndim != 2 or c_kv.shape[0] != k_pe.shape[0]:
        raise ValueError("c_kv and k_pe must have matching [tokens, width] shapes")
    if c_kv.dtype != k_pe.dtype or c_kv.device != k_pe.device:
        raise ValueError("latent and rotary cache must share dtype and device")
    lens = _lengths(seq_lens, c_kv.shape[0])
    counts = [(n + page - 1) // page for n in lens]
    latent = torch.cat((c_kv, k_pe), dim=-1)
    cache = latent.new_zeros(sum(counts) * page, latent.shape[-1])
    table = []
    source, block = 0, 0
    for n, count in zip(lens, counts):
        cache[block * page : block * page + n] = latent[source : source + n]
        table.append(list(range(block, block + count)) + [0] * (max(counts) - count))
        source += n
        block += count
    return cache.view(sum(counts), page, -1), torch.tensor(table, device=latent.device, dtype=torch.int32)


_BMM_MAX_NUMEL = 2**31 - 1


def _bmm(a, b):
    """``bmm_batch_invariant`` over head chunks small enough for its int32 batch offsets.

    The vendored Triton ``bmm_kernel`` computes ``pid_b * stride_*b`` in int32 (its ``*_LARGE`` flags widen
    only the row/column/K offsets), so an operand or output above 2**31 elements addresses out of bounds.
    Every batch entry (head) is computed independently of the others, so chunking over heads is bit-identical.
    """
    from isoexec.ops.mm.mm_batch_invariant import bmm_batch_invariant

    a, b = a.contiguous(), b.contiguous()
    H, M, K = a.shape
    per_head = max(M * K, b.shape[1] * b.shape[2], M * b.shape[2])
    if per_head > _BMM_MAX_NUMEL:
        raise ValueError(f"one head of bmm_batch_invariant exceeds 2**31 elements: {tuple(a.shape)} x {tuple(b.shape)}")
    step = max(1, _BMM_MAX_NUMEL // per_head)
    if step >= H:
        return bmm_batch_invariant(a, b)
    out = torch.empty(H, M, b.shape[2], device=a.device, dtype=a.dtype)
    for h0 in range(0, H, step):
        bmm_batch_invariant(a[h0:h0 + step], b[h0:h0 + step], out=out[h0:h0 + step])
    return out


def _absorb_fwd(q_nope, w_uk, q_pe=None):
    """``[T,H,nope] x [H,nope,latent] -> [T,H,latent]``: bits of ``ops/attention/mla_absorbed.absorb_query``.

    With ``q_pe`` returns the kernel query ``[W_uk^T q_nope | q_pe]`` (``[T,H,576]``, contiguous): the Triton
    kernel is stride-generic, so it reads ``q_nope`` through its transposed view and writes the latent columns of
    the preallocated query through a transposed view (no transpose copies, no cat). Same kernel, same operand
    values, same tiles: the same bits (``tests``: projections match the absorbed primitive).
    """
    T, H, _ = q_nope.shape
    latent = w_uk.shape[2]
    width = latent + (0 if q_pe is None else q_pe.shape[2])
    q = torch.empty(T, H, width, device=q_nope.device, dtype=q_nope.dtype)
    if max(q.numel(), q_nope.numel()) > _BMM_MAX_NUMEL:
        q[..., :latent] = _bmm(q_nope.transpose(0, 1), w_uk).transpose(0, 1)
    else:
        from isoexec.ops.mm.mm_batch_invariant import bmm_batch_invariant

        bmm_batch_invariant(q_nope.transpose(0, 1), w_uk, out=q[..., :latent].transpose(0, 1))
    if q_pe is not None:
        q[..., latent:] = q_pe
    return q


def _unabsorb_fwd(o_latent, w_uv, out=None):
    """``[T,H,latent] x [H,value,latent] -> [T,H,value]``: bits of ``ops/attention/mla_absorbed.unabsorb_output``.

    Reads ``o_latent`` and ``W_uv^T`` through transposed views and writes ``[T,H,value]`` (``out`` if given)
    through a transposed view: no copies around the kernel.
    """
    T, H, _ = o_latent.shape
    y = out if out is not None else torch.empty(T, H, w_uv.shape[1], device=o_latent.device, dtype=o_latent.dtype)
    if max(y.numel(), o_latent.numel()) > _BMM_MAX_NUMEL:
        y.copy_(_bmm(o_latent.transpose(0, 1), w_uv.transpose(1, 2)).transpose(0, 1))
    else:
        from isoexec.ops.mm.mm_batch_invariant import bmm_batch_invariant

        bmm_batch_invariant(o_latent.transpose(0, 1), w_uv.transpose(1, 2), out=y.transpose(0, 1))
    return y


def kernel_fwd(q, kv_cache, block_table, cu_seqlens_q, seqused_k, softmax_scale, seg_tokens, max_seqlen_q, **kw):
    """Call ``fwd.mla_sm90_fwd`` with the role-free contract."""
    from isoexec.ops.attention.mla_sm90.fwd import mla_sm90_fwd

    return mla_sm90_fwd(q, kv_cache, block_table, cu_seqlens_q, seqused_k, softmax_scale, seg_tokens, max_seqlen_q,
                        **kw)


def kernel_plan(cu_seqlens_q, seqused_k, seg_tokens, h_q, max_seqlen_q, **kw):
    """Call ``fwd.mla_sm90_plan`` with the role-free contract."""
    from isoexec.ops.attention.mla_sm90.fwd import mla_sm90_plan

    return mla_sm90_plan(cu_seqlens_q, seqused_k, seg_tokens, h_q, max_seqlen_q, **kw)


class _MLASM90Op(torch.autograd.Function):
    """Absorbed MLA autograd node with pinned forward and numerical FA3/projection backward."""

    @staticmethod
    def forward(ctx, q_nope, q_pe, c_kv, k_pe, w_uk, w_uv, lens, cu, lengths, scale, seg_tokens):
        q = _absorb_fwd(q_nope, w_uk, q_pe)
        cache, table = pack_latent(c_kv, k_pe, lens, PAGE)
        out, lse = kernel_fwd(q, cache.unsqueeze(2), table, cu, lengths, scale, seg_tokens, max(lens))
        y = _unabsorb_fwd(out, w_uv)
        ctx.save_for_backward(q_nope, q_pe, c_kv, k_pe, w_uk, w_uv, y, lse, cu)
        ctx.scale, ctx.max_seqlen = scale, max(lens)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        from isoexec.ops.attention.mla_sm90.bwd import mla_sm90_attention_grads

        q_nope, q_pe, c_kv, k_pe, w_uk, w_uv, y, lse, cu = ctx.saved_tensors
        grads = mla_sm90_attention_grads(grad_y.to(y.dtype), q_nope, q_pe, c_kv, k_pe, w_uk, w_uv, y, lse, cu,
                              ctx.max_seqlen, ctx.scale)
        leaves = (q_nope, q_pe, c_kv, k_pe, w_uk, w_uv)
        out = tuple(g.to(t.dtype) if need else None
                    for g, t, need in zip(grads, leaves, ctx.needs_input_grad[:6]))
        return (*out, None, None, None, None, None)


def mla_sm90_attention(q_nope, q_pe, c_kv, k_pe, w_uk, w_uv, *, seq_lens: Sequence[int],
                       scale: float, seg_tokens: int):
    """Packed causal absorbed MLA: ``[T, H, nope]`` queries over their own sequences -> ``[T, H, v]``.

    ``seq_lens`` are host integers, one per packed sequence. ``seg_tokens`` is the deployment-fixed segment
    size (a positive multiple of 128); it is numerical for every row whose border exceeds it, and must equal
    the engine's value (compared by digest) for trainer rows to match engine rows bitwise.
    """
    if q_nope.ndim != 3:
        raise ValueError("q_nope must have shape [tokens, heads, nope]")
    tokens, heads, nope = q_nope.shape
    if c_kv.ndim != 2 or k_pe.ndim != 2:
        raise ValueError("c_kv and k_pe must have shape [tokens, width]")
    expected = ((tokens, heads, ROPE), (tokens, LATENT), (tokens, ROPE), (heads, nope, LATENT))
    if tuple(tuple(t.shape) for t in (q_pe, c_kv, k_pe, w_uk)) != expected:
        raise ValueError("inconsistent MLA token/head/projection dimensions (latent 512, rope 64)")
    if w_uv.ndim != 3 or w_uv.shape[0] != heads or w_uv.shape[-1] != LATENT:
        raise ValueError("w_uv must have shape [heads, value, 512]")
    if any(t.dtype != torch.bfloat16 for t in (q_nope, q_pe, c_kv, k_pe, w_uk, w_uv)):
        raise ValueError("mla_sm90 operands must be bf16 (kv_dtype pin)")
    if any(t.device != q_nope.device for t in (q_pe, c_kv, k_pe, w_uk, w_uv)) or q_nope.device.type != "cuda":
        raise ValueError("mla_sm90 operands must share one CUDA device")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("MLA softmax scale must be finite and positive")
    if type(seg_tokens) is not int or seg_tokens <= 0 or seg_tokens % 128:
        raise ValueError("seg_tokens must be a positive multiple of 128")
    lens = _lengths(seq_lens, tokens)
    device = q_nope.device
    lengths = torch.tensor(lens, device=device, dtype=torch.int32)
    cu = torch.cat((lengths.new_zeros(1), lengths.cumsum(0, dtype=torch.int32)))

    return _MLASM90Op.apply(q_nope, q_pe, c_kv, k_pe, w_uk, w_uv, lens, cu, lengths, float(scale), seg_tokens)
