"""Runtime-free absorbed MLA from SkyRL-ZeroKL commit 628b322d3ab88a0f9e833e322907d4af3f3a7d67,
skyrl/backends/skyrl_train/zerokl/ops/mla/mla_core_absorbed.py.
Forward uses owned batch-invariant BMM and host-supplied FA3 with num_splits=1; backward differentiates
the unabsorbed expression and requires numerical, not bitwise, agreement."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import math
import operator

import torch

# The absorbed forward is the same FA3 single-split kernel the GQA core runs (gqa_sm90).
from isoexec.ops.attention.spec import NUM_SPLITS


def split_kv_up_weight(weight, *, num_heads: int, qk_nope_dim: int, v_head_dim: int, kv_lora_rank: int):
    """Return live [H, nope, latent] and [H, value, latent] parameter views.

HF kv_b_proj and Megatron linear_kv_up_proj use this per-head interleaving.
Views reflect in-place weight synchronization; callers must rebuild them when
the owning parameter/storage is replaced.
"""
    dims = (num_heads, qk_nope_dim, v_head_dim, kv_lora_rank)
    if any(type(n) is not int or n <= 0 for n in dims):
        raise ValueError("MLA dimensions must be positive integers")
    expected = (num_heads * (qk_nope_dim + v_head_dim), kv_lora_rank)
    if tuple(weight.shape) != expected:
        raise ValueError(f"kv-up weight shape {tuple(weight.shape)} != {expected}")
    stack = weight.view(num_heads, qk_nope_dim + v_head_dim, kv_lora_rank)
    return stack[:, :qk_nope_dim, :], stack[:, qk_nope_dim:, :]


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


class _PinnedBMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        from isoexec.ops.mm.mm_batch_invariant import bmm_batch_invariant

        ctx.save_for_backward(a, b)
        return bmm_batch_invariant(a, b)

    @staticmethod
    def backward(ctx, grad):
        a, b = ctx.saved_tensors
        return torch.ops.aten.bmm(grad, b.transpose(1, 2)), torch.ops.aten.bmm(a.transpose(1, 2), grad)


def absorb_query(q_nope, w_uk):
    """[T,H,nope] x [H,nope,latent] -> [T,H,latent], fixed K reduction."""
    return _PinnedBMM.apply(q_nope.transpose(0, 1), w_uk).transpose(0, 1).contiguous()


def unabsorb_output(o_latent, w_uv):
    """[T,H,latent] x [H,value,latent] -> [T,H,value], fixed K reduction."""
    return _PinnedBMM.apply(o_latent.transpose(0, 1), w_uv.transpose(1, 2)).transpose(0, 1).contiguous()


def absorbed_kernel(q_latent, q_pe, cache, block_table, cu_seqlens_q, seqused_k,
                    max_seqlen_q: int, max_seqlen_k: int, *, scale: float,
                    kv_lora_rank: int, flash_attn: Callable):
    """The shared trainer/engine FA3 call. The host owns the supplied entry point.

Queries are aligned to the right end of each sequence's valid cached prefix;
this is FA3's causal convention for decode and chunked prefill.
"""
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("MLA softmax scale must be finite and positive")
    return flash_attn(
        q=q_pe,
        k=cache[..., kv_lora_rank:].unsqueeze(-2),
        v=cache[..., :kv_lora_rank].unsqueeze(-2),
        q_v=q_latent,
        max_seqlen_q=max_seqlen_q,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_k=max_seqlen_k,
        seqused_k=seqused_k,
        block_table=block_table,
        softmax_scale=scale,
        causal=True,
        window_size=[-1, 0],
        num_splits=NUM_SPLITS,
    )


def _unabsorbed(q_nope, q_pe, c_kv, k_pe, w_uk, w_uv, cu, max_seq, scale, varlen_attn):
    heads = q_nope.shape[1]
    latent = c_kv.unsqueeze(0).expand(heads, -1, -1)
    # The installed batch-invariance runtime may rebind the Python torch.bmm
    # symbol to a forward-only kernel. Calling aten preserves its autograd node.
    k_nope = torch.ops.aten.bmm(latent, w_uk.transpose(1, 2)).transpose(0, 1)
    values = torch.ops.aten.bmm(latent, w_uv.transpose(1, 2)).transpose(0, 1).contiguous()
    queries = torch.cat((q_nope, q_pe), dim=-1).contiguous()
    keys = torch.cat((k_nope, k_pe.unsqueeze(1).expand(-1, heads, -1)), dim=-1).contiguous()
    return varlen_attn(queries, keys, values, cu, cu, max_seq, max_seq, scale)


class _AbsorbedMLA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q_nope, q_pe, c_kv, k_pe, w_uk, w_uv, lens, scale, page, flash_attn, varlen_attn):
        lengths = torch.tensor(lens, device=q_nope.device, dtype=torch.int32)
        cu = torch.cat((lengths.new_zeros(1), lengths.cumsum(0, dtype=torch.int32)))
        cache, table = pack_latent(c_kv, k_pe, lens, page)
        latent_out = absorbed_kernel(
            absorb_query(q_nope, w_uk), q_pe, cache, table, cu, lengths,
            max(lens), max(lens), scale=scale, kv_lora_rank=c_kv.shape[-1], flash_attn=flash_attn)
        ctx.save_for_backward(q_nope, q_pe, c_kv, k_pe, w_uk, w_uv, cu)
        ctx.max_seq, ctx.scale, ctx.varlen_attn = max(lens), scale, varlen_attn
        return unabsorb_output(latent_out, w_uv)

    @staticmethod
    def backward(ctx, grad):
        *saved, cu = ctx.saved_tensors
        leaves = [t.detach().requires_grad_(True) for t in saved]
        with torch.enable_grad():
            out = _unabsorbed(*leaves, cu, ctx.max_seq, ctx.scale, ctx.varlen_attn)
            grads = torch.autograd.grad(out, leaves, grad.contiguous())
        return (*grads, None, None, None, None, None)


def absorbed_mla(q_nope, q_pe, c_kv, k_pe, w_uk, w_uv, *, seq_lens: Sequence[int],
                 scale: float, flash_attn: Callable, varlen_attn: Callable, page: int = 64):
    """Packed trainer forward with the original unabsorbed-gradient contract.

``flash_attn`` is the pinned FA3 paged entry point (gqa_sm90.flash_attn_paged);
``varlen_attn`` is the gradient-capable varlen binding (gqa_sm90.varlen_attention). Neither provider is auto-discovered
or silently substituted. Runtime admission must qualify both supplied bindings.
"""
    if q_nope.ndim != 3:
        raise ValueError("q_nope must have shape [tokens, heads, nope]")
    tokens, heads, nope = q_nope.shape
    if c_kv.ndim != 2 or k_pe.ndim != 2:
        raise ValueError("c_kv and k_pe must have shape [tokens, width]")
    latent, rope = c_kv.shape[-1], k_pe.shape[-1]
    expected = ((tokens, heads, rope), (tokens, latent), (tokens, rope),
                (heads, nope, latent))
    if tuple(t.shape for t in (q_pe, c_kv, k_pe, w_uk)) != expected:
        raise ValueError("inconsistent MLA token/head/projection dimensions")
    if w_uv.ndim != 3 or w_uv.shape[0] != heads or w_uv.shape[-1] != latent:
        raise ValueError("w_uv must have shape [heads, value, latent]")
    if any(t.device != q_nope.device or t.dtype != q_nope.dtype for t in (q_pe, c_kv, k_pe, w_uk, w_uv)):
        raise ValueError("MLA operands must share dtype and device")
    if not callable(flash_attn) or not callable(varlen_attn):
        raise TypeError("MLA requires explicit forward and backward attention bindings")
    lens = _lengths(seq_lens, tokens)
    return _AbsorbedMLA.apply(q_nope, q_pe, c_kv, k_pe, w_uk, w_uv,
                              lens, scale, page, flash_attn, varlen_attn)
