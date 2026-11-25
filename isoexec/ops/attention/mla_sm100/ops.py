"""SM100 absorbed MLA kernels and a forward-only packed causal adapter."""

from __future__ import annotations

from isoexec.ops.attention.mla_sm100.fwd import mla_sm100_fwd, mla_sm100_plan

__all__ = ["kernel_fwd", "kernel_plan", "mla_sm100_attention"]


def kernel_fwd(q, kv_cache, block_table, cu_seqlens_q, seqused_k, softmax_scale, seg_tokens,
               max_seqlen_q, **kw):
    """The deterministic absorbed-MLA forward: (out bf16 [T, h_q, 512], lse float32 [T, h_q])."""
    return mla_sm100_fwd(q, kv_cache, block_table, cu_seqlens_q, seqused_k, softmax_scale,
                         seg_tokens, max_seqlen_q, **kw)


def kernel_plan(cu_seqlens_q, seqused_k, seg_tokens, h_q, max_seqlen_q, **kw):
    """The per-step plan; reusable across layers and CUDA-graph capturable."""
    return mla_sm100_plan(cu_seqlens_q, seqused_k, seg_tokens, h_q, max_seqlen_q, **kw)


import math
from collections.abc import Sequence
import torch
# These architecture-neutral projection helpers retain main's int32-offset chunking guard.
from isoexec.ops.attention.mla_sm90.ops import _lengths, _absorb_fwd, pack_latent, _unabsorb_fwd

LATENT, ROPE = 512, 64


def mla_sm100_attention(q_nope, q_pe, c_kv, k_pe, w_uk, w_uv, *, seq_lens: Sequence[int],
                       scale: float, seg_tokens: int):
    """Packed causal absorbed MLA: ``[T, H, nope]`` queries over their own sequences -> ``[T, H, v]``."""
    if torch.is_grad_enabled() and any(t.requires_grad for t in (q_nope, q_pe, c_kv, k_pe, w_uk, w_uv)):
        raise RuntimeError("SM100 packed MLA is forward-only; no backward implementation is registered")
    if q_nope.ndim != 3:
        raise ValueError("q_nope must have shape [tokens, heads, nope]")
    tokens, heads, nope = q_nope.shape
    if min(tokens, heads, nope) <= 0:
        raise ValueError("MLA token, head, and projection dimensions must be positive")
    if c_kv.ndim != 2 or k_pe.ndim != 2:
        raise ValueError("c_kv and k_pe must have shape [tokens, width]")
    expected = ((tokens, heads, ROPE), (tokens, LATENT), (tokens, ROPE), (heads, nope, LATENT))
    if tuple(tuple(t.shape) for t in (q_pe, c_kv, k_pe, w_uk)) != expected:
        raise ValueError("inconsistent MLA token/head/projection dimensions (latent 512, rope 64)")
    if w_uv.ndim != 3 or w_uv.shape[0] != heads or w_uv.shape[-1] != LATENT:
        raise ValueError("w_uv must have shape [heads, value, 512]")
    if any(t.dtype != torch.bfloat16 for t in (q_nope, q_pe, c_kv, k_pe, w_uk, w_uv)):
        raise ValueError("mla_sm100 operands must be bf16 (kv_dtype pin)")
    if any(t.device != q_nope.device for t in (q_pe, c_kv, k_pe, w_uk, w_uv)) or q_nope.device.type != "cuda":
        raise ValueError("mla_sm100 operands must share one CUDA device")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("MLA softmax scale must be finite and positive")
    if type(seg_tokens) is not int or seg_tokens <= 0 or seg_tokens % 128:
        raise ValueError("seg_tokens must be a positive multiple of 128")
    lens = _lengths(seq_lens, tokens)
    lengths = torch.tensor(lens, device=q_nope.device, dtype=torch.int32)
    cu = torch.cat((lengths.new_zeros(1), lengths.cumsum(0, dtype=torch.int32)))
    q = _absorb_fwd(q_nope, w_uk, q_pe)
    cache, table = pack_latent(c_kv, k_pe, lens)
    out, _ = kernel_fwd(q, cache.unsqueeze(2), table, cu, lengths, float(scale), seg_tokens, max(lens))
    return _unabsorb_fwd(out, w_uv)
