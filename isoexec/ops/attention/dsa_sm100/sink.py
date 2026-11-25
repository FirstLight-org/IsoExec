"""Explicit learned-sink DSA training path, ``sm100_dsa_sparse_sink_v1``.

The frozen attention leaf exports its raw FP32 numerator/denominator. A new root
adds exp(sink[h]) to the denominator before the sole BF16 output rounding.
The returned natural-log normalizer includes the sink and is not differentiable.
Backward uses ``sm100_dsa_sparse_sink_bwd_v2``: the existing no-atomic
attention backward recomputes Delta from FP32 scores and dP rather than the
BF16-rounded output, avoiding cancellation error for nearly one-hot rows.
The forward leaf and its accumulator recipe are unchanged;
its new sink gradient owns each head and visits tokens in ascending order.

This is an experimental native path, not an op-table promotion. Lists must be
strictly ascending with trailing -1 pads. Inputs and sink logits must be finite.
"""
from __future__ import annotations

import torch

from ._extension import extension
from .bwd import dsa_sparse_attn_bwd
from .fwd import check_ascending
from .plan import groups_for

__all__ = ["sparse_attention_sink", "sparse_attention_sink_forward"]

BACKWARD_RECIPE = "sm100_dsa_sparse_sink_bwd_v2"


def _validate(q, kv, lists, sink):
    if q.ndim != 3 or q.shape[0] == 0 or q.shape[2] not in (512, 576) or not 1 <= q.shape[1] <= 128:
        raise ValueError("q: [T>0, H<=128, D=512|576]")
    if kv.ndim != 2 or kv.shape[1] != q.shape[2] or kv.shape[0] == 0:
        raise ValueError("kv: nonempty [S,D] matching q")
    if q.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16:
        raise ValueError("q and kv must be BF16")
    if lists.ndim != 2 or lists.shape[0] != q.shape[0] or lists.dtype != torch.int32:
        raise ValueError("lists: int32 [T,K]")
    if sink.shape != (q.shape[1],) or sink.dtype != torch.float32:
        raise ValueError("sink: fp32 [H]")
    if not q.is_cuda or any(x.device != q.device for x in (kv, lists, sink)):
        raise ValueError("all inputs must share a CUDA device")


def sparse_attention_sink_forward(q, kv, lists, sink, softmax_scale, *, recipe=1, check=False):
    """Sink-inclusive output/LSE; inference-only primitive (use wrapper for training)."""
    _validate(q, kv, lists, sink)
    if recipe not in (1, 2):
        raise ValueError("recipe must be 1 or 2")
    if check:
        check_ascending(lists)
        if bool((lists >= kv.shape[0]).any()):
            raise ValueError("list entry outside KV storage")
        if not bool(torch.isfinite(sink).all()):
            raise ValueError("sink logits must be finite")
    t, h, _ = q.shape
    k = lists.shape[1]
    groups = groups_for(128 if h > 64 else 64, k)
    out = torch.empty((t, h, 512), dtype=torch.bfloat16, device=q.device)
    lse = torch.empty((t, h), dtype=torch.float32, device=q.device)
    partials = torch.empty((t, groups, h, 520), dtype=torch.float32, device=q.device)
    dummy = torch.empty((1, 1), dtype=torch.int32, device=q.device)
    with torch.cuda.device(q.device):
        extension("attn").sparse_sink_fwd(q.contiguous(), kv.contiguous(), dummy, None,
                                          lists.contiguous(), None, float(softmax_scale), k // groups,
                                          out, lse, partials, recipe, sink.contiguous())
    return out, lse


class _SparseAttentionSink(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, kv, lists, sink, softmax_scale, recipe):
        q, kv, lists, sink = (x.contiguous() for x in (q, kv, lists, sink))
        out, lse = sparse_attention_sink_forward(q, kv, lists, sink, softmax_scale, recipe=recipe)
        ctx.save_for_backward(q, kv, lists, sink, out, lse)
        ctx.scale = softmax_scale
        ctx.mark_non_differentiable(lse)
        return out, lse

    @staticmethod
    def backward(ctx, d_out, _d_lse):
        q, kv, lists, sink, out, lse = ctx.saved_tensors
        d_out = d_out.contiguous().to(torch.bfloat16)
        with torch.cuda.device(q.device):
            dq, dkv = dsa_sparse_attn_bwd(q, kv, lists, out, lse, d_out, ctx.scale,
                                          recompute_delta=True)
            dsink = torch.empty_like(sink)
            extension("attn").sink_grad(out, d_out, lse, sink, dsink)
        return dq, dkv, None, dsink, None, None


def sparse_attention_sink(q, kv, lists, sink, softmax_scale, *, recipe=1):
    """Differentiable native sink attention; grads to Q, shared KV, and FP32 sink."""
    return _SparseAttentionSink.apply(q, kv, lists, sink, softmax_scale, recipe)
