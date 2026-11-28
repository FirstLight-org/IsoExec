"""Python entry points for the SM100 DeepSeek-V4.1 shifted mHC (parameters, collapse, expand; forward and backward).

All kernels live in ``mhc_sm100/csrc`` (one extension, one numerical program per op); this module adds the DeepSeek
boundaries: fp32 ``fn``/``scale``/``base`` checkpoint parameters, rms epsilon 1e-20 and Sinkhorn epsilon 1e-6 as call facts,
the *shifted* collapse (``pre`` supplied by the caller from the previous sublayer) and the all-fp32 expand.
"""

from __future__ import annotations

import os

import torch

from isoexec.ops.residual.deepseek_mhc_sm100.spec import HIDDEN, IMPL_ID  # noqa: F401
from isoexec.ops.residual.mhc_sm100 import ops as _g
from isoexec.ops.residual.mhc_sm100._extension import extension


def _check(x, weight, scale, base):
    if x.shape[-1] != HIDDEN:
        raise ValueError(f"DeepSeek mHC serves H = {HIDDEN}")
    if scale.dtype != torch.float32 or base.dtype != torch.float32:
        raise ValueError("DeepSeek mHC scale/base are fp32 checkpoint parameters")


# ---------------------------------------------------------------------------------------------------- parameters
class _Params(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, scale, base, rms_eps, eps, iters):
        pre, post, comb, _, (proj, rstd) = _g.split_forward(x, weight, scale, base, rms_eps=rms_eps, eps=eps,
                                                            iters=iters, collapse=None, stats=True)
        ctx.save_for_backward(x, weight, scale, base, proj, rstd)
        ctx.cfg = (eps, iters)
        return pre, post, comb

    @staticmethod
    def backward(ctx, dpre, dpost, dcomb):
        x, weight, scale, base, proj, rstd = ctx.saved_tensors
        eps, iters = ctx.cfg
        dx, dw, dscale, dbase, _ = _g.split_backward(x, weight, scale, base, proj, rstd, dpost, dcomb, eps=eps,
                                                     iters=iters, dpre=dpre)
        return dx, dw.to(weight.dtype), dscale, dbase, None, None, None


def mhc_params(x, weight, scale, base, *, rms_eps, sinkhorn_eps, iterations):
    """``(pre, post, comb)`` for streams ``x [..., 4, H]`` (bf16): the V4.1 coefficients, differentiable."""
    _check(x, weight, scale, base)
    if torch.is_grad_enabled() and any(t.requires_grad for t in (x, weight, scale, base)):
        return _Params.apply(x, weight, scale, base, float(rms_eps), float(sinkhorn_eps), int(iterations))
    pre, post, comb, _ = _g.split_forward(x, weight, scale, base, rms_eps=rms_eps, eps=sinkhorn_eps,
                                          iters=iterations, collapse=None)
    return pre, post, comb


class _SplitShifted(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, scale, base, shifted_pre, rms_eps, eps, iters):
        pre, post, comb, col, (proj, rstd) = _g.split_forward(x, weight, scale, base, rms_eps=rms_eps, eps=eps,
                                                              iters=iters, collapse="shifted", shifted_pre=shifted_pre,
                                                              stats=True)
        ctx.save_for_backward(x, weight, scale, base, proj, rstd, shifted_pre)
        ctx.cfg = (eps, iters)
        return pre, post, comb, col

    @staticmethod
    def backward(ctx, dpre, dpost, dcomb, dcol):
        x, weight, scale, base, proj, rstd, shifted_pre = ctx.saved_tensors
        eps, iters = ctx.cfg
        dx, dw, dscale, dbase, dsp = _g.split_backward(x, weight, scale, base, proj, rstd, dpost, dcomb, eps=eps,
                                                       iters=iters, dpre=dpre, dcol=dcol, shifted_pre=shifted_pre)
        return dx, dw.to(weight.dtype), dscale, dbase, dsp, None, None, None


def mhc_split_shifted(x, weight, scale, base, shifted_pre, *, rms_eps, sinkhorn_eps, iterations):
    """Fused: ``(pre, post, comb, collapse(x, shifted_pre))`` in one pass over the streams (the collapse uses the
    caller's shifted ``pre``; ``pre`` returned here feeds the NEXT collapse)."""
    _check(x, weight, scale, base)
    if torch.is_grad_enabled() and any(t.requires_grad for t in (x, weight, scale, base, shifted_pre)):
        return _SplitShifted.apply(x, weight, scale, base, shifted_pre, float(rms_eps), float(sinkhorn_eps), int(iterations))
    pre, post, comb, col = _g.split_forward(x, weight, scale, base, rms_eps=rms_eps, eps=sinkhorn_eps, iters=iterations,
                                            collapse="shifted", shifted_pre=shifted_pre)
    return pre, post, comb, col


# ---------------------------------------------------------------------------------------------------- collapse
def _flat(x, pre):
    if x.dim() < 3 or x.shape[-2] != 4 or x.dtype != torch.bfloat16 or not x.is_cuda or x.shape[-1] % 8:
        raise ValueError("collapse expects CUDA bf16 streams [..., 4, H]")
    if pre.dtype != torch.float32 or pre.shape != (*x.shape[:-2], 4):
        raise ValueError("pre must be fp32 [..., 4]")
    H = x.shape[-1]
    return x.contiguous().view(-1, 4, H), pre.contiguous().view(-1, 4), H


def collapse_forward(x, pre):
    xf, pf, H = _flat(x, pre)
    return extension().collapse_forward(xf, pf).view(*x.shape[:-2], H)


def collapse_backward(x, pre, dcol):
    xf, pf, H = _flat(x, pre)
    g = dcol.contiguous().view(-1, H)
    dx = extension().collapse_backward_dx(g, pf).view(x.shape)
    dpre = extension().dcolx(xf, g).view(pre.shape)
    return dx, dpre


class _Collapse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, pre):
        ctx.save_for_backward(x, pre)
        return collapse_forward(x, pre)

    @staticmethod
    def backward(ctx, dcol):
        return collapse_backward(*ctx.saved_tensors, dcol)


def collapse(x, pre):
    """``bf16(sum_s pre[..., s] * x[..., s, :])`` (fp32 products and sums): the shifted pre-mix collapse."""
    if torch.is_grad_enabled() and (x.requires_grad or pre.requires_grad):
        return _Collapse.apply(x, pre)
    return collapse_forward(x, pre)


# ---------------------------------------------------------------------------------------------------- expand
class _Expand(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, residual, post, comb):
        ctx.save_for_backward(x, residual, post, comb)
        return _g.merge_forward(residual, x, post, comb, cast=False)

    @staticmethod
    def backward(ctx, grad):
        x, residual, post, comb = ctx.saved_tensors
        dres, dx, dpost, dcomb = _g.merge_backward(grad, residual, x, post, comb, cast=False)
        return dx, dres, dpost, dcomb


def expand(x, residual, post, comb):
    """``bf16(post[..., :, None] * x[..., None, :] + sum_s comb[..., s, :, None] * residual[..., s, None, :])``, fp32."""
    if residual.shape[-2] != 4 or x.shape != (*residual.shape[:-2], residual.shape[-1]):
        raise ValueError("expand expects x [..., H] and residual [..., 4, H]")
    if torch.is_grad_enabled() and any(t.requires_grad for t in (x, residual, post, comb)):
        return _Expand.apply(x, residual, post, comb)
    return _g.merge_forward(residual, x, post, comb, cast=False)


# ---------------------------------------------------------------------------------------------------- boundary (inference)
def boundary_rows(streams, shifted_pre, norm_weight, eps, *, out=None, post=None, comb=None, encode=False, cpt=0):
    """Inference row part of one mHC boundary in ONE launch (``mhc_sm100/csrc/deepseek_rows.cu``):
    ``new = expand(out, streams, post, comb)`` (or ``streams`` itself when ``out`` is None) and
    ``h = rms_norm(collapse(new, shifted_pre))`` (norms_sm100 recipe v2 at H = 5120, bf16 weight), plus the
    checkpoint FP8 encoding of ``h`` registered for its FP8 consumer when ``encode`` (its scales also in
    DeepGEMM's packed layout, byte for byte deepgemm_sm100 ``layout.pack_activation_scales``, so the DeepGEMM
    projections of ``h`` need no packing launch). Bit for bit
    ``expand`` followed by ``paged.collapse_rms_norm``. Returns ``(new, h)`` or None (caller falls back).
    The new streams' coefficients are NOT computed here (``mhc_params(new, ...)``): the collapse uses the
    caller's shifted ``pre``, so the caller may compute them concurrently."""
    import contextlib
    lead = streams.shape[:-2]
    if (streams.dim() < 3 or streams.shape[-2:] != (4, HIDDEN) or streams.dtype != torch.bfloat16 or not streams.is_cuda
            or norm_weight.dtype != torch.bfloat16 or norm_weight.numel() != HIDDEN
            or shifted_pre.dtype != torch.float32 or shifted_pre.shape != (*lead, 4)
            or (out is not None and (out.shape != (*lead, HIDDEN) or out.dtype != torch.bfloat16
                                     or post is None or comb is None or post.dtype != torch.float32
                                     or comb.dtype != torch.float32 or post.shape != (*lead, 4)
                                     or comb.shape != (*lead, 4, 4)))
            or (torch.is_grad_enabled() and any(t is not None and t.requires_grad
                                                for t in (streams, shifted_pre, norm_weight, out, post, comb)))):
        return None
    from isoexec.ops.attention.deepseek_v41_cache_sm100.paged import register_encoding
    x = streams.contiguous().view(-1, 4, HIDDEN)
    rows = x.shape[0]
    new = None
    if out is not None:
        new = torch.empty_like(x)
        out, post, comb = (out.contiguous().view(rows, HIDDEN), post.contiguous().view(rows, 4),
                           comb.contiguous().view(rows, 16))
    # as paged._norm_launch: ``h`` carries a version counter even under inference_mode, so the registered
    # encoding is validated against it (an in-place write before the consumer invalidates the encoding)
    with (torch.inference_mode(False) if encode and torch.is_inference_mode_enabled() else contextlib.nullcontext()):
        h = torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device=x.device)
    codes = scales = None
    if encode:
        codes = torch.empty((rows, HIDDEN), dtype=torch.float8_e4m3fn, device=x.device)
        scales = torch.empty((rows, HIDDEN // 32), dtype=torch.float8_e8m0fnu, device=x.device)
    packed = None
    if encode:
        packed = torch.empty((HIDDEN // 128, (rows + 3) // 4 * 4), dtype=torch.int32, device=x.device)
        extension().deepseek_rows_packed(x, out, post, comb, shifted_pre.contiguous().view(rows, 4),
                                         norm_weight.contiguous(), float(eps), new, h, codes, scales, int(cpt), packed)
    else:
        extension().deepseek_rows(x, out, post, comb, shifted_pre.contiguous().view(rows, 4), norm_weight.contiguous(),
                                  float(eps), new, h, codes, scales, int(cpt))
    if encode:
        register_encoding(h, codes, scales, packed.t()[:rows])
    new = streams if new is None else new.view(streams.shape)
    return new, h.view(*lead, HIDDEN)


#: Decode batches up to this many rows compute the mHC coefficients on a side stream (overlapping the sublayer
#: that consumes ``h``); larger batches keep them in order on the caller's stream. Launch structure only: the
#: coefficients are the same kernels on the same inputs either way. ``ISOEXEC_MHC_OVERLAP_MAX_ROWS`` overrides
#: (0 disables the overlap).
OVERLAP_MAX_ROWS = int(os.environ.get("ISOEXEC_MHC_OVERLAP_MAX_ROWS", "64"))
#: Which sublayers the side-stream coefficients may overlap. Launch structure only: the same kernels on the
#: same inputs either way.
OVERLAP_SUBLAYERS = frozenset({"attn", "ffn"})
#: Above OVERLAP_MAX_ROWS and up to this many rows only the Sinkhorn tail goes to the side stream (the projection stays
#: in order; the tail is a light T/8-CTA latency chain): every step size up to the batched-token budget (measured 10 to
#: 13 us less per boundary at M = 256..2048). 0 = off. Launch structure only: the same two kernels on the same inputs.
OVERLAP_TAIL_MAX_ROWS = 2048
#: Coefficients that overlap the ATTENTION sublayer (``sublayer="attn"``) go wholly to the side stream up to this many
#: rows (attention holds no persistent all-SM kernel); the FFN sublayer (router + MegaMoE, which needs every SM) keeps
#: the projection in order above ``OVERLAP_MAX_ROWS`` (tail-only overlap). Launch structure only.
OVERLAP_ATTN_MAX_ROWS = 2048
_SIDE: dict = {}


def _side_streams_on() -> frozenset:
    """``ISOEXEC_V41_SIDE_STREAMS``: which V4.1 overlap streams run (``mhc``, ``aux``, ``side``; ``all`` (default) or
    ``none``). Placement of work only: the same kernels on the same operands, no bit changes."""
    v = os.environ.get("ISOEXEC_V41_SIDE_STREAMS", "all").strip().lower()
    if v == "all":
        return frozenset({"mhc", "aux", "side"})
    return frozenset() if v in ("", "none") else frozenset(t.strip() for t in v.split(","))


def _side_stream(device):
    key = device.index if device.index is not None else torch.cuda.current_device()
    stream = _SIDE.get(key)
    if stream is None:
        stream = _SIDE[key] = torch.cuda.Stream(device)
    return stream


def mhc_params_async(x, weight, scale, base, *, rms_eps, sinkhorn_eps, iterations, overlap=None, sublayer=None):
    """Inference ``mhc_params`` that may run on a side stream: ``(pre, post, comb, event)``. When ``event`` is
    not None the coefficients are produced on the side stream and a consumer must ``wait_event(event)`` on its
    own stream first (``join``); the side stream waited for every prior op of the caller's stream, and the
    tensors are recorded on the caller's stream (the allocator never recycles them under a pending reader).
    ``overlap``: None = automatic (CUDA, no autograd, at most ``OVERLAP_MAX_ROWS`` rows, ``sublayer`` -- the
    sublayer the coefficients would overlap -- None or in ``OVERLAP_SUBLAYERS``)."""
    rows = x.numel() // (4 * HIDDEN) if x.shape[-1] == HIDDEN else 0
    if "mhc" not in _side_streams_on():     # the mature single-stream order (same kernels, same bits)
        return (*mhc_params(x, weight, scale, base, rms_eps=rms_eps, sinkhorn_eps=sinkhorn_eps,
                            iterations=iterations), None)
    if overlap is None:
        limit = max(OVERLAP_MAX_ROWS, OVERLAP_ATTN_MAX_ROWS) if sublayer == "attn" else OVERLAP_MAX_ROWS
        overlap = (x.is_cuda and not torch.is_grad_enabled() and 0 < rows <= limit
                   and (sublayer is None or sublayer in OVERLAP_SUBLAYERS))
    if not overlap:
        if (0 < rows <= OVERLAP_TAIL_MAX_ROWS and x.is_cuda and not torch.is_grad_enabled()
                and weight.dtype == torch.float32 and (sublayer is None or sublayer in OVERLAP_SUBLAYERS)):
            return _params_tail_overlapped(x, weight, scale, base, rms_eps, sinkhorn_eps, iterations)
        return (*mhc_params(x, weight, scale, base, rms_eps=rms_eps, sinkhorn_eps=sinkhorn_eps,
                            iterations=iterations), None)
    main = torch.cuda.current_stream(x.device)
    side = _side_stream(x.device)
    side.wait_stream(main)
    with torch.cuda.stream(side):
        pre, post, comb = mhc_params(x, weight, scale, base, rms_eps=rms_eps, sinkhorn_eps=sinkhorn_eps,
                                     iterations=iterations)
        event = torch.cuda.Event()
        event.record(side)
    x.record_stream(side)
    for t in (pre, post, comb):
        t.record_stream(main)
    return pre, post, comb, event


def _params_tail_overlapped(x, weight, scale, base, rms_eps, eps, iters):
    """Large batches: the projection in order on the caller's stream (``split_proj``), only the latency-bound Sinkhorn
    tail (``split_tail``) on the side stream: bit for bit ``mhc_params`` (the same two kernels on the same inputs)."""
    _check(x, weight, scale, base)
    lead = x.shape[:-2]
    xs = x.contiguous().view(-1, 4, HIDDEN)
    ws = extension().split_proj(xs, weight.contiguous(), 0)
    main = torch.cuda.current_stream(x.device)
    side = _side_stream(x.device)
    side.wait_stream(main)
    with torch.cuda.stream(side):
        pre, post, comb = extension().split_tail(ws, scale.contiguous(), base.contiguous(), float(rms_eps), float(eps),
                                                 int(iters))
        event = torch.cuda.Event()
        event.record(side)
    ws.record_stream(side)
    for t in (pre, post, comb):
        t.record_stream(main)
    return pre.view(*lead, 4), post.view(*lead, 4), comb.view(*lead, 4, 4), event


def join(event, device=None):
    """Make the current stream (of ``device``) wait for side-stream coefficients (``mhc_params_async``);
    None is a no-op."""
    if event is not None:
        torch.cuda.current_stream(device).wait_event(event)


__all__ = ["OVERLAP_MAX_ROWS", "OVERLAP_SUBLAYERS", "boundary_rows", "collapse", "collapse_backward", "collapse_forward", "expand", "join",
           "mhc_params", "mhc_params_async", "mhc_split_shifted"]
