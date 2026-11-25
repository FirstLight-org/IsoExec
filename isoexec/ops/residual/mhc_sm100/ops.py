"""Python entry points for the SM100 four-stream mHC kernels.

``split_forward`` / ``split_backward`` / ``merge_forward`` / ``merge_backward`` are the raw kernel wrappers. The
DeepSeek-V4.1 recipe (``deepseek_mhc_sm100``) builds its differentiable parameters, collapse and expand on them.
"""

from __future__ import annotations

import torch

from isoexec.ops.residual.mhc_sm100._extension import extension
from isoexec.ops.residual.mhc_sm100.spec import HIDDEN

MAX_ROWS = 128 * 512      # backward row limit per call (segment-count bound of the fixed trees)


def _check_split(x, fn, scale, base):
    if x.dim() < 3 or x.shape[-2] != 4 or x.dtype != torch.bfloat16 or not x.is_cuda:
        raise ValueError("mHC split expects CUDA bf16 streams [..., 4, H]")
    H = x.shape[-1]
    if H not in HIDDEN:
        raise ValueError(f"mHC split serves H in {HIDDEN}")
    if fn.shape != (24, 4 * H) or fn.dtype not in (torch.bfloat16, torch.float32) or fn.device != x.device:
        raise ValueError("fn must be bf16 or fp32 [24, 4H] on the stream device")
    if scale.shape != (3,) or base.shape != (24,) or scale.dtype != torch.float32 or base.dtype != torch.float32:
        raise ValueError("scale [3] and base [24] must be fp32")


def split_forward(streams, fn, scale, base, *, rms_eps, eps, iters, collapse="own", shifted_pre=None,
                  stats=False, mt=0, groups=0):
    """``streams [..., 4, H]`` -> ``(pre [...,4], post [...,4], comb [...,4,4], collapsed [..., H] | None[, (proj, rstd)])``.

    ``collapse``: ``"own"`` (collapse with this call's pre), ``"shifted"`` (with ``shifted_pre`` fp32 [...,4]
    -- DeepSeek shifted mixing) or ``None``. ``stats=True`` also returns ``(proj [...,24], rstd [...])`` for backward.
    ``mt`` (0 = automatic) selects rows per tile and ``groups`` (0 = automatic) the number of persistent CTA groups; both are
    ownership only and never change a value (tests hold them to the same bits).
    """
    _check_split(streams, fn, scale, base)
    lead, H = streams.shape[:-2], streams.shape[-1]
    x = streams.contiguous().view(-1, 4, H)
    mode = {None: 0, "own": 1, "shifted": 2}[collapse]
    ext = None
    if mode == 2:
        if shifted_pre is None or shifted_pre.dtype != torch.float32 or shifted_pre.shape != (*lead, 4):
            raise ValueError("shifted collapse needs fp32 shifted_pre [..., 4]")
        ext = shifted_pre.contiguous().view(-1, 4)
    pre, post, comb, proj, rstd, col = extension().split_forward(
        x, fn.contiguous(), scale.contiguous(), base.contiguous(), ext, mode, float(rms_eps), float(eps),
        int(iters), bool(stats), int(mt), int(groups))
    out = (pre.view(*lead, 4), post.view(*lead, 4), comb.view(*lead, 4, 4),
           col.view(*lead, H) if mode else None)
    if stats:
        return out + ((proj.view(*lead, 24), rstd.view(lead)),)
    return out


def split_backward(streams, fn, scale, base, proj, rstd, dpost, dcomb, *, eps, iters, dpre=None, dcol=None,
                   shifted_pre=None, groups=0):
    """Gradients of the split. Returns ``(dstreams bf16, dfn (fn dtype), dscale fp32 [3], dbase fp32 [24], dshifted_pre | None)``.

    ``dpre``: gradient of the returned ``pre`` (DeepSeek); ``dcol``: gradient of the collapsed output (bf16 [..., H]);
    ``shifted_pre``: the pre the collapse used, if it was the shifted one (then ``dcol`` produces ``dshifted_pre`` and
    no gradient to this call's own pre).
    """
    _check_split(streams, fn, scale, base)
    lead, H = streams.shape[:-2], streams.shape[-1]
    T = streams.numel() // (4 * H)
    if T > MAX_ROWS:
        raise ValueError(f"split backward serves at most {MAX_ROWS} rows per call")
    f = lambda t, *shape: t.contiguous().view(*shape)
    z = lambda *shape: torch.zeros(*shape, dtype=torch.float32, device=streams.device)
    dx, dfn, dscale, dbase, dcolx = extension().split_backward(
        f(streams, T, 4, H), fn.contiguous(), scale.contiguous(), base.contiguous(), f(proj, T, 24), f(rstd, T),
        None if dpre is None else f(dpre, T, 4),
        z(T, 4) if dpost is None else f(dpost, T, 4), z(T, 16) if dcomb is None else f(dcomb, T, 16),
        None if dcol is None else f(dcol, T, H),
        None if shifted_pre is None else f(shifted_pre, T, 4), float(eps), int(iters), int(groups))
    dsp = dcolx.view(*lead, 4) if (shifted_pre is not None and dcol is not None) else None
    return dx.view(streams.shape), dfn, dscale, dbase, dsp


def merge_forward(streams, output, post, comb, *, cast=True):
    """Merge ``[..., 4, H]`` streams, ``[..., H]`` output, fp32 ``post [..., 4]`` / ``comb [..., 4, 4]``.

    ``cast=True``: post/comb and the two merge terms rounded to bf16 at the merge; ``cast=False``: the
    DeepSeek ``expand`` recipe (all fp32, one rounding of the result) used by ``deepseek_mhc_sm100``.
    """
    if streams.dim() < 3 or streams.shape[-2] != 4 or streams.dtype != torch.bfloat16 or not streams.is_cuda:
        raise ValueError("mHC merge expects CUDA bf16 streams [..., 4, H]")
    H = streams.shape[-1]
    if output.shape != (*streams.shape[:-2], H) or output.dtype != torch.bfloat16:
        raise ValueError("output must be bf16 [..., H]")
    if post.dtype != torch.float32 or comb.dtype != torch.float32 or post.shape[-1:] != (4,) or comb.shape[-2:] != (4, 4):
        raise ValueError("post [..., 4] and comb [..., 4, 4] must be fp32")
    res = extension().merge_forward(streams.contiguous().view(-1, 4, H), output.contiguous().view(-1, H),
                                    post.contiguous().view(-1, 4), comb.contiguous().view(-1, 16), cast)
    return res.view(streams.shape)


def merge_backward(grad, streams, output, post, comb, *, cast=True):
    """``(dstreams bf16, doutput bf16, dpost fp32, dcomb fp32)`` for the merge."""
    H = streams.shape[-1]
    dx, do, dp, dc = extension().merge_backward(
        grad.contiguous().view(-1, 4, H), streams.contiguous().view(-1, 4, H), output.contiguous().view(-1, H),
        post.contiguous().view(-1, 4), comb.contiguous().view(-1, 16), cast)
    return dx.view(streams.shape), do.view(output.shape), dp.view(post.shape), dc.view(comb.shape)


def gather_rows(index, tensors):
    """``[t.index_select(0, index) for t in tensors]`` in ONE launch (a byte copy per row; 1..8 tensors with the
    same leading row count, any dtypes and row shapes): the CED decoder's segment gather. ``index`` int64 [R] with
    every entry in ``[0, T)``. Tensors that autograd tracks, host tensors and anything the kernel does not serve
    take ``index_select`` (the same values)."""
    tensors = list(tensors)
    if (not tensors or len(tensors) > 8 or not index.is_cuda or index.dtype != torch.int64 or index.dim() != 1
            or any(not t.is_cuda or t.dim() < 1 or t.shape[0] != tensors[0].shape[0] or t.device != index.device
                   for t in tensors)
            or (torch.is_grad_enabled() and any(t.requires_grad for t in tensors))):
        return [t.index_select(0, index) for t in tensors]
    srcs = [t.contiguous() for t in tensors]
    dsts = [torch.empty((index.shape[0], *t.shape[1:]), dtype=t.dtype, device=t.device) for t in srcs]
    extension().gather_rows(index.contiguous(), srcs, dsts)
    return dsts


def tf32_fn(fn):
    """The coefficient weight rounded to TF32 (round to nearest, ties away from zero: ``cvt.rna.tf32.f32``), the
    precision native ``mega_mhc`` multiplies with. Its value is exactly two BF16 pieces, so the projection with
    ``pieces=2`` is the exact product of the TF32 weight (the PROTOTYPE of the TF32-fn recipe option; not a default).
    Finite inputs only."""
    if fn.dtype != torch.float32:
        raise ValueError("fn must be fp32")
    u = fn.contiguous().view(torch.int32)
    r = ((u + 0x1000) & ~0x1FFF).view(torch.float32)
    return torch.where(torch.isfinite(fn), r, fn)


def split_proj(x, fn, *, groups=0, pieces=3):
    """The DeepSeek coefficient projection's slice-partial workspace ``[32, T, 32]`` (``split_forward``'s projection
    launch). ``pieces=3``: the exact FP32 fn (the recipe); ``pieces=2``: two BF16 pieces of ``fn`` -- exact only for a
    TF32 ``fn`` (``tf32_fn``), the prototype of the TF32 option."""
    return extension().split_proj(x.contiguous(), fn.contiguous(), int(groups), int(pieces))


def split_tail(ws, scale, base, *, rms_eps, eps, iters):
    """``(pre, post, comb)`` from a projection workspace (``split_forward``'s tail launch)."""
    return extension().split_tail(ws, scale.contiguous(), base.contiguous(), float(rms_eps), float(eps), int(iters))


__all__ = ["MAX_ROWS", "gather_rows", "split_proj", "split_tail", "tf32_fn", "merge_backward", "merge_forward", "split_backward", "split_forward"]
