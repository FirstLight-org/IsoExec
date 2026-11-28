"""Float64 / declared references for the DeepSeek shifted mHC (CPU-runnable)."""

from __future__ import annotations

import torch

from isoexec.ops.residual.mhc_sm100.reference import split_reference


def params_reference(x, weight, scale, base, *, rms_eps, sinkhorn_eps, iterations, dtype=torch.float64):
    """(pre, post, comb) in ``dtype``: mixes = (flat . W^T) * rsqrt(mean(flat^2) + rms_eps), as ``DeepSeekMHC``."""
    pre, post, comb, *_ = split_reference(x, weight, scale, base, rms_eps=rms_eps, hc_eps=sinkhorn_eps,
                                          iters=iterations, dtype=dtype)
    return pre, post, comb


def collapse_reference(x, pre, dtype=torch.float64):
    return (x.to(dtype) * pre.to(dtype).unsqueeze(-1)).sum(-2)


def expand_reference(x, residual, post, comb, dtype=torch.float64):
    mixed = (comb.to(dtype).unsqueeze(-1) * residual.to(dtype).unsqueeze(-2)).sum(-3)
    return post.to(dtype).unsqueeze(-1) * x.to(dtype).unsqueeze(-2) + mixed


def collapse_declared(x, pre):
    """Bitwise program of the native collapse in torch fp32 ops."""
    xf, p = x.float(), pre.float().unsqueeze(-1)
    acc = p[..., 0, :] * xf[..., 0, :]
    for s in range(1, 4):
        acc = acc + p[..., s, :] * xf[..., s, :]
    return acc.to(x.dtype)


def expand_declared(x, residual, post, comb):
    """Bitwise program of the native expand in torch fp32 ops."""
    xf, r = x.float(), residual.float()
    t1 = post.unsqueeze(-1) * xf.unsqueeze(-2)
    acc = comb[..., 0, :].unsqueeze(-1) * r[..., 0, :].unsqueeze(-2)
    for s in range(1, 4):
        acc = acc + comb[..., s, :].unsqueeze(-1) * r[..., s, :].unsqueeze(-2)
    return (t1 + acc).to(x.dtype)


def eager_params(x, weight, scale, base, *, rms_eps, sinkhorn_eps, iterations):
    """The DeepSeek-V4.1 mHC coefficients in eager PyTorch (normalize, project, sigmoid gates, Sinkhorn)."""
    n = 4
    flat = x.flatten(-2).float()
    mixes = torch.nn.functional.linear(flat, weight.float()) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + rms_eps)
    pre = torch.sigmoid(mixes[..., :n] * scale[0] + base[:n]) + sinkhorn_eps
    post = 2 * torch.sigmoid(mixes[..., n:2 * n] * scale[1] + base[n:2 * n])
    comb = (mixes[..., 2 * n:] * scale[2] + base[2 * n:]).unflatten(-1, (n, n))
    comb = comb.softmax(-1) + sinkhorn_eps
    comb = comb / (comb.sum(-2, keepdim=True) + sinkhorn_eps)
    for _ in range(iterations - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + sinkhorn_eps)
        comb = comb / (comb.sum(-2, keepdim=True) + sinkhorn_eps)
    return pre, post, comb


def eager_collapse(x, pre):
    return (x.float() * pre.unsqueeze(-1)).sum(-2).to(x.dtype)


def eager_expand(x, residual, post, comb):
    mixed = (comb.unsqueeze(-1) * residual.unsqueeze(-2)).sum(-3)
    return (post.unsqueeze(-1) * x.unsqueeze(-2) + mixed).to(x.dtype)
