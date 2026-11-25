"""Equation-level references for the four-stream mHC kernels (CPU-runnable, any dtype).

``split_reference`` / ``merge_reference`` transcribe the mHC split and the cast merge (``cast=True``) with a
caller-chosen compute dtype: float64 gives the accuracy oracle the native kernels are held to; with
``dtype=torch.float32`` it is the eager recipe (normalize first, then project) up to evaluation order.
The one declared reordering of the native program: the native kernel projects the raw bf16 streams and
scales the 24 results by rstd (``mix = proj * rstd``).
"""

from __future__ import annotations

import torch


def split_reference(streams, fn, scale, base, *, rms_eps, hc_eps, iters, dtype=torch.float64,
                    shifted_pre=None):
    """Returns (pre, post, comb, collapsed, proj, rstd) in ``dtype`` (collapsed in ``dtype`` too).

    ``shifted_pre`` (DeepSeek): collapse with a caller-supplied ``pre`` instead of the one computed here.
    """
    n = streams.shape[-2]
    x = streams.to(dtype)
    flat = x.flatten(-2)
    rstd = torch.rsqrt(flat.square().mean(-1, keepdim=True) + rms_eps)
    proj = flat @ fn.to(dtype).t()
    mixes = proj * rstd
    pre_m, post_m, comb_m = mixes.split((n, n, n * n), dim=-1)
    pre_b, post_b, comb_b = base.to(dtype).split((n, n, n * n))
    sc = scale.to(dtype)
    pre = torch.sigmoid(pre_m * sc[0] + pre_b) + hc_eps
    post = 2 * torch.sigmoid(post_m * sc[1] + post_b)
    comb = (comb_m.reshape(*comb_m.shape[:-1], n, n) * sc[2] + comb_b.reshape(n, n)).softmax(-1) + hc_eps
    comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
    cp = pre if shifted_pre is None else shifted_pre.to(dtype)
    collapsed = (cp.unsqueeze(-1) * x).sum(-2)
    return pre, post, comb, collapsed, proj, rstd.squeeze(-1)


def merge_reference(streams, output, post, comb, dtype=torch.float64):
    """The cast merge equation in ``dtype`` with NO intermediate casts (float64 oracle): post/comb are not
    rounded to the stream dtype here; the native recipe's casts are checked by ``merge_declared``."""
    return post.to(dtype).unsqueeze(-1) * output.to(dtype).unsqueeze(-2) + (
        comb.to(dtype).transpose(-1, -2) @ streams.to(dtype))


def merge_declared(streams, output, post, comb):
    """The native merge program in torch fp32 ops (bitwise comparable with the kernel): post and comb cast to the
    stream dtype, t1 = bf16(post_b * out), t2 = bf16(sequential fp32 sum over source streams of comb_b * x),
    result = bf16(t1 + t2)."""
    dt = streams.dtype
    pb, cb = post.to(dt).float(), comb.to(dt).float()
    x, o = streams.float(), output.float()
    t1 = (pb.unsqueeze(-1) * o.unsqueeze(-2)).to(dt).float()
    acc = cb[..., 0, :].unsqueeze(-1) * x[..., 0, :].unsqueeze(-2)
    for s in range(1, streams.shape[-2]):
        acc = acc + cb[..., s, :].unsqueeze(-1) * x[..., s, :].unsqueeze(-2)
    return (t1 + acc.to(dt).float()).to(dt)


def merge_backward_declared(grad, streams, output, post, comb):
    """The native merge backward in torch fp32 ops, including the declared reduction tree (bitwise comparable).

    ``grad/streams [T, 4, H]``, ``output [T, H]`` bf16; ``post [T, 4]``, ``comb [T, 4, 4]`` fp32. Returns
    ``(dstreams bf16, doutput bf16, dpost fp32, dcomb fp32)``.
    """
    dt = streams.dtype
    T, _, H = streams.shape
    g, x, o = grad.float(), streams.float(), output.float()
    pb, cb = post.to(dt).float(), comb.to(dt).float()
    dout = (g[:, 0] * pb[:, 0:1])
    for i in range(1, 4):
        dout = dout + g[:, i] * pb[:, i:i + 1]
    dx = []
    for s in range(4):
        a = cb[:, s, 0:1] * g[:, 0]
        for i in range(1, 4):
            a = a + cb[:, s, i:i + 1] * g[:, i]
        dx.append(a)
    dx = torch.stack(dx, 1).to(dt)
    # reductions: lane j owns chunks j, j+128, ... of 8 h; sequential fp32 within a lane
    nchunk = H // 8
    prods = []   # 20 products streams: [T, 20, H]
    for i in range(4):
        prods.append(g[:, i] * o)
    for s in range(4):
        for i in range(4):
            prods.append(g[:, i] * x[:, s])
    pr = torch.stack(prods, 1).view(T, 20, nchunk, 8)
    lane_acc = torch.zeros(T, 20, 128, device=g.device, dtype=torch.float32)
    for c0 in range(0, nchunk, 128):
        n = min(128, nchunk - c0)
        for e in range(8):
            lane_acc[:, :, :n] = lane_acc[:, :, :n] + pr[:, :, c0:c0 + n, e]
    v = lane_acc.view(T, 20, 4, 32)             # [warp, lane]
    for m in (16, 8, 4, 2, 1):                  # xor butterfly == pairing (l, l + m) at lane 0
        v = v[..., :m] + v[..., m:2 * m]
    w = v[..., 0]                               # [T, 20, 4 warps]
    tot = (w[..., 0] + w[..., 1]) + (w[..., 2] + w[..., 3])
    q = tot.to(dt).float()
    return dx, dout.to(dt), q[:, 0:4].contiguous(), q[:, 4:20].reshape(T, 4, 4).contiguous()


def eager_split(streams, fn, base, scale, *, rms_eps, hc_eps, sinkhorn_iters):
    """The eager hyper-connection split (normalize, project, Sinkhorn, own collapse) in PyTorch."""
    n = streams.shape[-2]
    flat = streams.flatten(-2).float()
    flat = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + rms_eps)
    pre, post, comb = torch.nn.functional.linear(flat, fn.float()).split((n, n, n * n), dim=-1)
    pre_b, post_b, comb_b = base.float().split((n, n, n * n))
    pre = (pre * scale[0] + pre_b).sigmoid() + hc_eps
    post = 2 * (post * scale[1] + post_b).sigmoid()
    comb = (comb.reshape(*comb.shape[:-1], n, n) * scale[2] + comb_b.reshape(n, n)).softmax(-1)
    comb = comb + hc_eps
    comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
    collapsed = (pre.unsqueeze(-1) * streams).sum(-2).to(streams.dtype)
    return post, comb, collapsed


def eager_merge(streams, output, post, comb):
    """The eager cast merge in PyTorch."""
    return post.to(streams.dtype).unsqueeze(-1) * output.unsqueeze(-2) + (
        comb.to(streams.dtype).transpose(-1, -2) @ streams)
