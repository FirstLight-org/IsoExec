"""Independent oracles for the Engram package (no native code).

* ``engram_residual_f64`` -- the checkpoint's gate/residual equation (``weight = q*k`` first) in float64,
  differentiable, the accuracy oracle for forward and backward.
* ``decode_fp8_rows`` -- ``Engram.embed`` of the checkpoint (``ParallelEngramEmbedding.forward`` at world
  size 1): fp32 multiply of the decoded e4m3 value by the UE8M0 scale, one narrowing.
* ``emulate_row_grad`` -- the declared duplicate-row reduction order, executed in fp32 torch ops on the CPU.
"""

from __future__ import annotations

import torch


def engram_residual_f64(x, key, value, q_weight, k_weight, eps, token_mask=None):
    h, k, v = x.double(), key.double(), value.double()
    w = q_weight.double() * k_weight.double()
    D = x.shape[-1]
    rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(k.square().mean(-1) + eps)
    dot = (h * w * k).sum(-1) * rstd * D ** -0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
    if token_mask is not None:
        gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
    return h + gate.unsqueeze(-1) * v.unsqueeze(-2), gate


def decode_fp8_rows(weight, scale, indices, dtype=torch.bfloat16):
    values = torch.nn.functional.embedding(indices, weight).float().unflatten(-1, (-1, 32))
    scales = torch.nn.functional.embedding(indices, scale).float()
    return (values * scales.unsqueeze(-1)).flatten(-2).to(dtype)


def emulate_row_grad(indices, grad):
    """Exact declared order: stable sort by row; per run, stripe w = entries w, w+8, ... summed sequentially
    in fp32; stripes merged ((p0+p1)+(p2+p3))+((p4+p5)+(p6+p7)). Returns (rows, g fp32)."""
    idx = indices.reshape(-1).cpu()
    g = grad.reshape(idx.numel(), -1).float().cpu()
    order = torch.sort(idx, stable=True).indices
    sidx = idx[order]
    rows, out = [], []
    start = 0
    n = idx.numel()
    while start < n:
        end = start
        while end < n and sidx[end] == sidx[start]:
            end += 1
        stripes = []
        for w in range(8):
            acc = torch.zeros(g.shape[1], dtype=torch.float32, device=g.device)
            for e in range(start + w, end, 8):
                acc = acc + g[order[e]]
            stripes.append(acc)
        p = stripes
        out.append(((p[0] + p[1]) + (p[2] + p[3])) + ((p[4] + p[5]) + (p[6] + p[7])))
        rows.append(int(sidx[start]))
        start = end
    return torch.tensor(rows, dtype=torch.int64), torch.stack(out)
