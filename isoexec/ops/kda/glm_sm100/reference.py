"""Float64 oracles of the KDA contract (README.md, the contract equation), packed varlen form.

``recurrence`` is the token recurrence of the op contract evaluated in float64 -- the same equation as
``isoexec.nn.GLMKimiDeltaRule(backend="reference")``, which evaluates it in fp32 on [B, T, H, K] -- and is
the accuracy oracle of the battery. It is differentiable (autograd through float64 torch), which is how
the backward battery obtains reference gradients, including gradients through the returned final state.
"""

from __future__ import annotations

import torch


def l2n(x, eps=1e-6):
    return x / torch.sqrt(x.square().sum(-1, keepdim=True) + eps)


def recurrence(q, k, v, g, beta, cu_seqlens, initial_state=None, *, dtype=torch.float64):
    """``(o [T, H, V], final [N, H, K, V])`` in ``dtype`` for packed ``[T, H, *]`` inputs."""
    q, k, v, g, beta = (t.to(dtype) for t in (q, k, v, g, beta))
    q = l2n(q) * (q.shape[-1] ** -0.5)
    k = l2n(k)
    cu = [int(c) for c in cu_seqlens.tolist()]
    n, H, K, V = len(cu) - 1, q.shape[1], q.shape[2], v.shape[2]
    outs, finals = [], []
    for s in range(n):
        S = (initial_state[s].to(dtype) if initial_state is not None
             else torch.zeros(H, K, V, dtype=dtype, device=q.device))
        for t in range(cu[s], cu[s + 1]):
            S = S * g[t].exp().unsqueeze(-1)
            mem = (S * k[t].unsqueeze(-1)).sum(-2)
            delta = (v[t] - mem) * beta[t].unsqueeze(-1)
            S = S + k[t].unsqueeze(-1) * delta.unsqueeze(-2)
            outs.append((S * q[t].unsqueeze(-1)).sum(-2))
        finals.append(S)
    o = torch.stack(outs) if outs else v.new_zeros(0, H, V, dtype=dtype)
    return o, torch.stack(finals)


def chunked(q, k, v, g, beta, cu_seqlens, initial_state=None, *, dtype=torch.float64, chunk=64):
    """The chunk form of the same function in ``dtype`` (no roundings): an independent check of the
    recipe's algebra (WY / per-channel decay factorization) separate from the kernel's rounding."""
    q, k, v, g, beta = (t.to(dtype) for t in (q, k, v, g, beta))
    q = l2n(q) * (q.shape[-1] ** -0.5)
    k = l2n(k)
    cu = [int(c) for c in cu_seqlens.tolist()]
    n, H, K, V = len(cu) - 1, q.shape[1], q.shape[2], v.shape[2]
    o = v.new_zeros(v.shape[0], H, V, dtype=dtype)
    finals = []
    for s in range(n):
        S = (initial_state[s].to(dtype) if initial_state is not None
             else torch.zeros(H, K, V, dtype=dtype, device=q.device))
        for c0 in range(cu[s], cu[s + 1], chunk):
            c1 = min(c0 + chunk, cu[s + 1])
            qc, kc, vc, bc = (t[c0:c1].transpose(0, 1) for t in (q, k, v, beta))   # [H, r, *]
            G = g[c0:c1].transpose(0, 1).cumsum(1)                                 # [H, r, K]
            r = c1 - c0
            dec = (G.unsqueeze(2) - G.unsqueeze(1))                                # [H, j, i, K]
            tri = torch.tril(torch.ones(r, r, dtype=torch.bool, device=q.device))
            dec = torch.where(tri[None, :, :, None], dec, torch.full_like(dec, -torch.inf)).exp()
            KK = torch.einsum("hjc,hic,hjic->hji", kc, kc, dec)
            KQ = torch.einsum("hjc,hic,hjic->hji", qc, kc, dec)
            Lm = torch.tril(KK, -1) * bc.unsqueeze(-1)
            X = torch.linalg.inv(torch.eye(r, dtype=dtype, device=q.device) + Lm)
            Y = X * bc.unsqueeze(1)
            u = Y @ vc
            w = Y @ (kc * G.exp())
            d = u - w @ S
            o[c0:c1] = (((qc * G.exp()) @ S) + torch.tril(KQ) @ d).transpose(0, 1)
            GL = G[:, -1:, :]
            S = S * GL.transpose(1, 2).exp() + (kc * (GL - G).exp()).transpose(1, 2) @ d
        finals.append(S)
    return o, torch.stack(finals)
