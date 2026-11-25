"""Float64 references for the gates: the gated delta rule itself, and the state a CPR record stands for."""

from __future__ import annotations

import math

import torch


def gdn_reference(q, k, v, a, b, A_log, dt_bias, cu_seqlens, initial_state=None, dtype=torch.float64):
    """Token recurrence in ``dtype``. Shapes as in ops.py; ``cu_seqlens`` may be a list or tensor.

    Returns ``(out [T, hv, 128], final_state [N, hv, 128, 128])``.
    """
    q, k, v, a, b = (x.to(dtype) for x in (q, k, v, a, b))
    T, hk, Dk = q.shape
    hv = v.shape[1]
    cu = [int(x) for x in (cu_seqlens.tolist() if torch.is_tensor(cu_seqlens) else cu_seqlens)]
    n = len(cu) - 1
    out = torch.zeros(T, hv, Dk, dtype=dtype, device=q.device)
    zero = torch.zeros(hv, Dk, Dk, dtype=dtype, device=q.device)
    S = [zero if initial_state is None else initial_state[i].to(dtype) for i in range(n)]
    kn = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    qn = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) / math.sqrt(Dk)
    x = a + dt_bias.to(dtype)
    sp = torch.where(x <= 20, torch.log1p(torch.exp(x)), x)
    g = torch.exp(-torch.exp(A_log.to(dtype)) * sp)
    beta = torch.sigmoid(b)
    rep = hv // hk
    for i in range(n):
        for t in range(cu[i], cu[i + 1]):
            kk = kn[t].repeat_interleave(rep, 0)
            qq = qn[t].repeat_interleave(rep, 0)
            Sn = S[i] * g[t][:, None, None]
            u = beta[t][:, None] * (v[t] - torch.einsum("hvk,hk->hv", Sn, kk))
            Sn = Sn + u[:, :, None] * kk[:, None, :]
            out[t] = torch.einsum("hvk,hk->hv", Sn, qq)
            S[i] = Sn
    return out, torch.stack(S)


def materialize(state, slots) -> torch.Tensor:
    """The float64 [n, hv, 128, 128] recurrent state each slot's record stands for."""
    idx = slots.long()
    S = state.S[idx].double()
    pos = state.pos[idx, :, 0].long()
    out = S.clone()
    for r in range(idx.numel()):
        for h in range(state.hv):
            j = int(pos[r, h]) % 64
            if j == 0:
                continue
            G = state.gb[idx[r], h, :j].double()
            dec = torch.exp(G[-1] - G)
            d = state.db[idx[r], h, :j].double() * dec[:, None]   # [i][v], row-major in the token index
            kk = state.kb[idx[r], h, :j].double()                 # [i][d]
            out[r, h] = S[r, h] * torch.exp(G[-1]) + d.t() @ kk
    return out
