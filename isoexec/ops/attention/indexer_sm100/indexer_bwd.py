"""The indexer backward (recipe ``sm100_dsa_indexer_bwd_v1``, SPEC 6.2; -- LOWEST
priority: PR #2262 trains with the indexer FROZEN, ``autograd.INDEXER_FROZEN = True``, so this never
runs on the training path).  Under the backward rule: no atomics, not bitwise-required.

    dq, dw, dk = dsa_indexer_bwd(q_idx, w, k_idx, lists, d_scores)

Contract = slime's ``indexer_bwd_interface``: ``q_idx [T, H, 128]`` bf16 (``IDX_STE = identity``:
the values the forward quantised, gradients w.r.t. them unchanged), ``w [T, H]`` fp32, ``k_idx
[S, 128]`` bf16, ``lists [T, K]`` (-1 pads), ``d_scores [T, K]`` fp32 (gradient of the selected
scores ``s[t, p] = sum_h w[t, h] relu(q[t, h] . k[idx[t, p]])``).  Returns ``dq`` bf16, ``dw`` fp32,
``dk [S, 128]`` fp32 (zeros for an unselected key).

Per chunk of rows: d = q Kg^T (fp32), r = relu(d), g = (d > 0) d_s w; dw = sum_p r d_s;
dq = g Kg; rows = g^T q; dk[j] += rows of the (t, p) selecting j (csrc/bwd reduce: a CTA owns a key
block; no atomics).
"""

from __future__ import annotations

import torch

# `reduce` (the atomics-free key-row reduction) lives in the sparse attention's backward extension (csrc/bwd/bwd.cu)
from isoexec.ops.attention.sparse_sm100._extension import extension

__all__ = ["dsa_indexer_bwd"]


def dsa_indexer_bwd(q_idx, w, k_idx, lists, d_scores, *, chunk=1024):
    T, H, D = q_idx.shape
    K = lists.shape[-1]
    lists = lists.reshape(T, K).contiguous()
    ext = extension("bwd")
    dq = torch.empty_like(q_idx)
    dw = torch.empty((T, H), dtype=torch.float32, device=q_idx.device)
    acc = torch.zeros((k_idx.shape[0], D), dtype=torch.float32, device=q_idx.device)
    wf = w.reshape(T, H).float()
    for t0 in range(0, T, chunk):
        t1 = min(T, t0 + chunk)
        ic = lists[t0:t1]
        valid = (ic >= 0)
        Kg = k_idx.index_select(0, ic.clamp(min=0).reshape(-1)).view(t1 - t0, K, D)
        qc = q_idx[t0:t1]
        d = torch.bmm(qc, Kg.transpose(1, 2), out_dtype=torch.float32)               # [c, H, K]
        ds = torch.where(valid, d_scores[t0:t1].float(), 0.0)[:, None, :]
        dw[t0:t1] = (torch.relu(d) * ds).sum(-1)
        g = torch.where(d > 0, ds * wf[t0:t1, :, None], 0.0)
        gb = g.to(torch.bfloat16)
        dq[t0:t1] = torch.bmm(gb, Kg, out_dtype=torch.float32).to(dq.dtype)
        rows = torch.bmm(gb.transpose(1, 2), qc, out_dtype=torch.float32)              # [c, K, D]
        ext.reduce(rows, ic, None, t0, acc)
    return dq, dw, acc
