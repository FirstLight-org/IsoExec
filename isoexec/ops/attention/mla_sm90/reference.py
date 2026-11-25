"""Float64 oracle for absorbed MLA: q [T,h,576] attends a paged [c_kv 512 | k_pe 64] cache with values c_kv.
Queries are right-aligned at p=seqused_k-q_len+i, attend keys [0,p], and return natural-log lse;
empty rows return out=0/lse=-inf and must be excluded from comparisons with the kernel's finite sentinel.
Autograd differentiates sum(dout*out) in float64, excluding lse; this oracle does not reproduce kernel rounding."""

from __future__ import annotations

import torch

LATENT = 512
ROPE = 64
WIDTH = LATENT + ROPE


def _host_ints(t) -> list:
    return [int(x) for x in (t.tolist() if isinstance(t, torch.Tensor) else t)]


def _check(q, kv_cache, block_table, cu_seqlens_q, seqused_k):
    if q.ndim != 3 or q.shape[-1] != WIDTH:
        raise ValueError(f"q must be [T, h_q, {WIDTH}], got {tuple(q.shape)}")
    if kv_cache.ndim != 4 or kv_cache.shape[2] != 1 or kv_cache.shape[-1] != WIDTH:
        raise ValueError(f"kv_cache must be [num_blocks, page, 1, {WIDTH}], got {tuple(kv_cache.shape)}")
    cu = _host_ints(cu_seqlens_q)
    used = _host_ints(seqused_k)
    if len(cu) != len(used) + 1 or cu[0] != 0 or cu[-1] != q.shape[0]:
        raise ValueError("cu_seqlens_q must be [0, ..., T] with B+1 entries")
    if block_table.ndim != 2 or block_table.shape[0] != len(used):
        raise ValueError("block_table must be [B, max_blocks]")
    return cu, used


def _request_rows(q, kv, block_table, b, start, end, used, page, scale, v_dim, chunk_rows):
    """float64 (out, lse) for query rows [start, end) of request b; kv is the float64 cache."""
    n_q = end - start
    h = q.shape[1]
    out = q.new_zeros((n_q, h, v_dim), dtype=torch.float64)
    lse = q.new_full((n_q, h), float("-inf"), dtype=torch.float64)
    if n_q == 0 or used <= 0:
        return out, lse
    n_blocks = (used + page - 1) // page
    pages = block_table[b, :n_blocks].long()
    keys = kv[pages].reshape(-1, kv.shape[-1])[:used]  # [used, 576]
    values = keys[:, :v_dim]
    positions = torch.arange(used - n_q, used, device=q.device)  # logical position of each row
    key_pos = torch.arange(used, device=q.device)
    outs, lses = [], []
    for r0 in range(0, n_q, chunk_rows):
        r1 = min(n_q, r0 + chunk_rows)
        qr = q[start + r0 : start + r1].to(torch.float64)  # [r, h, 576]
        score = torch.einsum("rhd,kd->rhk", qr, keys) * scale
        mask = key_pos.view(1, 1, -1) > positions[r0:r1].view(-1, 1, 1)  # future keys
        score = score.masked_fill(mask, float("-inf"))
        row_lse = torch.logsumexp(score, dim=-1)  # [r, h]
        prob = torch.softmax(score, dim=-1)
        # Rows with border <= 0 have all keys masked: softmax gives NaN; they are defined as empty.
        empty = (positions[r0:r1] < 0).view(-1, 1, 1)
        prob = torch.where(empty, torch.zeros_like(prob), prob)
        outs.append(torch.einsum("rhk,kv->rhv", prob, values))
        lses.append(row_lse)
    return torch.cat(outs), torch.cat(lses)


def mla_reference_fwd(q, kv_cache, block_table, cu_seqlens_q, seqused_k, softmax_scale: float, *,
                      v_dim: int = LATENT, chunk_rows: int = 256):
    """float64 ``(out [T, h_q, v_dim], lse [T, h_q])`` of absorbed causal MLA over a paged cache."""
    cu, used = _check(q, kv_cache, block_table, cu_seqlens_q, seqused_k)
    kv = kv_cache.to(torch.float64)[:, :, 0, :]  # [blocks, page, 576]
    page = kv_cache.shape[1]
    outs, lses = [], []
    for b in range(len(used)):
        o, l = _request_rows(q, kv, block_table, b, cu[b], cu[b + 1], used[b], page,
                             float(softmax_scale), v_dim, chunk_rows)
        outs.append(o)
        lses.append(l)
    if not outs:
        return (q.new_zeros((0, q.shape[1], v_dim), dtype=torch.float64),
                q.new_zeros((0, q.shape[1]), dtype=torch.float64))
    return torch.cat(outs), torch.cat(lses)


def reference_attention(q_nope, q_pe, c_kv, k_pe, w_uk, w_uv, *, seq_lens, scale: float):
    """Differentiable float64 reference of the packed trainer op ``mla_sm90_attention``.

    Unabsorbed algebra: ``k_nope = W_uk c``, ``v = W_uv c`` and softmax attention with keys
    ``[k_nope | k_pe]`` against queries ``[q_nope | q_pe]``. It equals the absorbed form exactly in
    float64 (``q_nope . W_uk c = (W_uk^T q_nope) . c``). Inputs may require grad; everything is
    promoted to float64 inside, so gradients arrive in the leaves' own dtype via autograd.
    """
    lens = [int(n) for n in seq_lens]
    q_nope, q_pe, c_kv, k_pe, w_uk, w_uv = (t.to(torch.float64) for t in (q_nope, q_pe, c_kv, k_pe, w_uk, w_uv))
    outs, start = [], 0
    for n in lens:
        qn, qp = q_nope[start:start + n], q_pe[start:start + n]
        c, kp = c_kv[start:start + n], k_pe[start:start + n]
        k_nope = torch.einsum("hnl,kl->khn", w_uk, c)  # [n, H, nope]
        v = torch.einsum("hvl,kl->khv", w_uv, c)  # [n, H, v]
        score = (torch.einsum("rhn,khn->hrk", qn, k_nope) + torch.einsum("rhp,kp->hrk", qp, kp)) * scale
        mask = torch.ones(n, n, dtype=torch.bool, device=score.device).triu(1)
        prob = score.masked_fill(mask, float("-inf")).softmax(-1)
        outs.append(torch.einsum("hrk,khv->rhv", prob, v))
        start += n
    return torch.cat(outs)
