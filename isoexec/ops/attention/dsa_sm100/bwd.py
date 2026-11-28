"""The sparse-attention backward (recipe ``sm100_dsa_sparse_bwd_v1``, SPEC 6.1), under the user's
backward rule: NO atomics anywhere, bitwise determinism not required, accuracy against float64.

    dq, dkv = dsa_sparse_attn_bwd(q, kv, lists, out, lse, dO, softmax_scale)

``q [T, h, D_QK]`` bf16, ``kv [S, D_QK]`` bf16 (D_QK = 576 training geometry, or 512; V = the first
512 dims), ``lists [T, K]`` ascending with trailing -1 pads, ``out [T, h, 512]`` / ``lse [T, h]``
(natural log) from the forward, ``dO [T, h, 512]``.  Returns ``dq [T, h, D_QK]`` bf16 and
``dkv [S, D_QK]`` bf16 (exact zeros for a key no query selected).

Program, per chunk of query rows (the chunk is execution only: the bits do not depend on it, see below):
  Delta = rowsum(out * dO)                                     (csrc/bwd delta, fp32)
  [S;dP]= [q;dO] kv[list]^T                                    (csrc/bwd ggemm_sd: the key rows TMA-gathered from
                                                                kv inside the GEMM, never materialised; fp32 out)
  P     = exp2(S scale log2e - lse log2e) (pads 0, bf16 as the forward rounded it),
  dS    = scale P (dP - Delta)                                 (csrc/bwd grad)
  dq    = dS kv[list]                                          (csrc/bwd ggemm_dq: gathered B, bf16 stored in place)
  rows  = [dS ; P]^T [q ; dO|0]  (dK + dV of every (t, p) in ONE GEMM: K = 2h; csrc/bwd ggemm_rows reads
                                 both operands in place)
  dKV[j] += rows[t, p] with list[t, p] == j                    (csrc/bwd reduce: a CTA owns a key block)
dKV is held in fp32 across chunks and rounded to bf16 once at the end.  Each key's sum is ONE fp32 chain over its
sources in ascending (row, slot) order, carried across chunks: dKV, dq and Delta are row-local or chained in row
order, so the gradient bits do not depend on the chunk size or on the other sequences of the call.  cuBLAS runs in its default
CUBLAS_ATOMICS_NOT_ALLOWED mode; README "Step 10" has the SASS scan of every kernel a call launches.
"""

from __future__ import annotations

import torch

from ._extension import extension

__all__ = ["chunk_rows", "dsa_sparse_attn_bwd"]


def chunk_rows(h: int, K: int, dqk: int, budget_bytes: int = 3 << 30, *, gathered: bool = False) -> int:
    """Rows per chunk, from the peak live temporaries of one chunk.

    gathered (the ggemm path): [dS ; P] bf16 (4hK bytes) lives throughout, beside first the fp32 [S ; dP] (8hK) and
    then -- after that is freed -- the fp32 key-side rows (4 K dqk). Fallback path: the gathered keys (bf16), the
    key-side fp32 rows, S, dP, [dS ; P] and the stacked [q ; dO] operand.
    """
    if gathered:
        per_row = h * K * 4 + max(h * K * 8, K * dqk * 4)
    else:
        per_row = K * dqk * (2 + 4) + h * K * (4 + 4 + 2 * 2 * 2)
    return max(1, min(1024, budget_bytes // per_row))


def _bmm_f32(a, b):
    """``aten::bmm.dtype`` (FP32 out) through the dispatcher: a host that rebinds the Python ``torch.bmm``
    (a trainer's batch-invariant install) keeps this backward on the registered provider chain."""
    return torch.ops.aten.bmm.dtype(a, b, torch.float32)


def dsa_sparse_attn_bwd(q, kv, lists, out, lse, dO, softmax_scale, *, topk_length=None, chunk=None,
                        recompute_delta=False):
    if recompute_delta and topk_length is not None:
        raise ValueError("sink score-delta requires explicit trailing list pads")
    T, h, dqk = q.shape
    K = lists.shape[-1]
    lists = lists.reshape(T, K)
    if lists.dtype != torch.int32:
        raise ValueError("lists: int32")
    ext = extension("bwd")
    dv = dO.shape[-1]
    out = out.contiguous()
    dO = dO.contiguous()
    lse = lse.reshape(T, h).contiguous().float()
    delta = torch.empty((T, h), dtype=torch.float32, device=q.device)
    if not recompute_delta:
        ext.delta(out, dO, delta)
    dq = torch.empty((T, h, dqk), dtype=torch.bfloat16, device=q.device)
    acc = torch.zeros((kv.shape[0], dqk), dtype=torch.float32, device=q.device)
    gathered = h % 64 == 0 and dqk % 64 == 0 and K % 64 == 0 and lists.is_contiguous()
    C = chunk or chunk_rows(h, K, dqk, gathered=gathered)
    q = q.contiguous()
    kv = kv.contiguous()
    for t0 in range(0, T, C):
        t1 = min(T, t0 + C)
        c = t1 - t0
        ic = lists[t0:t1]
        qc = q[t0:t1]
        Kg = sd = None
        if gathered:   # the key rows gathered inside the GEMMs (no [c, K, D] copy)
            sd = torch.empty((c, 2 * h, K), dtype=torch.float32, device=q.device)
            ext.ggemm_sd(qc, dO[t0:t1], kv, ic, sd)
            S, dP = sd[:, :h], sd[:, h:]
        else:
            Kg = kv.index_select(0, ic.clamp(min=0).reshape(-1)).view(c, K, dqk)
            S = _bmm_f32(qc, Kg.transpose(1, 2))
            dP = _bmm_f32(dO[t0:t1], Kg[:, :, :dv].transpose(1, 2))
        if recompute_delta:
            ext.score_delta(S, dP, lse[t0:t1], ic, delta[t0:t1], float(softmax_scale))
        A = torch.empty((c, 2 * h, K), dtype=torch.bfloat16, device=q.device)
        ext.grad(S, dP, lse[t0:t1], delta[t0:t1], ic, topk_length, t0, A, float(softmax_scale))
        del S, dP, sd   # the views alone do not free [S ; dP]: drop it before the fp32 rows are allocated
        if gathered:
            ext.ggemm_dq(A, kv, ic, dq[t0:t1])
        else:
            dq[t0:t1] = _bmm_f32(A[:, :h], Kg).to(torch.bfloat16)
        if gathered:   # [dS ; P]^T [q ; dO|0] with both operands read in place (MN-major), no stacked copy
            rows = torch.empty((c, K, dqk), dtype=torch.float32, device=q.device)
            ext.ggemm_rows(A, qc, dO[t0:t1], rows)
        else:
            B = torch.zeros((c, 2 * h, dqk), dtype=torch.bfloat16, device=q.device)
            B[:, :h] = qc
            B[:, h:, :dv] = dO[t0:t1]
            rows = _bmm_f32(A.transpose(1, 2), B)      # [c, K, dqk]
            del B
        del A, Kg
        ext.reduce(rows, ic, topk_length, t0, acc)
        del rows
    return dq, acc.to(torch.bfloat16)
