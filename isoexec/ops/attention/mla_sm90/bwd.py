"""Numerical backward for absorbed MLA using FA3's varlen backward (gqa_sm90) on decompressed per-head Q/K/V.
It consumes the forward's y and lse without recomputation, then forms projection gradients with cuBLAS BMM.
Backward is not deterministic: the Q/K width nope+64 (192 for DeepSeek) rounds to FA3's head-dim-256 backward,
which accumulates with atomics; this numerical-only backward accepts that explicitly."""

from __future__ import annotations


def mla_sm90_attention_grads(grad_y, q_nope, q_pe, c_kv, k_pe, w_uk, w_uv, y, lse, cu_seqlens, max_seqlen, scale):
    """Leaf gradients ``(d_q_nope, d_q_pe, d_c_kv, d_k_pe, d_w_uk, d_w_uv)`` of ``mla_sm90_attention``.

    ``q_nope`` [T, h, nope], ``q_pe`` [T, h, 64], ``c_kv`` [T, 512], ``k_pe`` [T, 64], ``w_uk`` [h, nope, 512],
    ``w_uv`` [h, v, 512] (bf16, packed causal self-attention over ``cu_seqlens``); ``y`` [T, h, v] is the op's
    output and ``lse`` [T, h] (float32, natural log) the forward's logsumexp. Activation grads are bf16 except
    the per-token head sums ``d_c_kv`` and ``d_k_pe`` (float32).
    """
    import torch

    from isoexec.ops.attention.gqa_sm90.ops import varlen_bwd

    T, h, nope = q_nope.shape
    c_h = c_kv.unsqueeze(0).expand(h, -1, -1)                                    # [h, T, 512]
    k_nope = torch.bmm(c_h, w_uk.transpose(1, 2)).transpose(0, 1)                 # [T, h, nope]
    v = torch.bmm(c_h, w_uv.transpose(1, 2)).transpose(0, 1).contiguous()         # [T, h, v]
    q = torch.cat((q_nope, q_pe), -1)
    k = torch.cat((k_nope, k_pe.unsqueeze(1).expand(-1, h, -1)), -1)
    dq, dk, dv = varlen_bwd(grad_y.contiguous(), q, k, v, y.contiguous(), lse.transpose(0, 1).contiguous(),
                            cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, scale, allow_nondeterministic=True)
    dk_nope = dk[..., :nope].transpose(0, 1)                                      # [h, T, nope]
    dv_h = dv.transpose(0, 1)                                                     # [h, T, v]
    d_c = (torch.bmm(dk_nope, w_uk) + torch.bmm(dv_h, w_uv)).sum(0, dtype=torch.float32)
    d_k_pe = dk[..., nope:].sum(1, dtype=torch.float32)
    d_w_uk = torch.bmm(dk_nope.transpose(1, 2), c_h)                              # [h, nope, 512]
    d_w_uv = torch.bmm(dv_h.transpose(1, 2), c_h)                                 # [h, v, 512]
    return dq[..., :nope], dq[..., nope:], d_c, d_k_pe, d_w_uk, d_w_uv
