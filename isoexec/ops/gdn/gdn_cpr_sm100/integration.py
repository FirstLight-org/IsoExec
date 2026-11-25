"""Deterministic SM100 CPR forward with FLA's chunked backward."""

from __future__ import annotations

import torch

from isoexec.ops.gdn.gdn_cpr_sm100 import ops
from isoexec.ops.gdn.gdn_cpr_sm90 import fla_backward

#: The implementation identity selected for this family's trainer and engine core.
IMPL = "sm100_cpr_v1"
_OWNED = ("gdn.l2norm", "gdn.gating")


def _load_fla():
    """FLA's backward stages, through the loader the sm90 package owns (arch-neutral Triton)."""
    from isoexec.ops.gdn.gdn_cpr_sm90 import fla_backward

    return fla_backward


def _qk_finish(dq_hv, dk_hv, dk2_hv, qn, kn, q_rstd, k_rstd):
    """Group sums over each key head's value heads, then the l2norm backward, in fp32."""
    T, hv, Dk = dq_hv.shape
    hk = qn.shape[1]
    rep = hv // hk

    def finish(dy_hv, y, rstd):
        dy = dy_hv.float().view(T, hk, rep, Dk).sum(2)
        y = y.float()
        return (rstd.float().unsqueeze(-1) * (dy - y * (dy * y).sum(-1, keepdim=True)))

    dq = finish(dq_hv, qn, q_rstd)
    dk = finish(dk_hv.float() + dk2_hv.float(), kn, k_rstd)
    return dq, dk


def _fla_grads(saved, a, b, A_log, dt_bias, cu_seqlens, do, cu_seqlens_cpu=None):
    """d(q, k, v, a, b, A_log, dt_bias) from FLA's chunked backward on the forward's saves."""
    fla_backward = _load_fla()
    F = fla_backward.load()
    C, scale = ops.C, ops.SCALE
    cu = cu_seqlens.to(torch.int64)
    qn, kn, v, A = (saved[n].unsqueeze(0) for n in ("qn", "kn", "v", "A"))
    a4, b4, do4 = (t.unsqueeze(0).contiguous() for t in (a, b, do))
    A_log, dt_bias = A_log.float().contiguous(), dt_bias.float().contiguous()
    beta = F.fused_beta_sigmoid(b4, scale=1.0)
    cu_cpu = cu.cpu() if cu_seqlens_cpu is None else cu_seqlens_cpu
    chunks = F.prepare_chunk_indices(cu, C, cu_seqlens_cpu=cu_cpu)
    g = F.gdn_gate_chunk_cumsum(g=a4, A_log=A_log, chunk_size=C, scale=F.RCP_LN2, dt_bias=dt_bias, cu_seqlens=cu,
                                chunk_indices=chunks)
    if saved["h"].numel():
        w, v_new = (saved[n].unsqueeze(0) for n in ("w", "v_new"))
        h = saved["h"][: len(chunks)].unsqueeze(0)
    else:
        w, u = F.recompute_w_u_fwd(k=kn, v=v, beta=beta, A=A, g=g, cu_seqlens=cu, chunk_indices=chunks)
        h, v_new, _ = F.chunk_gated_delta_rule_fwd_h(k=kn, w=w, u=u, g=g, initial_state=None,
                                                    output_final_state=False, cu_seqlens=cu, chunk_indices=chunks,
                                                    state_v_first=True, chunk_size=C)
    dv = F.chunk_bwd_dv_local(q=qn, k=kn, g=g, do=do4, scale=scale, cu_seqlens=cu, chunk_indices=chunks,
                              chunk_size=C)
    dh, _, dv = F.chunk_gated_delta_rule_bwd_dhu(q=qn, k=kn, w=w, g=g, h0=None, dht=None, do=do4, dv=dv,
                                                 scale=scale, cu_seqlens=cu, chunk_indices=chunks,
                                                 state_v_first=True, chunk_size=C)
    dq_hv, dk_hv, dw, dg = fla_backward.dqkwg(F, qn, kn, v_new, w, g, h, dv, do4, dh, scale, cu, chunks)
    dk2_hv, dv, db, dg2 = fla_backward.wy_bwd(F, kn, v, beta, A, dw, dv, g, cu, chunks)
    dg.add_(dg2)
    dg = fla_backward.reverse_cumsum(F, dg, cu, chunks)
    dg, dA_log, ddt_bias = F.gdn_gate_bwd(g=a4, A_log=A_log, dt_bias=dt_bias, dyg=dg)
    dq, dk = _qk_finish(dq_hv[0], dk_hv[0], dk2_hv[0], saved["qn"], saved["kn"], saved["q_rstd"], saved["k_rstd"])
    db = F.fused_beta_sigmoid_bwd(b4, db, scale=1.0)
    return dq, dk, dv[0], dg[0], db[0], dA_log, ddt_bias


class _Core(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, a, b, A_log, dt_bias, cu_seqlens, save_states, cu_seqlens_cpu):
        with torch.no_grad():
            o, saved = ops.trainer_forward(q, k, v, a, b, A_log, dt_bias, cu_seqlens, save_states=save_states,
                                           cu_seqlens_cpu=cu_seqlens_cpu)
        ctx.cu_seqlens_cpu = cu_seqlens_cpu
        ctx.names = tuple(saved)
        ctx.save_for_backward(a, b, A_log, dt_bias, cu_seqlens, *saved.values())
        ctx.dtypes = tuple(t.dtype for t in (q, k, v, a, b, A_log, dt_bias))
        return o

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, do):
        a, b, A_log, dt_bias, cu_seqlens, *rest = ctx.saved_tensors
        g = _fla_grads(dict(zip(ctx.names, rest)), a, b, A_log, dt_bias, cu_seqlens, do.to(torch.bfloat16),
                       ctx.cu_seqlens_cpu)
        return (*[gr.to(dt) for gr, dt in zip(g, ctx.dtypes)], None, None, None)


def gdn_cpr_sm100_core(q, k, v, a, b, A_log, dt_bias, cu_seqlens, *, max_seqlen=None, save_states=False,
                       cu_seqlens_cpu=None):
    """Trainer forward (packed varlen, zero initial state) with gradients."""
    del max_seqlen
    if torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in (q, k, v, a, b, A_log, dt_bias)):
        return _Core.apply(q, k, v, a, b, A_log, dt_bias, cu_seqlens, save_states, cu_seqlens_cpu)
    with torch.no_grad():
        return ops.prefill(q, k, v, a, b, A_log, dt_bias, cu_seqlens)
