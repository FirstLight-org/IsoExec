"""Deterministic CPR forward with FLA's chunked backward and fused Q/K gradient reduction.

Backward recomputes chunk states by default; ``save_states=True`` retains them instead.
The scalar reverse gate scan uses the backend's explicit native recipe. Other
FLA stages retain their native implementations; complete backward numerical
qualification is separate from the shared deterministic forward recipe.
"""

from __future__ import annotations

from typing import Any, cast

import torch

from isoexec.ops.gdn.gdn_cpr_sm90 import fla_backward, ops

#: The implementation identity selected for this family's trainer and engine core.
IMPL = "cuda_cpr_v1"
_OWNED = ("gdn.l2norm", "gdn.gating")


_CHUNKS: dict = {}


def chunk_indices(cu: torch.Tensor, cu_seqlens_cpu: torch.Tensor | None, chunk: int = 64) -> torch.Tensor:
    """FLA's ``prepare_chunk_indices`` (int64 [chunks, 2]: sequence, chunk within it) on ``cu``'s device.

    From the host boundaries (the trainer builds them once per micro-batch) the pairs are computed on the
    host and uploaded from pinned memory without a stream synchronization; every GDN layer of the
    micro-batch (forward recompute and backward alike) then reuses the one upload, keyed by the host
    tensor. Without host boundaries FLA's own derivation runs, which reads the lengths back (a sync).
    """
    if cu_seqlens_cpu is None:
        return fla_backward.load().prepare_chunk_indices(cu, chunk, cu_seqlens_cpu=cu.cpu())
    key = (id(cu_seqlens_cpu), cu_seqlens_cpu._version, chunk, cu.device)
    hit = _CHUNKS.get(key)
    if hit is not None and hit[0]() is cu_seqlens_cpu:
        return hit[1]
    import weakref

    import numpy as np

    host = cu_seqlens_cpu.numpy().astype(np.int64)
    counts = (np.diff(host) + chunk - 1) // chunk
    seq = np.repeat(np.arange(counts.size, dtype=np.int64), counts)
    within = np.arange(seq.size, dtype=np.int64) - np.repeat(np.cumsum(counts) - counts, counts)
    pinned = torch.from_numpy(np.stack([seq, within], 1)).pin_memory()
    out = pinned.to(cu.device, non_blocking=True)
    _CHUNKS.clear()   # one micro-batch at a time; the pinned source lives as long as its upload is cached
    _CHUNKS[key] = (weakref.ref(cu_seqlens_cpu), out, pinned)
    return out


def _fla_grads(saved, a, b, A_log, dt_bias, cu_seqlens, do, cu_seqlens_cpu=None):
    """d(q, k, v, a, b, A_log, dt_bias): FLA's ``chunk_gated_delta_rule_bwd`` kernels on the forward's saved
    intermediates (``ops.trainer_forward``), then the group sums + l2norm, gate and beta-sigmoid backward."""
    F = fla_backward.load()
    C, scale = 64, ops.SCALE
    cu = cu_seqlens.to(torch.int64)
    qn, kn, v, A = (saved[n].unsqueeze(0) for n in ("qn", "kn", "v", "A"))
    a4, b4, do4 = (t.unsqueeze(0).contiguous() for t in (a, b, do))
    A_log, dt_bias = A_log.float().contiguous(), dt_bias.float().contiguous()
    beta = F.fused_beta_sigmoid(b4, scale=1.0)
    chunks = chunk_indices(cu, cu_seqlens_cpu, C)
    g = F.gdn_gate_chunk_cumsum(g=a4, A_log=A_log, chunk_size=C, scale=F.RCP_LN2, dt_bias=dt_bias, cu_seqlens=cu,
                                chunk_indices=chunks)
    if saved["h"].numel():
        w, v_new = (saved[n].unsqueeze(0) for n in ("w", "v_new"))
        h = saved["h"][:len(chunks)].unsqueeze(0)
    else:   # lean saves: FLA's own recompute of w, u and the chunk states
        w, u = F.recompute_w_u_fwd(k=kn, v=v, beta=beta, A=A, g=g, cu_seqlens=cu, chunk_indices=chunks)
        h, v_new, _ = F.chunk_gated_delta_rule_fwd_h(k=kn, w=w, u=u, g=g, initial_state=None, output_final_state=False,
                                                     cu_seqlens=cu, chunk_indices=chunks, state_v_first=True,
                                                     chunk_size=C)
    dv = F.chunk_bwd_dv_local(q=qn, k=kn, g=g, do=do4, scale=scale, cu_seqlens=cu, chunk_indices=chunks, chunk_size=C)
    dh, _, dv = F.chunk_gated_delta_rule_bwd_dhu(q=qn, k=kn, w=w, g=g, h0=None, dht=None, do=do4, dv=dv, scale=scale,
                                                 cu_seqlens=cu, chunk_indices=chunks, state_v_first=True, chunk_size=C)
    dq_hv, dk_hv, dw, dg = fla_backward.dqkwg(F, qn, kn, v_new, w, g, h, dv, do4, dh, scale, cu, chunks)
    dk2_hv, dv, db, dg2 = fla_backward.wy_bwd(F, kn, v, beta, A, dw, dv, g, cu, chunks)
    dg.add_(dg2)
    dg = fla_backward.reverse_cumsum(F, dg, cu, chunks)
    dg, dA_log, ddt_bias = F.gdn_gate_bwd(g=a4, A_log=A_log, dt_bias=dt_bias, dyg=dg)
    # group sums over each key head's value heads and the l2norm backward, one pass (csrc/backward.cu)
    dq, dk = cast(Any, ops.extension()).bwd_qk_finish(dq_hv[0], dk_hv[0], dk2_hv[0], saved["qn"], saved["kn"], saved["q_rstd"],
                                           saved["k_rstd"])
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
    @cast(Any, torch.autograd.function.once_differentiable)
    def backward(ctx: Any, *grad_outputs: torch.Tensor):
        (do,) = grad_outputs
        a, b, A_log, dt_bias, cu_seqlens, *rest = ctx.saved_tensors
        g = _fla_grads(dict(zip(ctx.names, rest)), a, b, A_log, dt_bias, cu_seqlens, do.to(torch.bfloat16),
                       ctx.cu_seqlens_cpu)
        return (*[gr.to(dt) for gr, dt in zip(g, ctx.dtypes)], None, None, None)


def gdn_cpr_sm90_core(q, k, v, a, b, A_log, dt_bias, cu_seqlens, *, max_seqlen=None, save_states=False,
                       cu_seqlens_cpu=None):
    """Trainer forward (packed varlen, zero initial state) with gradients. Returns [T, hv, 128] bf16.

    ``max_seqlen`` is accepted for caller compatibility and is not needed. By default, backward
    recomputes w, u, chunk states and deltas; ``save_states=True`` retains w, h and v_new instead.
    ``cu_seqlens_cpu`` carries the trainer's once-per-batch host boundaries through forward and
    backward, enabling measured prefill plans without a per-layer device-to-host copy.
    """
    del max_seqlen
    if torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in (q, k, v, a, b, A_log, dt_bias)):
        return _Core.apply(q, k, v, a, b, A_log, dt_bias, cu_seqlens, save_states, cu_seqlens_cpu)
    with torch.no_grad():
        return ops.prefill(q, k, v, a, b, A_log, dt_bias, cu_seqlens, cu_seqlens_cpu=cu_seqlens_cpu)
