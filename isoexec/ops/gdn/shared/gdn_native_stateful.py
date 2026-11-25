"""The native recurrent GDN composition with a state hand-off: an initial state for the first packed
sequence and the final state of the last one.

This is the context-parallel form of ``gdn_ops.gdn_native_core``: the same engine kernel
(``fused_sigmoid_gating_delta_rule_update``, l2norm and gating in-kernel) over the same scratch pool,
except that the first sequence's pool row holds the fp32 state the previous rank ended with, and the
last sequence's final state is written back to its row. The kernel loads the initial state into its
fp32 register tile and then performs exactly the per-token arithmetic it performs when the earlier
tokens are in the same call, so a sequence cut at any token is exact (the property engine chunked
prefill already relies on).

State wire format: ``[HV, V, K]`` fp32, the kernel's pool-row layout.

Backward: FLA's chunk VJP with the initial state as a differentiable input and the final state's
cotangent as an output gradient, chained through the eager rebuild of the in-kernel l2norm/gating
(the trainer's FLA backward, ``ISOEXEC_GDN_FLA_BACKWARD``). Correct and deterministic, not a bitwise
twin of anything (the contract covers forward values only).
"""

from __future__ import annotations

from typing import Sequence

import torch


def _scratch(q, v, lengths: Sequence[int], h0, want_last: bool):
    """A pool with the hand-off rows and a stride-stable index grid (as ``_recurrent_scratch_state``)."""
    n = len(lengths)
    tmax = max(lengths)
    hv, vdim, kdim = v.shape[-2], v.shape[-1], q.shape[-1]
    ssm = q.new_zeros(n + 1, hv, vdim, kdim, dtype=torch.float32)
    if h0 is not None:
        if tuple(h0.shape) != (hv, vdim, kdim) or h0.dtype != torch.float32:
            raise ValueError(f"gdn state hand-off: initial state must be fp32 {(hv, vdim, kdim)}, got "
                             f"{tuple(h0.shape)} {h0.dtype}")
        ssm[1].copy_(h0)
    pad = max(64, 1 << (tmax - 1).bit_length())
    host = torch.zeros(n, pad, dtype=torch.int32, device="cpu")  # built on the host, one upload
    host[:, 0] = torch.arange(1, n + 1, dtype=torch.int32)
    if want_last:
        # The kernel stores the state after token t wherever the index is positive: row n keeps the
        # state after the last token of the last sequence (its token-0 store is overwritten).
        host[n - 1, lengths[-1] - 1] = n
    idx = host.to(device=q.device, non_blocking=True)
    return ssm, idx[:, :tmax]


def stateful_forward(q, k, v, a, b, A_log, dt_bias, *, cu_seqlens, lengths: Sequence[int], h0=None,
                     want_last: bool = False):
    """``(o [T, HV, V], h_last [HV, V, K] fp32 | None)`` for raw ``q, k [T, H, K]``, ``v [T, HV, V]``,
    ``a, b [T, HV]``. ``lengths`` are the host lengths of ``cu_seqlens``."""
    from isoexec.ops.gdn.shared.gdn_ops import gdn_native_core_kernel

    with torch.no_grad():
        ssm, idx = _scratch(q, v, lengths, h0, want_last)
        o = gdn_native_core_kernel(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), a, b, A_log, dt_bias,
                                   ssm_state=ssm, state_indices=idx, cu_seqlens=cu_seqlens)
    o = o.squeeze(0) if o.dim() == 4 else o
    return o, (ssm[len(lengths)].clone() if want_last else None)


def stateful_vjp(q, k, v, a, b, A_log, dt_bias, h0, do, dht, cu_seqlens, n: int):
    """Gradients of ``stateful_forward`` w.r.t. ``(q, k, v, a, b, A_log, dt_bias, h0)``."""
    from isoexec.ops.gdn.shared.gdn_fla_backward import get_fla_chunk_backward

    chunk = get_fla_chunk_backward()
    hv = v.shape[-2]
    with torch.enable_grad():
        leaves = [t.detach().requires_grad_(True) for t in (q, k, a, b, A_log, dt_bias)]
        ql, kl, al, bl, Al, dl = leaves
        qn = (ql.float() * torch.rsqrt((ql.float() ** 2).sum(-1, keepdim=True) + 1e-6)).to(q.dtype)
        kn = (kl.float() * torch.rsqrt((kl.float() ** 2).sum(-1, keepdim=True) + 1e-6)).to(k.dtype)
        if q.shape[-2] != hv:
            rep = hv // q.shape[-2]
            qn = qn.repeat_interleave(rep, dim=-2)
            kn = kn.repeat_interleave(rep, dim=-2)
        g = (-torch.exp(Al.float()) * torch.nn.functional.softplus(al.float() + dl.float())).unsqueeze(0)
        beta = bl.sigmoid().unsqueeze(0)
        # FLA's state layout is [N, H, K, V]; the kernel pool row is [HV, V, K].
        kdim, vdim = q.shape[-1], v.shape[-1]
        state0 = torch.zeros(n, hv, kdim, vdim, dtype=torch.float32, device=v.device)
        if h0 is not None:
            state0[0].copy_(h0.transpose(-1, -2))
        state0.requires_grad_(h0 is not None)
        inner = [t.detach().requires_grad_(True) for t in (qn, kn, v.unsqueeze(0), g, beta)]
        o, ht = chunk(inner[0].unsqueeze(0), inner[1].unsqueeze(0), inner[2], inner[3], inner[4], scale=None,
                      initial_state=state0 if h0 is not None else None, output_final_state=dht is not None,
                      cu_seqlens=cu_seqlens, use_qk_l2norm_in_kernel=False, use_beta_sigmoid_in_kernel=False)
        outs, cots = [o], [do.unsqueeze(0).to(o.dtype)]
        if dht is not None:
            last = torch.zeros_like(ht)
            last[n - 1].copy_(dht.transpose(-1, -2))
            outs.append(ht)
            cots.append(last)
        wrt = [inner[0], inner[1], inner[2], inner[3], inner[4]] + ([state0] if h0 is not None else [])
        grads = torch.autograd.grad(outs, wrt, cots, allow_unused=True)
        dqn, dkn, dv, dg, dbeta = (gr if gr is not None else torch.zeros_like(t) for gr, t in zip(grads[:5], wrt[:5]))
        dh0 = grads[5][0].transpose(-1, -2).contiguous() if h0 is not None else None
        prep = torch.autograd.grad(
            [qn, kn, g, beta], leaves,
            [dqn.to(qn.dtype), dkn.to(kn.dtype), dg.to(g.dtype), dbeta.to(beta.dtype)],
            allow_unused=True,
        )
    prep = [torch.zeros_like(t) if gr is None else gr for gr, t in zip(prep, leaves)]
    return (prep[0].to(q.dtype), prep[1].to(k.dtype), dv.squeeze(0).to(v.dtype), prep[2].to(a.dtype),
            prep[3].to(b.dtype), prep[4].to(A_log.dtype), prep[5].to(dt_bias.dtype), dh0)


class _NativeStateful(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, a, b, A_log, dt_bias, h0, cu_seqlens, lengths, want_last):  # noqa: D102
        o, ht = stateful_forward(q, k, v, a, b, A_log, dt_bias, cu_seqlens=cu_seqlens, lengths=lengths, h0=h0,
                                 want_last=want_last)
        ctx.save_for_backward(q, k, v, a, b, A_log, dt_bias, *(() if h0 is None else (h0,)))
        ctx.has_h0, ctx.want_last, ctx.cu, ctx.n = h0 is not None, want_last, cu_seqlens, len(lengths)
        if ht is None:
            ht = o.new_zeros(0, dtype=torch.float32)
        return o, ht

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, do, dht):  # noqa: D102
        saved = ctx.saved_tensors
        q, k, v, a, b, A_log, dt_bias = saved[:7]
        h0 = saved[7] if ctx.has_h0 else None
        if not ctx.want_last:
            dht = None
        elif dht is None:
            dht = torch.zeros(v.shape[-2], v.shape[-1], q.shape[-1], dtype=torch.float32, device=v.device)
        if do is None:
            do = torch.zeros_like(v)
        grads = stateful_vjp(q, k, v, a, b, A_log, dt_bias, h0, do, dht, ctx.cu, ctx.n)
        return (*grads, None, None, None)


def gdn_native_stateful(q, k, v, a, b, A_log, dt_bias, *, cu_seqlens, lengths: Sequence[int], h0=None,
                        want_last: bool = False):
    """Differentiable :func:`stateful_forward`; ``h_last`` is ``None`` unless ``want_last``."""
    tensors = (q, k, v, a, b, A_log, dt_bias, h0)
    needs_grad = torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in tensors)
    if not needs_grad:
        return stateful_forward(q, k, v, a, b, A_log, dt_bias, cu_seqlens=cu_seqlens, lengths=lengths, h0=h0,
                                want_last=want_last)
    o, ht = _NativeStateful.apply(q, k, v, a, b, A_log, dt_bias, h0, cu_seqlens, tuple(lengths), bool(want_last))
    return o, (ht if want_last else None)


__all__ = ["gdn_native_stateful", "stateful_forward", "stateful_vjp"]
