"""Contiguous context-parallel causal attention: key/value prefixes to the query's rank.

Each rank computes attention for its own query rows only. Its first local sequence may have started
on lower ranks; those ranks send it that sequence's key/value rows (the *prefix*, the minimum data a
query can see), and the rank runs the arch's varlen core over ``[prefix | local]`` keys with its
queries right-aligned to their key range. That is the chunked-prefill case of the causal varlen core:
every query row is computed by the same kernel over the same keys, with key segments anchored at the
sequence's first key, so CP=k forward bits equal CP=1 bits by construction. No partial softmax
crosses a rank and no alignment of shard boundaries to key segments is needed.

Backward: the core's own backward on the local rows. The prefix rows' key/value cotangents go back to
their owners, which add them to their own in fp32, in ascending contributor rank, and round once
(a deterministic, standard context-parallel backward; not bitwise CP-invariant, which the contract
does not ask of gradients).
"""

from __future__ import annotations

from typing import Callable

import torch

from isoexec.ops.collectives.cp_contiguous import ContiguousPlan, require_plan
from isoexec.ops.collectives.cp_transport import forward_sends_enabled, received

#: The movement this core performs, recorded on the ``cp_gather`` slot and in the core's ``owns``.
CP_ATTENTION_IMPL = "contiguous_p2p_prefix_fwd_ranked_sum_bwd"


def _send_prefix_rows(k, v, plan: ContiguousPlan, transport) -> None:
    for t in plan.kv_send:
        transport.send(k[t.src_start:], t.dst)
        transport.send(v[t.src_start:], t.dst)


def _receive_prefix(k, v, plan: ContiguousPlan, transport) -> tuple[torch.Tensor, torch.Tensor]:
    prefix = plan.prefix
    kp = torch.empty((prefix, *k.shape[1:]), dtype=k.dtype, device=k.device)
    vp = torch.empty((prefix, *v.shape[1:]), dtype=v.dtype, device=v.device)
    for t in plan.kv_recv:  # ascending source rank = ascending key position
        transport.recv_into(kp[t.dst_start:t.dst_start + t.rows], t.src)
        transport.recv_into(vp[t.dst_start:t.dst_start + t.rows], t.src)
    return kp, vp


def exchange_forward(k, v, plan: ContiguousPlan, transport, key) -> tuple[torch.Tensor, torch.Tensor]:
    """``[prefix | local]`` keys and values for this rank; sends this rank's rows to higher ranks first."""
    if forward_sends_enabled():
        _send_prefix_rows(k, v, plan, transport)
    if plan.prefix == 0:
        return k, v
    kp, vp = received(plan, key, lambda: _receive_prefix(k, v, plan, transport))
    return torch.cat((kp, k), dim=0), torch.cat((vp, v), dim=0)


class _KvPrefixExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, k, v, plan, transport, key):  # noqa: D102
        ctx.plan, ctx.transport = plan, transport
        return exchange_forward(k, v, plan, transport, key)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dk_ext, dv_ext):  # noqa: D102
        plan, transport = ctx.plan, ctx.transport
        prefix = plan.prefix
        # Prefix cotangents go home first: lower ranks wait on them, this rank waits on nobody yet.
        for t in plan.kv_recv:
            rows = slice(t.dst_start, t.dst_start + t.rows)
            transport.send(dk_ext[rows], t.src)
            transport.send(dv_ext[rows], t.src)
        dk, dv = dk_ext[prefix:], dv_ext[prefix:]
        if plan.kv_send:
            dk, dv = dk.float(), dv.float()  # fresh fp32 accumulators (the slices above are not written)
            for t in plan.kv_send:  # ascending contributor rank
                dk[t.src_start:] += transport.recv((t.rows, *dk.shape[1:]), dk_ext.dtype, dk.device, t.dst).float()
                dv[t.src_start:] += transport.recv((t.rows, *dv.shape[1:]), dv_ext.dtype, dv.device, t.dst).float()
            dk, dv = dk.to(dk_ext.dtype), dv.to(dv_ext.dtype)
        return dk.contiguous(), dv.contiguous(), None, None, None


def make_cp_attention_core(core_qk: Callable[..., torch.Tensor], *, transport_factory: Callable[[], object],
                           cp: int, rank: int, site: str) -> Callable[..., torch.Tensor]:
    """Wrap a varlen causal core with separate query/key boundaries into the trainer core signature
    ``core(q, k, v, cu_seqlens, max_q, scale)``, where ``cu_seqlens`` is the local boundary tensor a
    :class:`ContiguousPlan` is registered on.

    ``core_qk(q, k, v, cu_q, cu_k, max_q, max_k, scale)`` must right-align each sequence's queries to
    its keys (bottom-right causal), as FA3's varlen forward does.
    """
    state: dict = {}

    def transport():
        if "t" not in state:
            state["t"] = transport_factory()
        return state["t"]

    key = (site, "kv")

    def core(q, k, v, cu_seqlens, max_q, scale):
        del max_q  # the plan's local maximum (and key maximum) are exact
        plan = require_plan(cu_seqlens, cp=cp, rank=rank, site=site)
        t = transport()
        if plan.kv_send or plan.prefix:
            if torch.is_grad_enabled() and (k.requires_grad or v.requires_grad):
                k, v = _KvPrefixExchange.apply(k, v, plan, t, key)
            else:
                k, v = exchange_forward(k, v, plan, t, key)
        tensors = plan.device_tensors(q.device)
        return core_qk(q, k, v, tensors["local_cu"], tensors["key_cu"], plan.max_q, plan.max_k, scale)

    core.cp_layout = "contiguous"
    return core


__all__ = ["CP_ATTENTION_IMPL", "exchange_forward", "make_cp_attention_core"]
