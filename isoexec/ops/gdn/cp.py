"""Contiguous context parallelism for the GDN mixer: conv-tail and recurrent-state hand-off.

Rank ``r``'s first local piece continues a sequence from rank ``r - 1`` whenever the shard boundary
cuts a sequence (:class:`~isoexec.ops.collectives.cp_contiguous.ContiguousPlan`). Two values cross
that link, and each makes rank ``r`` compute exactly what a CP=1 forward computes for its rows:

* **conv tail** -- the last ``min(w - 1, rows before the boundary)`` pre-conv rows of the sequence.
  Rank ``r`` prepends them to its first piece and drops their outputs, so every output row reads the
  same taps with the same sequence-start masking.
* **recurrent state** -- the fp32 state ``[HV, V, K]`` after the sequence's last token on rank
  ``r - 1``, loaded as the initial state of rank ``r``'s first piece (``gdn_native_stateful``).

The tail is sent as soon as the in-projection exists and the state as soon as the core finishes, so
rank ``r`` runs layer ``L`` while rank ``r - 1`` runs ``L + 1``. In the backward the cotangents of both
travel the reverse chain. The core is any *stateful* core ``(…, cu_seqlens, lengths, h0, want_last)
-> (o, h_last)``; a chunk-aligned (CPR) state hand-off fits the same interface.
"""

from __future__ import annotations

from typing import Callable

import torch

from isoexec.ops.collectives.cp_contiguous import require_plan
from isoexec.ops.collectives.cp_transport import recv_from_prev, send_to_next

#: The movement these wrappers perform (``collectives.cp_state``).
CP_STATE_IMPL = "contiguous_p2p_state_chain"


def _lazy(factory: Callable[[], object]) -> Callable[[], object]:
    state: dict = {}

    def get():
        if "t" not in state:
            state["t"] = factory()
        return state["t"]

    return get


def make_cp_conv(conv: Callable[..., torch.Tensor], *, transport_factory: Callable[[], object], cp: int, rank: int,
                 site: str) -> Callable[..., torch.Tensor]:
    """Wrap a packed causal convolution ``conv(x [T, C], w [C, 1, W], cu_seqlens)`` with the tail hand-off."""
    transport = _lazy(transport_factory)
    key = (site, "conv_tail")

    def cp_conv(x, w, cu_seqlens):
        plan = require_plan(cu_seqlens, cp=cp, rank=rank, site=site)
        t = transport()
        h_in, h_out = plan.halo(int(w.shape[-1]))
        rows = x.shape[0]
        if h_out and h_out <= rows:  # the outgoing tail is local: send before waiting on anything
            x = send_to_next(x, x[rows - h_out:], plan, t)
        if h_in:
            halo = recv_from_prev(x, (h_in, *x.shape[1:]), x.dtype, plan, t, key)
            ext = torch.cat((halo, x), dim=0)
            cu = plan.halo_cu_tensor(int(w.shape[-1]), x.device)
        else:
            ext, cu = x, cu_seqlens
        if h_out > rows:  # this rank's rows are fewer than the tail: it passes part of its halo on
            ext = send_to_next(ext, ext[ext.shape[0] - h_out:], plan, t)
        y = conv(ext, w, cu)
        return y[h_in:] if h_in else y

    return cp_conv


def make_cp_gdn_core(stateful: Callable[..., tuple], *, transport_factory: Callable[[], object], cp: int, rank: int,
                     site: str) -> Callable[..., torch.Tensor]:
    """Wrap a stateful core into the ``gdn.core`` signature ``core(q, k, v, a, b, A_log, dt_bias, cu_seqlens)``."""
    transport = _lazy(transport_factory)
    key = (site, "state")

    def cp_core(q, k, v, a, b, A_log, dt_bias, cu_seqlens, *, cu_seqlens_cpu=None, **unused):
        del cu_seqlens_cpu, unused  # the plan holds the host boundaries
        plan = require_plan(cu_seqlens, cp=cp, rank=rank, site=site)
        t = transport()
        h0 = None
        if plan.recv_prev:
            shape = (v.shape[-2], v.shape[-1], q.shape[-1])
            h0 = recv_from_prev(v, shape, torch.float32, plan, t, key)
        lengths = [b_ - a_ for a_, b_ in zip(plan.local_cu, plan.local_cu[1:])]
        o, h_last = stateful(q, k, v, a, b, A_log, dt_bias, cu_seqlens=cu_seqlens, lengths=lengths, h0=h0,
                             want_last=plan.send_next)
        if plan.send_next:
            o = send_to_next(o, h_last, plan, t)
        return o

    return cp_core


def native_stateful_core() -> Callable[..., tuple]:
    """The production stateful core of the native recurrent composition."""
    from isoexec.ops.gdn.shared.gdn_native_stateful import gdn_native_stateful

    return gdn_native_stateful


__all__ = ["CP_STATE_IMPL", "make_cp_conv", "make_cp_gdn_core", "native_stateful_core"]
