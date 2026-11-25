"""Point-to-point transport and autograd primitives for contiguous context parallelism.

Context-parallel hand-offs only ever flow from a lower rank to a higher rank in the forward (key/value
prefixes, recurrent state, convolution tails) and back in the backward (their cotangents). Every
message is matched by construction: both ends derive its existence and shape from the same
:class:`~isoexec.ops.collectives.cp_contiguous.ContiguousPlan`, and both issue the messages of one
peer pair in the same program order. Sends never block the sender's compute; a receive makes only
the consumer wait. Rank ``r`` can therefore run layer ``L`` while rank ``r - 1`` runs ``L + 1``
(the wavefront), and no collective sits in the context-parallel forward.

Transports:

* :class:`DistTransport` -- ``torch.distributed`` ``isend``/``irecv`` on the CP process group (NCCL
  on GPUs, multi-node; gloo on CPU).
* :class:`MailboxTransport` -- an in-process mailbox for simulating ``cp`` ranks one after another
  in a single process (bitwise checks of any degree on one device).

Activation recompute. A checkpointed region's first pass runs in ``record`` mode: every tensor it
receives is kept on the microbatch's plan. Its recompute (inside the backward) runs in ``replay``
mode: no message is sent or received in the forward direction, the recorded tensors are reused.
Without this the recompute would re-run the forward hand-offs inside the backward, where rank
``r - 1`` reaches region ``k`` only after its backward of ``k + 1`` waits on rank ``r``: the ranks
would serialize. The backward hand-offs are never recorded.
"""

from __future__ import annotations

import contextlib
import contextvars
from collections import deque
from typing import Any, Callable

import torch

#: ``None`` (plain), ``"record"`` (a checkpoint region's first pass) or ``"replay"`` (its recompute).
_MODE: contextvars.ContextVar[str | None] = contextvars.ContextVar("isoexec_cp_mode", default=None)


@contextlib.contextmanager
def cp_region(mode: str | None):
    """Run a block as a checkpoint region's first pass (``record``) or its recompute (``replay``)."""
    if mode not in (None, "record", "replay"):
        raise ValueError(f"cp_region mode must be None, 'record' or 'replay', got {mode!r}")
    token = _MODE.set(mode)
    try:
        yield
    finally:
        _MODE.reset(token)


def cp_mode() -> str | None:
    return _MODE.get()


class CpTransportError(RuntimeError):
    """A context-parallel message did not match its plan."""


class DistTransport:
    """``isend``/``irecv`` over a CP process group, addressed by group rank."""

    def __init__(self, group) -> None:
        import torch.distributed as dist

        self.group = group
        self.cp = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        # Sends stay referenced until complete: the payload must outlive the transfer.
        self._pending: deque = deque()

    def _prune(self) -> None:
        while self._pending and self._pending[0][0].is_completed():
            self._pending.popleft()

    def send(self, tensor: torch.Tensor, dst: int) -> None:
        import torch.distributed as dist

        payload = tensor.contiguous()
        work = dist.isend(payload, group=self.group, group_dst=dst)
        self._pending.append((work, payload))
        self._prune()

    def recv(self, shape, dtype, device, src: int) -> torch.Tensor:
        buffer = torch.empty(shape, dtype=dtype, device=device)
        self.recv_into(buffer, src)
        return buffer

    def recv_into(self, buffer: torch.Tensor, src: int) -> None:
        import torch.distributed as dist

        if not buffer.is_contiguous():
            raise CpTransportError("context-parallel receive buffers must be contiguous")
        dist.irecv(buffer, group=self.group, group_src=src).wait()

    def flush(self) -> None:
        """Wait for every outstanding send (CPU transports; GPU sends are stream-ordered)."""
        while self._pending:
            work, _ = self._pending.popleft()
            work.wait()

    def warmup(self, device) -> None:
        """Create every peer pair's communicator before the first layer, in one global order.

        ``send``/``recv_into`` are unbatched ``isend``/``irecv``. Under lazy NCCL initialisation each one
        creates its pair's own 2-rank communicator on first use, and both ranks of the pair must create it
        together. In a layer's exchange every rank first sends to its higher peers while the top rank first
        receives from rank 0, so at CP >= 3 with no other traffic first (an attention-only model) the pair
        creations wait on each other in a cycle and the forward hangs. Here every rank visits the pairs
        (i, j), i < j, in lexicographic order -- one total order, so no cycle can form -- with the same
        unbatched ops the exchange uses. (``batch_isend_irecv`` would run on the group's own communicator
        and leave the pair communicators uncreated.)
        """
        import torch.distributed as dist

        token = torch.zeros(1, dtype=torch.float32, device=device)
        sink = torch.empty(1, dtype=torch.float32, device=device)
        for low in range(self.cp):
            for high in range(low + 1, self.cp):
                if self.rank == low:
                    dist.isend(token, group=self.group, group_dst=high).wait()
                elif self.rank == high:
                    dist.irecv(sink, group=self.group, group_src=low).wait()
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)


class MailboxTransport:
    """One simulated rank of an in-process CP world; all ranks of a world share ``box``.

    Ranks run one after another: a receive must find its message already sent (forward in
    ascending rank order, backward in descending order).
    """

    def __init__(self, box: dict, cp: int, rank: int) -> None:
        self.box, self.cp, self.rank = box, cp, rank

    def send(self, tensor: torch.Tensor, dst: int) -> None:
        self.box.setdefault((self.rank, dst), deque()).append(tensor.detach().clone())

    def recv(self, shape, dtype, device, src: int) -> torch.Tensor:
        queue = self.box.get((src, self.rank))
        if not queue:
            raise CpTransportError(f"rank {self.rank}: no message from rank {src} (simulated ranks must run in "
                                   "dependency order)")
        tensor = queue.popleft()
        if tuple(tensor.shape) != tuple(shape) or tensor.dtype != dtype:
            raise CpTransportError(f"rank {self.rank}: message from {src} is {tuple(tensor.shape)} {tensor.dtype}, "
                                   f"expected {tuple(shape)} {dtype}")
        return tensor.to(device)

    def recv_into(self, buffer: torch.Tensor, src: int) -> None:
        buffer.copy_(self.recv(buffer.shape, buffer.dtype, buffer.device, src))

    def flush(self) -> None:
        pass

    def warmup(self, device) -> None:
        pass


def _stash_get(plan, key):
    try:
        return plan.stash.pop(key)  # one replay per record: release it as the backward walks back
    except KeyError:
        raise CpTransportError(f"context-parallel replay of {key!r}: nothing was recorded in this microbatch's "
                               "first pass") from None


def received(plan, key, receive: Callable[[], Any]):
    """``receive()`` in plain/record mode (recording the result), the recorded value in replay."""
    mode = cp_mode()
    if mode == "replay":
        return _stash_get(plan, key)
    value = receive()
    if mode == "record":
        plan.stash[key] = value
    return value


def forward_sends_enabled() -> bool:
    """Forward-direction sends are skipped in replay: the receiver replays its recorded copy."""
    return cp_mode() != "replay"


class _RecvFromPrev(torch.autograd.Function):
    """Forward: receive a tensor from ``rank - 1``. Backward: send its cotangent back to it.

    ``anchor`` is any tensor that requires grad (it gives the node a place in the graph so the
    backward runs); it receives no gradient.
    """

    @staticmethod
    def forward(ctx, anchor, shape, dtype, plan, transport, key):  # noqa: D102
        src = plan.rank - 1

        def receive():
            return transport.recv(shape, dtype, anchor.device, src)

        value = received(plan, key, receive)
        ctx.transport, ctx.src = transport, src
        return value

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad):  # noqa: D102
        ctx.transport.send(grad, ctx.src)
        return None, None, None, None, None, None


class _SendToNext(torch.autograd.Function):
    """Forward: send ``value`` to ``rank + 1`` and return ``carrier`` unchanged. Backward: receive
    ``value``'s cotangent from ``rank + 1``. ``carrier`` must be on the loss path."""

    @staticmethod
    def forward(ctx, carrier, value, plan, transport):  # noqa: D102
        dst = plan.rank + 1
        if forward_sends_enabled():
            transport.send(value.detach(), dst)
        ctx.transport, ctx.dst = transport, dst
        ctx.value_meta = (tuple(value.shape), value.dtype, value.device)
        return carrier.view_as(carrier)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad):  # noqa: D102
        shape, dtype, device = ctx.value_meta
        value_grad = ctx.transport.recv(shape, dtype, device, ctx.dst)
        return grad, value_grad, None, None


def recv_from_prev(anchor: torch.Tensor, shape, dtype, plan, transport, key) -> torch.Tensor:
    """A tensor from ``rank - 1`` whose cotangent flows back to it (see :class:`_RecvFromPrev`)."""
    if torch.is_grad_enabled() and anchor.requires_grad:
        return _RecvFromPrev.apply(anchor, tuple(shape), dtype, plan, transport, key)
    return received(plan, key, lambda: transport.recv(tuple(shape), dtype, anchor.device, plan.rank - 1))


def send_to_next(carrier: torch.Tensor, value: torch.Tensor, plan, transport) -> torch.Tensor:
    """Send ``value`` to ``rank + 1``; ``carrier`` (returned) brings its cotangent back."""
    if torch.is_grad_enabled() and (carrier.requires_grad or value.requires_grad):
        return _SendToNext.apply(carrier, value, plan, transport)
    if forward_sends_enabled():
        transport.send(value.detach(), plan.rank + 1)
    return carrier


__all__ = [
    "CpTransportError",
    "DistTransport",
    "MailboxTransport",
    "cp_mode",
    "cp_region",
    "forward_sends_enabled",
    "received",
    "recv_from_prev",
    "send_to_next",
]
