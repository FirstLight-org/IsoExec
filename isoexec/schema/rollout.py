"""The engine requests behind each packed trainer sequence (model-neutral carrier).

A model whose program depends on how a sequence was generated (DeepSeek V4.1's CED decoder segments start at
``max(0, prompt_len - W)`` of each engine request) reads, during a trainer forward, which engine requests
produced each packed sequence: per sequence, in order, ``(prompt_len, generated)`` with ``prompt_len`` the
request's prompt length (``len(RequestOutput.prompt_token_ids)``, the absolute position of its first
generated token) and ``generated`` its generated-token count. The trainer integration derives them from the
batch it already holds and carries them around each forward (``carry``); components that read them declare
``Component.READS_ROLLOUT_REQUESTS`` so the integration installs the carrier only where it is read.

Standard library only (the schema rule): no torch.
"""
from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Iterator, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class RolloutRequests:
    """``requests[s]``: the engine requests of packed sequence ``s``, ``((prompt_len, generated), ...)`` in order;
    ``cu_seqlens``: the packed boundaries of the forward they describe (alignment padding included); ``lengths``:
    the real (unpadded) token count of each sequence."""

    requests: tuple[tuple[tuple[int, int], ...], ...]
    cu_seqlens: tuple[int, ...]
    lengths: tuple[int, ...]

    def __post_init__(self):
        n = len(self.cu_seqlens) - 1
        if n < 1 or len(self.requests) != n or len(self.lengths) != n:
            raise ValueError(f"rollout requests describe {len(self.requests)} sequences, lengths "
                             f"{len(self.lengths)}, packing {n}")
        for s, (reqs, real) in enumerate(zip(self.requests, self.lengths)):
            padded = self.cu_seqlens[s + 1] - self.cu_seqlens[s]
            if not 1 <= real <= padded:
                raise ValueError(f"sequence {s}: {real} real tokens in a packed slot of {padded}")
            end = 0
            for prompt_len, generated in reqs:
                if prompt_len < max(1, end) or generated < 0 or prompt_len + generated > real:
                    raise ValueError(f"sequence {s}: request ({prompt_len}, {generated}) does not follow the "
                                     f"previous request ending at {end} inside {real} real tokens")
                end = prompt_len + generated

    @staticmethod
    def single(prompt_lens: Sequence[int], generated: Sequence[int], cu_seqlens: Sequence[int],
               lengths: Sequence[int] | None = None) -> RolloutRequests:
        """One engine request per sequence."""
        lengths = [int(p) + int(g) for p, g in zip(prompt_lens, generated)] if lengths is None else lengths
        return RolloutRequests(tuple(((int(p), int(g)),) for p, g in zip(prompt_lens, generated)),
                               tuple(int(c) for c in cu_seqlens), tuple(int(n) for n in lengths))


_CURRENT: contextvars.ContextVar[RolloutRequests | None] = contextvars.ContextVar(
    "isoexec_rollout_requests", default=None)


@contextlib.contextmanager
def carry(requests: RolloutRequests | None) -> Iterator[RolloutRequests | None]:
    """The rollout requests of the trainer forward run inside."""
    token = _CURRENT.set(requests)
    try:
        yield requests
    finally:
        _CURRENT.reset(token)


def current() -> RolloutRequests | None:
    return _CURRENT.get()


__all__ = ["RolloutRequests", "carry", "current"]
