"""Full-row evidence between Tinker's sampling and model-pass requests."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from collections.abc import Iterator, Sequence

import numpy as np

from isoexec.checks import Refusal


MAX_PENDING_FULL_LOGPROB_BYTES = 8 * 1024**3


@dataclass(frozen=True)
class SampleRows:
    """Independently produced engine rows for one exact prompt and response."""

    model_id: str
    version: int
    prompt: tuple[int, ...]
    response: tuple[int, ...]
    rows: np.ndarray
    sampled: np.ndarray

    @property
    def key(self) -> tuple[str, tuple[int, ...]]:
        return self.model_id, self.prompt + self.response

    @property
    def nbytes(self) -> int:
        return self.rows.nbytes + self.sampled.nbytes


class TinkerState:
    """One full-parameter model, its acknowledged sampler version and pending rows.

    The native Tinker engine drains sampling coroutines before every model pass,
    weight mutation or unload. Reserve/commit contain no awaits, so concurrent
    sampling coroutines cannot oversubscribe this store's byte limit.
    """

    def __init__(self, byte_limit: int = MAX_PENDING_FULL_LOGPROB_BYTES):
        if type(byte_limit) is not int or byte_limit < 1:
            raise ValueError("Tinker full-logprob byte limit must be a positive integer")
        self.byte_limit = byte_limit
        self.version = 0
        self.synced_version: int | None = None
        self.checkpoint_ids: set[str] = set()
        self.completed_backward_calls = 0
        self.compared_rows_per_sample: tuple[int, ...] | None = None
        self.failed = False
        self._rows: dict[tuple[str, tuple[int, ...]], SampleRows] = {}
        self._bytes = 0
        self._reserved = 0

    def require_healthy(self) -> None:
        if self.failed:
            raise Refusal("IsoExec Tinker model pass/update failed; unload and recreate the model before continuing")

    def acknowledge_sync(self, checkpoint_id: str) -> None:
        self.require_healthy()
        if self._reserved:
            raise Refusal("Cannot acknowledge a weight transfer while Tinker sampling is in flight")
        self.synced_version = self.version
        self.checkpoint_ids.add(checkpoint_id)

    def require_sampler(self, checkpoint_id: str) -> int:
        self.require_healthy()
        if self.synced_version != self.version or checkpoint_id not in self.checkpoint_ids:
            raise Refusal("Full comparison requires sampling from the current acknowledged save_weights_for_sampler")
        return self.version

    @contextmanager
    def reserve(self, nbytes: int) -> Iterator[None]:
        self.require_healthy()
        if nbytes < 0 or self._bytes + self._reserved + nbytes > self.byte_limit:
            raise Refusal("Pending Tinker full-logprob rows exceed the 8 GiB diagnostic bound; "
                          "complete the current training/update before requesting more samples")
        self._reserved += nbytes
        try:
            yield
        finally:
            self._reserved -= nbytes

    def commit(self, records: Sequence[SampleRows], reservation: int) -> None:
        """Publish a complete sample request, or retain none of its new rows."""
        self.require_healthy()
        if reservation < 0 or reservation > self._reserved:
            raise Refusal("Tinker sample rows must have an active byte reservation")
        added: dict[tuple[str, tuple[int, ...]], SampleRows] = {}
        for record in records:
            if record.version != self.version or self.synced_version != self.version:
                raise Refusal("Weights changed while Tinker sampling was in flight")
            if (record.rows.dtype != np.float32 or record.rows.ndim != 2
                    or record.rows.shape[0] != len(record.response)
                    or record.sampled.dtype != np.float32 or record.sampled.shape != (len(record.response),)):
                raise Refusal("Tinker sampling requires FP32 rows aligned to every response token")
            if not record.response:
                continue  # No trainable response or row bytes; do not accumulate zero-byte cache entries.
            previous = added.get(record.key, self._rows.get(record.key))
            if previous is not None:
                if (previous.prompt != record.prompt or previous.rows.shape != record.rows.shape
                        or not np.array_equal(previous.rows.view(np.uint32), record.rows.view(np.uint32))
                        or not np.array_equal(previous.sampled.view(np.uint32), record.sampled.view(np.uint32))):
                    raise Refusal("Identical Tinker token sequences have ambiguous full-logprob evidence")
                continue
            added[record.key] = record
        size = sum(record.nbytes for record in added.values())
        if self._bytes + self._reserved - reservation + size > self.byte_limit:
            raise Refusal("Decoded Tinker full-logprob rows exceed the reserved byte budget")
        for record in added.values():
            record.rows.flags.writeable = False
            record.sampled.flags.writeable = False
        self._rows.update(added)
        self._bytes += size

    def find(self, model_id: str, tokens: Sequence[int], *, required: bool) -> SampleRows | None:
        self.require_healthy()
        result = self._rows.get((model_id, tuple(tokens)))
        if result is None or result.version != self.version:
            if required:
                raise Refusal("Full comparison requires the exact prompt/response sampled at the current weights; "
                              "no matching Tinker rollout rows are available")
            return None
        return result

    def require_update(self) -> None:
        self.require_healthy()
        if self._reserved:
            raise Refusal("Cannot update weights while Tinker sampling is in flight")
        if self.completed_backward_calls < 1:
            raise Refusal("IsoExec Tinker optim_step requires a successfully compared forward_backward request")

    def updated(self) -> None:
        self.version += 1
        self.synced_version = None
        self.checkpoint_ids.clear()
        self.completed_backward_calls = 0
        self.clear_rows()

    def fail(self) -> None:
        self.failed = True
        self.completed_backward_calls = 0
        self.clear_rows()

    def clear_rows(self) -> None:
        self._rows.clear()
        self._bytes = 0
        self.compared_rows_per_sample = None
