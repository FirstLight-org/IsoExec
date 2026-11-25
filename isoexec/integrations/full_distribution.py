"""Bit-preserving full-logprob payloads shared by framework adapters."""

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


def full_comparison_enabled(model: Mapping[str, Any] | None) -> bool:
    """Read the shared IsoExec option without accepting truthy non-boolean values."""
    if model is None:
        return False
    if not isinstance(model, Mapping):
        raise ValueError("isoexec must be a mapping")
    enabled = model.get("full_logprob_comparison", False)
    if type(enabled) is not bool:
        raise ValueError("isoexec.full_logprob_comparison must be a boolean")
    return enabled

try:
    import pybase64 as _base64
except ImportError:
    import base64 as _base64


@dataclass(frozen=True)
class FullVocabularyEntry:
    """One vocabulary entry of a :class:`FullVocabularyRow`, shaped like an engine logprob record."""

    logprob: float
    rank: int | None = None
    decoded_token: str | None = None


class FullVocabularyRow(Mapping):
    """One response position's complete vocabulary logprobs as a float32 array indexed by token ID.

    Engines report such a row as one Python object per vocabulary entry. For a 248,320-token
    vocabulary that costs tens of megabytes per generated token against the megabyte the values
    need. This mapping serves ``row[token_id].logprob`` from the array instead.
    """

    __slots__ = ("_values", "sampled_rank", "sampled_token_id")

    def __init__(self, values: np.ndarray, sampled_token_id: int, sampled_rank: int | None = None):
        if not isinstance(values, np.ndarray) or values.dtype != np.dtype("<f4") or values.ndim != 1:
            raise ValueError("a full vocabulary row must be a one-dimensional little-endian float32 array")
        if values.size == 0 or not 0 <= sampled_token_id < values.size:
            raise ValueError("a full vocabulary row needs a sampled token inside its vocabulary")
        # Not named ``values``: that would shadow the mapping method of the same name.
        self._values = values
        self.sampled_token_id = sampled_token_id
        self.sampled_rank = sampled_rank

    def __len__(self) -> int:
        return int(self._values.size)

    def __iter__(self) -> Iterator[int]:
        # The engine's own container lists the sampled token first; readers that take the first
        # entry rely on it. The rest follow by token ID.
        yield self.sampled_token_id
        for token_id in range(len(self)):
            if token_id != self.sampled_token_id:
                yield token_id

    def __contains__(self, token_id: object) -> bool:
        return bool(
            isinstance(token_id, (int, np.integer))
            and not isinstance(token_id, bool)
            and 0 <= token_id < len(self)
        )

    def __getitem__(self, token_id: int) -> FullVocabularyEntry:
        if token_id not in self:
            raise KeyError(token_id)
        rank = self.sampled_rank if token_id == self.sampled_token_id else None
        return FullVocabularyEntry(float(self._values[token_id]), rank)

    @property
    def sampled_logprob(self) -> float:
        return float(self._values[self.sampled_token_id])


def full_vocabulary_row(token_ids: np.ndarray, logprobs: np.ndarray) -> FullVocabularyRow:
    """Reorder one sampled-first, rank-ordered engine row into token-ID order without changing a bit.

    ``token_ids[0]`` is the sampled token and ``token_ids[1:]`` must name every vocabulary entry
    exactly once; anything else is refused so an incomplete row can never be packed as complete.
    """
    if not isinstance(token_ids, np.ndarray) or not isinstance(logprobs, np.ndarray):
        raise ValueError("a full vocabulary row needs NumPy token IDs and logprobs")
    if token_ids.ndim != 1 or token_ids.size < 2 or logprobs.shape != token_ids.shape:
        raise ValueError("a full vocabulary row needs the sampled entry plus one entry per vocabulary token")
    if token_ids.dtype.kind not in "iu" or logprobs.dtype != np.dtype("<f4"):
        raise ValueError("a full vocabulary row needs integer token IDs and float32 logprobs")
    vocab_size = token_ids.size - 1
    order = token_ids[1:]
    if int(order.min()) < 0 or int(order.max()) >= vocab_size:
        raise ValueError("a full vocabulary row names a token outside its vocabulary")
    # Length and range checks don't guarantee each token_ids is named exactly once; check explicitly.
    seen = np.zeros(vocab_size, dtype=bool)
    seen[order] = True
    if not seen.all():
        raise ValueError("a full vocabulary row must name every vocabulary token exactly once")
    values = np.empty(vocab_size, dtype="<f4")
    values[order] = logprobs[1:]
    sampled = int(token_ids[0])
    if not 0 <= sampled < vocab_size:
        raise ValueError("a full vocabulary row samples a token outside its vocabulary")
    # Use slices to keep the <f4 bytes; a scalar's tobytes() will follow the host byte order.
    if values[sampled : sampled + 1].tobytes() != logprobs[:1].tobytes():
        raise ValueError("the sampled logprob bits differ from its vocabulary entry")
    # For rank derivation, please refer to:
    # https://github.com/vllm-project/vllm/blob/38dc8ee50055622ab0d5fe6bb32a5aee526bcbf5/vllm/logprobs.py#L178-L185
    sampled_rank = int(np.flatnonzero(order == sampled)[0]) + 1
    return FullVocabularyRow(values, sampled, sampled_rank)


def full_logprob_rows(token_ids: list[int], resp_logprobs: Any, vocab_size: int) -> np.ndarray:
    """Complete per-token vocabulary logprobs as a little-endian float32 ``[num_tokens, vocab_size]`` array."""
    if resp_logprobs is None or len(resp_logprobs) != len(token_ids):
        raise ValueError("full logprobs must contain one row per response token")
    if vocab_size <= 0 or any(tid < 0 or tid >= vocab_size for tid in token_ids):
        raise ValueError("invalid vocabulary or response token IDs for full logprobs")
    rows = np.empty((len(token_ids), vocab_size), dtype="<f4")
    for index, row in enumerate(resp_logprobs):
        if row is None or len(row) != vocab_size:
            raise ValueError("full logprobs must contain every vocabulary token")
        if isinstance(row, FullVocabularyRow):
            rows[index] = row._values
            continue
        try:
            rows[index] = [row[tid].logprob for tid in range(vocab_size)]
        except (KeyError, AttributeError, TypeError, ValueError) as exc:
            raise ValueError("invalid full logprob row") from exc
    if not np.isfinite(rows).all():
        raise ValueError("full logprobs must be finite")
    return rows


def pack_full_logprobs(token_ids: list[int], resp_logprobs: Any, vocab_size: int) -> dict[str, Any]:
    """Pack complete per-token vocabulary logprobs as little-endian float32."""
    rows = full_logprob_rows(token_ids, resp_logprobs, vocab_size)
    return {
        "data": _base64.b64encode(memoryview(rows)).decode("ascii"),
        "shape": list(rows.shape),
        "dtype": "float32",
    }


def decode_packed_full_logprobs(payload: dict[str, Any]) -> np.ndarray:
    """Decode and validate a packed full-logprob matrix."""
    if not isinstance(payload, dict):
        raise ValueError("full_logprobs must be a packed object")
    try:
        shape = tuple(payload["shape"])
        dtype = payload["dtype"]
        decode = getattr(_base64, "b64decode_as_bytearray", None)
        data = (
            decode(payload["data"], validate=True)
            if decode is not None
            else bytearray(_base64.b64decode(payload["data"], validate=True))
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid packed full_logprobs payload") from exc
    if dtype != "float32":
        raise ValueError(f"invalid packed full_logprobs dtype: {dtype!r}")
    if len(shape) != 2 or any(type(dim) is not int or dim < 0 for dim in shape) or shape[1] == 0:
        raise ValueError(f"invalid packed full_logprobs shape: {shape}")
    if len(data) != math.prod(shape) * 4:
        raise ValueError("packed full_logprobs byte count does not match shape")
    rows = np.frombuffer(data, dtype="<f4").reshape(shape)
    if not np.isfinite(rows).all():
        raise ValueError("full logprobs must be finite")
    return rows
