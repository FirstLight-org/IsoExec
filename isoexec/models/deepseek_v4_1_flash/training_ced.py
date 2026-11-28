"""Trainer side of the CED program: what the rollout carries, how a batch becomes decoder segments, and
where the scored logits go.

What the rollout must carry, per trained sequence (``RolloutTurns``): for every engine request that produced
part of it, in order, the request's prompt length (``len(RequestOutput.prompt_token_ids)``, which is the
absolute position of its first generated token, because each turn's prompt is the whole conversation so far,
token in, token out) and its generated-token count (``len(CompletionOutput.token_ids)``). These are the
``Turn``s ``(prompt_end, end = prompt_end + generated)``. The loss mask alone does not determine them: two
turns with no observation tokens between them would form one run, and a run's start is the prompt length only
when the engine prompt is the exact token prefix of the trained sequence. A single-turn rollout may derive its
turn from the prompt length and the response length (``single_turn``); a multi-turn rollout without explicit
turns is refused (``CED_TURNS_REQUIRED``).

Batch layouts: SkyRL hands the trainer padded rows (left-padded prompt + right-padded response, an attention
mask); ``pack_padded`` packs the real tokens (positions restart at 0 per sequence) and keeps the map back to
the padded response columns. Packed batches (``cu_seqlens``) are used as they are.

Scoring: ``DecoderSegments.for_program(mode, cu_seqlens, turns, window)`` gives the decoder rows; the logits at
the rows that score a generated token (``segments.score``) produce that token's log-probability, scattered
back to its response column; every other column (observation tokens of a multi-turn rollout, padding) is 0 and
outside the loss mask. Backward runs through the gathered decoder rows into ``decoder.entry`` (the decoder
global KV, straight-through over the cache quantizers) and the encoder; the indexer has no gradient.
"""
from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from .program import DecoderSegments, Turn


@dataclass(frozen=True)
class RolloutTurns:
    """Per trained sequence, the engine requests that generated it: ``(prompt_len, generated)`` in order. A
    request with no generated token (a padding row: SkyRL's dummy rows carry no response) is a turn ending at its
    prompt: its segment runs the decoder on the last ``W`` prompt positions, as the engine's prefill of such a
    request would, and scores nothing."""

    requests: tuple[tuple[tuple[int, int], ...], ...]

    def turns(self) -> list[list[Turn]]:
        out = []
        for s, reqs in enumerate(self.requests):
            seq, end = [], 0
            for prompt_len, generated in reqs:
                if prompt_len < max(1, end) or generated < 0:
                    raise ValueError(f"sequence {s}: turn ({prompt_len}, {generated}) does not follow the previous "
                                     f"turn ending at {end}")
                seq.append(Turn(prompt_len, prompt_len + generated))
                end = prompt_len + generated
            out.append(seq)
        return out


def single_turn(prompt_lens: Sequence[int], response_lens: Sequence[int], loss_mask: torch.Tensor | None = None
                ) -> RolloutTurns:
    """One engine request per sequence: the turn is ``(prompt_len, prompt_len + response_len)``. With a loss
    mask ``[B, num_actions]``, a mask with more than one run of generated tokens is refused (multi-turn rollouts
    must carry their turns)."""
    if loss_mask is not None:
        m = loss_mask.to(torch.int8)
        runs = ((m[:, 1:] == 1) & (m[:, :-1] == 0)).sum(1) + (m[:, 0] == 1)
        if (runs > 1).any():
            raise ValueError("CED_TURNS_REQUIRED: a multi-turn rollout must carry each turn's prompt length "
                             "(the engine requests' prompt_token_ids lengths)")
    return RolloutTurns(tuple(((int(p), int(r)),) for p, r in zip(prompt_lens, response_lens)))


@dataclass
class PackedBatch:
    """Real tokens of a padded batch, packed: ``ids [T]``, ``cu_seqlens [B + 1]``, and for each sequence the
    padded column of its first response token (``response_col``) and its prompt length."""

    ids: torch.Tensor
    cu_seqlens: list[int]
    prompt_lens: list[int]
    response_lens: list[int]


def pack_padded(input_ids: torch.Tensor, attention_mask: torch.Tensor, num_actions: int) -> PackedBatch:
    """SkyRL's padded rows ``[B, P + num_actions]`` (left-padded prompt, right-padded response) -> packed."""
    B, width = input_ids.shape
    P = width - num_actions
    mask = attention_mask.bool()
    ids, cu, prompts, responses = [], [0], [], []
    for b in range(B):
        prompt = mask[b, :P]
        response = mask[b, P:]
        if prompt.any() and not prompt[int(prompt.long().argmax()):].all():
            raise ValueError(f"row {b}: the prompt must be left-padded (contiguous real tokens)")
        r = int(response.sum())
        if not response[:r].all():
            raise ValueError(f"row {b}: the response must be right-padded")
        p = int(prompt.sum())
        ids.append(torch.cat([input_ids[b, P - p:P], input_ids[b, P:P + r]]))
        cu.append(cu[-1] + p + r)
        prompts.append(p)
        responses.append(r)
    return PackedBatch(torch.cat(ids), cu, prompts, responses)


def score(logits: torch.Tensor, segments: DecoderSegments, ids: torch.Tensor, logprob_fn) -> torch.Tensor:
    """Log-probabilities of the tokens the scored decoder rows predict: ``[R_scored]`` in segment order."""
    rows = torch.nonzero(segments.score).squeeze(-1)
    targets = ids[segments.rows[rows] + 1]
    return logprob_fn(logits[rows], targets)


def scatter_to_response(values: torch.Tensor, segments: DecoderSegments, batch: PackedBatch,
                        num_actions: int) -> torch.Tensor:
    """Scored log-probabilities -> ``[B, num_actions]`` response columns (0 where nothing is scored)."""
    out = torch.zeros((len(batch.prompt_lens), num_actions), dtype=values.dtype)
    rows = torch.nonzero(segments.score).squeeze(-1)
    for v, r in zip(values, rows.tolist()):
        b = int(segments.sequence[r])
        column = int(segments.positions[r]) + 1 - batch.prompt_lens[b]
        if not 0 <= column < num_actions:
            raise ValueError(f"scored position of sequence {b} falls outside its response")
        out[b, column] = v
    return out


def segments_for_forward(mode: str, cu_seqlens: Sequence[int], window: int) -> DecoderSegments:
    """The decoder segments of a trainer forward over the packing ``cu_seqlens``: those set explicitly
    (``decoder_segments``) or, otherwise, derived from the rollout requests the trainer integration carries
    (``isoexec.schema.rollout``) by the same rule as the engine (``engine_ced.segment_starts``: a function of
    each request's prompt length only). ``full_decoder`` needs neither."""
    from isoexec.schema import rollout

    from .config import FULL_DECODER
    cu = tuple(int(c) for c in cu_seqlens)
    explicit = _SEGMENTS.get()
    if explicit is not None:
        return explicit
    carried = rollout.current()
    if carried is None:
        if mode == FULL_DECODER:
            return DecoderSegments.all_rows(cu)
        raise ValueError("CED_SEGMENTS_MISSING: a trainer forward of the bounded_replay program needs the rollout's "
                         "decoder segments (training_ced.decoder_segments) or the engine requests behind each "
                         "sequence (isoexec.schema.rollout.carry, set by the trainer integration)")
    if carried.cu_seqlens != cu:
        raise ValueError(f"CED_ROLLOUT_PACKING: the carried rollout requests describe the packing "
                         f"{carried.cu_seqlens[:8]}..., the forward packs {cu[:8]}...")
    return DecoderSegments.for_program(mode, cu, RolloutTurns(carried.requests).turns(), window)


_SEGMENTS: contextvars.ContextVar = contextvars.ContextVar("isoexec_v41_decoder_segments", default=None)


@contextlib.contextmanager
def decoder_segments(segments: DecoderSegments):
    """The decoder segments of the trainer forwards run inside (``decoder.entry`` reads them)."""
    token = _SEGMENTS.set(segments)
    try:
        yield segments
    finally:
        _SEGMENTS.reset(token)


def current_segments() -> DecoderSegments | None:
    return _SEGMENTS.get()


__all__ = ["PackedBatch", "RolloutTurns", "current_segments", "decoder_segments", "pack_padded", "scatter_to_response",
           "score", "segments_for_forward", "single_turn"]
