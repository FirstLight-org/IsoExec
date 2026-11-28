"""The CED program: the encoder-decoder handoff, the decoder global KV and the decoder segments.

The encoder runs on every token. The decoder runs on **decoder segments** (``DecoderSegments``): a segment
``[s, e)`` of one sequence is the set of positions whose decoder rows are computed together; the decoder's
sliding window inside it is truncated at ``s`` on every decoder layer after the boundary (the boundary layer's
own window keys are token-wise in the handoff, so its window is not truncated), while the decoder global KV
(projected from the encoder output by ``decoder.entry`` for every token) is read over the whole causal prefix.

Two programs (``config.CED_PREFILL_MODES``):

* ``bounded_replay`` (release): for a turn whose prompt ends at ``b`` (the first generated position) and whose
  generated tokens end at ``e``, one segment ``[max(0, b - W), e)`` with ``W`` the replay window (the sliding
  window, 128). The rule is a function of the prompt length only: invariant to scheduler chunks, graph mode
  and logprob requests. Engine and trainer evaluate the same segments; the trainer takes ``b`` and ``e`` of
  every turn from the rollout.
* ``full_decoder`` (reference parity only): one segment ``[0, T)`` per sequence (the official reference code).

Nothing here assumes one rank holds a whole sequence: positions are absolute, segments name their sequence
and absolute start, and the handoff values are per token.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from .config import BOUNDED_REPLAY, CED_PREFILL_MODES


def replay_start(prompt_end: int, window: int) -> int:
    """First decoder position of a turn whose prompt ends at ``prompt_end`` (Decoder SWA Bounded Replay)."""
    if prompt_end < 1 or window < 1:
        raise ValueError(f"a turn needs a non-empty prompt and a positive replay window, got {prompt_end}, {window}")
    return max(0, prompt_end - window)


@dataclass(frozen=True)
class Turn:
    """One generation turn of a sequence, in absolute positions: the prompt ends at ``prompt_end`` (the first
    generated position) and the generated tokens end at ``end``."""

    prompt_end: int
    end: int

    def __post_init__(self):
        if not 1 <= self.prompt_end <= self.end:
            raise ValueError(f"turn needs 1 <= prompt_end <= end, got {self.prompt_end}, {self.end}")


@dataclass(frozen=True)
class Segment:
    """Decoder rows of one sequence ``[start, end)`` (absolute positions) computed together, and the position
    ranges ``[lo, hi)`` whose logits score generated tokens (position ``p`` scores token ``p + 1``)."""

    sequence: int
    start: int
    end: int
    scored: tuple[tuple[int, int], ...] = ()

    def scores(self, p: int) -> bool:
        return any(lo <= p < hi for lo, hi in self.scored)

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(eq=False)
class EncoderOutput:
    """THE handoff (report eq. (1), ``H_{L/2}``): the only values that cross the CED boundary, per token.

    ``streams`` ``[T, hc, H]`` BF16: the residual streams leaving the last encoder block (``X_b``), with the
    last encoder FFN expand settled; ``pre`` ``[T, hc]`` FP32: the last encoder FFN's shifted collapse
    weights (``pre_{b-1}``). It is the pipeline payload at the CED cut."""

    streams: torch.Tensor
    pre: torch.Tensor


@dataclass(eq=False)
class DecoderGlobalKV:
    """What ``decoder.entry`` produces for EVERY token and every decoder block reads.

    ``main_kv`` / ``index_k``: the boundary layer's ratio-1 compressed KV and index keys (cache records; the
    trainer holds their dequantized rows), read over the whole causal prefix; ``window_kv``: the boundary
    layer's own sliding-window keys (token-wise in the handoff, so exact for every row); ``rows``: the
    boundary layer's attention input ``h_b`` and attention mHC coefficients (``pre_a``, ``post``, ``comb``) per
    token, gathered at segment rows by the decoder (``DecoderSegments.gather``)."""

    main_kv: Any
    index_k: Any
    window_kv: Any
    rows: dict
    rows_at: dict | None = None      # ``rows`` gathered at the decoder rows (the segment gather G)


@dataclass(eq=False)
class DecoderRows:
    """The decoder rows of one forward, as the decoder kernels take them (R2/R3 in model/REQUESTS.md).

    ``index [R]`` packed token row, ``positions [R]`` absolute, ``sequence [R]`` packed sequence (whose global
    KV the row reads over its whole causal prefix), ``window_floor [R]`` the lowest window position of the
    decoder layers after the boundary (the segment start), ``window_index [R, W]`` (trainer) the decoder row
    of each window slot inside the same segment (-1 empty) or None (engine: the pages and the floor),
    ``cu_seqlens [S + 1]`` the packed token boundaries."""

    index: Any
    positions: Any
    sequence: Any
    window_floor: Any
    window_index: Any
    cu_seqlens: Any

    @staticmethod
    def from_segments(segments: DecoderSegments, window: int, device=None) -> DecoderRows:
        to = (lambda t: t.to(device)) if device is not None else (lambda t: t)
        return DecoderRows(to(segments.rows), to(segments.positions), to(segments.sequence),
                           to(segments.seg_start), to(segments.window_rows(window)),
                           to(torch.tensor(segments.cu_seqlens, dtype=torch.int64)))

    @staticmethod
    def from_plan(plan, positions, token_to_request, cu_seqlens) -> DecoderRows:
        """Engine: the step plan (``engine_ced.CedStepPlan``) over the step's packed tokens (built once per step,
        by ``decoder.entry``). The host rows reach the device through pinned staging, asynchronously: a pageable
        copy would hold the host until the device drained every launch queued before it."""
        dev = positions.device
        if dev.type == "cuda":
            idx, starts = (torch.from_numpy(a).pin_memory().to(dev, non_blocking=True)
                           for a in (plan.rows, plan.segment_start))
        else:
            idx, starts = torch.from_numpy(plan.rows), torch.from_numpy(plan.segment_start)
        seq = token_to_request.index_select(0, idx).long()
        return DecoderRows(idx, positions.index_select(0, idx), seq, starts.index_select(0, seq), None, cu_seqlens)


class DecoderSegments:
    """Which packed rows the decoder computes and where each row's decoder window starts.

    Over a packed batch (``cu_seqlens`` of ``S`` sequences, ``T`` tokens): ``rows [R]`` the packed token row of
    each decoder row (a token may appear in several segments of a multi-turn sequence), ``positions [R]`` its
    absolute position, ``seg_start [R]`` the absolute start of its segment (the window clamp), ``sequence [R]``
    its sequence, ``cu_rows [G + 1]`` segment boundaries inside the decoder rows, ``score [R]`` whether the row's
    logits score a generated token."""

    def __init__(self, segments: Sequence[Segment], cu_seqlens: Sequence[int]):
        self.segments = tuple(segments)
        self.cu_seqlens = tuple(int(c) for c in cu_seqlens)
        rows, positions, starts, seqs, score, cu = [], [], [], [], [], [0]
        for g in self.segments:
            base, length = self.cu_seqlens[g.sequence], self.cu_seqlens[g.sequence + 1] - self.cu_seqlens[g.sequence]
            if not 0 <= g.start < g.end <= length or any(not g.start <= lo <= hi <= g.end for lo, hi in g.scored):
                raise ValueError(f"segment {g} does not fit sequence {g.sequence} of length {length}")
            for p in range(g.start, g.end):
                rows.append(base + p)
                positions.append(p)
                starts.append(g.start)
                seqs.append(g.sequence)
                score.append(g.scores(p))
            cu.append(len(rows))
        self.rows = torch.tensor(rows, dtype=torch.int64)
        self.positions = torch.tensor(positions, dtype=torch.int64)
        self.seg_start = torch.tensor(starts, dtype=torch.int64)
        self.sequence = torch.tensor(seqs, dtype=torch.int64)
        self.score = torch.tensor(score, dtype=torch.bool)
        self.cu_rows = torch.tensor(cu, dtype=torch.int64)

    def __len__(self) -> int:
        return int(self.rows.numel())

    # ------------------------------------------------------------------------------------------ builders
    @staticmethod
    def all_rows(cu_seqlens: Sequence[int], turns: Sequence[Sequence[Turn]] | None = None) -> DecoderSegments:
        """``full_decoder``: one segment ``[0, T_s)`` per sequence, scored at every turn's generated tokens
        (default: every row but the last)."""
        cu = [int(c) for c in cu_seqlens]
        segs = []
        for s in range(len(cu) - 1):
            length = cu[s + 1] - cu[s]
            scored = (((0, length - 1),) if turns is None else
                      tuple((t.prompt_end - 1, t.end - 1) for t in turns[s]))
            segs.append(Segment(s, 0, length, scored))
        return DecoderSegments(segs, cu)

    @staticmethod
    def replay(cu_seqlens: Sequence[int], turns: Sequence[Sequence[Turn]], window: int) -> DecoderSegments:
        """``bounded_replay``: per turn ``k`` of sequence ``s``, the segment ``[max(0, b_k - W), e_k)`` scored
        from ``b_k - 1``. ``turns[s]`` lists the sequence's turns (from the rollout); the segments of one
        sequence may overlap (a later turn replays positions an earlier one generated)."""
        cu = [int(c) for c in cu_seqlens]
        if len(turns) != len(cu) - 1:
            raise ValueError("one turn list per packed sequence")
        segs = []
        for s, seq_turns in enumerate(turns):
            if not seq_turns:
                raise ValueError(f"sequence {s} has no turn")
            for t in seq_turns:
                if t.end > cu[s + 1] - cu[s]:
                    raise ValueError(f"turn {t} exceeds sequence {s}")
                segs.append(Segment(s, replay_start(t.prompt_end, window), t.end, ((t.prompt_end - 1, t.end - 1),)))
        return DecoderSegments(segs, cu)

    @staticmethod
    def for_program(mode: str, cu_seqlens: Sequence[int], turns: Sequence[Sequence[Turn]],
                    window: int) -> DecoderSegments:
        """The segments of program ``mode`` for a rollout described by its turns."""
        if mode not in CED_PREFILL_MODES:
            raise ValueError(f"unknown CED prefill program {mode!r}")
        if mode == BOUNDED_REPLAY:
            return DecoderSegments.replay(cu_seqlens, turns, window)
        return DecoderSegments.all_rows(cu_seqlens, turns)   # one decoder state per token

    # ------------------------------------------------------------------------------------------ uses
    def gather(self, values: torch.Tensor) -> torch.Tensor:
        """Packed per-token ``values [T, ...]`` at the decoder rows (``[R, ...]``)."""
        return values.index_select(0, self.rows.to(values.device))

    def window_rows(self, window: int) -> torch.Tensor:
        """``[R, window]`` decoder-row index of each truncated window slot (slot ``j`` of a row at position ``p``
        holds position ``p - window + 1 + j``), ``-1`` where the position precedes the segment start."""
        R = len(self)
        out = torch.full((R, window), -1, dtype=torch.int64)
        for g in range(len(self.segments)):
            a, z = int(self.cu_rows[g]), int(self.cu_rows[g + 1])
            for r in range(a, z):
                for j in range(window):
                    q = r - window + 1 + j
                    if q >= a:
                        out[r, j] = q
        return out


@dataclass(frozen=True)
class CedState:
    """One state class of the definition: who owns it, its record, its lifetime and whether it crosses calls.

    ``tokens_per_entry``: tokens per record (the compression ratio of a global KV; 1 for windows); ``window``:
    a ring of that many positions (0 = the whole prefix); ``lifetime``: ``request`` (the sequence), ``segment``
    (one decoder segment: rebuilt by every replay, never prefix-cached), ``group`` (until the ratio-r group
    completes), ``forward`` (one call); ``axis``: the CP axis it lives on (absolute positions; a CP rank may hold
    any slice of them, nothing assumes one rank holds the sequence)."""

    name: str
    owner: str
    stack: str
    record_bytes: int
    tokens_per_entry: int
    window: int
    lifetime: str
    crosses_calls: bool
    reuse: str
    axis: str = "token"


    def portable(self):
        """This state as the portable contract declares it (``isoexec.state.contract.StateSpec``)."""
        from isoexec.schema import replicated
        from isoexec.state.contract import StateSpec
        if self.tokens_per_entry == 0 or self.lifetime == "group":
            return StateSpec(self.name, (self.record_bytes,), "u8", replicated(), "per_request")
        windowed = self.window > 0
        return StateSpec(self.name, (self.record_bytes,), "u8", replicated(), "per_token", self.tokens_per_entry,
                         "sliding_window" if windowed else "full_history", self.window if windowed else None,
                         scope="segment" if self.lifetime == "segment" else "request")


def ced_states(cfg) -> tuple[CedState, ...]:
    """The state classes of geometry ``cfg``, record widths from the cache package's declared constants."""
    from isoexec.ops.attention.deepseek_v41_cache_sm100 import spec

    KV, IX, W = spec.KV_RECORD_BYTES, spec.INDEX_RECORD_BYTES, cfg.replay_window
    out = []
    for k in cfg.encoder:
        own = f"encoder.blocks.{k.layer}"
        out.append(CedState("swa_kv", own, "encoder", KV, 1, W, "request", True, "none"))
        if k.mode.value == "full":
            out.append(CedState("global_kv", own, "encoder", KV, k.ratio, 0, "request", True, "all"))
            out.append(CedState("index_k", own, "encoder", IX, k.ratio, 0, "request", True, "all"))
            if k.ratio > 1:
                out.append(CedState("compressor_state", own, "encoder", 2 * 4 * cfg.head_dim, 1, k.ratio - 1,
                                    "group", True, "align"))
    out += [CedState("global_kv", "decoder.entry", "decoder", KV, 1, 0, "request", True, "all"),
            CedState("index_k", "decoder.entry", "decoder", IX, 1, 0, "request", True, "all"),
            CedState("swa_kv", "decoder.entry", "decoder", KV, 1, W, "request", True, "none")]
    for k in cfg.decoder[1:]:
        out.append(CedState("swa_kv", f"decoder.blocks.{k.layer - cfg.boundary}", "decoder", KV, 1, W, "segment",
                            True, "none"))
    out += [CedState("segment_start", "model", "decoder", 8, 0, 0, "segment", True, "none", axis="request"),
            CedState("engram_history", "embed", "encoder", 8 * (cfg.engram_max_ngram_size - 1), 0, 0, "request",
                     True, "none", axis="request")]
    return tuple(out)


#: Context-parallel axes per region (declared now; the CP schedule is designed later): ``(query axis, key axis,
#: values that would cross a cut at absolute position c)``. Every axis is anchored at absolute position 0.
REGION_CP_AXES = {
    'embed': ('token', 'ngram context (previous N - 1 ids)',
        'N - 1 token ids'),
    'encoder.swa': ('token, causal', 'window [i - W + 1, i]',
        'previous W - 1 window records per layer, or per-segment softmax summaries with the sink counted once'),
    'encoder.compressor': ('groups {r j .. r j + r - 1}', '-',
        'the pending token (kv, score) when c is not a multiple of r'),
    'encoder.csa2': ('token, causal', 'compressed entries j < (i + 1) // r',
        ('index scores or per-segment top-k candidates (a degree-invariant global top-k); selected records or '
         'attention summaries')),
    'handoff': ('token', '-',
        'nothing'),
    'decoder.entry': ('token', '-',
        'nothing (ratio 1)'),
    'segment_gather': ('decoder rows', '-',
        'segment start per sequence on every rank'),
    'decoder.swa': ('token, causal', 'window [max(s, i - W + 1), i]',
        'as encoder.swa, with the clamp'),
    'decoder.indexer': ('token, causal', 'all entries j <= i; candidate blocks anchored at 0',
        'block maxima, index scores, candidate block lists'),
    'decoder.reuse_reindex': ('token, causal', "the boundary's pool",
        'selected records or attention summaries'),
    'moe_mhc_norms_head': ('token', '-',
        'nothing'),
}

#: Values of one forward that never cross calls (recomputed by every forward that needs them).
FORWARD_ONLY = ("topk", "candidate_pool", "mhc_carry", "routing", "decoder_rows")


def engine_decoder_rows(prompt_end: int, chunk_start: int, chunk_end: int, window: int, mode: str) -> range:
    """Positions of a prefill chunk ``[chunk_start, chunk_end)`` the engine runs through the decoder: those at or
    after the request's segment start (``bounded_replay``) or all (``full_decoder``). A function of the prompt
    length and the positions only, never of the chunking."""
    start = replay_start(prompt_end, window) if mode == BOUNDED_REPLAY else 0
    return range(max(chunk_start, start), chunk_end)


__all__ = ["FORWARD_ONLY", "REGION_CP_AXES", "CedState", "DecoderGlobalKV", "DecoderRows", "DecoderSegments",
           "EncoderOutput",
           "Segment", "Turn", "ced_states", "engine_decoder_rows", "replay_start"]
