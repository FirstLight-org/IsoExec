"""The CED program on vLLM 0.33's Model Runner V2: which rows of a step run the decoder, and why chunking
cannot change it.

Request lifecycle. A request enters with its prompt (``NewRequestData.prompt_len``, held by the runner in
``req_states.prompt_len``); its decoder segment starts at ``s = max(0, prompt_len - W)`` under
``bounded_replay`` (``W`` = the replay window, 128) and at 0 under ``full_decoder``. Every step, for each
scheduled request with ``c`` computed tokens and ``q`` scheduled tokens (positions ``[c, c + q)``):

* the encoder and ``decoder.entry`` run on all ``q`` rows (the decoder global KV, index keys and the boundary
  layer's window keys are written for every token);
* the decoder runs on the rows at positions ``>= s`` (``plan_step``): a prefill chunk entirely below ``s``
  runs no decoder row, a decode step always runs its row.

Invariant (chunk independence). The union over the steps of a request of its decoder rows is exactly
``[s, end)``, each position once, for every chunking the scheduler chooses, because ``s`` depends on the
prompt length only and each row is computed from causal state (window keys of positions ``>= s`` of this
request, the global KV of every earlier position) that no step boundary changes. A preempted request that
is recomputed (``prefill_len > prompt_len``) keeps ``s`` from ``prompt_len`` and recomputes the same rows.
Tests: ``test_ced_engine_cpu.py`` drives the CPU reference engine with these plans under random schedules.

Not admitted under ``bounded_replay`` (named refusals, ``refuse_request``): prompt logprobs or prompt-token
logprob requests (rows below ``s`` have no decoder state); returned routed experts (they would cover rows the
decoder never ran). Multi-turn: each turn is a fresh request (prefix caching is refused), so it has its own
segment; earlier turns' decoder windows go with their requests.

Capture plan (FULL_DECODE_ONLY, then FULL_AND_PIECEWISE):
* FULL_DECODE_ONLY: a decode-only step has ``plan.all_rows`` (every row is at ``>= prompt_len > s``), so the
  whole forward is one graph per capture size, decoder rows == tokens; no row selection is captured. The
  segment floor never binds there either: a decode row at ``p >= prompt_len`` reads ``[p - W + 1, p]`` and
  ``p - W + 1 > prompt_len - W = s``, so captured decode is the unclamped program and reads no per-request
  segment start. The floor binds only on prefill rows (positions in ``[s, prompt_len)`` and recomputed rows).
* FULL_AND_PIECEWISE: the encoder regions and ``decoder.entry`` are piecewise at the step's token count; the
  break sits at the handoff / entry -> segment gather (G), which is eager (it reads the host plan); the decoder
  regions run at the decoder row count ``R`` padded to a decoder bucket. ``R <= sum_r min(q_r, W) + decode
  rows``: at most ``W`` = 128 prefill rows per prefilling request. Warm-up / capture sizes for the decoder
  regions are the decoder buckets (1..max_num_seqs * 128 bounded by the chunk size), separate from the
  encoder's token buckets.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import BOUNDED_REPLAY, CED_PREFILL_MODES, FULL_DECODER


@dataclass(frozen=True)
class CedStepPlan:
    """The decoder rows of one engine step.

    ``rows [R]`` packed token rows (ascending), ``cu_rows [num_reqs + 1]`` per-request boundaries inside
    ``rows``, ``segment_start [num_reqs]`` absolute segment start of each request (the window floor of the
    decoder layers after the boundary), ``all_rows``: every token of the step is a decoder row."""

    rows: np.ndarray
    cu_rows: np.ndarray
    segment_start: np.ndarray
    all_rows: bool
    tokens: int = -1          # the step's scheduled token count (a plan is used by the forward of that step only)
    dummy: bool = False       # vLLM's dummy step (profiling / warm-up / capture): no request behind it

    @property
    def count(self) -> int:
        return int(self.rows.size)


def segment_starts(prompt_len: np.ndarray, window: int, mode: str) -> np.ndarray:
    if mode not in CED_PREFILL_MODES:
        raise ValueError(f"unknown CED prefill program {mode!r}")
    prompt_len = np.asarray(prompt_len, dtype=np.int64)
    if (prompt_len < 1).any():
        raise ValueError("every request needs a non-empty prompt")
    if mode == FULL_DECODER:
        return np.zeros_like(prompt_len)
    return np.maximum(prompt_len - window, 0)


def plan_step(prompt_len, num_computed, num_scheduled, *, window: int, mode: str) -> CedStepPlan:
    """Host plan of one step from the runner's per-request numpy arrays (no device read)."""
    prompt_len = np.asarray(prompt_len, dtype=np.int64)
    computed = np.asarray(num_computed, dtype=np.int64)
    scheduled = np.asarray(num_scheduled, dtype=np.int64)
    if not prompt_len.shape == computed.shape == scheduled.shape or (scheduled < 0).any() or (computed < 0).any():
        raise ValueError("per-request prompt lengths, computed and scheduled counts must align")
    starts = segment_starts(prompt_len, window, mode)
    q0 = np.concatenate(([0], np.cumsum(scheduled)))
    first = np.clip(starts - computed, 0, scheduled)              # leading rows of the step below s
    rows = [np.arange(q0[i] + first[i], q0[i + 1]) for i in range(len(scheduled))]
    cu = np.concatenate(([0], np.cumsum([len(r) for r in rows])))
    flat = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)
    return CedStepPlan(flat.astype(np.int64), cu.astype(np.int64), starts.astype(np.int64),
                       bool(flat.size == q0[-1]), int(q0[-1]))


def dummy_plan(num_scheduled) -> CedStepPlan:
    """The plan of one of vLLM's dummy steps (memory profiling, eager warm-up, graph warm-up and capture): no
    request stands behind it (``InputBatch.make_dummy``: idx_mapping 0..n-1 over unset request slots, positions 0,
    nothing computed, prefill lengths 0). Convention: every scheduled token is a decoder row with segment start 0,
    i.e. the decode-step program over all rows. It is the largest decoder workload a step of that token count can
    have (a real step's decoder rows are a subset), so profiling measures an upper bound and warm-up / capture
    exercise the shapes a decode step runs."""
    scheduled = np.asarray(num_scheduled, dtype=np.int64)
    T = int(scheduled.sum())
    return CedStepPlan(np.arange(T, dtype=np.int64), np.concatenate(([0], np.cumsum(scheduled))).astype(np.int64),
                       np.zeros(scheduled.size, dtype=np.int64), True, T, True)


def dummy_split(num_reqs: int, num_tokens: int) -> list[int]:
    """Scheduled tokens per dummy request, as ``InputBatch.make_dummy`` splits them (extra tokens last)."""
    base, extra = divmod(num_tokens, num_reqs)
    return [base + (i >= num_reqs - extra) for i in range(num_reqs)]


def is_dummy_batch(input_batch) -> bool:
    """vLLM 0.33 V2 marks no batch as dummy; ``InputBatch.make_dummy`` sets every request's ``prefill_len`` to 0,
    while a real request always has ``prefill_len >= prompt_len >= 1`` (``RequestState.prefill_len``)."""
    n = input_batch.num_reqs
    return n > 0 and not np.asarray(input_batch.prefill_len_np[:n]).any()


#: Request-id prefixes of vLLM 0.33's own kernel warm-up requests (``v1/worker/gpu/warmup.py``: ``_warmup_{i}_``
#: with ``SamplingParams.for_sampler_warmup()``, which asks for prompt logprobs, and ``_v2_mixed_warmup_*``). Their
#: outputs are discarded and their prompts are a few tokens (segment start 0: every prompt row is a decoder row).
VLLM_WARMUP_PREFIXES = ("_warmup_", "_v2_mixed_warmup")


def refuse_request(mode: str, *, prompt_logprobs: bool, prompt_logprob_token_ids: bool,
                   return_routed_experts: bool, req_id: str = "") -> None:
    """Named refusals of a request the program cannot serve as asked (vLLM's own warm-up requests excepted)."""
    if mode != BOUNDED_REPLAY or req_id.startswith(VLLM_WARMUP_PREFIXES):
        return
    if prompt_logprobs or prompt_logprob_token_ids:
        raise ValueError("CED_PROMPT_LOGPROBS: under bounded_replay the decoder has no state for prompt rows "
                         "before the replay window; prompt logprobs exist only in the full_decoder program")
    if return_routed_experts:
        raise ValueError("CED_ROUTED_EXPERTS: under bounded_replay the decoder's routing exists only for its "
                         "segment rows")


_CURRENT: dict = {"plan": None}


def publish_plan(plan: CedStepPlan) -> None:
    """The plan of the step being prepared (read by ``decoder.entry`` / the segment gather of that forward)."""
    _CURRENT["plan"] = plan


def current_plan() -> CedStepPlan | None:
    return _CURRENT["plan"]


def plan_from_runner(input_batch, req_states, *, window: int, mode: str) -> CedStepPlan:
    """``plan_step`` from vLLM 0.33 V2's ``InputBatch`` and ``RequestState`` (host arrays only)."""
    n = input_batch.num_reqs
    if is_dummy_batch(input_batch):
        return dummy_plan(input_batch.num_scheduled_tokens[:n])
    idx = input_batch.idx_mapping_np[:n]
    return plan_step(req_states.prompt_len.np[idx], input_batch.num_computed_tokens_np[:n],
                     input_batch.num_scheduled_tokens[:n], window=window, mode=mode)


__all__ = ["VLLM_WARMUP_PREFIXES", "CedStepPlan", "current_plan", "dummy_plan", "dummy_split", "is_dummy_batch", "plan_from_runner", "plan_step",
           "publish_plan", "refuse_request", "segment_starts"]
