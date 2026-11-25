"""The engine requests behind each trained sequence, carried from SkyRL's microbatches into the trainer forward.

A stage that reads them (``MegatronModel.reads_rollout_requests``: DeepSeek V4.1's CED decoder segments) gets,
around each microbatch forward, ``isoexec.schema.rollout.RolloutRequests`` derived from the microbatch SkyRL
already holds; other stages are untouched. Pairing follows the auxiliary-loss injector (``aux_loss.py``): the
stage's ``forward_context`` is set for one schedule and consumes the microbatches in order, each forward
checked against its own packing.

SkyRL's batch layout (``skyrl/train/dataset/preprocess.py`` ``convert_prompts_responses_to_batch_tensors``):
each row is ONE left-padded block ``[pad | prompt | response]``; ``attention_mask`` marks the real tokens and
``response_mask [B, num_actions]`` (right-aligned) the response tokens. The prompt is the engine request's
token-in prompt (``prompt_token_ids``), so per row ``prompt_len = real - response`` and ``generated =
response``: exactly the numbers the engine's planner keys on (``engine_ced.plan_from_runner`` reads
``req_states.prompt_len``). The rule is by prompt length only.

Single-turn only: one engine request per row (the V4.1 recipe's generator is ``generator.batched=true``,
single-turn by contract). A loss mask with more than one run of generated tokens is a multi-turn rollout whose
turn boundaries SkyRL does not carry: refused (``CED_TURNS_REQUIRED``).
"""
from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import torch

from isoexec.schema import rollout

from .aux_batch import VALID_ROWS


def _host(value: Any) -> torch.Tensor:
    return value.detach().to(device="cpu") if isinstance(value, torch.Tensor) else torch.as_tensor(value)


def requests_from_microbatch(batch: dict, cu_seqlens: Sequence[int]) -> rollout.RolloutRequests:
    """The single-turn engine requests of SkyRL microbatch ``batch``, over the forward's packed boundaries
    ``cu_seqlens`` (``packed_seq_params.cu_seqlens_q_padded``: alignment padding included)."""
    if batch.get("sub_seq_lengths") is not None or batch.get("sub_seq_lengths_list") is not None:
        raise ValueError("ROLLOUT_SUBSEQ_PACKING: controller sub-sequence packing puts several trajectories in one "
                         "row; their prompt/response split is not carried (set trainer packing off for this model)")
    response_mask = batch.get("response_mask")
    if response_mask is None:
        raise ValueError("ROLLOUT_RESPONSE_MASK_MISSING: the microbatch carries no response_mask, so the prompt "
                         "length of each row (the engine request's prompt) is unknown; SkyRL's forward-only "
                         "microbatches need it (skyrl-patches/0001-forward-microbatch-response-mask.patch)")
    attention = _host(batch["attention_mask"]).to(torch.bool)
    response = _host(response_mask).to(torch.bool)
    B, A = response.shape
    if attention.shape[0] != B or int(batch.get("num_actions", A)) != A:
        raise ValueError(f"response_mask {tuple(response.shape)} does not match attention_mask "
                         f"{tuple(attention.shape)} / num_actions {batch.get('num_actions')}")
    loss = batch.get("loss_mask")
    loss = None if loss is None else _host(loss)
    padding_batch = bool(batch.get("is_padding_batch", False))
    valid = batch.get(VALID_ROWS)
    valid = None if valid is None else _host(valid).to(torch.bool)
    cu = [int(c) for c in cu_seqlens]
    if len(cu) != B + 1:
        raise ValueError(f"ROLLOUT_PACKING: {B} rows, packed boundaries for {len(cu) - 1} sequences")
    real = attention.sum(1).tolist()
    generated = response.sum(1).tolist()
    prompts, gens = [], []
    for b in range(B):
        active = loss is not None and bool((loss[b] > 0).any())
        dummy = padding_batch or (valid is not None and b < valid.numel() and not bool(valid[b]))
        p, g = real[b] - generated[b], generated[b]
        if p < 1:
            # SkyRL's padding rows (worker_utils._create_padding_microbatch: one attended token, response_mask all
            # ones) carry no request; they run as a prompt with nothing generated and score nothing.
            if active and not dummy:
                raise ValueError(f"ROLLOUT_GEOMETRY: row {b} has {real[b]} real tokens and {g} response tokens "
                                 "but an active loss mask")
            p, g = real[b], 0
        elif g and not bool(attention[b, -A:][response[b]].all()):
            raise ValueError(f"ROLLOUT_GEOMETRY: row {b}: response tokens are not the trailing real tokens")
        if loss is not None and g:
            m = (loss[b] > 0).to(torch.int8)
            runs = int(((m[1:] == 1) & (m[:-1] == 0)).sum()) + int(m[0] == 1)
            if runs > 1:
                raise ValueError(f"CED_TURNS_REQUIRED: row {b}'s loss mask has {runs} runs of generated tokens (a "
                                 "multi-turn rollout); each turn's engine prompt length is not carried through "
                                 "SkyRL, and the V4.1 recipe's generator is single-turn (generator.batched=true)")
        prompts.append(p)
        gens.append(g)
    return rollout.RolloutRequests.single(prompts, gens, cu, real)


def admit_generator(cfg: Any, stage: Any) -> None:
    """Refuse, before any rollout, a generator that can produce several engine requests per trained sequence
    for a stage that reads them: SkyRL does not carry turn boundaries, and forward-only microbatches carry no
    loss mask from which a multi-turn row could even be recognised. Single-turn = ``generator.batched=true``
    (``generate_batched``: one request per prompt) or ``max_turns == 1`` without step-wise trajectories."""
    if stage is None or not getattr(stage, "reads_rollout_requests", False):
        return
    # The trainer worker holds SkyRL's TrainerConfig, not the full config: config.resolve records the generator's
    # turn settings on the resolved options (``generator_turns``) for this check.
    generator = getattr(cfg, "generator", None)
    if generator is None:
        turns = getattr(getattr(cfg, "_isoexec_config", None), "generator_turns", None)
        if turns is None:
            raise ValueError("CED_TURNS_REQUIRED: the generator's turn settings did not reach the trainer worker "
                             "(IsoExec's resolved options carry no generator_turns)")
        from types import SimpleNamespace
        generator = SimpleNamespace(**turns)
    batched = bool(getattr(generator, "batched", False))
    turns = getattr(generator, "max_turns", 1)
    step_wise = bool(getattr(generator, "step_wise_trajectories", False))
    if not batched and (turns != 1 or step_wise):
        raise ValueError(f"CED_TURNS_REQUIRED: generator.max_turns={turns!r}, step_wise_trajectories={step_wise} "
                         "produce several engine requests per trained sequence; their prompt lengths (each turn's "
                         "decoder segment) are not carried through SkyRL. This model admits single-turn rollouts "
                         "(generator.batched=true or max_turns=1), as the V4.1 recipe runs")


def _boundaries(packed_seq_params) -> list[int]:
    cu = packed_seq_params.cu_seqlens_q_padded
    if cu is None:
        cu = packed_seq_params.cu_seqlens_q
    return cu.detach().to(device="cpu").tolist()


@contextmanager
def microbatch_requests(stage: Any, micro_batches: Sequence[dict], *, chunks: int = 1) -> Iterator[None]:
    """Carry each microbatch's rollout requests into the stage's forward of that microbatch (in schedule
    order). A no-op for stages that do not read them."""
    if stage is None or not getattr(stage, "reads_rollout_requests", False):
        yield
        return
    if chunks != 1:
        raise ValueError("ROLLOUT_VIRTUAL_PIPELINE: rollout requests are paired with one model chunk per rank")
    if stage.forward_context is not None:
        raise RuntimeError("Rollout requests cannot overlap another microbatch schedule on the same stage")
    index = 0

    def context(packed_seq_params):
        nonlocal index
        if index >= len(micro_batches):
            raise RuntimeError(f"ROLLOUT_PAIRING: forward {index + 1} of a schedule of {len(micro_batches)} "
                               "microbatches")
        batch = micro_batches[index]
        index += 1
        return rollout.carry(requests_from_microbatch(batch, _boundaries(packed_seq_params)))

    stage.forward_context = context
    try:
        yield
        if index != len(micro_batches):
            raise RuntimeError(f"ROLLOUT_PAIRING: {index} forwards for {len(micro_batches)} microbatches")
    finally:
        stage.forward_context = None


__all__ = ["admit_generator", "microbatch_requests", "requests_from_microbatch"]
