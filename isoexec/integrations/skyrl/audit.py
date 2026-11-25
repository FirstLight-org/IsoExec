"""The checks SkyRL's trainer runs around its policy forward, every step and before the update.

Trainer == engine: the trainer's logprob of each trainable response token must equal, bit for bit, the
logprob the engine reported when it sampled the token. With the full comparison on, every vocabulary
entry of every trainable row is compared in the workers during that forward, and a collector confirms
that every row and vocabulary slice was compared by some rank.
"""

from typing import Any

import torch

from isoexec.checks import Refusal
from isoexec.integrations.full_distribution import full_comparison_enabled

from .online_prescore import TOKEN, begin_online_prescore, finish_online_prescore


def before_policy_forward(isoexec: Any, training_input: Any, policy_data_fwd_pass: Any, *, global_step: int) -> None:
    """Called before SkyRL's policy forward: with the full comparison on, register what it has to compare.

    ``policy_data_fwd_pass`` is the batch SkyRL sends to the policy workers. The plan rides its metadata
    to the workers, which report to the collector what they compared.
    """
    if not full_comparison_enabled(isoexec):
        return
    if policy_data_fwd_pass.get("rollout_full_logprobs") is None:
        raise Refusal("IsoExec's full comparison is on but the rollout carried no full-vocabulary rows")
    if not bool((policy_data_fwd_pass["loss_mask"] > 0).any()):
        raise Refusal("IsoExec has no trainable response tokens to audit")
    boundaries = training_input.metadata.get("policy_mini_batch_boundaries")
    begin_online_prescore(policy_data_fwd_pass, boundaries, global_step)


def after_policy_forward(training_input: Any, policy_data_fwd_pass: Any, action_log_probs: Any) -> dict[str, int]:
    """Refuse the step unless the policy forward's logprobs equal the rollout's; return the metrics to log."""
    rollout = training_input.get("rollout_logprobs")
    if rollout is None or action_log_probs is None:
        raise Refusal("IsoExec requires rollout_logprobs and a pre-update policy forward")
    mask = training_input["loss_mask"] > 0
    if not bool(mask.any()):
        raise Refusal("IsoExec has no trainable response tokens to audit")
    # SkyRL's forward output can be longer than the batch; it slices it the same way.
    trainer = action_log_probs[: len(training_input["sequences"])]
    engine_logprobs = rollout[mask].to(torch.float32).contiguous()
    trainer_logprobs = trainer[mask].to(torch.float32).contiguous()
    # Equal bits are not enough on their own: NaN or -inf on both sides would also be equal.
    if not bool(torch.isfinite(engine_logprobs).all()) or not bool(torch.isfinite(trainer_logprobs).all()):
        raise Refusal("trainer != engine: a trainable response-token logprob is not finite")
    engine_bits = engine_logprobs.view(torch.int32)
    trainer_bits = trainer_logprobs.view(torch.int32)
    tokens = int(engine_bits.numel())
    mismatches = int((engine_bits != trainer_bits).sum().item())
    print(f"[ISOEXEC-SKYRL] trainer vs rollout logprobs: {mismatches} of {tokens} trainable tokens differ", flush=True)
    if mismatches:
        raise Refusal(f"trainer != engine: {mismatches} of {tokens} trainable response-token logprobs differ")
    metrics = {"isoexec/rollout_train_logprob_tokens": tokens, "isoexec/rollout_train_logprob_mismatches": mismatches}
    if TOKEN in (policy_data_fwd_pass.metadata or {}):
        receipt = finish_online_prescore(policy_data_fwd_pass)
        metrics["policy/full_logprobs_verified_rows"] = int(receipt["rows"])
    return metrics
