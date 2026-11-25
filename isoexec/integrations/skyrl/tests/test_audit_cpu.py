"""The trainer == engine check around SkyRL's policy forward, on SkyRL's real batch class."""

import pytest
import torch
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch

from isoexec.checks import Refusal
from isoexec.integrations.skyrl import audit
from isoexec.integrations.skyrl.audit import after_policy_forward, before_policy_forward

ROLLOUT = torch.tensor([[-0.25, -0.75, -1.5], [-2.0, -0.125, 0.0]])
FULL = {"model_package": "unused", "full_logprob_comparison": True}


def batch(**fields):
    return TrainingInputBatch({"sequences": torch.zeros(2, 4, dtype=torch.long), "rollout_logprobs": ROLLOUT.clone(),
                               "loss_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]), **fields})


def test_a_step_passes_when_every_trainable_token_matches_bit_for_bit():
    trainer = ROLLOUT.clone()
    trainer[0, 2] = 9.0  # outside the loss mask: not a trainable token
    padded = torch.cat((trainer, torch.full((1, 3), 7.0)))  # SkyRL's forward output can be longer than the batch
    data = batch()
    assert after_policy_forward(data, data, padded) == {
        "isoexec/rollout_train_logprob_tokens": 5, "isoexec/rollout_train_logprob_mismatches": 0}


def test_a_step_is_refused_when_a_trainable_token_differs_in_any_bit():
    trainer = ROLLOUT.clone()
    trainer[1, 0] = torch.nextafter(ROLLOUT[1, 0], torch.tensor(0.0))
    trainer[1, 2] = -0.0  # a different bit pattern counts
    data = batch()
    with pytest.raises(Refusal, match="2 of 5"):
        after_policy_forward(data, data, trainer)


@pytest.mark.parametrize("value", [float("-inf"), float("nan")])
def test_a_step_is_refused_when_both_sides_hold_the_same_non_finite_logprob(value):
    rollout = ROLLOUT.clone()
    rollout[1, 0] = value
    data = batch(rollout_logprobs=rollout)
    with pytest.raises(Refusal, match="not finite"):
        after_policy_forward(data, data, rollout.clone())


def test_a_step_is_refused_when_there_is_nothing_to_compare():
    data = batch()
    with pytest.raises(Refusal, match="pre-update policy forward"):
        after_policy_forward(data, data, None)
    data = batch(rollout_logprobs=None)
    with pytest.raises(Refusal, match="rollout_logprobs"):
        after_policy_forward(data, data, ROLLOUT.clone())
    data = batch(loss_mask=torch.zeros(2, 3, dtype=torch.long))
    with pytest.raises(Refusal, match="no trainable response tokens"):
        after_policy_forward(data, data, ROLLOUT.clone())


def test_full_comparison_is_refused_before_the_forward_when_the_rollout_has_no_rows():
    data = batch()
    with pytest.raises(Refusal, match="no full-vocabulary rows"):
        before_policy_forward(FULL, data, data, global_step=1)
    # Off: the sampled-token check needs nothing prepared.
    before_policy_forward({"model_package": "unused"}, data, data, global_step=1)


def test_full_comparison_registers_the_plan_before_the_forward_and_reports_the_verified_rows_after(monkeypatch):
    registered = []

    def begin(data_fwd_pass, boundaries, global_step):
        registered.append((boundaries, global_step))
        data_fwd_pass.metadata[audit.TOKEN] = "plan"

    monkeypatch.setattr(audit, "begin_online_prescore", begin)
    monkeypatch.setattr(audit, "finish_online_prescore", lambda data_fwd_pass: {"rows": 5})
    training_input = batch()
    training_input.metadata = {"policy_mini_batch_boundaries": [(0, 2)]}
    policy_data_fwd_pass = batch(rollout_full_logprobs=torch.zeros(2, 3, 4))
    policy_data_fwd_pass.metadata = {}

    before_policy_forward(FULL, training_input, policy_data_fwd_pass, global_step=3)
    assert registered == [([(0, 2)], 3)]
    metrics = after_policy_forward(training_input, policy_data_fwd_pass, ROLLOUT.clone())
    assert metrics["policy/full_logprobs_verified_rows"] == 5
