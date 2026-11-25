"""CPU checks for the N-step IsoExec verl trainer: optimizer receipts per step, config switch, validator."""

import json

import pytest
from omegaconf import OmegaConf

from isoexec.checks import Refusal
from isoexec.integrations.verl import config as iso_config
from isoexec.integrations.verl.trainer import validate_multistep_trainer_config, verify_optimizer_steps


def _write(root, rank, steps):
    receipts = root / "receipts"
    receipts.mkdir(parents=True, exist_ok=True)
    with open(receipts / f"optimizer-rank{rank:04d}.jsonl", "a") as handle:
        for step in steps:
            handle.write(json.dumps({"event": "optimizer_step", "step": step, "rank": rank, "grad_norm": 0.5,
                                     "changed": True}) + "\n")


def test_two_updates_per_step_are_counted_per_window(tmp_path):
    for rank in range(4):
        _write(tmp_path, rank, [1, 2, 3, 4])
    assert verify_optimizer_steps(tmp_path, 4, 1, 2)["updates"] == 2
    assert verify_optimizer_steps(tmp_path, 4, 3, 4)["updates"] == 2
    with pytest.raises(Refusal):
        verify_optimizer_steps(tmp_path, 4, 5, 6)


def test_a_missing_rank_update_is_refused(tmp_path):
    for rank in range(3):
        _write(tmp_path, rank, [1, 2])
    _write(tmp_path, 3, [1])
    with pytest.raises(Refusal):
        verify_optimizer_steps(tmp_path, 4, 1, 2)


def _base_config(**isoexec):
    return OmegaConf.create({
        "model_engine": "megatron",
        "isoexec": {"enable": True, **isoexec},
        "trainer": {"val_before_train": True, "v1": {"trainer_mode": "sync"}},
        "actor_rollout_ref": {"model": {}, "rollout": {"engine_kwargs": {"vllm": {}}}},
    })


def test_multi_step_switch_keeps_validation_and_selects_the_multistep_trainer():
    config = _base_config(multi_step=True)
    iso_config.apply(config)
    assert config.trainer.v1.trainer_mode == iso_config.MULTI_STEP_TRAINER_MODE
    assert config.trainer.val_before_train is True


def test_default_switch_is_the_one_update_qualification():
    config = _base_config()
    iso_config.apply(config)
    assert config.trainer.v1.trainer_mode == iso_config.TRAINER_MODE
    assert config.trainer.val_before_train is False


def _trainer_config(**overrides):
    base = {
        "trainer": {"critic_warmup": 0, "save_freq": -1, "v1": {"sync": {"parameter_sync_step": 1}}},
        "data": {"train_batch_size": 64, "gen_batch_size": None},
        "algorithm": {"rollout_correction": {"bypass_mode": False}},
        "actor_rollout_ref": {"actor": {"ppo_mini_batch_size": 32, "ppo_epochs": 1},
                              "rollout": {"calculate_log_probs": True, "n": 5}},
    }
    return OmegaConf.merge(OmegaConf.create(base), OmegaConf.create(overrides))


def test_validator_admits_several_updates_per_step_and_refuses_checkpointing():
    validate_multistep_trainer_config(_trainer_config())
    with pytest.raises(ValueError):
        validate_multistep_trainer_config(_trainer_config(trainer={"save_freq": 10}))
    with pytest.raises(ValueError):
        validate_multistep_trainer_config(_trainer_config(actor_rollout_ref={"actor": {"ppo_mini_batch_size": 48}}))


def test_mismatch_policy_defaults_to_refuse_and_admits_record():
    from isoexec.integrations.verl.trainer import mismatch_policy

    assert mismatch_policy(_trainer_config()) == "refuse"
    recording = _trainer_config(isoexec={"on_mismatch": "record"})
    validate_multistep_trainer_config(recording)
    assert mismatch_policy(recording) == "record"
    with pytest.raises(ValueError):
        validate_multistep_trainer_config(_trainer_config(isoexec={"on_mismatch": "ignore"}))


def test_sampled_agreement_counts_bit_patterns():
    import torch

    from isoexec.integrations.verl.trainer import sampled_logprob_agreement

    trainer = torch.tensor([[0.0, -1.0, -2.5, -7.0]])
    engine = torch.tensor([[-0.0, -1.0, -2.0, -9.0]])
    mask = torch.tensor([[1, 1, 1, 0]])
    agreement = sampled_logprob_agreement(trainer, engine, mask)
    assert agreement == {"tokens": 3, "samples": 1, "mismatched_tokens": 2, "max_abs_diff": 0.5}
