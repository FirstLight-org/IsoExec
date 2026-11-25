"""verl: trainer context parallelism with an engine without CP is admitted; engine PCP/DCP is refused."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from isoexec.checks import Refusal

_ROLLOUT = dict(calculate_log_probs=True, enable_chunked_prefill=False, enable_prefix_caching=False,
                enforce_eager=False, logprobs_mode="raw_logprobs", dtype="bfloat16", topk_log_probs=0)


def _rollout(vllm=None):
    return SimpleNamespace(**_ROLLOUT, engine_kwargs=OmegaConf.create({"vllm": vllm or {}}))


@pytest.mark.parametrize("key", ["prefill_context_parallel_size", "decode_context_parallel_size"])
def test_rollout_server_refuses_engine_context_parallelism(key):
    from isoexec.integrations.verl.rollout import validate_rollout_config

    validate_rollout_config(_rollout())
    validate_rollout_config(_rollout({key: 1}))
    with pytest.raises(Refusal, match="engine_kwargs.vllm: engine context parallelism is not supported"):
        validate_rollout_config(_rollout({key: 2}))


def _trainer_config(vllm):
    return OmegaConf.create({
        "trainer": {"critic_warmup": 0, "save_freq": -1, "total_training_steps": 1, "val_before_train": False,
                    "test_freq": -1, "v1": {"sync": {"parameter_sync_step": 1}}},
        "data": {"train_batch_size": 64, "gen_batch_size": None},
        "algorithm": {"rollout_correction": {"bypass_mode": False}},
        "actor_rollout_ref": {"actor": {"ppo_mini_batch_size": 64, "ppo_epochs": 1},
                              "rollout": {"calculate_log_probs": True, "n": 5, "engine_kwargs": {"vllm": vllm}}},
    })


@pytest.mark.parametrize("validator", ["validate_trainer_config", "validate_multistep_trainer_config"])
@pytest.mark.parametrize("key", ["prefill_context_parallel_size", "decode_context_parallel_size"])
def test_trainer_validators_refuse_engine_context_parallelism_before_workers(validator, key):
    from isoexec.integrations.verl import trainer

    getattr(trainer, validator)(_trainer_config({}))
    with pytest.raises(Refusal, match="use trainer CP with an engine without CP"):
        getattr(trainer, validator)(_trainer_config({key: 2}))


@pytest.mark.parametrize("cp", [2, 4, 8])
def test_trainer_context_parallelism_with_an_engine_without_cp_is_admitted(cp):
    from isoexec.integrations.verl.engine import validate_engine_config
    from isoexec.integrations.verl.rollout import validate_rollout_config
    from isoexec.integrations.verl.tests.test_verl_engine_config_cpu import _engine_config, _model_config
    from isoexec.integrations.verl.trainer import validate_multistep_trainer_config

    validate_engine_config(_engine_config(context_parallel_size=cp), _model_config())
    validate_rollout_config(_rollout())
    validate_multistep_trainer_config(_trainer_config({}))
