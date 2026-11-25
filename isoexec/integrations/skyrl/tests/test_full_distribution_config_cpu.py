"""Framework-independent tests of the SkyRL adapter's admission policy."""

from types import SimpleNamespace as NS
from typing import Any, cast

import pytest

from isoexec.integrations.skyrl.full_distribution_config import (
    validate_full_distribution_config as validate_logprob_comparison,
    validate_full_distribution_runtime,
)


def _full_mode_config():
    megatron = NS(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=None,
        transformer_config_kwargs={},
    )
    placement = NS(
        policy_num_gpus_per_node=1,
        policy_num_nodes=1,
        colocate_all=True,
        asymmetric_colocation=False,
    )
    trainer = NS(
        strategy="megatron",
        fully_async=NS(enabled=False),
        mtp=NS(enabled=False),
        fused_lm_head_logprob=False,
        remove_microbatch_padding=True,
        recompute_old_logprobs_per_minibatch=True,
        placement=placement,
        policy=NS(megatron_config=megatron, model=NS(lora=NS(rank=0))),
        algorithm=NS(dynamic_sampling=NS(type=None), temperature=1.0),
    )
    engine = NS(
        backend="vllm",
        num_engines=1,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        expert_parallel_size=1,
        run_engines_locally=True,
        enable_pd=False,
        speculative_config=None,
        fp8_weight_sync_mode=None,
        weight_sync_backend="nccl",
    )
    generator = NS(
        inference_engine=engine,
        batched=True,
        vision_language_generator=False,
        max_turns=1,
        step_wise_trajectories=False,
        sampling_params=NS(temperature=1.0, logprobs=1),
        eval_sampling_params=NS(logprobs=1),
    )
    return NS(isoexec={"full_logprob_comparison": True}, trainer=trainer, generator=generator)


def test_full_mode_accepts_the_bounded_profile():
    cfg = _full_mode_config()
    validate_logprob_comparison(cfg)
    assert cfg.generator.eval_sampling_params.logprobs is None


@pytest.mark.parametrize("tp", [2, 4])
def test_full_mode_admits_explicit_expert_tensor_parallel_with_ep1(tp):
    cfg = _full_mode_config()
    cfg.trainer.placement.policy_num_gpus_per_node = tp
    cfg.trainer.placement.asymmetric_colocation = True
    mc = cfg.trainer.policy.megatron_config
    mc.tensor_model_parallel_size = tp
    mc.expert_tensor_parallel_size = tp
    validate_logprob_comparison(cfg)
    mc.expert_model_parallel_size = tp
    with pytest.raises(ValueError, match="expert_tensor_parallel_size"):
        validate_logprob_comparison(cfg)


def test_real_framework_configuration_reaches_the_package_policy(monkeypatch):
    config_module = pytest.importorskip("skyrl.train.config")
    from isoexec.integrations.skyrl import bootstrap
    monkeypatch.setattr(bootstrap, "resolve", lambda cfg: None)
    cfg = config_module.SkyRLTrainConfig.from_cli_overrides(
        [
            "trainer.strategy=megatron",
            "isoexec.full_logprob_comparison=true",
            "trainer.remove_microbatch_padding=true",
            "trainer.recompute_old_logprobs_per_minibatch=true",
            "trainer.algorithm.temperature=1.0",
            "trainer.placement.policy_num_gpus_per_node=1",
            "generator.inference_engine.tensor_parallel_size=1",
            "generator.batched=true",
        ]
    )
    bootstrap.validate_config(cfg)
    assert cfg.generator.eval_sampling_params.logprobs is None
    cfg.trainer.placement.policy_num_gpus_per_node = 4
    cfg.trainer.placement.asymmetric_colocation = True
    cfg.trainer.policy.megatron_config.context_parallel_size = 4
    with pytest.raises(ValueError, match="context_parallel_size in \\(1, 2\\)"):
        bootstrap.validate_config(cfg)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda cfg: setattr(cfg.trainer, "strategy", "fsdp"),
        lambda cfg: setattr(cfg.generator, "batched", False),
        lambda cfg: setattr(cfg.trainer.fully_async, "enabled", True),
        lambda cfg: setattr(cfg.trainer.mtp, "enabled", True),
        lambda cfg: setattr(cfg.trainer, "fused_lm_head_logprob", True),
        lambda cfg: setattr(cfg.trainer, "remove_microbatch_padding", False),
        # Non-colocated is supported only with NCCL broadcast weight sync.
        lambda cfg: (
            setattr(cfg.trainer.placement, "colocate_all", False),
            setattr(cfg.generator.inference_engine, "weight_sync_backend", "delta"),
        ),
        lambda cfg: setattr(cfg.generator, "vision_language_generator", True),
        lambda cfg: setattr(
            cfg.trainer.policy.megatron_config, "expert_tensor_parallel_size", 2
        ),
        # A colocated engine cannot outgrow the policy GPUs, whichever degree multiplies it (pipeline
        # stages included); a degree that is not a positive integer refuses on its own.
        lambda cfg: setattr(
            cfg.generator.inference_engine, "pipeline_parallel_size", 2
        ),
        lambda cfg: setattr(
            cfg.generator.inference_engine, "pipeline_parallel_size", 0
        ),
        lambda cfg: setattr(cfg.generator.inference_engine, "expert_parallel_size", 2),
        lambda cfg: setattr(cfg.generator.sampling_params, "temperature", 0.5),
        lambda cfg: setattr(cfg.generator.sampling_params, "logprobs", None),
    ],
)
def test_full_mode_rejects_unsupported_profiles(mutate):
    cfg = _full_mode_config()
    mutate(cfg)

    with pytest.raises(ValueError, match="full logprob comparison requires"):
        validate_logprob_comparison(cfg)


def test_runtime_and_native_rl_call_plan_have_distinct_requirements():
    cfg = _full_mode_config()
    cfg.trainer.recompute_old_logprobs_per_minibatch = False
    validate_full_distribution_runtime(cfg)
    with pytest.raises(ValueError, match="recompute_old_logprobs_per_minibatch=true"):
        validate_logprob_comparison(cfg)


def test_full_mode_admits_data_parallel_trainer_only_with_asymmetric_colocation():
    cfg = _full_mode_config()
    cfg.trainer.placement.policy_num_gpus_per_node = 2

    with pytest.raises(ValueError, match="asymmetric_colocation"):
        validate_logprob_comparison(cfg)

    cfg.trainer.placement.asymmetric_colocation = True
    validate_logprob_comparison(cfg)

    # Tensor parallelism over the two policy GPUs is admitted under the same contract.
    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 2
    validate_logprob_comparison(cfg)

    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 4
    with pytest.raises(ValueError, match="dividing the 2 policy GPUs"):
        validate_logprob_comparison(cfg)

    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 2
    cfg.trainer.placement.asymmetric_colocation = False
    with pytest.raises(ValueError, match="asymmetric_colocation"):
        validate_logprob_comparison(cfg)


def test_full_mode_admits_non_colocated_data_parallel_engines_with_nccl_broadcast():
    cfg = _full_mode_config()
    cfg.trainer.placement.colocate_all = False
    cfg.generator.inference_engine.weight_sync_backend = "nccl"
    cfg.generator.inference_engine.data_parallel_size = 2
    validate_logprob_comparison(cfg)

    cfg.generator.inference_engine.weight_sync_backend = "delta"
    with pytest.raises(ValueError, match="weight_sync_backend=nccl when not colocated"):
        validate_logprob_comparison(cfg)


@pytest.mark.parametrize("colocated", [False, True])
@pytest.mark.parametrize("tp", [1, 2])
def test_full_mode_admits_engine_expert_parallelism_only_across_data_parallel_ranks(
    colocated, tp
):
    cfg = _full_mode_config()
    engine = cfg.generator.inference_engine
    cfg.trainer.placement.colocate_all = colocated
    cfg.trainer.placement.policy_num_gpus_per_node = 2 * tp
    engine.weight_sync_backend = "nccl"
    engine.tensor_parallel_size = tp
    engine.data_parallel_size = 2
    engine.expert_parallel_size = 2 * tp  # IsoExec owns dispatch on the DP x TP group.
    validate_logprob_comparison(cfg)

    engine.expert_parallel_size = 4 * tp  # not DP x TP
    with pytest.raises(
        ValueError,
        match="expert_parallel_size=1, or = data_parallel_size x tensor_parallel_size",
    ):
        validate_logprob_comparison(cfg)
    engine.expert_parallel_size = 2 * tp
    engine.data_parallel_size = 1  # EP without the DP ranks it spans
    with pytest.raises(
        ValueError,
        match="expert_parallel_size=1, or = data_parallel_size x tensor_parallel_size",
    ):
        validate_logprob_comparison(cfg)


def test_full_mode_admits_independent_engine_replicas_only_when_non_colocated():
    cfg = _full_mode_config()
    cfg.generator.inference_engine.num_engines = 2
    with pytest.raises(ValueError, match="full logprob comparison requires"):
        validate_logprob_comparison(cfg)

    cfg.trainer.placement.colocate_all = False
    cfg.generator.inference_engine.weight_sync_backend = "nccl"
    validate_logprob_comparison(cfg)


def test_full_mode_admits_pipeline_parallel_trainer_only_with_asymmetric_colocation():
    cfg = _full_mode_config()
    cfg.trainer.placement.policy_num_gpus_per_node = 2
    cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = 2
    with pytest.raises(ValueError, match="full logprob comparison requires"):
        validate_logprob_comparison(cfg)

    cfg.trainer.placement.asymmetric_colocation = True
    validate_logprob_comparison(cfg)

    # PP spans exactly the policy GPUs, with TP=1 and EP=1.
    cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = 4
    with pytest.raises(ValueError, match="full logprob comparison requires"):
        validate_logprob_comparison(cfg)
    cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = 2
    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 2
    with pytest.raises(ValueError, match="full logprob comparison requires"):
        validate_logprob_comparison(cfg)
    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 1
    cfg.trainer.policy.megatron_config.context_parallel_size = 2
    with pytest.raises(ValueError, match="full logprob comparison requires"):
        validate_logprob_comparison(cfg)


def test_full_mode_admits_context_parallel_at_degree_two_only():
    # CP=2 on two policy GPUs: one replica's tokens are cut over the pair, the engine keeps GPU 0.
    cfg = _full_mode_config()
    cfg.trainer.placement.policy_num_gpus_per_node = 2
    cfg.trainer.placement.asymmetric_colocation = True
    cfg.trainer.policy.megatron_config.context_parallel_size = 2
    validate_logprob_comparison(cfg)

    # It composes with TP and PP as one more model-parallel factor of the policy GPUs.
    cfg.trainer.placement.policy_num_gpus_per_node = 8
    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 2
    cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = 2
    validate_logprob_comparison(cfg)

    # Any other degree is refused by name; expert parallelism composes with CP (MoE is per token).
    cfg.trainer.policy.megatron_config.context_parallel_size = 4
    cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = 1
    with pytest.raises(ValueError, match="context_parallel_size in \\(1, 2\\)"):
        validate_logprob_comparison(cfg)
    cfg.trainer.policy.megatron_config.context_parallel_size = 2
    cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = 2
    cfg.trainer.policy.megatron_config.expert_model_parallel_size = 2
    validate_logprob_comparison(cfg)


def test_full_mode_admits_expert_parallelism_across_the_context_parallel_cut():
    # CP peers are expert peers: EP tiles TP x CP x dense DP, so CP2 EP2 holds one replica on two GPUs.
    cfg = _full_mode_config()
    mc = cfg.trainer.policy.megatron_config
    cfg.trainer.placement.asymmetric_colocation = True
    cfg.trainer.placement.policy_num_gpus_per_node = 2
    mc.context_parallel_size, mc.expert_model_parallel_size = 2, 2
    validate_logprob_comparison(cfg)
    # TP2 x CP2 on four GPUs: EP2 is each TP group, EP4 spans the cut.
    cfg.trainer.placement.policy_num_gpus_per_node = 4
    mc.tensor_model_parallel_size = 2
    for ep in (2, 4):
        mc.expert_model_parallel_size = ep
        validate_logprob_comparison(cfg)
    # EP must divide the TP x CP x dense DP mesh.
    for ep in (3, 8):
        mc.expert_model_parallel_size = ep
        with pytest.raises(ValueError, match="divisor of TP x CP x dense DP = 4"):
            validate_logprob_comparison(cfg)


def test_full_mode_admits_expert_parallel_trainer_only_with_asymmetric_colocation():
    cfg = _full_mode_config()
    cfg.trainer.placement.policy_num_gpus_per_node = 2
    cfg.trainer.policy.megatron_config.expert_model_parallel_size = 2
    with pytest.raises(ValueError, match="full logprob comparison requires"):
        validate_logprob_comparison(cfg)

    cfg.trainer.placement.asymmetric_colocation = True
    validate_logprob_comparison(cfg)

    # Expert owners tile the policy mesh at expert TP=1.
    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 2
    validate_logprob_comparison(cfg)
    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 1
    cfg.trainer.policy.megatron_config.expert_model_parallel_size = 4
    with pytest.raises(ValueError, match="full logprob comparison requires"):
        validate_logprob_comparison(cfg)
    cfg.trainer.policy.megatron_config.expert_model_parallel_size = 2
    cfg.trainer.policy.megatron_config.expert_tensor_parallel_size = 2
    with pytest.raises(ValueError, match="full logprob comparison requires"):
        validate_logprob_comparison(cfg)


def test_full_mode_admits_trainer_compositions_that_tile_the_policy_gpus():
    cfg = _full_mode_config()
    mc = cfg.trainer.policy.megatron_config
    cfg.trainer.placement.policy_num_gpus_per_node = 4
    cfg.trainer.placement.asymmetric_colocation = True
    mc.tensor_model_parallel_size, mc.pipeline_model_parallel_size = (
        2,
        2,
    )  # TP x PP, dense DP 1
    validate_logprob_comparison(cfg)
    mc.transformer_config_kwargs["sequence_parallel"] = True  # + SP on the TP axis
    validate_logprob_comparison(cfg)
    mc.pipeline_model_parallel_size = 1  # TP x DP (+SP), and EP = TP x dense DP = 4
    mc.expert_model_parallel_size = 4
    validate_logprob_comparison(cfg)
    mc.expert_model_parallel_size = (
        2  # EP = TP: two replicas of each expert shard (expert DP 2)
    )
    validate_logprob_comparison(cfg)
    mc.tensor_model_parallel_size = 4  # TP4, EP2: EP not a multiple of TP (expert DP 2)
    validate_logprob_comparison(cfg)
    mc.tensor_model_parallel_size = 2
    mc.expert_model_parallel_size = 3  # not a divisor of TP x CP x dense DP
    with pytest.raises(
        ValueError, match="expert_model_parallel_size 1, or a divisor of TP x CP x dense DP"
    ):
        validate_logprob_comparison(cfg)
    mc.expert_model_parallel_size = 1
    mc.tensor_model_parallel_size = 1  # SP without TP
    with pytest.raises(ValueError, match="sequence_parallel"):
        validate_logprob_comparison(cfg)


def test_full_mode_admits_colocated_engines_on_a_prefix_of_the_policy_gpus():
    cfg = _full_mode_config()
    engine = cfg.generator.inference_engine
    cfg.trainer.placement.policy_num_gpus_per_node = 4
    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 2
    cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = 2
    engine.tensor_parallel_size = 2  # one TP=2 engine on policy GPUs 0-1
    with pytest.raises(ValueError, match="asymmetric_colocation=true"):
        validate_logprob_comparison(cfg)
    cfg.trainer.placement.asymmetric_colocation = True
    validate_logprob_comparison(cfg)
    engine.num_engines = 2  # two TP=2 replicas tile all four GPUs: no asymmetry needed
    cfg.trainer.placement.asymmetric_colocation = False
    validate_logprob_comparison(cfg)
    engine.num_engines = 3
    with pytest.raises(ValueError, match="fitting the 4 policy GPUs"):
        validate_logprob_comparison(cfg)


def test_full_mode_admits_tensor_parallel_engine_only_when_non_colocated():
    cfg = _full_mode_config()
    cfg.generator.inference_engine.tensor_parallel_size = 2
    with pytest.raises(ValueError, match="full logprob comparison requires"):
        validate_logprob_comparison(cfg)

    cfg.trainer.placement.colocate_all = False
    cfg.generator.inference_engine.weight_sync_backend = "nccl"
    validate_logprob_comparison(cfg)

    # Engine pipeline stages compose with it: every worker of every stage receives the weight stream.
    cfg.generator.inference_engine.pipeline_parallel_size = 2
    validate_logprob_comparison(cfg)


def test_full_mode_admits_engine_pipeline_stages_inside_the_policy_gpus():
    # Colocated: TP2 x PP2 engine on the four GPUs of a TP2 x PP2 trainer.
    cfg = _full_mode_config()
    cfg.trainer.placement.policy_num_gpus_per_node = 4
    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 2
    cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = 2
    cfg.generator.inference_engine.tensor_parallel_size = 2
    cfg.generator.inference_engine.pipeline_parallel_size = 2
    validate_logprob_comparison(cfg)

    # One more stage than GPUs is refused by the colocation rule, not by a pipeline rule.
    cfg.generator.inference_engine.pipeline_parallel_size = 4
    with pytest.raises(
        ValueError, match="colocated engines \\(8 GPUs\\) fitting the 4 policy GPUs"
    ):
        validate_logprob_comparison(cfg)

    cfg.generator.inference_engine.pipeline_parallel_size = 0
    with pytest.raises(ValueError, match="pipeline_parallel_size >= 1"):
        validate_logprob_comparison(cfg)


def test_full_mode_rejects_data_parallel_engines_when_colocated():
    cfg = _full_mode_config()
    cfg.generator.inference_engine.data_parallel_size = 2
    with pytest.raises(ValueError, match="data_parallel_size=1 when colocated"):
        validate_logprob_comparison(cfg)


def test_colocated_engine_ep_still_requires_enough_policy_gpus():
    cfg = _full_mode_config()
    cfg.trainer.placement.policy_num_gpus_per_node = 1
    cfg.generator.inference_engine.data_parallel_size = 2
    cfg.generator.inference_engine.expert_parallel_size = 2
    with pytest.raises(ValueError, match="colocated engines.*fitting"):
        validate_logprob_comparison(cfg)
