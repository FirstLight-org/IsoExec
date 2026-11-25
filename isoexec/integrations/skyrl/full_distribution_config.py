"""Admission policy for SkyRL's full-distribution diagnostic integration."""

from isoexec.integrations.full_distribution import full_comparison_enabled


def validate_full_distribution_runtime(cfg):
    """Validate the model, topology and scoring shared by RL and Tinker callers."""
    trainer, generator = cfg.trainer, cfg.generator
    engine = generator.inference_engine
    if not full_comparison_enabled(cfg.isoexec):
        return
    megatron = trainer.policy.megatron_config
    placement = trainer.placement
    policy_gpus = placement.policy_num_gpus_per_node
    tp, pp = megatron.tensor_model_parallel_size, megatron.pipeline_model_parallel_size
    cp, ep = megatron.context_parallel_size, megatron.expert_model_parallel_size
    # Context parallelism multiplies the model-parallel footprint like TP and PP do: a CP group of 2
    # holds ONE replica's tokens, so dense DP is what is left after TP x PP x CP.
    model_parallel = tp * pp * cp if isinstance(cp, int) and cp >= 1 else 0
    dense_dp = (
        policy_gpus // model_parallel
        if model_parallel and policy_gpus % model_parallel == 0
        else 0
    )
    engine_gpus = (
        engine.num_engines
        * engine.tensor_parallel_size
        * engine.pipeline_parallel_size
        * engine.data_parallel_size
    )
    sequence_parallel = megatron.transformer_config_kwargs.get(
        "sequence_parallel", False
    )

    # Every requirement is named, so a refusal says which one failed instead of reciting the whole profile.
    requirements = [
        (trainer.strategy == "megatron", "trainer.strategy=megatron"),
        (engine.backend == "vllm", "inference_engine.backend=vllm"),
        (
            not trainer.fully_async.enabled,
            "synchronous training (fully_async disabled)",
        ),
        (not trainer.mtp.enabled, "trainer.mtp disabled"),
        (
            not trainer.fused_lm_head_logprob,
            "trainer.fused_lm_head_logprob=false (the full row needs the logits)",
        ),
        (
            trainer.remove_microbatch_padding,
            "trainer.remove_microbatch_padding=true (packed sequences)",
        ),
        (trainer.policy.model.lora.rank == 0, "no LoRA"),
        (placement.policy_num_nodes == 1, "a single policy node"),
        # Require TP and PP to divide the available policy GPUs.
        (
            dense_dp >= 1,
            f"tensor x pipeline x context parallel size ({tp} x {pp} x {cp}) dividing the {policy_gpus} policy GPUs",
        ),
        # CP=2 is supported for dense attention; each rank scores its own predictor-token rows.
        (cp in (1, 2), "context_parallel_size in (1, 2)"),
        # Whole experts (ETP=1): the expert mesh EP x expert-DP covers the dense TP x CP x DP mesh.
        (
            ep == 1 or (dense_dp >= 1 and (tp * cp * dense_dp) % ep == 0),
            f"expert_model_parallel_size 1, or a divisor of TP x CP x dense DP = {tp * cp * dense_dp}",
        ),
        (
            megatron.expert_tensor_parallel_size in ((None, 1, tp) if ep == 1 else (None, 1)),
            "expert_tensor_parallel_size unset, 1, or TP when EP=1; MoE native groups must match the model placement",
        ),
        (
            megatron.transformer_config_kwargs.get(
                "virtual_pipeline_model_parallel_size"
            )
            is None,
            "no virtual pipeline parallelism",
        ),
        (
            isinstance(sequence_parallel, bool) and (not sequence_parallel or tp > 1),
            "transformer_config_kwargs.sequence_parallel is a bool and needs tensor_model_parallel_size > 1",
        ),
        # -- engines
        (engine.run_engines_locally, "inference_engine.run_engines_locally=true"),
        (not engine.enable_pd, "no prefill/decode disaggregation"),
        # The engine host validates pipeline stage ownership and residual-stream transfers.
        (
            isinstance(engine.pipeline_parallel_size, int)
            and engine.pipeline_parallel_size >= 1,
            "inference_engine.pipeline_parallel_size >= 1",
        ),
        # Use host-owned engine EP across DP and TP; keep vLLM's own dispatch disabled.
        (
            engine.expert_parallel_size == 1
            or (
                engine.data_parallel_size > 1
                and engine.expert_parallel_size
                == engine.data_parallel_size * engine.tensor_parallel_size
            ),
            "inference_engine.expert_parallel_size=1, or = data_parallel_size x tensor_parallel_size with "
            "data_parallel_size > 1",
        ),
        (engine.speculative_config is None, "no speculative decoding"),
        (engine.fp8_weight_sync_mode is None, "no fp8 weight sync"),
        (
            trainer.algorithm.temperature == 1.0,
            "trainer.algorithm.temperature=1",
        ),
    ]
    if placement.colocate_all:
        # Pair colocated engine and trainer ranks by GPU, then reshard the complete model for that destination.
        requirements += [
            (
                engine_gpus <= policy_gpus,
                f"colocated engines ({engine_gpus} GPUs) fitting the {policy_gpus} policy GPUs",
            ),
            (
                engine_gpus >= policy_gpus or placement.asymmetric_colocation,
                "placement.asymmetric_colocation=true when the engines use fewer GPUs than the policy",
            ),
            (
                engine.data_parallel_size == 1
                or (
                    engine.data_parallel_size > 1
                    and engine.expert_parallel_size
                    == engine.data_parallel_size * engine.tensor_parallel_size
                ),
                "inference_engine.data_parallel_size=1 when colocated, or IsoExec engine EP = DP x TP with DP > 1",
            ),
        ]
    else:
        # Broadcast full logical weights from trainer rank zero; each remote engine extracts its own shard.
        requirements += [
            (
                engine.weight_sync_backend == "nccl",
                "inference_engine.weight_sync_backend=nccl when not colocated",
            ),
            (
                engine.data_parallel_size >= 1,
                "inference_engine.data_parallel_size >= 1",
            ),
        ]
    failed = [reason for ok, reason in requirements if not ok]
    if failed:
        raise ValueError(
            "full logprob comparison requires synchronous, packed Megatron/vLLM "
            "at temperature=1; unmet: "
            + "; ".join(failed)
        )


def validate_full_distribution_config(cfg):
    """Validate the native RL loop's data production and prescore call plan."""
    validate_full_distribution_runtime(cfg)
    if not full_comparison_enabled(cfg.isoexec):
        return
    trainer, generator = cfg.trainer, cfg.generator
    requirements = [
        (generator.batched, "generator.batched=true"),
        (not generator.vision_language_generator, "text-only generator"),
        (generator.max_turns == 1, "single-turn generation"),
        (not generator.step_wise_trajectories, "no step-wise trajectories"),
        (trainer.algorithm.dynamic_sampling.type is None, "no dynamic sampling"),
        (generator.sampling_params.temperature == 1.0, "temperature=1 on both sides"),
        (generator.sampling_params.logprobs is not None, "generator.sampling_params.logprobs set"),
        (trainer.recompute_old_logprobs_per_minibatch,
         "trainer.recompute_old_logprobs_per_minibatch=true so the driver's exact call plan matches native prescore"),
    ]
    failed = [reason for ok, reason in requirements if not ok]
    if failed:
        raise ValueError("full logprob comparison requires single-turn batched text generation; unmet: "
                         + "; ".join(failed))
    if generator.eval_sampling_params is not None:
        generator.eval_sampling_params.logprobs = None
