"""Resolve the optional IsoExec mapping from SkyRL's existing configuration."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from isoexec.checks import dependencies
from isoexec.config import IsoExecConfig
from isoexec.integrations.full_distribution import full_comparison_enabled
from .model_folder import _sections, configuration_path, prepare_checkpoints, source_config, validate_model_config

#: The engine's CUDA graph modes, all on vLLM Model Runner V2 and bitwise identical to each other.
#: FULL_AND_PIECEWISE (the default) replays FULL graphs for uniform decode and vLLM's breakable
#: PIECEWISE graphs, with attention and GDN cores as eager breaks, for mixed batches.
CUDAGRAPH_MODES = ("NONE", "FULL_DECODE_ONLY", "FULL_AND_PIECEWISE")


@dataclass
class SkyRLConfig(IsoExecConfig):
    """Worker options carried by SkyRL's existing TrainerConfig serialization."""

    model_descriptor: dict[str, Any] | None = None
    generator_turns: dict[str, Any] | None = None


def _cudagraph_mode(engine) -> str:
    """Resolve the engine CUDA graph mode; graphs are captured without compilation.

    The engine runs vLLM Model Runner V2: ``FULL_AND_PIECEWISE`` (the default) captures FULL decode
    graphs and breakable PIECEWISE graphs for mixed batches; ``enforce_eager`` selects ``NONE``;
    ``FULL_DECODE_ONLY`` runs mixed batches eagerly.
    """
    compilation = engine.engine_init_kwargs.get("compilation_config", {})
    if not isinstance(compilation, dict):
        raise ValueError(
            "generator.inference_engine.engine_init_kwargs.compilation_config must be a mapping for IsoExec"
        )
    if compilation.get("mode", 0) not in (0, "NONE"):
        raise ValueError(
            "IsoExec requires generator.inference_engine.engine_init_kwargs.compilation_config.mode=0; CUDA graphs remain supported"
        )
    mode = compilation.get("cudagraph_mode", "NONE" if engine.enforce_eager else "FULL_AND_PIECEWISE")
    if mode not in CUDAGRAPH_MODES:
        raise ValueError(f"IsoExec runs vLLM Model Runner V2 with cudagraph_mode in {CUDAGRAPH_MODES}, got {mode!r}")
    return mode


def _worker_environment(cfg, options) -> dict[str, str]:
    """The process environment every Ray actor of the job (trainer and vLLM server) starts with.

    Kernel selections, the vLLM plugins and the Model Runner V2 / breakable-graph switches must be in
    place before an actor imports CUDA code or builds a ``VllmConfig``, so they ride the job's
    runtime environment (SkyRL's ``trainer.extra_runtime_env_vars``) rather than any in-process setting.
    """
    from isoexec.runtimes.environment import resolved_environment
    from isoexec.runtimes.vllm.registration import engine_environment
    from isoexec.runtimes.vllm.vllm_patches import ISOEXEC_VLLM_ENV

    from .weight_transfer import PLUGINS

    env = {key: value for key, value in os.environ.items() if key.startswith("ISOEXEC")}
    if "NVTE_TENSOR_HANDLE_POOL_SIZE_MB" in os.environ:
        env["NVTE_TENSOR_HANDLE_POOL_SIZE_MB"] = os.environ["NVTE_TENSOR_HANDLE_POOL_SIZE_MB"]
    env.update(cfg.trainer.extra_runtime_env_vars)
    env.update(ISOEXEC_VLLM_ENV)
    env.update(resolved_environment(cfg.trainer.policy.model.path))
    # vLLM decides when it builds the engine config whether PIECEWISE graphs can run without
    # torch.compile; without this the server actor silently runs FULL_DECODE_ONLY (which VllmModel refuses).
    env.update(engine_environment(options.cudagraph_mode))
    env["VLLM_PLUGINS"] = PLUGINS
    return env


def resolve(cfg):
    """Validate an IsoExec run and resolve its checkpoints once, in the driver, before any worker starts.

    The options ride ``cfg.trainer`` (SkyRL's worker configuration) to every worker, and the actors'
    environment rides ``cfg.trainer.extra_runtime_env_vars`` into SkyRL's Ray runtime environment.
    """
    if cfg.isoexec is None:
        return None
    validate_model_config(cfg.isoexec)
    trainer, engine = cfg.trainer, cfg.generator.inference_engine
    # Check dependency compatibility before workers start.
    rows = dependencies.check(host="skyrl")
    gap = dependencies.unverified(rows)
    if gap:
        print(
            f"[ISOEXEC-DEPENDENCIES] accepted but NOT gate-verified here: {', '.join(gap)}. "
            f"These are inside the declared range and IsoExec will run, but the test gate has not "
            f"been run against them; a bitwise claim is only as strong as where it was measured.",
            flush=True,
        )
    checks = {
        "trainer.strategy": (trainer.strategy, "megatron"),
        "generator.inference_engine.backend": (engine.backend, "vllm"),
        "trainer.bf16": (trainer.bf16, True),
        "trainer.remove_microbatch_padding": (trainer.remove_microbatch_padding, True),
        "trainer.policy.use_torch_compile": (trainer.policy.use_torch_compile, False),
        "trainer.policy.model.lora.rank": (trainer.policy.model.lora.rank, 0),
        "trainer.policy.model.fake_int4_qat.enabled": (trainer.policy.model.fake_int4_qat.enabled, False),
        "trainer.mtp.enabled": (trainer.mtp.enabled, False),
        "trainer.fused_lm_head_logprob": (trainer.fused_lm_head_logprob, False),
        # Sample-support scores come from SkyRL's own vocabulary-parallel scorer, not the model's head.
        "trainer.algorithm.enable_sample_support_replay": (trainer.algorithm.enable_sample_support_replay, False),
        "trainer.policy.megatron_config.moe_enable_routing_replay": (
            trainer.policy.megatron_config.moe_enable_routing_replay,
            False,
        ),
        "trainer.policy.megatron_config.freeze_moe_router": (trainer.policy.megatron_config.freeze_moe_router, False),
        "trainer.policy.megatron_config.freeze_dsa_indexer": (trainer.policy.megatron_config.freeze_dsa_indexer, False),
    }
    for name, (actual, expected) in checks.items():
        if actual != expected:
            raise ValueError(f"IsoExec requires {name}={expected!r}, got {actual!r}")
    full_distribution = full_comparison_enabled(cfg.isoexec)
    # IsoExec owns engine expert dispatch across DP and TP groups.
    engine_ep = engine.expert_parallel_size
    if not isinstance(engine_ep, int) or isinstance(engine_ep, bool) or engine_ep < 1:
        raise ValueError(f"generator.inference_engine.expert_parallel_size must be a positive integer, got {engine_ep!r}")
    if engine_ep != 1 and (
        engine.data_parallel_size <= 1 or engine_ep != engine.data_parallel_size * engine.tensor_parallel_size
    ):
        raise ValueError(
            "IsoExec hosts engine expert parallelism only across the engine's data-parallel "
            f"ranks: generator.inference_engine.expert_parallel_size must be 1 or data_parallel_size x "
            f"tensor_parallel_size with data_parallel_size > 1, got expert_parallel_size={engine_ep}, "
            f"data_parallel_size={engine.data_parallel_size}, tensor_parallel_size={engine.tensor_parallel_size}"
        )
    engine_pp = engine.pipeline_parallel_size
    if not isinstance(engine_pp, int) or isinstance(engine_pp, bool) or engine_pp < 1:
        raise ValueError(
            f"generator.inference_engine.pipeline_parallel_size must be a positive integer, got {engine_pp!r}"
        )
    cp = trainer.policy.megatron_config.context_parallel_size
    if cp not in (1, 2) or isinstance(cp, bool):
        raise ValueError(
            f"IsoExec requires trainer.policy.megatron_config.context_parallel_size in (1, 2), got {cp!r}"
        )
    pp = trainer.policy.megatron_config.pipeline_model_parallel_size
    if not isinstance(pp, int) or isinstance(pp, bool) or pp < 1:
        raise ValueError(
            f"trainer.policy.megatron_config.pipeline_model_parallel_size must be a positive integer, got {pp!r}"
        )
    if engine.external_proxy_url is not None or engine.external_server_urls is not None:
        raise ValueError("IsoExec launches its own vLLM servers; external inference servers are unsupported")
    if engine.enable_pd:
        raise ValueError("IsoExec does not support prefill/decode disaggregation (enable_pd)")
    if engine.fp8_weight_sync_mode is not None:
        raise ValueError("IsoExec does not support generator.inference_engine.fp8_weight_sync_mode")
    if engine.weight_sync_backend != "nccl":
        # SkyRL resolves "nccl" to its NCCL broadcast, or to CUDA IPC when colocated; IsoExec receives both.
        raise ValueError(
            f"IsoExec requires generator.inference_engine.weight_sync_backend='nccl', got {engine.weight_sync_backend!r}"
        )
    if engine.model_dtype not in ("auto", "bfloat16", "bf16"):
        raise ValueError("generator.inference_engine.model_dtype must be bfloat16 for IsoExec")
    if "logprobs_mode" in engine.engine_init_kwargs:
        # SkyRL admits only its own mode there; the IsoExec engine always reports raw_logprobs at the
        # training temperature (registration.register_native).
        raise ValueError("Leave generator.inference_engine.engine_init_kwargs.logprobs_mode unset for IsoExec")
    # SkyRL loads configs and tokenizers from model.path itself, so a single weight file is not a checkpoint.
    for name in _sections(trainer):
        if Path(getattr(trainer, name).model.path).expanduser().is_file():
            raise ValueError(
                f"IsoExec with SkyRL requires trainer.{name}.model.path to be a checkpoint directory or Hub "
                "repository, not a weight file"
            )
    prepare_checkpoints(cfg)
    # Refuse a missing or stale native model source before workers start.
    source_config(cfg.isoexec, trainer, trainer.policy.model.path)
    options = SkyRLConfig(
        model_descriptor=cfg.isoexec,
        scoring="contracted",
        score_temperature=(trainer.algorithm.temperature if trainer.algorithm.temperature is not None
                           else cfg.generator.sampling_params.temperature),
        full_distribution_comparison=full_distribution,
        engine_tp=engine.tensor_parallel_size, engine_ep=engine.expert_parallel_size,
        max_model_len=engine.engine_init_kwargs.get("max_model_len", 0),
        max_num_batched_tokens=engine.max_num_batched_tokens or 0,
    )
    from .trainer_profile import trainer_capture_bound

    options.trainer_capture_max_tokens = trainer_capture_bound(cfg)
    options.tokenizer_path = configuration_path(trainer.policy.model.path)
    options.cudagraph_mode = _cudagraph_mode(engine)
    # Opt-in through the engine kwargs only: SkyRL's own enable_chunked_prefill defaults on and is not
    # plumbed to the engine, so it is not read as a request.
    options.enable_chunked_prefill = bool(engine.engine_init_kwargs.get("enable_chunked_prefill", False))
    generator = cfg.generator
    options.generator_turns = {
        "batched": bool(getattr(generator, "batched", False)),
        "max_turns": getattr(generator, "max_turns", 1),
        "step_wise_trajectories": bool(getattr(generator, "step_wise_trajectories", False)),
    }
    trainer._isoexec_config = options
    trainer.extra_runtime_env_vars = _worker_environment(cfg, options)
    return options


def configure_isoexec_vllm_engine(cfg, args):
    """Turn SkyRL's vLLM server arguments for the policy into the IsoExec engine's, in place.

    ``args`` is the namespace ``build_vllm_cli_args(cfg)`` returns; ``resolve`` already ran in the driver.
    SkyRL's weight-sync transport is kept and only its receive engine is replaced.
    """
    from isoexec.runtimes.vllm.registration import register_native

    from .aux_batch import install_padding_tracking
    from .inference import engine_defaults
    from .weight_transfer import configure

    # SkyRL calls this in the trainer's process before training starts: mark synthetic padding rows
    # before that trainer dispatches batch slices to the workers.
    install_padding_tracking()
    options = cfg.trainer._isoexec_config
    if options.full_distribution_comparison:
        # A request for complete rows asks for more logprobs than vLLM's default cap admits.
        requested = dict(cfg.generator.inference_engine.engine_init_kwargs)
        engine_defaults(requested)
        args.max_logprobs = requested["max_logprobs"]
    checkpoint = cfg.trainer.policy.model.path
    source, raw = source_config(cfg.isoexec, cfg.trainer, checkpoint)
    artifact = getattr(cfg, "_isoexec_checkpoint_artifacts", {}).get("policy")
    if args.model not in (checkpoint, configuration_path(checkpoint)):
        raise ValueError("Engine model must use the resolved policy checkpoint artifact")
    args.model = args.tokenizer = configuration_path(checkpoint)
    if artifact is not None:
        args.revision = args.tokenizer_revision = args.code_revision = artifact["revision"]
    changes = register_native(source, raw, checkpoint=checkpoint, engine_kwargs=vars(args),
                              score_temperature=options.score_temperature,
                              cudagraph_mode=options.cudagraph_mode,
                              enable_chunked_prefill=options.enable_chunked_prefill)
    for name, value in changes.items():
        setattr(args, name, value)
    configure(args)
    # The engine's side of the policy worker's choice (workers.IsoExecPolicyWorker._build_weight_source): under the
    # full comparison a colocated (IPC) sync carries every Parameter twice and the engine compares the copies.
    transfer = getattr(args, "weight_transfer_config", None)
    args.additional_config["isoexec"]["verify_weight_updates"] = bool(
        options.full_distribution_comparison and transfer is not None and transfer.backend == "isoexec_ipc")
