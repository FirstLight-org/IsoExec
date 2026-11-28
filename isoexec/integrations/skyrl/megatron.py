"""SkyRL Megatron lifecycle around IsoExec's host-owned model and optimizer."""

import math
import os
from dataclasses import replace
from functools import partial
from typing import Any, List, cast
import torch
import torch.nn as nn
import megatron.core.parallel_state as mpu

from isoexec.integrations.skyrl import model_folder


def validate_mesh(worker, mc, lora_config):
    if lora_config is not None:
        raise ValueError("IsoExec does not support trainer.policy.model.lora")
    # Qualify CP degree and model rows separately; only the supported two-rank layout is admitted.
    cp = mc.context_parallel_size
    if cp not in (1, 2):
        raise ValueError(f"IsoExec admits context_parallel_size 1 or 2, got {cp!r}")
    # Expert parallelism composes with CP: routing, experts and the combine are per token, and the EP
    # group (formed by Megatron over TP x CP x DP) may span CP ranks; dispatch moves whole rows only.
    # Weight synchronization and HF export gather the stages into a complete model stream.
    ep, tp = mc.expert_model_parallel_size, mc.tensor_model_parallel_size
    etp = mc.expert_tensor_parallel_size
    # Global tensor ownership distinguishes dense ETP=1 from the EP=1 expert
    # layout. Check that model-dependent coupling before materialization below.
    allowed = (1, tp) if ep == 1 else (1,)
    if etp is not None and etp not in allowed:
        raise ValueError(f"expert_tensor_parallel_size={etp} must be one of {allowed} "
                         f"for expert_model_parallel_size={ep}")
    if ep > 1:
        # The live expert mesh must be the one the degrees imply (EP | TP x CP x DP; any expert-DP degree).
        from isoexec.runtimes.parallel import expert_data_parallel_degree, expert_data_parallel_refusal

        dp = mpu.get_data_parallel_world_size(with_context_parallel=False)
        expert_dp = expert_data_parallel_degree(tp, dp, ep, cp)
        refusal = expert_data_parallel_refusal(expert_dp, int(mpu.get_expert_data_parallel_world_size()))
        if refusal:
            raise ValueError(refusal)
        print(
            f"[ISOEXEC-TOPOLOGY] expert parallelism: tp={tp} cp={cp} dp={dp} ep={ep} expert_dp={expert_dp} "
            f"ep_group={sorted(torch.distributed.get_process_group_ranks(cast(torch.distributed.ProcessGroup, mpu.get_expert_model_parallel_group())))} "
            f"expert_dp_rank={int(mpu.get_expert_data_parallel_rank())}",
            flush=True,
        )


def validate_model_options(mc):
    from skyrl.train.config.config import DEFAULT_TRANSFORMER_CONFIG_KWARGS

    for name in ("fp8", "fp8_recipe", "fp8_param", "fp8_amax_compute_algo"):
        if getattr(mc, name) is not None:
            raise ValueError(f"IsoExec does not support megatron_config.{name}")
    if mc.model_config_kwargs:
        raise ValueError("IsoExec does not support megatron_config.model_config_kwargs")
    if mc.mtp_num_layers not in (None, 0):
        raise ValueError("IsoExec does not support megatron_config.mtp_num_layers")
    if mc.moe_router_load_balancing_type not in ("none", "aux_loss"):
        raise ValueError("IsoExec Megatron auxiliary loss supports only 'none' or 'aux_loss'")
    coefficient = mc.moe_aux_loss_coeff
    if (isinstance(coefficient, bool) or not isinstance(coefficient, (int, float))
            or not math.isfinite(coefficient) or coefficient < 0):
        raise ValueError("moe_aux_loss_coeff must be a finite nonnegative number")
    if (
        mc.moe_router_dtype != "fp32"
        or mc.moe_router_score_function not in (None, "softmax")
        or mc.moe_router_enable_expert_bias
    ):
        raise ValueError("IsoExec requires checkpoint router semantics and moe_router_dtype='fp32'")
    # wrap_ddp_and_optimizer installs dtype-grouped gradient statistics for precision-aware Adam.
    for name, value in mc.transformer_config_kwargs.items():
        if name == "recompute_num_layers":
            if not isinstance(value, int) or value < 1:
                raise ValueError("megatron_config.transformer_config_kwargs.recompute_num_layers must be positive")
        elif name == "sequence_parallel":
            # IsoExec owns SP over the TP group; native stages read this flag.
            if not isinstance(value, bool):
                raise ValueError("megatron_config.transformer_config_kwargs.sequence_parallel must be a bool")
            if value and mc.tensor_model_parallel_size < 2:
                raise ValueError("sequence_parallel=true needs tensor_model_parallel_size >= 2 (SP shards over the TP group)")
        elif name == "deterministic_mode":
            if type(value) is not bool:
                raise ValueError("megatron_config.transformer_config_kwargs.deterministic_mode must be a bool")
        elif name not in DEFAULT_TRANSFORMER_CONFIG_KWARGS or value != DEFAULT_TRANSFORMER_CONFIG_KWARGS[name]:
            raise ValueError(
                f"IsoExec does not support megatron_config.transformer_config_kwargs.{name}={value!r}"
            )


def build_module(self, *, wrap_with_ddp: bool, ddp_config, lora_config, bf16: bool) -> List[nn.Module]:
    section = getattr(self.cfg, self._isoexec_section_name)
    selected = model_folder.source_config(self.cfg._isoexec_config.model_descriptor, self.cfg, section.model.path)
    from megatron.core.distributed import DistributedDataParallelConfig
    from megatron.core.process_groups_config import ProcessGroupCollection

    from isoexec.runtimes.megatron.factory import build_model, wrap_ddp_and_optimizer
    from skyrl.backends.skyrl_train.distributed.megatron.optimizer import init_megatron_optim_config
    from skyrl.train.config import get_config_as_dict

    mc = section.megatron_config
    validate_mesh(self, mc, lora_config)
    validate_model_options(mc)
    if not bf16:
        raise ValueError("Native SkyRL model construction requires trainer.bf16=True")
    recompute = bool(self.cfg.gradient_checkpointing)
    deterministic = mc.transformer_config_kwargs.get("deterministic_mode", False)
    from isoexec.runtimes.trainer_installs import install_trainer_process

    install_trainer_process(section.model.path, source=selected[0], raw_config=selected[1],
                            deterministic_attention_backward=deterministic)
    model = build_model(
        *selected, checkpoint_path=section.model.path, training=wrap_with_ddp,
        max_tokens=self.cfg._isoexec_config.trainer_capture_max_tokens,
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
        transformer_config_kwargs={
            "tensor_model_parallel_size": mc.tensor_model_parallel_size,
            "pipeline_model_parallel_size": mc.pipeline_model_parallel_size,
            "context_parallel_size": mc.context_parallel_size,
            "expert_model_parallel_size": mc.expert_model_parallel_size,
            "expert_tensor_parallel_size": mc.expert_tensor_parallel_size,
            "sequence_parallel": mc.transformer_config_kwargs.get("sequence_parallel", False),
            "deterministic_mode": deterministic,
            "recompute_granularity": "full" if recompute else None,
            "recompute_method": "uniform" if recompute else None,
            "recompute_num_layers": mc.transformer_config_kwargs.get("recompute_num_layers", 1) if recompute else None,
        },
        # SkyRL shards packed rows with Megatron's zigzag cut (model_utils._get_tokens_on_this_cp_rank) and
        # scores them with the same cut, so the model must bind the zigzag CP form on every arch.
        cp_layout="zigzag" if mc.context_parallel_size > 1 else None,
    )
    self._isoexec_optimizer = None
    if not wrap_with_ddp:
        return [model]
    ddp_options: dict[str, Any] = {"use_distributed_optimizer": True}
    if ddp_config is not None:
        ddp_options.update(get_config_as_dict(ddp_config))
    optim_config = init_megatron_optim_config(section.optimizer_config, mc.optimizer_config_kwargs)
    optimizer_builder = None
    disk_config = getattr(mc, "optimizer_offload_to_disk", None)
    if disk_config is not None and disk_config.dir is not None:
        from skyrl.backends.skyrl_train.distributed.megatron.optimizer_offload_to_disk import (
            build_optimizer_offloaded_to_disk,
        )

        settings = replace(disk_config, dir=os.path.join(disk_config.dir, self.cfg.run_name))
        optimizer_builder = partial(build_optimizer_offloaded_to_disk, settings=settings)
    ddp, optimizer = wrap_ddp_and_optimizer(
        model, ddp_config=DistributedDataParallelConfig(**ddp_options), optimizer_config=optim_config,
        optimizer_builder=optimizer_builder,
    )
    self._isoexec_optimizer = optimizer
    return [ddp]


def init_configs(worker, model_path, megatron_config):
    from transformers import AutoConfig
    from skyrl.utils.tok import get_tokenizer

    config_path = model_folder.configuration_path(model_path)
    source, raw = model_folder.source_config(worker.cfg._isoexec_config.model_descriptor, worker.cfg, model_path)
    if getattr(source._module("model"), "CONFIG_CLASS", None) is not None:
        # A package that ships its own config class (DeepSeek V4.1: model_type deepseek_v41, which the installed
        # transformers does not register) is parsed by that class, as the engine and the model build parse it.
        worker.strategy.hf_config = source.build_config(raw)
    else:
        worker.strategy.hf_config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)
    worker.tokenizer = get_tokenizer(config_path, trust_remote_code=True)
    worker.megatron_config = megatron_config
    worker.bridge = None
    worker.provider = None
    worker.is_vlm = False
    worker.enable_router_replay = False
    worker.enable_sample_support_replay = worker.cfg.algorithm.enable_sample_support_replay
    worker._logical_model_path = model_path
    worker._deferred_fp8_param_weight_load = False
    worker._fp8_param_unquantized_state_dict = None


def finish_optimizer_step(model):
    from isoexec.runtimes.megatron.optimizer import finish_native_optimizer_step
    from isoexec.runtimes.megatron.replicas import check_after_optimizer_step

    result = finish_native_optimizer_step(model.actor_module)
    # Every rank, after every step and so before any weight sync: replicated copies must still be equal.
    check_after_optimizer_step(model.actor_module)
    return result
