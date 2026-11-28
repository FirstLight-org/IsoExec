"""Run colocated SkyRL V4.1 Flash GRPO on an existing eight-node Ray cluster.

Use the IsoExec-enabled SkyRL environment on every node. Checkpoint and GSM8K
paths must be visible to every worker. Pass --validate-only to check the
configuration before connecting to Ray. Extra key=value arguments go to SkyRL.
"""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("overrides", metavar="key=value", nargs="*")
    args = parser.parse_args()
    model = str(args.checkpoint.resolve())
    data = args.data.resolve()
    config = {
        "data.train_data": [str(data / "train.parquet")],
        "data.val_data": [str(data / "validation.parquet")],
        "isoexec": {"model_package": "isoexec.models.deepseek_v4_1_flash"},
        "trainer.strategy": "megatron",
        "trainer.bf16": True,
        "trainer.policy.model.path": model,
        "trainer.ref.model.path": model,
        "trainer.placement.colocate_all": True,
        "trainer.placement.policy_num_nodes": 8,
        "trainer.placement.policy_num_gpus_per_node": 8,
        "trainer.policy.megatron_config.tensor_model_parallel_size": 1,
        "trainer.policy.megatron_config.pipeline_model_parallel_size": 1,
        "trainer.policy.megatron_config.expert_model_parallel_size": 64,
        "trainer.policy.megatron_config.expert_tensor_parallel_size": 1,
        "trainer.policy.megatron_config.moe_enable_routing_replay": False,
        "trainer.policy.megatron_config.transformer_config_kwargs.sequence_parallel": False,
        "trainer.policy.megatron_config.transformer_config_kwargs.deterministic_mode": True,
        "trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_cpu_offload": True,
        "trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_offload_fraction": 1.0,
        "trainer.policy.megatron_config.optimizer_config_kwargs.overlap_cpu_optimizer_d2h_h2d": False,
        "trainer.policy.megatron_config.optimizer_config_kwargs.use_precision_aware_optimizer": False,
        "trainer.policy.megatron_config.ddp_config.grad_reduce_in_fp32": False,
        "trainer.policy.megatron_config.empty_cuda_cache": True,
        "trainer.policy.optimizer_config.lr": 1e-6,
        "trainer.policy.optimizer_config.max_grad_norm": 1.0,
        "trainer.gradient_checkpointing": True,
        "trainer.remove_microbatch_padding": True,
        "trainer.fused_lm_head_logprob": False,
        "trainer.mtp.enabled": False,
        "trainer.algorithm.advantage_estimator": "grpo",
        "trainer.algorithm.loss_reduction": "sequence_mean",
        "trainer.algorithm.use_kl_loss": False,
        "trainer.train_batch_size": 64,
        "trainer.policy_mini_batch_size": 64,
        "trainer.micro_train_batch_size_per_gpu": 1,
        "trainer.micro_forward_batch_size_per_gpu": 4,
        "trainer.max_training_steps": 10,
        "trainer.max_prompt_length": 512,
        "trainer.eval_before_train": False,
        "trainer.eval_interval": 10,
        "trainer.ckpt_interval": -1,
        "trainer.resume_mode": None,
        "trainer.logger": "console",
        "generator.inference_engine.backend": "vllm",
        "generator.inference_engine.run_engines_locally": True,
        "generator.inference_engine.distributed_executor_backend": "ray",
        "generator.inference_engine.num_engines": 4,
        "generator.inference_engine.tensor_parallel_size": 8,
        "generator.inference_engine.pipeline_parallel_size": 2,
        "generator.inference_engine.expert_parallel_size": 1,
        "generator.inference_engine.gpu_memory_utilization": 0.6,
        "generator.inference_engine.weight_sync_backend": "nccl",
        "generator.inference_engine.enable_prefix_caching": False,
        "generator.inference_engine.enforce_eager": False,
        "generator.inference_engine.max_num_batched_tokens": 8192,
        "generator.inference_engine.engine_init_kwargs": {
            "max_model_len": 2048, "max_num_seqs": 128,
            "enable_expert_parallel": True, "enable_chunked_prefill": True,
            "compilation_config": {"mode": 0, "cudagraph_mode": "FULL_AND_PIECEWISE"},
        },
        "generator.batched": True,
        "generator.max_turns": 1,
        "generator.n_samples_per_prompt": 8,
        "generator.sampling_params.max_generate_length": 1024,
        "environment.env_class": "gsm8k",
    }
    overrides = [f"{key}={json.dumps(value, separators=(',', ':'))}" for key, value in config.items()]
    overrides.extend(args.overrides)
    os.environ.setdefault("NVTE_TENSOR_HANDLE_POOL_SIZE_MB", "512")
    os.environ.setdefault("ISOEXEC_V41_PIECEWISE_BREAKS", "core")
    if args.validate_only:
        from skyrl.train.config import SkyRLTrainConfig
        from skyrl.train.utils import validate_cfg

        validate_cfg(SkyRLTrainConfig.from_cli_overrides(overrides))
        print("SkyRL configuration validated")
        return
    os.environ.setdefault("RAY_ADDRESS", "auto")
    os.execv(sys.executable, [sys.executable, "-m", "skyrl.train.entrypoints.main_base", *overrides])


if __name__ == "__main__":
    main()
