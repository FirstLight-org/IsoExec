#!/usr/bin/env python3
"""Train Qwen3.5-0.8B for one GRPO step on GSM8K with IsoExec, in the runtime `make setup` created.

Needs one SM90 GPU. SkyRL runs its own PPO loop with its ``isoexec`` config field set, with the vLLM
engine on Model Runner V2 (FULL_AND_PIECEWISE graphs). Every step, before the policy update, the
trainer's logprob of each trainable response token is compared bit for bit with the logprob the engine
sampled it with, and with ``full_logprob_comparison`` every vocabulary entry of each trainable row is
compared as well. The run fails on the first value that differs, so exit code 0 is the result. Extra
key=value arguments go to SkyRL, for example trainer.logger=wandb or trainer.max_training_steps=2.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import tomllib

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))
import setup_env  # noqa: E402

MODEL = 'Qwen/Qwen3.5-0.8B'
REVISION = '2fc06364715b967f1860aea9cf38778875588b17'


def overrides(model: Path, data: Path, run_dir: Path) -> dict:
    """BF16, Megatron TP1/PP1 and one colocated vLLM engine (Model Runner V2, CUDA graphs) on one GPU."""
    return {
        'data.train_data': [str(data / 'train.parquet')], 'data.val_data': [str(data / 'validation.parquet')],
        'isoexec': {"model_package": 'isoexec.models.qwen3_5', "full_logprob_comparison": True},
        'trainer.strategy': 'megatron', 'trainer.bf16': True,
        'trainer.policy.model.path': str(model), 'trainer.ref.model.path': str(model),
        'trainer.policy.language_model_only': True, 'trainer.ref.language_model_only': True,
        'trainer.gradient_checkpointing': False, 'trainer.remove_microbatch_padding': True,
        'trainer.fused_lm_head_logprob': False, 'trainer.recompute_old_logprobs_per_minibatch': True,
        'trainer.mtp.enabled': False,
        'trainer.placement.colocate_all': True, 'trainer.placement.policy_num_nodes': 1,
        'trainer.placement.policy_num_gpus_per_node': 1, 'trainer.placement.ref_num_gpus_per_node': 1,
        'trainer.policy.megatron_config.tensor_model_parallel_size': 1,
        'trainer.ref.megatron_config.tensor_model_parallel_size': 1,
        'trainer.policy.megatron_config.pipeline_model_parallel_size': 1,
        'trainer.ref.megatron_config.pipeline_model_parallel_size': 1,
        'trainer.policy.megatron_config.context_parallel_size': 1,
        'trainer.policy.megatron_config.expert_model_parallel_size': 1,
        'trainer.ref.megatron_config.expert_model_parallel_size': 1,
        'trainer.policy.megatron_config.expert_tensor_parallel_size': 1,
        'trainer.ref.megatron_config.expert_tensor_parallel_size': 1,
        'trainer.policy.megatron_config.transformer_config_kwargs.sequence_parallel': False,
        'trainer.ref.megatron_config.transformer_config_kwargs.sequence_parallel': False,
        'trainer.policy.megatron_config.transformer_config_kwargs.deterministic_mode': True,
        'trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_cpu_offload': False,
        'trainer.policy.megatron_config.optimizer_config_kwargs.use_precision_aware_optimizer': False,
        'trainer.policy.optimizer_config.offload_after_step': True, 'trainer.policy.optimizer_config.lr': 1e-6,
        'trainer.train_batch_size': 4, 'trainer.policy_mini_batch_size': 4,
        'trainer.micro_train_batch_size_per_gpu': 1, 'trainer.micro_forward_batch_size_per_gpu': 1,
        'trainer.epochs': 1, 'trainer.max_training_steps': 1, 'trainer.update_epochs_per_batch': 1,
        'trainer.eval_before_train': False, 'trainer.eval_interval': -1,
        'trainer.ckpt_interval': -1, 'trainer.hf_save_interval': -1, 'trainer.resume_mode': None,
        'trainer.max_prompt_length': 512, 'trainer.algorithm.advantage_estimator': 'grpo',
        'trainer.algorithm.use_kl_loss': True, 'trainer.algorithm.kl_loss_coef': 0.001,
        'trainer.algorithm.temperature': 1.0, 'trainer.algorithm.dynamic_sampling.type': None,
        'trainer.enable_ray_gpu_monitor': False, 'trainer.print_example_interval': -1,
        'trainer.dump_eval_results': False, 'trainer.logger': 'console',
        'trainer.project_name': 'isoexec-tiny', 'trainer.run_name': 'qwen3.5-0.8b',
        'trainer.log_path': str(run_dir / 'logs'), 'trainer.ckpt_path': str(run_dir / 'checkpoints'),
        'trainer.export_path': str(run_dir / 'exports'),
        'generator.inference_engine.backend': 'vllm', 'generator.inference_engine.language_model_only': True,
        'generator.inference_engine.run_engines_locally': True, 'generator.inference_engine.num_engines': 1,
        'generator.inference_engine.tensor_parallel_size': 1, 'generator.inference_engine.pipeline_parallel_size': 1,
        'generator.inference_engine.data_parallel_size': 1, 'generator.inference_engine.expert_parallel_size': 1,
        'generator.inference_engine.weight_sync_backend': 'nccl',
        'generator.inference_engine.gpu_memory_utilization': 0.35,
        'generator.inference_engine.enforce_eager': False,
        'generator.inference_engine.enable_prefix_caching': False,
        'generator.inference_engine.enable_chunked_prefill': False,
        'generator.inference_engine.max_num_batched_tokens': 2048,
        'generator.inference_engine.engine_init_kwargs': {
            'max_model_len': 2048, 'max_num_seqs': 8,
            'compilation_config': {'mode': 0, 'cudagraph_mode': 'FULL_AND_PIECEWISE'},
        },
        'generator.batched': True, 'generator.n_samples_per_prompt': 5, 'generator.max_turns': 1,
        'generator.sampling_params.max_generate_length': 1024,
        'generator.sampling_params.temperature': 1.0, 'generator.sampling_params.top_p': 1.0,
        'generator.sampling_params.logprobs': 0, 'environment.env_class': 'gsm8k',
    }


def skyrl_source(python: Path, env: dict) -> Path:
    """The SkyRL checkout installed editable into the managed venv (setup leaves SkyRL to the user)."""
    probe = 'import pathlib, skyrl; print(pathlib.Path(skyrl.__file__).resolve().parents[1])'
    found = subprocess.run([python, '-I', '-c', probe], env=env, capture_output=True, text=True)
    checkout = Path(found.stdout.strip()) if found.returncode == 0 else None
    if checkout is None or not (checkout / 'pyproject.toml').is_file():
        sys.exit(f'no editable SkyRL in {python}; install one from your SkyRL checkout: uv pip install '
                 f'--python {python} --no-deps -e <SkyRL checkout> -e <SkyRL checkout>/skyrl-gym')
    return checkout


def runtime(root: Path) -> tuple[dict, Path, Path]:
    """The environment and interpreter that `make setup` created under ``root``, and the SkyRL checkout
    installed into it."""
    if not (root / 'receipt.json').is_file():
        sys.exit(f'{root / "receipt.json"} is missing; run make setup first')
    with setup_env.MANIFEST.open('rb') as stream:
        env = setup_env.process_env(tomllib.load(stream), root, root / 'venv')
    # SkyRL's ray.init connects to RAY_ADDRESS, else to a cluster already running on
    # this machine; "local" starts a private one for this run, on the GPUs CUDA_VISIBLE_DEVICES shows.
    env.setdefault('RAY_ADDRESS', 'local')
    if env['RAY_ADDRESS'] == 'local':
        # A managed node's RAY_* settings (resource override, event export) describe its own cluster,
        # not this private one.
        for key in [key for key in env if key.startswith('RAY_') and key not in ('RAY_ADDRESS', 'RAY_ENABLE_UV_RUN_RUNTIME_ENV')]:
            env.pop(key)
    python = root / 'venv/bin/python'
    return env, python, skyrl_source(python, env)


def download_model(python: Path, env: dict) -> Path:
    """The local snapshot of ``MODEL`` at ``REVISION``, downloaded into the runtime's Hugging Face cache if needed."""
    download = f'from huggingface_hub import snapshot_download; print(snapshot_download({MODEL!r}, revision={REVISION!r}))'
    return Path(subprocess.run([python, '-c', download], env=env, check=True, capture_output=True,
                               text=True).stdout.splitlines()[-1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(os.environ.get('ISOEXEC_ROOT', '~/.isoexec')))
    args, extra = parser.parse_known_args()
    root = args.root.expanduser().resolve()
    env, python, skyrl = runtime(root)
    data = root / 'data/gsm8k'
    if not all((data / f'{split}.parquet').is_file() for split in ('train', 'validation')):
        subprocess.run([python, 'examples/train/gsm8k/gsm8k_dataset.py', '--output_dir', data],
                       cwd=skyrl, env=env, check=True)
    model = download_model(python, env)
    run_dir = root / 'runs' / time.strftime('%Y%m%d-%H%M%S')
    command = [str(python), '-m', 'skyrl.train.entrypoints.main_base']
    command += [f"{key}={json.dumps(value, separators=(',', ':'))}"
                for key, value in overrides(model, data, run_dir).items()]
    command += extra
    print('[run-tiny] ' + shlex.join(command), flush=True)
    return subprocess.run(command, cwd=skyrl, env=env).returncode


if __name__ == '__main__':
    sys.exit(main())
