# Native reference for the six-layer SkyRL run

Use the same `MODEL`, `DATA`, `RUN`, `IX_ROOT` and four allocated B200s as in the [IsoExec experiment](v41-six-layer-skyrl.md). Run the arms sequentially. The native arm uses SkyRL `0a29e57e` and its locked Megatron and vLLM 0.33.0 environment.

## Prepare the native environment

In a separate shell, set the paths above and select a fresh checkout. Build SkyRL with the Megatron extra. The two patches use the checkpoint's production Engram hash IDs in the native trainer and engine. The trainer keeps the large host tables file-backed. The native engine uses its fallback cache path when the cache row stride is unaligned.

```bash
export NATIVE_RUN="$RUN/native"
export NATIVE_SRC="$NATIVE_RUN/SkyRL"
mkdir -p "$NATIVE_RUN"
git clone https://github.com/NovaSky-AI/SkyRL.git "$NATIVE_SRC"
git -C "$NATIVE_SRC" checkout 0a29e57ec9812e7d409a44a5ce932e5d56e6d6d0
unset VIRTUAL_ENV PYTHONPATH
export UV_PROJECT_ENVIRONMENT="$NATIVE_RUN/venv"
(cd "$NATIVE_SRC" && uv sync --frozen --extra megatron --python 3.12)
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
git -C "$NATIVE_SRC" apply "$IX_ROOT/examples/skyrl/v41-six-layer-native-skyrl.patch"
site=$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
(cd "$site" && patch -p1 < "$IX_ROOT/examples/skyrl/v41-six-layer-native-vllm.patch")
```

## Launch the native step

Run this in the native environment with the four devices exposed. Use a separate optimizer directory for each arm. This is the native run's resolved command with portable paths.

```bash
export PYTHONPATH="$NATIVE_SRC:$NATIVE_SRC/skyrl-gym"
export SKYRL_PYTHONPATH_EXPORT=1 RAY_ADDRESS=local
export NVTE_TENSOR_HANDLE_POOL_SIZE_MB=512 VLLM_ALLOW_INSECURE_SERIALIZATION=1
export SKYRL_DSV41_ENGRAM_TABLE_DEVICE=cpu SKYRL_DSV41_FREEZE_ROUTED_EXPERTS=0
export SKYRL_FROZEN_OFFLOAD_DIR="$NATIVE_RUN/frozen"
export SKYRL_OPTIMIZER_OFFLOAD_TO_DISK_CHUNK_NUMEL=16000000
export RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES=2147483648
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S=3600
mkdir -p "$NATIVE_RUN/optimizer" "$SKYRL_FROZEN_OFFLOAD_DIR"
args=(
  "data.train_data=['$DATA/train.parquet']"
  "data.val_data=['$DATA/validation.parquet']"
  trainer.strategy=megatron
  trainer.algorithm.advantage_estimator=grpo
  trainer.algorithm.use_kl_loss=false
  trainer.algorithm.loss_reduction=sequence_mean
  "trainer.policy.model.path=$MODEL"
  trainer.placement.colocate_all=true
  trainer.placement.policy_num_nodes=1
  trainer.placement.policy_num_gpus_per_node=4
  trainer.policy.megatron_config.tensor_model_parallel_size=1
  trainer.policy.megatron_config.pipeline_model_parallel_size=1
  trainer.policy.megatron_config.context_parallel_size=1
  trainer.policy.megatron_config.expert_model_parallel_size=4
  trainer.policy.megatron_config.expert_tensor_parallel_size=1
  trainer.policy.megatron_config.mtp_num_layers=0
  trainer.policy.megatron_config.moe_grouped_gemm=true
  trainer.policy.megatron_config.moe_token_dispatcher_type=alltoall
  trainer.policy.megatron_config.moe_enable_routing_replay=false
  generator.inference_engine.enable_return_routed_experts=false
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_granularity=selective
  'trainer.policy.megatron_config.transformer_config_kwargs.recompute_modules=[moe,mhc,layernorm]'
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_method=null
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_num_layers=null
  trainer.policy.megatron_config.transformer_config_kwargs.mlp_chunks_for_training=64
  trainer.policy.megatron_config.transformer_config_kwargs.gradient_accumulation_fusion=false
  trainer.policy.megatron_config.transformer_config_kwargs.disable_parameter_transpose_cache=true
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_cpu_offload=false
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_offload_fraction=0.0
  trainer.policy.megatron_config.optimizer_config_kwargs.overlap_cpu_optimizer_d2h_h2d=false
  trainer.policy.megatron_config.empty_cuda_cache=true
  trainer.policy.megatron_config.optimizer_config_kwargs.use_precision_aware_optimizer=false
  trainer.policy.megatron_config.ddp_config.grad_reduce_in_fp32=true
  trainer.policy.optimizer_config.lr=1e-6
  trainer.policy.optimizer_config.max_grad_norm=1.0
  trainer.remove_microbatch_padding=false
  trainer.use_expandable_segments=true
  trainer.micro_forward_batch_size_per_gpu=4
  trainer.micro_train_batch_size_per_gpu=1
  trainer.train_batch_size=4
  trainer.policy_mini_batch_size=4
  trainer.update_epochs_per_batch=1
  trainer.epochs=1
  trainer.max_training_steps=1
  trainer.max_prompt_length=512
  trainer.eval_batch_size=256
  trainer.eval_before_train=false
  trainer.eval_interval=-1
  trainer.ckpt_interval=-1
  trainer.resume_mode=null
  "trainer.ckpt_path=$NATIVE_RUN/checkpoints"
  generator.inference_engine.backend=vllm
  generator.inference_engine.run_engines_locally=true
  generator.inference_engine.num_engines=1
  generator.inference_engine.tensor_parallel_size=4
  generator.inference_engine.pipeline_parallel_size=1
  generator.inference_engine.gpu_memory_utilization=0.80
  generator.inference_engine.enforce_eager=false
  'generator.inference_engine.engine_init_kwargs={"kv_cache_memory_bytes":2147483648,"max_model_len":2048,"max_num_seqs":128,"enable_chunked_prefill":true,"enable_prefix_caching":false,"attention_backend":"FLASHMLA_SPARSE_DSV41","enable_expert_parallel":true}'
  generator.sampling_params.max_generate_length=64
  generator.batched=true
  generator.n_samples_per_prompt=2
  environment.env_class=gsm8k
  trainer.logger=console
  trainer.project_name=v41-six-layer
  trainer.run_name=native
  generator.inference_engine.max_num_batched_tokens=256
  trainer.algorithm.use_entropy_loss=true
  "trainer.log_path=$NATIVE_RUN/logs"
  "trainer.export_path=$NATIVE_RUN/exports"
  "trainer.policy.megatron_config.optimizer_offload_to_disk.dir=$NATIVE_RUN/optimizer"
  trainer.policy.megatron_config.optimizer_offload_to_disk.moment_storage_dtype=fp32
  trainer.policy.megatron_config.optimizer_offload_to_disk.staging_buffer_mb=64
)
set -o pipefail
python -m skyrl.train.entrypoints.main_base "${args[@]}" 2>&1 | tee "$NATIVE_RUN/skyrl.log"
```

Read the unrounded `minibatch_rollout_logprobs_abs_diff_mean` and `_max` alongside the optimizer step and weight sync. The reference values and token counts are in the [six-layer results table](v41-six-layer-skyrl.md#check-the-result).
