# Run one DeepSeek V4.1 GRPO step with SkyRL on four B200s

This is the six-layer GSM8K experiment. SkyRL generates two responses for each of four prompts, scores the generated tokens with Megatron, updates the policy once and synchronizes the new weights to vLLM. Both runtimes load the same BF16 checkpoint. The trainer uses TP1/PP1/EP4 and the colocated rollout uses one TP4/PP1 engine with expert parallelism enabled.

## Environment

Run the commands below in Bash from the root of this branch. Use a Linux host with four available B200s, CUDA 13, and local disk for checkpoint conversion, optimizer moments and frozen-weight offload. The checkpoint payload alone is 346 GiB. The two full-sized FP8 Engram tables also need host memory accessible to the workers.

If the SkyRL environment already contains the IsoExec integration and V4.1 hooks, activate it and set `SKYRL_SRC` to that checkout. To use FirstLight's pinned environment from scratch:

```bash
export IX_ROOT="$PWD"
export RUNTIME="$HOME/.isoexec-v41-dev"
make setup-dev ISOEXEC_ROOT="$RUNTIME"
source "$RUNTIME/venv/bin/activate"
export SKYRL_SRC="$RUNTIME/skyrl"
git -C "$SKYRL_SRC" apply --check "$IX_ROOT/examples/skyrl/v41-six-layer.patch"
git -C "$SKYRL_SRC" apply "$IX_ROOT/examples/skyrl/v41-six-layer.patch"
```

The patch is for the SkyRL revision in `environments/default.toml` (`d2f629d9`). It carries the rollout response mask through policy scoring, forwards Megatron's optimizer options to disk-backed Adam, and binds each worker's host allocations to the NUMA node of its active GPU. In an integrated checkout these changes are already present. In either environment, use this branch on the driver and workers:

```bash
export IX_ROOT="$PWD"
export PYTHONPATH="$IX_ROOT:$SKYRL_SRC:$SKYRL_SRC/skyrl-gym${PYTHONPATH:+:$PYTHONPATH}"
export SKYRL_PYTHONPATH_EXPORT=1
export RUN=/path/to/new-run-directory
export MODEL=/path/to/six-layer-bf16
export DATA="$RUN/gsm8k-six"
mkdir -p "$RUN" "$DATA"
```

## Checkpoint and prompts

With the experimental six-layer checkpoint already mounted at `$MODEL`, go straight to the data step. To build it from the released `deepseek-ai/DeepSeek-V4.1-Flash` weights, point `SOURCE` at the complete local snapshot and choose a fresh output directory:

```bash
export SOURCE=/path/to/DeepSeek-V4.1-Flash
python examples/prepare_v41_six_layer.py --source "$SOURCE" --plan
python examples/prepare_v41_six_layer.py --source "$SOURCE" --out "$MODEL"
```

The six production layers are **0, 1, 14, 20, 24, 25**, stored as rows 0–5. Row 3 is the encoder/decoder boundary. The two Engram tables live on rows 1 and 2 and retain production hash IDs **1 and 14**. Hidden size is 5,120, with 384 routed experts, six selected per token, and expert width 2,304. The plan should identify **7,072 tensors and 371,149,391,136 payload bytes**. Conversion expands FP4/FP8 projections to BF16, copies the encoded Engram tables and sets the tokenizer's default to chat mode.

Prepare SkyRL's GSM8K Parquet files, then select the same 24 training prompts in their original order. The validation split is the standard GSM8K test set.

```bash
python "$SKYRL_SRC/examples/train/gsm8k/gsm8k_dataset.py" --output_dir "$RUN/gsm8k-full"
python - <<'PY'
import os
import shutil
from pathlib import Path
import pyarrow.parquet as pq

src = Path(os.environ["RUN"]) / "gsm8k-full"
dst = Path(os.environ["DATA"])
indices = [175, 399, 593, 636, 757, 803, 818, 839, 997, 1071, 1102, 1202,
           1261, 1380, 1531, 1546, 1647, 1689, 1764, 1894, 2002, 2098, 2114, 2161]
pq.write_table(pq.read_table(src / "train.parquet").take(indices), dst / "train.parquet")
shutil.copyfile(src / "validation.parquet", dst / "validation.parquet")
PY
```

## Launch

Set `CUDA_VISIBLE_DEVICES` to four GPUs allocated to this job. `RAY_ADDRESS=local` starts a local Ray instance. The existing recipe supplies the SkyRL model package, optimizer, GRPO loss and vLLM settings. The overrides below select the measured single-node run, including its 2 GiB KV cache, disk-backed FP32 Adam moments and unfrozen routed experts.

The six-layer runs use chunked prefill with a 256-token scheduler budget and disable prefix caching. On this branch the SkyRL integration selects vLLM Model Runner V2 and enables breakable CUDA graphs. The FirstLight and native runs each captured 35 piecewise and 19 full graph sizes. The earlier IsoExec reference run used `FULL_DECODE_ONLY`. To reproduce that earlier mode, append `generator.inference_engine.engine_init_kwargs.compilation_config.cudagraph_mode=FULL_DECODE_ONLY` to `args` before validation and launch.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export RAY_ADDRESS=local
export RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES=2147483648
export SKYRL_FROZEN_OFFLOAD_DIR="$RUN/frozen"
export SKYRL_OPTIMIZER_OFFLOAD_TO_DISK_CHUNK_NUMEL=16000000
export ISOEXEC_DS_FREEZE_ROUTED_EXPERTS=0
export ISOEXEC_V41_PIECEWISE_BREAKS=core
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
mkdir -p "$RUN/optimizer" "$SKYRL_FROZEN_OFFLOAD_DIR"

args=(
  --checkpoint "$MODEL" --data "$DATA"
  trainer.seed=42
  trainer.placement.policy_num_nodes=1 trainer.placement.policy_num_gpus_per_node=4
  trainer.policy.megatron_config.expert_model_parallel_size=4
  trainer.policy.megatron_config.ddp_config.grad_reduce_in_fp32=true
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_cpu_offload=false
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_offload_fraction=0.0
  "trainer.policy.megatron_config.optimizer_offload_to_disk.dir=$RUN/optimizer"
  trainer.policy.megatron_config.optimizer_offload_to_disk.moment_storage_dtype=fp32
  trainer.policy.megatron_config.optimizer_offload_to_disk.staging_buffer_mb=64
  trainer.train_batch_size=4 trainer.policy_mini_batch_size=4
  trainer.max_training_steps=1 trainer.eval_interval=-1 trainer.algorithm.use_entropy_loss=true
  'trainer.extra_runtime_env_vars={"OMP_NUM_THREADS":"8","MKL_NUM_THREADS":"8","OPENBLAS_NUM_THREADS":"1","ISOEXEC_DS_FREEZE_ROUTED_EXPERTS":"0"}'
  generator.inference_engine.num_engines=1 generator.inference_engine.tensor_parallel_size=4
  generator.inference_engine.pipeline_parallel_size=1 generator.inference_engine.gpu_memory_utilization=0.8
  generator.inference_engine.max_num_batched_tokens=256
  generator.inference_engine.engine_init_kwargs.kv_cache_memory_bytes=2147483648
  generator.inference_engine.engine_init_kwargs.cudagraph_metrics=true
  generator.n_samples_per_prompt=2 generator.sampling_params.max_generate_length=64
  "trainer.ckpt_path=$RUN/checkpoints" "trainer.log_path=$RUN/logs" "trainer.export_path=$RUN/exports"
)
python "$IX_ROOT/examples/run_v41_gsm8k.py" --validate-only "${args[@]}"
set -o pipefail
python "$IX_ROOT/examples/run_v41_gsm8k.py" "${args[@]}" 2>&1 | tee "$RUN/skyrl.log"
```

## Check the result

Look for `trainer/global_step = 1`, a nonzero `grad_norm` and a completed `timing/sync_weights`. Compare `policy/rollout_train_logprobs_abs_diff_mean` and `_max` for generated tokens. SkyRL's console summary rounds values to four decimal places. Its `minibatch_rollout_logprobs_abs_diff_*` records give the unrounded values for this single minibatch.

```bash
grep -E 'trainer/global_step|timing/sync_weights|grad_norm|rollout_(train_)?logprobs_abs_diff_(mean|max)' "$RUN/skyrl.log"
```

| Six-layer run | Generated tokens | Mean absolute difference (nats) | Maximum (nats) | First step (s) |
| --- | ---: | ---: | ---: | ---: |
| SkyRL native Megatron + vLLM | 512 | 0.215999484 | 2.735767841 | 1,200.707 |
| IsoExec before the port, full decode graphs | 512 | 0 | 0 | 1,024.408 |
| FirstLight IsoExec, piecewise prefill and full decode graphs | 512 | 0 | 0 | 1,364.711 |

The FirstLight run used IsoExec `6d43a57f`, the patched SkyRL revision above and vLLM 0.33.0. During rollout, vLLM dispatched five 256-token batches and one 57-token batch padded to 64 in piecewise mode. It also reported 63 full-graph dispatches. Megatron recorded a gradient norm of 0.2006974 and the updated weights synchronized in 14.5709 s. The first step includes initial compilation and the disk-backed optimizer update. The B200 host was shared during timing.

The native and pre-port measurements are the earlier reference pair (IsoExec `9db5de37`, integrated SkyRL `2d580b25`). The [native launch](v41-six-layer-native.md) used its own V4.1-capable Megatron/vLLM environment, the same checkpoint, tokenizer and prompts, and native `FLASHMLA_SPARSE_DSV41` attention. Its reduced-depth loaders honored `engram_hash_layer_ids` on both sides. Each run sampled its own responses. The reported differences compare rollout and trainer within each run at equal weights.
