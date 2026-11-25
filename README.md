# IsoExec: A Cross-Framework Abstraction for Training-Inference Determinism

> One model. One numerical path. Same bits.

IsoExec is a cross-framework execution layer that enables RL trainers and rollout engines to compute bit-identical
execution. It eliminates training-inference mismatch by introducing:

- **One model definition.** An `nn.Module` built from `isoexec.nn` components and shared between the trainer and rollout
  engine.
- **One numerical path.** Operations bind to architecture-specific kernels with matching numerical behavior across
  runtimes.
- **Bit-identical verification.** Trainer and inference compare full-vocabulary
  logprobs and ensure zero mismatch.

## Quickstart

Install `isoexec` and its dependencies:

```sh
git clone https://github.com/FirstLight-org/IsoExec.git
cd IsoExec
make setup
```

`make setup` installs the third-party packages of the SkyRL lock pinned in
`environments/default.toml`, not SkyRL itself. Install SkyRL into the same environment from a
checkout that has SkyRL's `isoexec` config field (until it lands upstream, the
`feat/skyrl-isoexec` branch of `JiangJiaWei1103/SkyRL`). `--no-deps` keeps the packages setup
installed:

```sh
git clone -b feat/skyrl-isoexec https://github.com/JiangJiaWei1103/SkyRL.git ../SkyRL
uv pip install --python ~/.isoexec/venv/bin/python --no-deps -e ../SkyRL -e ../SkyRL/skyrl-gym
```

Next, train Qwen3.5-0.8B for one GRPO step on GSM8K using one GPU:

```sh
make run-tiny
```

## Supported Hardware

| Platform | Specification |
| --- | --- |
| NVIDIA | B200; H100/H200 |

## Supported Models

| Model | Package | Checkpoints |
| --- | --- | --- |
| Qwen3.5 | `isoexec.models.qwen3_5` | `Qwen/Qwen3.5-0.8B`, `Qwen/Qwen3.5-9B` |
| Qwen3.5 MoE | `isoexec.models.qwen3_5_moe` | `Qwen/Qwen3.5-35B-A3B-Base` |
| Qwen2 MoE | `isoexec.models.qwen2_moe` | `Qwen/Qwen1.5-MoE-A2.7B` |
| Qwen3 | `isoexec.models.qwen3` | `Qwen/Qwen3-8B` (SM90) |
| Qwen3 MoE | `isoexec.models.qwen3_moe` | `Qwen/Qwen3-30B-A3B` (SM90) |

## Integrations

- **verl.** `isoexec.enable=true` on verl's Megatron/vLLM GRPO trainer selects IsoExec's Megatron
  engine, its vLLM Model Runner V2 rollout server and its trainer lifecycle together
  (`isoexec.integrations.verl`). `examples/verl_mrv2/run_case.sh` is the launcher for the
  qualification cases.
- **SkyRL.** IsoExec runs inside SkyRL through SkyRL's `isoexec` config field, as below.

## SkyRL

IsoExec runs inside [SkyRL](https://github.com/skyrl-project/SkyRL), installed as in the
Quickstart, through SkyRL's `isoexec` config field. Add one argument to a SkyRL Megatron/vLLM
recipe:

```sh
python -m skyrl.train.entrypoints.main_base \
    'isoexec={"model_package":"isoexec.models.qwen3_5"}' \
    trainer.strategy=megatron trainer.remove_microbatch_padding=true ...
```

`isoexec.model_package` selects the model; `examples/run_tiny.py` is a complete one-GPU recipe. The
policy and reference run on IsoExec's Megatron workers and the policy's vLLM engines serve the same
model on vLLM Model Runner V2, with `FULL_AND_PIECEWISE` CUDA graphs by default (`NONE` and
`FULL_DECODE_ONLY` are also accepted through `engine_init_kwargs.compilation_config.cudagraph_mode`).
Everything else (data, generator, PPO loop, placement, colocation, weight-sync transport) is SkyRL's.
Settings the IsoExec model cannot honor (LoRA, MTP, FP8, fused LM-head logprobs, sample-support
replay, torch.compile, checkpoint save/restore and HF export) are refused before any worker starts.

Every step, before the policy update, the trainer's logprob of each trainable response token is
compared bit for bit with the logprob the engine sampled it with; the step fails on the first token
that differs (`isoexec/rollout_train_logprob_mismatches` is 0 otherwise). With
`isoexec.full_logprob_comparison=true` the engine also returns each response token's full-vocabulary
row, and during the same forward the trainer compares every entry of every trainable row
(`policy/full_logprobs_verified_rows` counts them).

With the full comparison and colocated (CUDA IPC) weight sync, each sync sends every Parameter twice
and each engine rank compares the second copy with the bytes the first one wrote
(`[ISOEXEC-WEIGHT-SYNC] PASS`); non-colocated (NCCL) syncs are not checked this way.

### SkyRL Tinker API

SkyRL's Tinker API server builds IsoExec's backend when `backend_config` has the `isoexec`
mapping as one entry (dotted `isoexec.*` keys are refused). Use `backend=megatron` and a
full-parameter client (`rank=0`); the full comparison needs `runtime_role=combined`. LoRA, critic
models, multimodal inputs, sample-support replay/score centering, external inference servers,
training-checkpoint save/restore and persistent HF sampler export are refused.

```yaml
backend: megatron
runtime_role: combined
backend_config:
  isoexec:
    model_package: isoexec.models.qwen3_5
    full_logprob_comparison: true
  generator.inference_engine.engine_init_kwargs:
    max_model_len: 2048
```

A finite `trainer.algorithm.max_seq_len` or engine `max_model_len` bounds training inputs and
packed capture. The full comparison requires equal forward and training microbatch sizes and
sampling at temperature 1. Use `save_weights_and_get_sampling_client()` to acknowledge the current
weights, then sample and submit the exact prompt and response as next-token training data, with
prompt-token loss weights zero. The backend keeps each sample's full FP32 rows on the server,
compares every loss-active response position in a policy forward before the backward, and admits
`optim_step` only after a compared backward. Forward-only requests can score other text; they do not
authorize an update. Pending rows are bounded to 8 GiB and released at update and unload. After an
update, synchronize again and sample with the new client. A failed model pass or update requires
unloading and recreating the model.

Without `full_logprob_comparison`, Tinker requests run on IsoExec's workers and engine but nothing
compares the trainer with the engine.

With a preinstalled environment, start the API with
`uv run --isolated --no-project -- /path/to/python -m skyrl.tinker.api ...`; the engine subprocess
uses that same interpreter.

See [STATUS](./STATUS.md) for what is natively qualified.

## Host Requirements

- Linux x86_64 (glibc 2.28 or newer)
- NVIDIA driver 580.65.06 or newer (the CUDA 13.0 minimum)
- CUDA 13 toolkit; setup uses `$CUDA_HOME`, else `$CUDA_PATH`, else the toolkit whose `nvcc` is on `PATH`
- `gcc`, `g++`, `git`, `make`, `uv`, and `libnuma` (Debian/Ubuntu: `libnuma1`)

## Development

```sh
make setup-dev
```

After changing the SkyRL lock or `[overrides]` in `environments/default.toml`, run `make setup-dev` with a new 
`ISOEXEC_ROOT`: only this path checks `[[metadata_conflicts]]` against the installed lock; `make setup` does not rerun
this check.

## Documentation

- [Support a new model](./docs/support-new-model.md)
- [GDN modes: CPR and recurrent](./docs/gdn-modes.md)
- [Compiler design](./isoexec/compiler/DESIGN.md)
- [Status and limits](./STATUS.md)

## Acknowledgements

IsoExec builds on the work of these projects:

- [FlashMLA](https://github.com/deepseek-ai/FlashMLA) (DeepSeek): the SM90 and SM100 MLA kernels
  are derived from it.
- [FlashAttention](https://github.com/Dao-AILab/flash-attention) (Tri Dao): FA3 (vendored in
  `isoexec/ops/attention/gqa_sm90/fa3`) is the SM90 GQA attention kernel for trainer and engine, and an
  FA3 helper is used in the MLA kernels. FA4's SM100 backward (vendored in
  `isoexec/ops/_vendor/flash_attn_cute`) is the SM100 GQA backward, and the SM100 GQA forward's
  execution is derived from FA4's forward.
- [QuACK](https://github.com/Dao-AILab/quack) (Tri Dao and contributors): the CuTe DSL helpers FA4's
  backward imports, vendored beside it.
- [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) (Songlin Yang, Yu
  Zhang and contributors) and [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) (Tri Dao):
  the GDN kernels. The forward and convolution kernels are vendored by way of vLLM; the trainer's GDN
  backward is vendored directly from flash-linear-attention 0.5.2 (`isoexec/ops/gdn/shared/vendored_fla_bwd`).
- [vLLM](https://github.com/vllm-project/vllm): the inference runtime, and the source of the vendored
  GDN, convolution and batch-invariant GEMM kernels.
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) (NVIDIA): the training runtime, and the source
  of the vendored RoPE and attention output-gate references.
- [CUTLASS](https://github.com/NVIDIA/cutlass) (NVIDIA): CuTe/CUTLASS headers for the native kernels,
  fetched at a pinned commit by `make setup`.
- [verl](https://github.com/volcengine/verl): an RL integration.
- [SkyRL](https://github.com/skyrl-project/SkyRL): an RL integration.
- [torchtitan](https://github.com/pytorch/torchtitan): the varlen attention code adapted from it.

## License

[Apache-2.0](LICENSE). Third-party code included in this repository keeps its own license; see
[NOTICE](NOTICE) for the files, copyright holders and full license texts.
