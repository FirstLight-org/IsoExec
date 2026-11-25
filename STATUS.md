# IsoExec status

This release ships one model API, two runtimes and two RL integrations:

| Area | Shipped |
| --- | --- |
| Model packages | `isoexec.models.qwen3_5` (Qwen3.5 dense), `isoexec.models.qwen3_5_moe` (Qwen3.5 MoE text), `isoexec.models.qwen2_moe` (Qwen1.5-MoE), `isoexec.models.qwen3` (Qwen3 dense), `isoexec.models.qwen3_moe` (Qwen3 MoE) |
| Trainer runtime | Megatron (`isoexec.runtimes.megatron`) |
| Inference runtime | vLLM (`isoexec.runtimes.vllm`) |
| Integrations | verl (`isoexec.integrations.verl`): verl's GRPO trainer with IsoExec's Megatron engine and vLLM Model Runner V2 rollouts, enabled by `isoexec.enable=true` |
| | SkyRL (`isoexec.integrations.skyrl`): Megatron training with vLLM rollouts (Model Runner V2) on SkyRL, through SkyRL's `isoexec` config field |

A model package is an `nn.Module` built from `isoexec.nn` components (`model.py`) plus its
Hugging Face checkpoint conversions (`weight.py`), selected with
`isoexec.model_package`. See [Support a new model](docs/support-new-model.md) and the
[compiler design](isoexec/compiler/DESIGN.md).

## Native qualification

SkyRL, on 4 x H200 (SM90) with vLLM 0.33.0 (Model Runner V2) and the SkyRL revision pinned in
`environments/default.toml`, 2026-11-22 to 2026-11-23. Each run is one training step, colocated
unless noted, text-only, BF16. Every engine runs `FULL_AND_PIECEWISE` CUDA graphs, prefix caching
and chunked prefill.

| Run | Model | GPUs | Engine | Trainer | Passed |
| --- | --- | --- | --- | --- | --- |
| `main_base` case 1 | Qwen/Qwen3.5-0.8B `2fc06364` | 1 | TP1, CUDA IPC sync | TP1 PP1 | PRESCORE (12565 rows), WEIGHT-SYNC; CPR prefix caching on |
| `main_base` case 2 | Qwen/Qwen1.5-MoE-A2.7B `1a758c50` | 4 | TP2 DP2 EP4 | TP1 CP2 DP2 EP4 ETP1, SP off | PRESCORE (9939 rows), WEIGHT-SYNC on 4 ranks |
| `main_base` case 3 | Qwen/Qwen1.5-MoE-A2.7B `1a758c50` | 4 | TP2 PP2 | TP2 PP2 EP2 ETP1, SP | PRESCORE (6197 rows), WEIGHT-SYNC |
| Tinker | Qwen/Qwen3.5-0.8B `2fc06364` | 1 | TP1, CUDA IPC sync | TP1 PP1 | PRESCORE (384 rows in each of 2 rounds), WEIGHT-SYNC |
| Tinker, not colocated | Qwen/Qwen3.5-0.8B `2fc06364` | 2 | TP1, NCCL sync | TP1 PP1 | PRESCORE (2 rounds) |

- PRESCORE (`[ISOEXEC-PRESCORE]`): every trainable row of the policy forward equals the engine's
  FP32 full-vocabulary row bit for bit (`full_logprob_comparison: true`).
- WEIGHT-SYNC (`[ISOEXEC-WEIGHT-SYNC]`): each engine rank's applied parameter bytes equal the
  trainer's. Checked for CUDA IPC syncs only; NCCL syncs are not byte-checked.
- The runs also carried test-only checks that are not part of the adapter: every gradient value was
  finite before each optimizer step, and the step changed 261 (case 1) and 262 (both Tinker runs) of
  362 parameters. The MoE steps changed none; a GSM8K step on the base model can have all-zero
  gradients.
- In each Tinker run the scheduler admitted one 896-token prefix-cache hit, in a round whose
  PRESCORE passed. Case 1 admitted none.

On 2026-10-04, with an earlier version of this adapter, Qwen/Qwen3.5-0.8B (revision
`2fc06364715b967f1860aea9cf38778875588b17`) completed real SkyRL runs on H100 at PP1 (4 GPUs) and
PP2 (8 GPUs): Megatron TP2/DP2 with SP, two vLLM TP2 engines, colocated, text-only, BF16, MTP off.
Each run passed the W0 full-vocabulary logprob comparison, one native optimizer update, W1 weight
sync and the W1 comparison. Details and the remaining one-ULP BF16 weight differences after the
update are recorded in the
[native qualification snapshot](isoexec/compiler/DESIGN.md#skyrl-native-qualification-snapshot).

`make run-tiny` repeats a one-GPU, one-step GRPO run of Qwen3.5-0.8B on GSM8K.

The verl integration runs on Model Runner V2 (CUDA-graph mode `FULL_AND_PIECEWISE`). Its
qualification cases (dense, MoE, Qwen1.5-MoE) are launched by `examples/verl_mrv2/run_case.sh`:
W0 weight sync, a rollout whose every full-vocabulary row is compared with Megatron rescoring, one
optimizer update, W1 sync and a second comparison.

## Limits

- Larger models, PP4 and parallel layouts other than the ones above are not natively qualified; CPU tests cover
  `qwen3_5_moe`, `qwen2_moe`, `qwen3` and `qwen3_moe` geometry, checkpoint conversion and partitioning
  (and Transformers parity for `qwen3`/`qwen3_moe` on tiny configs). The Qwen3 plain-gamma q/k norm
  runs on SM90 only; SM100 refuses it until its kernel path is verified there.
- Native training checkpoint save/restore and Hugging Face export are unsupported; SkyRL
  refuses positive `trainer.ckpt_interval`/`trainer.hf_save_interval` and resume requests.
- MTP and speculative decoding are unsupported.
- Performance and long-run endurance are not qualified. Native Hugging Face bitwise fidelity
  is not claimed.
