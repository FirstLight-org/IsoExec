# IsoExec package

The top-level `isoexec` package exports `IsoExecConfig`, `ModelSource` and
`ModelSourceError`. See the repository [README](../README.md) for installation,
[Support a new model](../docs/support-new-model.md) for the model package contract,
the [compiler design](compiler/DESIGN.md) for the internals, and [STATUS](../STATUS.md)
for measured support and limits.

Importing the package initializes no CUDA context and imports no trainer or
serving framework. The SkyRL integration constructs the Megatron trainer model and
registers the vLLM engine model in each worker.

| Directory | Responsibility |
| --- | --- |
| `models/` | Built-in model packages (`model.py` + `weight.py`) and the `ModelSource` descriptor. |
| `nn/` | Model components (`Attention`, `GDN`, `MLP`, `MoE`, norms, embedding, head). |
| `compiler/` | Source reconstruction, export capture, stage partitioning, placement and checkpoint realization. |
| `core/`, `ops/` | Operator registration, numerical kernels and their bindings. |
| `schema/`, `state/`, `storage/`, `weights/` | Typed tensor/state contracts, packed storage and checkpoint weight operations. |
| `checks/` | Configuration, dependency, logprob and weight compatibility checks. |
| `runtimes/` | Megatron trainer and vLLM engine adapters. |
| `sync/` | Trainer-to-engine weight synchronization. |
| `integrations/` | SkyRL configuration, training, scoring and full-distribution transport. |

The native training loop recomputes policy scores before an update and checks
exact equality against rollout scores for valid tokens from the same policy.
These checks enforce agreement; architecture support alone does not certify every
checkpoint, parallel layout or training feature.

SkyRL keeps configuration fields and the call sites that supply live batches,
worker configuration and weight transfers. The implementation lives in
`integrations/skyrl/`: `bootstrap.py` validates and resolves the run at the end of
SkyRL's `validate_cfg`; `config.py` writes the actors' environment and edits the
policy's vLLM arguments; `inference.py` prepares full-row requests and returns the
rows; `audit.py` brackets the native policy forward with the comparison and checks
the optimizer step; `model_wrapper.py` supplies scoring and gradient callbacks
through a subclass of SkyRL's native wrapper; and `tinker.py` is the backend SkyRL's
Tinker engine builds when `backend_config` has the `isoexec` mapping, with
`tinker_state.py` holding the sampled rows between requests. The native RL loop, batching, deferred gradient synchronization and
transport remain owned by SkyRL. Shared computations continue to live in
`runtimes/`, `ops/` and `checks/`, rather than being copied into each integration.
Enable full-vocabulary comparison with `isoexec.full_logprob_comparison=true`.
This single IsoExec option selects the inference response mode and the trainer's
pre-update comparison; SkyRL has no separate comparison/output-mode config fields.
`weight_transfer.py` registers the IsoExec receiver through vLLM's plugin registry,
and `weight_receiver.py` preserves the native NCCL/IPC receive path while finishing
registered model updates and releasing imported IPC buffers.
Capture bounds are derived from SkyRL's existing batch and token limits. A custom
trajectory producer can supply `isoexec.trainer_capture_max_tokens` when
those limits cannot bound its packed input; this adds no SkyRL configuration field.
