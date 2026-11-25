"""verl's vLLM rollout server with the IsoExec model on vLLM Model Runner V2.

verl keeps its server actor, agent loop, request admission, sleep/wake and colocated weight
transport. This server only:

- selects Model Runner V2 and the IsoExec engine environment before vLLM builds its config;
- registers the native model (``registration.register_native``) through verl's engine kwargs,
  refusing verl settings that would contradict it instead of overriding them silently;
- requests complete raw FP32 rows (``logprobs=-1``) and publishes each request's rows to the run
  directory under its token-derived key (``full_distribution.write_rows``);
- opens generation for a weight version only after every worker of its replica acknowledged it.

Selected with ``actor_rollout_ref.rollout.agent.custom_async_server`` naming
``IsoExecvLLMHttpServer`` in this module.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import numpy as np
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

from isoexec.integrations.full_distribution import FullVocabularyRow

#: FULL graphs for uniform decode; mixed (prefill-bearing) batches replay vLLM's breakable PIECEWISE
#: graphs with attention and GDN cores as eager breaks (runtimes/vllm/graph_breaks.py). Bitwise
#: identical to NONE and FULL_DECODE_ONLY on the same engine and requests.
CUDAGRAPH_MODE = "FULL_AND_PIECEWISE"


def cudagraph_mode(engine_kwargs: dict) -> str:
    """``rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode`` when the caller set it, else
    :data:`CUDAGRAPH_MODE`. ``register_native`` admits NONE, FULL_DECODE_ONLY and FULL_AND_PIECEWISE,
    all bitwise identical on the same engine and requests, and refuses any other mode."""
    import json

    compilation = engine_kwargs.get("compilation_config") or {}
    if isinstance(compilation, str):
        compilation = json.loads(compilation)
    mode = compilation.get("cudagraph_mode") if isinstance(compilation, dict) else getattr(compilation,
                                                                                            "cudagraph_mode", None)
    return CUDAGRAPH_MODE if mode is None else str(getattr(mode, "name", mode))


def configure_process_environment(model_path: str) -> dict[str, str]:
    """Export the IsoExec engine environment in this server process before vLLM reads it.

    vLLM's multiprocessing workers inherit this environment.
    """
    from isoexec.runtimes.environment import resolved_environment
    from isoexec.runtimes.vllm.vllm_patches import apply_vllm_isoexec_env

    env = resolved_environment(model_path)
    for key, value in env.items():
        os.environ[key] = value
    os.environ["VLLM_PLUGINS"] = "isoexec"
    apply_vllm_isoexec_env()
    return env


def validate_rollout_config(config: Any) -> None:
    """Settings the IsoExec engine requires; a conflicting value is refused, never replaced."""
    expected = {
        "calculate_log_probs": True,
        "enforce_eager": False,
        "logprobs_mode": "raw_logprobs",
        "dtype": "bfloat16",
        "topk_log_probs": 0,
    }
    for name, value in expected.items():
        actual = getattr(config, name, None)
        if name == "topk_log_probs":
            actual = actual or 0
        if actual != value:
            raise ValueError(f"IsoExec rollout requires actor_rollout_ref.rollout.{name}={value!r}, got {actual!r}")
    from isoexec.runtimes.vllm.gdn_engine_patch import (
        STATE_RESUMING_ENGINE_ARGS,
        enable_state_features,
        resumable_engine_args_for,
    )

    mtp = getattr(config, "mtp", None)
    speculative = bool(mtp is not None and getattr(mtp, "enable", False) and getattr(mtp, "enable_rollout", False))
    admitted, reason = resumable_engine_args_for(speculative=speculative)
    for name in STATE_RESUMING_ENGINE_ARGS:
        if name not in admitted and getattr(config, name, False):
            raise ValueError(f"IsoExec rollout requires actor_rollout_ref.rollout.{name}=False here: {reason}")
    # The engine flags an enabled feature needs under CPR (the driver forwards them; vLLM's engine processes
    # inherit this process's environment). A flag set off explicitly is refused here, before the engine.
    enable_state_features([name for name in STATE_RESUMING_ENGINE_ARGS if getattr(config, name, False)])
    if getattr(config, "enable_rollout_routing_replay", False):
        raise ValueError("IsoExec rollout does not support rollout routing replay")
    refuse_engine_context_parallel(config)


def refuse_engine_context_parallel(rollout_config: Any) -> None:
    """Refuse vLLM prefill/decode context parallelism (``rollout.engine_kwargs.vllm``) for the IsoExec engine.

    Trainer context parallelism (``megatron.context_parallel_size``, any degree) with an engine without
    context parallelism is the supported configuration.
    """
    from isoexec.checks import Refusal
    from isoexec.runtimes.vllm.validation import engine_kwargs_context_parallel_refusal

    engine_kwargs = getattr(rollout_config, "engine_kwargs", None) or {}
    vllm_kwargs = engine_kwargs.get("vllm", None) if hasattr(engine_kwargs, "get") else None
    refusal = engine_kwargs_context_parallel_refusal(vllm_kwargs)
    if refusal is not None:
        raise Refusal(f"actor_rollout_ref.rollout.engine_kwargs.vllm: {refusal}")


def carry_state_features(config: Any, engine_kwargs: dict) -> None:
    """Put the rollout's chunked-prefill / prefix-caching choice into ``engine_kwargs`` before registration.

    verl passes these to vLLM from the rollout config, with ``engine_kwargs`` (what ``register_native`` sees
    and returns) applied last, so the registration must decide on the values the engine will really get.
    An explicit ``rollout.engine_kwargs.vllm`` value wins, as it does in verl."""
    from isoexec.runtimes.vllm.gdn_engine_patch import STATE_RESUMING_ENGINE_ARGS

    for name in STATE_RESUMING_ENGINE_ARGS:
        value = getattr(config, name, None)
        if value is not None and name not in engine_kwargs:
            engine_kwargs[name] = bool(value)


def expected_workers(config: Any) -> int:
    """Workers in one replica: verl's TP x DP x PP, times vLLM's prefill context parallel size."""
    from verl.workers.config.rollout import rollout_context_parallel_size

    return (config.tensor_model_parallel_size * config.data_parallel_size
            * config.pipeline_model_parallel_size * rollout_context_parallel_size(config))


class IsoExecvLLMHttpServer(vLLMHttpServer):
    """verl vLLM server serving an IsoExec model with full-row publication and version acks."""

    def __init__(self, config, model_config, *args, **kwargs):
        # verl converts its Hydra configs here; vLLM's engine config is built later, in launch_server,
        # so the IsoExec environment and plugin are still in place before vLLM reads them.
        super().__init__(config, model_config, *args, **kwargs)
        configure_process_environment(str(self.model_config.local_path))
        from isoexec.runtimes.vllm.registration import register_plugin

        register_plugin()
        validate_rollout_config(self.config)
        self._isoexec_ready_version: int | None = None
        self._isoexec_ready_lock = asyncio.Lock()

    # ------------------------------------------------------------- engine config
    def _preprocess_engine_kwargs(self, engine_kwargs: dict) -> None:
        super()._preprocess_engine_kwargs(engine_kwargs)
        from isoexec.runtimes.vllm.registration import register_native

        from .model import checkpoint_directory, source_config

        path = str(checkpoint_directory(self.model_config.local_path))
        source, raw = source_config(path)
        from .full_distribution import full_rows_enabled

        required = {"logprobs_mode": "raw_logprobs", **({"max_logprobs": -1} if full_rows_enabled() else {})}
        for key, value in required.items():
            if key in engine_kwargs and engine_kwargs[key] != value:
                raise ValueError(f"rollout.engine_kwargs.vllm.{key} must be {value!r} for full logprobs")
            engine_kwargs[key] = value
        temperature = float(self.config.temperature)
        carry_state_features(self.config, engine_kwargs)
        native = register_native(source, raw, checkpoint=path, engine_kwargs=engine_kwargs,
                                 score_temperature=temperature, cudagraph_mode=cudagraph_mode(engine_kwargs))
        model = native.pop("model")
        if str(checkpoint_directory(model)) != path:
            raise ValueError(f"IsoExec registration selected {model}, not the rollout checkpoint {path}")
        engine_kwargs.clear()
        engine_kwargs.update(native)
        from isoexec.ops.gdn import native_state_enabled

        if native_state_enabled():
            # The GDN state lives in vLLM's mamba pages; "auto" resolves the ssm cache to bf16.
            if engine_kwargs.get("mamba_ssm_cache_dtype", "float32") != "float32":
                raise ValueError("IsoExec native GDN state requires mamba_ssm_cache_dtype=float32")
            engine_kwargs["mamba_ssm_cache_dtype"] = "float32"

    def _get_worker_extension_cls(self) -> str:
        return "isoexec.integrations.verl.worker_extension.IsoExecWorkerExtension"

    def _resolve_sleep_level(self) -> int:
        # Level 2 is exact for an IsoExec model (runtimes/vllm/weights_sleep): the Parameters, which every
        # coverage-checked update rewrites, are discarded; every other weights-pool byte is kept in a pinned
        # stash and restored at the weights wake. ISOEXEC_VLLM_EXACT_SLEEP=0 falls back to level 1 (offload
        # and restore the whole pool). verl's own reasons for level 1 (MTP drafter weights, LoRA adapters) stand.
        from isoexec.runtimes.vllm.weights_sleep import exact_sleep_enabled

        return super()._resolve_sleep_level() if exact_sleep_enabled() else 1

    # ------------------------------------------------------------------ requests
    def _preprocess_sampling_params(self, sampling_params: dict[str, Any]) -> None:
        if sampling_params.get("logprobs", None) is None:
            raise ValueError("IsoExec rollout requires per-token logprobs on every request")
        if sampling_params.get("n", 1) != 1:
            raise ValueError("IsoExec rollout requires n=1 per request")
        from .full_distribution import full_rows_enabled, rows_suspended, run_root

        # The sampled token's raw logprob only (compared bitwise by the trainer). The full-vocabulary debug
        # mode asks for every row, except during validation, which publishes none and may decode greedily.
        sampling_params["logprobs"] = -1 if full_rows_enabled() and not rows_suspended(run_root()) else 0

    async def generate(self, prompt_ids, sampling_params, request_id, *args, **kwargs):
        await self._isoexec_require_acks()
        return await super().generate(prompt_ids, sampling_params, request_id, *args, **kwargs)

    async def _isoexec_require_acks(self) -> None:
        version = self.global_steps
        if version is None:
            raise RuntimeError("IsoExec rollout: no weight version has been synchronized yet")
        if self._isoexec_ready_version == version:
            return
        async with self._isoexec_ready_lock:
            if self._isoexec_ready_version == version:
                return
            self._isoexec_check_acks(int(version))
            self._isoexec_ready_version = version

    def _isoexec_check_acks(self, version: int) -> None:
        from .full_distribution import append_receipt, run_root
        from .worker_extension import ack_directory

        root = run_root()
        directory = ack_directory(root, version)
        prefix = f"replica{self.replica_rank}-"
        acks = [json.loads(path.read_text()) for path in sorted(directory.glob(f"{prefix}*.json"))] \
            if directory.exists() else []
        count = expected_workers(self.config)
        ranks = {(ack["dp_rank"], ack["rank"]) for ack in acks}
        if len(acks) != count or len(ranks) != count or any(ack["version"] != version for ack in acks):
            raise RuntimeError(
                f"IsoExec rollout replica {self.replica_rank}: weight version {version} has {len(acks)} worker "
                f"acknowledgements, expected {count}; generation stays closed"
            )
        runners = sorted({ack["runner_class"] for ack in acks})
        if not all(ack["use_v2_model_runner"] for ack in acks) or any(
                not runner.startswith("vllm.v1.worker.gpu.model_runner.") for runner in runners):
            raise RuntimeError(f"IsoExec rollout requires vLLM Model Runner V2 on every worker, got {runners}")
        append_receipt(root, "rollout-gate.jsonl", {
            "event": "generation_opened", "replica_rank": self.replica_rank, "version": version,
            "workers": count, "runner_classes": runners,
            "parameter_sha256": sorted({ack["parameter_sha256"] for ack in acks}),
        })

    def _postprocess_output(self, output, final_res, sampling_params):
        output = super()._postprocess_output(output, final_res, sampling_params)
        if output.stop_reason == "aborted" or not final_res.outputs:
            raise RuntimeError("IsoExec rollout: an aborted request has no complete rows to compare")
        from .full_distribution import full_rows_enabled, rows_suspended, run_root, write_rows

        if not full_rows_enabled() or rows_suspended(run_root()):
            # Validation generation: nothing rescores these requests, so no rows are published.
            return output
        completion = final_res.outputs[0]
        rows: list[Any] | None = completion.logprobs  # type: ignore[assignment]
        token_ids = list(completion.token_ids)
        if rows is None or len(rows) != len(token_ids):
            raise RuntimeError("IsoExec rollout: complete rows must cover every generated token")
        if not all(isinstance(row, FullVocabularyRow) for row in rows):
            raise RuntimeError("IsoExec rollout: vLLM returned per-entry logprob containers, not complete rows")
        matrix = np.stack([row._values for row in rows]).astype("<f4", copy=False)
        if any(row.sampled_token_id != token for row, token in zip(rows, token_ids, strict=True)):
            raise RuntimeError("IsoExec rollout: a complete row's sampled token differs from the response token")
        reference = write_rows(run_root(), version=int(self.global_steps), prompt_ids=list(final_res.prompt_token_ids),
                               response_ids=token_ids, rows=matrix, request_id=str(final_res.request_id))
        output.extra_fields["isoexec_full_rows"] = reference
        return output


__all__ = ["CUDAGRAPH_MODE", "IsoExecvLLMHttpServer", "configure_process_environment", "cudagraph_mode", "expected_workers",
           "refuse_engine_context_parallel", "validate_rollout_config"]
