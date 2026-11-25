"""Register the IsoExec model with vLLM and install the engine stack in each vLLM process."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from importlib import import_module
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from isoexec.compiler.source import ModelSource

#: Architecture names ``register_native`` writes into ``hf_overrides``. Graphs with GDN state layers
#: take vLLM's hybrid cache configuration; graphs without them must not.
NATIVE_ARCH = "VllmModel"
NATIVE_DENSE_ARCH = "DenseVllmModel"


def register_native_model_class() -> None:
    """Register the native constructors and hybrid configuration without building a model."""
    registry = import_module("vllm.model_executor.models.registry").ModelRegistry
    registry.register_model(NATIVE_ARCH, "isoexec.runtimes.vllm.factory:VllmModel")
    registry.register_model(NATIVE_DENSE_ARCH, "isoexec.runtimes.vllm.factory:DenseVllmModel")
    configs = import_module("vllm.model_executor.models.config")
    configs.MODELS_CONFIG_MAP.setdefault(NATIVE_ARCH, configs.HybridAttentionMambaModelConfig)


#: CUDA graph modes that replay PIECEWISE graphs (breakable, without torch.compile).
PIECEWISE_MODES = ("FULL_AND_PIECEWISE",)


def engine_environment(cudagraph_mode: str) -> dict[str, str]:
    """The environment every process that builds a ``VllmConfig`` for ``cudagraph_mode`` needs.

    vLLM decides at config build whether PIECEWISE graphs can run without torch.compile, from
    ``VLLM_USE_BREAKABLE_CUDAGRAPH``; without it ``FULL_AND_PIECEWISE`` silently becomes
    ``FULL_DECODE_ONLY``. Launchers that build the engine in another process (a Ray actor, an API
    server) must carry this into it.
    """
    return {"VLLM_USE_BREAKABLE_CUDAGRAPH": "1"} if cudagraph_mode in PIECEWISE_MODES else {}


def requested_cudagraph_mode(vllm_config) -> str | None:
    """The mode ``register_native`` was asked for, as recorded in the engine's ``additional_config``."""
    return ((getattr(vllm_config, "additional_config", None) or {}).get("isoexec") or {}).get("cudagraph_mode")
#: Engine settings a model package may choose with ``engine_settings()`` on its model class, with
#: release's defaults (``None``: not set here). A model that declares ``None`` leaves the setting to
#: vLLM's own selection. Everything else ``register_native``
#: pins is part of the IsoExec engine contract and is not model-selectable.
ENGINE_SETTING_DEFAULTS: Mapping[str, Any] = MappingProxyType({
    "attention_backend": "CUSTOM",
    "enable_chunked_prefill": False,
    "kv_cache_dtype": None,
    "block_size": None,
})


def model_engine_settings(model: object) -> dict[str, Any]:
    """Validated engine settings an author model declares; ``{}`` when it declares none."""
    declare = getattr(model, "engine_settings", None)
    if not callable(declare):
        return {}
    settings = dict(declare())
    unknown = sorted(set(settings) - set(ENGINE_SETTING_DEFAULTS))
    if unknown:
        raise ValueError(f"engine_settings(): {unknown} are not model-selectable engine settings "
                         f"(selectable: {sorted(ENGINE_SETTING_DEFAULTS)})")
    return settings


def register_native(source: ModelSource, raw_config: Mapping[str, Any], *, checkpoint: str,
                    engine_kwargs: Mapping[str, Any] | None = None,
                    score_temperature: float = 1.0, cudagraph_mode: str = "NONE",
                    enable_chunked_prefill: bool = False) -> dict[str, Any]:
    """Return native LLM kwargs for the BF16 model.

    ``raw_config`` is the original HF config.json before architecture overrides.
    Source configuration stays in native additional_config for worker-side
    construction. The ordinary vLLM loader owns checkpoint iteration and loading.
    An exact file supplies its directory for model configuration and its filename
    as the native loader's weight pattern. ``cudagraph_mode`` is NONE,
    FULL_DECODE_ONLY or FULL_AND_PIECEWISE; graphs are captured without
    compilation. FULL_AND_PIECEWISE replays mixed batches through vLLM's breakable
    PIECEWISE graphs (exported ``VLLM_USE_BREAKABLE_CUDAGRAPH=1``, inherited by the
    engine processes) with attention and GDN cores as eager breaks. Models with GDN
    layers select the hybrid native class; others select the dense class.
    Chunked prefill and prefix caching follow the GDN state composition
    (``gdn_state_engine_args``): under the native recurrent composition, which resumes
    exactly from vLLM's own state blocks, the ``engine_kwargs`` values (or vLLM's
    defaults) stand; under CPR on its private pool, an explicit ``True`` stands and turns
    on the engine flags it needs (the pool's chunked-prefill resume, the prefix-cache
    boundary store); otherwise both are forced off. Prefix caching on a hybrid model also
    turns chunked prefill on (vLLM's mamba prefix caching requires it).
    ``enable_chunked_prefill=True`` is the caller's explicit opt-in, which the GDN binding
    admits only for compositions that resume exactly
    (``gdn_engine_patch.assert_engine_args_compatible``).
    A model package may choose the engine settings listed in ``ENGINE_SETTING_DEFAULTS``
    (``engine_settings()`` on its model class: its own cache encoding, block size,
    chunked prefill); a model's declaration takes precedence over the caller's
    """
    from pathlib import Path

    import isoexec.nn
    from isoexec.compiler.weights import local_checkpoint_path

    if cudagraph_mode not in ("NONE", "FULL_DECODE_ONLY", "FULL_AND_PIECEWISE"):
        raise ValueError(f"cudagraph_mode={cudagraph_mode!r}: expected NONE, FULL_DECODE_ONLY or FULL_AND_PIECEWISE")
    for name, value in engine_environment(cudagraph_mode).items():
        if os.environ.get(name, value) != value:
            raise ValueError(f"cudagraph_mode={cudagraph_mode!r} requires {name}={value}")
        # This process (an in-process ``LLM``) builds its config here; other launchers carry it.
        os.environ[name] = value
    from isoexec.checks import Refusal

    from .validation import engine_kwargs_context_parallel_refusal

    refusal = engine_kwargs_context_parallel_refusal(engine_kwargs)
    if refusal is not None:
        raise Refusal(refusal)
    kwargs = dict(engine_kwargs or {})
    additional = dict(kwargs.get("additional_config") or {})
    settings = dict(additional.get("isoexec") or {})
    settings.update(model_source=source.to_dict(), raw_model_config=dict(raw_config),
                    score_temperature=score_temperature, cudagraph_mode=cudagraph_mode)
    path = Path(checkpoint).expanduser()
    settings.pop("checkpoint_path", None)
    if path.is_file():
        path = local_checkpoint_path(path)
        settings["checkpoint_path"] = str(path)
        kwargs["served_model_name"] = kwargs.get("served_model_name") or [str(path)]
        checkpoint = str(path.parent)
    additional["isoexec"] = settings
    overrides = dict(kwargs.get("hf_overrides") or {})
    meta = source.build_meta(raw_config)
    hybrid = any(isinstance(module, isoexec.nn.GDN) for module in meta.modules())
    declared = model_engine_settings(meta)
    overrides["architectures"] = [NATIVE_ARCH if hybrid else NATIVE_DENSE_ARCH]
    if raw_config.get("quantization_config") is not None:
        # The package's checkpoint conversions read the stored encodings themselves (raw_config keeps
        # the declaration); vLLM must not also apply its own quantization method to the layers.
        overrides["quantization_config"] = None
    compilation = compilation_config_values(kwargs.get("compilation_config"))
    compilation["mode"] = 0  # CompilationMode.none
    compilation["cudagraph_mode"] = cudagraph_mode
    # pik collectives need one eager pass per shape before capture freezes them. Model Runner V2
    # runs exactly one eager warmup per FULL and PIECEWISE descriptor before capturing it.
    compilation["cudagraph_num_of_warmups"] = max(1, compilation.get("cudagraph_num_of_warmups", 0))
    if cudagraph_mode != "NONE" and int(kwargs.get("max_num_seqs") or 0) > 512:
        compilation.setdefault("max_cudagraph_capture_size", kwargs["max_num_seqs"])
    kwargs.update(model=checkpoint, additional_config=additional, hf_overrides=overrides,
                  load_format="auto", dtype="bfloat16", enforce_eager=cudagraph_mode == "NONE",
                  compilation_config=compilation,
                  attention_backend="CUSTOM", logprobs_mode="raw_logprobs",
                  disable_cascade_attn=True, mamba_ssm_cache_dtype="float32")
    if enable_chunked_prefill:
        # The caller's explicit opt-in (the private-pool resume, ISOEXEC_GDN_CHUNKED_PREFILL): the engine
        # gate (gdn_engine_patch.assert_engine_args_compatible) admits or refuses it.
        kwargs["enable_chunked_prefill"] = True
    # Same rule for every graph (a dense one has no GDN state, but the verl driver deciding its
    # config cannot tell the two apart, and the attention path is the same split-exact one).
    forced = gdn_state_engine_args(kwargs, hybrid=hybrid)
    if enable_chunked_prefill:
        forced.pop("enable_chunked_prefill", None)
    kwargs.update(forced)
    if declared:
        # V4.1 owns its paged caches and the encodings passed to its attention backend.
        kwargs["enable_prefix_caching"] = False
        for name, value in declared.items():
            if value is None:
                kwargs.pop(name, None)
            else:
                kwargs[name] = value
    register_plugin()
    return kwargs


def gdn_state_engine_args(kwargs: Mapping[str, Any], *, hybrid: bool = True) -> dict[str, Any]:
    """Chunked prefill and prefix caching for a GDN model: the caller's choice where admitted, else off.

    The decision is the engine gate's own (``gdn_engine_patch.resumable_engine_args_for``): under the
    native recurrent composition both features resume exactly from vLLM's state blocks, so a value
    the caller set (or vLLM's default, when unset) stands. Under CPR on its private pool a feature the
    caller set to ``True`` stands and this process exports the engine flags it needs
    (``gdn_engine_patch.enable_state_features``; each engine process also turns them on from its own
    configuration); an unset one is off. Otherwise each is forced off, printing why. On a ``hybrid``
    model, prefix caching turns chunked prefill on (vLLM requires it for mamba prefix caching). Returns
    only the keys that change.
    """
    from .gdn_engine_patch import (
        STATE_RESUMING_ENGINE_ARGS,
        cpr_pool_composition,
        enable_state_features,
        resumable_engine_args_for,
    )

    allowed, reason = resumable_engine_args_for(speculative=kwargs.get("speculative_config") is not None)
    explicit = cpr_pool_composition()  # the private pool runs a feature only when asked for it
    out: dict[str, Any] = {}
    for name in STATE_RESUMING_ENGINE_ARGS:
        if name in allowed and (not explicit or kwargs.get(name) is not None):
            continue
        if kwargs.get(name, None) is not False:
            if kwargs.get(name):
                print(f"[ISOEXEC-VLLM] {name}: {kwargs[name]!r} -> False ({reason})", flush=True)
            out[name] = False
    final = {name: out.get(name, kwargs.get(name)) for name in STATE_RESUMING_ENGINE_ARGS}
    if hybrid and final["enable_prefix_caching"] and final["enable_chunked_prefill"] is False:
        print("[ISOEXEC-VLLM] enable_chunked_prefill: False -> True (vLLM's prefix caching on a hybrid model "
              "requires chunked prefill)", flush=True)
        out["enable_chunked_prefill"] = final["enable_chunked_prefill"] = True
    on = [name for name in STATE_RESUMING_ENGINE_ARGS if final[name]]
    flags = enable_state_features(on)
    if flags:
        print(f"[ISOEXEC-VLLM] {', '.join(on)} under the CPR GDN composition: set "
              + " ".join(f"{k}={v}" for k, v in flags.items()), flush=True)
    return out


def _drop_unset(config):
    """Recursively omit dataclass fields whose value is their None default. CLI JSON round trips
    otherwise turn absent optional booleans into explicit null values that vLLM rejects.
    """
    out = {}
    for field in fields(config):
        if not field.init:
            continue
        value = getattr(config, field.name)
        if field.default is None and value is None:
            continue
        if is_dataclass(value) and not isinstance(value, type):
            value = _drop_unset(value)
        out[field.name] = value
    return out


def compilation_config_values(config):
    """Preserve the native CLI's dataclass or JSON configuration as plain kwargs."""
    if isinstance(config, str):
        try:
            config = json.loads(config)
        except json.JSONDecodeError as error:
            raise ValueError("compilation_config: expected a JSON object") from error
    if config is None:
        config = {}
    if is_dataclass(config) and not isinstance(config, type):
        config = _drop_unset(config)
    if not isinstance(config, Mapping):
        raise ValueError("compilation_config: expected a mapping, dataclass, or JSON object")
    return dict(config)


_PLUGIN_INSTALLED = False


def register_plugin() -> None:
    """Register the native model classes, attention backend and vLLM hooks in this process.

    Runs where ``register_native`` builds the kwargs and, through the general plugin, in every vLLM
    process (API server, engine core, each worker): vLLM's model registry and the patched classes
    are per process, and spawned processes resolve ``hf_overrides.architectures`` against their own
    registry. Importing this module must not initialize CUDA.
    """
    global _PLUGIN_INSTALLED
    # vLLM forwards its VLLM_* environment namespace to the processes it spawns, so the plugin in
    # each of them knows this job opted in.
    os.environ["VLLM_ISOEXEC_REGISTERED"] = "1"
    if _PLUGIN_INSTALLED:
        return
    register_native_model_class()
    from isoexec.ops.attention import (
        varlen_backend,  # noqa: F401  -- @register_backend(CUSTOM)
    )
    from isoexec.runtimes.vllm.vllm_patches import (
        patch_vllm_cumem_sleep,
        patch_vllm_full_vocabulary_logprobs,
        patch_vllm_kernel_warmup,
        patch_vllm_load_timers,
        patch_vllm_v2_cudagraph_census,
        patch_vllm_v2_fused_sampler,
        patch_vllm_v2_request_census,
        patch_vllm_v2_sampler_logprobs_rowinv,
        patch_vllm_v2_shard_sampler,
    )

    # Model Runner V2 gathers sampled and full-vocabulary scores in its own module: the rowinv producer.
    patch_vllm_v2_sampler_logprobs_rowinv()
    patch_vllm_v2_cudagraph_census()
    patch_vllm_v2_request_census()
    # Eligible V2 sampling steps run on each rank's vocab shard (O(rows) exchange, same bits).
    patch_vllm_v2_shard_sampler()
    # LogprobsProcessor lives in the API server and holds each logprobs=-1 row until the
    # response finishes, so this hook has to be installed there, not only in the workers.
    patch_vllm_full_vocabulary_logprobs()
    # Plain decode steps (no top-k/top-p, min_p or per-request seed) take the arch's decode sampler.
    patch_vllm_v2_fused_sampler()
    # In every worker process, before any engine sleeps: see the patch for the cumem double unmap
    # (first seen on vLLM 0.28, still present in 0.33).
    patch_vllm_cumem_sleep()
    # IsoExec models own their kernels: vLLM's SM100 CuTe-DSL router warmup has nothing to build.
    patch_vllm_kernel_warmup()
    # Per-component cold-start timers of the worker (core/load_timers), emitted once after the first step.
    patch_vllm_load_timers()

    # vLLM caches the GDN backend's batch-invariance answer; lift the veto before anything asks.
    # GDN core binding lifts it again, idempotently.
    if os.environ.get("ISOEXEC_GDN") == "1":
        from isoexec.runtimes.vllm.gdn_engine_patch import lift_gdn_batch_invariance_veto

        lift_gdn_batch_invariance_veto()

    # Engine-side context parallelism (vLLM PCP/DCP) is refused for the native architectures while vLLM
    # builds its engine configuration; trainer CP with an engine without CP is the supported configuration.
    from isoexec.runtimes.vllm.validation import install_context_parallel_refusal

    install_context_parallel_refusal((NATIVE_ARCH, NATIVE_DENSE_ARCH))

    # CPR prefix caching: the engine core names the checkpoint index and the scheduler admits only hits
    # every worker holds a checkpoint for (inert unless the engine runs prefix caching under the CPR composition).
    from isoexec.runtimes.vllm.cpr_apc import install_scheduler_admission

    install_scheduler_admission()

    # Publish readiness only after every installer succeeds; failed attempts retry.
    _PLUGIN_INSTALLED = True
    print("[ISOEXEC-PLUGIN] native model registered (CUSTOM varlen backend, sampler pins)", flush=True)


__all__ = [
    "NATIVE_ARCH",
    "ENGINE_SETTING_DEFAULTS",
    "NATIVE_DENSE_ARCH",
    "compilation_config_values",
    "engine_environment",
    "model_engine_settings",
    "register_native",
    "requested_cudagraph_mode",
    "register_native_model_class",
    "register_plugin",
]
