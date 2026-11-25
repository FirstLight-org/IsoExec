"""Configuration checks shared by vLLM construction entry points."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

import torch

from isoexec.ops.attention.spec import KV_DTYPE

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def structured_outputs_refusal(vllm_config: VllmConfig) -> str | None:
    """Return a refusal when a structured-output backend is explicitly selected."""
    backend = vllm_config.structured_outputs_config.backend
    if backend != "auto":
        return (
            f"structured outputs are pinned to the {backend!r} grammar backend; the grammar bitmask is "
            "applied to `logits` IN PLACE in Model Runner V2 (v1/worker/gpu/model_runner.py -> "
            "StructuredOutputsWorker.apply_grammar_bitmask) BEFORE the sampler computes raw_logprobs, "
            "so the returned logprob would carry the mask. Lift path: return the mask with the rollout"
        )
    return None


def kv_cache_refusal(vllm_config: VllmConfig, *, model_kv_dtype: str | None = None) -> str | None:
    """Return a refusal unless the model uses BF16 and the KV cache uses BF16 or the cache encoding the
    model package declares (``engine_settings()['kv_cache_dtype']``) for the cache layers it owns."""
    cache_config = vllm_config.cache_config
    model_config = vllm_config.model_config
    kv_dtype = str(cache_config.cache_dtype)
    model_dtype = model_config.dtype
    accepted = ("auto", "bfloat16") if model_kv_dtype is None else (model_kv_dtype,)
    if kv_dtype not in accepted or model_dtype != torch.bfloat16:
        return (
            f"A36: attention.core kv_dtype is pinned {KV_DTYPE} (this refuses every fp8/nvfp4 KV cache); "
            f"cache_dtype={kv_dtype!r} model dtype={model_dtype}"
        )
    return None


#: Engine-side context parallelism (vLLM prefill/decode context parallelism) is refused for IsoExec models.
#: Trainer context parallelism (any degree) with an engine without context parallelism is the supported
#: configuration: the trainer's CP=k forward is bitwise equal to its CP=1 forward, hence to the engine.
ENGINE_CONTEXT_PARALLEL_REFUSAL = (
    "engine context parallelism is not supported for IsoExec models; use trainer CP with an engine without CP"
)


def context_parallel_refusal(prefill_context_parallel_size: Any = 1, decode_context_parallel_size: Any = 1) -> str | None:
    """Return a refusal when vLLM prefill or decode context parallelism is requested (``None``/unset means 1)."""
    pcp = int(prefill_context_parallel_size or 1)
    dcp = int(decode_context_parallel_size or 1)
    if pcp == 1 and dcp == 1:
        return None
    return (f"{ENGINE_CONTEXT_PARALLEL_REFUSAL} (got prefill_context_parallel_size={pcp}, "
            f"decode_context_parallel_size={dcp})")


def engine_kwargs_context_parallel_refusal(engine_kwargs: Mapping[str, Any] | None) -> str | None:
    """``context_parallel_refusal`` for vLLM engine keyword arguments."""
    kwargs = engine_kwargs or {}
    return context_parallel_refusal(kwargs.get("prefill_context_parallel_size"),
                                    kwargs.get("decode_context_parallel_size"))


def parallel_config_context_parallel_refusal(parallel_config: Any) -> str | None:
    """``context_parallel_refusal`` for a vLLM ``ParallelConfig``."""
    return context_parallel_refusal(getattr(parallel_config, "prefill_context_parallel_size", 1),
                                    getattr(parallel_config, "decode_context_parallel_size", 1))


def install_context_parallel_refusal(native_architectures: tuple[str, ...]) -> None:
    """Refuse prefill/decode context parallelism for the IsoExec native architectures while vLLM builds its
    engine configuration (``ModelConfig.verify_with_parallel_config``, run by ``VllmConfig`` on the driver,
    before any worker starts), whatever front end constructs the engine. Other models keep vLLM's checks.
    """
    from vllm.config.model import ModelConfig

    from isoexec.checks import Refusal

    verify = ModelConfig.verify_with_parallel_config
    if getattr(verify, "_isoexec_cp_refusal", False):
        return
    native = frozenset(native_architectures)

    def verify_with_parallel_config(self, parallel_config):
        if native & set(getattr(self, "architectures", None) or ()):
            refusal = parallel_config_context_parallel_refusal(parallel_config)
            if refusal is not None:
                raise Refusal(refusal)
        return verify(self, parallel_config)

    setattr(verify_with_parallel_config, "_isoexec_cp_refusal", True)
    ModelConfig.verify_with_parallel_config = verify_with_parallel_config
