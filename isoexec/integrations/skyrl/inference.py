"""Full-vocabulary logprobs on SkyRL's ``/skyrl/v1/generate`` endpoint and engine arguments."""

from __future__ import annotations

from typing import Any

import numpy as np

from isoexec.integrations.full_distribution import full_comparison_enabled, full_logprob_rows

__all__ = ["engine_defaults", "full_comparison_enabled", "full_logprobs_from_output", "prepare_full_logprobs_capture"]


def engine_defaults(engine_kwargs: dict[str, Any]) -> None:
    """Request complete raw rows without silently overriding an explicit conflict."""
    for key, value in {"max_logprobs": -1, "logprobs_mode": "raw_logprobs"}.items():
        if key in engine_kwargs and engine_kwargs[key] != value:
            raise ValueError(f"engine_init_kwargs.{key} must be {value!r} for full logprobs")
        engine_kwargs[key] = value


def prepare_full_logprobs_capture(body: dict[str, Any], sampling_params: dict[str, Any],
                                  model_config: Any, lora_request: Any) -> None:
    """Check that this request and engine can return full-vocabulary rows, and make the engine produce them.

    SkyRL calls this before vLLM allocates generation resources, for a request with
    ``return_full_logprobs``. ``sampling_params`` is edited in place; a request that cannot return the
    rows raises ``ValueError``.
    """
    if body.get("return_sample_support", False):
        raise ValueError("full logprobs and sample-support capture cannot be requested together")
    if body.get("features"):
        raise ValueError("full logprobs require text-only generation")
    if lora_request is not None:
        raise ValueError("IsoExec does not support LoRA")
    if sampling_params.get("n", 1) != 1:
        raise ValueError("full logprobs require n=1")
    if model_config.logprobs_mode != "raw_logprobs":
        raise ValueError("full logprobs require the engine's raw_logprobs mode")
    sampling_params["logprobs"] = -1
    if not sampling_params.get("stop"):
        sampling_params["detokenize"] = False


def full_logprobs_from_output(completion: Any, vocab_size: int) -> np.ndarray:
    """A completion's full-vocabulary logprobs as a float32 ``[num_tokens, vocab_size]`` array.

    ``completion`` is the vLLM ``CompletionOutput`` SkyRL's endpoint reads from ``final_res.outputs[0]``; the
    request has a single completion because ``prepare_full_logprobs_capture`` refuses ``n != 1``.
    """
    return full_logprob_rows(list(completion.token_ids), completion.logprobs, vocab_size)
