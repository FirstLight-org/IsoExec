"""Model-distribution logprobs at the configured training temperature.

Sampling still uses vLLM's original logits and processor order. Only the
logprob producer sees the independently widened and tempered scoring rows.
"""

import weakref
import numpy as np
import torch
from isoexec.checks import Refusal

_OWNER = None
_TEMPERATURE = 1.0


def install(model: torch.nn.Module, temperature: float) -> None:
    global _OWNER, _TEMPERATURE
    if not np.isfinite(temperature) or temperature <= 0:
        raise Refusal("score_temperature: expected a finite positive temperature")
    device = next(model.parameters()).device
    model.register_buffer(
        "_isoexec_score_temperature",
        torch.tensor(temperature, dtype=torch.float32, device=device),
        persistent=False,
    )
    _OWNER = weakref.ref(model)
    _TEMPERATURE = temperature


def prepare_logits(logits: torch.Tensor) -> torch.Tensor:
    if _OWNER is None or _TEMPERATURE == 1.0:
        return logits
    model = _OWNER()
    if model is None:
        raise Refusal("score_temperature: registered model no longer exists")
    # A direct divide writes the independent scoring row once. Cloning and
    # dividing in-place would add a redundant full-vocabulary copy kernel.
    if logits.dtype == torch.float32:
        return logits / model._isoexec_score_temperature
    return logits.float().div_(model._isoexec_score_temperature)


def temperature_tensor() -> torch.Tensor | None:
    if _OWNER is None or _TEMPERATURE == 1.0:
        return None
    model = _OWNER()
    if model is None:
        raise Refusal("score_temperature: registered model no longer exists")
    return model._isoexec_score_temperature
