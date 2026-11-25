"""Divide logits by an fp32 tensor temperature, matching both engines.
A Python scalar can lower to reciprocal multiplication and change rounding."""

from __future__ import annotations

import torch


def temperature_divide(logits_fp32: torch.Tensor, temperature: float | torch.Tensor) -> torch.Tensor:
    """Divide by an FP32 tensor temperature; scalar reciprocal multiplication changes bits."""
    if temperature == 1.0:
        return logits_fp32
    return logits_fp32 / torch.as_tensor(temperature, dtype=torch.float32, device=logits_fp32.device)


__all__ = [
    "temperature_divide",
]
