"""Trainer residual addition: cast the sublayer output to the residual dtype, then add with one bf16 round."""

from __future__ import annotations

import torch


def residual_add_megatron(h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """mcore ``_bias_dropout_add_func`` (unfused, no bias, p=0): ``x.to(residual.dtype)`` then
    ``residual + x``."""
    if y.dtype != h.dtype:
        y = y.to(h.dtype)
    return h + y


__all__ = ["residual_add_megatron"]
