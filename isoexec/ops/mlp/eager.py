"""Trainer SwiGLU: silu(gate) * (up + 0.0).
The zero offset maps negative zero to positive zero and must be retained."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def silu_mul_megatron(g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """mcore ``MLP.forward`` glu (local spec, no fusion): ``activation_func(x_glu) * (x_linear +
    glu_linear_offset)`` with offset 0.0 -- the ``+ 0.0`` is kept (it maps -0.0 to +0.0)."""
    return F.silu(g) * (u + 0.0)


__all__ = ["silu_mul_megatron"]
