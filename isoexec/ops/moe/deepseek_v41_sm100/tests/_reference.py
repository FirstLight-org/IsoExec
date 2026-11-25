"""Torch statements of the activation and the comparison metric the batteries use."""
from __future__ import annotations

import torch


def swiglu_program(g: torch.Tensor, u: torch.Tensor, limit: float, weight: torch.Tensor | None = None):
    """fp32 ``(g, u)`` -> bf16 h: the clamped SwiGLU, the route weight applied before the down projection.
    Evaluated with torch CUDA float32 ops (IEEE division, libdevice ``expf``; one op per kernel, nothing
    contracted) it is bitwise the native kernel's graph."""
    L = torch.tensor(limit, dtype=torch.float32, device=g.device)
    gc = torch.where(g > L, L, g)
    uc = torch.where(u > L, L, torch.where(u < -L, -L, u))
    s = gc / (1.0 + torch.exp(-gc))
    h = s * uc
    if weight is not None:
        h = h * weight.float()[:, None]
    return h.to(torch.bfloat16)


def rel(a, b):
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm().clamp_min(1e-300)).item()
