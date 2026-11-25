"""Public entry point for the fused sm100 ``mlp.act`` kernel."""

from __future__ import annotations

import torch

from isoexec.ops.mlp.mlp_sm100._extension import extension


def silu_mul_fused(g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """One-kernel ``silu(g) * (u + 0.0)``, bit-identical to ``ops.mlp.eager.silu_mul_megatron``."""
    return _SiluMul.apply(g, u)


class _SiluMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, g, u):
        ctx.save_for_backward(g, u)
        return extension().silu_mul(g, u)

    @staticmethod
    def backward(ctx, dy):
        g, u = ctx.saved_tensors
        dg = torch.ops.aten.silu_backward(dy * (u + 0.0), g) if ctx.needs_input_grad[0] else None
        du = dy * torch.nn.functional.silu(g) if ctx.needs_input_grad[1] else None
        return dg, du


__all__ = ["silu_mul_fused"]
