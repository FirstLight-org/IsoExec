"""Public entry point for the fused sm90 ``mlp.act`` kernel."""

from __future__ import annotations

import torch

from isoexec.ops.mlp.mlp_sm90._extension import extension


def _in_domain(g: torch.Tensor, u: torch.Tensor) -> bool:
    return (g.is_cuda and u.is_cuda and g.dtype == torch.bfloat16 and u.dtype == torch.bfloat16
            and g.shape == u.shape and g.device == u.device)


def silu_mul_fused(g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """One-kernel ``silu(g) * (u + 0.0)``, bit-identical to ``ops.mlp.eager.silu_mul_megatron``.

    Operands outside the kernel's domain run the eager statement itself (the same program).
    """
    if not _in_domain(g, u):
        from isoexec.ops.mlp.eager import silu_mul_megatron

        return silu_mul_megatron(g, u)
    if torch.is_grad_enabled() and (g.requires_grad or u.requires_grad):
        return _SiluMul.apply(g, u)
    return extension().silu_mul(g, u)


class _SiluMul(torch.autograd.Function):
    """Fused forward; backward is the eager program's VJP, op for op, in one kernel.

    ``silu_mul_bwd`` reproduces ``silu_backward(dy * (u + 0.0), g)`` and ``dy * silu(g)`` bit for bit
    (the ``+ 0.0`` and every bf16 round included); a ``dy`` outside its domain runs the eager VJP.
    """

    @staticmethod
    def forward(ctx, g, u):
        ctx.save_for_backward(g, u)
        return extension().silu_mul(g, u)

    @staticmethod
    def backward(ctx, dy):
        g, u = ctx.saved_tensors
        if dy.dtype == torch.bfloat16 and dy.is_cuda and dy.shape == g.shape:
            dg, du = extension().silu_mul_bwd(dy, g, u)
            return (dg if ctx.needs_input_grad[0] else None), (du if ctx.needs_input_grad[1] else None)
        dg = torch.ops.aten.silu_backward(dy * (u + 0.0), g) if ctx.needs_input_grad[0] else None
        du = dy * torch.nn.functional.silu(g) if ctx.needs_input_grad[1] else None
        return dg, du


__all__ = ["silu_mul_fused"]
