"""Fused forward with the existing eager activation's autograd backward."""

import torch


class _SiluMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, g, u):
        from .ops import silu_mul_fused
        ctx.save_for_backward(g, u)
        return silu_mul_fused(g, u)

    @staticmethod
    def backward(ctx, dy):
        from isoexec.ops.mlp.eager import silu_mul_megatron
        g, u = ctx.saved_tensors
        with torch.enable_grad():
            g, u = g.detach().requires_grad_(), u.detach().requires_grad_()
            return torch.autograd.grad(silu_mul_megatron(g, u), (g, u), dy)


def silu_mul(g, u):
    return _SiluMul.apply(g, u)
