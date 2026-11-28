"""Native autograd for the engine's frozen chain and eight-leaf projection programs.

Backward uses BF16 operands and mm_sm100/general's deterministic reductions. The
weight gradient reduces over tokens, padded with absent zero pairs to 64 for the
native GEMM ABI; backward may depend on token count. No serving forward is changed.
"""

import torch
from torch.autograd.function import once_differentiable


def backward_mm(a, b):
    """Native BF16 backward GEMM, including token counts not divisible by 64."""
    from isoexec.ops.mm.mm_sm100.general import ops

    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise ValueError("engine projection backward requires BF16 operands")
    k = a.shape[1]
    kp = (k + 63) // 64 * 64
    if kp != k:
        # Preserve the original operand orientations: gradients retain their declared
        # MN-major segment reduction, rather than adopting the forward chain layout.
        ap = (a.new_zeros(kp, a.shape[0]).t() if a.stride(0) == 1
              else a.new_zeros(a.shape[0], kp))
        bp = (b.new_zeros(b.shape[1], kp).t() if b.stride(0) == 1
              else b.new_zeros(kp, b.shape[1]))
        ap[:, :k].copy_(a)
        bp[:k].copy_(b)
        a, b = ap, bp
    return ops.mm(a, b)


class ChainLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        from .engine import linear
        ctx.save_for_backward(x, weight)
        return linear(x, weight)

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        x, weight = ctx.saved_tensors
        dy = dy.to(torch.bfloat16).contiguous()
        dx = backward_mm(dy, weight) if ctx.needs_input_grad[0] else None
        dw = backward_mm(dy.t(), x) if ctx.needs_input_grad[1] else None
        return dx, dw


class LeafLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, leaves, root_fp32):
        from .engine import o_proj_leaves
        ctx.save_for_backward(x, weight)
        return o_proj_leaves(x, weight, leaves, root_fp32)

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        x, weight = ctx.saved_tensors
        dy = dy.to(torch.bfloat16).contiguous()
        dx = backward_mm(dy, weight) if ctx.needs_input_grad[0] else None
        dw = backward_mm(dy.t(), x) if ctx.needs_input_grad[1] else None
        return dx, dw, None, None
