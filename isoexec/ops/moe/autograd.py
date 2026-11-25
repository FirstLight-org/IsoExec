"""Grouped expert derivatives shared by SM90 and SM100 numerical providers."""

from typing import Any, cast

import torch
from torch.autograd.function import once_differentiable


def expert_autograd(ops: Any, counts: dict[str, int]) -> type[torch.autograd.Function]:
    """Retain each provider's actual rounded operands and grouped-GEMM derivatives."""
    class _Experts(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, probs, gate_up, down, plan, leaves, weight_after_fc2, defer_weight_after_fc2):
            h = ops.fc1_glu(x, gate_up, plan, None if weight_after_fc2 else probs)
            out = ops.fc2(h, down, plan, leaves=leaves)
            weighted = weight_after_fc2 and not defer_weight_after_fc2
            if weighted:
                out = out.to(x.dtype)
            ctx.save_for_backward(x, probs, gate_up, down, h, *((out,) if weighted else ()))
            ctx.plan = plan
            from isoexec.ops.mm.wgrad import expert_sink

            ctx.sinks = (expert_sink(gate_up), expert_sink(down))
            ctx.weight_after_fc2 = weight_after_fc2
            ctx.weighted = weighted
            if weighted:
                return (out * probs[:, None].to(x.dtype)).float()
            return out

        @staticmethod
        @once_differentiable
        def backward(ctx, *grad_outputs):  # pyright: ignore[reportIncompatibleMethodOverride]
            (dout,) = grad_outputs
            x, probs, gate_up, down, h = ctx.saved_tensors[:5]
            plan = ctx.plan
            counts["backward"] += 1
            # Recompute the actual rounded operands, including CUDA's SiLU provider.
            inter = ops.fc1(x, gate_up, plan)
            offsets = plan.offsets[1:].contiguous()
            mm = torch.ops.aten._grouped_mm.default
            upstream = dout.to(x.dtype)
            dy = ((upstream * probs[:, None].to(x.dtype)) if ctx.weighted else upstream).contiguous()
            from isoexec.ops.mm.wgrad import expert_weight_grads

            # Gradient-accumulation fusion: each expert's block straight into its parameters' main_grad.
            dw2 = expert_weight_grads(dy, h, offsets, down, ctx.sinks[1])
            if dw2 is None:
                dw2 = mm(dy.t(), h, offsets)
            dh = mm(dy, down, offsets)
            if ctx.weight_after_fc2:
                counts["after_fc2_backward"] += 1
                dp = (upstream * ctx.saved_tensors[5]).sum(-1).float() if ctx.weighted else None
            else:
                unweighted = torch.empty_like(h)
                cast(Any, ops.extension()).activation(inter, torch.ones_like(probs), unweighted)
                dp = (dh.float() * unweighted.float()).sum(-1)
            # Respect the casts in the pointwise chain when propagating its VJP.
            product_grad = dh if ctx.weight_after_fc2 else (dh.float() * probs[:, None]).to(x.dtype)
            g, u = inter.chunk(2, dim=-1)
            with torch.enable_grad():
                gr = g.detach().requires_grad_(True)
                ur = u.detach().requires_grad_(True)
                product = torch.nn.functional.silu(gr) * ur
                dg, du = torch.autograd.grad(product, (gr, ur), product_grad)
            di = torch.cat((dg, du), dim=-1)
            dx = mm(di, gate_up, offsets)
            dw1 = expert_weight_grads(di, x, offsets, gate_up, ctx.sinks[0])
            if dw1 is None:
                dw1 = mm(di.t(), x, offsets)
            return dx, dp, dw1, dw2, None, None, None, None
    return _Experts
