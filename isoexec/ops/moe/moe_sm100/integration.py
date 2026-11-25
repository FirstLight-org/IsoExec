"""The `moe.expert_mlp` seam on sm100, in the shape `ops/moe/resolve.py` asks packages for."""

from __future__ import annotations

import torch

from isoexec.ops.moe.carriers import ExpertOutputs
from isoexec.ops.moe.autograd import expert_autograd
from isoexec.ops.moe.moe_sm100 import ops
from isoexec.ops.moe.moe_sm100.spec import FC1_GLU_ACT

#: The registered impl id (`moe_sm100/spec.py`). The id IS the recipe.
IMPL = "sm100_tcgen05_v1"
AFTER_FC2_IMPL = "sm100_tcgen05_after_fc2_v1"

#: The shared expert's ids. v2 names every GEMM's provider (``ops/moe/sm100.py``,
#: ``ops.shared_expert``): v1 reached ``matmul_persistent`` / ``aten::mm``, which a trainer and an
#: engine install differently.
SHARED_IMPL = "sm100_after_fc2_shared_v2"
EAGER_SHARED_IMPL = "sm100_eager_v2"

#: The activation the fused FC1 epilogue implements, as the binding records it. SM90 spells its
#: own (`cuda_silu_mul_v1`); this is a different epilogue instantiation under a different id
#: because the accumulation history feeding it is tcgen05's, not wgmma's.
ACT = "sm100_cuda_silu_mul_v1"

_COUNTS = {"forward": 0, "backward": 0, "after_fc2_forward": 0, "after_fc2_backward": 0}
_Experts = expert_autograd(ops, _COUNTS)


class _Gather(torch.autograd.Function):
    """Expert-major token rows and route probabilities, with a fixed-order gather adjoint.

    ``index_select``'s own backward scatter-adds the ``k`` copies of each token with float atomics;
    this one sums them in route-slot order (``restore[t, j]`` is route ``(t, j)``'s row) instead.
    """

    @staticmethod
    def forward(ctx, x, probs, inverse, token_indices, restore):
        ctx.save_for_backward(restore)
        ctx.dtype = x.dtype
        return x.index_select(0, token_indices), probs.flatten().index_select(0, inverse)

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_x, grad_p = grad_outputs
        (restore,) = ctx.saved_tensors
        rows = restore.long()
        dx = dp = None
        if grad_x is not None and ctx.needs_input_grad[0]:
            dx = grad_x.index_select(0, rows[:, 0]).float()
            for slot in range(1, rows.size(1)):
                dx = dx + grad_x.index_select(0, rows[:, slot]).float()
            dx = dx.to(ctx.dtype)
        if grad_p is not None and ctx.needs_input_grad[1]:
            dp = grad_p.index_select(0, rows.flatten()).view(rows.shape)
        return dx, dp, None, None, None


def stats() -> dict:
    return dict(_COUNTS)


def expert_mlp(x, routing, gate_up, down, *, n_leaves: int = 8, compact_root: bool = False,
               weight_after_fc2: bool = False, defer_weight_after_fc2: bool = False):
    """Dispatch, FC1 with the fused GLU epilogue, FC2 with the segment tree -> `ExpertOutputs`."""
    if compact_root:
        raise ValueError(
            "moe_sm100 does not offer compact (operand-dtype) FC2 roots; the multi-leaf root is "
            "an fp32 tree value and narrowing it would round the transport twice"
        )
    if defer_weight_after_fc2 and not weight_after_fc2:
        raise ValueError("Deferred routing weights require weight_after_fc2=True")
    k = routing.indices.size(1)
    plan = ops.route(routing.indices.contiguous().to(torch.int32), gate_up.size(0))
    inverse = plan.inverse.long()
    token_indices = inverse.div(k, rounding_mode="floor")
    if torch.is_grad_enabled() and (x.requires_grad or routing.probs.requires_grad):
        grouped_x, grouped_p = _Gather.apply(x, routing.probs, inverse, token_indices, plan.restore)
    else:
        grouped_x = x.index_select(0, token_indices)
        grouped_p = routing.probs.flatten().index_select(0, inverse)
    if torch.is_grad_enabled() and any(t.requires_grad for t in (grouped_x, grouped_p, gate_up, down)):
        rows = _Experts.apply(grouped_x, grouped_p, gate_up, down, plan.grouped, n_leaves,
                              weight_after_fc2, defer_weight_after_fc2)
    else:
        h = ops.fc1_glu(grouped_x, gate_up, plan.grouped, None if weight_after_fc2 else grouped_p)
        rows = ops.fc2(h, down, plan.grouped, leaves=n_leaves)
        if weight_after_fc2 and not defer_weight_after_fc2:
            rows = (rows.to(x.dtype) * grouped_p[:, None].to(x.dtype)).float()
    _COUNTS["forward"] += 1
    if weight_after_fc2:
        _COUNTS["after_fc2_forward"] += 1
    return ExpertOutputs(
        rows,
        "permuted",
        routing,
        # ``extra`` follows SM90's convention exactly: int64, one TOKEN index per grouped row --
        # not this package's ``inverse``, which is a flat (token, slot) route index. A consumer
        # that reads ``extra`` must get the same thing from either arch, and
        # ``moe_combine_backward.build_combine_rows`` is one such consumer.
        extra=token_indices,
        combine_rows=plan.combine_rows,
    )


#: A consumer must opt in to anything; nothing here is inferred from a tensor's dtype.
expert_mlp.supports_compact_root = False

__all__ = ["ACT", "EAGER_SHARED_IMPL", "FC1_GLU_ACT", "IMPL", "SHARED_IMPL", "expert_mlp", "stats"]
