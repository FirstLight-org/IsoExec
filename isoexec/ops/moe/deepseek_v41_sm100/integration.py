"""``isoexec.nn.DeepSeekExperts(backend="sm100")`` binding of ``deepseek_v41_bf16_actor_v1``.

The mixture keeps its checkpoint holders (gate weight/bias, per-expert w1/w2/w3 ``CheckpointLinear``
with their raw FP4/FP8 bytes untouched) and their decoded actor masters. The masters are packed
expert-major (w1 rows then w3 rows; w2) by re-pointing each holder's ``master_weight`` at its slice
of one storage (``packing``: values unchanged, aliases thereafter).
BF16 masters run as they are; FP32 masters pass through one differentiable cast to BF16 (the actor's
compute dtype, as the megatron_bf16 path casts them). Rollout and training run the same forward.
"""
from __future__ import annotations

import torch

from .packing import Stack, packed_slices

from . import ops, training
from .spec import RECIPE


def packed_masters(mixture):
    experts = [*mixture.experts, mixture.shared_experts]
    lins = [lin for e in experts for lin in (e.w1, e.w3, e.w2)]
    for lin in lins:
        if lin.master_weight is None:
            raise RuntimeError(f"{RECIPE} runs the decoded actor masters: initialize/enable them first")
    w1 = experts[0].w1.master_weight
    G, I, H = len(experts), w1.shape[0], w1.shape[1]
    gu = [w for e in experts for w in (e.w1.master_weight, e.w3.master_weight)]
    dn = [e.w2.master_weight for e in experts]
    if any(p.dtype != w1.dtype for p in gu + dn) or dn[0].shape != (H, I):
        raise ValueError("expert masters must share one dtype and the [I, H] / [H, I] geometry")
    with torch.no_grad():
        packed_slices(gu, (G, 2 * I, H))
        packed_slices(dn, (G, H, I))
    gu_t, dn_t = Stack.apply((G, 2 * I, H), *gu), Stack.apply((G, H, I), *dn)
    return gu_t.to(torch.bfloat16), dn_t.to(torch.bfloat16)


def experts_forward(experts_module, mixture, x):
    """``DeepSeekExperts.forward(module, x, linear)`` for ``backend="sm100"``."""
    shape = x.shape
    rows = x.reshape(-1, shape[-1])
    if rows.dtype != torch.bfloat16:
        raise ValueError(f"{RECIPE} requires BF16 activations")
    router = experts_module.router
    weight, bias = mixture.gate.weight, mixture.gate.bias
    if weight.dtype != torch.bfloat16 or bias.dtype != torch.float32:
        raise ValueError(f"{RECIPE} routes with the BF16 held router weight and the FP32 selection bias")
    gu, dn = packed_masters(mixture)
    kw = dict(top_k=router.topk, routed_scale=router.routed_scale, limit=experts_module.clamp)
    if experts_module.clamp <= 0:
        raise ValueError(f"{RECIPE} implements the clamped SwiGLU (positive swiglu_limit)")
    grad = torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad or gu.requires_grad)
    if rows.size(0) == 0:
        return x.clone()
    if grad:
        y = training.moe_train(rows, weight, bias, gu, dn, **kw)
    else:
        y = ops.moe_forward(rows, weight, bias, gu, dn, **kw)
    return y.reshape(shape)


__all__ = ["experts_forward", "packed_masters"]
