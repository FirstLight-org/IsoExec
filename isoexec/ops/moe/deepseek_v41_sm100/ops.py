"""Forward of ``deepseek_v41_bf16_actor_v1``: every stage a native kernel.

* route     router_sm100 ``sqrtsoftplus_bias_select`` (``rows.route``), shared expert
            = slot k, id E, weight 1.0; moe_sm100's exact-rank plan
* dispatch  moe_sm100 ``dispatch`` (bf16 rows + per-row route weight)
* FC1       moe_sm100 ``fc1``: bf16 x bf16 -> fp32 one K chain, rounded once to bf16 (gate | up)
* act       ``swiglu_rows``: h = RNE_bf16(w * (silu(min(g, L)) * clamp(u, -L, L))), fp32 (DeepSeek's rule,
            the route weight before the down projection; the shared rows' weight is exactly 1)
* FC2       moe_sm100 ``fc2`` with one K segment: fp32 chain, root rounded to bf16
* combine   moe_sm100 ``combine``: one fp32 chain over the token's rows in ascending expert id (shared
            last), one bf16 rounding
``moe_forward_mega`` evaluates this program on canonical BF16 MegaMoE (``mega.py``: one fused kernel, same bits;
no-grad passes); ``moe_forward`` / ``stages`` keep the stage-by-stage grouped evaluation (the training forward,
whose stage tensors the backward reads, and the bitwise reference).
Masters are the trainable BF16 tensors ``gate_up [E + 1, 2I, H]`` (w1 rows then w3 rows) and
``down [E + 1, H, I]`` (w2), the shared expert last.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from . import rows as cs

from .spec import FC2_LEAVES, RECIPE, ROUTING


def _base():
    from isoexec.ops.moe.moe_sm100 import ops as base
    return base


def swiglu_rows(proj, w, limit):
    return cs.swiglu_rows(proj, w, limit)


def route(x, router_w, bias, *, top_k, routed_scale, num_experts, return_logits=False):
    return cs.route(x, router_w, bias, top_k=top_k, routed_scale=routed_scale, num_experts=num_experts,
                     return_logits=return_logits, routing=ROUTING)


@dataclass
class Stages:
    rt: object
    x_rows: torch.Tensor     # [R, H] bf16 dispatched rows
    gp: torch.Tensor         # [R] fp32 route weight per row (shared rows 1.0)
    proj: torch.Tensor       # [R, 2I] bf16 FC1 projection
    h: torch.Tensor          # [R, I] bf16 activation (weighted)
    roots: torch.Tensor      # [R, H] fp32 (bf16-valued) FC2 roots
    y: torch.Tensor          # [T, H] bf16


def check(x, gate_up, down):
    if x.dtype != torch.bfloat16 or gate_up.dtype != torch.bfloat16 or down.dtype != torch.bfloat16:
        raise ValueError(f"{RECIPE} runs BF16 activations and BF16 masters")
    if gate_up.ndim != 3 or down.ndim != 3 or gate_up.size(0) != down.size(0) or gate_up.size(1) != 2 * down.size(2) \
            or gate_up.size(2) != down.size(1) or x.size(-1) != gate_up.size(2):
        raise ValueError("masters must be gate_up [E + 1, 2I, H] and down [E + 1, H, I]")


def stages(x, router_w, bias, gate_up, down, *, top_k, routed_scale, limit, return_logits=False) -> Stages:
    base = _base()
    check(x, gate_up, down)
    x = x.contiguous()
    rt = route(x, router_w, bias, top_k=top_k, routed_scale=routed_scale, num_experts=gate_up.size(0) - 1,
               return_logits=return_logits)
    x_rows, gp = base.dispatch(x, rt.probs_ext, rt.plan)
    proj = base.fc1(x_rows, gate_up, rt.plan.grouped)
    h = swiglu_rows(proj, gp, limit)
    roots = base.fc2(h, down, rt.plan.grouped, leaves=FC2_LEAVES)
    y = base.combine(roots, rt.plan, dtype=torch.bfloat16)
    return Stages(rt, x_rows, gp, proj, h, roots, y)


def moe_forward(x, router_w, bias, gate_up, down, *, top_k, routed_scale, limit):
    """``[T, H]`` bf16 -> ``[T, H]`` bf16 (rollout and the training forward are this program)."""
    if x.size(0) == 0:
        return torch.empty_like(x)
    with torch.no_grad():
        return stages(x, router_w, bias, gate_up, down, top_k=top_k, routed_scale=routed_scale, limit=limit).y


def mega_workspace(x, gate_up, down, top_k):
    """The single-rank BF16 MegaMoE workspace of this geometry (pooled; created outside capture, no collective)."""
    from . import mega
    return mega.Workspace.prepare(None, x.device, gate_up.size(0) - 1, int(top_k), x.size(1), down.size(2))


def moe_forward_mega(x, router_w, bias, gate_up, down, *, top_k, routed_scale, limit, skip=None):
    """``moe_forward`` on canonical BF16 MegaMoE (``mega.py``): route, then one fused kernel (FC1, weighted SwiGLU,
    FC2, fused shared expert, canonical combine). EP = 1, no host sync, graph-capturable once warm; no-grad only.
    ``skip [n]`` bool: rows routed to no expert (``mega.skip_rows``)."""
    from . import mega
    check(x, gate_up, down)
    mega.check_no_grad_pass(x, router_w, gate_up, down)
    if x.size(0) == 0:
        return torch.empty_like(x)
    with torch.no_grad():
        ws = mega_workspace(x, gate_up, down, top_k)
        if not ws.warmed and not torch.cuda.is_current_stream_capturing():
            mega.warm(ws, gate_up, down, limit)      # every tile geometry, so any later capture size is ready
        ids, weights = mega.route_select(x.contiguous(), router_w, bias, top_k=top_k, routed_scale=routed_scale)
        mega.skip_rows(ids, weights, skip, top_k)
        return mega.experts(ws, x, ids, weights, gate_up, down, limit=limit)


__all__ = ["Stages", "check", "mega_workspace", "moe_forward", "moe_forward_mega", "route", "stages", "swiglu_rows"]
