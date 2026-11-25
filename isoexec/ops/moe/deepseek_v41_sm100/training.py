"""Backward of ``deepseek_v41_bf16_actor_v1``: the exact derivative of the forward program with
straight-through bf16 roundings, no atomics.

dY_r = dO_t (the combine is unweighted); dh = dY @ W_down (bf16 x bf16 -> fp32, ``rows.dgrad``: placeholder on mm_sm100/general);
``act_backward_weighted`` (dh * w, torch.clamp masks, dw = fixed per-row tree of dh * silu(g') * u');
wgrad (``rows.wgrad``: placeholder, mm_sm100/general's K-segment tree over the token axis) (operands: dY with the forward's h,
(dg | du) with the dispatched bf16 rows); FC1 dgrad; router_sm100 sqrt-softplus backward through the
selected uncorrected scores (no bias gradient); token gradient one fp32 chain over (routed rows in
ascending expert id, shared row, router dx), one bf16 rounding.
"""
from __future__ import annotations

import torch

from . import rows as cs
from . import rows as cs_training

from . import ops
from .packing import save_operands, saved_operands
from .spec import ROUTING


class _Train(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, router_w, bias, gate_up, down, top_k, routed_scale, limit):
        ops.check(x, gate_up, down)
        ctx.empty = x.size(0) == 0
        if x.size(0) == 0:
            save_operands(ctx, x, router_w, gate_up, down)
            return x.new_empty(x.shape)
        with torch.no_grad():
            st = ops.stages(x, router_w, bias, gate_up, down, top_k=top_k, routed_scale=routed_scale, limit=limit,
                            return_logits=True)
        ctx.rt = st.rt
        # a trainer's stacked expert copies are saved as their parameters and re-stacked in backward
        save_operands(ctx, x.contiguous(), router_w, bias, gate_up, down, st.x_rows, st.gp, st.proj)
        ctx.cfg = (top_k, routed_scale, limit)
        return st.y

    @staticmethod
    def backward(ctx, dy):
        if ctx.empty:
            x, router_w, gu, dn = saved_operands(ctx)
            return (torch.zeros_like(x), torch.zeros_like(router_w), None,
                    torch.zeros_like(gu), torch.zeros_like(dn), None, None, None)
        from isoexec.ops.moe.moe_sm100 import ops as base
        from isoexec.ops.moe.router_sm100 import ops as router
        x, router_w, bias, gu, dn, x_rows, gp, proj = saved_operands(ctx)
        top_k, routed_scale, limit = ctx.cfg
        rt = ctx.rt
        plan = rt.plan
        offsets = plan.grouped.offsets
        G = gu.size(0)
        T, H = x.shape
        R = plan.grouped.rows
        cs_training.bind_device(dy)
        with torch.no_grad():
            dy = dy.to(torch.bfloat16).contiguous()
            ones = torch.ones_like(rt.probs_ext)
            dY, _ = base.dispatch(dy, ones, plan)                       # dO of each route row (bytes)
            dh = cs_training.dgrad(dY, dn, offsets)
            dgu, hb, dw = cs.act_backward_weighted(dh, proj, gp, limit)
            d_dn = cs_training.wgrad_into(dY, hb, offsets, G, dn.dtype)
            d_gu = cs_training.wgrad_into(dgu, x_rows, offsets, G, gu.dtype)
            rows = torch.empty((R + T, H), dtype=torch.float32, device=x.device)
            cs_training.dgrad(dgu, gu, offsets, out=rows[:R])
            dp = dw.index_select(0, plan.restore[:, :top_k].reshape(-1).long()).view(T, top_k)
            dxr, dwr, _, _ = router.route_backward(dp.contiguous(), x, router_w, rt.logits, routing=ROUTING,
                                                   top_k=top_k, expert_bias=bias, routed_scale=routed_scale,
                                                   num_shared=1)
            rows[R:].copy_(dxr)                                          # exact bf16 -> fp32 widening
            tail = torch.arange(R, R + T, dtype=torch.int32, device=x.device)[:, None]
            dx = cs_training.fold_rows(rows, torch.cat((plan.combine_rows, tail), 1).contiguous())
        return dx, dwr.to(router_w.dtype), None, d_gu, d_dn, None, None, None


def moe_train(x, router_w, bias, gate_up, down, *, top_k, routed_scale, limit):
    """Training form of ``ops.moe_forward`` (same forward bits) with the native backward."""
    return _Train.apply(x, router_w, bias, gate_up, down, int(top_k), float(routed_scale), float(limit))


__all__ = ["moe_train"]
