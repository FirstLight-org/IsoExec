"""Deterministic expert parallelism of the DeepSeek-V4.1 MoE (``isoexec/ops/moe/ep_fixed_order.py``):

* ``ep_moe_train`` / ``ep_moe_forward`` -- ``deepseek_v41_bf16_actor_v1`` (BF16 masters through
  moe_sm100's grouped GEMMs, sqrt-softplus routing, route weight before the down projection);
* ``mega_ep_forward`` -- the same program's rollout on canonical BF16 MegaMoE over a real EP group (``mega.py``;
  no host sync).

Outputs, dx and owned experts' gradients are bitwise the single-rank recipe on the concatenated batch.
Replicated router and shared-at-home gradients are summed in fixed EP-rank order here; callers
must not reduce them over the same EP group again.
Local operands are this rank's experts in ``ExpertMap.local_experts(rank)`` order.
"""
from __future__ import annotations

import torch

from isoexec.ops.moe import ep_fixed_order as epf
from . import rows as cs
from . import rows as cs_training

from . import ops
from .packing import save_operands, saved_operands
from .spec import FC2_LEAVES, ROUTING


def _route(x, router_w, bias, top_k, routed_scale, E, return_logits=False):
    return cs.route_rows(x, router_w, bias, top_k=top_k, routed_scale=routed_scale, E=E, routing=ROUTING,
                         return_logits=return_logits)


def _fold_f32(buf, rows):
    """moe_sm100's combine program (acc = +0; acc = RN32(acc + row j), j ascending in canonical
    order; one bf16 rounding) over the bf16-valued roots read through ``rows [T, S]``: the bf16 rows
    widened exactly to fp32, then moe_sm100's combine (``rows.fold_rows``)."""
    return cs.fold_rows(buf.float(), rows.contiguous())


def _owner_forward(rows, lids, probs, gu, dn, limit):
    from isoexec.ops.moe.moe_sm100 import ops as base

    n, H = rows.size(0), dn.size(1)
    if n == 0:
        return torch.empty((0, H), dtype=torch.bfloat16, device=rows.device), None
    plan = epf.owner_plan(lids, gu.size(0))
    x_rows, gp = base.dispatch(rows, probs.view(n, 1).contiguous(), plan)
    proj = base.fc1(x_rows, gu, plan.grouped)
    h = ops.swiglu_rows(proj, gp, limit)
    roots = base.fc2(h, dn, plan.grouped, leaves=FC2_LEAVES)
    # fc2's one-segment root is rounded to bf16 already (stored fp32): ship it as bf16, exactly
    return epf.to_receive_order(roots, plan).to(torch.bfloat16), (plan, x_rows, gp, proj)


def _exchange_forward(x, router_w, bias, gu, dn, *, top_k, routed_scale, limit, emap, comm, return_logits):
    owner_tab, local_tab = emap.tables(comm.rank, x.device)
    ids_ext, probs_ext, logits = _route(x, router_w, bias, top_k, routed_scale, emap.E, return_logits)
    rt = epf._plan_routes(ids_ext, owner_tab, comm)
    rows, lids, probs = epf.dispatch_all(rt, comm, x, local_tab.index_select(0, ids_ext.reshape(-1).long()),
                                         probs_ext.reshape(-1))
    roots, state = _owner_forward(rows, lids, probs, gu, dn, limit)
    buf = epf.ret(rt, comm, roots)                    # send-buffer order, read through rt.home_rows()
    return rt, state, buf, logits


def ep_moe_forward(x, router_w, bias, gate_up_local, down_local, *, top_k, routed_scale, limit, emap, comm):
    """Rollout of the BF16 actor at EP = ``comm.size`` (the training forward's bits)."""
    T, H = x.shape
    with torch.no_grad():
        rt, _, buf, _ = _exchange_forward(x.contiguous(), router_w, bias, gate_up_local, down_local, top_k=top_k,
                                         routed_scale=routed_scale, limit=limit, emap=emap, comm=comm,
                                         return_logits=False)
    return _fold_f32(buf, rt.home_rows()) if T else x.new_empty((0, H))


class _EPTrain(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, router_w, bias, gu, dn, cfg):
        top_k, routed_scale, limit, emap, comm = cfg
        T, H = x.shape
        with torch.no_grad():
            x = x.contiguous()
            rt, state, buf, logits = _exchange_forward(x, router_w, bias, gu, dn, top_k=top_k, routed_scale=routed_scale,
                                                       limit=limit, emap=emap, comm=comm, return_logits=True)
            y = _fold_f32(buf, rt.home_rows()) if T else x.new_empty((0, H))
        ctx.cfg, ctx.rt, ctx.state, ctx.logits = cfg, rt, state, logits
        # a trainer's stacked expert copies are saved as their parameters and re-stacked in backward
        save_operands(ctx, x, router_w, bias, gu, dn)
        return y

    @staticmethod
    def backward(ctx, dy):
        from isoexec.ops.moe.router_sm100 import ops as router

        x, router_w, bias, gu, dn = saved_operands(ctx)
        top_k, routed_scale, limit, emap, comm = ctx.cfg
        rt, state = ctx.rt, ctx.state
        cs_training.bind_device(dy)
        T, H = x.shape
        S = top_k + 1
        dev = x.device
        with torch.no_grad():
            r_dY = epf.dispatch_rows(rt, comm, dy.to(torch.bfloat16).contiguous())   # dO of every route (bytes)
            n, G = r_dY.size(0), gu.size(0)
            l_dx = torch.empty((n, H), dtype=torch.float32, device=dev)
            l_dp = torch.empty(n, dtype=torch.float32, device=dev)
            d_gu, d_dn = torch.zeros_like(gu), torch.zeros_like(dn)
            if n:
                plan, x_rows, gp, proj = state
                off = plan.grouped.offsets
                dYg = epf.to_grouped(r_dY, plan)
                dgu, hb, dw = cs.act_backward_weighted(cs_training.dgrad(dYg, dn, off), proj, gp, limit)
                d_dn = cs_training.wgrad_into(dYg, hb, off, G, dn.dtype)
                d_gu = cs_training.wgrad_into(dgu, x_rows, off, G, gu.dtype)
                l_dx = epf.to_receive_order(cs_training.dgrad(dgu, gu, off), plan)
                l_dp = epf.to_receive_order(dw, plan)
            dX = epf.ret(rt, comm, l_dx)
            dp_rows = epf.ret(rt, comm, l_dp)
            if T:
                hr = rt.home_rows()
                dp = dp_rows.index_select(0, hr[:, :top_k].reshape(-1).long()).view(T, top_k)
                dxr, dwr, _, _ = router.route_backward(dp, x, router_w, ctx.logits, routing=ROUTING, top_k=top_k,
                                                       expert_bias=bias, routed_scale=routed_scale, num_shared=1)
                rows = torch.cat((dX, dxr.float()), 0)
                tail = torch.arange(T * S, T * S + T, dtype=torch.int32, device=dev)[:, None]
                dx = cs_training.fold_rows(rows, torch.cat((hr, tail), 1).contiguous())
            else:
                dx, dwr = torch.empty_like(x), torch.zeros_like(router_w, dtype=torch.float32)
            dwr, d_gu, d_dn = epf.reduce_replicated_grads(comm, emap, dwr, d_gu, d_dn)
            dwr = dwr.to(router_w.dtype)
        return dx, dwr, None, d_gu, d_dn, None


def ep_moe_train(x, router_w, bias, gate_up_local, down_local, *, top_k, routed_scale, limit, emap, comm):
    """Training form of the BF16 actor at EP = ``comm.size`` (bitwise ``training.moe_train`` on the
    concatenated batch: output, dx, every local expert's gradient)."""
    if getattr(comm, 'static_routes', False):
        raise NotImplementedError('fixed-capacity EP transport is inference-only')
    ops.check(x, gate_up_local, down_local)
    cfg = (int(top_k), float(routed_scale), float(limit), emap, comm)
    return _EPTrain.apply(x, router_w, bias, gate_up_local, down_local, cfg)


# ------------------------------------------------------------------------------- MegaMoE EP ---
def mega_ep_forward(x, router_w, bias, gate_up_local, down_local, *, top_k, routed_scale, limit, emap, ws, width,
                    shared=None, out_rows=None, skip=None):
    """Rollout of the BF16 actor at EP = ``ws.world`` on canonical BF16 MegaMoE: in-kernel NVLink dispatch, FC1,
    weighted SwiGLU, FC2, fused shared expert and home fold (``mega.py``). Collective: every EP rank calls it with
    the same ``width`` (>= its own rows). No host sync. Bitwise ``ep_moe_forward`` / the single-rank recipe.
    ``shared=(gate_up, down)``: routed-only local masters plus the shared expert's (no concatenation copy).
    ``out_rows=M``: ``x`` is this rank's canonical home slice of M replicated rows; returns the replicated [M, H]
    (the fused NVLink gather instead of the caller's all-gather). ``skip [n]`` bool: rows routed to no expert
    (``mega.skip_rows``: a padded graph step's padding rows)."""
    from . import mega
    if shared is None:
        ops.check(x, gate_up_local, down_local)
    mega.check_expert_map(emap, ws.world, ws.rank)
    mega.check_no_grad_pass(x, router_w, gate_up_local, down_local, *(shared or ()))
    with torch.no_grad():
        x = x.contiguous()
        n = x.size(0)
        ids, weights = (mega.route_select(x, router_w, bias, top_k=top_k, routed_scale=routed_scale) if n
                        else (None, None))
        mega.skip_rows(ids, weights, skip, top_k)
        return mega.experts(ws, x, ids, weights, gate_up_local, down_local, limit=limit, width=width, shared=shared,
                            out_rows=out_rows)


__all__ = ["ep_moe_forward", "ep_moe_train", "mega_ep_forward"]
