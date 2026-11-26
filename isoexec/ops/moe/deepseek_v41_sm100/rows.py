"""BF16 row operations and grouped backward GEMMs of the V4.1 experts, on base kernels and torch.

PLACEHOLDER (BF16 kernel items, ``BF16-SCOPE``): the clamped-SwiGLU rows, their weighted backward and the
grouped dgrad / wgrad run as torch elementwise ops and per-expert ``mm_sm100/general`` GEMMs. They are
deterministic run to run (fixed launch configurations, no floating-point atomics), correct, and slow (a
host sync for the expert row offsets and one GEMM launch per expert). Each states its program:

* ``swiglu_rows``  ``h = RNE_bf16((silu(min(g, L)) * clamp(u, -L, L)) * w)``, every operation an fp32 IEEE
  operation in this order, ``silu(x) = x / (1 + expf(-x))`` (ATen's formula). The same scalar program as
  the canonical BF16 MegaMoE's activation (``_vendor/deep_gemm`` ``isoexec_canonical``).
* ``act_backward_weighted``  the exact derivative through the clamp masks (the gradient passes at equality,
  zero for clamped values); ``dw`` (the row weight's gradient) is ``sum_i dh_i * s_i * u'_i`` reduced by
  ``torch.sum`` over the row (a fixed order for a fixed shape; not a declared tree).
* ``dgrad``  per expert ``dX_e = dY_e @ W_e`` (bf16 x bf16 -> fp32) on ``mm_sm100/general``: one ascending K
  chain per output at the V4.1 shapes (K = hidden 5120 or 2 x expert width 4608, below the 16384 at which
  ``general``'s published min_segment = 8192 first gives two segments under the schedule policy's rule 3).
* ``wgrad``  per expert ``dW_e = dY_e^T @ X_e``: the token axis zero-padded to a multiple of 64 (exact zero
  products), reduced by ``mm_sm100/general``'s K-segment tree (``sm100_tcgen05_ksegtree_v1``: the schedule
  policy's rule 3 on the kernel's published min_segment = 8192 and max_segments 16 -- segments of at least
  8192 tokens, balanced adjacent-pair fold, FP32 root).
* ``fold_rows``  moe_sm100's combine program (one fp32 chain per token over its rows in canonical slot
  order, one bf16 rounding).

Routing (``route``, ``route_rows``) is router_sm100's fused selection plus moe_sm100's plan.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

#: The routing the V4.1 experts are served with (router_sm100 ``sqrtsoftplus_bias_select``).
ROUTING_SQRTSOFTPLUS = "sqrtsoftplus_bias_select"
ROUTINGS = (ROUTING_SQRTSOFTPLUS,)


@dataclass
class Routing:
    ids: torch.Tensor           # [T, k] int32 routed ids, ascending expert id
    weights: torch.Tensor       # [T, k] fp32 normalized * routed_scaling_factor
    plan: object                # moe_sm100 RoutePlan over [T, k + 1] (shared slot last)
    probs_ext: torch.Tensor     # [T, k + 1] fp32 (shared weight 1.0)
    logits: torch.Tensor | None = None


def route(x, router_w, bias, *, top_k, routed_scale, num_experts, return_logits=False, shared=True,
          routing=ROUTING_SQRTSOFTPLUS) -> Routing:
    """router_sm100's fused selection (the shared expert emitted as slot ``k``, id ``E``, weight 1.0) and
    moe_sm100's exact-rank counting-sort plan."""
    from isoexec.ops.moe.moe_sm100 import ops as base
    from isoexec.ops.moe.router_sm100 import ops as router

    if routing not in ROUTINGS:
        raise ValueError(f"routing must be one of {ROUTINGS}")
    with torch.no_grad():   # selection is discrete; training differentiates through the router backward
        out = router.route_fused(x, router_w, routing=routing, top_k=top_k, expert_bias=bias,
                                 routed_scale=routed_scale, num_shared=1 if shared else 0,
                                 return_logits=return_logits)
    ids_ext, probs_ext = out[0].contiguous(), out[1].contiguous()
    plan = base.route(ids_ext, num_experts + 1)
    return Routing(ids_ext[:, :top_k], probs_ext[:, :top_k], plan, probs_ext, out[2] if return_logits else None)


def route_rows(x, router_w, bias, *, top_k, routed_scale, E, routing=ROUTING_SQRTSOFTPLUS, return_logits=False):
    """``(ids [T, k+1] int32 with the shared id E last, probs [T, k+1] fp32, logits | None)``."""
    T = x.size(0)
    if T == 0:
        ids_ext = torch.empty((0, top_k + 1), dtype=torch.int32, device=x.device)
        return ids_ext, torch.empty((0, top_k + 1), dtype=torch.float32, device=x.device), None
    rt = route(x, router_w, bias, top_k=top_k, routed_scale=routed_scale, num_experts=E,
               return_logits=return_logits, routing=routing)
    sid = torch.full((T, 1), E, dtype=torch.int32, device=x.device)
    ids_ext = torch.cat((rt.ids[:, :top_k].to(torch.int32), sid), 1)
    return ids_ext, rt.probs_ext.contiguous(), rt.logits


def _gate_up(raw):
    I = raw.size(1) // 2
    return raw[:, :I].float(), raw[:, I:].float()


def swiglu_rows(proj, w, limit):
    """``[R, 2I]`` bf16 (gate | up) and fp32 row weights ``[R]`` (or None) -> ``h [R, I]`` bf16."""
    g, u = _gate_up(proj)
    L = float(limit)
    gc, uc = torch.clamp_max(g, L), torch.clamp(u, -L, L)
    v = torch.mul(torch.div(gc, torch.add(torch.exp(torch.neg(gc)), 1.0)), uc)
    if w is not None:
        v = torch.mul(v, w.view(-1, 1))
    return v.to(torch.bfloat16)


def act_backward_weighted(dh: torch.Tensor, raw: torch.Tensor, w: torch.Tensor, limit: float):
    """``(dgu [R, 2I] bf16, h_w [R, I] bf16, dw [R] fp32)``: the weighted activation's backward."""
    g, u = _gate_up(raw)
    L = float(limit)
    gc, uc = torch.clamp_max(g, L), torch.clamp(u, -L, L)
    den = torch.add(torch.exp(torch.neg(gc)), 1.0)
    sig = torch.div(1.0, den)
    sl = torch.div(gc, den)
    pr = torch.mul(sl, uc)
    wr = w.view(-1, 1)
    dw = torch.mul(dh, pr).sum(1)
    dp = torch.mul(dh, wr)
    ds = torch.mul(sig, torch.add(torch.mul(gc, torch.sub(1.0, sig)), 1.0))
    dg = torch.where(g <= L, torch.mul(torch.mul(dp, uc), ds), torch.zeros_like(g))
    du = torch.where((u >= -L) & (u <= L), torch.mul(dp, sl), torch.zeros_like(u))
    dgu = torch.cat((dg, du), 1).to(torch.bfloat16)
    return dgu, torch.mul(pr, wr).to(torch.bfloat16), dw


def bind_device(t):
    """Make t's device current in this thread (the autograd engine runs backward on its own device thread)."""
    torch.cuda.set_device(t.device)


def _bounds(offsets):
    return [int(v) for v in offsets.tolist()]


def _general():
    from isoexec.ops.mm.mm_sm100.general import ops as gen
    return gen


def dgrad(rows_bf16, w, offsets, out=None):
    """``out[o_e:o_{e+1}] = rows[o_e:o_{e+1}] @ w[e]`` (``w [G, K, N]``), fp32, one K chain per output."""
    if out is None:
        out = torch.empty((rows_bf16.size(0), w.size(2)), dtype=torch.float32, device=rows_bf16.device)
    b = _bounds(offsets)
    gen = _general()
    for e in range(len(b) - 1):
        lo, hi = b[e], b[e + 1]
        if hi > lo:
            out[lo:hi] = gen.bmm(rows_bf16[lo:hi].unsqueeze(0), w[e].unsqueeze(0), out_dtype=torch.float32)[0]
    return out


def wgrad(dy_bf16, x_bf16, offsets, groups):
    """``dW[e] = dy[o_e:o_{e+1}]^T @ x[o_e:o_{e+1}]`` fp32, the token axis padded to 64 with zero rows."""
    out = torch.zeros((groups, dy_bf16.size(1), x_bf16.size(1)), dtype=torch.float32, device=dy_bf16.device)
    b = _bounds(offsets)
    gen = _general()
    for e in range(min(groups, len(b) - 1)):
        lo, hi = b[e], b[e + 1]
        n = hi - lo
        if n == 0:
            continue
        pad = (-n) % 64
        dy, x = dy_bf16[lo:hi], x_bf16[lo:hi]
        if pad:
            dy = torch.cat((dy, dy.new_zeros((pad, dy.size(1)))), 0)
            x = torch.cat((x, x.new_zeros((pad, x.size(1)))), 0)
        out[e] = gen.bmm(dy.t().unsqueeze(0), x.contiguous().unsqueeze(0), out_dtype=torch.float32)[0]
    return out


def wgrad_into(dy_bf16, x_bf16, offsets, groups, dtype):
    """``wgrad(...).to(dtype)``."""
    return wgrad(dy_bf16, x_bf16, offsets, groups).to(dtype)


def fold_rows(rows_f32, combine_rows):
    """One fp32 chain per token over ``combine_rows[t]`` in column order, one bf16 rounding (moe_sm100)."""
    from isoexec.ops.moe.moe_sm100._extension import extension

    out = torch.empty((combine_rows.size(0), rows_f32.size(1)), dtype=torch.bfloat16, device=rows_f32.device)
    if out.numel():
        extension().combine(rows_f32, combine_rows, out)
    return out


__all__ = ["ROUTINGS", "ROUTING_SQRTSOFTPLUS", "Routing", "act_backward_weighted", "bind_device", "dgrad",
           "fold_rows", "route", "route_rows", "swiglu_rows", "wgrad", "wgrad_into"]
