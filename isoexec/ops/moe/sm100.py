"""SM100 bindings retain the native recipes' identities and component carrier types."""

import torch
from typing import Any, cast

from isoexec.ops._resolve import refuse, resolution


def forward_only(fn):
    def call(*args, **kw):
        import torch
        if torch.is_grad_enabled() and any(isinstance(t, torch.Tensor) and t.requires_grad for t in (*args, *kw.values())):
            raise ValueError("SM100 BF16 MoE is forward-only; no grouped-GEMM backward is provided")
        return fn(*args, **kw)
    return call


class _SortedTopK(torch.autograd.Function):
    """The native SM100 sorted top-k forward with the adjoint of a softmax over the selected logits."""

    @staticmethod
    def forward(ctx, logits, top_k):
        from isoexec.ops.moe.moe_sm100.ops import route_mechanics

        indices, probs = route_mechanics(logits, top_k)
        ctx.save_for_backward(indices, probs)
        ctx.shape = logits.shape
        ctx.mark_non_differentiable(indices)
        return indices, probs

    @staticmethod
    def backward(ctx, *grad_outputs):
        _, grad = grad_outputs
        indices, probs = ctx.saved_tensors
        if grad is None:
            return None, None
        selected = probs * (grad - (grad * probs).sum(-1, keepdim=True))
        # Top-k indices are distinct within a row, so the scatter writes each element once.
        return probs.new_zeros(ctx.shape).scatter_(1, indices.long(), selected), None


def mechanics(routing):
    def call(logits, spec, want_scores_full=True, *, expert_bias=None):
        import torch
        from isoexec.ops.moe.carriers import RoutingResult
        from isoexec.ops.moe.moe_sm100.ops import route_mechanics
        trainable = torch.is_grad_enabled() and any(isinstance(t, torch.Tensor) and t.requires_grad
                                                    for t in (logits, expert_bias))
        if routing == "sigmoid_noaux_bias_select":
            if trainable:
                raise ValueError("SM100 sigmoid routing is forward-only")
            if want_scores_full:
                raise ValueError("SM100 sigmoid routing does not expose full auxiliary scores")
            if expert_bias is None:
                raise ValueError("sigmoid routing requires expert_bias")
            idx, probs = route_mechanics(logits, spec.top_k, expert_bias=expert_bias,
                                         routed_scale=spec.routed_scale)
            return RoutingResult(idx, probs, None)
        idx, probs = _SortedTopK.apply(logits, spec.top_k) if trainable else route_mechanics(logits, spec.top_k)
        # The auxiliary full-row softmax is the trainer's own (the engine never requests it).
        full = None
        if want_scores_full:
            from isoexec.ops.mm.mm_batch_invariant import softmax_batch_invariant

            full = softmax_batch_invariant(logits, -1, torch.float32)
        return RoutingResult(idx, probs, full)
    return call


def combine(routed, shared):
    import torch

    if torch.is_grad_enabled() and routed.rows.requires_grad:
        raise ValueError("SM100 combine is forward-only")
    from isoexec.ops.moe.moe_sm100.ops import extension
    if routed.layout != "permuted" or routed.combine_rows is None:
        raise ValueError("SM100 combine requires permuted expert rows and canonical combine_rows")
    out = torch.empty((routed.combine_rows.shape[0], routed.rows.shape[1]), dtype=torch.float32,
                      device=routed.rows.device)
    cast(Any, extension()).combine(routed.rows, routed.combine_rows, out)
    return out if shared is None else out + shared


class _Combine(torch.autograd.Function):
    """The native SM100 fixed-order fold with its exact row-gather adjoint."""

    @staticmethod
    def forward(ctx, rows, combine_rows, token_indices):
        from isoexec.ops.moe.moe_sm100.ops import extension

        ctx.save_for_backward(token_indices)
        ctx.dtype = rows.dtype
        output = rows.new_empty((combine_rows.shape[0], rows.shape[1]), dtype=torch.float32)
        cast(Any, extension()).combine(rows, combine_rows, output)
        return output

    @staticmethod
    def backward(ctx, *grad_outputs):
        (gradient,) = grad_outputs
        (token_indices,) = ctx.saved_tensors
        return gradient.index_select(0, token_indices).to(ctx.dtype), None, None


def combine_trainable(routed, shared):
    """Use the selected SM100 provider, including when autograd is enabled."""
    if routed.layout != "permuted" or routed.combine_rows is None:
        raise ValueError("SM100 combine requires permuted expert rows and canonical combine_rows")
    if not isinstance(routed.extra, torch.Tensor) or routed.extra.shape != (routed.rows.shape[0],):
        raise ValueError("SM100 combine requires one token index per expert row")
    output = _Combine.apply(routed.rows, routed.combine_rows, routed.extra.long())
    return output if shared is None else output + shared


def route_score_fp32_vendored(x, router_w):
    """fp32 router logits on the vendored batch-invariant Triton GEMM, named directly.

    ``torch.mm`` would reach whichever ``aten::mm`` registration the host installed last, and the
    trainer's ``matmul_persistent`` carries wrappers an engine's does not; this is one kernel on
    every host. Autograd reaches the router weight through the fp32 casts.
    """
    from isoexec.ops.mm.trainer import mm_vendored

    return mm_vendored(x.to(torch.float32), router_w.to(torch.float32))


def shared_expert_sm100(x, gate, up, down, expert_gate, *, row_parallel=None, glu_epilogue=None, tp_copy=None):
    """The shared expert with every GEMM on a named provider: gate | up on this arch's ``mm``
    (``mm_sm100``, whose K segmentation reads K alone, so the packed gate | up and any TP column shard
    of it are the separate projections' program), the N = 1 gate on the vendored Triton GEMM, which
    ``mm_sm100`` declines, and the fc2 on the placement's fixed-tree ``row_parallel`` (pik)."""
    from isoexec.ops.collectives.trainer import _identity, row_parallel_trainer
    from isoexec.ops.mm.sm100 import mm_linear
    from isoexec.ops.mm.trainer import mm_vendored
    from isoexec.ops.moe.eager import shared_expert_megatron, shared_glu_epilogue

    return shared_expert_megatron(
        x, gate, up, down, expert_gate,
        row_parallel=row_parallel_trainer if row_parallel is None else row_parallel,
        glu_epilogue=shared_glu_epilogue if glu_epilogue is None else glu_epilogue,
        tp_copy=_identity if tp_copy is None else tp_copy,
        mm=mm_linear, gate_mm=mm_vendored,
    )


def resolve_sm100(op, *, site, routing, side, weight_after_fc2=False):
    from isoexec.ops.moe.moe_sm100 import integration, ops
    provider = "ops.moe.moe_sm100"
    if op == "moe.expert_mlp":
        impl = integration.AFTER_FC2_IMPL if weight_after_fc2 else integration.IMPL
        return resolution(op, impl, integration.expert_mlp, provider=provider, site=site,
                          inputs={"act": integration.ACT})
    if op == "moe.route_score":
        # The router weight is fp32 (``nn/moe.ROUTER_DTYPE``); the native chain reads operand-dtype
        # weights, so every routing scores with the vendored fp32 Triton GEMM, called by name.
        return resolution(op, "sm100_router_gemm_fp32_v1", route_score_fp32_vendored, provider=provider,
                          site=site)
    if op == "moe.route_mechanics" and routing == "softmax":
        from isoexec.ops.moe.eager import route_mechanics_softmax
        return resolution(op, "sm100_softmax_full_v1", route_mechanics_softmax, provider=provider, site=site)
    if op == "moe.combine" and weight_after_fc2:
        fn = combine_trainable
        if side == "engine":
            from isoexec.ops.moe.moe_sm90.moe_pik_combine_owner import engine_combine_owner
            fn = engine_combine_owner(fn)
        return resolution(op, "sm100_after_fc2_combine_v1", fn, provider=provider, site=site)
    if op == "moe.shared_expert" and weight_after_fc2:
        fn = shared_expert_sm100
        if side == "engine":
            from isoexec.ops.mm.sm100 import mm_linear
            from isoexec.ops.mm.trainer import mm_vendored
            from isoexec.ops.moe.moe_sm90.moe_preamble_o12 import engine_shared_expert_owner
            fn = engine_shared_expert_owner(fn, mm=mm_linear, gate_mm=mm_vendored)
        return resolution(op, integration.SHARED_IMPL, fn, provider=provider, site=site)
    if op == "moe.route_mechanics":
        impl = {"softmax_renorm": "sm100_fused_topk_v1",
                "sigmoid_noaux_bias_select": "sm100_sigmoid_noaux_bias_select_v1"}.get(routing)
        if impl is not None:
            return resolution(op, impl, mechanics(routing), provider=provider, site=site)
    if op == "moe.combine":
        return resolution(op, "sm100_gather_korder_v1", combine, provider=provider, site=site)
    if op == "moe.shared_expert":
        # This native recipe is local-only; callers needing TP/SP must bind a qualified seam.
        return resolution(op, integration.EAGER_SHARED_IMPL, forward_only(ops.shared_expert), provider=provider,
                          site=site)
    raise refuse(op, site=site, asked={"routing": routing, "side": side}, on_offer=())
