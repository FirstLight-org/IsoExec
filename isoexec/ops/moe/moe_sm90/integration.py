"""Shared SM90 CUDA expert forward with compact grouped-GEMM training backward.
EP moves bytes and ETP folds fixed-width leaves; backward recomputes intermediates and has no forward bitwise contract."""

import os
from typing import Any, cast

import torch
from isoexec.ops.moe.autograd import expert_autograd

from isoexec.ops.moe.moe_sm90 import ops

_AFTER_FC2_GATHER = os.environ.get("ISOEXEC_MOE_AFTER_FC2_GATHER") == "1"
_COUNTS = {"forward": 0, "backward": 0, "after_fc2_forward": 0, "after_fc2_backward": 0,
           "after_fc2_gather_forward": 0}


def stats():
    return dict(_COUNTS)


_Experts = expert_autograd(ops, _COUNTS)

def _compact_root(h, down, plan):
    """FC2's single rounded BF16 root, stored into the owner combine's wire buffer when it grants.

    The root IS the BF16 value the combine would otherwise copy into symmetric staging, so ``out=``
    removes a copy of identical bytes; ``owner_wire_sink`` returns None (the allocating path) whenever
    it cannot prove that, and P/H come from the plan, never from an assumed routing width.
    """
    from isoexec.ops.moe.moe_sm90.moe_pik_combine_owner import owner_wire_sink

    return ops.fc2_compact_root(h, down, plan, out=owner_wire_sink(plan.rows, down.size(1), h.device))


def expert_mlp(x, routing, gate_up, down, *, n_leaves=8, compact_root=False, weight_after_fc2=False,
               defer_weight_after_fc2=False):
    """Unified expert seam, with explicit lossless storage for one local leaf.

    Default callers receive FP32 roots. The ETP owner may opt into compact BF16
    roots because it widens before the cross-rank tree. Training retains FP32
    output and the existing VJP even when that storage option is requested.
    """
    from isoexec.ops.moe.carriers import ExpertOutputs
    from isoexec.ops.moe.moe_sm90.moe_fused_permute import compact_route_positions_topk, route_to_permuted_rows

    if defer_weight_after_fc2 and not weight_after_fc2:
        raise ValueError("Deferred routing weights require weight_after_fc2=True")
    # The owner's compact wire must never contain an unweighted FC2 result.
    compact_root = compact_root and not weight_after_fc2
    if (not weight_after_fc2 or _AFTER_FC2_GATHER) and not torch.is_grad_enabled() and ops._can_gather_fc1(x, gate_up, routing.indices, routing.probs):
        if routing.indices.numel() <= 256 and gate_up.size(0) <= 256:
            plan, indices, probs, combine_rows = ops.small_route_plan(
                routing.indices.contiguous(), routing.probs.contiguous(), gate_up.size(0)
            )
            emit_rows = None
        else:
            indices, _, counts, probs = compact_route_positions_topk(routing.indices, routing.probs, gate_up.size(0))
            plan = ops.make_plan_from_counts(counts, indices.numel())
            combine_rows = torch.empty_like(routing.indices, dtype=torch.int32)
            emit_rows = combine_rows
        if weight_after_fc2:
            _COUNTS["after_fc2_gather_forward"] += 1
        h = ops._fc1_from_routes(x, gate_up, plan, indices, routing.indices, None if weight_after_fc2 else probs, emit_rows)
        if compact_root and n_leaves == 1 and x.dtype == torch.bfloat16:
            rows = _compact_root(h, down, plan)
        else:
            rows = ops.fc2(h, down, plan, leaves=n_leaves)
        if weight_after_fc2 and not defer_weight_after_fc2:
            rows = (rows.to(x.dtype) * probs[:, None].to(x.dtype)).float()
    else:
        permuted, probs, counts, indices, combine_rows = route_to_permuted_rows(x, routing, gate_up.shape[0])
        plan = ops.make_plan_from_counts(counts, permuted.shape[0])
        if torch.is_grad_enabled() and any(t.requires_grad for t in (permuted, probs, gate_up, down)):
            rows = _Experts.apply(permuted, probs, gate_up, down, plan, n_leaves,
                                  weight_after_fc2, defer_weight_after_fc2)
        else:
            with torch.no_grad():
                h = ops.fc1_glu(permuted, gate_up, plan, None if weight_after_fc2 else probs)
                if compact_root and n_leaves == 1 and h.size(1) == 64 and h.dtype == torch.bfloat16:
                    rows = _compact_root(h, down, plan)
                else:
                    rows = ops.fc2(h, down, plan, leaves=n_leaves)
                if weight_after_fc2 and not defer_weight_after_fc2:
                    rows = (rows.to(x.dtype) * probs[:, None].to(x.dtype)).float()
    _COUNTS["forward"] += 1
    if weight_after_fc2:
        _COUNTS["after_fc2_forward"] += 1
    return ExpertOutputs(rows, "permuted", routing, extra=indices, combine_rows=combine_rows)


# A consumer must opt in; this is never inferred from a tensor's dtype.
cast(Any, expert_mlp).supports_compact_root = True


#: The registered impl id (``ops/moe/moe_sm90/spec.py``) and the pins it declares.
IMPL = "cuda_wgmma_v1"
AFTER_FC2_IMPL = "cuda_wgmma_after_fc2_v1"

# Activation implemented by the fused fc1 epilogue, shared by binding and resolver records.
ACT = "cuda_silu_mul_v1"


# Forward-only engine recipes use their eager stage as fallback; the resolver selects them by side.

#: ``moe.route_mechanics`` on the engine. The ONE id that changes: ``ops/moe/moe_sm90/spec.py``
#: declares it in twin class ``moe.route.sorted_topk`` with ``deterministic_sorted_topk``, and
#: ``core/registry.TWIN_PROOFS`` names the battery that proves the pair.
ENGINE_MECHANICS_IMPL = "fused_o2"


def engine_route_mechanics(eager_fn):
    """The fused O2 router (top-k + softmax + dense scatter), declining to ``eager_fn``."""
    from isoexec.ops.moe.moe_sm90.moe_router_o2_kernel import engine_route_mechanics_o2

    return engine_route_mechanics_o2(eager_fn)


def engine_combine(eager_fn):
    """The owner combine: the same fixed tree and the same k-order fp32 fold, run for each token by
    the rank that owns it. Exposes ``owner_group``, which is what tells the mixture to defer both
    fc2 cross-rank trees to it. Impl id unchanged (``pik_leaf_tree``)."""
    from isoexec.ops.moe.moe_sm90.moe_pik_combine_owner import engine_combine_owner

    return engine_combine_owner(eager_fn)


def engine_shared_expert(eager_fn):
    """The shared expert that can hand its fc2 tree to :func:`engine_combine`, with the one-launch
    glu admitted per width. Impl id unchanged (``eager``)."""
    from isoexec.ops.moe.moe_sm90.moe_preamble_o12 import engine_shared_expert_owner

    return engine_shared_expert_owner(eager_fn)
