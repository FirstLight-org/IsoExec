"""Plain-torch MoE references: fp32 router, family routing mechanics, bf16 expert GEMM boundaries and shared SwiGLU.
Routing probabilities apply before fc2; combination keeps its ordered fp32 sum and shared addition unrounded
until the residual stage, while GEMM and activation definitions come from their owning families."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isoexec.core.bindings import BoundOp
from isoexec.schema import Mlp
from isoexec.ops.mlp.reference import ref_silu_mul
from isoexec.ops.mm.reference import ref_mm
from isoexec.ops.moe.carriers import ExpertOutputs, RoutingResult
from isoexec.ops.moe.routing import ROUTING_FUNCTIONS

if TYPE_CHECKING:  # the plan is DUCK-TYPED at runtime -- ops/reference.py imports this module, never the reverse
    from isoexec.ops.reference import ReferencePlan


def ref_route_score(x: torch.Tensor, router_w: torch.Tensor) -> torch.Tensor:
    """The fp32 router GEMM: logits [T, E] (weight held fp32; bf16 activations upcast exactly)."""
    return x.to(torch.float32) @ router_w.to(torch.float32).t()


def ref_route_mechanics(scores: torch.Tensor, spec: Mlp, want_scores_full: bool = True, **kwargs) -> RoutingResult:
    return ROUTING_FUNCTIONS[spec.routing](scores, spec, want_scores_full=want_scores_full, **kwargs)


def ref_expert_mlp(x: torch.Tensor, routing: RoutingResult, gate_up: torch.Tensor, down: torch.Tensor,
                   *, weight_after_fc2: bool = False) -> ExpertOutputs:
    """Per (token, slot): fc2(act(fc1(x)) * p) with bf16 GEMM boundaries, fp32 out (probs PRE-fc2)."""
    T, k = routing.indices.shape
    E, twoI, H = gate_up.shape
    I = twoI // 2  # noqa: E741
    rows = torch.zeros(T, k, H, dtype=torch.float32, device=x.device)
    for e in range(E):
        tok, slot = (routing.indices == e).nonzero(as_tuple=True)
        if tok.numel() == 0:
            continue
        xs = x[tok]
        gu = ref_mm(xs, gate_up[e])
        h = ref_silu_mul(gu[:, :I], gu[:, I:])
        if weight_after_fc2:
            y = ref_mm(h, down[e]).to(x.dtype) * routing.probs[tok, slot, None].to(x.dtype)
        else:
            h = (h.to(torch.float32) * routing.probs[tok, slot][:, None]).to(x.dtype)
            y = ref_mm(h, down[e])
        rows[tok, slot] = y.to(torch.float32)
    return ExpertOutputs(rows=rows, layout="per_slot", routing=routing)


def ref_shared_expert(x, gate, up, down, expert_gate) -> torch.Tensor:
    h = ref_silu_mul(ref_mm(x, gate), ref_mm(x, up))
    y = ref_mm(h, down).to(torch.float32)
    if expert_gate is not None:
        g = torch.sigmoid(x.to(torch.float32) @ expert_gate.to(torch.float32).t())  # [T, 1]
        y = y * g
    return y


def ref_combine(routed: ExpertOutputs, shared: torch.Tensor | None) -> torch.Tensor:
    """fp32 k-order sum (slot order == descending-score order == the routing's fixed order), then
    the shared-expert add; NO rounding (the seam casts)."""
    if routed.layout == "combined":
        acc = routed.rows
    else:
        acc = routed.rows[:, 0, :].clone()
        for j in range(1, routed.rows.shape[1]):
            acc = acc + routed.rows[:, j, :]
    if shared is not None:
        acc = acc + shared.to(torch.float32)
    return acc


def bindings(plan: "ReferencePlan") -> dict[str, BoundOp]:
    """This family's contribution to ``ops/reference.py::reference_bindings``."""
    del plan
    return {
        "moe.route_score": BoundOp(
            "moe.route_score", "ref.router_gemm_fp32", None, "ops.reference", fn=ref_route_score
        ),
        "moe.route_mechanics": BoundOp(
            "moe.route_mechanics", "ref.stable_topk_softmax", None, "ops.reference", fn=ref_route_mechanics
        ),
        "moe.expert_mlp": BoundOp("moe.expert_mlp", "ref.experts", None, "ops.reference", fn=ref_expert_mlp),
        "moe.shared_expert": BoundOp("moe.shared_expert", "ref.shared", None, "ops.reference", fn=ref_shared_expert),
        "moe.combine": BoundOp("moe.combine", "ref.k_order_combine", None, "ops.reference", fn=ref_combine),
    }


__all__ = [
    "bindings",
    "ref_combine",
    "ref_expert_mlp",
    "ref_route_mechanics",
    "ref_route_score",
    "ref_shared_expert",
]
