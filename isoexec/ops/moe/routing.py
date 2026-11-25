"""MoE routing on fp32 [T,E] logits with stable (score descending, expert index ascending) selection.
Softmax routing applies max/sub/exp/sum/div over selected logits; sigmoid/no-aux routing uses bias only for selection.
RoutingResult carries int32 [T,k] indices, fp32 probabilities and optional full scores."""

from __future__ import annotations

from typing import Callable

import torch

from isoexec.schema import Mlp
from isoexec.ops.moe.carriers import RoutingResult


def stable_topk(scores: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k over the last dim with the pinned tie-break (score desc, index asc). Returns
    (values fp32 [T,k], indices int32 [T,k]) in descending-score order."""
    if scores.dtype != torch.float32:
        raise TypeError(f"routing scores must be fp32, got {scores.dtype}")
    order = torch.sort(scores, dim=-1, descending=True, stable=True)
    vals = order.values[..., :k].clone(memory_format=torch.contiguous_format)
    idx = order.indices[..., :k].to(dtype=torch.int32, memory_format=torch.contiguous_format)
    return vals, idx


def softmax_rows(x: torch.Tensor) -> torch.Tensor:
    """fp32 softmax over the last dim as the batch-invariant decomposition (amax, exp, sum, div)."""
    if x.dtype != torch.float32:
        raise TypeError(f"softmax input must be fp32, got {x.dtype}")
    m = torch.amax(x, dim=-1, keepdim=True)
    e = torch.exp(x - m)
    return e / torch.sum(e, dim=-1, keepdim=True)


def softmax_renorm(logits: torch.Tensor, spec: Mlp, *, want_scores_full: bool = True) -> RoutingResult:
    """``softmax_renorm``: top-k on the fp32 logits, softmax over the selected."""
    if spec.routing != "softmax_renorm":
        raise ValueError(f"softmax_renorm called for routing {spec.routing!r}")
    vals, idx = stable_topk(logits, spec.top_k)
    probs = softmax_rows(vals)
    full = softmax_rows(logits) if want_scores_full else None
    return RoutingResult(indices=idx, probs=probs.contiguous(), scores_full=full)


def softmax(logits: torch.Tensor, spec: Mlp, *, want_scores_full: bool = True) -> RoutingResult:
    """Top-k full-softmax probabilities, without renormalizing the selected set."""
    full = softmax_rows(logits)
    probs, indices = stable_topk(full, spec.top_k)
    return RoutingResult(indices, probs, full if want_scores_full else None)


def sigmoid_noaux_bias_select(logits: torch.Tensor, spec: Mlp, want_scores_full: bool = True, *, expert_bias):
    """Select using sigmoid scores plus correction bias; weight using unbiased scores.

GLM's single expert group needs no group masking. The correction bias influences
only selection, so it never receives a gradient. Selected weights renormalize
before the checkpoint's routed scale is applied.
"""
    if spec.routing != "sigmoid_noaux_bias_select" or logits.dtype != torch.float32:
        raise ValueError("sigmoid/noaux routing requires its declared spec and fp32 logits")
    if expert_bias.shape != (spec.num_experts,) or expert_bias.dtype != torch.float32:
        raise ValueError("expert correction bias must be fp32 [num_experts]")
    scores = torch.sigmoid(logits)
    _, idx = stable_topk(scores + expert_bias, spec.top_k)
    selected = torch.gather(scores, -1, idx.long())
    probs = selected / (selected.sum(-1, keepdim=True) + 1e-20)
    return RoutingResult(idx, (probs * spec.routed_scale).contiguous(), scores if want_scores_full else None)


ROUTING_FUNCTIONS: dict[str, Callable[..., RoutingResult]] = {
    "softmax": softmax,
    "softmax_renorm": softmax_renorm,
    "sigmoid_noaux_bias_select": sigmoid_noaux_bias_select,
}


#: routing name -> the score fn applied to the fp32 router logits BEFORE the mechanics. Identity for
#: the softmax family: its softmax lives in the mechanics, over the selected set.
SCORE_FUNCTIONS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "softmax": lambda logits: logits,
    "softmax_renorm": lambda logits: logits,
    "sigmoid_noaux_bias_select": lambda logits: logits,
}


def route(logits_fp32: torch.Tensor, spec: Mlp, *, expert_bias: torch.Tensor | None = None) -> RoutingResult:
    """Reference composition score-fn -> mechanics for one row's routing (plain torch)."""
    score_fn, mech = SCORE_FUNCTIONS[spec.routing], ROUTING_FUNCTIONS[spec.routing]
    if logits_fp32.dtype != torch.float32:
        raise TypeError("router logits must be fp32")
    if spec.routing == "sigmoid_noaux_bias_select":
        if expert_bias is None:
            raise ValueError("sigmoid/noaux routing requires the checkpoint correction bias")
        return mech(score_fn(logits_fp32), spec, expert_bias=expert_bias)
    if expert_bias is not None:
        raise ValueError("softmax_renorm does not consume a correction bias")
    return mech(score_fn(logits_fp32), spec)


__all__ = [
    "ROUTING_FUNCTIONS",
    "SCORE_FUNCTIONS",
    "route",
    "softmax_renorm",
    "softmax_rows",
    "stable_topk",
]
