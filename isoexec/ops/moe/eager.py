"""Trainer routing, shared-expert and combination stages around the expert GEMM provider.
Router projection uses the installed autograd-aware MM override; placement-aware callers supply collective operations."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
import torch.nn.functional as F

from isoexec.core.linear import packed_view
from isoexec.ops.moe.carriers import ExpertOutputs, RoutingResult
from isoexec.schema import Mlp as MlpSpec
from isoexec.ops.collectives.trainer import _identity, row_parallel_trainer
from isoexec.ops.mm.trainer import mm_trainer


def route_score_fp32(x: torch.Tensor, router_w: torch.Tensor) -> torch.Tensor:
    """Compute router scores with the trainer's deterministic FP32 GEMM, preserving autograd."""
    return torch.mm(x.to(torch.float32).contiguous(), router_w.to(torch.float32).t())


def route_mechanics_megatron(logits: torch.Tensor, spec: MlpSpec, want_scores_full: bool = True) -> RoutingResult:
    """Select sorted top-k experts and normalize their scores using the trainer's softmax."""
    from isoexec.ops.mm.mm_batch_invariant import softmax_batch_invariant

    vals, idx = torch.topk(logits, k=spec.top_k, dim=1, sorted=True)
    probs = softmax_batch_invariant(vals, -1, torch.float32)
    full = softmax_batch_invariant(logits, -1, torch.float32) if want_scores_full else None
    return RoutingResult(indices=idx.to(torch.int32), probs=probs.contiguous(), scores_full=full)


def route_mechanics_softmax(logits: torch.Tensor, spec: MlpSpec, want_scores_full: bool = True) -> RoutingResult:
    """Keep the original full-softmax mass of the selected experts."""
    from isoexec.ops.mm.mm_batch_invariant import softmax_batch_invariant

    full = softmax_batch_invariant(logits, -1, torch.float32)
    probs, indices = torch.topk(full, k=spec.top_k, dim=-1, sorted=True)
    return RoutingResult(indices.to(torch.int32), probs.contiguous(), full if want_scores_full else None)


def shared_glu_epilogue(inter: torch.Tensor) -> torch.Tensor:
    """Apply the shared GLU epilogue, preserving the SiLU, zero-offset, and product rounding steps."""
    x_glu, x_linear = torch.chunk(inter, 2, dim=-1)
    return F.silu(x_glu) * (x_linear + 0.0)


def shared_expert_glu(
    x: torch.Tensor, gate: torch.Tensor, up: torch.Tensor, *, epilogue=shared_glu_epilogue, mm=mm_trainer
) -> torch.Tensor:
    """Compute the shared expert's fused gate/up projection and GLU activation. ``mm`` is the
    projection's GEMM, ``y = x @ w.T``; an arch package that owns it passes its own."""
    inter = mm(x, packed_view(gate, up))
    return epilogue(inter)


def shared_expert_gate_score(x: torch.Tensor, expert_gate: torch.Tensor, *, mm=mm_trainer) -> torch.Tensor:
    """``torch.sigmoid(F.linear(x, gate_weight))`` -> ``[T, 1]`` in ``x.dtype`` (bf16), the shared
    expert's already-rounded gate. Same sharing rule as ``shared_expert_glu``."""
    logits = mm(x, expert_gate)  # F.linear(x, gate_weight) -> aten::mm chain (N=1, Triton)
    return torch.sigmoid(logits)


def shared_expert_megatron(
    x,
    gate,
    up,
    down,
    expert_gate,
    *,
    row_parallel=row_parallel_trainer,
    glu_epilogue=shared_glu_epilogue,
    tp_copy=_identity,
    mm=mm_trainer,
    gate_mm=mm_trainer,
) -> torch.Tensor:
    """Compute the shared expert and its optional BF16 sigmoid gate with trainer operators.
    ``mm`` / ``gate_mm`` are the fc1 and gate GEMMs (``y = x @ w.T``)."""
    h = shared_expert_glu(tp_copy(x), gate, up, epilogue=glu_epilogue, mm=mm)
    y = row_parallel(h, down)
    if expert_gate is not None:
        y = y * shared_expert_gate_score(x, expert_gate, mm=gate_mm)
    return y


def combine_fixed_order(routed: ExpertOutputs, shared: torch.Tensor | None) -> torch.Tensor:
    """``moe_combine_backward.fused_combine`` (what the trainer's rebound ``unpermute`` runs:
    ascending-expert k-order fp32 sum, no rounding) then mcore ``MoELayer.postprocess``'s
    ``output + shared_expert_output`` (fp32 + bf16 -> fp32). The bf16 round is the residual seam's."""
    from isoexec.ops.moe import fused_combine

    if routed.layout != "permuted":
        raise ValueError(f"combine_fixed_order expects permuted expert rows, got layout {routed.layout!r}")
    T = routed.routing.indices.shape[0]
    from isoexec.ops.moe import build_combine_rows

    rows = cast(Callable[..., torch.Tensor], build_combine_rows)(routed.extra, T, rows=routed.combine_rows)
    out = cast(Callable[..., torch.Tensor], fused_combine)(routed.rows, routed.extra, T, rows=rows)
    if shared is not None:
        out = out + shared
    return out


__all__ = [
    "combine_fixed_order",
    "route_mechanics_megatron",
    "route_score_fp32",
    "shared_expert_gate_score",
    "shared_expert_glu",
    "shared_expert_megatron",
    "shared_glu_epilogue",
]
