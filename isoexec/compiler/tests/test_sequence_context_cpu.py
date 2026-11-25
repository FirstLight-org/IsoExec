"""Implicit authored sequence inputs preserve real component and pipeline semantics."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

import isoexec.nn as inn
from isoexec.compiler.capture.capture import capture
from isoexec.compiler.capture.graph import ExecutionGraph
from isoexec.compiler.tests.test_output_contract_cpu import _equal
from isoexec.compiler.text import text_inputs, text_roots
from isoexec.ops.table import OpTable
from isoexec.runtimes.parallel import trivial_placement
from isoexec.schema import SequenceContext, make_sequence_context


class _Model(nn.Module):
    def __init__(self, *, moe: bool = False) -> None:
        super().__init__()
        self.embed = inn.Embedding(vocab=32, hidden=16, device="cpu")
        self.layers = nn.ModuleList(
            [inn.MoE(hidden=16, num_experts=4, top_k=2, intermediate=8,
                     shared_intermediate=8, device="cpu") for _ in range(2)]
            if moe else [
                inn.Attention(hidden=16, heads=4, kv_heads=2, head_dim=4,
                              rope_theta=10000, device="cpu"),
                inn.GDN(hidden=16, key_heads=2, value_heads=4, key_head_dim=4,
                        value_head_dim=4, conv_kernel=4, device="cpu"),
            ]
        )
        self.head = inn.LMHead(vocab=32, hidden=16, device="cpu")


class _ExplicitModel(_Model):
    def forward(self, input_ids: torch.Tensor, seq: SequenceContext) -> torch.Tensor:
        hidden = self.embed(input_ids)
        for layer in self.layers:
            hidden = hidden + layer(hidden, seq)
        return self.head(hidden)


class _ImplicitModel(_Model):
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embed(input_ids)
        for layer in self.layers:
            hidden = hidden + layer(hidden)
        return self.head(hidden)


def _capture_pair(explicit: _Model, implicit: _Model) -> tuple[ExecutionGraph, ExecutionGraph]:
    generator = torch.Generator().manual_seed(61)
    with torch.no_grad():
        for parameter in explicit.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.2)
    implicit.load_state_dict(explicit.state_dict())
    examples = text_inputs(explicit, device="cpu", max_tokens=16)
    before = capture(explicit, examples.args, examples.kwargs,
                     dynamic_shapes=examples.dynamic_shapes, root_layouts=examples.root_layouts)
    ids, seq = examples.args
    after = capture(implicit, (ids,), seq=seq, dynamic_shapes=examples.dynamic_shapes,
                    root_layouts=examples.root_layouts)
    return before, after


def _place(*models: nn.Module) -> None:
    for model in models:
        inn.place(model, trivial_placement(side="trainer"), op_table=OpTable(mode="reference"))


def _roots(lengths: tuple[int, ...]) -> dict[str, Any]:
    # The same token values with different valid packing make metadata observable.
    ids = torch.tensor([1, 5, 7, 3, 9, 2], dtype=torch.int64)
    assert sum(lengths) == ids.numel()
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    cu_cpu = torch.tensor(offsets, dtype=torch.int32)
    seq = make_sequence_context(
        torch.cat([torch.arange(length) for length in lengths]), cu_cpu.clone(),
        max_q=max(lengths), max_kv=max(lengths), cu_seqlens_cpu=cu_cpu,
    )
    return text_roots(ids, seq)


_BATCHES = ((2, 4), (3, 1, 2))
_Output = tuple[torch.Tensor, dict[str, torch.Tensor]]
_Result = tuple[list[_Output], dict[str, torch.Tensor]]


def _run(model: nn.Module, forward: Callable[..., Any], *, aux_only: bool = False) -> _Result:
    model.zero_grad(set_to_none=True)
    outputs: list[_Output] = []
    losses: list[torch.Tensor] = []
    for lengths in _BATCHES:
        result = forward(**_roots(lengths))
        logits, aux = result if isinstance(result, tuple) else (result, {})
        assert torch.isfinite(logits).all()
        loss = logits.new_zeros((), dtype=torch.float32) if aux_only else logits.float().square().mean()
        for scores in aux.values():
            assert scores.shape == (sum(lengths), 4) and torch.isfinite(scores).all()
            loss = loss + (scores[:, 0].mean() if aux_only else 0.01 * scores[:, 0].float().square().mean())
        losses.append(loss)
        outputs.append((logits.detach().clone(), {key: value.detach().clone() for key, value in aux.items()}))
    # Both forwards precede backward: a latest-batch context must not affect replay.
    torch.stack(losses).sum().backward()
    gradients = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            assert aux_only, name
            continue
        assert torch.isfinite(parameter.grad).all(), name
        gradients[name] = parameter.grad.clone()
    return outputs, gradients


def _equal_result(actual: _Result, expected: _Result) -> None:
    for (actual_logits, actual_aux), (expected_logits, expected_aux) in zip(actual[0], expected[0], strict=True):
        _equal(actual_logits, expected_logits)
        assert actual_aux.keys() == expected_aux.keys()
        for name in expected_aux:
            _equal(actual_aux[name], expected_aux[name])
    assert actual[1].keys() == expected[1].keys()
    for name in expected[1]:
        _equal(actual[1][name], expected[1][name])


def test_implicit_sequence_recompute_uses_each_forward_batch() -> None:
    explicit, implicit = _ExplicitModel(), _ImplicitModel()
    _, after = _capture_pair(explicit, implicit)
    _place(explicit, implicit)

    def recompute(forward: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
        def wrapped(input_ids: torch.Tensor, seq: SequenceContext) -> torch.Tensor:
            def run(_anchor: torch.Tensor) -> torch.Tensor:
                # Recreate the per-forward RoPE cache while retaining this call's metadata.
                return forward(**text_roots(input_ids, seq))

            anchor = torch.empty((), requires_grad=True)
            return cast(torch.Tensor, checkpoint(run, anchor, use_reentrant=True))

        return wrapped

    # Preserve the checkpoint schedule in the independent explicit-input oracle.
    expected = _run(explicit, recompute(explicit))
    _equal_result(_run(implicit, recompute(after.program)), expected)
