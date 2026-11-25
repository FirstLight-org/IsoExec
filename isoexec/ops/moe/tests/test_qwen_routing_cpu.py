"""Qwen full-softmax routing and probability multiplication after the expert projection."""

import torch

from isoexec.ops.mlp.reference import ref_silu_mul
from isoexec.ops.mm.reference import ref_mm
from isoexec.ops.moe.carriers import RoutingResult
from isoexec.ops.moe.reference import ref_expert_mlp
from isoexec.ops.moe.routing import route
from isoexec.schema import Mlp


def test_softmax_topk_keeps_original_probability_mass_and_aux_gradient():
    logits = torch.tensor([[0.4, 0.3, 0.2, 0.1]]).log().requires_grad_()
    spec = Mlp(intermediate_size=8, num_experts=4, top_k=2, routing="softmax")
    result = route(logits, spec)
    torch.testing.assert_close(result.probs, torch.tensor([[0.4, 0.3]]))
    assert result.scores_full is not None
    assert result.probs.sum() < 1
    result.scores_full[:, 0].sum().backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0


def test_after_fc2_values_and_gradients_follow_the_explicit_weighting_order():
    gen = torch.Generator().manual_seed(193)
    x = torch.randn(3, 16, generator=gen).bfloat16().requires_grad_()
    w1 = torch.randn(1, 16, 16, generator=gen).bfloat16().requires_grad_()
    w2 = torch.randn(1, 16, 8, generator=gen).bfloat16().requires_grad_()
    probs = torch.tensor([[0.3], [0.6], [0.7]], requires_grad=True)
    routing = RoutingResult(torch.zeros(3, 1, dtype=torch.int32), probs, None)
    actual = ref_expert_mlp(x, routing, w1, w2, weight_after_fc2=True).rows[:, 0]
    gate, up = ref_mm(x, w1[0]).chunk(2, dim=-1)
    expected = (ref_mm(ref_silu_mul(gate, up), w2[0]).bfloat16() * probs.bfloat16()).float()
    assert torch.equal(actual, expected)
    before = ref_expert_mlp(x, routing, w1, w2).rows[:, 0]
    assert not torch.equal(actual, before)
    operands = (x, w1, w2, probs)
    actual_grads = torch.autograd.grad(actual.square().sum(), operands, retain_graph=True)
    expected_grads = torch.autograd.grad(expected.square().sum(), operands)
    for got, want in zip(actual_grads, expected_grads, strict=True):
        assert torch.equal(got, want)
