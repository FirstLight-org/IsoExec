"""Check sigmoid/no-aux routing against an independent Python-sort oracle.
Bias selects experts, while unbiased sigmoid scores determine normalized and scaled weights."""

from __future__ import annotations

import pytest
import torch

from isoexec.ops.moe.routing import sigmoid_noaux_bias_select
from isoexec.schema import Mlp

#: Enough experts that top-k is a real selection, and a routed scale that is not 1.0 so a dropped
#: multiply is visible rather than invisible.
SPEC = Mlp(intermediate_size=128, num_experts=8, top_k=3, routing="sigmoid_noaux_bias_select", routed_scale=2.5)


def _oracle(logits: torch.Tensor, bias: torch.Tensor, spec: Mlp):
    """The rule, written naively: ``(indices [T, k] int, probs [T, k] fp32)``.

    Independent of ``stable_topk``/``gather`` on purpose -- see the module docstring.
    """
    indices, probs = [], []
    for row in logits.tolist():
        scores = [1.0 / (1.0 + torch.exp(torch.tensor(-x, dtype=torch.float64)).item()) for x in row]
        biased = [s + b for s, b in zip(scores, bias.tolist())]
        order = sorted(range(len(biased)), key=lambda i: (-biased[i], i))[: spec.top_k]
        chosen = [scores[i] for i in order]
        total = sum(chosen)
        indices.append(order)
        probs.append([c / total * spec.routed_scale for c in chosen])
    return indices, probs


@pytest.fixture
def case():
    """One seeded row set and one correction bias, shared by the cases that need both."""
    gen = torch.Generator().manual_seed(0x51C0)
    logits = (torch.randn(24, SPEC.num_experts, generator=gen) * 3).to(torch.float32)
    bias = (torch.randn(SPEC.num_experts, generator=gen) * 0.5).to(torch.float32)
    return logits, bias


def test_selection_and_weights_match_the_naive_oracle(case):
    """The whole rule at once: which experts, in which order, with which weights."""
    logits, bias = case
    got = sigmoid_noaux_bias_select(logits, SPEC, expert_bias=bias)
    want_idx, want_probs = _oracle(logits, bias, SPEC)

    assert got.indices.dtype == torch.int32
    assert got.probs.dtype == torch.float32
    assert got.indices.tolist() == want_idx
    # fp64 oracle against an fp32 kernel: the comparison is on the VALUES, and the tolerance is the
    # fp32 round the kernel is entitled to and nothing wider.
    assert torch.allclose(got.probs.double(), torch.tensor(want_probs, dtype=torch.float64), rtol=0, atol=1e-6)


def test_the_bias_selects_and_does_not_weight():
    """Two biases selecting the same experts must produce identical weights from the unbiased scores."""
    logits = torch.tensor([[3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0]], dtype=torch.float32)
    flat = torch.zeros(SPEC.num_experts, dtype=torch.float32)
    # Lifts the same three experts the flat bias already picks, by the same amount, so the ORDER and
    # the SET are untouched and only a weight read off the biased score could move.
    lifted = torch.tensor([0.4, 0.4, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)

    a = sigmoid_noaux_bias_select(logits, SPEC, expert_bias=flat)
    b = sigmoid_noaux_bias_select(logits, SPEC, expert_bias=lifted)
    assert a.indices.tolist() == b.indices.tolist() == [[0, 1, 2]]
    assert torch.equal(a.probs, b.probs), "the correction bias reached the weights"


def test_the_selected_weights_renormalize_and_carry_the_routed_scale(case):
    """``probs`` sums to ``routed_scale`` per row, and the scale is the spec's, not a constant 1."""
    logits, bias = case
    got = sigmoid_noaux_bias_select(logits, SPEC, expert_bias=bias)
    assert torch.allclose(got.probs.sum(-1), torch.full((logits.shape[0],), SPEC.routed_scale), rtol=0, atol=1e-6)

    unit = Mlp(intermediate_size=128, num_experts=SPEC.num_experts, top_k=SPEC.top_k,
               routing="sigmoid_noaux_bias_select", routed_scale=1.0)
    assert torch.allclose(sigmoid_noaux_bias_select(logits, unit, expert_bias=bias).probs * SPEC.routed_scale,
                          got.probs, rtol=0, atol=1e-6)


def test_ties_break_on_ascending_index(case):
    """The pinned tie rule, on the BIASED score, because that is what selection sorts by.

    Two experts whose biased scores are exactly equal must be taken in index order on every host, or
    two peers route the same token to two different experts and diverge with no numerical difference
    to point at.
    """
    del case
    logits = torch.tensor([[1.0, 1.0, 1.0, 1.0, -5.0, -5.0, -5.0, -5.0]], dtype=torch.float32)
    bias = torch.zeros(SPEC.num_experts, dtype=torch.float32)
    got = sigmoid_noaux_bias_select(logits, SPEC, expert_bias=bias)
    assert got.indices.tolist() == [[0, 1, 2]]


def test_a_row_depends_on_that_row_only(case):
    """Shuffle invariance: batching may not change any row's routing."""
    logits, bias = case
    perm = torch.randperm(logits.shape[0], generator=torch.Generator().manual_seed(11))
    straight = sigmoid_noaux_bias_select(logits, SPEC, expert_bias=bias)
    shuffled = sigmoid_noaux_bias_select(logits[perm], SPEC, expert_bias=bias)
    assert torch.equal(shuffled.indices, straight.indices[perm])
    assert torch.equal(shuffled.probs, straight.probs[perm])


def test_the_declared_inputs_are_enforced(case):
    """bf16 logits, a bias of the wrong width and the wrong routing each refuse rather than run."""
    logits, bias = case
    with pytest.raises(ValueError):
        sigmoid_noaux_bias_select(logits.to(torch.bfloat16), SPEC, expert_bias=bias)
    with pytest.raises(ValueError):
        sigmoid_noaux_bias_select(logits, SPEC, expert_bias=bias[:-1])
    with pytest.raises(ValueError):
        sigmoid_noaux_bias_select(logits, SPEC, expert_bias=bias.to(torch.float64))
