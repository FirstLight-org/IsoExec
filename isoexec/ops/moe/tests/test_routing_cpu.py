"""L0 (CPU): softmax_renorm routing -- fp32 end to end, the pinned tie-break (score desc, index asc),
the softmax-over-selected == renormalized-softmax identity, scores_full only on request, and
batch-shuffle invariance (a row's routing depends on that row only)."""

import pytest
import torch

from isoexec.ops.moe.routing import route, softmax_renorm, softmax_rows, stable_topk
from isoexec.schema import Mlp

SPEC = Mlp(intermediate_size=128, num_experts=8, top_k=2)


@pytest.mark.parametrize("tokens", [0, 1, 4])
@pytest.mark.parametrize("top_k", [1, 2, 8])
def test_topk_canonical_strides_preserve_legacy_values_indices_and_gradients(tokens: int, top_k: int):
    scores = torch.tensor([[1.0, 3.0, 3.0, 0.5, 3.0, 2.0, 2.0, 0.0]]).repeat(tokens, 1).requires_grad_()
    order = torch.sort(scores, dim=-1, descending=True, stable=True)
    expected_values = order.values[..., :top_k].contiguous()
    expected_indices = order.indices[..., :top_k].to(torch.int32).contiguous()
    expected_gradient, = torch.autograd.grad(expected_values.square().sum(), scores)

    values, indices = stable_topk(scores, top_k)
    gradient, = torch.autograd.grad(values.square().sum(), scores)
    assert values.stride() == indices.stride() == (top_k, 1)
    assert torch.equal(values.view(torch.uint8), expected_values.view(torch.uint8))
    assert torch.equal(indices, expected_indices)
    assert torch.equal(gradient.view(torch.uint8), expected_gradient.view(torch.uint8))


def test_fp32_only():
    for dt in (torch.bfloat16, torch.float16):
        try:
            softmax_renorm(torch.zeros(3, 8, dtype=dt), SPEC)
        except TypeError:
            pass
        else:
            raise AssertionError(f"{dt} scores must refuse")


def test_stable_tie_break_index_asc():
    s = torch.tensor([[1.0, 3.0, 3.0, 0.5, 3.0, 2.0, 2.0, 0.0]], dtype=torch.float32)
    vals, idx = stable_topk(s, 4)
    assert idx.tolist() == [[1, 2, 4, 5]] and vals.tolist() == [[3.0, 3.0, 3.0, 2.0]]
    assert idx.dtype == torch.int32
    r = softmax_renorm(s, SPEC)
    assert r.indices.tolist() == [[1, 2]] and torch.allclose(r.probs, torch.tensor([[0.5, 0.5]]))


def test_probs_match_naive_renormalized_softmax():
    torch.manual_seed(0)
    logits = torch.randn(64, 8) * 3
    r = softmax_renorm(logits, SPEC)
    full = torch.softmax(logits, -1)
    naive_sel = full.gather(-1, r.indices.long())
    naive = naive_sel / naive_sel.sum(-1, keepdim=True)
    assert torch.allclose(r.probs, naive, atol=2e-6, rtol=1e-5)
    assert torch.equal(r.indices, torch.topk(logits, 2, dim=-1).indices.to(torch.int32))  # no ties here
    assert torch.allclose(r.probs.sum(-1), torch.ones(64))
    assert r.scores_full is not None
    assert r.probs.dtype == torch.float32 and r.scores_full.shape == (64, 8)
    assert torch.allclose(r.scores_full, full, atol=1e-6)


def test_scores_full_only_when_asked():
    logits = torch.randn(5, 8)
    assert softmax_renorm(logits, SPEC, want_scores_full=False).scores_full is None
    assert softmax_renorm(logits, SPEC, want_scores_full=True).scores_full is not None


def test_batch_shuffle_invariance():
    torch.manual_seed(1)
    logits = torch.randn(40, 8)
    perm = torch.randperm(40)
    a = softmax_renorm(logits, SPEC)
    b = softmax_renorm(logits[perm], SPEC)
    assert torch.equal(a.indices[perm], b.indices) and torch.equal(a.probs[perm], b.probs)
    assert torch.equal(softmax_rows(logits)[perm], softmax_rows(logits[perm]))


def test_route_composition_and_an_unregistered_routing_refuses():
    import dataclasses

    logits = torch.randn(7, 8)
    r = route(logits, SPEC)
    assert torch.equal(r.indices, softmax_renorm(logits, SPEC).indices)
    try:
        route(logits, dataclasses.replace(SPEC, routing="sigmoid_noaux"))
    except ValueError as exc:
        assert "Mlp.routing" in str(exc)
    else:
        raise AssertionError("an unimplemented routing refuses")
