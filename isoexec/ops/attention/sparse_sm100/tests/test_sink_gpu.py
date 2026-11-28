"""Learned-sink semantics and native accuracy."""
from __future__ import annotations

import pytest
import torch


def oracle(q, kv, lists, sink, scale):
    """Independent FP64 softmax with an explicit extra key and zero value."""
    keys = kv[lists.clamp(min=0).long()]
    scores = torch.einsum("thd,tkd->thk", q, keys) * scale
    scores = scores.masked_fill(lists[:, None] < 0, -torch.inf)
    scores = torch.cat((scores, sink[None, :, None].expand(q.shape[0], -1, 1)), -1)
    probabilities = scores.softmax(-1)[..., :-1]
    out = torch.einsum("thk,tkd->thd", probabilities, keys[..., :min(512, q.shape[-1])])
    return out, scores.logsumexp(-1)


def test_sink_gradient_formula_cpu():
    torch.manual_seed(13)
    q = torch.randn(3, 4, 8, dtype=torch.float64, requires_grad=True)
    kv = torch.randn(7, 8, dtype=torch.float64, requires_grad=True)
    sink = torch.randn(4, dtype=torch.float64, requires_grad=True)
    lists = torch.tensor([[0, 2, 4], [1, -1, -1], [-1, -1, -1]])
    out, lse = oracle(q, kv, lists, sink, 8**-0.5)
    do = torch.randn_like(out)
    dsink, = torch.autograd.grad((out * do).sum(), sink)
    formula = -(torch.exp(sink[None] - lse) * (out * do).sum(-1)).sum(0)
    torch.testing.assert_close(dsink, formula, rtol=1e-12, atol=1e-12)
    assert torch.equal(out[-1], torch.zeros_like(out[-1]))
    assert torch.equal(lse[-1], sink)


def test_empty_query_rejected_before_cuda_launch():
    from isoexec.ops.attention.sparse_sm100.sink import sparse_attention_sink_forward

    with pytest.raises(ValueError, match="T>0"):
        sparse_attention_sink_forward(torch.empty(0, 64, 512, dtype=torch.bfloat16),
                                      torch.empty(1, 512, dtype=torch.bfloat16),
                                      torch.empty(0, 640, dtype=torch.int32), torch.zeros(64), 512**-.5)


def _relative(actual, expected):
    return float((actual.detach().double() - expected.detach()).abs().max() / expected.detach().abs().max().clamp(min=1e-30))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires SM100")
def test_score_delta_matches_unrounded_fp64_expectation():
    from isoexec.ops.attention.sparse_sm100._extension import extension

    generator = torch.Generator(device="cuda").manual_seed(144)
    scores = torch.randn(3, 4, 640, device="cuda", generator=generator)
    dp = torch.randn(3, 4, 640, device="cuda", generator=generator)
    indices = torch.full((3, 640), -1, device="cuda", dtype=torch.int32)
    indices[0] = torch.arange(640, device="cuda")
    indices[1, 0] = 0
    sink = torch.tensor([-30., -5., 10., 30.], device="cuda", dtype=torch.float64)
    masked = (scores.double() * .125).masked_fill(indices[:, None] < 0, -torch.inf)
    normalizer = torch.cat((masked, sink[None, :, None].expand(3, -1, 1)), -1).logsumexp(-1)
    expected = ((masked - normalizer[..., None]).exp() * dp.double()).sum(-1)
    actual = torch.empty((3, 4), device="cuda")
    extension("bwd").score_delta(scores, dp, normalizer.float(), indices, actual, .125)
    torch.testing.assert_close(actual.double(), expected, rtol=2e-5, atol=1e-7)
    assert torch.equal(actual[-1], torch.zeros_like(actual[-1]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires SM100")
@pytest.mark.parametrize("heads,width,k,recipe", [(64, 512, 640, 1), (64, 512, 640, 2),
                                                (40, 512, 640, 1), (128, 576, 2048, 1)])
def test_native_sink_accuracy(heads, width, k, recipe):
    from isoexec.ops.attention.sparse_sm100.sink import sparse_attention_sink

    torch.manual_seed(23)
    q = (torch.randn(4, heads, width, device="cuda") * .3).bfloat16().requires_grad_()
    kv = (torch.randn(k + 9, width, device="cuda") * .3).bfloat16().requires_grad_()
    sink = torch.linspace(-20, 20, heads, device="cuda", requires_grad=True)
    lists = torch.full((4, k), -1, dtype=torch.int32, device="cuda")
    for t, count in enumerate((k, 1, 0, k - 1)):
        lists[t, :count] = torch.arange(count, device="cuda")
    do = (torch.randn(4, heads, 512, device="cuda") * .3).bfloat16()
    out, lse = sparse_attention_sink(q, kv, lists, sink, width**-.5, recipe=recipe)
    assert not lse.requires_grad
    (out * do).sum().backward()
    q64, kv64, s64 = (x.detach().double().requires_grad_() for x in (q, kv, sink))
    expected, normalizer = oracle(q64, kv64, lists, s64, width**-.5)
    (expected * do.double()).sum().backward()
    errors = [_relative(a, b) for a, b in ((out, expected), (q.grad, q64.grad),
                                           (kv.grad, kv64.grad), (sink.grad, s64.grad))]
    assert max(errors) <= 1e-2, errors
    torch.testing.assert_close(lse.double(), normalizer, rtol=2e-5, atol=2e-4)
    assert torch.equal(out[2], torch.zeros_like(out[2]))
    assert torch.equal(q.grad[2], torch.zeros_like(q.grad[2]))
    assert torch.equal(kv.grad[k:], torch.zeros_like(kv.grad[k:]))
    assert torch.equal(lse[2], sink)
    first = tuple(x.grad.clone() for x in (q, kv, sink))
    for x in (q, kv, sink):
        x.grad = None
    again, _ = sparse_attention_sink(q, kv, lists, sink, width**-.5, recipe=recipe)
    (again * do).sum().backward()
    assert torch.equal(again, out)
    assert all(torch.equal(x.grad, g) for x, g in zip((q, kv, sink), first, strict=True))
