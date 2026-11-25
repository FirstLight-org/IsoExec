"""CPU regression for current fixed-order combine engagement and backward execution."""

import torch

from isoexec.ops.moe.moe_sm90 import moe_combine_backward as C


def _combine_backward_stats():
    # inlined: the public module keeps the state but not this test accessor
    out = dict(C._COUNTS)
    out["decline_reasons"] = dict(C._DECLINES)
    return out


def test_fused_combine_counts_only_after_forward_and_backward_are_served(monkeypatch):
    tokens, topk, hidden = 5, 3, 7
    rows = torch.arange(tokens * topk).view(tokens, topk)
    permuted = torch.randn(tokens * topk, hidden, requires_grad=True)

    def fake_forward(permuted_tokens, sorted_indices, restore_shape, *, permuted_probs=None, rows=None, out_dtype=None):
        # the stand-in only has to accept `out_dtype`; the counter under test is engagement
        del sorted_indices, restore_shape, permuted_probs, out_dtype
        return permuted_tokens.index_select(0, rows.reshape(-1)).view(tokens, topk, hidden).sum(dim=1)

    def fake_backward(
        dout,
        combine_rows,
        num_permuted,
        *,
        out_dtype,
        permuted=None,
        permuted_probs=None,
        sorted_indices=None,
        impl="auto",
    ):
        del permuted, permuted_probs, sorted_indices, impl
        grad = torch.empty((num_permuted, hidden), dtype=out_dtype)
        grad[combine_rows.reshape(-1)] = dout[:, None, :].expand(-1, topk, -1).reshape(-1, hidden)
        return grad, None

    monkeypatch.setattr(C, "fused_fixed_order_combine", fake_forward)
    monkeypatch.setattr(C, "combine_backward", fake_backward)
    before = _combine_backward_stats()

    out = C.fused_combine(permuted, torch.arange(tokens * topk) % tokens, tokens, rows=rows)
    after_forward = _combine_backward_stats()
    assert after_forward["forward_served"] == before["forward_served"] + 1
    assert after_forward["backward_served"] == before["backward_served"]

    out.sum().backward()
    after_backward = _combine_backward_stats()
    assert after_backward["backward_served"] == before["backward_served"] + 1
    assert permuted.grad is not None
