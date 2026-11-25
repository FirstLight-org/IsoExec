"""The fused-wgrad placeholder contract (ops/mm/wgrad.py), on CPU with a stand-in for the fused GEMM.

The fused GEMM accumulates into ``main_grad`` and hands autograd a cached placeholder. These tests pin:
the placeholder is zeros and stays zeros; an unfused contribution to the same weight in the SAME backward
reaches ``main_grad`` (summed, never dropped); a DDP hook ordered after the re-arm hook (a second wrap)
adds only zeros.
"""

from __future__ import annotations

import torch

from isoexec.ops.mm import wgrad


class _FusedLinear(torch.autograd.Function):
    """``x @ w.T`` whose weight gradient goes straight into ``w.main_grad`` (what weight_grad does on GPU)."""

    @staticmethod
    def forward(ctx, x, w):
        ctx.save_for_backward(x, w)
        wgrad._arm(w)
        return x @ w.t()

    @staticmethod
    def backward(ctx, dy):
        x, w = ctx.saved_tensors
        w.main_grad.add_(dy.t() @ x)
        w.grad_added_to_main_grad = True
        return dy @ w, wgrad._dummy(w.shape, w.dtype, w.device)


def _ddp_hook(p):
    """Megatron DDP's accumulator hook (distributed_data_parallel._make_backward_post_hook)."""

    def hook(*_):
        if p.grad is not None and (not p.grad_added_to_main_grad or getattr(p, "zero_out_wgrad", False)):
            p.main_grad.add_(p.grad.data)
        p.grad = None

    return hook


def _param(seed, ddp_first=True):
    g = torch.Generator().manual_seed(seed)
    w = torch.nn.Parameter(torch.randn(6, 5, generator=g))
    w.main_grad = torch.zeros(6, 5)
    w.grad_added_to_main_grad = False
    acc = w.expand_as(w).grad_fn.next_functions[0][0]
    if ddp_first:
        acc.register_hook(_ddp_hook(w))
    return w, acc, g


def _placeholders_are_zero():
    return all(bool((b == 0).all()) for b in wgrad._DUMMIES.values())


def test_placeholder_is_zeros_and_recognized():
    d = wgrad._dummy((3, 4), torch.float32, torch.device("cpu"))
    assert bool((d == 0).all()) and wgrad.is_placeholder(d) and wgrad.is_placeholder(d[1:])
    assert not wgrad.is_placeholder(torch.zeros(3, 4))


def test_fused_only_backward_adds_once():
    w, acc, g = _param(0)
    x, dy = torch.randn(7, 5, generator=g), torch.randn(7, 6, generator=g)
    before = dict(wgrad.COUNTS)
    _FusedLinear.apply(x, w).backward(dy)
    assert torch.equal(w.main_grad, torch.zeros(6, 5).add_(dy.t() @ x))
    assert w.grad is None and w.grad_added_to_main_grad is False
    assert wgrad.COUNTS == before and _placeholders_are_zero()


def test_unfused_contribution_in_the_same_backward_is_summed_not_dropped():
    """The weight also feeds a plain matmul in the same graph (MTP sharing a head, LoRA, ...)."""
    w, acc, g = _param(1)
    x, x2 = torch.randn(7, 5, generator=g), torch.randn(4, 5, generator=g)
    dy, dy2 = torch.randn(7, 6, generator=g), torch.randn(4, 6, generator=g)
    n0 = wgrad.COUNTS["unfused_in_same_backward"]
    y = _FusedLinear.apply(x, w)
    y2 = x2 @ w.t()
    torch.autograd.backward([y, y2], [dy, dy2])
    want = torch.zeros(6, 5).add_(dy.t() @ x)
    want.add_(dy2.t() @ x2)  # DDP's add of the unfused part (zeros + g == g)
    assert torch.equal(w.main_grad, want)
    assert wgrad.COUNTS["unfused_in_same_backward"] == n0 + 1
    assert w.grad is None and w.grad_added_to_main_grad is False and _placeholders_are_zero()


def test_a_ddp_hook_registered_after_the_rearm_adds_only_zeros():
    """A second DDP wrap after the first fused forward orders its hook after ours: the flag is already cleared
    when it runs, so it adds param.grad -- the placeholder, which must be zeros."""
    w, acc, g = _param(2, ddp_first=False)
    x, dy = torch.randn(7, 5, generator=g), torch.randn(7, 6, generator=g)
    y = _FusedLinear.apply(x, w)  # arms first
    acc.register_hook(_ddp_hook(w))  # the re-wrap's hook
    y.backward(dy)
    assert torch.equal(w.main_grad, torch.zeros(6, 5).add_(dy.t() @ x).add_(torch.zeros(6, 5)))
    assert _placeholders_are_zero()


def test_zero_out_wgrad_adds_only_zeros():
    w, acc, g = _param(3)
    w.zero_out_wgrad = True
    x, dy = torch.randn(7, 5, generator=g), torch.randn(7, 6, generator=g)
    _FusedLinear.apply(x, w).backward(dy)
    assert torch.equal(w.main_grad, torch.zeros(6, 5).add_(dy.t() @ x))
    assert _placeholders_are_zero()
