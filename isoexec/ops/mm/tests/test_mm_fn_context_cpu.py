"""Every ``_MmFn`` provider leaves the autograd context as the shared ``_MmFn.backward`` reads it.

``_SM100Mm`` (``ops/mm/sm100.py``) overrides ``forward`` to call the sm100 GEMM; it once saved only ``(x, w)``,
and the shared backward then raised ``AttributeError: ... no attribute 'wsink'`` on every trainer backward
through ``mm_linear``. These run each provider's forward and the shared backward on CPU, with the provider's
kernel replaced by ``torch.mm`` (the context is what is under test, not the kernel's bits).
"""

from __future__ import annotations

import pytest
import torch

from isoexec.ops.mm import sm100, trainer
from isoexec.ops.mm.mm_sm100 import ops as sm100_ops


def _cpu_mm(a, b, *, out=None):
    return a @ b


@pytest.fixture
def cpu_kernels(monkeypatch):
    monkeypatch.setattr(sm100_ops, "mm_sm100", _cpu_mm)
    monkeypatch.setattr(trainer, "_vendored_kernel", lambda: _cpu_mm)


def _operands(n):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(5, 16, generator=g, dtype=torch.float64).requires_grad_(True)
    w = torch.randn(n, 16, generator=g, dtype=torch.float64).requires_grad_(True)
    dy = torch.randn(5, n, generator=g, dtype=torch.float64)
    return x, w, dy


@pytest.mark.parametrize("n", [8, 13])  # 13: the forward pads N to the 16-byte pitch; the backward must not see it
@pytest.mark.parametrize("fn", ["sm100", "vendored"])
def test_backward_runs_and_gives_the_linear_gradients(cpu_kernels, fn, n):
    x, w, dy = _operands(n)
    y = sm100.mm_linear(x, w) if fn == "sm100" else trainer.mm_vendored(x, w)
    assert y.shape == (5, n)
    ctx = y.grad_fn
    assert ctx.wsink is None  # no grad-accumulation sink on a plain tensor
    assert ctx.tp_link is None
    xs, ws = ctx.saved_tensors
    assert ws.shape == (n, 16)  # the unpadded weight is what the backward differentiates against
    y.backward(dy)
    torch.testing.assert_close(x.grad, dy @ w.detach(), rtol=0, atol=0)
    torch.testing.assert_close(w.grad, dy.t() @ x.detach(), rtol=0, atol=0)


def test_a_weight_without_grad_has_no_sink(cpu_kernels):
    x, w, dy = _operands(8)
    w = w.detach()
    y = sm100.mm_linear(x, w)
    assert y.grad_fn.wsink is None
    y.backward(dy)
    torch.testing.assert_close(x.grad, dy @ w, rtol=0, atol=0)
