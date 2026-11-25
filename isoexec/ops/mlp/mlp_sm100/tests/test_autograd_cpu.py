"""The fused forward retains the existing eager backward's rounding schedule."""
from types import SimpleNamespace

import pytest
import torch

from isoexec.ops.mlp.eager import silu_mul_megatron
from isoexec.ops.mlp.mlp_sm100 import ops


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("needs", [(True, True), (True, False), (False, True)])
def test_backward_matches_existing_eager(monkeypatch, dtype, needs):
    monkeypatch.setattr(ops, "extension", lambda: SimpleNamespace(silu_mul=silu_mul_megatron))
    torch.manual_seed(17)
    g, u = [torch.randn(9, 73).to(dtype).requires_grad_(need) for need in needs]
    eg, eu = [x.detach().clone().requires_grad_(need) for x, need in zip((g, u), needs)]
    dy = torch.randn_like(g)
    got = ops.silu_mul_fused(g, u)
    expected = silu_mul_megatron(eg, eu)
    assert torch.equal(got, expected)
    got.backward(dy)
    expected.backward(dy)
    for x, ex, need in zip((g, u), (eg, eu), needs):
        if need:
            assert torch.equal(x.grad.view(torch.uint8), ex.grad.view(torch.uint8))
        else:
            assert x.grad is None and ex.grad is None
