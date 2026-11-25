"""Bounded native compressor training: all gradients, FP64 oracle, and repeats.

Uses the production 512-channel latent and FP32 projection/pooled BF16 boundary.
This is a single-GPU component gate, not a published-model qualification.
"""
import pytest
import torch
from torch import nn

from isoexec.ops.attention.deepseek_v41_cache_sm100 import ops

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a leased GPU")


def _round_bf16_ste(value):
    rounded = value.to(torch.bfloat16).double()
    return value + (rounded - value).detach()


def _oracle(x, kv_weight, gate_weight, norm_weight, *, batch, tokens, ratio, eps):
    """Independent channel-softmax pooling and RMSNorm with declared cast STE."""
    groups = tokens // ratio
    values = (x @ kv_weight.t()).view(batch, tokens, 512)[:, :groups * ratio]
    scores = (x @ gate_weight.t()).view(batch, tokens, 512)[:, :groups * ratio]
    values = values.reshape(batch, groups, ratio, 512)
    scores = scores.reshape(batch, groups, ratio, 512)
    pooled = (values * scores.softmax(dim=2)).sum(dim=2).reshape(-1, 512)
    pooled = _round_bf16_ste(pooled)
    normalized = pooled * torch.rsqrt(pooled.square().mean(-1, keepdim=True) + eps)
    return _round_bf16_ste(normalized * norm_weight)


@pytest.mark.parametrize("ratio", [2, 4])
def test_compressor_all_gradients_repeat_and_update_vs_fp64(ratio):
    batch, tokens, hidden, eps = 2, 17, 512, 1e-6
    generator = torch.Generator(device="cuda").manual_seed(710 + ratio)
    comp = nn.Module()
    comp.wkv = nn.Linear(hidden, 512, bias=False, device="cuda", dtype=torch.bfloat16)
    comp.wgate = nn.Linear(hidden, 512, bias=False, device="cuda", dtype=torch.bfloat16)
    comp.norm = nn.Module()
    comp.norm.weight = nn.Parameter(torch.ones(512, device="cuda", dtype=torch.bfloat16))
    with torch.no_grad():
        for linear in (comp.wkv, comp.wgate):
            linear.weight.copy_(torch.randn(linear.weight.shape, generator=generator,
                                           device="cuda") / hidden ** .5)
    x = torch.randn((batch * tokens, hidden), generator=generator,
                    device="cuda").bfloat16().requires_grad_()
    dy = torch.randn((batch * (tokens // ratio), 512), generator=generator,
                     device="cuda").bfloat16()
    leaves = (x, comp.wkv.weight, comp.wgate.weight, comp.norm.weight)
    optimizer = torch.optim.SGD(comp.parameters(), lr=0.025)
    for step in range(2):
        y, projection = ops.compressor_full(comp, x, batch, tokens, ratio, eps)
        assert projection.dtype == torch.float32 and projection.requires_grad
        actual = torch.autograd.grad(y, leaves, dy)
        repeated, _ = ops.compressor_full(comp, x, batch, tokens, ratio, eps)
        repeated_grads = torch.autograd.grad(repeated, leaves, dy)
        assert torch.equal(y, repeated)
        assert all(torch.equal(a, b) for a, b in zip(actual, repeated_grads))
        reference_leaves = tuple(t.detach().double().requires_grad_() for t in leaves)
        reference = _oracle(*reference_leaves, batch=batch, tokens=tokens, ratio=ratio, eps=eps)
        expected = torch.autograd.grad(reference, reference_leaves, dy.double())
        errors = {}
        for name, got, want in zip(("input", "wkv", "wgate", "norm"), actual, expected):
            assert torch.isfinite(got).all() and got.norm() > 0, name
            relative = float((got.double() - want).norm() / want.norm())
            peak = float((got.double() - want).abs().max() / want.abs().max())
            errors[name] = (relative, peak)
            # BF16 gradient GEMMs round the FP32 pool adjoint at their input;
            # FP64 oracle retains it. Bounds match the actor component gate.
            assert relative < .03 and peak < .03, (name, relative, peak)
        assert torch.count_nonzero(actual[0].view(batch, tokens, hidden)[:, -1]) == 0
        print(f"compressor ratio={ratio} step={step}: (relativeL2, peakrelative)={errors}", flush=True)
        before = comp.wkv.weight.detach().clone()
        for parameter, gradient in zip(comp.parameters(), actual[1:]):
            parameter.grad = gradient
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        assert not torch.equal(before, comp.wkv.weight)
