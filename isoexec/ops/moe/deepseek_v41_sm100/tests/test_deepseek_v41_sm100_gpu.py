"""GPU battery of deepseek_v41_bf16_actor_v1 (ops tier). Module binding:
isoexec/nn/tests/test_deepseek_experts_sm100_gpu.py."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from isoexec.core.arch import live_device_arch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or live_device_arch() != "sm100",
                                reason="needs a leased SM100 GPU")

from isoexec.ops.moe.deepseek_v41_sm100 import ops, training  # noqa: E402
from isoexec.ops.moe.deepseek_v41_sm100.tests import _reference as ref  # noqa: E402
from isoexec.ops.moe.deepseek_v41_sm100.tests._reference import rel  # noqa: E402

DEV = "cuda"
# a router_sm100-served sqrt-softplus geometry (E = 256, top-6); DeepSeek-V4.1-Flash is E = 384, top-6
E, K, H, I, SCALE, L = 256, 6, 512, 256, 1.5, 10.0
KW = dict(top_k=K, routed_scale=SCALE, limit=L)


def _gen(seed):
    return torch.Generator(device=DEV).manual_seed(seed)


def _setup(seed, grad=False):
    g = _gen(seed)
    gu = (torch.randn(E + 1, 2 * I, H, generator=g, device=DEV) * 0.05).to(torch.bfloat16).requires_grad_(grad)
    dn = (torch.randn(E + 1, H, I, generator=g, device=DEV) * 0.05).to(torch.bfloat16).requires_grad_(grad)
    rw = (torch.randn(E, H, generator=g, device=DEV) * 0.05).to(torch.bfloat16).requires_grad_(grad)
    rb = torch.randn(E, generator=g, device=DEV) * 0.01
    return gu, dn, rw, rb


def _oracle(x, rw, gu, dn, ids):
    """float64 DeepSeekExperts equation with the given selection (route weight before down)."""
    x64, rw64, gu64, dn64 = x.double(), rw.double(), gu.double(), dn.double()
    s = F.softplus(F.linear(x64, rw64)).sqrt().gather(-1, ids)
    wts = s / (s.sum(-1, keepdim=True) + 1e-20) * SCALE

    def mlp(e, z, w=None):
        h = F.silu(F.linear(z, gu64[e, :I]).clamp_max(L)) * F.linear(z, gu64[e, I:]).clamp(-L, L)
        return F.linear(h if w is None else h * w[:, None], dn64[e])
    out = mlp(E, x64)
    for e in ids.unique().tolist():
        t, j = torch.where(ids == e)
        out = out.index_add(0, t, mlp(e, x64[t], wts[t, j]))
    return out


def test_forward_program_and_oracle():
    gu, dn, rw, rb = _setup(1)
    x = (torch.randn(91, H, generator=_gen(1), device=DEV)).to(torch.bfloat16)
    st = ops.stages(x, rw, rb, gu, dn, **KW)
    # the activation stage is the declared FP32 rule on the bf16 projection, bitwise
    assert torch.equal(st.h, ref.swiglu_program(st.proj[:, :I].float(), st.proj[:, I:].float(), L, st.gp))
    assert torch.equal(st.gp[st.rt.plan.combine_rows[:, -1].long()], torch.ones(91, device=DEV))
    e = rel(st.y, _oracle(x, rw, gu, dn, st.rt.ids.long()))
    print(f"deepseek bf16 actor: rel vs fp64 equation {e:.3e}")
    assert e < 2e-2                     # bf16 roundings of the projections, h and roots (the recipe)


def test_relations_bitwise():
    gu, dn, rw, rb = _setup(2)
    x = (torch.randn(257, H, generator=_gen(2), device=DEV)).to(torch.bfloat16)
    y = ops.moe_forward(x, rw, rb, gu, dn, **KW)
    assert torch.equal(y, ops.moe_forward(x, rw, rb, gu, dn, **KW))
    for lo, hi in ((0, 1), (5, 6), (10, 80), (256, 257)):
        assert torch.equal(y[lo:hi], ops.moe_forward(x[lo:hi].clone(), rw, rb, gu, dn, **KW)), (lo, hi)
    assert ops.moe_forward(x[:0], rw, rb, gu, dn, **KW).shape == (0, H)


def test_training_identity_gradients_determinism():
    gu, dn, rw, rb = _setup(3, grad=True)
    x = torch.randn(77, H, generator=_gen(3), device=DEV).to(torch.bfloat16).requires_grad_()
    y = training.moe_train(x, rw, rb, gu, dn, **KW)
    assert torch.equal(y, ops.moe_forward(x.detach(), rw.detach(), rb, gu.detach(), dn.detach(), **KW))
    dy = torch.randn_like(y)
    y.backward(dy)
    first = [t.grad.clone() for t in (x, rw, gu, dn)]
    for t in (x, rw, gu, dn):
        t.grad = None
    training.moe_train(x, rw, rb, gu, dn, **KW).backward(dy)
    assert all(torch.equal(a, t.grad) for a, t in zip(first, (x, rw, gu, dn)))
    ids = ops.route(x.detach(), rw.detach(), rb, top_k=K, routed_scale=SCALE, num_experts=E).ids.long()
    x64, rw64, gu64, dn64 = (t.detach().double().requires_grad_() for t in (x, rw, gu, dn))
    _oracle(x64, rw64, gu64, dn64, ids).backward(dy.double())
    used = ids.unique().tolist() + [E]
    errs = {"x": rel(first[0], x64.grad), "router": rel(first[1], rw64.grad),
            "gate_up": rel(first[2][used], gu64.grad[used]), "down": rel(first[3][used], dn64.grad[used])}
    print("deepseek bf16 actor grads rel vs fp64:", {n: f"{v:.3e}" for n, v in errs.items()})
    assert max(errs.values()) < 3e-2


def test_zero_local_tokens_training_backward_is_defined_and_repeatable():
    """Empty local sequences must remain differentiable before EP transport."""
    gu, dn, rw, rb = _setup(31, grad=True)
    x = torch.empty((0, H), dtype=torch.bfloat16, device=DEV, requires_grad=True)
    leaves = (x, rw, gu, dn)
    runs = []
    for _ in range(2):
        output = training.moe_train(x, rw, rb, gu, dn, **KW)
        assert output.shape == (0, H) and output.requires_grad
        gradients = torch.autograd.grad(output.sum(), leaves)
        assert all(torch.isfinite(g).all() and torch.count_nonzero(g) == 0 for g in gradients)
        runs.append(gradients)
    assert all(torch.equal(a, b) for a, b in zip(*runs))
