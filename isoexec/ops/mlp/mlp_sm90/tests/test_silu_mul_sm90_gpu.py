"""SM90 fused ``mlp.act`` is the eager ``F.silu(g) * (u + 0.0)``, bit for bit, forward and backward."""

from __future__ import annotations

import pytest
import torch

from isoexec.ops._testing import bitwise_equal
from isoexec.ops.mlp.eager import silu_mul_megatron
from isoexec.ops.mlp.mlp_sm90.ops import silu_mul_fused

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0), reason="requires H100/SM90"
)

BF16 = torch.bfloat16
# Qwen3.8-27B intermediate 17408 at TP1/2/4/8, plus other production widths and a ragged one.
WIDTHS = (4352, 8704, 17408, 2176, 11008, 512, 131)
MS = (1, 2, 7, 64, 129, 256, 1024, 4096, 8192)
POPULATIONS = ("normal", "subnormal", "large", "signed_zero", "special")


def _population(shape, pop, gen):
    base = torch.randn(shape, device="cuda", dtype=torch.float32, generator=gen)
    if pop == "normal":
        return base.to(BF16)
    if pop == "subnormal":
        return (base * 2.0**-120).to(BF16)
    if pop == "large":
        return (base * 30.0).to(BF16)
    if pop == "signed_zero":
        z = torch.zeros(shape, device="cuda", dtype=torch.float32)
        z.view(-1)[::2] = -0.0
        z.view(-1)[1::4] = base.view(-1)[1::4]
        return z.to(BF16)
    pool = torch.tensor([float("inf"), float("-inf"), float("nan"), 3.3895e38, -3.3895e38, 1e-40, -1e-40, 0.0,
                         -0.0, 88.7, -88.7, 87.3, -87.3, 103.9, -103.9, 1.0, -1.0], device="cuda")
    idx = torch.randint(0, pool.numel(), shape, device="cuda", generator=gen)
    return torch.where(torch.rand(shape, device="cuda", generator=gen) < 0.5, pool[idx], base).to(BF16)


@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("m", MS)
def test_forward_bitwise_production_shapes(m, width):
    gen = torch.Generator(device="cuda").manual_seed(m * 131 + width)
    g = _population((m, width), "large", gen)
    u = _population((m, width), "normal", gen)
    with torch.no_grad():
        assert bitwise_equal(silu_mul_fused(g, u), silu_mul_megatron(g, u))


@pytest.mark.parametrize("gpop", POPULATIONS)
@pytest.mark.parametrize("upop", POPULATIONS)
def test_forward_bitwise_value_populations(gpop, upop):
    gen = torch.Generator(device="cuda").manual_seed(hash((gpop, upop)) & 0xFFFF)
    for m, width in ((7, 4352), (129, 131), (1024, 17408)):
        g, u = _population((m, width), gpop, gen), _population((m, width), upop, gen)
        with torch.no_grad():
            assert bitwise_equal(silu_mul_fused(g, u), silu_mul_megatron(g, u)), (m, width)


def test_misaligned_and_strided_operands():
    gen = torch.Generator(device="cuda").manual_seed(3)
    big = _population((65, 4353), "normal", gen)
    g, u = big[1:, 1:], big[:-1, :-1]  # non-contiguous, odd offsets: the host compacts, scalar path
    with torch.no_grad():
        assert bitwise_equal(silu_mul_fused(g, u), silu_mul_megatron(g, u))
    flat = _population((4352 * 3 + 1,), "normal", gen)
    g, u = flat[1:4353], flat[4354:4354 + 4352]  # 2-byte offset bases: scalar path
    with torch.no_grad():
        assert bitwise_equal(silu_mul_fused(g, u), silu_mul_megatron(g, u))


@pytest.mark.parametrize("needs", [(True, True), (True, False), (False, True)])
@pytest.mark.parametrize("m", (1, 64, 4096))
def test_backward_matches_eager_autograd(m, needs):
    gen = torch.Generator(device="cuda").manual_seed(m)
    g0, u0 = _population((m, 4352), "large", gen), _population((m, 4352), "normal", gen)
    dy = _population((m, 4352), "normal", gen)
    g, u = g0.clone().requires_grad_(needs[0]), u0.clone().requires_grad_(needs[1])
    eg, eu = g0.clone().requires_grad_(needs[0]), u0.clone().requires_grad_(needs[1])
    got, ref = silu_mul_fused(g, u), silu_mul_megatron(eg, eu)
    assert bitwise_equal(got.detach(), ref.detach())
    got.backward(dy)
    ref.backward(dy)
    for a, b, need in zip((g, u), (eg, eu), needs):
        assert (a.grad is None) == (not need)
        if need:
            assert bitwise_equal(a.grad, b.grad)


def _eager_vjp(dy, g, u):
    """The eager program's autograd VJP: ``silu_backward(dy * (u + 0.0), g)`` and ``dy * silu(g)``."""
    eg, eu = g.clone().requires_grad_(True), u.clone().requires_grad_(True)
    silu_mul_megatron(eg, eu).backward(dy)
    return eg.grad, eu.grad


@pytest.mark.parametrize("dpop", POPULATIONS)
@pytest.mark.parametrize("gpop", POPULATIONS)
@pytest.mark.parametrize("upop", ("normal", "signed_zero", "special"))
def test_backward_kernel_bitwise_value_populations(dpop, gpop, upop):
    """The one-kernel VJP is the eager four-kernel VJP bit for bit, -0/inf/nan/subnormal included."""
    from isoexec.ops.mlp.mlp_sm90._extension import extension

    gen = torch.Generator(device="cuda").manual_seed(hash((dpop, gpop, upop)) & 0xFFFF)
    for m, width in ((7, 4352), (129, 131), (1024, 17408 // 4)):
        dy, g, u = (_population((m, width), p, gen) for p in (dpop, gpop, upop))
        dg, du = extension().silu_mul_bwd(dy, g, u)
        rg, ru = _eager_vjp(dy, g, u)
        assert bitwise_equal(dg, rg), (m, width)
        assert bitwise_equal(du, ru), (m, width)


@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("m", (1, 129, 5800))
def test_backward_kernel_bitwise_production_shapes(m, width):
    from isoexec.ops.mlp.mlp_sm90._extension import extension

    gen = torch.Generator(device="cuda").manual_seed(m * 7 + width)
    dy, g, u = _population((m, width), "normal", gen), _population((m, width), "large", gen), _population(
        (m, width), "normal", gen)
    dg, du = extension().silu_mul_bwd(dy, g, u)
    rg, ru = _eager_vjp(dy, g, u)
    assert bitwise_equal(dg, rg) and bitwise_equal(du, ru)


def test_backward_kernel_strided_operands():
    """Column views of one packed [m, 2I] output (row-strided path) and odd offsets (compacting path)."""
    from isoexec.ops.mlp.mlp_sm90._extension import extension

    gen = torch.Generator(device="cuda").manual_seed(11)
    packed = _population((257, 2 * 4352), "large", gen)
    g, u = packed[:, :4352], packed[:, 4352:]
    dy = _population((257, 4352), "normal", gen)
    for a, b, c in ((dy, g, u), (dy[1:, 1:], g[:-1, 1:], u[1:, :-1])):
        dg, du = extension().silu_mul_bwd(a, b, c)
        rg, ru = _eager_vjp(a.contiguous(), b.contiguous(), c.contiguous())
        assert dg.is_contiguous() and du.is_contiguous()
        assert bitwise_equal(dg, rg) and bitwise_equal(du, ru)
