"""Fused residual join + RMSNorm (``sm100_add_rms_v3``) is the unfused program, bit for bit, forward and backward.

The unfused program is what the captured graph runs without the fusion pass: ``h = residual + x``
(``aten.add`` on two bf16 tensors) and then the bound sm100 RMS recipe for the width (``ops.rms_norm``, with its
C++ autograd). Both outputs (``norm(h)`` and ``h``) and every gradient (x, residual, weight) are
compared as integer views.
"""

from __future__ import annotations

import pytest
import torch

from isoexec.ops._testing import bitwise_equal
from isoexec.ops.norms.norms_sm100 import ops as O
from isoexec.ops.norms.norms_sm100.tests.test_sm100_gpu import rows, weight

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10, reason="requires SM100"
)

EPS = 1e-6
MS = (1, 3, 7, 128, 256, 1000, 8192)


def _pair(m, n, seed):
    x = rows(m, n, scale=1.5, seed=seed)  # a sublayer output
    r = rows(m, n, scale=8.0, seed=seed + 100)  # the residual stream, wider range
    if m > 2:
        r[1] = -x[1]  # exact cancellation row: h = +-0
    return x, r


@pytest.mark.parametrize("zero_centered", [True, False])
@pytest.mark.parametrize("n", [5120, 4096, 2048, 1024])
@pytest.mark.parametrize("m", MS)
def test_forward_matches_unfused_and_recipe(m, n, zero_centered):
    x, r = _pair(m, n, seed=m * 7 + n)
    w = weight(n, zero_centered=zero_centered)
    with torch.no_grad():
        y, h = O.add_rms_norm_fused(x, r, w, EPS, zero_centered)
        h_ref = r + x
        y_ref = O.rms_norm(h_ref, w, EPS, zero_centered)
    assert bitwise_equal(h, h_ref)
    assert bitwise_equal(y, y_ref)


@pytest.mark.parametrize("m", MS)
@pytest.mark.parametrize("zero_centered", [True, False])
@pytest.mark.parametrize("n", [2048, 4096, 5120])
def test_backward_matches_unfused_autograd(m, zero_centered, n):
    x0, r0 = _pair(m, n, seed=m + 3)
    w0 = weight(n, zero_centered=zero_centered)
    g = torch.Generator(device="cuda").manual_seed(m)
    dy = (torch.randn(m, n, generator=g, device="cuda") * 0.3).bfloat16()
    dh = (torch.randn(m, n, generator=g, device="cuda") * 0.7).bfloat16()

    def run(fused):
        x, r, w = (t.detach().clone().requires_grad_(True) for t in (x0, r0, w0))
        if fused:
            y, h = O.add_rms_norm_fused(x, r, w, EPS, zero_centered)
        else:
            h = r + x
            y = O.rms_norm(h, w, EPS, zero_centered)
        torch.autograd.backward([y, h], [dy, dh])
        return y.detach(), h.detach(), x.grad, r.grad, w.grad

    got, ref = run(True), run(False)
    for name, a, b in zip(("y", "h", "dx", "dresidual", "dw"), got, ref):
        assert bitwise_equal(a, b), name


def test_backward_without_residual_gradient():
    """The last join of a stage: only the norm output is consumed."""
    m, n = 33, 5120
    x0, r0 = _pair(m, n, seed=9)
    w0 = weight(n, zero_centered=True)
    dy = torch.randn(m, n, device="cuda").bfloat16()

    def run(fused):
        x, r, w = (t.detach().clone().requires_grad_(True) for t in (x0, r0, w0))
        y = O.add_rms_norm_fused(x, r, w, EPS, True)[0] if fused else O.rms_norm(r + x, w, EPS, True)
        y.backward(dy)
        return x.grad, r.grad, w.grad

    for a, b in zip(run(True), run(False)):
        assert bitwise_equal(a, b)


def test_out_of_domain_operands_run_the_unfused_statement():
    m, n = 4, 5120
    x, r = _pair(m, n, seed=1)
    w = weight(n, zero_centered=True)
    with torch.no_grad():
        y, h = O.add_rms_norm_fused(x[:1], r, w, EPS, True)  # a broadcast join: not the kernel's domain
        h_ref = r + x[:1]
        assert bitwise_equal(h, h_ref) and bitwise_equal(y, O.rms_norm(h_ref, w, EPS, True))


def test_bound_through_the_fusion_pass_resolution():
    """The resolver binds the sm100 add + RMS identity matching the width's recipe, owning the join."""
    from isoexec.ops.norms.resolve import resolve

    b = resolve("norms.rms.pre_mlp", site="t", convention="rms_zero_centered", width=4096,
                residual="residual.add.attn")
    assert b.impl == "sm100_add_rms_v3" and b.owns == {"residual.add.attn": "sm100_add_rms_v3"}
    x, r = _pair(7, 4096, seed=4)
    w = weight(4096, zero_centered=True)
    with torch.no_grad():
        y, h = b.fn(x, r, w, EPS)
        assert bitwise_equal(h, r + x) and bitwise_equal(y, O.rms_norm(r + x, w, EPS, True))
