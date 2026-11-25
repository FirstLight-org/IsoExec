"""The production ``norms.qk`` binding for each gamma convention: the engine (no autograd) and trainer
(autograd) forms are the recipe's bits, and the convention is what selects ``g = 1 + w`` or ``g = w``.

Run: CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=. $PY -m pytest -q isoexec/ops/attention/rope_sm90/tests/test_bound_conventions_gpu.py
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from isoexec.ops._testing import bitwise_equal
from isoexec.ops.attention.rope_sm90 import build_table
from isoexec.ops.attention.rope_sm90 import reference as R

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0), reason="requires SM90"
)

DEV = "cuda"
SITE = "model.layers.0.mixer.qk_norm"


def _bind(convention):
    from isoexec.ops.table import OpTable

    return OpTable().bind("norms.qk", site=SITE, inputs={"eps": 1e-6}, convention=convention)


def _stage(bound, theta, rot):
    rotary = SimpleNamespace(style="neox", theta=theta, rotary_dim=rot)
    spec = SimpleNamespace(mrope_section=None, mrope_interleaved=False)

    def run(q, k, wq, wk, positions):
        return bound.fn(q, k, wq, wk, 1e-6, seq=SimpleNamespace(positions=positions), rotary=rotary, spec=spec)

    return run


@pytest.mark.parametrize("convention", ["rms", "rms_zero_centered"])
@pytest.mark.parametrize("hq,hk,d,rot,theta", [(32, 8, 128, 128, 1e6), (4, 1, 128, 128, 1e6),
                                               (32, 4, 128, 128, 1e6), (2, 1, 256, 64, 1e7)])
def test_engine_and_trainer_forms_are_the_recipe(convention, hq, hk, d, rot, theta):
    g = torch.Generator(device=DEV).manual_seed(hq * 7 + d + (convention == "rms"))
    zc = convention == "rms_zero_centered"
    bound = _bind(convention)
    assert bound.impl == "sm90_qk_norm_rope_v2" and bound.inputs["convention"] == convention
    run = _stage(bound, theta, rot)
    T = 211
    # q|k as column views of one packed projection row, as the attention forward hands them over.
    P = torch.randn(T, (hq + 2 * hk) * d, device=DEV, generator=g).bfloat16()
    q, k = P[:, :hq * d].view(T, hq, d), P[:, hq * d:(hq + hk) * d].view(T, hk, d)
    wq = (torch.randn(d, device=DEV, generator=g) * 0.3 + (0.0 if zc else 1.0)).bfloat16()
    wk = (torch.randn(d, device=DEV, generator=g) * 0.3 + (0.0 if zc else 1.0)).bfloat16()
    pos = torch.randint(0, 40960, (T,), device=DEV, generator=g)
    tab = build_table(theta, rot, 40960, DEV)
    rq, _ = R.qk_norm_rope_recipe(q, wq, pos, tab.table, 1e-6, zero_centered=zc)
    rk, _ = R.qk_norm_rope_recipe(k, wk, pos, tab.table, 1e-6, zero_centered=zc)

    with torch.no_grad():  # engine form
        eq, ek = run(q, k, wq, wk, pos)
    assert bitwise_equal(eq, rq) and bitwise_equal(ek, rk)

    # trainer form: autograd records the backward; the forward bits are the same program.
    tq, tk = q.detach().clone().requires_grad_(), k.detach().clone().requires_grad_()
    twq, twk = wq.clone().requires_grad_(), wk.clone().requires_grad_()
    yq, yk = run(tq, tk, twq, twk, pos)
    assert yq.grad_fn is not None
    assert bitwise_equal(yq.detach(), rq) and bitwise_equal(yk.detach(), rk)
    gq, gk, gwq, gwk = torch.autograd.grad((yq, yk), (tq, tk, twq, twk), (torch.ones_like(yq), torch.ones_like(yk)))
    assert all(torch.isfinite(x.float()).all() for x in (gq, gk, gwq, gwk))

    # decode: one token per call is the same bits as the prefill pack.
    with torch.no_grad():
        for t in (0, 1, 100, T - 1):
            dq, dk = run(q[t:t + 1], k[t:t + 1], wq, wk, pos[t:t + 1])
            assert bitwise_equal(dq[0], eq[t]) and bitwise_equal(dk[0], ek[t])


def test_the_convention_selects_the_gain_and_neither_substitutes_for_the_other():
    """Plain gamma ``g`` and zero-centred ``g - 1`` (exact in fp32) are the same gains: same bits. The
    plain checkpoint read through the zero-centred binding (the +-1 bug class) is not."""
    g = torch.Generator(device=DEV).manual_seed(3)
    T, hq, hk, d = 64, 8, 2, 128
    q = torch.randn(T, hq, d, device=DEV, generator=g).bfloat16()
    k = torch.randn(T, hk, d, device=DEV, generator=g).bfloat16()
    gamma = (torch.rand(d, device=DEV, generator=g) * 1.5 + 0.5).bfloat16().float()
    pos = torch.arange(T, device=DEV)
    with torch.no_grad():
        plain = _stage(_bind("rms"), 1e6, 128)(q, k, gamma, gamma, pos)
        centred = _stage(_bind("rms_zero_centered"), 1e6, 128)(q, k, gamma - 1, gamma - 1, pos)
        wrong = _stage(_bind("rms_zero_centered"), 1e6, 128)(q, k, gamma, gamma, pos)
    assert torch.equal(1 + (gamma - 1), gamma)
    assert bitwise_equal(plain[0], centred[0]) and bitwise_equal(plain[1], centred[1])
    assert not bitwise_equal(plain[0], wrong[0])
