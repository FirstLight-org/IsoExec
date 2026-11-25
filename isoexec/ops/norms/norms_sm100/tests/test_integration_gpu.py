from __future__ import annotations

import pytest
import torch

from isoexec.ops._testing import bitwise_equal
from isoexec.ops.norms.norms_sm100 import integration as I
from isoexec.ops.norms.norms_sm100 import ops as O

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0), reason="requires B200/SM100"
)
EPS = 1e-6


def test_bound_callables_have_the_op_table_signature():
    """``norms.rms.*`` is ``fn(x, w, eps)``; ``norms.gated_out`` is ``fn(x, z, w, eps)``."""
    x = (torch.randn(129, 2048, device="cuda") * 3).bfloat16()
    w = (torch.randn(2048, device="cuda") * 0.5).bfloat16()
    for zc in (False, True):
        assert bitwise_equal(I.rms(zc)(x, w, EPS), O.rms_norm(x, w, EPS, zc))
    t, hv, dv = 64, 8, 128
    gx = (torch.randn(t, hv, dv, device="cuda")).bfloat16()
    gz = (torch.randn(t, hv, dv, device="cuda") * 4).bfloat16()
    gw = (torch.randn(dv, device="cuda") * 0.5).bfloat16()
    out = I.gated(True)(gx, gz, gw, EPS)
    assert out.shape == (t, hv * dv)
    assert bitwise_equal(out, O.gated_rms_norm(gx, gz, gw, EPS, True))


def test_trainer_and_engine_forward_agree_bitwise():
    """The same row through the trainer's (grad-enabled) and the engine's (no_grad) bound callable."""
    x = (torch.randn(2048, 2048, device="cuda") * 3).bfloat16()
    w = (torch.randn(2048, device="cuda") * 0.5).bfloat16()
    fn = I.rms(True)
    trainer = fn(x.detach().clone().requires_grad_(True), w.detach().clone().requires_grad_(True), EPS)
    with torch.no_grad():
        engine_decode = fn(x[:1].clone(), w, EPS)
        engine_prefill = fn(x[:512].clone(), w, EPS)
    assert bitwise_equal(engine_decode, trainer.detach()[:1])
    assert bitwise_equal(engine_prefill, trainer.detach()[:512])
