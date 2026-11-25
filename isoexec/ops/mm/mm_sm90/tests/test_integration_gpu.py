"""The installed sm90 mm provider: identity read back off the object, served vs fall-through decided by the
shape alone, and the trainer's ``mm`` entry point reaching the same bits as the kernel."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from isoexec.core.arch import live_device_arch  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or live_device_arch() != "sm90",
    reason="the sm90 mm kernel runs on a Hopper device only",
)


def _bits(t):
    return t.view(torch.int16)


@pytest.fixture
def installed(monkeypatch):
    from isoexec.ops.mm import mm_batch_invariant as V
    from isoexec.ops.mm.mm_sm90 import integration

    saved = V.matmul_persistent
    monkeypatch.setenv("ISOEXEC_MM_SM90_SEGTREE", "1")
    assert integration.install_mm_sm90()
    yield integration
    V.matmul_persistent = saved
    integration._WRAPPER = None


def test_identity_and_served_shapes(installed):
    from isoexec.ops.mm import mm_batch_invariant as V
    from isoexec.ops.mm.mm_sm90 import ops
    from isoexec.ops.mm.trainer import mm_impl_id, mm_trainer

    assert mm_impl_id() == "sm90_wgmma_segtree"
    assert installed.install_mm_sm90()  # idempotent by identity
    x = torch.randn(37, 5120, device="cuda").bfloat16()
    w = (torch.randn(4120, 5120, device="cuda") * 0.02).bfloat16()
    before = installed.stats()
    y = mm_trainer(x, w)
    assert installed.stats()["served"] == before["served"] + 1
    assert torch.equal(_bits(y), _bits(ops.mm_sm90(x, w.t())))
    # bias, fp32 and K % 8 fall through to the vendored kernel, counted, never refused
    stock = V._isoexec_stock_matmul_persistent
    b = torch.randn(4120, device="cuda").bfloat16()
    yb = V.matmul_persistent(x, w.t(), bias=b)
    assert torch.equal(_bits(yb), _bits(stock(x, w.t(), bias=b)))
    xf, wf = x.float(), w.float()
    assert torch.equal(V.matmul_persistent(xf, wf.t()), stock(xf, wf.t()))
    assert installed.stats()["declined"] == before["declined"] + 2


def test_autograd_forward_is_the_kernel_and_backward_is_unchanged(installed):
    from isoexec.ops.mm.mm_sm90 import ops
    from isoexec.ops.mm.trainer import mm_trainer

    x = torch.randn(64, 1536, device="cuda").bfloat16().requires_grad_(True)
    w = (torch.randn(512, 1536, device="cuda") * 0.02).bfloat16().requires_grad_(True)
    y = mm_trainer(x, w)
    assert torch.equal(_bits(y.detach()), _bits(ops.mm_sm90(x.detach(), w.detach().t())))
    y.float().sum().backward()
    assert x.grad is not None and w.grad is not None
