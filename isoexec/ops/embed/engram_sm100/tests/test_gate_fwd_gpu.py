"""The gate forward (row-in-registers kernel) against the backward's recomputation of the same scalars
(``gate_scalars``, the row-loop program both kernels declare) bit for bit, and its output against
``bf16(x + gate * value)`` in IEEE FP32 (product rounded, then the add)."""
import pytest
import torch

from isoexec.core.arch import live_device_arch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or live_device_arch() != "sm100",
                                reason="needs a leased SM100 GPU")

from isoexec.ops.embed.engram_sm100._extension import extension  # noqa: E402


@pytest.mark.parametrize("T,D,wdtype,masked", [(1, 5120, torch.bfloat16, False), (37, 5120, torch.bfloat16, True),
                                               (256, 5120, torch.float32, False), (3, 1024, torch.bfloat16, False),
                                               (2, 8200, torch.bfloat16, False), (64, 5120, torch.bfloat16, True), (65, 5120, torch.bfloat16, False)])
def test_forward_gate_is_the_backward_recompute(T, D, wdtype, masked):
    S = 4
    g = torch.Generator(device="cuda").manual_seed(T + D)
    x = torch.randn(T, S, D, generator=g, device="cuda").bfloat16()
    kv = torch.randn(T, 5, D, generator=g, device="cuda").bfloat16()
    key, val = kv[:, :4], kv[:, 4]
    qw = (1 + 0.1 * torch.randn(S, D, generator=g, device="cuda")).to(wdtype)
    kw = (1 + 0.1 * torch.randn(S, D, generator=g, device="cuda")).to(wdtype)
    mask = (torch.rand(T, generator=g, device="cuda") > 0.3) if masked else None
    eps, scale = 1e-20, D ** -0.5
    out, gate = torch.empty_like(x), torch.empty(T, S, device="cuda")
    extension().gate_fwd(x, key, val, qw, kw, mask, out, gate, eps, scale)
    dout = torch.randn(T, S, D, generator=g, device="cuda").bfloat16()
    nseg = max(1, -(-T // 64))
    bufs = [torch.empty_like(x), torch.empty_like(key), torch.empty_like(val), torch.empty_like(qw), torch.empty_like(kw)]
    gate_b = torch.empty(T * S, device="cuda")
    extension().gate_bwd(x, key, val, qw, kw, mask, dout, *bufs, gate_b, torch.empty(T * S, device="cuda"),
                         torch.empty(nseg, S * D, device="cuda"), eps, scale)
    assert torch.equal(gate.view(-1).view(torch.int32), gate_b.view(torch.int32))
    want = (x.float() + gate[..., None] * val.float()[:, None, :]).bfloat16()
    assert torch.equal(out.view(torch.int16), want.view(torch.int16))
