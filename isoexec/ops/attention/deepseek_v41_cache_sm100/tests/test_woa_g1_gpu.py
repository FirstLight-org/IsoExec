"""wo_a at one output group per rank (TP8): mm_sm100's dense GEMM (``ops.WOA_G1_DENSE``) == DeepGEMM's grouped einsum,
bit for bit, at decode and prefill row counts, for a contiguous and a strided (head-sliced) group-major input."""
import pytest
import torch

from isoexec.core.arch import live_device_arch
from isoexec.ops.attention.deepseek_v41_cache_sm100 import ops

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or live_device_arch() != "sm100",
                                reason="sm100 kernels")


def _out(og, wa, R, dense):
    old = ops.WOA_G1_DENSE
    ops.WOA_G1_DENSE = dense
    try:
        with torch.no_grad():
            return ops._grouped_out(og, wa, 1, R, deepgemm=True)
    finally:
        ops.WOA_G1_DENSE = old


@pytest.mark.parametrize("M", [1, 8, 16, 33, 128, 512, 2048, 8192])
def test_g1_dense_equals_grouped_einsum(M):
    g = torch.Generator(device="cuda").manual_seed(M)
    Dg, R = 4096, 1024
    og = (torch.randn((1, M, Dg), generator=g, device="cuda") * 3).bfloat16()
    wa = (torch.randn((R, Dg), generator=g, device="cuda") * 0.02).bfloat16()
    a, b = _out(og, wa, R, True), _out(og, wa, R, False)
    assert a.shape == b.shape == (M, R)
    assert torch.equal(a.view(torch.int16), b.view(torch.int16))
    # the leaf's token-major output read in place as the one group (row stride H * 512 with heads sliced)
    wide = torch.zeros((M, Dg + 512), dtype=torch.bfloat16, device="cuda")
    wide[:, 256:256 + Dg] = og[0]
    ogv = wide[:, 256:256 + Dg].unsqueeze(0)
    assert torch.equal(_out(ogv, wa, R, True).view(torch.int16), b.view(torch.int16))
