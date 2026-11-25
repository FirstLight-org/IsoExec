"""Bitwise gate of the DeepGEMM BF16 projections against the incumbent programs: the dense GEMM against
``mm_sm100`` at segment count 1 (forward and the ``F.linear`` backward), the grouped einsum against
moe_sm100's grouped GEMM (``attention/deepseek_v41_cache_sm100.grouped_gemm``), over M = 1..2048."""
import pytest
import torch
from torch.nn import functional as F

from isoexec.core.arch import live_device_arch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or live_device_arch() != "sm100",
                                reason="needs a leased SM100 GPU")

from isoexec.ops.mm.deepgemm_sm100 import bf16, ops  # noqa: E402
from isoexec.ops.mm.mm_sm100 import ops as mm_ops  # noqa: E402

MS = [1, 2, 3, 7, 8, 9, 16, 31, 64, 127, 128, 255, 256, 511, 512, 1000, 1024, 2047, 2048]


def _bits(t):
    return t.contiguous().view(torch.int16)


@pytest.mark.parametrize("n,k", [(32768, 1280), (16384, 1280), (4096, 1280)])
def test_dense_is_mm_sm100(n, k):
    if not bf16.serves(n, k):
        pytest.skip("segment count > 1 at this shape: not served")
    g = torch.Generator(device="cuda").manual_seed(n + k)
    w = (torch.randn(n, k, generator=g, device="cuda") * 0.05).to(torch.bfloat16)
    for m in MS:
        x = torch.randn(m, k, generator=g, device="cuda").to(torch.bfloat16)
        want = mm_ops.mm_sm100(x, w.t())
        got = bf16._gemm(x, w)
        assert torch.equal(_bits(got), _bits(want)), m


def test_dense_backward_is_linear():
    g = torch.Generator(device="cuda").manual_seed(3)
    w = (torch.randn(4096, 1280, generator=g, device="cuda") * 0.05).to(torch.bfloat16)
    x = torch.randn(77, 1280, generator=g, device="cuda").to(torch.bfloat16)
    dy = torch.randn(77, 4096, generator=g, device="cuda").to(torch.bfloat16)
    grads = []
    for fn in (lambda a, b: bf16.DGLinearFn.apply(a, b), F.linear):
        xr, wr = x.clone().requires_grad_(), w.clone().requires_grad_()
        fn(xr, wr).backward(dy)
        grads.append((xr.grad, wr.grad))
    assert all(torch.equal(_bits(a), _bits(b)) for a, b in zip(*grads))


@pytest.mark.parametrize("G,R,D", [(8, 1024, 4096), (4, 1024, 4096)])
def test_grouped_einsum_is_grouped_gemm(G, R, D):
    from isoexec.ops.attention.deepseek_v41_cache_sm100.ops import grouped_gemm
    g = torch.Generator(device="cuda").manual_seed(G)
    wa = (torch.randn(G, R, D, generator=g, device="cuda") * 0.02).to(torch.bfloat16)
    for m in MS:
        og = torch.randn(G, m, D, generator=g, device="cuda").to(torch.bfloat16)
        want = grouped_gemm(og.reshape(G * m, D), wa, G, m).view(G, m, R).permute(1, 0, 2)
        got = torch.empty((m, G, R), dtype=torch.bfloat16, device="cuda")
        ops.bf16_grouped_bhr_hdr_bhd(og.permute(1, 0, 2), wa, got)
        assert torch.equal(_bits(got), _bits(want)), m


@pytest.mark.parametrize("n,k", [(5120, 1024), (12288, 1280), (4096, 1280)])
def test_padded_64_row_tile_is_the_default_tile(n, k):
    """The padded 64-row swap-AB tile (requested with set_block_size_multiple_of(.., 64)) changes ownership only:
    BF16 GEMM and the one-leaf FP32 root equal the default launch bitwise at every M."""
    from isoexec.ops.mm.deepgemm_sm100 import segtree
    dg = ops._dg()
    g = torch.Generator(device="cuda").manual_seed(n + 7 * k)
    w = (torch.randn(n, k, generator=g, device="cuda") * 0.05).to(torch.bfloat16)
    for m in MS:
        x = torch.randn(m, k, generator=g, device="cuda").to(torch.bfloat16)
        want = (bf16._gemm(x, w), segtree.linear(x, w, 1, out_dtype=torch.float32, noswap=False))
        dg.set_block_size_multiple_of((1, 64))
        try:
            got = (bf16._gemm(x, w), segtree.linear(x, w, 1, out_dtype=torch.float32, noswap=False))
        finally:
            dg.set_block_size_multiple_of((1, 1))
        assert torch.equal(_bits(got[0]), _bits(want[0])), m
        assert torch.equal(got[1].view(torch.int32), want[1].view(torch.int32)), m


def test_shape_table_changes_no_bit(monkeypatch):
    """The measured launch table (bf16.SHAPE_TABLE: early mode / non-swapped layout per M band) == the default launch,
    eager under inference_mode, at the fused wq_b | idx wq_b shape across and between the bands."""
    g = torch.Generator(device="cuda").manual_seed(41)
    w = (torch.randn(12288, 1280, generator=g, device="cuda") * 0.03).to(torch.bfloat16)
    for m in (1, 7, 8, 32, 33, 64, 96, 128, 129, 160, 256):
        x = torch.randn(m, 1280, generator=g, device="cuda").to(torch.bfloat16)
        with torch.inference_mode():
            got = bf16._gemm(x, w)
            monkeypatch.setattr(bf16, "SHAPE_TABLE", {})
            want = bf16._gemm(x, w)
            monkeypatch.undo()
        assert torch.equal(got.view(torch.int16), want.view(torch.int16)), m
