"""``fp8_rowwise_lookup`` against the exact decode statement (``reference.decode_fp8_rows``) bit for bit, every output
dtype, decode and prefill row counts, and the error word for an out-of-table index."""
import pytest
import torch

from isoexec.core.arch import live_device_arch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or live_device_arch() != "sm100",
                                reason="needs a leased SM100 GPU")

from isoexec.ops.embed.engram_sm100.ops import fp8_rowwise_lookup  # noqa: E402
from isoexec.ops.embed.engram_sm100.reference import decode_fp8_rows  # noqa: E402


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("M", [1, 128, 2048])
def test_lookup_is_the_exact_decode(dtype, M):
    g = torch.Generator(device="cuda").manual_seed(M)
    rows, D = 1 << 16, 256
    weight = (torch.randn(rows, D, generator=g, device="cuda") * 3).to(torch.float8_e4m3fn)
    scale = torch.randint(100, 150, (rows, D // 32), generator=g, device="cuda", dtype=torch.uint8).view(torch.float8_e8m0fnu)
    idx = torch.randint(0, rows, (M, 24), generator=g, device="cuda")
    got = fp8_rowwise_lookup(weight, scale, idx, out_dtype=dtype)
    want = decode_fp8_rows(weight, scale, idx, dtype)
    assert torch.equal(got.view(torch.uint8), want.view(torch.uint8))


def test_out_of_table_index_raises():
    weight = torch.zeros(16, 256, device="cuda").to(torch.float8_e4m3fn)
    scale = torch.full((16, 8), 127, device="cuda", dtype=torch.uint8).view(torch.float8_e8m0fnu)
    with pytest.raises(ValueError):
        fp8_rowwise_lookup(weight, scale, torch.tensor([[3, 16]], device="cuda"))
