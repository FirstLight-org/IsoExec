"""Engram's ``wkv`` column split over TP (``DeepSeekV41Block._engram_kv``) on one GPU: every rank's column block
through the actor's linear provider, interleaved in rank order, equals the replicated projection bit for bit at
the split's row counts (the weight's published segment count is 1 at both widths). The all-gather itself only
moves bytes; the TP2 engine gates run it."""
import pytest
import torch

from isoexec.core.arch import live_device_arch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or live_device_arch() != "sm100",
                                reason="needs a leased SM100 GPU")

from isoexec.models.deepseek_v4_1_flash.block import ENGRAM_TP_SPLIT_ROWS, _WeightRows  # noqa: E402
from isoexec.models.deepseek_v4_1_flash.providers import BF16Linear  # noqa: E402
from isoexec.ops.mm.mm_sm100 import segments as seg  # noqa: E402


@pytest.mark.parametrize("n,k", [(5 * 4096, 1024), (5 * 4096, 2048), (5 * 4096, 4096)])
@pytest.mark.parametrize("tp", [2, 4])
def test_rank_columns_interleave_to_the_replicated_projection(n, k, tp):
    if seg.segment_count(k, torch.bfloat16) != 1:
        pytest.skip("the split is refused at this shape (segment count > 1)")
    lin = BF16Linear()
    g = torch.Generator(device="cuda").manual_seed(n + k + tp)
    w = (torch.randn(n, k, generator=g, device="cuda") * 0.03).to(torch.bfloat16)
    holder = _WeightRows(w)
    c = n // tp
    with torch.no_grad():
        for m in range(1, ENGRAM_TP_SPLIT_ROWS + 1):
            x = torch.randn(m, k, generator=g, device="cuda").to(torch.bfloat16)
            want = lin(holder, x)
            parts = torch.stack([lin(_WeightRows(w[r * c:(r + 1) * c]), x) for r in range(tp)])   # [tp, m, c]
            got = parts.permute(1, 0, 2).reshape(m, n)
            assert torch.equal(got.view(torch.int16), want.view(torch.int16)), m
