"""The dual-rail leaf on an fp8 cache view of any row count (defect 13b: an engine pipeline stage whose allocation ends
within a page after a layer's last record gets a row-bounded view, ``engine.record_rows``): every plan (one-CTA leaf,
split, persistent) gives the bits of the same records held in whole 64-token pages, for the main and the extra
(compressed) cache, with -1 pads and lists that reach the last row."""
import pytest
import torch

from isoexec.core.arch import live_device_arch  # noqa: E402

from isoexec.ops.attention.sparse_sm100.tests import _inputs as inputs
from isoexec.ops.attention.sparse_sm100.dualrail import dualrail_attn

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or live_device_arch() != "sm100",
                                reason="the dual-rail leaf runs on a Blackwell device only")
SCALE = 512 ** -0.5


@pytest.mark.parametrize("plan", ["one", "2,4", "persist"])            # ISOEXEC_DUALRAIL_PLAN: leaf, split, persistent
@pytest.mark.parametrize("rows,extra_rows", [(3 * 64 + 17, 2 * 64 + 1), (64 + 63, 5), (128, 64 + 31)])
def test_row_bounded_cache_equals_whole_pages(monkeypatch, plan, rows, extra_rows):
    monkeypatch.setenv("ISOEXEC_DUALRAIL_PLAN", plan)
    g = torch.Generator(device="cuda").manual_seed(rows * 7 + extra_rows)
    R, H = 9, 64
    pages, epages = -(-rows // 64), -(-extra_rows // 64)
    q = (torch.randn((R, H, 512), generator=g, device="cuda") * 2).bfloat16()
    full = inputs.quantize_v41(torch.randn((pages * 64, 512), generator=g, device="cuda").bfloat16().view(-1, 64, 512))
    efull = inputs.quantize_v41((torch.randn((epages * 64, 512), generator=g, device="cuda") * 3).bfloat16()
                                .view(-1, 64, 512))
    bounded = full.reshape(-1, 528)[:rows].clone()          # its own allocation: nothing after the last row
    ebounded = efull.reshape(-1, 528)[:extra_rows].clone()
    K, KE = 128, 64
    lists = torch.full((R, K), -1, dtype=torch.int32, device="cuda")
    n = min(K - 5, rows)
    lists[:, :n] = torch.rand((R, rows), generator=g, device="cuda").topk(n, -1).indices.int()
    lists[0, 0] = rows - 1                                  # the last record
    elists = torch.full((R, KE), -1, dtype=torch.int32, device="cuda")
    ne = min(KE - 3, extra_rows)
    elists[:, :ne] = torch.rand((R, extra_rows), generator=g, device="cuda").topk(ne, -1).indices.int()
    elists[1, 0] = extra_rows - 1
    sink = torch.randn(H, generator=g, device="cuda")
    o, l = dualrail_attn(q, bounded, lists, SCALE, sink=sink, extra_kv=ebounded, extra_lists=elists)
    ow, lw = dualrail_attn(q, full, lists, SCALE, sink=sink, extra_kv=efull, extra_lists=elists)
    assert torch.equal(o.view(torch.int16), ow.view(torch.int16))
    assert torch.equal(l.view(torch.int32), lw.view(torch.int32))
