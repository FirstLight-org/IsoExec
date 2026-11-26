"""Stress persistent query-stage generations with cold, independently addressed keys.

The expected scores are analytic powers-of-two/integer products; no copy of the
kernel supplies the oracle. The unfixed kernel failed this workload intermittently.
Roughly 0.8 GiB peak allocation.
"""
import pytest
import torch
from isoexec.core.arch import live_device_arch
from isoexec.ops.attention.indexer_sm100._extension import extension
from isoexec.ops.attention.sparse_sm100.tests._inputs import fp8_row_quantize

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or live_device_arch() != "sm100",
    reason="requires SM100",
)


def test_cold_persistent_tail_query_generations():
    rows, pages, width = 2048, 36, 8192
    generator = torch.Generator(device="cuda").manual_seed(41)
    kc, ks = fp8_row_quantize(torch.ones((64, 128), device="cuda"))
    page = torch.cat((kc.view(torch.uint8).reshape(-1), ks.view(torch.uint8).reshape(-1)))
    cache = page.repeat(rows * pages, 1).reshape(rows * pages, 64, 1, 132)
    table = torch.zeros((rows, 128), device="cuda", dtype=torch.int32)
    table[:, :pages] = torch.randperm(rows * pages, generator=generator, device="cuda").reshape(rows, pages).int()
    factors = (2. ** (torch.arange(rows, device="cuda") % 5)).float()
    q, scales = fp8_row_quantize(factors[:, None, None].expand(rows, 64, 128).contiguous())
    weights = (torch.arange(rows, device="cuda") % 127 + 1).float()
    w = weights[:, None].expand(rows, 64).contiguous().bfloat16()
    lengths = torch.full((rows,), 2049, device="cuda", dtype=torch.int32)
    expected = (factors * weights * 1024)[:, None]
    output = torch.empty((rows, width), device="cuda", dtype=torch.float32)
    kernel = extension("indexer")
    def run():
        kernel.indexer_tree_fwd(q.view(torch.uint8), scales, cache, None, w,
                               None, lengths, table, output, 1, 16, 148, 2)
    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    # Neighboring full/tail tiles plus three ring generations. Checking every
    # replay prevents a later correct write from hiding an earlier wrong score.
    for length in (2047, 2048, 2049, 2117, 2175):
        lengths.fill_(length)
        run()
        assert torch.equal(output[:, :length], expected.expand(rows, length))
        for replay in range(1000):
            graph.replay()
            assert torch.equal(output[:, :length], expected.expand(rows, length)), (length, replay)
