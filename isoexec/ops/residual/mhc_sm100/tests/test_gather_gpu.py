"""``gather_rows``: one launch equals ``index_select`` per tensor bit for bit (the CED segment gather's tensors and
dtypes: bf16 streams [T, 4, H] and h [T, H], fp32 pre / post [T, 4] and comb [T, 4, 4]), any row order and repeats,
empty index, and under CUDA-graph replay."""
import pytest
import torch

from isoexec.core.arch import live_device_arch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or live_device_arch() != "sm100",
                                reason="needs a leased SM100 GPU")

from isoexec.ops.residual.mhc_sm100.ops import gather_rows  # noqa: E402


def _tensors(T, H=5120, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return [torch.randn(T, 4, H, generator=g, device="cuda").bfloat16(), torch.randn(T, H, generator=g, device="cuda").bfloat16(),
            torch.randn(T, 4, generator=g, device="cuda"), torch.randn(T, 4, generator=g, device="cuda"),
            torch.randn(T, 4, 4, generator=g, device="cuda")]


def _same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.view(torch.uint8), b.view(torch.uint8))


@pytest.mark.parametrize("T,R", [(1, 1), (7, 3), (300, 128), (8192, 2048), (5, 0)])
def test_gather_rows_is_index_select(T, R):
    ts = _tensors(T, seed=T)
    g = torch.Generator(device="cuda").manual_seed(R)
    index = torch.randint(0, T, (R,), generator=g, device="cuda")
    with torch.no_grad():
        got = gather_rows(index, ts)
    assert all(_same(a, t.index_select(0, index)) for a, t in zip(got, ts))


def test_gather_rows_odd_row_bytes_and_graph():
    ts = _tensors(64, H=24) + [torch.arange(64 * 3, device="cuda", dtype=torch.uint8).view(64, 3)]
    index = torch.tensor([63, 0, 5, 5, 17], device="cuda")
    with torch.no_grad():
        gather_rows(index, ts)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = gather_rows(index, ts)
        index.copy_(torch.tensor([1, 2, 3, 62, 40], device="cuda"))
        graph.replay()
    assert all(_same(a, t.index_select(0, index)) for a, t in zip(out, ts))
