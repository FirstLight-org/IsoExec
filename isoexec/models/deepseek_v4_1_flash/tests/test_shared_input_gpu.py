"""Shared-input projections as one GEMM (``providers.BF16Linear.packed`` over ``pack_rows``): ``wq_a | wkv`` packed
equals the two separate projections bit for bit at every row count (the packed weight on mm_sm100, the members on
DeepGEMM up to ``NARROW_ROWS`` rows and mm_sm100 above), for every TP column shard of the packed weight, and under
CUDA-graph replay."""
import pytest
import torch
from torch import nn

from isoexec.core.arch import live_device_arch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or live_device_arch() != "sm100",
                                reason="needs a leased SM100 GPU")

from isoexec.models.deepseek_v4_1_flash.providers import BF16Linear, _Rows, pack_rows  # noqa: E402

MS = [1, 2, 3, 7, 8, 16, 31, 64, 127, 128, 129, 255, 256, 257, 511, 512, 1000, 2048, 4096, 8192]
K, NS_ = 5120, (1280, 512)


def _bits(t):
    return t.contiguous().view(torch.int16)


def _members(seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    hs = [_Rows(nn.Parameter((torch.randn(n, K, generator=g, device="cuda") * K ** -0.5).to(torch.bfloat16),
                             requires_grad=False)) for n in NS_]
    return g, hs


def test_packed_equals_separate_projections():
    lin = BF16Linear()
    g, hs = _members(1)
    with torch.no_grad():
        x0 = torch.randn(3, K, generator=g, device="cuda").to(torch.bfloat16)
        ref0 = [lin(h, x0) for h in hs]
        w = pack_rows([h.weight for h in hs])
        assert w is not None and lin.packs(hs)
        assert all(torch.equal(_bits(lin(h, x0)), _bits(r)) for h, r in zip(hs, ref0))   # values unchanged
        for m in MS:
            x = torch.randn(m, K, generator=g, device="cuda").to(torch.bfloat16)
            sep = [lin(_Rows(h.weight.detach().clone()), x) for h in hs]
            got = lin.packed(hs, x)
            for a, b in zip(got, sep):
                assert torch.equal(_bits(a), _bits(b)), m


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_column_shards_of_the_packed_weight(tp):
    lin = BF16Linear()
    g, hs = _members(2)
    with torch.no_grad():
        w = pack_rows([h.weight for h in hs])
        n = w.shape[0] // tp
        for m in (1, 8, 128, 300, 2048):
            x = torch.randn(m, K, generator=g, device="cuda").to(torch.bfloat16)
            full = torch.cat(lin.packed(hs, x), -1)
            shards = torch.cat([lin(_Rows(w[r * n:(r + 1) * n]), x) for r in range(tp)], -1)
            assert torch.equal(_bits(shards), _bits(full)), (tp, m)


def test_packed_graph_replay_equals_eager():
    lin = BF16Linear()
    g, hs = _members(3)
    with torch.no_grad():
        pack_rows([h.weight for h in hs])
        for m in (1, 8, 128, 512):
            x = torch.randn(m, K, generator=g, device="cuda").to(torch.bfloat16)
            want = [t.clone() for t in lin.packed(hs, x)]
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                lin.packed(hs, x)                                   # warm (JIT, plans) outside capture
            torch.cuda.current_stream().wait_stream(s)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = lin.packed(hs, x)
            x.copy_(torch.randn(m, K, generator=g, device="cuda").to(torch.bfloat16))
            graph.replay()
            fresh = [lin(_Rows(h.weight.detach().clone()), x) for h in hs]
            assert all(torch.equal(_bits(a), _bits(b)) for a, b in zip(out, fresh)), m
            assert not torch.equal(_bits(out[0]), _bits(want[0]))      # the replay read the new input
