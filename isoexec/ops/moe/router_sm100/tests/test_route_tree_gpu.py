"""Bitwise gate of the segment-root router path against the fused router kernel: the logits program's
segments (one, the schedule policy's count: ``ops.declared_segments``) as DeepGEMM segment GEMMs (``mm/deepgemm_sm100/segtree.partials``) plus the tree-folding selection
(``route_select_tree``) give ``route_fused``'s ids and weights at EVERY row count the engine (``route``: the tree
form at every row count) or the trainer (always ``route_fused``) runs: rollout == trainer needs this equality."""
import pytest
import torch

from isoexec.core.arch import live_device_arch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or live_device_arch() != "sm100",
                                reason="needs a leased SM100 GPU")

from isoexec.ops.mm.deepgemm_sm100 import segtree  # noqa: E402
from isoexec.ops.moe.router_sm100 import ops  # noqa: E402

# every decode row count, the graph buckets and the prefill chunk up to 2048, and the trainer's whole-sequence batches
TS = (list(range(1, 65)) + [65, 96, 127, 128, 129, 255, 256, 384, 511, 512, 768, 1000, 1024, 1536, 2047, 2048,
                            3000, 4096, 8192, 16384])


# the served sqrt-softplus geometries (top_k 6; E 256 / 384 / 512) at hidden 4096 and the checkpoints' 5120
@pytest.mark.parametrize("E,H,k", [(256, 5120, 6), (384, 5120, 6), (256, 4096, 6), (512, 4096, 6)])
def test_tree_select_is_route_fused_at_every_row_count(E, H, k):
    g = torch.Generator(device="cuda").manual_seed(E + H)
    w = (torch.randn(E, H, generator=g, device="cuda") * 0.03).to(torch.bfloat16)
    bias = torch.randn(E, generator=g, device="cuda") * 0.01
    kw = dict(routing=ops.SQRTSOFTPLUS, top_k=k, expert_bias=bias, routed_scale=2.5, num_shared=1)
    xs = torch.randn(max(TS), H, generator=g, device="cuda").to(torch.bfloat16)
    xs[3] = xs[2]                                                     # duplicated rows (ties across rows)
    for T in TS:
        x = xs[:T].contiguous()
        want_i, want_w = ops.route_fused(x, w, **kw)
        got_i, got_w = ops.route_select_tree(segtree.partials(x, w, ops.declared_segments(H)), **kw)
        assert torch.equal(got_i, want_i) and torch.equal(got_w.view(torch.int32), want_w.view(torch.int32)), T
        r_i, r_w = ops.route(x, w, **kw)
        assert torch.equal(r_i, want_i) and torch.equal(r_w.view(torch.int32), want_w.view(torch.int32)), T


def test_tree_select_graph_replay_is_eager():
    g = torch.Generator(device="cuda").manual_seed(1)
    w = (torch.randn(256, 5120, generator=g, device="cuda") * 0.03).to(torch.bfloat16)
    bias = torch.randn(256, generator=g, device="cuda") * 0.01
    kw = dict(routing=ops.SQRTSOFTPLUS, top_k=6, expert_bias=bias, routed_scale=2.5, num_shared=1)
    x = torch.randn(32, 5120, generator=g, device="cuda").to(torch.bfloat16)
    want = ops.route(x, w, **kw)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = ops.route(x, w, **kw)
    x.copy_(torch.randn(32, 5120, generator=g, device="cuda").to(torch.bfloat16))
    graph.replay()
    fused = ops.route_fused(x, w, **kw)
    assert torch.equal(out[0], fused[0]) and torch.equal(out[1], fused[1])
