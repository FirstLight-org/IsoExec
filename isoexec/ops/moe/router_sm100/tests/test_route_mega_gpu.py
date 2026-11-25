"""Bitwise gate of the one-launch router (``route_mega``: the vendored DeepGEMM MegaGate with the logits program's
segments (one, the schedule policy's count) in tensor memory and ``select.cuh`` as its gate) against ``route`` (segment GEMMs + the tree-folding
selection): ids and weights equal bit for bit at every row count, under every launch-plan ownership (split-K slices
owning whole segments, single CTA or CTA pair), with masks, ties and non-finite logits, run to run, and under CUDA
graph replay."""
import pytest
import torch

from isoexec.core.arch import live_device_arch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or live_device_arch() != "sm100",
                                reason="needs a leased SM100 GPU")

from isoexec.ops.moe.deepseek_v41_sm100.spec import ROUTING  # noqa: E402
from isoexec.ops.moe.router_sm100 import ops  # noqa: E402

TS = (1, 2, 7, 16, 33, 64, 128, 512, 2048, 8192)
E, H, K, S, SCALE = 384, 5120, 6, 1, 1.5          # DeepSeek V4.1: 384 experts, top-6 + 1 shared slot


def _same(got, want, what):
    gi, gw = got
    wi, ww = want
    assert gi.dtype == wi.dtype == torch.int32 and gw.dtype == ww.dtype == torch.float32, what
    assert gi.shape == wi.shape and gw.shape == ww.shape, what
    bad = (gi != wi).any(1) | (gw.view(torch.int32) != ww.view(torch.int32)).any(1)
    assert not bool(bad.any()), f"{what}: {int(bad.sum())} rows differ, first {int(bad.nonzero()[0])}"


def _ref(x, w, **kw):
    """The two-launch tree program (the committed router before route_mega became the default), else the fused kernel."""
    from isoexec.ops.mm.deepgemm_sm100 import segtree
    H = x.shape[1]
    segments = ops.declared_segments(H)
    if segtree.serves(w.shape[0], H, segments):
        return ops.route_select_tree(segtree.partials(x.contiguous(), w.contiguous(), segments), **kw)
    return ops.route_fused(x, w, **kw)


def _v41(seed, T):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn(E, H, generator=g, device="cuda") * 0.03).to(torch.bfloat16)
    bias = torch.randn(E, generator=g, device="cuda") * 0.01
    x = torch.randn(T, H, generator=g, device="cuda").to(torch.bfloat16)
    kw = dict(routing=ROUTING, top_k=K, expert_bias=bias, routed_scale=SCALE, num_shared=S)
    return x, w, bias, kw


def test_route_mega_is_route_at_every_row_count():
    x, w, _, kw = _v41(0, max(TS))
    x[3] = x[2]                                                       # duplicated rows
    for T in TS:
        xt = x[:T].contiguous()
        _same(ops.route_mega(xt, w, **kw), _ref(xt, w, **kw), f"T={T}")


@pytest.mark.parametrize("split", [1])        # one segment (the schedule policy's count): K is never split
@pytest.mark.parametrize("ctas", [1, 2])
def test_every_ownership_is_route(split, ctas):
    """The heuristic's choices (split-K slices of whole segments; one CTA or a CTA pair per expert group, whose
    128-expert accumulators lie on the two lane halves) are ownership only."""
    x, w, _, kw = _v41(1, 8192)
    for T in (1, 33, 512, 2048, 8192):
        xt = x[:T].contiguous()
        _same(ops.route_mega(xt, w, num_split_k=split, num_mma_ctas=ctas, **kw), _ref(xt, w, **kw),
              f"T={T} split={split} ctas={ctas}")


def test_ties_mask_and_nonfinite_logits():
    x, w, bias, kw = _v41(2, 512)
    w[7] = w[5]                                                       # identical keys: the lower id must win
    bias[7] = bias[5]
    w[200] = w[100]
    x[10] = float("nan")                                              # every logit NaN: canonicalized to -inf
    x[11, :8] = float("inf")
    mask = torch.rand(512, device="cuda") > 0.3
    for T in (1, 16, 17, 512):
        xt = x[:T].contiguous()
        _same(ops.route_mega(xt, w, **kw), _ref(xt, w, **kw), f"T={T}")
        m = mask[:T].contiguous()
        _same(ops.route_mega(xt, w, mask=m, **kw), _ref(xt, w, mask=m, **kw), f"T={T} mask")


def test_without_bias_and_other_geometries():
    x, w, _, kw = _v41(3, 2048)
    kw = dict(kw, expert_bias=None)
    for T in (1, 64, 2048):
        _same(ops.route_mega(x[:T], w, **kw), _ref(x[:T], w, **kw), f"no bias T={T}")
    g = torch.Generator(device="cuda").manual_seed(4)
    for e, h, k, s in ((256, 4096, 6, 1), (512, 4096, 6, 1), (256, 5120, 8, 0)):
        w2 = (torch.randn(e, h, generator=g, device="cuda") * 0.03).to(torch.bfloat16)
        b2 = torch.randn(e, generator=g, device="cuda") * 0.01
        x2 = torch.randn(1024, h, generator=g, device="cuda").to(torch.bfloat16)
        kw2 = dict(routing=ops.SQRTSOFTPLUS, top_k=k, expert_bias=b2, routed_scale=2.5, num_shared=s)
        for T in (1, 33, 1024):
            _same(ops.route_mega(x2[:T], w2, **kw2), _ref(x2[:T], w2, **kw2), f"E={e} H={h} T={T}")


def test_sigmoid_grouped_geometry():
    """DeepSeek V3-class routing (sigmoid, 8 groups, top-4 groups, top-8): the selection core's grouped path, and a
    256-expert group on a CTA pair (UMMA M = 256)."""
    g = torch.Generator(device="cuda").manual_seed(5)
    w = (torch.randn(256, 7168, generator=g, device="cuda") * 0.03).to(torch.bfloat16)
    bias = torch.randn(256, generator=g, device="cuda") * 0.01
    x = torch.randn(4096, 7168, generator=g, device="cuda").to(torch.bfloat16)
    kw = dict(routing=ops.GROUPED, top_k=8, n_group=8, topk_group=4, expert_bias=bias, routed_scale=2.5)
    for T in (1, 64, 512, 4096):
        _same(ops.route_mega(x[:T], w, **kw), _ref(x[:T], w, **kw), f"T={T}")
        for split in (1,):                  # one segment: K is never split
            for ctas in (1, 2):
                _same(ops.route_mega(x[:T], w, num_split_k=split, num_mma_ctas=ctas, **kw), _ref(x[:T], w, **kw),
                      f"T={T} split={split} ctas={ctas}")


def test_run_to_run():
    x, w, _, kw = _v41(6, 2048)
    first = ops.route_mega(x, w, **kw)
    for _ in range(5):
        _same(ops.route_mega(x, w, **kw), first, "run to run")


def test_graph_replay_is_eager():
    """Captured after a warm-up ON THE CAPTURE STREAM (the JIT compile and that stream's barrier workspace are
    first-call work); replayed several times with new inputs, two row counts in one graph (two kernel nodes sharing
    the stream's barrier lines), equal to eager ``_ref`` (the tree program) every time."""
    x, w, _, kw = _v41(7, 512)
    stream = torch.cuda.Stream()
    sizes = (1, 64, 512)
    bufs = {T: x[:T].clone() for T in sizes}
    with torch.cuda.stream(stream):
        for T in sizes:
            ops.route_mega(bufs[T], w, **kw)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outs = {T: ops.route_mega(bufs[T], w, **kw) for T in sizes}
    g = torch.Generator(device="cuda").manual_seed(8)
    for rep in range(4):
        for T in sizes:
            bufs[T].copy_(torch.randn(T, H, generator=g, device="cuda").to(torch.bfloat16))
        graph.replay()
        torch.cuda.synchronize()
        for T in sizes:
            _same(outs[T], _ref(bufs[T], w, **kw), f"replay {rep} T={T}")


def test_default_route_is_the_tree_program_and_capture_falls_back():
    """route() now takes route_mega outside MEGA_SKIP: still the tree program's bits at every row count; inside a
    capture an unwarmed (stream, T) takes the two-launch path (no compile in capture), with the same bits."""
    from isoexec.ops.mm.deepgemm_sm100 import segtree
    for T in (1, 8, 64, 128, 512, 2048, 8192):
        x, w, _, kw = _v41(100 + T, T)
        want = ops.route_select_tree(segtree.partials(x, w, ops.declared_segments(H)), **kw)
        _same(ops.route(x, w, **kw), want, f"default T={T}")
    x, w, _, kw = _v41(5, 3)
    want = ops.route_select_tree(segtree.partials(x, w, ops.declared_segments(H)), **kw)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        segtree.partials(x, w, ops.declared_segments(H))                    # warm the two-launch path only (not route_mega at T = 3)
        ops.route_select_tree(segtree.partials(x, w, ops.declared_segments(H)), **kw)
    torch.cuda.current_stream().wait_stream(stream)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=stream):
        got = ops.route(x, w, **kw)
    g.replay()
    torch.cuda.synchronize()
    _same(got, want, "captured fallback")
