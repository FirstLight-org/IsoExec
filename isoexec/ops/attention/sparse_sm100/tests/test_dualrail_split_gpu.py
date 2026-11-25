"""The dual-rail leaf's split execution plan (csrc/dualrail/split.cuh) == the one-CTA plan, bit for bit.

Every plan parameter (tiles per qk CTA, column CTAs), V4.1 fp8 window cache with and without the fp8 extra (compressed) cache,
8 / 16 / 64 heads, rows 1..64, lazy-rescale-heavy q scales, -1 padding (partial and empty lists), topk_length arrays,
and inside a CUDA graph. The one-CTA plan's SASS is unchanged from the recipe's qualified build, so this is the
plan-invariance relation of the recipe (README "Dual-rail split plan")."""
import os

import pytest
import torch

from isoexec.core.arch import live_device_arch  # noqa: E402

from isoexec.ops.attention.sparse_sm100.dualrail import dualrail_attn, split_plan

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or live_device_arch() != "sm100",
                                reason="the dual-rail leaf runs on a Blackwell device only")
SCALE = 512 ** -0.5


def _e4m3(shape, g, scale):
    return (torch.randn(shape, generator=g, device="cuda") * scale).to(torch.float8_e4m3fn).view(torch.uint8)


def _case(B, r, H, seed, qscale=4.0, n_comp=4096):
    g = torch.Generator(device="cuda").manual_seed(seed)
    main = torch.empty((B * 3, 64, 528), dtype=torch.uint8, device="cuda")
    main[..., :512] = _e4m3(main[..., :512].shape, g, 50.0)
    main[..., 512:] = torch.randint(118, 126, main[..., 512:].shape, generator=g, device="cuda", dtype=torch.uint8)
    perm = torch.randperm(B * 3 * 64, generator=g, device="cuda")
    lists = perm[: B * 128].view(B, 128).int().contiguous()
    extra = extra_lists = None
    if r:
        extra = torch.empty((n_comp // 64, 64, 528), dtype=torch.uint8, device="cuda")
        extra[..., :512] = _e4m3(extra[..., :512].shape, g, 50.0)
        extra[..., 512:] = torch.randint(114, 122, extra[..., 512:].shape, generator=g, device="cuda", dtype=torch.uint8)
        extra_lists = torch.stack([torch.randperm(n_comp, generator=g, device="cuda")[:512].sort().values
                                   for _ in range(B)]).int().contiguous()
    q = (torch.randn((B, H, 512), generator=g, device="cuda") * qscale).bfloat16()
    sink = torch.randn(H, generator=g, device="cuda")
    if B >= 3:
        lists[1, 100:] = -1
        if r:
            extra_lists[1, 300:] = -1
            extra_lists[2] = -1
        else:
            lists[2] = -1
    return q, main, lists, extra, extra_lists, sink


def _run(c, plan, tl=None, etl=None):
    q, main, lists, extra, extra_lists, sink = c
    os.environ["ISOEXEC_DUALRAIL_PLAN"] = plan
    try:
        return dualrail_attn(q, main, lists, SCALE, sink=sink, topk_length=tl, extra_kv=extra, extra_lists=extra_lists,
                             extra_topk_length=etl)
    finally:
        os.environ.pop("ISOEXEC_DUALRAIL_PLAN", None)


def _eq(a, b):
    return torch.equal(a[0].view(torch.int16), b[0].view(torch.int16)) and torch.equal(a[1].view(torch.int32), b[1].view(torch.int32))


@pytest.mark.parametrize("r", [0, 1])
@pytest.mark.parametrize("H", [8, 16, 32, 64])
@pytest.mark.parametrize("B", [1, 3, 8, 33])
@pytest.mark.parametrize("qscale", [1.0, 16.0])
def test_every_split_plan_equals_one_cta(r, H, B, qscale):
    c = _case(B, r, H, seed=B * 100 + H + r, qscale=qscale)
    T = 2 + (8 if r else 0)
    ref = _run(c, "one")
    for tpc in sorted({1, 3, T}):
        for nc in (2, 4, 8):
            assert _eq(ref, _run(c, f"{tpc},{nc}")), (tpc, nc)


@pytest.mark.parametrize("r", [0, 1])
def test_topk_length_arrays(r):
    B = 9
    c = _case(B, r, 16, seed=91 + r)
    g = torch.Generator().manual_seed(5)
    tl = torch.randint(0, 129, (B,), generator=g).int().cuda()
    etl = torch.randint(0, 513, (B,), generator=g).int().cuda() if r else None
    ref = _run(c, "one", tl, etl)
    for plan in ("1,8", "2,4", "10,2"):
        assert _eq(ref, _run(c, plan, tl, etl)), plan


def test_default_plan_in_graph_and_repeat():
    c = _case(8, 1, 16, seed=3)
    assert split_plan(8, 10, 16, True) is not None
    ref = _run(c, "one")
    q, main, lists, extra, extra_lists, sink = c
    out = torch.empty_like(q)
    lse = torch.empty(q.shape[:2], dtype=torch.float32, device="cuda")
    fn = lambda: dualrail_attn(q, main, lists, SCALE, sink=sink, extra_kv=extra, extra_lists=extra_lists, out=out, lse=lse)
    fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    for _ in range(3):
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert _eq(ref, (out, lse))


def test_batch_rows_independent_under_split():
    """A row's bits do not depend on the batch it runs in (or on which plan that batch size selects)."""
    c = _case(48, 1, 16, seed=17)
    q, main, lists, extra, extra_lists, sink = c
    full = dualrail_attn(q, main, lists, SCALE, sink=sink, extra_kv=extra, extra_lists=extra_lists)
    for i in (0, 5, 47):
        one = dualrail_attn(q[i:i + 1], main, lists[i:i + 1], SCALE, sink=sink, extra_kv=extra, extra_lists=extra_lists[i:i + 1])
        assert torch.equal(one[0].view(torch.int16), full[0][i:i + 1].view(torch.int16))
        assert torch.equal(one[1].view(torch.int32), full[1][i:i + 1].view(torch.int32))
