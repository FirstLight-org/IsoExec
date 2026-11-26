"""The dual-rail leaf's persistent execution plan (csrc/dualrail/persist.cuh) == the one-CTA plan, bit for bit; and the
row invariance of a prefill chunk: a row's bits do not depend on the chunk it is computed in (its size, the other rows,
its position) and equal its decode-shaped launch (one row, the decode plans).

V4.1 fp8 window cache alone and with the fp8 extra (compressed) cache, the bf16-kv mode (trainer rows), 8 / 16 / 64 / 128
heads, 1 .. 1000 rows (several rows per CTA through cluster launch control), lazy-rescale-heavy q scales, -1 padding
(partial, leading, and empty lists), extra_topk_length arrays, indices past the cache (a zero row, as the leaf's TMA
fill), repeat launches and a CUDA graph."""
import os

import pytest
import torch

from isoexec.core.arch import live_device_arch  # noqa: E402

from isoexec.ops.attention.sparse_sm100.dualrail import dualrail_attn

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or live_device_arch() != "sm100",
                                reason="the dual-rail leaf runs on a Blackwell device only")
SCALE = 512 ** -0.5


def _e4m3(shape, g, scale):
    return (torch.randn(shape, generator=g, device="cuda") * scale).to(torch.float8_e4m3fn).view(torch.uint8)


def _records(pages, g, lo, hi):
    t = torch.empty((pages, 64, 528), dtype=torch.uint8, device="cuda")
    t[..., :512] = _e4m3(t[..., :512].shape, g, 50.0)
    t[..., 512:] = torch.randint(lo, hi, t[..., 512:].shape, generator=g, device="cuda", dtype=torch.uint8)
    return t


def _case(R, H, seed, *, extra=True, qscale=4.0, n_win=4096, n_comp=4096, K=128):
    g = torch.Generator(device="cuda").manual_seed(seed)
    main = _records(n_win // 64, g, 118, 126)
    lists = torch.stack([torch.randperm(n_win, generator=g, device="cuda")[:K] for _ in range(R)]).int().contiguous()
    ex = ex_lists = None
    if extra:
        ex = _records(n_comp // 64, g, 114, 122)
        ex_lists = torch.stack([torch.randperm(n_comp, generator=g, device="cuda")[:512].sort().values
                                for _ in range(R)]).int().contiguous()
    q = (torch.randn((R, H, 512), generator=g, device="cuda") * qscale).bfloat16()
    sink = torch.randn(H, generator=g, device="cuda")
    if R >= 6:
        lists[1, 100:] = -1                       # partial
        lists[3, :77] = -1                        # leading pads (the first window rows of a prompt)
        lists[4, 5] = n_win + 64 * 7              # past the cache: a zero row, valid
        if extra:
            ex_lists[1, 300:] = -1
            ex_lists[2] = -1
            lists[5] = -1
            ex_lists[5] = -1                      # empty row
        else:
            lists[2] = -1                         # empty row
    return dict(q=q, kv=main, lists=lists, extra_kv=ex, extra_lists=ex_lists, sink=sink)


def _run(c, plan, rows=None, etl=None, q=None):
    os.environ["ISOEXEC_DUALRAIL_PLAN"] = plan
    try:
        sl = slice(None) if rows is None else rows
        return dualrail_attn(c["q"][sl] if q is None else q, c["kv"], c["lists"][sl].contiguous(), SCALE, sink=c["sink"],
                             extra_kv=c["extra_kv"],
                             extra_lists=None if c["extra_lists"] is None else c["extra_lists"][sl].contiguous(),
                             extra_topk_length=etl)
    finally:
        os.environ.pop("ISOEXEC_DUALRAIL_PLAN", None)


def _bits(a, b):
    return (torch.equal(a[0].view(torch.int16), b[0].view(torch.int16))
            and torch.equal(a[1].view(torch.int32), b[1].view(torch.int32)))


def _ndiff(a, b):
    return int((a[0].view(torch.int16) != b[0].view(torch.int16)).sum()), int((a[1].view(torch.int32) != b[1].view(torch.int32)).sum())


@pytest.mark.parametrize("extra", [False, True])
@pytest.mark.parametrize("H", [8, 16, 64, 128])
@pytest.mark.parametrize("R", [1, 7, 149, 1000])
@pytest.mark.parametrize("qscale", [1.0, 16.0])
def test_persist_equals_one_cta_fp8(extra, H, R, qscale):
    c = _case(R, H, seed=R * 1000 + H * 10 + extra, extra=extra, qscale=qscale)
    ref, got = _run(c, "one"), _run(c, "persist")
    assert _bits(ref, got), _ndiff(ref, got)


@pytest.mark.parametrize("K", [128, 640])
@pytest.mark.parametrize("H", [8, 64])
@pytest.mark.parametrize("R", [3, 300])
def test_persist_equals_one_cta_bf16(K, H, R):
    g = torch.Generator(device="cuda").manual_seed(K + H + R)
    S = 6144
    kv = (torch.randn((S, 512), generator=g, device="cuda") * 2).bfloat16()
    lists = torch.stack([torch.randperm(S, generator=g, device="cuda")[:K] for _ in range(R)]).int().contiguous()
    if R >= 3:
        lists[1, K // 3:] = -1
        lists[2] = -1
    c = dict(q=(torch.randn((R, H, 512), generator=g, device="cuda") * 6).bfloat16(), kv=kv, lists=lists,
             extra_kv=None, extra_lists=None, sink=torch.randn(H, generator=g, device="cuda"))
    ref, got = _run(c, "one"), _run(c, "persist")
    assert _bits(ref, got), _ndiff(ref, got)


def test_extra_topk_length_array():
    R, H = 300, 64
    c = _case(R, H, seed=5)
    g = torch.Generator(device="cuda").manual_seed(9)
    etl = torch.randint(0, 513, (R,), generator=g, device="cuda", dtype=torch.int32)
    etl[:4] = torch.tensor([0, 1, 64, 512], dtype=torch.int32)
    assert _bits(_run(c, "one", etl=etl), _run(c, "persist", etl=etl))


@pytest.mark.parametrize("H", [8, 64])
def test_prefill_rows_are_chunk_and_position_invariant(H):
    """A prompt's rows in one chunk == the same rows in other chunkings, in another order, and one row at a time (the
    decode-shaped launch: the one-CTA / split plans at R = 1), bit for bit."""
    R = 600
    c = _case(R, H, seed=77 + H, qscale=8.0)
    whole = _run(c, "persist")
    # one row per launch (decode shape, default plan selection), a sample of rows incl. the special ones
    for r in list(range(8)) + [123, 299, 599]:
        one = dualrail_attn(c["q"][r:r + 1], c["kv"], c["lists"][r:r + 1].contiguous(), SCALE, sink=c["sink"],
                            extra_kv=c["extra_kv"], extra_lists=c["extra_lists"][r:r + 1].contiguous())
        assert _bits((whole[0][r:r + 1], whole[1][r:r + 1]), one), r
    # other chunkings (the persistent plan per chunk)
    for cuts in ((0, 1, 2, 150, 151, 600), (0, 333, 600), (0, 128, 256, 384, 512, 600)):
        for a, b in zip(cuts[:-1], cuts[1:]):
            if b - a == 0:
                continue
            part = _run(c, "persist", rows=slice(a, b))
            assert _bits((whole[0][a:b], whole[1][a:b]), part), (a, b)
    # another order (position in the chunk)
    perm = torch.randperm(R, generator=torch.Generator().manual_seed(3)).cuda()
    os.environ["ISOEXEC_DUALRAIL_PLAN"] = "persist"
    try:
        shuf = dualrail_attn(c["q"][perm].contiguous(), c["kv"], c["lists"][perm].contiguous(), SCALE, sink=c["sink"],
                             extra_kv=c["extra_kv"], extra_lists=c["extra_lists"][perm].contiguous())
    finally:
        os.environ.pop("ISOEXEC_DUALRAIL_PLAN", None)
    assert _bits((whole[0][perm], whole[1][perm]), shuf)
    # the q rows as a strided view of a wider tensor (another batch layout)
    wide = torch.zeros((R, H + 8, 512), dtype=torch.bfloat16, device="cuda")
    wide[:, 4:4 + H] = c["q"]
    assert _bits(whole, _run(c, "persist", q=wide[:, 4:4 + H]))


def test_repeat_and_graph():
    R, H = 700, 64
    c = _case(R, H, seed=11, qscale=16.0)
    ref = _run(c, "persist")
    for _ in range(20):
        assert _bits(ref, _run(c, "persist"))
    out = torch.empty_like(c["q"])
    lse = torch.empty(c["q"].shape[:2], dtype=torch.float32, device="cuda")
    os.environ["ISOEXEC_DUALRAIL_PLAN"] = "persist"
    try:
        def call():
            dualrail_attn(c["q"], c["kv"], c["lists"], SCALE, sink=c["sink"], extra_kv=c["extra_kv"],
                          extra_lists=c["extra_lists"], out=out, lse=lse)
        call()
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            call()
        out.zero_()
        gr.replay()
        torch.cuda.synchronize()
    finally:
        os.environ.pop("ISOEXEC_DUALRAIL_PLAN", None)
    assert _bits(ref, (out, lse))


def test_not_admissible_falls_back():
    """topk_length or K < 128: the persistent plan is not admissible; a forced 'persist' runs the one-CTA plan."""
    c = _case(9, 64, seed=3)
    tl = torch.full((9,), 100, dtype=torch.int32, device="cuda")
    os.environ["ISOEXEC_DUALRAIL_PLAN"] = "persist"
    try:
        a = dualrail_attn(c["q"], c["kv"], c["lists"], SCALE, sink=c["sink"], topk_length=tl, extra_kv=c["extra_kv"],
                          extra_lists=c["extra_lists"])
    finally:
        os.environ.pop("ISOEXEC_DUALRAIL_PLAN", None)
    b = dualrail_attn(c["q"], c["kv"], c["lists"], SCALE, sink=c["sink"], topk_length=tl, extra_kv=c["extra_kv"],
                      extra_lists=c["extra_lists"])
    assert _bits(a, b)


@pytest.mark.parametrize("extra", [False, True])
@pytest.mark.parametrize("H", [8, 64])
@pytest.mark.parametrize("ratio", [0, 2])
def test_persist_fused_rope_equals_one_cta(extra, H, ratio):
    """The fused Q RoPE / inverse RoPE (fused_rope.cuh) in the persistent plan == in the one-CTA plan, each flag alone
    and both."""
    from isoexec.ops.attention.deepseek_v41_cache_sm100 import ops as cops
    from isoexec.ops.attention.deepseek_v41_cache_sm100.tests import _config
    cos, sin = cops.rope_table(_config.published_config(), ratio, 8192, "cuda")
    R = 400
    c = _case(R, H, seed=H * 7 + extra + 31 * ratio, extra=extra, qscale=8.0)
    pos = torch.randint(0, 8192, (R,), generator=torch.Generator(device="cuda").manual_seed(H), device="cuda",
                        dtype=torch.int32)
    for rq, ro in ((True, False), (False, True), (True, True)):
        outs = []
        for plan in ("one", "persist"):
            os.environ["ISOEXEC_DUALRAIL_PLAN"] = plan
            try:
                outs.append(dualrail_attn(c["q"], c["kv"], c["lists"], SCALE, sink=c["sink"], extra_kv=c["extra_kv"],
                                          extra_lists=c["extra_lists"], rope=(cos, sin, pos), rope_q=rq, rope_o=ro))
            finally:
                os.environ.pop("ISOEXEC_DUALRAIL_PLAN", None)
        assert _bits(outs[0], outs[1]), (rq, ro, _ndiff(outs[0], outs[1]))
