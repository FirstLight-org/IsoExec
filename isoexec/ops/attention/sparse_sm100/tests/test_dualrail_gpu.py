"""The dual-rail leaf (sm100_sparse_dualrail_v1 / _sink_v1): accuracy and its relations, all bitwise unless a
tolerance is named -- the fp8 cache and the fp8 + fp8 extra (compressed) cache == the bf16 path on their exact
dequantization; heads per launch (32 / 16 / 8 == the 64-head run, 128 == two 64-head blocks); batch permutation, a lone
row, repeat."""
import pytest
import torch

from isoexec.core.arch import live_device_arch  # noqa: E402

from isoexec.ops.attention.sparse_sm100.tests import _inputs as inputs
from isoexec.ops.attention.sparse_sm100.dualrail import dualrail_attn

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or live_device_arch() != "sm100",
                                reason="the dual-rail leaf runs on a Blackwell device only")
SCALE = 512 ** -0.5


def _oracle(q, kv, lists, sink=None):
    R, H = q.shape[:2]
    out = torch.zeros(R, H, 512, dtype=torch.float64, device=q.device)
    lse = torch.full((R, H), float("inf"), dtype=torch.float64, device=q.device)
    for r in range(R):
        ids = lists[r][lists[r] >= 0].long()
        if ids.numel() == 0:
            if sink is not None:
                lse[r] = sink.double()
            continue
        k = kv[ids].double()
        s = (q[r].double() @ k.T) * SCALE
        m = s.max(-1).values if sink is None else torch.maximum(s.max(-1).values, sink.double())
        e = torch.exp(s - m[:, None])
        den = e.sum(-1) + (0 if sink is None else torch.exp(sink.double() - m))
        out[r] = (e / den[:, None]) @ k
        lse[r] = m + torch.log(den)
    return out, lse


def _case(R, K, H, seed, S=4096):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = (torch.randn((R, H, 512), generator=g, device="cuda") * 2).bfloat16()
    kv = torch.randn((S, 512), generator=g, device="cuda").bfloat16()
    n = min(K - 13, S)
    lists = torch.full((R, K), -1, dtype=torch.int32, device="cuda")
    lists[:, :n] = torch.rand((R, S), generator=g, device="cuda").topk(n, -1).indices.int()   # list order: not sorted
    lists[1, 70:] = -1
    if R > 4:
        lists[4] = -1
    sink = torch.randn(H, generator=g, device="cuda") * 2
    return q, kv, lists, sink


def _rel(a, b):
    return float((a.double() - b).abs().max() / b.abs().max())


@pytest.mark.parametrize("K,H", [(128, 64), (640, 64), (640, 128), (640, 32)])
def test_bf16_and_fp8_vs_float64(K, H):
    q, kv, lists, sink = _case(9, K, H, seed=K + H)
    cache = inputs.quantize_v41(kv.view(-1, 64, 512))
    kvq = inputs.dequantize_v41(cache).reshape(-1, 512).contiguous()
    for s in (None, sink):
        ob, lb = dualrail_attn(q, kvq, lists, SCALE, sink=s)
        o8, l8 = dualrail_attn(q, cache, lists, SCALE, sink=s)
        assert torch.equal(ob.view(torch.int16), o8.view(torch.int16)) and torch.equal(lb.view(torch.int32), l8.view(torch.int32))
        want, wl = _oracle(q, kvq, lists, s)
        assert _rel(ob, want) <= 1e-2
        fin = torch.isfinite(wl)
        assert torch.equal(torch.isfinite(lb), fin)
        assert float((lb.double() - wl)[fin].abs().max()) <= 2e-3
        assert (ob[4] == 0).all() if q.shape[0] > 4 else True
        if s is not None and q.shape[0] > 4:
            assert torch.equal(lb[4], s)


@pytest.mark.parametrize("K,KE", [(704, 0), (2112, 0), (192, 512), (128, 576)])
def test_wider_lists_refused_by_name(K, KE):
    """The single-chain leaf is the declared program up to 640 slots per row (window + extra); wider is refused."""
    q, kv, lists, sink = _case(3, K, 64, seed=K)
    cache = inputs.quantize_v41(kv.view(-1, 64, 512))
    kw = {}
    if KE:
        kw = dict(extra_kv=cache, extra_lists=torch.full((3, KE), -1, dtype=torch.int32, device="cuda"))
    with pytest.raises(ValueError, match="declared single-chain width 640"):
        dualrail_attn(q, cache, lists, SCALE, sink=sink, **kw)


def test_topk_length():
    q, kv, lists, sink = _case(6, 640, 64, seed=3)
    tl = torch.tensor([640, 0, 65, 300, 1, 64], dtype=torch.int32, device="cuda")
    o, l = dualrail_attn(q, kv, lists, SCALE, sink=sink, topk_length=tl)
    cut = lists.clone()
    for r in range(6):
        cut[r, int(tl[r]):] = -1
    o2, l2 = dualrail_attn(q, kv, cut, SCALE, sink=sink)
    assert torch.equal(o, o2) and torch.equal(l.view(torch.int32), l2.view(torch.int32))


def test_fp8_plus_fp8_extra_equals_bf16_on_dequant():
    """Window records as the main list, compressed records (the same 528 B format, its own pool) as the extra list."""
    g = torch.Generator(device="cuda").manual_seed(5)
    R, S, Se = 7, 2048, 1024
    q = (torch.randn((R, 64, 512), generator=g, device="cuda") * 2).bfloat16()
    kv = torch.randn((S, 512), generator=g, device="cuda").bfloat16()
    ex = (torch.randn((Se, 512), generator=g, device="cuda") * 3).bfloat16()
    c8 = inputs.quantize_v41(kv.view(-1, 64, 512))
    ce = inputs.quantize_v41(ex.view(-1, 64, 512))
    d8 = inputs.dequantize_v41(c8).reshape(-1, 512)
    de = inputs.dequantize_v41(ce).reshape(-1, 512)
    lo = torch.rand((R, S), generator=g, device="cuda").topk(128, -1).indices.int()
    le = torch.rand((R, Se), generator=g, device="cuda").topk(500, -1).indices.int()
    le = torch.cat([le, torch.full((R, 12), -1, dtype=torch.int32, device="cuda")], 1)
    sink = torch.randn(64, generator=g, device="cuda")
    o, l = dualrail_attn(q, c8, lo.contiguous(), SCALE, sink=sink, extra_kv=ce, extra_lists=le.contiguous())
    arena = torch.cat([d8, de]).contiguous()
    lb = torch.cat([lo, torch.where(le >= 0, le + S, le)], 1).contiguous()
    ob, lbb = dualrail_attn(q, arena, lb, SCALE, sink=sink)
    assert torch.equal(o.view(torch.int16), ob.view(torch.int16)) and torch.equal(l.view(torch.int32), lbb.view(torch.int32))
    want, _ = _oracle(q, arena, lb, sink)
    assert _rel(o, want) <= 1e-2


@pytest.mark.parametrize("kind", ["bf16", "fp8"])
def test_heads_batch_and_repeat_relations(kind):
    q, kv, lists, sink = _case(11, 640, 64, seed=17)       # the declared width (MAX_LIST_SLOTS)
    src = kv if kind == "bf16" else inputs.quantize_v41(kv.view(-1, 64, 512))
    o, l = dualrail_attn(q, src, lists, SCALE, sink=sink)
    for hr in (32, 16, 8):
        for h0 in range(0, 64, hr):
            oh, lh = dualrail_attn(q[:, h0:h0 + hr].contiguous(), src, lists, SCALE, sink=sink[h0:h0 + hr])
            assert torch.equal(oh, o[:, h0:h0 + hr]) and torch.equal(lh.view(torch.int32), l[:, h0:h0 + hr].view(torch.int32))
    q2 = torch.cat([q, q.flip(1)], 1).contiguous()
    o2, _ = dualrail_attn(q2, src, lists, SCALE, sink=torch.cat([sink, sink.flip(0)]))
    assert torch.equal(o2[:, :64], o)
    perm = torch.randperm(11, device="cuda")
    op, _ = dualrail_attn(q[perm].contiguous(), src, lists[perm].contiguous(), SCALE, sink=sink)
    assert torch.equal(op, o[perm])
    one, _ = dualrail_attn(q[3:4].contiguous(), src, lists[3:4].contiguous(), SCALE, sink=sink)
    assert torch.equal(one[0], o[3])
    again, la = dualrail_attn(q, src, lists, SCALE, sink=sink)
    assert torch.equal(again, o) and torch.equal(la.view(torch.int32), l.view(torch.int32))
