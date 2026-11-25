"""The three domains of the sparse attention through the ONE entry ``domain.sparse_attention``, and the step program
"blocks that hold only empty positions are not run".

* window only (SWA), window ++ selected (SWA + CSA, sink) and selected only (DSA, no sink) against a float64 oracle,
  with lists that leave empty positions, whole empty blocks, an empty part and an empty row;
* per-part lengths: a launch that is told each part's occupied prefix returns the bits of the launch that runs every
  block -- on the one-CTA, the split and the persistent plan, BF16 keys and FP8 records, at every fill around a block
  and a part boundary; a row alone == the row in a batch; repeat;
* selected only without a sink on the segmented leaf at the V3.2 / GLM-5 layout (keys 576 wide, value = the first
  512) and at 512: float64 oracle, a row alone == in a batch, repeat, the length relation.
"""
import os

import pytest
import torch

from isoexec.core.arch import live_device_arch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or live_device_arch() != "sm100",
                                reason="the sparse-attention leaves run on a Blackwell device only")

from isoexec.ops.attention.sparse_sm100.domain import Domain, domain_of, sparse_attention  # noqa: E402
from isoexec.ops.attention.sparse_sm100.tests import _inputs as inputs  # noqa: E402

SCALE = 512 ** -0.5


def _bits(a, b):
    return torch.equal(a[0].view(torch.int16), b[0].view(torch.int16)) and torch.equal(a[1].view(torch.int32), b[1].view(torch.int32))


def _oracle(q, parts, sink, v_dim=512):
    """float64: softmax over the occupied positions of every part, in any order (the sum is exact enough in fp64)."""
    R, H = q.shape[:2]
    out = torch.zeros(R, H, v_dim, dtype=torch.float64, device=q.device)
    for r in range(R):
        ks = [kv[l[r][l[r] >= 0].long()].double() for kv, l in parts]
        k = torch.cat(ks) if ks else None
        if k is None or k.shape[0] == 0:
            continue
        s = (q[r].double() @ k.T) * SCALE
        m = s.max(-1).values if sink is None else torch.maximum(s.max(-1).values, sink.double())
        e = torch.exp(s - m[:, None])
        den = e.sum(-1) + (0 if sink is None else torch.exp(sink.double() - m))
        out[r] = (e / den[:, None]) @ k[:, :v_dim]
    return out


def _prefix_lists(R, cap, S, fills, g):
    """[R, cap] lists whose row r holds fills[r % len] occupied leading positions (distinct rows of an S-row pool)."""
    lists = torch.full((R, cap), -1, dtype=torch.int32, device="cuda")
    length = torch.zeros(R, dtype=torch.int32, device="cuda")
    for r in range(R):
        n = min(fills[r % len(fills)], cap)
        lists[r, :n] = torch.randperm(S, generator=g, device="cuda")[:n].int()
        length[r] = n
    return lists.contiguous(), length


def _pools(seed, S=2048, Sc=1024):
    g = torch.Generator(device="cuda").manual_seed(seed)
    kw = torch.randn((S, 512), generator=g, device="cuda").bfloat16()
    kc = (torch.randn((Sc, 512), generator=g, device="cuda") * 2).bfloat16()
    cw, cc = inputs.quantize_v41(kw.view(-1, 64, 512)), inputs.quantize_v41(kc.view(-1, 64, 512))
    dw, dc = inputs.dequantize_v41(cw).reshape(-1, 512).contiguous(), inputs.dequantize_v41(cc).reshape(-1, 512).contiguous()
    return g, (cw, dw), (cc, dc)


W_FILLS = (128, 127, 65, 64, 63, 1, 0, 100)          # window: full, one short, around the block boundary, one key, empty
K_FILLS = (512, 511, 449, 448, 447, 65, 64, 63, 1, 0, 300)   # selected: exactly k, fewer, around block boundaries, none


def test_domain_names_and_refusals():
    l = torch.zeros((2, 128), dtype=torch.int32)
    kv = torch.zeros((4, 512), dtype=torch.bfloat16)
    assert domain_of(window=(kv, l)) == Domain(128, 0, False) and domain_of(window=(kv, l)).name == "window"
    assert domain_of(window=(kv, l), selected=(kv, l), sink=torch.zeros(2)).name == "window+selected"
    assert domain_of(selected=(kv, l)).name == "selected"
    with pytest.raises(ValueError):
        Domain(0, 0, False)
    with pytest.raises(ValueError):
        Domain(100, 0, False)
    with pytest.raises(ValueError, match="segmented leaf serves the selected-only domain"):
        sparse_attention(torch.zeros(2, 64, 512), SCALE, window=(kv, l), leaf="segmented")


@pytest.mark.parametrize("H", (64, 32))
@pytest.mark.parametrize("sink_on", (True, False))
def test_window_only(H, sink_on):
    g, (cw, dw), _ = _pools(1)
    R = len(W_FILLS)
    q = (torch.randn((R, H, 512), generator=g, device="cuda") * 2).bfloat16()
    sink = torch.randn(H, generator=g, device="cuda") if sink_on else None
    lists, length = _prefix_lists(R, 128, dw.shape[0], W_FILLS, g)
    ref = sparse_attention(q, SCALE, window=(dw, lists), sink=sink)
    want = _oracle(q, [(dw, lists)], sink)
    assert float((ref[0].double() - want).abs().max() / want.abs().max()) <= 1e-2
    assert (ref[0][6] == 0).all(), "an empty row is exact zeros"
    for kv in (dw, cw):                                  # BF16 keys and the FP8 records: the same bits, with and without lengths
        assert _bits(sparse_attention(q, SCALE, window=(kv, lists), sink=sink), ref)
        assert _bits(sparse_attention(q, SCALE, window=(kv, lists), sink=sink, window_length=length), ref)
    one = sparse_attention(q[2:3].contiguous(), SCALE, window=(cw, lists[2:3].contiguous()), sink=sink,
                           window_length=length[2:3].contiguous())
    assert torch.equal(one[0][0], ref[0][2])


def _both(R, H, seed):
    g, (cw, dw), (cc, dc) = _pools(seed)
    q = (torch.randn((R, H, 512), generator=g, device="cuda") * 2).bfloat16()
    sink = torch.randn(H, generator=g, device="cuda")
    wl, wn = _prefix_lists(R, 128, dw.shape[0], W_FILLS, g)
    sl, sn = _prefix_lists(R, 512, dc.shape[0], K_FILLS, g)
    return q, sink, (cw, dw, wl, wn), (cc, dc, sl, sn)


@pytest.mark.parametrize("H", (64, 32))
def test_window_and_selected_with_sink(H):
    R = 88                                                # every (window fill, selected fill) pair: 8 x 11
    q, sink, (cw, dw, wl, wn), (cc, dc, sl, sn) = _both(R, H, 2)
    arena = torch.cat([dw, dc]).contiguous()
    sl_arena = torch.where(sl >= 0, sl + dw.shape[0], sl).contiguous()
    ref = sparse_attention(q, SCALE, window=(arena, wl), selected=(arena, sl_arena), sink=sink)     # one BF16 list
    want = _oracle(q, [(dw, wl), (dc, sl)], sink)
    assert float((ref[0].double() - want).abs().max() / want.abs().max()) <= 1e-2
    full = sparse_attention(q, SCALE, window=(cw, wl), selected=(cc, sl), sink=sink)                # two FP8 caches
    assert _bits(full, ref), "FP8 records != BF16 on their exact dequantization"
    # the step program on each plan the launch shape can take: lengths of both parts, and of the selected part alone
    for plan in ("one", "2,4", "persist"):
        os.environ["ISOEXEC_DUALRAIL_PLAN"] = plan
        try:
            assert _bits(sparse_attention(q, SCALE, window=(cw, wl), selected=(cc, sl), sink=sink), ref), plan
            got = sparse_attention(q, SCALE, window=(cw, wl), selected=(cc, sl), sink=sink, selected_length=sn)
            assert _bits(got, ref), f"plan {plan}: selected lengths changed the bits"
            if plan != "persist":                         # the persistent plan takes no window length
                got = sparse_attention(q, SCALE, window=(cw, wl), selected=(cc, sl), sink=sink, window_length=wn,
                                       selected_length=sn)
                assert _bits(got, ref), f"plan {plan}: window + selected lengths changed the bits"
        finally:
            os.environ.pop("ISOEXEC_DUALRAIL_PLAN", None)
    # a row alone, a permuted batch, repeat -- with lengths
    for r in (0, 5, 17, 87):
        one = sparse_attention(q[r:r + 1].contiguous(), SCALE, window=(cw, wl[r:r + 1].contiguous()),
                               selected=(cc, sl[r:r + 1].contiguous()), sink=sink,
                               window_length=wn[r:r + 1].contiguous(), selected_length=sn[r:r + 1].contiguous())
        assert torch.equal(one[0][0], ref[0][r]), f"row {r} alone"
    perm = torch.randperm(R, device="cuda")
    got = sparse_attention(q[perm].contiguous(), SCALE, window=(cw, wl[perm].contiguous()), selected=(cc, sl[perm].contiguous()),
                           sink=sink, selected_length=sn[perm].contiguous())
    assert torch.equal(got[0], ref[0][perm])
    assert _bits(sparse_attention(q, SCALE, window=(cw, wl), selected=(cc, sl), sink=sink, selected_length=sn), ref)


@pytest.mark.parametrize("cap", (128, 512, 640))
def test_selected_only_without_a_sink_on_the_dualrail_leaf(cap):
    g, _, (cc, dc) = _pools(3)
    R = len(K_FILLS)
    q = (torch.randn((R, 64, 512), generator=g, device="cuda") * 2).bfloat16()
    sl, sn = _prefix_lists(R, cap, dc.shape[0], K_FILLS, g)
    assert domain_of(selected=(dc, sl)) == Domain(0, cap, False)
    ref = sparse_attention(q, SCALE, selected=(dc, sl))
    want = _oracle(q, [(dc, sl)], None)
    assert float((ref[0].double() - want).abs().max() / want.abs().max()) <= 1e-2
    assert torch.isinf(ref[1][9]).all() and (ref[0][9] == 0).all(), "an empty row: out 0, lse +inf without a sink"
    for kv in (dc, cc):
        assert _bits(sparse_attention(q, SCALE, selected=(kv, sl)), ref)
        assert _bits(sparse_attention(q, SCALE, selected=(kv, sl), selected_length=sn), ref)


@pytest.mark.parametrize("dqk", (576, 512))
@pytest.mark.parametrize("H", (64, 128))
def test_selected_only_without_a_sink_on_the_segmented_leaf(dqk, H):
    """(c): the V3.2 / GLM-5 layout -- keys 576 wide, value the first 512 -- and 512 / 512; K = 2048, BF16 keys."""
    T, K, S = 12, 2048, 3000
    q, kv, lists, _ = inputs.attn_train_inputs(7 + dqk, T, H, K, ctx=S, d_qk=dqk)
    lists = lists.view(T, K).contiguous()
    lists[3, 70:] = -1
    lists[5] = -1
    scale = dqk ** -0.5
    ref = sparse_attention(q, scale, selected=(kv, lists), leaf="segmented")
    assert domain_of(selected=(kv, lists)) == Domain(0, K, False)
    R, _ = q.shape[:2]
    want = torch.zeros(R, H, 512, dtype=torch.float64, device="cuda")
    for r in range(R):
        ids = lists[r][lists[r] >= 0].long()
        if ids.numel():
            k = kv[ids].double()
            want[r] = torch.softmax((q[r].double() @ k.T) * scale, -1) @ k[:, :512]
    assert float((ref[0].double() - want).abs().max() / want.abs().max()) <= 1e-2
    assert (ref[0][5] == 0).all()
    assert _bits(sparse_attention(q, scale, selected=(kv, lists), leaf="segmented"), ref), "repeat"
    for r in (0, 3, 11):
        one = sparse_attention(q[r:r + 1].contiguous(), scale, selected=(kv, lists[r:r + 1].contiguous()), leaf="segmented")
        assert torch.equal(one[0][0], ref[0][r]) and torch.equal(one[1][0].view(torch.int32), ref[1][r].view(torch.int32))
    length = (lists >= 0).sum(1, dtype=torch.int32).contiguous()
    assert _bits(sparse_attention(q, scale, selected=(kv, lists), selected_length=length, leaf="segmented"), ref)
