"""The gathered index-score route (``ops.index_scores(gather=True)``: the step's sequences' paged index keys packed
contiguous, scored by the contiguous kernel) == the paged route, bit for bit on ``[0, ke)`` of every row.

Relations:
* kernel level: one sequence (prefill from 0 and a chunk after cached tokens, up to T = 4096 rows) and several ragged
  sequences with rows of different sequences adjacent (Q blocks straddle sequence boundaries), decode rows and padded
  rows (context 0) mixed in; compress ratio 2 and 1, 64- and 128-entry pages, 32 and 64 index heads;
* the packed keys are the sequences' keys byte for byte at ``offsets[s]``, and the row ranges are their offsets;
* the gathered route captured in a CUDA graph and replayed over new positions == the eager paged route;
* end to end: paged attention layers over a mixed prefill / decode schedule give the same outputs with the route
  forced on, forced off and chosen by the rule; a CUDA-graph decode step with the route forced on replays bit-exact.
"""
import pytest
import torch

from isoexec.ops.attention.deepseek_v41_cache_sm100 import ops, paged
from isoexec.ops.attention.deepseek_v41_cache_sm100.tests import test_paged_gpu as tpg
from isoexec.ops.attention.dsa_sm100 import index_logits as il
from isoexec.ops.attention.dsa_sm100.tests._inputs import fp8_row_quantize

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _win(a, ke):
    mask = torch.arange(a.shape[1], device=a.device)[None, :] < ke.long()[:, None]
    return a.contiguous().view(torch.int16)[mask]


def _step(seed, plan, ratio, P, H, pad=0):
    """plan: (seq, p0, n) per sequence of the step (tokens of a sequence contiguous, in plan order). Sequence s holds
    (p0 + n) // ratio index keys in a pool of P-entry pages at shuffled page ids (spare pages hold random bytes).
    Returns the operands of ops.index_scores and the reference keys per sequence."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    seqs = max(s for s, _, _ in plan) + 1
    C = {s: (a + n) // ratio for s, a, n in plan}
    npg = max(1, max((c + P - 1) // P for c in C.values()))
    pages = seqs * npg + 3
    pool = torch.randint(0, 256, (pages, P * 132), generator=g, device="cuda", dtype=torch.uint8)
    bt = torch.randperm(pages, generator=g, device="cuda")[: seqs * npg].view(seqs, npg).to(torch.int32).contiguous()
    keys = {}
    for s, c in C.items():
        kc, ks = fp8_row_quantize(torch.randn((max(c, 1), 128), generator=g, device="cuda"))
        kc, ks = kc.view(torch.uint8)[:c], ks.view(-1)[:c]
        keys[s] = (kc, ks)
        for p in range((c + P - 1) // P):
            n = min(P, c - p * P)
            pool[bt[s, p].long(), : n * 128] = kc[p * P:p * P + n].reshape(-1)
            pool[bt[s, p].long(), P * 128: P * 128 + n * 4] = ks[p * P:p * P + n].contiguous().view(torch.uint8)
    tok_seq, pos = [], []
    for s, a, n in plan:
        tok_seq += [s] * n
        pos += list(range(a, a + n))
    tok_seq = torch.tensor(tok_seq + [plan[-1][0]] * pad, dtype=torch.int32, device="cuda")
    pos = torch.tensor(pos + [0] * pad, dtype=torch.int32, device="cuda")
    ke = ((pos + 1) // ratio).to(torch.int32)
    if pad:
        ke[-pad:] = 0                                     # padded rows (index_bounds: context 0)
    N = tok_seq.numel()
    q = torch.randn((N, H, 128), generator=g, device="cuda") * 2
    codes, sf = fp8_row_quantize(q)
    w = (torch.randn((N, H), generator=g, device="cuda") * 0.05).to(torch.bfloat16).contiguous()
    return dict(codes=codes.contiguous(), sf=sf.view(N, H).contiguous(), w=w, ke=ke.contiguous(),
                tok_seq=tok_seq.contiguous(), pos=pos, cache=pool.view(pages, P, 1, 132), bt=bt, width=npg * P,
                keys=keys, ratio=ratio)


def _scores(d, gather, **kw):
    return ops.index_scores(d["codes"], d["sf"], d["w"], d["ke"], width=d["width"], cache=d["cache"],
                            block_table=d["bt"], bt_rows=d["tok_seq"], gather=gather, **kw)


CASES = [
    # one sequence: prefill from 0, a chunk after cached tokens
    ("prefill", [(0, 0, 4096)], 2, 64, 32, 0),
    ("chunk", [(0, 4096, 4096)], 2, 128, 32, 0),
    ("chunk_r1", [(0, 2048, 2048)], 1, 64, 32, 0),
    ("prefill_r1_p128", [(0, 0, 3001)], 1, 128, 32, 0),
    # several ragged sequences, adjacent rows of different sequences, decode rows and padded rows mixed in
    ("mixed", [(0, 0, 1000), (1, 300, 517), (2, 4000, 1), (3, 0, 3), (4, 129, 1), (5, 0, 2000)], 2, 64, 32, 5),
    ("mixed_r1", [(3, 77, 5), (0, 0, 1023), (1, 1, 1), (2, 500, 777), (4, 0, 2)], 1, 128, 32, 3),
    ("mixed_h64", [(0, 0, 513), (1, 1000, 255), (2, 63, 2)], 2, 64, 64, 2),
]


@pytest.mark.parametrize("name,plan,ratio,P,H,pad", CASES, ids=[c[0] for c in CASES])
def test_gathered_route_equals_paged_route(name, plan, ratio, P, H, pad):
    d = _step(len(name) * 101 + ratio * 7 + P + H, plan, ratio, P, H, pad)
    ref = _scores(d, False)
    got = _scores(d, True)
    assert torch.equal(_win(got, d["ke"]), _win(ref, d["ke"]))
    # the step's shared ranges (paged.py computes them once per step and ratio) give the same bits
    ranges = ops.index_key_ranges(d["ke"], d["tok_seq"], d["bt"].shape[0], d["width"])
    again = _scores(d, True, ranges=ranges)
    assert torch.equal(_win(again, d["ke"]), _win(ref, d["ke"]))
    assert got.shape == ref.shape and got.stride() == ref.stride()


def test_packed_keys_and_ranges():
    plan = [(2, 0, 700), (0, 64, 300), (3, 9, 1), (1, 0, 2)]
    d = _step(17, plan, 2, 64, 32, pad=2)
    seqs = d["bt"].shape[0]
    ks, ke_out, offsets = ops.index_key_ranges(d["ke"], d["tok_seq"], seqs, d["width"])
    kmax = {s: 0 for s in range(seqs)}
    for s, e in zip(d["tok_seq"].tolist(), d["ke"].tolist()):
        kmax[s] = max(kmax[s], e)
    want, o = [], 0
    for s in range(seqs):
        want.append(o)
        o += kmax[s]
    assert offsets.tolist() == want + [o]
    seq_l = d["tok_seq"].long()
    assert torch.equal(ks, offsets[seq_l])
    assert torch.equal(ke_out, offsets[seq_l] + d["ke"])
    codes, sf = il.gather_keys(d["cache"], d["bt"], offsets, il.gather_capacity(d["ke"].numel(), seqs, d["width"]))
    for s in range(seqs):
        k = kmax[s]
        if k:
            kc, ksc = d["keys"][s]
            assert torch.equal(codes[want[s]:want[s] + k], kc[:k]), s
            assert torch.equal(sf[want[s]:want[s] + k], ksc[:k]), s


def test_gathered_route_in_cuda_graph():
    plan = [(0, 0, 900), (1, 200, 300), (2, 50, 1), (3, 0, 61)]
    d = _step(23, plan, 2, 64, 32, pad=4)
    run = lambda: _scores(d, True)                                         # noqa: E731
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        run()
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run()
    keys = torch.tensor([(a + n) // 2 for _, a, n in sorted(plan)], device="cuda", dtype=torch.int32)
    for k in range(3):
        # replay k > 0: new device metadata in the captured tensors (rows reassigned to other sequences, other
        # contexts, other queries), ke kept within each row's sequence's keys
        if k:
            d["tok_seq"].copy_(d["tok_seq"].roll(k))
            d["ke"].copy_(torch.minimum(torch.clamp(d["ke"] - k, min=0), keys[d["tok_seq"].long()]))
            for t in ("codes", "sf", "w"):
                v = d[t].view(torch.uint8) if d[t].element_size() == 1 else d[t]   # no roll kernel for fp8
                v.copy_(v.roll(1, 0))
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(_win(out, d["ke"]), _win(_scores(d, False), d["ke"])), k


@pytest.fixture
def force(monkeypatch):
    def set_(mode):
        monkeypatch.setattr(paged, "INDEX_GATHER", mode)
    return set_


def test_paged_layers_same_outputs_with_route_on_off(force):
    cfg, layers, xs = tpg._setup(3, seed=83)
    win, comp, P = tpg._tables(3, 29)
    slots = {0: 1, 1: 2, 2: 0}
    got = {}
    for mode in (False, True, None):
        force(mode)
        pools = tpg._pools(cfg, 3, P)
        got[mode] = [tpg._step(cfg, layers, xs, pools, plan, win, comp, slots) for plan in tpg.SCHEDULE[:6]]
    for mode in (True, None):
        for k, (a, b) in enumerate(zip(got[mode], got[False])):
            for i, (x, y) in enumerate(zip(a, b)):
                assert tpg._eq(x, y), (mode, k, i)


def test_paged_decode_graph_with_route_forced_on(force):
    force(True)
    tpg.test_paged_decode_step_cuda_graph_replay()
