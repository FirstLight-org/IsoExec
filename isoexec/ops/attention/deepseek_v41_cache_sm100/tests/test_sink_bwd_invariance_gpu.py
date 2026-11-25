"""The trainer sparse-attention backward (``ops._SinkFn``: compact lists + ``dsa_sparse_attn_bwd`` + ``sink_grad``)
is invariant to batch composition and row chunking, and bitwise run to run.

Relations (equalities on integer views):
* batch composition (informational, xfail: backward kernels need run-to-run determinism only): a sequence's dq rows and dkv rows are the same bits alone and at every position of a
  microbatch of B = 2 / 3 sequences (T not a multiple of any chunk or block size);
* chunking: ``dsa_sparse_attn_bwd`` with forced chunks of 1, 7, 64, 1000 rows == the default plan (dq, dkv);
* run to run: dq, dkv, dsink x 5;
* accuracy: dq, dkv, dsink against a float64 reference (relative L2 < 0.015).
"""
import pytest
import torch

from isoexec.ops.attention.deepseek_v41_cache_sm100 import ops
from isoexec.ops.attention.deepseek_v41_cache_sm100._extension import extension

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

H, D, W, TOPK = 64, 512, 128, 512
SCALE = D ** -0.5


def _topk(T, ratio, g):
    C = T // ratio
    out = torch.full((T, TOPK), -1, dtype=torch.int32, device="cuda")
    vis = (torch.arange(T, device="cuda") + 1) // ratio
    r = torch.rand((T, C), generator=g, device="cuda")
    r = torch.where(torch.arange(C, device="cuda")[None] < vis[:, None], r, -1.0)
    k = min(TOPK, C)
    sel = r.topk(k, -1)
    ids = torch.where(sel.values >= 0, sel.indices, 1 << 30).sort(-1).values
    out[:, :k] = torch.where(ids == (1 << 30), -1, ids).int()
    return out


def _seq(seed, T, ratio):
    g = torch.Generator(device="cuda").manual_seed(seed)
    S = T + T // ratio
    return dict(q=(torch.randn((T, H, D), generator=g, device="cuda") * 2).bfloat16(),
                dO=torch.randn((T, H, D), generator=g, device="cuda").bfloat16(),
                kv=(torch.randn((S, D), generator=g, device="cuda") * 0.5).bfloat16(),
                topk=_topk(T, ratio, g), T=T, S=S)


def _sink():
    g = torch.Generator(device="cuda").manual_seed(5)
    return torch.randn(H, generator=g, device="cuda")


def _batch(seqs):
    T, S, B = seqs[0]["T"], seqs[0]["S"], len(seqs)
    q = torch.cat([s["q"] for s in seqs])
    dO = torch.cat([s["dO"] for s in seqs])
    kv = torch.cat([s["kv"] for s in seqs])
    topk = torch.cat([s["topk"] for s in seqs]).contiguous()
    lists = torch.empty((B * T, W + TOPK), dtype=torch.int32, device="cuda")
    extension().build_lists(topk, None, B * T, T, S, 0, T, W, ops.list_mode(3), lists)
    return q, dO, kv, lists


def _grads(q, dO, kv, lists, sink):
    q, kv, sink = (t.detach().clone().requires_grad_() for t in (q, kv, sink))
    out = ops.sink_attention(q, kv, lists, sink, SCALE, recipe=3)
    return torch.autograd.grad(out, (q, kv, sink), dO)


def _bits(t):
    return t.contiguous().view(torch.int16) if t.dtype == torch.bfloat16 else t.contiguous().view(torch.int32)


@pytest.mark.xfail(strict=False, reason="informational (user ruling 2026-11-19): backward kernels need run-to-run determinism only, not batch-composition invariance")
@pytest.mark.parametrize("ratio", [2, 1])
def test_batch_composition(ratio):
    T = 300
    seqs = [_seq(100 * ratio + k, T, ratio) for k in range(3)]
    sink = _sink()
    S = seqs[0]["S"]
    alone = [_grads(*_batch([s]), sink) for s in seqs]
    for members in ([0, 1], [2, 0, 1], [1, 2]):
        dq, dkv, _ = _grads(*_batch([seqs[m] for m in members]), sink)
        for pos, m in enumerate(members):
            assert torch.equal(_bits(dq[pos * T:(pos + 1) * T]), _bits(alone[m][0])), (members, pos)
            assert torch.equal(_bits(dkv[pos * S:(pos + 1) * S]), _bits(alone[m][1])), (members, pos)


def test_chunking_and_repeat():
    from isoexec.ops.attention.dsa_sm100.bwd import dsa_sparse_attn_bwd
    s = _seq(7, 777, 2)
    sink = _sink()
    q, dO, kv, lists = _batch([s])
    out, lse = ops.sink_forward(q, kv, lists, sink, SCALE, 3)
    compact = extension().compact_lists(lists)
    ref = dsa_sparse_attn_bwd(q, kv, compact, out, lse, dO, SCALE, recompute_delta=True)
    for chunk in (1, 7, 64, 1000):
        got = dsa_sparse_attn_bwd(q, kv, compact, out, lse, dO, SCALE, recompute_delta=True, chunk=chunk)
        assert torch.equal(_bits(got[0]), _bits(ref[0])), chunk
        assert torch.equal(_bits(got[1]), _bits(ref[1])), chunk
    first = _grads(q, dO, kv, lists, sink)
    for _ in range(4):
        again = _grads(q, dO, kv, lists, sink)
        assert all(torch.equal(_bits(a), _bits(b)) for a, b in zip(first, again))


def test_accuracy_vs_float64():
    s = _seq(11, 200, 2)
    sink = _sink()
    q, dO, kv, lists = _batch([s])
    got = _grads(q, dO, kv, lists, sink)
    q64, kv64, s64 = (t.detach().double().requires_grad_() for t in (q, kv, sink))
    sel = kv64[lists.clamp_min(0).long()]
    sc = torch.einsum("thd,tkd->thk", q64, sel) * SCALE
    sc = sc.masked_fill(lists[:, None, :] < 0, -torch.inf)
    sc = torch.cat((sc, s64[None, :, None].expand(sc.shape[0], -1, 1)), dim=-1)
    ref = torch.einsum("thk,tkd->thd", sc.softmax(-1)[..., :-1], sel)
    want = torch.autograd.grad(ref, (q64, kv64, s64), dO.double())
    for name, a, w in zip(("dq", "dkv", "dsink"), got, want):
        err = float((a.double() - w).norm() / w.norm())
        assert err < 0.015, (name, err)
