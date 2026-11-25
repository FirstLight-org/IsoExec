"""The sparse-attention backward providers on IsoExec's forward (BF16 KV rows, recipe-3 lists with pads, sink term,
64 heads, K = 128 + 512, ratios 2 / 1, B > 1): float64 accuracy, repeat-run bits (eager x5, CUDA-graph replay), and
agreement between the cuDNN deterministic provider and IsoExec's own. ``python -m <this module> digest OUT [dev]``
writes sha256 digests of one fixed backward (fresh-process / cross-device comparisons)."""
import hashlib
import json
import sys
import types

import pytest
import torch

from isoexec.ops.attention.deepseek_v41_cache_sm100 import ops
from isoexec.ops.attention.deepseek_v41_cache_sm100.tests import test_sink_bwd_invariance_gpu as inv

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
PROVIDERS = ["cudnn_det", "isoexec"]


@pytest.fixture
def provider(monkeypatch, request):
    monkeypatch.setattr(ops, "BWD_PROVIDER", request.param)
    if ops.backward_provider() != request.param:
        pytest.skip(f"{request.param} not available here")
    return request.param


def _case(B, T, ratio, seed):
    seqs = [inv._seq(seed + k, T, ratio) for k in range(B)]
    q, dO, kv, lists = inv._batch(seqs)
    return q, dO, kv, lists, inv._sink()


def _bwd(q, dO, kv, lists, sink):
    out, lse = ops.sink_forward(q, kv, lists, sink, inv.SCALE, 3)
    ctx = types.SimpleNamespace(saved_tensors=(q, kv, lists, sink, out, lse), scale=inv.SCALE, recipe=3,
                                sink_dtype=torch.float32)
    return lambda: ops._SinkFn.backward(ctx, dO)


@pytest.mark.parametrize("provider", PROVIDERS, indirect=True)
@pytest.mark.parametrize("B,T,ratio", [(1, 1100, 2), (2, 300, 1)])
def test_accuracy_vs_float64(provider, B, T, ratio):
    q, dO, kv, lists, sink = _case(B, T, ratio, 40 + ratio)
    got = inv._grads(q, dO, kv, lists, sink)
    q64, kv64, s64 = (t.detach().double().requires_grad_() for t in (q, kv, sink))
    sel = kv64[lists.clamp_min(0).long()]
    sc = torch.einsum("thd,tkd->thk", q64, sel) * inv.SCALE
    sc = sc.masked_fill(lists[:, None, :] < 0, -torch.inf)
    sc = torch.cat((sc, s64[None, :, None].expand(sc.shape[0], -1, 1)), dim=-1)
    ref = torch.einsum("thk,tkd->thd", sc.softmax(-1)[..., :-1], sel)
    want = torch.autograd.grad(ref, (q64, kv64, s64), dO.double())
    for name, a, w in zip(("dq", "dkv", "dsink"), got, want):
        err = float((a.double() - w).norm() / w.norm())
        print(f"{provider} B{B} T{T} r{ratio} {name} relL2={err:.3e}", flush=True)
        assert err < 0.015, (name, err)
    assert ops.BWD_PROGRAM_TAGS[provider] in ops.backward_programs_used()


@pytest.mark.parametrize("provider", PROVIDERS, indirect=True)
def test_repeat_eager_and_graph(provider):
    q, dO, kv, lists, sink = _case(2, 640, 2, 7)
    run = _bwd(q, dO, kv, lists, sink)
    first = [t.clone() for t in run() if t is not None]
    for _ in range(4):
        again = [t for t in run() if t is not None]
        assert all(torch.equal(inv._bits(a), inv._bits(b)) for a, b in zip(first, again))
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        run()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        outs = [t for t in run() if t is not None]
    for _ in range(3):
        for t in outs:
            t.zero_()
        g.replay()
        torch.cuda.synchronize()
        assert all(torch.equal(inv._bits(a), inv._bits(b)) for a, b in zip(first, outs))


def test_cudnn_det_agrees_with_isoexec(monkeypatch):
    q, dO, kv, lists, sink = _case(1, 1100, 2, 9)
    res = {}
    for p in PROVIDERS:
        monkeypatch.setattr(ops, "BWD_PROVIDER", p)
        if ops.backward_provider() != p:
            pytest.skip(f"{p} not available")
        res[p] = [t.double() for t in _bwd(q, dO, kv, lists, sink)() if t is not None]
    for name, a, b in zip(("dq", "dkv", "dsink"), res["cudnn_det"], res["isoexec"]):
        err = float((a - b).norm() / b.norm())
        print(f"cudnn_det vs isoexec {name} rel={err:.3e}", flush=True)
        assert err < 0.02, (name, err)


def digest(path, dev=0):
    torch.cuda.set_device(dev)
    ops.BWD_PROVIDER = "cudnn_det"
    assert ops.backward_provider() == "cudnn_det"
    q, dO, kv, lists, sink = _case(2, 1100, 2, 123)
    outs = [t for t in _bwd(q, dO, kv, lists, sink)() if t is not None]
    torch.cuda.synchronize()
    row = {"device": str(torch.cuda.get_device_properties(dev).uuid), "programs": ops.backward_programs_used()}
    for name, t in zip(("dq", "dkv", "dsink"), outs):
        row[name] = hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[1] == "digest":
    digest(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 0)
