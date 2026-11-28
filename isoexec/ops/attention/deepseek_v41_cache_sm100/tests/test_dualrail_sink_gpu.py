"""Recipe 3 (sm100_sparse_dualrail_sink_v1) through this package's sink attention: forward and backward vs
float64 autograd, the backward deterministic, and recipe 3 close to recipe 1 (different rounding programs)."""
import pytest
import torch

from isoexec.ops.attention.deepseek_v41_cache_sm100 import ops

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def test_dualrail_sink_forward_backward_vs_float64():
    g = torch.Generator(device="cuda").manual_seed(7)
    T, H, S, K = 40, 16, 1500, 640
    q = (torch.randn((T, H, 512), generator=g, device="cuda")).bfloat16()
    kv = torch.randn((S, 512), generator=g, device="cuda").bfloat16()
    lists = torch.full((T, K), -1, dtype=torch.int32, device="cuda")
    for t in range(T):
        n = min(K, 30 + 37 * t)
        lists[t, :n] = torch.randperm(S, generator=g, device="cuda")[:n].sort().values.int()
    sink = torch.randn(H, generator=g, device="cuda")
    dout = torch.randn((T, H, 512), generator=g, device="cuda").bfloat16()
    runs = []
    for _ in range(2):
        q1, k1, s1 = (x.clone().requires_grad_() for x in (q, kv, sink))
        o = ops.sink_attention(q1, k1, lists, s1, 512 ** -0.5, 3)
        runs.append((o,) + torch.autograd.grad(o, (q1, k1, s1), dout))
    assert all(torch.equal(a, b) for a, b in zip(*runs))
    q6, k6, s6 = (x.detach().double().requires_grad_() for x in (q, kv, sink))
    outs = []
    for t in range(T):
        ids = lists[t][lists[t] >= 0].long()
        sc = (q6[t] @ k6[ids].T) * 512 ** -0.5
        z = torch.cat([sc, s6[:, None]], 1).softmax(-1)
        outs.append(z[:, :-1] @ k6[ids])
    want = torch.stack(outs)
    eq, ek, es = torch.autograd.grad(want, (q6, k6, s6), dout.double())
    o, gq, gk, gs = runs[0]

    def rel(a, b):
        return float((a.double() - b).abs().max() / b.abs().max())
    errs = (rel(o, want), rel(gq, eq), rel(gk, ek), rel(gs, es))
    assert max(errs) <= 2e-2, errs
    o1 = ops.sink_attention(q, kv, lists, sink, 512 ** -0.5, 1)
    assert rel(o, o1.double()) <= 1e-2
