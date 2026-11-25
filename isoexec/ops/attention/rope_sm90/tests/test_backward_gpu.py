"""rope_sm90 backward gates: gradients against float64 autograd (no invariance claim), connectivity,
graph capture, and a loose cross-check against TransformerEngine's fused RoPE.

Run: CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=. $PY -m pytest -q isoexec/ops/attention/rope_sm90/tests/test_backward_gpu.py
"""

from __future__ import annotations

import pytest
import torch

from isoexec.ops.attention.rope_sm90 import build_table, qk_norm_rope, rope
from isoexec.ops.attention.rope_sm90 import reference as R

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0), reason="requires SM90"
)

DEV = "cuda"
BF = torch.bfloat16
# Mean relative error bounds: fp32 grads are exact fp32 calculus; bf16 grads carry one bf16 rounding
# (~1.4e-3 mean relative measured), fp16 grads one fp16 rounding (~1.8e-4 measured).
TOL = {torch.float32: 1e-6, torch.bfloat16: 5e-3, torch.float16: 5e-4}


def rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).abs().sum() / b.abs().sum().clamp_min(1e-300))


def rnd(*shape, dtype=BF, scale=1.0):
    return (torch.randn(*shape, device=DEV) * scale).to(dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("hq,hk,D,rot", [(32, 8, 128, 128), (20, 1, 64, 64), (2, 1, 256, 64), (4, 0, 128, 32)])
@pytest.mark.parametrize("layout", [("neox", "neox"), ("interleaved", "neox"), ("interleaved", "interleaved"),
                                    ("neox", "interleaved")])
@pytest.mark.parametrize("mrope", [False, True])
def test_rope_grads_match_float64(dtype, hq, hk, D, rot, layout, mrope):
    torch.manual_seed(hq + D + rot)
    sections = (rot // 2 - 2 * (rot // 8), rot // 8, rot // 8) if mrope else None
    tab = build_table(1e6, rot, 4096, DEV, mrope_section=sections, mrope_interleaved=mrope)
    T = 131
    q = rnd(T, hq, D, dtype=dtype).requires_grad_()
    k = rnd(T, hk, D, dtype=dtype).requires_grad_() if hk else None
    pos = torch.randint(0, 4096, (3, T) if mrope else (T,), device=DEV)
    si, so = layout
    qo, ko = rope(q, k, pos, tab, style=si, out_style=so)
    gq = rnd(T, hq, D, dtype=dtype)
    outs, ins, gouts = [qo], [q], [gq]
    if hk:
        gk = rnd(T, hk, D, dtype=dtype)
        outs, ins, gouts = [qo, ko], [q, k], [gq, gk]
    grads = torch.autograd.grad(outs, ins, gouts)
    cos, sin = R.gather_cos_sin(tab.table, pos, tab.m1, tab.m2)
    for x, g, go in zip(ins, grads, gouts):
        x64 = x.detach().double().requires_grad_()
        y = R.rope_float64(x64, cos.double(), sin.double(), rot, si == "interleaved", so == "interleaved")
        (ref,) = torch.autograd.grad(y, x64, go.double())
        assert g.dtype == x.dtype and g.shape == x.shape
        assert rel(g, ref) < TOL[dtype], (rel(g, ref), dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("wdtype", ["same", torch.float32])
@pytest.mark.parametrize("d,rot", [(256, 64), (256, 0), (128, 128), (64, 32)])
@pytest.mark.parametrize("zc", [True, False])
@pytest.mark.parametrize("hq,hk", [(4, 1), (6, 1), (2, 2)])
def test_qk_norm_rope_grads_match_float64(dtype, wdtype, d, rot, zc, hq, hk):
    torch.manual_seed(d + rot + hq)
    wd = dtype if wdtype == "same" else wdtype
    tab = build_table(1e7, rot, 4096, DEV) if rot else None
    T = 77
    q, k = rnd(T, hq, d, dtype=dtype).requires_grad_(), rnd(T, hk, d, dtype=dtype).requires_grad_()
    wq = rnd(d, dtype=wd, scale=0.5).add_(0.0 if zc else 1.0).requires_grad_()
    wk = rnd(d, dtype=wd, scale=0.5).add_(0.0 if zc else 1.0).requires_grad_()
    pos = torch.randint(0, 4096, (T,), device=DEV)
    yq, yk = qk_norm_rope(q, k, wq, wk, pos, tab, 1e-6, zero_centered=zc)
    gq, gk = rnd(T, hq, d, dtype=dtype), rnd(T, hk, d, dtype=dtype)
    grads = torch.autograd.grad((yq, yk), (q, k, wq, wk), (gq, gk))
    ins = [t.detach().double().requires_grad_() for t in (q, k, wq, wk)]
    cos = sin = None
    if rot:
        cos, sin = (t.double() for t in R.gather_cos_sin(tab.table, pos, 0, 0))
    zq = R.qk_norm_rope_float64(ins[0], ins[2], cos, sin, 1e-6, rot, zc)
    zk = R.qk_norm_rope_float64(ins[1], ins[3], cos, sin, 1e-6, rot, zc)
    refs = torch.autograd.grad((zq, zk), ins, (gq.double(), gk.double()))
    for name, g, r, x in zip(("q", "k", "wq", "wk"), grads, refs, (q, k, wq, wk)):
        assert g.dtype == x.dtype and g.shape == x.shape, name
        bound = TOL[dtype] if name in ("q", "k") else max(TOL[dtype], TOL[wd]) * 2
        assert rel(g, r) < bound, (name, rel(g, r))


def test_partial_requires_grad_and_unused_outputs():
    tab = build_table(1e7, 64, 4096, DEV)
    T = 33
    q, k = rnd(T, 2, 256).requires_grad_(), rnd(T, 1, 256)
    wq, wk = rnd(256, scale=0.5), rnd(256, scale=0.5).requires_grad_()
    pos = torch.randint(0, 4096, (T,), device=DEV)
    yq, yk = qk_norm_rope(q, k, wq, wk, pos, tab, 1e-6)
    gq, gwk = torch.autograd.grad(yq.float().square().sum() + yk.float().sum(), (q, wk))
    assert gq.shape == q.shape and gwk.shape == wk.shape
    assert torch.isfinite(gq.float()).all() and torch.isfinite(gwk.float()).all()
    # only q's output used
    yq, _ = qk_norm_rope(q, k, wq, wk, pos, tab, 1e-6)
    (g2,) = torch.autograd.grad(yq.float().sum(), (q,))
    assert g2.abs().sum() > 0
    qo, ko = rope(q, k.requires_grad_(), pos, tab)
    (g3,) = torch.autograd.grad(ko.float().sum(), (k,))
    assert g3.abs().sum() > 0


def test_grad_connectivity_through_views():
    """Grads flow back into the packed projection and MLA views the ops read."""
    tab = build_table(1e6, 64, 4096, DEV)
    T, H = 50, 5
    Q = rnd(T, H, 256).requires_grad_()
    KV = rnd(T, 576).requires_grad_()
    pos = torch.randint(0, 4096, (T,), device=DEV)
    q_pe = Q[..., 192:]
    k_pe = KV[:, 512:].unsqueeze(1)
    qo, ko = rope(q_pe, k_pe, pos, tab, style="interleaved", out_style="neox")
    gQ, gKV = torch.autograd.grad(qo.float().sum() + ko.float().sum(), (Q, KV))
    assert gQ[..., :192].abs().sum() == 0 and gQ[..., 192:].abs().sum() > 0
    assert gKV[:, :512].abs().sum() == 0 and gKV[:, 512:].abs().sum() > 0


def test_fwd_bwd_capturable_in_cuda_graph():
    tab = build_table(1e7, 64, 4096, DEV)
    T = 64
    sq, sk = rnd(T, 4, 256).requires_grad_(), rnd(T, 1, 256).requires_grad_()
    wq, wk = torch.nn.Parameter(rnd(256, scale=0.5)), torch.nn.Parameter(rnd(256, scale=0.5))
    spos = torch.zeros(T, dtype=torch.long, device=DEV)
    sg = rnd(T, 4, 256)

    def step():
        yq, yk = qk_norm_rope(sq, sk, wq, wk, spos, tab, 1e-6)
        return torch.autograd.grad((yq, yk), (sq, sk, wq, wk), (sg, torch.ones_like(yk)))

    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = step()
    with torch.no_grad():
        sq.copy_(rnd(T, 4, 256))
        spos.copy_(torch.randint(0, 4096, (T,), device=DEV))
    graph.replay()
    ref = step()
    for a, b in zip(outs, ref):
        assert rel(a, b) < 1e-6


def test_loose_agreement_with_transformer_engine_rope():
    te_rope = pytest.importorskip("transformer_engine.pytorch.attention.rope")
    for inter, (hq, hk, D, rot, theta) in ((False, (32, 8, 128, 128, 640000.0)), (True, (20, 1, 64, 64, 1e6))):
        S, nseq = 256, 4
        T = S * nseq
        tab = build_table(theta, rot, S, DEV)
        q, k = rnd(T, hq, D).requires_grad_(), rnd(T, hk, D).requires_grad_()
        gq, gk = rnd(T, hq, D), rnd(T, hk, D)
        pos = torch.arange(T, device=DEV) % S
        style = "interleaved" if inter else "neox"
        a = rope(q, k, pos, tab, style=style)
        ga = torch.autograd.grad(a, (q, k), (gq, gk))
        cu = torch.arange(0, T + 1, S, device=DEV, dtype=torch.int32)
        f0 = torch.outer(torch.arange(S, device=DEV, dtype=torch.float64),
                         torch.tensor(R.inv_freq_double(theta, rot), device=DEV, dtype=torch.float64)).float()
        emb = (f0.repeat_interleave(2, -1) if inter else torch.cat((f0, f0), -1))[:, None, None, :]
        b = tuple(te_rope.apply_rotary_pos_emb(x, emb, tensor_format="thd", fused=True, cu_seqlens=cu,
                                               interleaved=inter) for x in (q, k))
        gb = torch.autograd.grad(b, (q, k), (gq, gk))
        for x, y in zip(a + ga, b + gb):
            assert rel(x, y) < 1e-2, rel(x, y)
