"""RoPE and fused q/k-norm + RoPE backward: gradients against float64, run-to-run bits, plan
equivalence and the no-float-atomics gate."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from isoexec.ops._testing import bitwise_equal
from isoexec.ops.attention.rope_sm100 import build_table, qk_norm_rope, rope
from isoexec.ops.attention.rope_sm100 import reference as R

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0), reason="requires SM100"
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


# Release geometries using the fused kernel: Qwen3.5 dense and MoE (d 256, partial rotary 0.25 ->
# rot 64, interleaved MRoPE sections (11, 11, 10)) at TP=1 and at sharded head counts.
QWEN35 = dict(d=256, rot=64, theta=1e7, sections=(11, 11, 10))


def _qwen_case(T, hq, hk, *, wdtype=torch.float32, seed=0, mrope=True):
    g = torch.Generator(device=DEV).manual_seed(seed + T + 31 * hq + hk)
    d = QWEN35["d"]
    tab = build_table(QWEN35["theta"], QWEN35["rot"], 65536, DEV, mrope_section=QWEN35["sections"] if mrope else None,
                      mrope_interleaved=mrope)
    # q and k are column views of one packed projection, as Attention hands them over.
    packed = torch.randn(T, (2 * hq + 2 * hk) * d, device=DEV, generator=g, dtype=BF)
    q = packed[:, :hq * d].view(T, hq, d)
    k = packed[:, hq * d:(hq + hk) * d].view(T, hk, d)
    wq = (torch.randn(d, device=DEV, generator=g) * 0.3).to(wdtype)
    wk = (torch.randn(d, device=DEV, generator=g) * 0.3).to(wdtype)
    pos = torch.randint(0, 65536, (3, T) if mrope else (T,), device=DEV, generator=g)
    gq = torch.randn(T, hq, d, device=DEV, generator=g, dtype=BF)
    gk = torch.randn(T, hk, d, device=DEV, generator=g, dtype=BF)
    return packed, q, k, wq, wk, pos, tab, gq, gk


def _grads(packed, q, k, wq, wk, pos, tab, gq, gk):
    P = packed.detach().requires_grad_()
    hq, hk, d = q.shape[1], k.shape[1], q.shape[2]
    qv, kv = P[:, :hq * d].view(-1, hq, d), P[:, hq * d:(hq + hk) * d].view(-1, hk, d)
    wq_, wk_ = wq.detach().requires_grad_(), wk.detach().requires_grad_()
    yq, yk = qk_norm_rope(qv, kv, wq_, wk_, pos, tab, 1e-6)
    gP, gwq, gwk = torch.autograd.grad((yq, yk), (P, wq_, wk_), (gq, gk))
    return yq.detach(), yk.detach(), gP[:, :hq * d].view_as(q), gP[:, hq * d:(hq + hk) * d].view_as(k), gwq, gwk


def _rel_chunked(a, b, rows=1024):
    num = den = 0.0
    for s in range(0, a.shape[0], rows):
        x, y = a[s:s + rows].double(), b[s:s + rows].double()
        num += float((x - y).abs().sum())
        den += float(y.abs().sum())
    return num / max(den, 1e-300)


def _float64_grads(q, k, wq, wk, pos, tab, gq, gk, chunk=512):
    """Float64 autograd of the composition, token-chunked so the reference stays small."""
    cos, sin = (t.double() for t in R.gather_cos_sin(tab.table, pos, tab.m1, tab.m2))
    dq, dk = [], []
    dwq = torch.zeros(wq.shape, dtype=torch.float64, device=DEV)
    dwk = torch.zeros_like(dwq)
    for s in range(0, q.shape[0], chunk):
        sl = slice(s, s + chunk)
        ins = [q[sl].double().requires_grad_(), k[sl].double().requires_grad_(),
               wq.double().requires_grad_(), wk.double().requires_grad_()]
        zq = R.qk_norm_rope_float64(ins[0], ins[2], cos[sl], sin[sl], 1e-6, tab.rotary_dim, True)
        zk = R.qk_norm_rope_float64(ins[1], ins[3], cos[sl], sin[sl], 1e-6, tab.rotary_dim, True)
        a, b, c, e = torch.autograd.grad((zq, zk), ins, (gq[sl].double(), gk[sl].double()))
        dq.append(a.float())
        dk.append(b.float())
        dwq += c
        dwk += e
    return torch.cat(dq), torch.cat(dk), dwq, dwk


@pytest.mark.parametrize("T,hq,hk,wdtype,mrope", [
    (4096, 8, 2, torch.float32, True), (4096, 16, 2, torch.float32, True), (4096, 4, 1, torch.float32, True),
    (4096, 8, 1, torch.float32, True), (16384, 16, 4, torch.float32, True), (16384, 32, 2, torch.float32, True),
    (16384, 2, 1, torch.float32, True), (8192, 16, 4, torch.bfloat16, False), (4096, 4, 1, torch.bfloat16, True),
])
def test_qwen35_training_shapes_match_float64_and_repeat_bitwise(T, hq, hk, wdtype, mrope):
    """Qwen3.5 rows at trainer token counts: gradients against float64, then a second run bit-for-bit."""
    case = _qwen_case(T, hq, hk, wdtype=wdtype, mrope=mrope)
    packed, q, k, wq, wk, pos, tab, gq, gk = case
    got = _grads(*case)
    ref = _float64_grads(q, k, wq, wk, pos, tab, gq, gk)
    for name, a, b in zip(("dq", "dk", "dwq", "dwk"), got[2:], ref):
        assert a.dtype == (BF if name in ("dq", "dk") else wq.dtype)
        err = _rel_chunked(a, b)
        assert err < (TOL[BF] if name in ("dq", "dk") or wdtype == BF else 1e-5), (name, err)
    # The forward an autograd-recording trainer runs is the engine forward, bit for bit.
    with torch.no_grad():
        eq, ek = qk_norm_rope(q, k, wq, wk, pos, tab, 1e-6)
    assert bitwise_equal(got[0], eq) and bitwise_equal(got[1], ek)
    again = _grads(*case)
    for a, b in zip(got, again):
        assert bitwise_equal(a, b)


@pytest.mark.parametrize("wdtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("d,rot,mrope", [(256, 64, True), (256, 64, False), (256, 0, False),
                                         (128, 128, False), (64, 32, False)])
@pytest.mark.parametrize("T", [1, 77, 1003])
def test_qk_backward_plans_agree_bitwise(wdtype, d, rot, mrope, T):
    """Operands off the vector alignment take the grid-of-rows backward; aligned ones the warp-row
    one. Both are the same recipe: every dx and dw bit agrees, short last token groups included."""
    g = torch.Generator(device=DEV).manual_seed(T + d + rot)
    hq, hk = 5, 2
    tab = None
    pos = torch.randint(0, 4096, (3, T) if mrope else (T,), device=DEV, generator=g)
    if rot:
        tab = build_table(1e7, rot, 4096, DEV, mrope_section=QWEN35["sections"] if mrope else None,
                          mrope_interleaved=mrope)
    q = torch.randn(T, hq, d, device=DEV, generator=g).to(BF)
    k = torch.randn(T, hk, d, device=DEV, generator=g).to(BF)
    wq = (torch.randn(d, device=DEV, generator=g) * 0.3).to(wdtype)
    wk = (torch.randn(d, device=DEV, generator=g) * 0.3).to(wdtype)
    gq = torch.randn(T, hq, d, device=DEV, generator=g).to(BF)
    gk = torch.randn(T, hk, d, device=DEV, generator=g).to(BF)

    def odd(x):  # same values on a 4-byte- but not 16-byte-aligned storage offset
        buf = torch.empty(x.numel() + 2, dtype=x.dtype, device=DEV)
        y = buf[2:].view(x.shape)
        y.copy_(x)
        return y

    def run(q, k, gq, gk):
        q, k = q.detach().requires_grad_(), k.detach().requires_grad_()
        wq_, wk_ = wq.detach().requires_grad_(), wk.detach().requires_grad_()
        out = qk_norm_rope(q, k, wq_, wk_, pos, tab, 1e-6)
        return (*out, *torch.autograd.grad(out, (q, k, wq_, wk_), (gq, gk)))

    a = run(q, k, gq, gk)
    b = run(odd(q), odd(k), odd(gq), odd(gk))
    for x, y in zip(a, b):
        assert bitwise_equal(x.contiguous(), y.contiguous())


def test_qk_backward_empty_and_grouping_refusal():
    from isoexec.ops.attention.rope_sm100._extension import extension

    tab = build_table(1e7, 64, 64, DEV)
    q = torch.empty(0, 2, 256, device=DEV, dtype=BF).requires_grad_()
    k = torch.empty(0, 1, 256, device=DEV, dtype=BF).requires_grad_()
    wq, wk = (torch.randn(256, device=DEV).requires_grad_() for _ in range(2))
    pos = torch.empty(0, dtype=torch.long, device=DEV)
    yq, yk = qk_norm_rope(q, k, wq, wk, pos, tab, 1e-6)
    gq, gk, gwq, gwk = torch.autograd.grad((yq, yk), (q, k, wq, wk), (yq, yk))
    assert gq.shape == q.shape and not gwq.count_nonzero() and not gwk.count_nonzero()
    q, k = torch.randn(17, 2, 256, device=DEV, dtype=BF), torch.randn(17, 1, 256, device=DEV, dtype=BF)
    pos = torch.arange(17, device=DEV)
    _, _, inv = extension().qknorm_rope_forward(q, k, wq, wk, pos, tab.table, 0, 0, 1e-6, True, True, False)
    with pytest.raises(RuntimeError, match="requires gshift=4"):
        extension().qknorm_rope_backward(q, k, q, k, wq, wk, inv, pos, tab.table, 0, 0, True, True, 0)


def test_kernels_have_no_float_atomics():
    """Gate: no ATOM/RED instruction in any kernel of the extension (the backward reductions are
    plain stores into fixed workspaces plus fixed trees)."""
    if shutil.which("cuobjdump") is None:
        pytest.skip("cuobjdump not on PATH")
    from isoexec.ops.attention.rope_sm100._extension import extension

    sass = subprocess.run(["cuobjdump", "-sass", str(Path(extension().__file__))], capture_output=True, text=True,
                          check=True).stdout
    funcs, cur = {}, None
    for line in sass.splitlines():
        m = re.search(r"Function : (\S+)", line)
        if m:
            cur = m.group(1)
            funcs[cur] = []
        elif cur:
            funcs[cur].append(line)
    assert any("qknorm_bwd" in name for name in funcs) and any("qknorm_dw_fold" in name for name in funcs)
    bad = {name: [ln for ln in lines if re.search(r"\b(ATOM|ATOMG|ATOMS|RED|REDG)(\.|\s)", ln)]
           for name, lines in funcs.items()}
    assert not {name: lines for name, lines in bad.items() if lines}
