"""GPU battery for the DeepSeek-V4.1 shifted mHC (params / collapse / expand / fused split; forward and backward)."""

from __future__ import annotations

import pytest
import torch

from isoexec.core.arch import live_device_arch
from isoexec.ops.residual.deepseek_mhc_sm100 import ops
from isoexec.ops.residual.deepseek_mhc_sm100 import reference as R
from isoexec.ops.residual.deepseek_mhc_sm100.tests._data import EPS, ITERS, KW, RMS_EPS, H, make, probs, rnd

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or live_device_arch() != "sm100",
                                reason="the sm100 mHC kernels run on a Blackwell device only")
TS = [1, 7, 16, 33, 256, 513, 1100]


def params(x, w, scale, base):
    return ops.mhc_params(x, w, scale, base, **KW)


def oracle(x, w, scale, base):
    return R.params_reference(x, w, scale, base, rms_eps=RMS_EPS, sinkhorn_eps=EPS, iterations=ITERS)


# ------------------------------------------------------------------------------------------ parameters, forward
@pytest.mark.parametrize("T", TS)
def test_params_accuracy_vs_float64_oracle(T):
    x, w, scale, base = make(T)
    with torch.no_grad():
        pre, post, comb = params(x, w, scale, base)
    rp, rpost, rcomb = oracle(x, w, scale, base)
    d = lambda a, b: (a.double() - b).abs().max().item()
    assert d(pre, rp) < 3e-6 and d(post, rpost) < 3e-6 and d(comb, rcomb) < 3e-6   # fp32 evaluation of a 20480-term projection (<= ~25 ulp at 2.0)


@pytest.mark.parametrize("T", [64, 700])
def test_params_vs_eager_module_equations(T):
    x, w, scale, base = make(T)
    with torch.no_grad():
        a = R.eager_params(x, w, scale, base, rms_eps=RMS_EPS, sinkhorn_eps=EPS, iterations=ITERS)
        b = params(x, w, scale, base)
    for u, v in zip(a, b):
        assert (u - v).abs().max().item() < 4e-6        # fp32 evaluation of the same equation


def test_params_repeat_batch_invariance_and_padding():
    x, w, scale, base = make(1024)
    with torch.no_grad():
        full = params(x, w, scale, base)
        for _ in range(3):
            assert all(torch.equal(a, b) for a, b in zip(full, params(x, w, scale, base)))
        for i in (0, 1, 17, 1023):
            one = params(x[i:i + 1].contiguous(), w, scale, base)
            assert all(torch.equal(a[i:i + 1], b) for a, b in zip(full, one))
        for lo, hi in ((0, 37), (300, 811)):
            sub = params(x[lo:hi].contiguous(), w, scale, base)
            assert all(torch.equal(a[lo:hi], b) for a, b in zip(full, sub))
        # [batch, sequence] leading shape
        sh = params(x.view(8, 128, 4, H), w, scale, base)
        assert all(torch.equal(a.view(b.shape), b) for a, b in zip(full, sh))


def test_params_bf16_fn_is_the_same_program_on_exact_inputs():
    x, w, scale, base = make(200)
    wb = w.bfloat16()
    with torch.no_grad():
        a = params(x, wb, scale, base)
        b = params(x, wb.float(), scale, base)          # bf16 values are exactly representable: pieces (hi, 0, 0)
    assert all(torch.equal(u, v) for u, v in zip(a, b))


def test_params_extremes_and_refusals():
    x, w, scale, base = make(32, x_scale=500.0)
    with torch.no_grad():
        pre, post, comb = params(x, w, scale, base)
    assert torch.isfinite(comb).all()
    rp, rpost, rcomb = oracle(x, w, scale, base)
    assert (comb.double() - rcomb).abs().max() < 3e-6
    z = torch.zeros_like(x)                              # all-zero residual: rms floor 1e-20 keeps mixes finite
    with torch.no_grad():
        zp, zpost, zcomb = params(z, w, scale, base)
    assert torch.isfinite(zcomb).all() and torch.allclose(zpost.double(), 2 * torch.sigmoid(base[4:8].double()).expand(32, 4), atol=1e-6)
    with pytest.raises(ValueError):
        params(x[..., :4096].contiguous(), w[:, :4 * 4096].contiguous(), scale, base)
    with pytest.raises(ValueError):
        params(x.float(), w, scale, base)


# ------------------------------------------------------------------------------------------ parameters, backward
def grads_oracle(x, w, scale, base, dpre, dpost, dcomb, extra=None):
    xs, ws, ss, bs = (t.double().requires_grad_(True) for t in (x, w, scale, base))
    pre, post, comb = R.params_reference(xs, ws, ss, bs, rms_eps=RMS_EPS, sinkhorn_eps=EPS, iterations=ITERS)
    loss = (pre * dpre.double()).sum() + (post * dpost.double()).sum() + (comb * dcomb.double()).sum()
    loss.backward()
    return xs.grad, ws.grad, ss.grad, bs.grad


@pytest.mark.parametrize("T", [1, 16, 100, 513, 1100])
def test_params_backward_vs_float64_autograd(T):
    x, w, scale, base = make(T)
    xs, ws, ss, bs = (t.clone().requires_grad_(True) for t in (x, w, scale, base))
    pre, post, comb = params(xs, ws, ss, bs)
    dpre, dpost, dcomb = rnd(T, 4, seed=1), rnd(T, 4, seed=2), rnd(T, 4, 4, seed=3)
    torch.autograd.backward([pre, post, comb], [dpre, dpost, dcomb])
    gx, gw, gs, gb = grads_oracle(x, w, scale, base, dpre, dpost, dcomb)
    rel = lambda a, b: ((a.double() - b).abs().max() / b.abs().max().clamp_min(1e-30)).item()
    assert xs.grad.dtype == torch.bfloat16 and ws.grad.dtype == torch.float32
    assert rel(xs.grad, gx) < 2 ** -8 and rel(ws.grad, gw) < 2e-5
    assert rel(ss.grad, gs) < 2e-5 and rel(bs.grad, gb) < 2e-5


def test_params_backward_repeat_and_row_invariance():
    T = 1100
    x, w, scale, base = make(T)
    pre, post, comb = (t for t in params(x, w, scale, base))
    d = (rnd(T, 4, seed=1), rnd(T, 4, seed=2), rnd(T, 4, 4, seed=3))

    def run(xx, dd):
        xs, ws, ss, bs = (t.clone().requires_grad_(True) for t in (xx, w, scale, base))
        o = params(xs, ws, ss, bs)
        torch.autograd.backward(list(o), list(dd))
        return xs.grad, ws.grad, ss.grad, bs.grad

    a, b = run(x, d), run(x, d)
    assert all(torch.equal(u, v) for u, v in zip(a, b))
    sub = run(x[200:231].contiguous(), tuple(t[200:231].contiguous() for t in d))
    assert torch.equal(sub[0], a[0][200:231])        # dx of a row does not depend on the batch


# ------------------------------------------------------------------------------------------ collapse
@pytest.mark.parametrize("T", [0, 1, 33, 1000])
def test_collapse_bitwise_declared_and_f64(T):
    x, *_ = make(max(T, 1))
    x = x[:T]
    pre = torch.softmax(rnd(T, 4, seed=4), -1)
    y = ops.collapse(x, pre)
    assert torch.equal(y, R.collapse_declared(x, pre))
    if T:
        ex = R.collapse_reference(x, pre)
        assert ((y.double() - ex).abs() <= ex.abs() * 2 ** -8 + 4e-7 * x.double().abs().sum(-2)).all()
    ref = R.eager_collapse(x, pre) if T else None
    if T:
        assert ((y.double() - ref.double()).abs() <= ref.double().abs() * 2 ** -8 + 1e-6).all()


def test_collapse_backward():
    T = 300
    x, *_ = make(T)
    pre = torch.softmax(rnd(T, 4, seed=4), -1).requires_grad_(True)
    xs = x.clone().requires_grad_(True)
    dcol = rnd(T, H, seed=5, dtype=torch.bfloat16)
    ops.collapse(xs, pre).backward(dcol)
    xd, pd = x.double().requires_grad_(True), pre.detach().double().requires_grad_(True)
    (R.collapse_reference(xd, pd) * dcol.double()).sum().backward()
    assert torch.equal(xs.grad, (pre.detach().unsqueeze(-1) * dcol.float().unsqueeze(-2)).to(torch.bfloat16))
    assert ((pre.grad.double() - pd.grad).abs() / pd.grad.abs().max()).max() < 1e-5


# ------------------------------------------------------------------------------------------ expand
@pytest.mark.parametrize("T", [0, 1, 33, 600])
def test_expand_bitwise_declared_and_f64(T):
    x, *_ = make(max(T, 1))
    res = x[:T]
    out = rnd(T, H, seed=6, dtype=torch.bfloat16)
    post, comb = probs(T, 7)
    y = ops.expand(out, res, post, comb)
    assert torch.equal(y, R.expand_declared(out, res, post, comb))
    if T:
        ex = R.expand_reference(out, res, post, comb)
        mag = (post.double().unsqueeze(-1) * out.double().unsqueeze(-2)).abs() + (
            comb.double().unsqueeze(-1) * res.double().unsqueeze(-2)).abs().sum(-3)
        assert ((y.double() - ex).abs() <= ex.abs() * 2 ** -8 + 4e-7 * mag).all()
        e = R.eager_expand(out, res, post, comb)
        assert ((y.double() - e.double()).abs() <= e.double().abs() * 2 ** -8 + 1e-6 * mag).all()


@pytest.mark.parametrize("T", [1, 40, 600])
def test_expand_backward_vs_float64(T):
    x, *_ = make(T)
    res = x.clone().requires_grad_(True)
    out = rnd(T, H, seed=6, dtype=torch.bfloat16).requires_grad_(True)
    post, comb = (t.requires_grad_(True) for t in probs(T, 7))
    g = rnd(T, 4, H, seed=8, dtype=torch.bfloat16, sc=0.5)
    ops.expand(out, res, post, comb).backward(g)
    ts = [t.detach().double().requires_grad_(True) for t in (out, res, post, comb)]
    (R.expand_reference(*ts) * g.double()).sum().backward()
    rel = lambda a, b: ((a.double() - b).abs().max() / b.abs().max()).item()
    assert rel(out.grad, ts[0].grad) < 2 ** -8 and rel(res.grad, ts[1].grad) < 2 ** -8
    assert rel(post.grad, ts[2].grad) < 2e-5 and rel(comb.grad, ts[3].grad) < 2e-5
    assert post.grad.dtype == torch.float32


def test_expand_backward_repeat_and_row_invariance():
    T = 500
    x, *_ = make(T)
    out = rnd(T, H, seed=6, dtype=torch.bfloat16)
    post, comb = probs(T, 7)
    g = rnd(T, 4, H, seed=8, dtype=torch.bfloat16, sc=0.5)
    from isoexec.ops.residual.mhc_sm100 import ops as G
    a = G.merge_backward(g, x, out, post, comb, cast=False)
    assert all(torch.equal(u, v) for u, v in zip(a, G.merge_backward(g, x, out, post, comb, cast=False)))
    one = G.merge_backward(g[7:8].contiguous(), x[7:8].contiguous(), out[7:8].contiguous(), post[7:8].contiguous(),
                           comb[7:8].contiguous(), cast=False)
    assert all(torch.equal(u, v[7:8]) for u, v in zip(one, a))


# ------------------------------------------------------------------------------------------ fused shifted split
def test_fused_shifted_split_equals_separate_ops_bitwise():
    T = 640
    x, w, scale, base = make(T)
    sp = torch.softmax(rnd(T, 4, seed=9), -1)
    with torch.no_grad():
        pre, post, comb, col = ops.mhc_split_shifted(x, w, scale, base, sp, **KW)
        p2, q2, c2 = params(x, w, scale, base)
    assert torch.equal(pre, p2) and torch.equal(post, q2) and torch.equal(comb, c2)
    assert torch.equal(col, ops.collapse(x, sp))


def test_fused_shifted_split_backward_vs_float64():
    T = 300
    x, w, scale, base = make(T)
    sp = torch.softmax(rnd(T, 4, seed=9), -1)
    xs, ws, ss, bs, sps = (t.clone().requires_grad_(True) for t in (x, w, scale, base, sp))
    pre, post, comb, col = ops.mhc_split_shifted(xs, ws, ss, bs, sps, **KW)
    dpre, dpost, dcomb = rnd(T, 4, seed=1), rnd(T, 4, seed=2), rnd(T, 4, 4, seed=3)
    dcol = rnd(T, H, seed=5, dtype=torch.bfloat16)
    torch.autograd.backward([pre, post, comb, col], [dpre, dpost, dcomb, dcol])
    xd, wd, sd, bd, spd = (t.double().requires_grad_(True) for t in (x, w, scale, base, sp))
    p, q, c = R.params_reference(xd, wd, sd, bd, rms_eps=RMS_EPS, sinkhorn_eps=EPS, iterations=ITERS)
    (p * dpre.double()).sum().add((q * dpost.double()).sum()).add((c * dcomb.double()).sum()).add(
        (R.collapse_reference(xd, spd) * dcol.double()).sum()).backward()
    rel = lambda a, b: ((a.double() - b).abs().max() / b.abs().max().clamp_min(1e-30)).item()
    assert rel(xs.grad, xd.grad) < 2 ** -8 and rel(ws.grad, wd.grad) < 2e-5
    assert rel(ss.grad, sd.grad) < 2e-5 and rel(bs.grad, bd.grad) < 2e-5 and rel(sps.grad, spd.grad) < 1e-5


# ------------------------------------------------------------------------------------------ composition (state across sublayers)
def chain(x0, w1, w2, scale, base, pre0, T_slices):
    """Two sublayers with the shifted state: attention input = collapse(x, pre_prev); ffn input = collapse(x1, ap)."""
    out = []
    for sl in T_slices:
        x = x0[sl]
        ap, post1, comb1 = params(x, w1, scale, base)
        a_in = ops.collapse(x, pre0[sl])
        attn = (a_in.float() * 0.5).bfloat16()
        x1 = ops.expand(attn, x, post1, comb1)
        fp, post2, comb2 = params(x1, w2, scale, base)
        f_in = ops.collapse(x1, ap)
        ffn = (f_in.float() * -0.25).bfloat16()
        x2 = ops.expand(ffn, x1, post2, comb2)
        out.append((a_in, x1, f_in, x2, fp))
    return out


def test_prefill_vs_decode_composition_is_bitwise():
    T = 192
    x0, w1, scale, base = make(T)
    w2 = make(T, seed=5)[1]
    pre0 = torch.zeros(T, 4, device="cuda"); pre0[:, 0] = 1
    with torch.no_grad():
        full = chain(x0, w1, w2, scale, base, pre0, [slice(0, T)])[0]
        for t in (0, 77, 191):                        # one token at a time (decode), state carried per token
            one = chain(x0, w1, w2, scale, base, pre0, [slice(t, t + 1)])[0]
            for a, b in zip(full, one):
                assert torch.equal(a[t:t + 1], b)
        halves = chain(x0, w1, w2, scale, base, pre0, [slice(0, 100), slice(100, T)])
        for k in range(5):
            assert torch.equal(torch.cat([halves[0][k], halves[1][k]]), full[k])


def test_two_sublayer_chain_vs_float64_reference():
    T = 96
    x0, w1, scale, base = make(T)
    w2 = make(T, seed=5)[1]
    pre0 = torch.zeros(T, 4, device="cuda"); pre0[:, 0] = 1
    with torch.no_grad():
        a_in, x1, f_in, x2, fp = chain(x0, w1, w2, scale, base, pre0, [slice(0, T)])[0]
    ap, post1, comb1 = R.params_reference(x0, w1, scale, base, rms_eps=RMS_EPS, sinkhorn_eps=EPS, iterations=ITERS)
    ra = R.collapse_reference(x0, pre0)
    rx1 = R.expand_reference(0.5 * ra, x0, post1, comb1)
    fp_, post2, comb2 = R.params_reference(rx1, w2, scale, base, rms_eps=RMS_EPS, sinkhorn_eps=EPS, iterations=ITERS)
    rf = R.collapse_reference(rx1, ap)
    rx2 = R.expand_reference(-0.25 * rf, rx1, post2, comb2)
    # the native chain re-rounds to bf16 after every op: bound the drift by a few bf16 roundings of the magnitudes
    for got, want in ((a_in, ra), (x1, rx1), (f_in, rf), (x2, rx2)):
        assert ((got.double() - want).abs().max() / want.abs().max()).item() < 2 ** -5
    assert (fp.double() - fp_).abs().max() < 5e-3


def test_independent_of_cta_schedule_groups():
    from isoexec.ops.residual.mhc_sm100 import ops as G
    x, w, scale, base = make(900)
    ref = G.split_forward(x, w, scale, base, rms_eps=RMS_EPS, eps=EPS, iters=ITERS, collapse=None, stats=True)
    for groups in (1, 2, 5, 13):
        got = G.split_forward(x, w, scale, base, rms_eps=RMS_EPS, eps=EPS, iters=ITERS, collapse=None, stats=True,
                              groups=groups)
        assert all(torch.equal(a, b) for a, b in zip(ref[:3] + ref[4], got[:3] + got[4])), groups
    proj, rstd = ref[4]
    d = (rnd(900, 4, seed=1), rnd(900, 4, seed=2), rnd(900, 16, seed=3).view(900, 4, 4))
    a = G.split_backward(x, w, scale, base, proj, rstd, d[1], d[2], eps=EPS, iters=ITERS, dpre=d[0])
    for groups in (1, 2):
        b = G.split_backward(x, w, scale, base, proj, rstd, d[1], d[2], eps=EPS, iters=ITERS, dpre=d[0], groups=groups)
        assert all(torch.equal(u, v) for u, v in zip(a[:4], b[:4]))


def test_train_and_rollout_paths_are_the_same_bits():
    x, w, scale, base = make(300)
    with torch.no_grad():
        r = params(x, w, scale, base)
    leaves = [t.clone().requires_grad_(True) for t in (x, w, scale, base)]
    t_ = params(*leaves)
    assert all(torch.equal(a, b) for a, b in zip(r, t_))
    out = rnd(300, H, seed=6, dtype=torch.bfloat16)
    post, comb = probs(300, 7)
    with torch.no_grad():
        e0 = ops.expand(out, x, post, comb)
    assert torch.equal(e0, ops.expand(out.clone().requires_grad_(True), x, post, comb))


@pytest.mark.parametrize("iters", [1, 3, 32])
def test_backward_iteration_counts_vs_float64(iters):
    T = 50
    x, w, scale, base = make(T)
    kw = dict(rms_eps=RMS_EPS, sinkhorn_eps=EPS, iterations=iters)
    xs, ws, ss, bs = (t.clone().requires_grad_(True) for t in (x, w, scale, base))
    o = ops.mhc_params(xs, ws, ss, bs, **kw)
    d = (rnd(T, 4, seed=1), rnd(T, 4, seed=2), rnd(T, 4, 4, seed=3))
    torch.autograd.backward(list(o), list(d))
    xd, wd, sd, bd = (t.double().requires_grad_(True) for t in (x, w, scale, base))
    p, q, c = R.params_reference(xd, wd, sd, bd, rms_eps=RMS_EPS, sinkhorn_eps=EPS, iterations=iters)
    ((p * d[0].double()).sum() + (q * d[1].double()).sum() + (c * d[2].double()).sum()).backward()
    rel = lambda a, b: ((a.double() - b).abs().max() / b.abs().max().clamp_min(1e-30)).item()
    assert rel(xs.grad, xd.grad) < 2 ** -8 and rel(ws.grad, wd.grad) < 2e-5
    assert rel(ss.grad, sd.grad) < 2e-5 and rel(bs.grad, bd.grad) < 2e-5


def test_cast_boundaries_are_witnessed():
    # DeepSeek expand keeps post/comb fp32 and rounds once: it differs from the GLM (bf16-cast) merge on some elements
    from isoexec.ops.residual.mhc_sm100 import ops as G
    x, *_ = make(256)
    out = rnd(256, H, seed=6, dtype=torch.bfloat16)
    post, comb = probs(256, 7)
    ds = ops.expand(out, x, post, comb)
    glm = G.merge_forward(x, out, post, comb, cast=True)
    assert torch.equal(ds, R.expand_declared(out, x, post, comb)) and not torch.equal(ds, glm)
