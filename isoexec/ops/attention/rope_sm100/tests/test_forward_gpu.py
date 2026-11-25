"""rope_sm100 v1 forward gates: recipe identity and invariance, all on integer bit views.

Run: CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=. $PY -m pytest -q isoexec/ops/attention/rope_sm100/tests/test_forward_gpu.py
"""

from __future__ import annotations

import math

import pytest
import torch

from isoexec.ops._testing import assert_hazard_exercised, bitwise_equal
from isoexec.ops.attention.rope_sm100 import build_table, qk_norm_rope, rope
from isoexec.ops.attention.rope_sm100 import reference as R

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0), reason="requires SM100"
)

DEV = "cuda"
BF = torch.bfloat16
DTYPES = (torch.bfloat16, torch.float16, torch.float32)
QWEN = dict(theta=1e7, rot=64, d=256, sections=(11, 11, 10), interleaved=True, max_pos=262144)
MIMO = dict(theta=640000.0, rot=128, d=128, max_pos=32768)
GLM = dict(theta=1e6, rot=64, max_pos=202752)

_TABLES = {}


def table(theta, rot, max_pos, sections=None, interleaved=False):
    key = (theta, rot, max_pos, sections, interleaved)
    if key not in _TABLES:
        _TABLES[key] = build_table(theta, rot, max_pos, DEV, mrope_section=sections, mrope_interleaved=interleaved)
    return _TABLES[key]


def rnd(*shape, dtype=BF, scale=1.0, gen=None):
    return (torch.randn(*shape, device=DEV, generator=gen) * scale).to(dtype)


def pos_of(T, hi, gen=None):
    return torch.randint(0, hi, (T,), device=DEV, generator=gen)


# table
@pytest.mark.parametrize("theta,rot,max_pos", [(1e7, 64, 262144), (640000.0, 128, 32768), (1e6, 64, 202752)])
def test_table_rows_match_host_libm(theta, rot, max_pos):
    tab = table(theta, rot, max_pos)
    g = torch.Generator().manual_seed(rot)
    ps = sorted(set(p for p in [0, 1, 2, 3, 4095, 4096, 65535, max_pos - 2, max_pos - 1]
                    + torch.randint(0, max_pos, (400,), generator=g).tolist() if p < max_pos))
    host = R.table_rows_host(theta, rot, ps).to(DEV)
    dev = tab.table[torch.tensor(ps, device=DEV)]
    diff = (host.view(torch.int32) != dev.view(torch.int32)).sum().item()
    assert diff == 0, f"{diff} table entries differ from glibc double cos/sin rounded to fp32"


def test_table_build_is_repeatable_and_prefix_stable():
    a = build_table(1e7, 64, 70000, DEV).table
    b = build_table(1e7, 64, 262144, DEV).table[:70000]
    assert bitwise_equal(a, b)


# recipe identity
LAYOUTS = [("neox", "neox"), ("interleaved", "interleaved"), ("interleaved", "neox"), ("neox", "interleaved")]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("hq,hk,D,rot", [(32, 8, 128, 128), (20, 1, 64, 64), (2, 1, 256, 64), (5, 3, 256, 256),
                                         (4, 1, 128, 32), (1, 0, 64, 64), (40, 40, 16, 16)])
@pytest.mark.parametrize("layout", LAYOUTS)
def test_rope_matches_recipe(dtype, hq, hk, D, rot, layout):
    g = torch.Generator(device=DEV).manual_seed(hq * 1000 + D)
    tab = table(1e6, rot, 70000)
    T = 67
    q = rnd(T, hq, D, dtype=dtype, gen=g)
    k = rnd(T, hk, D, dtype=dtype, gen=g) if hk else None
    pos = pos_of(T, 70000, g)
    si, so = layout
    qo, ko = rope(q, k, pos, tab, style=si, out_style=so)
    kw = dict(in_interleaved=si == "interleaved", out_interleaved=so == "interleaved")
    assert bitwise_equal(qo, R.rope_recipe(q, pos, tab.table, **kw))
    if hk:
        assert bitwise_equal(ko, R.rope_recipe(k, pos, tab.table, **kw))
    else:
        assert ko is None


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("wdtype", ["same", torch.float32])
@pytest.mark.parametrize("d,rot", [(256, 64), (256, 256), (256, 0), (128, 32), (128, 128), (64, 64), (64, 16)])
@pytest.mark.parametrize("zc", [True, False])
def test_qk_norm_rope_matches_recipe(dtype, wdtype, d, rot, zc):
    g = torch.Generator(device=DEV).manual_seed(d + rot + zc)
    wd = dtype if wdtype == "same" else wdtype
    tab = table(1e7, rot, 70000) if rot else None
    for hq, hk in ((2, 1), (6, 1), (16, 4), (3, 3)):
        T = 45
        q, k = rnd(T, hq, d, dtype=dtype, gen=g), rnd(T, hk, d, dtype=dtype, gen=g)
        wq, wk = rnd(d, dtype=wd, scale=0.5, gen=g), rnd(d, dtype=wd, scale=0.5, gen=g)
        pos = pos_of(T, 70000, g)
        yq, yk = qk_norm_rope(q, k, wq, wk, pos, tab, 1e-6, zero_centered=zc)
        t = None if tab is None else tab.table
        rq, _ = R.qk_norm_rope_recipe(q, wq, pos, t, 1e-6, zero_centered=zc)
        rk, _ = R.qk_norm_rope_recipe(k, wk, pos, t, 1e-6, zero_centered=zc)
        assert bitwise_equal(yq, rq) and bitwise_equal(yk, rk), (hq, hk)


def test_recipe_is_not_the_old_recipe():
    """Version bump witness: rope_sm100 v1 bits differ from rope.apply eager (bf16 three rounds)."""
    from isoexec.core.rope import rotary_freqs
    # The vendored megatron reference is architecture-neutral and lives once, in rope_sm90.
    from isoexec.ops.attention.rope_sm90.rope_reference import rope_apply_eager

    tab = table(1e7, 64, 70000)
    T = 512
    q, k = rnd(T, 2, 256), rnd(T, 1, 256)
    pos = pos_of(T, 70000)
    old = rope_apply_eager(q, k, rotary_freqs(pos, 1e7, 64), style="neox", rotary_dim=64)
    new = rope(q, k, pos, tab)
    assert not bitwise_equal(old[0], new[0])


# invariance
def _pack(rows, extra_T, D_shape, gen, poison):
    """Place `rows` (list of (x_row [H,D], pos)) at random offsets of a pack with random/poisoned mates."""
    T = len(rows) + extra_T
    H, D = D_shape
    x = rnd(T, H, D, gen=gen)
    if poison:
        x[torch.rand(T, device=DEV, generator=gen) < 0.5] = float("nan")
    pos = pos_of(T, 70000, gen)
    slots = torch.randperm(T, device=DEV, generator=gen)[: len(rows)]
    for (xr, p), s in zip(rows, slots.tolist()):
        x[s] = xr
        pos[s] = p
    return x, pos, slots.tolist()


@pytest.mark.parametrize("site", ["mimo", "glm_mla", "qwen_qk"])
def test_batch_composition_and_padding_invariance(site):
    g = torch.Generator(device=DEV).manual_seed(11)
    n = 5
    if site == "mimo":
        tab, H, D, style = table(640000.0, 128, 70000), 32, 128, ("neox", "neox")
    elif site == "glm_mla":
        tab, H, D, style = table(1e6, 64, 70000), 20, 64, ("interleaved", "neox")
    else:
        tab, H, D, style = table(1e7, 64, 70000), 2, 256, None
    xs = [(rnd(H, D, gen=g), int(p)) for p in pos_of(n, 70000, g).tolist()]
    ks = [rnd(1, D, gen=g) for _ in range(n)]
    wq, wk = rnd(D, scale=0.5, gen=g), rnd(D, scale=0.5, gen=g)

    def run(x, k, pos):
        if style is None:
            return qk_norm_rope(x, k, wq, wk, pos, tab, 1e-6)
        return rope(x, k, pos, tab, style=style[0], out_style=style[1])

    singles = [run(xr[None], ks[i][None], torch.tensor([p], device=DEV)) for i, (xr, p) in enumerate(xs)]
    nulls = 0
    for extra in (0, 1, 6, 58, 507, 4091):
        for poison in (False, True):
            x, pos, slots = _pack(xs, extra, (H, D), g, poison)
            k = rnd(x.shape[0], 1, D, gen=g)
            if poison:
                k[torch.rand(x.shape[0], device=DEV, generator=g) < 0.5] = float("inf")
            for i, s in enumerate(slots):
                k[s] = ks[i]
            nulls += int(torch.isnan(x).any(dim=(1, 2)).sum())
            yq, yk = run(x, k, pos)
            for i, s in enumerate(slots):
                assert bitwise_equal(yq[s], singles[i][0][0]), (extra, poison, i)
                assert bitwise_equal(yk[s], singles[i][1][0]), (extra, poison, i)
    assert_hazard_exercised("null_lane", nulls)


@pytest.mark.parametrize("kind", ["rope", "qk"])
def test_head_count_invariance_across_tp(kind):
    """Engine TP8 / trainer TP4 / TP1 ranks see different head slices of the same logical heads."""
    g = torch.Generator(device=DEV).manual_seed(5)
    T, HQ, HK, d = 97, 16, 4, 256
    tab = table(1e7, 64, 70000)
    q, k = rnd(T, HQ, d, gen=g), rnd(T, HK, d, gen=g)
    wq, wk = rnd(d, scale=0.5, gen=g), rnd(d, scale=0.5, gen=g)
    pos = pos_of(T, 70000, g)

    def run(qs, ks):
        if kind == "rope":
            return rope(qs, ks, pos, tab)
        return qk_norm_rope(qs, ks, wq, wk, pos, tab, 1e-6)

    full_q, full_k = run(q, k)
    for tp in (2, 4, 8):
        hq, hk = HQ // tp, max(1, HK // tp)
        for r in range(tp):
            kr = (r * HK) // tp if HK >= tp else (r * HK) // tp
            yq, yk = run(q[:, r * hq:(r + 1) * hq], k[:, kr:kr + hk])
            assert bitwise_equal(yq, full_q[:, r * hq:(r + 1) * hq].contiguous()), (tp, r)
            assert bitwise_equal(yk, full_k[:, kr:kr + hk].contiguous()), (tp, r)
    if kind == "rope":  # the k operand does not change q's bits (one launch vs one operand)
        assert bitwise_equal(rope(q, None, pos, tab)[0], full_q)


@pytest.mark.parametrize("kind", ["rope", "qk"])
def test_decode_prefill_trainer_parity(kind):
    """decode (one token per call), prefill pack, trainer forward with autograd recording."""
    g = torch.Generator(device=DEV).manual_seed(9)
    T, hq, hk, d = 300, 4, 1, 256
    tab = table(1e7, 64, 70000)
    q, k = rnd(T, hq, d, gen=g), rnd(T, hk, d, gen=g)
    wq, wk = rnd(d, scale=0.5, gen=g), rnd(d, scale=0.5, gen=g)
    pos = torch.arange(T, device=DEV) + 1000

    def run(qs, ks, ps, grad=False):
        if grad:
            qs, ks = qs.clone().requires_grad_(), ks.clone().requires_grad_()
            wq_, wk_ = wq.clone().requires_grad_(), wk.clone().requires_grad_()
        else:
            wq_, wk_ = wq, wk
        with torch.set_grad_enabled(grad):
            if kind == "rope":
                out = rope(qs, ks, ps, tab)
            else:
                out = qk_norm_rope(qs, ks, wq_, wk_, ps, tab, 1e-6)
        return [o.detach() for o in out]

    prefill = run(q, k, pos)
    trainer = run(q, k, pos, grad=True)
    assert bitwise_equal(prefill[0], trainer[0]) and bitwise_equal(prefill[1], trainer[1])
    for t in range(0, T, 37):
        dq, dk = run(q[t:t + 1], k[t:t + 1], pos[t:t + 1])
        assert bitwise_equal(dq[0], prefill[0][t]) and bitwise_equal(dk[0], prefill[1][t])
    # chunked prefill: split anywhere
    for split in (1, 17, 128, 299):
        a = run(q[:split], k[:split], pos[:split])
        b = run(q[split:], k[split:], pos[split:])
        assert bitwise_equal(torch.cat((a[0], b[0])), prefill[0]) and bitwise_equal(torch.cat((a[1], b[1])), prefill[1])


@pytest.mark.parametrize("kind", ["rope_mla", "qk"])
def test_cuda_graph_replay_with_null_lanes(kind):
    g = torch.Generator(device=DEV).manual_seed(3)
    TMAX = 64
    if kind == "qk":
        tab, hq, hk, d = table(1e7, 64, 70000), 2, 1, 256
    else:
        tab, hq, hk, d = table(1e6, 64, 70000), 5, 1, 64
    sq, sk = torch.zeros(TMAX, hq, d, dtype=BF, device=DEV), torch.zeros(TMAX, hk, d, dtype=BF, device=DEV)
    spos = torch.zeros(TMAX, dtype=torch.long, device=DEV)
    wq, wk = rnd(d, scale=0.5, gen=g), rnd(d, scale=0.5, gen=g)

    def step():
        if kind == "qk":
            return qk_norm_rope(sq, sk, wq, wk, spos, tab, 1e-6)
        return rope(sq, sk, spos, tab, style="interleaved", out_style="neox")

    with torch.no_grad():
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outs = step()
        for n in (1, 7, 64):
            q, k, pos = rnd(n, hq, d, gen=g), rnd(n, hk, d, gen=g), pos_of(n, 70000, g)
            sq[:n], sk[:n], spos[:n] = q, k, pos
            sq[n:] = float("nan")  # NULL lanes: garbage beyond the live batch
            spos[n:] = 0
            graph.replay()
            if kind == "qk":
                ref = qk_norm_rope(q, k, wq, wk, pos, tab, 1e-6)
            else:
                ref = rope(q, k, pos, tab, style="interleaved", out_style="neox")
            assert bitwise_equal(outs[0][:n], ref[0]) and bitwise_equal(outs[1][:n], ref[1]), n


def test_noncontiguous_mla_views_and_misaligned_operands():
    g = torch.Generator(device=DEV).manual_seed(4)
    T, H, nope, rope_d, kv_lora = 129, 20, 192, 64, 512
    tab = table(1e6, 64, 70000)
    Q = rnd(T, H, nope + rope_d, gen=g)
    KV = rnd(T, kv_lora + rope_d, gen=g)
    pos = pos_of(T, 70000, g)
    q_pe = Q.split((nope, rope_d), dim=-1)[1]
    k_pe = KV.split((kv_lora, rope_d), dim=-1)[1].unsqueeze(1)
    assert not q_pe.is_contiguous() and not k_pe.is_contiguous()
    a = rope(q_pe, k_pe, pos, tab, style="interleaved", out_style="neox")
    b = rope(q_pe.contiguous(), k_pe.contiguous(), pos, tab, style="interleaved", out_style="neox")
    assert bitwise_equal(a[0], b[0]) and bitwise_equal(a[1], b[1])
    # misaligned: odd element offset (declined by the pair loads, copied, same bits)
    buf = rnd(T * H * 64 + 1, gen=g)
    odd = buf[1:].view(T, H, 64)
    assert odd.data_ptr() % 4 != 0
    c = rope(odd, None, pos, tab, style="interleaved", out_style="neox")[0]
    assert bitwise_equal(c, rope(odd.contiguous(), None, pos, tab, style="interleaved", out_style="neox")[0])
    # transposed head/token axes (odd strides) and qk column views of a packed projection
    P = rnd(T, 2 * 256 + 256 + 256, gen=g)
    qv, kv = P[:, :512].view(T, 2, 256), P[:, 512:768].view(T, 1, 256)
    wq, wk = rnd(256, scale=0.5, gen=g), rnd(256, scale=0.5, gen=g)
    t2 = table(1e7, 64, 70000)
    x = qk_norm_rope(qv, kv, wq, wk, pos, t2, 1e-6)
    y = qk_norm_rope(qv.contiguous(), kv.contiguous(), wq, wk, pos, t2, 1e-6)
    assert bitwise_equal(x[0], y[0]) and bitwise_equal(x[1], y[1])
    perm = rnd(H, T, 64, gen=g).transpose(0, 1)
    assert bitwise_equal(rope(perm, None, pos, tab)[0], rope(perm.contiguous(), None, pos, tab)[0])


def test_large_positions_and_mrope_text_rows():
    g = torch.Generator(device=DEV).manual_seed(8)
    tab = table(1e7, 64, 262144, (11, 11, 10), True)
    assert tab.m1 and tab.m2
    T = 300
    q, k = rnd(T, 2, 256, gen=g), rnd(T, 1, 256, gen=g)
    wq, wk = rnd(256, scale=0.5, gen=g), rnd(256, scale=0.5, gen=g)
    pos = torch.randint(262144 - 5000, 262144, (T,), device=DEV, generator=g)
    pos[:4] = torch.tensor([262143, 262142, 131072, 0], device=DEV)
    p3 = pos.expand(3, T).contiguous()
    for fn in (lambda p: rope(q, k, p, tab), lambda p: qk_norm_rope(q, k, wq, wk, p, tab, 1e-6)):
        a, b = fn(pos), fn(p3)
        assert bitwise_equal(a[0], b[0]) and bitwise_equal(a[1], b[1])
    # genuine 3-axis positions follow the Qwen3.5 interleaved column map
    p3 = torch.stack((pos, torch.randint(0, 64, (T,), device=DEV, generator=g), torch.randint(0, 64, (T,), device=DEV,
                                                                                               generator=g)))
    yq, _ = qk_norm_rope(q, k, wq, wk, p3, tab, 1e-6)
    rq, _ = R.qk_norm_rope_recipe(q, wq, p3, tab.table, 1e-6, tab.m1, tab.m2)
    assert bitwise_equal(yq, rq)
    zq, _ = rope(q, k, p3, tab)
    assert bitwise_equal(zq, R.rope_recipe(q, p3, tab.table, tab.m1, tab.m2))
    assert not bitwise_equal(zq, rope(q, k, pos, tab)[0])
    # the column map equals common/rope.py's MRoPE selection
    from isoexec.core.rope import resolve_rope
    from isoexec.schema import Attn

    spec = Attn(num_q_heads=16, num_kv_heads=2, head_dim=256, rope_theta=1e7, partial_rotary=0.25,
                mrope_section=(11, 11, 10), mrope_interleaved=True)
    ang = resolve_rope(spec)(p3[:, :5])
    inv = torch.tensor(R.inv_freq_double(1e7, 64), device=DEV)
    axis = torch.tensor([((tab.m1 >> j) & 1) + 2 * ((tab.m2 >> j) & 1) for j in range(32)], device=DEV)
    expect = p3[axis][:, :5].t().double() * inv
    assert torch.allclose(ang.double(), expect, rtol=1e-6)


def test_signed_zero_subnormal_overflow_hazards():
    tab = table(1e6, 64, 70000)
    T = 64
    pos = torch.arange(T, device=DEV)
    # signed zeros and exact zeros
    x = torch.zeros(T, 4, 64, dtype=torch.float32, device=DEV)
    x[::2] = -0.0
    x[:, :, ::3] = torch.finfo(torch.float32).tiny / 8  # subnormal fp32 inputs
    x[5, 1, 7] = 3.0e38
    x[6, 2, 9] = float("inf")
    y = rope(x, None, pos, tab)[0]
    r = R.rope_recipe(x, pos, tab.table)
    finite = torch.isfinite(r)
    assert bitwise_equal(torch.where(finite, y, 0), torch.where(finite, r, 0))
    assert torch.equal(torch.isnan(y), torch.isnan(r))
    neg0 = int(((y == 0) & torch.signbit(y)).sum())
    sub = int(((r != 0) & (r.abs() < torch.finfo(torch.float32).tiny)).sum())
    assert_hazard_exercised("signed_zero", neg0)
    assert_hazard_exercised("subnormal", sub)
    assert_hazard_exercised("overflow", int((~finite).sum()))
    # qk norm: all-zero row (inv = 1/sqrt(eps)), -0 rows, huge rows, bf16 subnormal-scale rows
    for dtype in (BF, torch.float32):
        q = torch.zeros(T, 2, 256, dtype=dtype, device=DEV)
        q[1] = -0.0
        q[2] = 1e30 if dtype == torch.float32 else 3e38
        q[3, :, :7] = 1e-40 if dtype == torch.float32 else 1e-38
        q[4] = torch.randn(2, 256, device=DEV).to(dtype)
        k = q[:, :1].clone()
        w = torch.zeros(256, dtype=dtype, device=DEV)
        w[::5] = -1.0  # 1 + w == +0: exact zero gains
        tq = table(1e7, 64, 70000)
        yq, yk = qk_norm_rope(q, k, w, w, pos, tq, 1e-6)
        rq, _ = R.qk_norm_rope_recipe(q, w, pos, tq.table, 1e-6)
        fin = torch.isfinite(rq)
        assert bitwise_equal(torch.where(fin, yq, 0), torch.where(fin, rq, 0))
        assert torch.equal(torch.isnan(yq), torch.isnan(rq))
        assert_hazard_exercised("signed_zero_qk", int(((yq == 0) & torch.signbit(yq)).sum()))


def test_empty_and_declined_shapes():
    tab = table(1e6, 64, 1000)
    e = torch.empty(0, 4, 64, dtype=BF, device=DEV)
    pe = torch.empty(0, dtype=torch.long, device=DEV)
    a, b = rope(e, e[:, :1], pe, tab)
    assert a.shape == (0, 4, 64) and b.shape == (0, 1, 64)
    w = torch.zeros(64, dtype=BF, device=DEV)
    a, b = qk_norm_rope(e, e[:, :1], w, w, pe, tab, 1e-6)
    assert a.shape == (0, 4, 64)
    # Unsupported shapes require an explicit reference call; production refuses.
    t6 = build_table(1e4, 6, 100, DEV)
    x = rnd(9, 3, 10)
    p = pos_of(9, 100)
    with pytest.raises(ValueError, match="SM100 rope declined"):
        rope(x, None, p, t6)
    t8 = build_table(1e4, 8, 100, DEV)
    q = rnd(9, 3, 100)
    wq = rnd(100, scale=0.5)
    with pytest.raises(ValueError, match="SM100 qk_norm_rope forward declined"):
        qk_norm_rope(q, q, wq, wq, p, t8, 1e-6)
    with pytest.raises(ValueError):
        rope(x, None, p.int(), t6)


@pytest.mark.parametrize("hq,hk,D,rot,style", [(32, 8, 128, 128, "neox"), (2, 1, 256, 64, "neox"),
                                               (20, 1, 64, 64, "interleaved")])
def test_inplace_equals_out_of_place(hq, hk, D, rot, style):
    g = torch.Generator(device=DEV).manual_seed(21)
    tab = table(1e6, rot, 70000)
    T = 333
    P = rnd(T, (hq + hk) * D + 64, gen=g)  # q|k column views of one projection, plus a tail
    q, k = P[:, : hq * D].view(T, hq, D), P[:, hq * D: (hq + hk) * D].view(T, hk, D)
    pos = pos_of(T, 70000, g)
    ref = rope(q, k, pos, tab, style=style)
    tail = P[:, -64:].clone()
    out = rope(q, k, pos, tab, style=style, inplace=True)
    assert out[0].data_ptr() == q.data_ptr()
    assert bitwise_equal(q, ref[0]) and bitwise_equal(k, ref[1]) and bitwise_equal(P[:, -64:], tail)
    with pytest.raises(ValueError):
        rope(q, k, pos, tab, style="interleaved", out_style="neox", inplace=True)
