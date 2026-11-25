"""The SM100 native recurrent GDN core: one function on the trainer and the engine, bit for bit.

- engine prefill (whole or chunk-split) then decode on vLLM-layout mamba pages == trainer forward, with
  conv_sm100 installed on both sides as production installs it;
- the trainer forward is invariant to packing (a sequence alone or among others) and to the head split
  (trainer TP2 ranks vs the engine's TP1);
- engine decode of a row is invariant to the decode batch it rides in;
- the trainer backward is run-to-run deterministic and packing-invariant.
"""

from __future__ import annotations

import pytest
import torch

from isoexec.ops._testing import bitwise_equal

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="gdn_recurrent_sm100 is proven on sm_100 only",
)

DEV = "cuda"
D = 128
W = 4
#: (key heads, value heads) of the GDN layers at TP1 (the engine); the trainer's TP2 rank holds half of each.
GEOMETRIES = {"qwen3.5-9b": (16, 32), "qwen3.6-27b": (16, 48)}
HK, HV = GEOMETRIES["qwen3.5-9b"]  # set per test by the `heads` fixture


@pytest.fixture(autouse=True, params=list(GEOMETRIES))
def heads(request, monkeypatch):
    hk, hv = GEOMETRIES[request.param]
    monkeypatch.setattr(__import__(__name__, fromlist=["HK"]), "HK", hk)
    monkeypatch.setattr(__import__(__name__, fromlist=["HV"]), "HV", hv)
    return hk, hv


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    from isoexec.ops import _dispatch, _resolve
    from isoexec.ops.gdn.conv_sm100 import integration as conv

    monkeypatch.setattr(_dispatch, "ARCH", "sm100")
    monkeypatch.setenv("ISOEXEC_GDN_KERNEL", "recurrent")
    monkeypatch.setenv("ISOEXEC_GDN_NATIVE_KERNELS", "1")
    monkeypatch.setenv("ISOEXEC_GDN_NATIVE_STATE", "1")
    monkeypatch.setenv("ISOEXEC_GDN_CHUNKED_PREFILL", "1")
    monkeypatch.setenv("ISOEXEC_GDN_FLA_BACKWARD", "1")
    monkeypatch.delenv("ISOEXEC_GDN_CPR_APC", raising=False)
    assert _dispatch.selected("gdn") == "gdn.gdn_recurrent_sm100"
    conv.install()
    yield
    conv.uninstall()
    _resolve._registry.cache_clear()


def _rand(g, *shape, scale=1.0):
    return (torch.randn(*shape, generator=g, device=DEV) * scale).to(torch.bfloat16)


def _inputs(lens, seed, hk=None, hv=None):
    hk, hv = hk or HK, hv or HV
    g = torch.Generator(device=DEV).manual_seed(seed)
    T = sum(lens)
    x = _rand(g, T, (2 * hk + hv) * D)  # pre-conv q|k|v
    a, b = _rand(g, T, hv), _rand(g, T, hv)
    w = _rand(g, (2 * hk + hv) * D, W, scale=0.5)
    A_log = _rand(g, hv, scale=0.5)
    dt_bias = _rand(g, hv)
    cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32, device=DEV)
    return x, a, b, w, A_log, dt_bias, cu


def _split(y, hk=None, hv=None):
    hk, hv = hk or HK, hv or HV
    T = y.shape[0]
    q, k, v = y.split([hk * D, hk * D, hv * D], dim=-1)
    return q.reshape(T, hk, D), k.reshape(T, hk, D), v.reshape(T, hv, D)


def _trainer(x, a, b, w, A_log, dt_bias, cu):
    """The trainer's layer core: conv.short then gdn.core, through the family facades."""
    from isoexec.ops import gdn

    y = gdn.gdn_native_conv(x, w, None, cu_seqlens=cu, activation="silu")
    hv = a.shape[1]
    q, k, v = _split(y, hk=(x.shape[1] // D - hv) // 2, hv=hv)
    return gdn.gdn_core(q, k, v, a, b, A_log, dt_bias, cu)


def _pool(blocks, w, A_log, dt_bias):
    from isoexec.ops import gdn

    dim = w.shape[0]
    ssm = torch.zeros(blocks, HV, D, D, device=DEV, dtype=torch.float32)  # vLLM MambaSpec (HV, V, K) fp32
    conv = torch.zeros(blocks, W - 1, dim, device=DEV, dtype=torch.bfloat16).transpose(-1, -2)  # (D, W-1) view
    return gdn.build_state_pool(
        max_num_seqs=blocks, chunk_size=64, ssm_state=ssm, conv_state=conv, conv_weight=w, conv_bias=None,
        A_log=A_log, dt_bias=dt_bias, num_k_heads=HK, head_k_dim=D, num_v_heads=HV, head_v_dim=D,
        activation="silu", dtype=torch.bfloat16, device=DEV,
    )


@pytest.mark.parametrize("split", [None, [37, 20]])
def test_engine_prefill_then_decode_equals_trainer_bitwise(split):
    lens, n_decode = [130, 45], 9
    x, a, b, w, A_log, dt_bias, cu = _inputs(lens, seed=3)
    with torch.no_grad():
        want = _trainer(x, a, b, w, A_log, dt_bias, cu)
    pool = _pool(16, w, A_log, dt_bias)
    assert pool._native and pool._native_kernels
    slots_cpu = [5, 11]
    slots = torch.tensor(slots_cpu, dtype=torch.int32, device=DEV)
    starts = [0, lens[0]]
    out = torch.full_like(want, float("nan"))
    bounds = [[0, lens[i] - n_decode] for i in range(2)] if split is None else \
        [[0, split[i], lens[i] - n_decode] for i in range(2)]
    for call in range(len(bounds[0]) - 1):
        spans = [(starts[i] + bounds[i][call], starts[i] + bounds[i][call + 1]) for i in range(2)]
        qsl = [0]
        for lo, hi in spans:
            qsl.append(qsl[-1] + hi - lo)
        cat = lambda t: torch.cat([t[lo:hi] for lo, hi in spans]).contiguous()  # noqa: E731
        qcu = torch.tensor(qsl, dtype=torch.int32, device=DEV)
        with torch.no_grad():
            o = pool.prefill(slots, slots_cpu, cat(x), cat(a), cat(b), qsl,
                             has_initial_state=[call > 0] * 2, prefill_query_start_loc=qcu)
        for i, (lo, hi) in enumerate(spans):
            out[lo:hi] = o[qsl[i]:qsl[i + 1]]
    for step in range(n_decode):
        t = torch.tensor([starts[i] + lens[i] - n_decode + step for i in range(2)], device=DEV)
        with torch.no_grad():
            o = pool.decode(slots, x[t].clone(), a[t].contiguous(), b[t].contiguous())
        out[t] = o
    pre = torch.cat([torch.arange(starts[i], starts[i] + lens[i] - n_decode) for i in range(2)]).to(DEV)
    dec = torch.cat([torch.arange(starts[i] + lens[i] - n_decode, starts[i] + lens[i]) for i in range(2)]).to(DEV)
    assert bitwise_equal(out[pre], want[pre]), "prefill tokens"
    assert bitwise_equal(out[dec], want[dec]), "decode tokens"


def test_trainer_forward_is_packing_and_head_split_invariant():
    lens = [200, 1, 64, 129]
    x, a, b, w, A_log, dt_bias, cu = _inputs(lens, seed=7)
    with torch.no_grad():
        packed = _trainer(x, a, b, w, A_log, dt_bias, cu)
        lo = sum(lens[:3])
        alone = _trainer(x[lo:].contiguous(), a[lo:].contiguous(), b[lo:].contiguous(), w, A_log, dt_bias,
                         torch.tensor([0, lens[3]], dtype=torch.int32, device=DEV))
        assert bitwise_equal(packed[lo:], alone), "a sequence alone == the same sequence in a pack"
        # A TP2 rank: key heads [0, HK/2), value heads [0, HV/2), and their conv channels.
        hk, hv = HK // 2, HV // 2
        cols = torch.cat([torch.arange(0, hk * D), torch.arange(HK * D, (HK + hk) * D),
                          torch.arange(2 * HK * D, (2 * HK + hv) * D)]).to(DEV)
        rank = _trainer(x[:, cols].contiguous(), a[:, :hv].contiguous(), b[:, :hv].contiguous(),
                        w[cols].contiguous(), A_log[:hv].contiguous(), dt_bias[:hv].contiguous(), cu)
        assert bitwise_equal(packed[:, :hv], rank), "the TP2 rank's heads == the same heads at TP1"


def test_engine_decode_is_batch_invariant():
    x, a, b, w, A_log, dt_bias, _ = _inputs([64], seed=11)
    n = 64
    alone, crowd = _pool(n + 1, w, A_log, dt_bias), _pool(n + 1, w, A_log, dt_bias)
    g = torch.Generator(device=DEV).manual_seed(5)
    for p in (alone, crowd):
        p.ssm_state.copy_(torch.randn(p.ssm_state.shape, generator=torch.Generator(device=DEV).manual_seed(1),
                                      device=DEV) * 0.1)
    slots = torch.arange(1, n + 1, dtype=torch.int32, device=DEV)
    for _ in range(4):
        xs, as_, bs = _rand(g, n, x.shape[1]), _rand(g, n, HV), _rand(g, n, HV)
        with torch.no_grad():
            many = crowd.decode(slots, xs.clone(), as_, bs)
            one = alone.decode(slots[7:8], xs[7:8].clone(), as_[7:8].contiguous(), bs[7:8].contiguous())
        assert bitwise_equal(many[7:8], one), "row 7 decoded alone == row 7 in a batch of 64"
    assert bitwise_equal(crowd.ssm_state[8], alone.ssm_state[8])


def _grads(x, a, b, w, A_log, dt_bias, cu, do):
    from isoexec.ops import gdn

    leaves = [t.detach().clone().requires_grad_(True) for t in (x, a, b, A_log, dt_bias)]
    xl, al, bl, Al, dl = leaves
    with torch.no_grad():
        y = gdn.gdn_native_conv(xl, w, None, cu_seqlens=cu, activation="silu")
    q, k, v = (t.detach().requires_grad_(True) for t in _split(y))
    o = gdn.gdn_core(q, k, v, al, bl, Al, dl, cu)
    o.backward(do)
    return [t.grad for t in (q, k, v, al, bl, Al, dl)]


def test_trainer_backward_is_deterministic_and_packing_invariant():
    lens = [300, 77, 129]
    x, a, b, w, A_log, dt_bias, cu = _inputs(lens, seed=13)
    do = _rand(torch.Generator(device=DEV).manual_seed(2), sum(lens), HV, D)
    A_log, dt_bias = A_log.float(), dt_bias.float()
    first = _grads(x, a, b, w, A_log, dt_bias, cu, do)
    second = _grads(x, a, b, w, A_log, dt_bias, cu, do)
    for name, g1, g2 in zip(("dq", "dk", "dv", "da", "db", "dA_log", "ddt_bias"), first, second):
        assert torch.isfinite(g1).all(), name
        assert bitwise_equal(g1, g2), f"{name}: two backward passes differ"
    lo = lens[0]
    tail = [t[lo:].contiguous() for t in (x, a, b)]
    alone = _grads(*tail, w, A_log, dt_bias,
                   torch.tensor([0, lens[1], lens[1] + lens[2]], dtype=torch.int32, device=DEV), do[lo:].contiguous())
    for name, g1, g2 in zip(("dq", "dk", "dv", "da", "db"), first, alone):
        assert bitwise_equal(g1[lo:], g2), f"{name}: packed rows differ from the same sequences packed alone"


def test_engine_decode_null_lanes_are_zero_and_inert():
    """A padded graph lane (the null block, id 0) decodes to the zero row and leaves real lanes' bits alone."""
    x, a, b, w, A_log, dt_bias, _ = _inputs([8], seed=17)
    padded, plain = _pool(9, w, A_log, dt_bias), _pool(9, w, A_log, dt_bias)
    g = torch.Generator(device=DEV).manual_seed(9)
    real = torch.tensor([3, 5], dtype=torch.int32, device=DEV)
    lanes = torch.tensor([3, 0, 5, 0], dtype=torch.int32, device=DEV)
    for _ in range(3):
        xs, as_, bs = _rand(g, 4, x.shape[1]), _rand(g, 4, HV), _rand(g, 4, HV)
        keep = torch.tensor([0, 2], device=DEV)
        with torch.no_grad():
            got = padded.decode(lanes, xs.clone(), as_, bs)
            want = plain.decode(real, xs[keep].clone(), as_[keep].contiguous(), bs[keep].contiguous())
        assert bitwise_equal(got[keep], want), "real lanes"
        assert bitwise_equal(got[1::2], torch.zeros_like(got[1::2])), "null lanes decode to +0"
    assert bitwise_equal(padded.ssm_state, plain.ssm_state)
