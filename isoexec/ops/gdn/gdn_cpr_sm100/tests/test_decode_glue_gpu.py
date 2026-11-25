"""The CPR sm100 engine state's decode is the former decode chain, bit for bit.

Former chain (``_unfused_decode`` below): the q|k|v columns of the in_proj GEMM output are cat'ed into
a packed buffer, alpha/beta compacted, rows resolved by the Triton row kernel, the conv updated in
place, q/k/v/alpha/beta split out by the fused copy kernel, the null lanes mapped by ``where``, the
core run, and the LRU clock/stamp written by two more launches.

``CprSm100GDN.decode``: the q|k|v column window and the alpha/beta views go in as they are, one
``decode_rows`` launch resolves the rows, the null lanes and the stamp (for every pool of a decode
group at once), the conv reads the strided window into a fresh output, and the core reads row-strided
q/k/v and the strided alpha/beta.

Compared as integer views after every step: the core output, the conv state, the recurrent records,
``last_used`` and the clock, with padded/unknown lanes and a prefill (with deferred alpha/beta) first.
"""

from __future__ import annotations

import pytest
import torch

from isoexec.ops.gdn.gdn_cpr_sm100.tests._fixtures import SM100, biteq, state_equal

pytestmark = SM100

DEV = "cuda"
BF = torch.bfloat16
HK, HV, DK, DV, W = 16, 32, 128, 128, 4
KD, VD = HK * DK, HV * DV
DQKV = 2 * KD + VD
WIDTH = DQKV + VD + 2 * HV  # q|k|v|z|beta|alpha: the in_proj GEMM row
CAP = 300


@pytest.fixture(autouse=True)
def _flags(monkeypatch):
    for k, v in {"ISOEXEC_GDN_KERNEL": "cpr", "ISOEXEC_GDN_NATIVE_KERNELS": "1", "ISOEXEC_GDN_NATIVE_STATE": "0",
                 "ISOEXEC_GDN_CPR_SLEEP": "0", "ISOEXEC_GDN_CPR_APC": "0"}.items():
        monkeypatch.setenv(k, v)
    from isoexec.ops.gdn.conv_sm100 import integration

    integration.install()
    yield


def _pool(seed):
    from isoexec.ops.gdn.gdn_cpr_sm100.engine_state import build_cpr_sm100_gdn

    g = torch.Generator(device=DEV).manual_seed(seed)
    cw = (torch.randn(DQKV, W, device=DEV, generator=g) * 0.4).to(BF)
    A_log = torch.log(torch.empty(HV, device=DEV).uniform_(0.002, 0.03, generator=g)).to(BF)
    dt = (torch.randn(HV, device=DEV, generator=g) * 0.5).to(BF)
    return build_cpr_sm100_gdn(max_num_seqs=CAP, chunk_size=64, conv_weight=cw, conv_bias=None, A_log=A_log,
                               dt_bias=dt, num_k_heads=HK, head_k_dim=DK, num_v_heads=HV, head_v_dim=DV,
                               activation="silu", dtype=BF, device=DEV)


def _gemm_rows(T, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return (torch.randn(T, WIDTH, device=DEV, generator=g) * 1.3).to(BF)


def _views(buf):
    T = buf.shape[0]
    q = buf[:, :KD].unflatten(-1, (HK, DK))
    k = buf[:, KD:2 * KD].unflatten(-1, (HK, DK))
    v = buf[:, 2 * KD:DQKV].unflatten(-1, (HV, DV))
    b = buf[:, DQKV + VD:DQKV + VD + HV]
    a = buf[:, DQKV + VD + HV:]
    assert q.shape[0] == T
    return q, k, v, a, b


def _decode(pool, slots, buf):
    """What runtimes/vllm/_gdn.run_core + gdn_layer_core hand the state on a pure-decode step."""
    from isoexec.runtimes.vllm._gdn import _adjacent_qkv

    q, k, v, a, b = _views(buf)
    mixed = _adjacent_qkv(q, k, v)
    assert mixed is not None and mixed.stride(0) == WIDTH
    return pool.decode(slots, mixed, a, b)


def _unfused_decode(pool, slots, buf):
    """The former decode chain, launch for launch, on its packed operands."""
    from isoexec.ops.gdn.gdn_cpr_sm100.engine_state import conv, ops

    q, k, v, a, b = _views(buf)
    x = torch.cat((q.flatten(1), k.flatten(1), v.flatten(1)), dim=-1)
    rows, rows32 = pool._rows_pair(slots)
    y = conv.causal_conv1d_update(x, pool.conv_state, pool.conv_weight, pool.conv_bias, pool.activation,
                                  conv_state_indices=rows32)
    q, k, v, a, b = pool._split_qkv_raw(y, a.contiguous(), b.contiguous())
    o = ops.decode(q, k, v, a, b, pool.A_log, pool.dt_bias, pool.records,
                   torch.where(rows32 > 0, rows32, pool._null32))
    pool._clock += 1
    pool.last_used[rows] = pool._clock
    return o


def _state_equal(p, r):
    """Conv windows, LRU stamps and clock whole; each live record as the record contract defines it
    (bytes past a record's open chunk are explicitly unspecified, see ``_fixtures.state_equal``)."""
    rows = sorted(p._slot2row.values())
    assert rows == sorted(r._slot2row.values())
    return (biteq(p.conv_state, r.conv_state) and state_equal(p.records, r.records, rows)
            and torch.equal(p.last_used, r.last_used) and torch.equal(p._clock, r._clock))


@pytest.mark.parametrize("n", [1, 7, 256])
def test_decode_is_the_unfused_chain(n):
    fused, ref = _pool(1), _pool(1)
    live = list(range(5, 5 + n))
    # Prefill every live slot (alpha/beta deferred: strided views reach the prefill split).
    lens = [1 + (i * 37) % 70 for i in range(n)]
    T = sum(lens)
    qsl = [0]
    for L in lens:
        qsl.append(qsl[-1] + L)
    pre = _gemm_rows(T, 11)
    for pool in (fused, ref):
        q, k, v, a, b = _views(pre)
        x = torch.cat((q.flatten(1), k.flatten(1), v.flatten(1)), dim=-1)
        o = pool.prefill(torch.tensor(live, device=DEV, dtype=torch.int32), live, x, a, b, qsl)
        if pool is fused:
            o_f = o
        else:
            assert biteq(o_f, o)
    assert _state_equal(fused, ref)

    # Decode steps: lane order shuffled, plus padded lanes (an unknown slot -> null row 0, and a
    # negative slot clamped to 0) as a captured graph's replay padding carries.
    g = torch.Generator().manual_seed(n)
    for step in range(6):
        order = [live[i] for i in torch.randperm(n, generator=g).tolist()]
        slots = order + [CAP + 50, -3]
        st = torch.tensor(slots, device=DEV, dtype=torch.int32)
        buf = _gemm_rows(len(slots), 100 + step)
        o_ref = _unfused_decode(ref, st, buf.clone())
        o_fus = _decode(fused, st, buf)
        assert biteq(o_fus, o_ref), f"step {step}: core output"
        assert _state_equal(fused, ref), f"step {step}: state"


def test_decode_rows_kernel_matches_the_aten_chain():
    from isoexec.ops.gdn.gdn_cpr_sm100._extension import extension

    g = torch.Generator(device=DEV).manual_seed(3)
    for n in (1, 7, 128, 256, 1500):
        m = torch.randint(0, 400, (900,), device=DEV, generator=g, dtype=torch.int64)
        m[::5] = 0
        for dt in (torch.int32, torch.int64):
            slots = torch.randint(-5, 1000, (n,), device=DEV, generator=g).to(dt)
            clock = torch.tensor(17, device=DEV, dtype=torch.int64)
            lu = torch.zeros(400, device=DEV, dtype=torch.int64)
            rows = m[slots.long().clamp(0, m.numel() - 1)]
            r32 = rows.to(torch.int32)
            lane = torch.where(r32 > 0, r32, torch.full((), -1, device=DEV, dtype=torch.int32))
            lu_ref = lu.clone()
            lu_ref[rows] = 18
            got32 = torch.empty(n, device=DEV, dtype=torch.int32)
            gotl = torch.empty(n, device=DEV, dtype=torch.int32)
            extension().decode_rows(slots, [m], [got32], [gotl], [clock], [lu])
            assert torch.equal(got32, r32) and torch.equal(gotl, lane)
            assert int(clock) == 18 and torch.equal(lu, lu_ref)


def _group(pools):
    from isoexec.ops.gdn.gdn_cpr_sm100.engine_state import DecodeGroup

    g = DecodeGroup()
    for p in pools:
        g.add(p)
    return g


def _shared(slots):
    st = torch.tensor(slots, device=DEV, dtype=torch.int32)
    st._isoexec_step_shared = True  # what gdn_engine_patch._decode_slots marks under the metadata alias
    return st


@pytest.mark.parametrize("n", [1, 7, 256])
def test_grouped_layers_resolve_rows_once_per_step(n):
    """Three layers of one model: the first decode of a step resolves all three pools' rows and stamps."""
    L = 3
    fused = [_pool(10 + i) for i in range(L)]
    ref = [_pool(10 + i) for i in range(L)]
    _group(fused)
    live = list(range(2, 2 + n))
    for p in fused + ref:
        p._assign_many(live)
    g = torch.Generator().manual_seed(n)
    for step in range(4):
        order = [live[i] for i in torch.randperm(n, generator=g).tolist()] + [CAP + 7, 0]
        st = _shared(order)
        bufs = [_gemm_rows(len(order), 300 + 10 * step + i) for i in range(L)]
        for i in range(L):
            o_ref = _unfused_decode(ref[i], st.clone(), bufs[i].clone())
            o_fus = _decode(fused[i], st, bufs[i])
            assert biteq(o_fus, o_ref), f"step {step} layer {i}"
        for i in range(L):
            assert _state_equal(fused[i], ref[i]), f"step {step} layer {i}: state"


def test_a_layer_decoded_twice_on_one_slot_tensor_is_stamped_twice_not_thrice():
    L = 3
    fused = [_pool(20 + i) for i in range(L)]
    ref = [_pool(20 + i) for i in range(L)]
    _group(fused)
    live = [3, 4, 5]
    for p in fused + ref:
        p._assign_many(live)
    st = _shared(live)
    calls = [0, 0, 1, 2]  # layer 0 twice: its second call finds its entry taken, others pending
    for j, i in enumerate(calls):
        buf = _gemm_rows(3, 500 + j)
        o_ref = _unfused_decode(ref[i], st.clone(), buf.clone())
        o_fus = _decode(fused[i], st, buf)
        assert biteq(o_fus, o_ref)
    for i in range(L):
        assert _state_equal(fused[i], ref[i]), i


def test_grouped_graph_replay():
    L, n = 3, 9
    fused = [_pool(30 + i) for i in range(L)]
    ref = [_pool(30 + i) for i in range(L)]
    _group(fused)
    live = list(range(1, 1 + n))
    for p in fused + ref:
        p._assign_many(live)
    statics = [_gemm_rows(n + 1, 70 + i) for i in range(L)]
    st = _shared(live + [CAP + 1])
    for i in range(L):  # one eager round to warm
        _decode(fused[i], st, statics[i].clone())
        _unfused_decode(ref[i], st.clone(), statics[i].clone())
    graph = torch.cuda.CUDAGraph()
    st_cap = _shared(live + [CAP + 1])
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.cuda.graph(graph):
        outs = [_decode(fused[i], st_cap, statics[i]) for i in range(L)]
    torch.cuda.current_stream().wait_stream(s)
    for step in range(3):
        news = [_gemm_rows(n + 1, 900 + 10 * step + i) for i in range(L)]
        for i in range(L):
            statics[i].copy_(news[i])
        graph.replay()
        refs = [_unfused_decode(ref[i], st.clone(), news[i].clone()) for i in range(L)]
        torch.cuda.synchronize()
        for i in range(L):
            assert biteq(outs[i], refs[i]), f"replay {step} layer {i}"
            assert _state_equal(fused[i], ref[i]), f"replay {step} layer {i}: state"


def test_one_launch_serves_more_pools_than_it_carries():
    """45 pools on one slot tensor (two launches of up to 40): each pool's rows, lanes and stamp."""
    from isoexec.ops.gdn.gdn_cpr_sm100._extension import extension

    g = torch.Generator(device=DEV).manual_seed(4)
    L, n = 45, 300
    maps = [torch.randint(0, 300, (700,), device=DEV, generator=g, dtype=torch.int64) for _ in range(L)]
    slots = torch.randint(-5, 900, (n,), device=DEV, generator=g).to(torch.int32)
    clocks = [torch.tensor(i, device=DEV, dtype=torch.int64) for i in range(L)]
    lus = [torch.zeros(300, device=DEV, dtype=torch.int64) for _ in range(L)]
    rows = [torch.empty(n, device=DEV, dtype=torch.int32) for _ in range(L)]
    lanes = [torch.empty(n, device=DEV, dtype=torch.int32) for _ in range(L)]
    extension().decode_rows(slots, maps, rows, lanes, clocks, lus)
    for i in range(L):
        r = maps[i][slots.long().clamp(0, 699)]
        lu = torch.zeros(300, device=DEV, dtype=torch.int64)
        lu[r] = i + 1
        assert torch.equal(rows[i], r.to(torch.int32))
        assert torch.equal(lanes[i], torch.where(r > 0, r, -1).to(torch.int32))
        assert int(clocks[i]) == i + 1 and torch.equal(lus[i], lu)
