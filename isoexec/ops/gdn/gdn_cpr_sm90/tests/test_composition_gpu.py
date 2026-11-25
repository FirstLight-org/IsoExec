"""Batch composition and packing invariance, head counts, input layouts, parameter dtypes, graph replay.

Every equality claim is compared on integer views (``ops._testing.bitwise_equal``). Claim map:
tests/README.md.
"""

from __future__ import annotations

import pytest
import torch

from isoexec.ops._testing import bitwise_equal
from isoexec.ops.gdn.gdn_cpr_sm90 import ops as G
from isoexec.ops.gdn.gdn_cpr_sm90.tests._fixtures import (
    BF,
    DECODE_PLANS,
    DEV,
    SM90,
    decode_stream,
    fresh,
    ids,
    make,
    prefill,
    rows,
    state_equal,
)

pytestmark = SM90


def test_prefill_row_is_independent_of_batchmates_and_position():
    """A sequence alone == the same sequence anywhere in packs of other sequences, output and state."""
    lanes = [37, 130, 64, 1, 90]
    d = make(lanes, 2, 4, seed=21)
    target = 1
    tr = slice(d["cu"][target], d["cu"][target + 1])
    st_a = fresh(2, 4)
    alone = prefill(d, tr, cu=[0, lanes[target]], state=st_a, slot_in=ids([1]))
    for order in ([0, 1, 2, 3, 4], [4, 3, 1], [1, 0], [2, 4, 1, 3, 0]):
        sel = torch.cat([torch.arange(d["cu"][i], d["cu"][i + 1], device=DEV) for i in order])
        cu = [0]
        for i in order:
            cu.append(cu[-1] + lanes[i])
        st = fresh(len(order) + 1, 4)
        o = prefill(d, sel, cu=cu, state=st, slot_in=ids(range(1, len(order) + 1)))
        k = order.index(target)
        assert bitwise_equal(o[cu[k]:cu[k + 1]], alone)
        assert state_equal(st, st_a, [k + 1], [1])


@pytest.mark.parametrize("split", [None] + DECODE_PLANS)
def test_decode_batch_composition_and_lane_order(split):
    """Decode: a sequence alone == with N-1 others in any lane order (any plan, the default heuristic too)."""
    lanes = [70] * 9
    d = make(lanes, 2, 4, seed=22)
    st_a = fresh(2, 4)
    alone = decode_stream(d, [70], st_a, plan=split)[:70]
    for order in ([0, 1, 2, 3, 4, 5, 6, 7, 8], [8, 0, 3], [5, 0]):
        st = fresh(10, 4)
        o = decode_stream(d, lanes, st, order=order, plan=split)
        assert bitwise_equal(o[:70], alone)
        assert state_equal(st, st_a, [1])


@pytest.mark.parametrize("hk,hv", [(3, 6), (1, 5), (5, 5), (2, 12)])
def test_non_power_of_two_heads(hk, hv):
    lanes = [66, 3]
    d = make(lanes, hk, hv, seed=23)
    st_p, st_d = fresh(3, hv), fresh(3, hv)
    o = prefill(d, state=st_p, slot_in=ids([1, 2]))
    od = decode_stream(d, lanes, st_d, plan=lambda t: DECODE_PLANS[t % len(DECODE_PLANS)])
    assert bitwise_equal(od, o) and state_equal(st_d, st_p, [1, 2])
    assert bitwise_equal(prefill(d), o)


def test_strided_views_of_a_fused_buffer():
    """q/k/v as column views of one packed [T, (2hk + hv) * 128] buffer == contiguous copies (prefill and
    decode, with a strided ``out``)."""
    lanes, hk, hv = [70, 20], 2, 4
    d = make(lanes, hk, hv, seed=24)
    T = sum(lanes)
    mixed = torch.cat([d["q"].flatten(1), d["k"].flatten(1), d["v"].flatten(1)], 1).contiguous()
    qv = mixed[:, : hk * 128].unflatten(1, (hk, 128))
    kv = mixed[:, hk * 128: 2 * hk * 128].unflatten(1, (hk, 128))
    vv = mixed[:, 2 * hk * 128:].unflatten(1, (hv, 128))
    assert not qv.is_contiguous()
    st1, st2 = fresh(3, hv), fresh(3, hv)
    sl = ids([1, 2])
    o1 = G.prefill(qv, kv, vv, d["a"], d["b"], d["A_log"], d["dt_bias"], d["cut"], state=st1, slot_in=sl)
    o2 = prefill(d, state=st2, slot_in=sl)
    assert bitwise_equal(o1, o2) and state_equal(st1, st2, [1, 2])
    big = torch.zeros(2, hv + 2, 128, device=DEV, dtype=BF)
    out = big[:, 1: hv + 1]   # strided output rows
    r = torch.tensor([10, T - 1], device=DEV)
    G.decode(qv[r], kv[r], vv[r], d["a"][r], d["b"][r], d["A_log"], d["dt_bias"], st1, sl, out=out)
    o4 = G.decode(*rows(d, r), d["A_log"], d["dt_bias"], st2, sl)
    assert bitwise_equal(out, o4) and state_equal(st1, st2, [1, 2])


def test_bf16_parameters_are_read_natively():
    """bf16 A_log/dt_bias == their exact fp32 widening, bit for bit (prefill and decode)."""
    lanes = [40, 70]
    d = make(lanes, 2, 4, seed=25)
    d16 = dict(d, A_log=d["A_log"].to(BF), dt_bias=d["dt_bias"].to(BF))
    d32 = dict(d, A_log=d16["A_log"].float(), dt_bias=d16["dt_bias"].float())
    st1, st2 = fresh(3, 4), fresh(3, 4)
    assert bitwise_equal(prefill(d16, state=st1, slot_in=ids([1, 2])), prefill(d32, state=st2, slot_in=ids([1, 2])))
    r = torch.tensor([0, 50], device=DEV)
    o1 = G.decode(*rows(d16, r), d16["A_log"], d16["dt_bias"], st1, ids([1, 2]))
    o2 = G.decode(*rows(d32, r), d32["A_log"], d32["dt_bias"], st2, ids([1, 2]))
    assert bitwise_equal(o1, o2) and state_equal(st1, st2, [1, 2])


def test_degenerate_shapes():
    d = make([0], 2, 4, seed=26)
    assert prefill(d).shape == (0, 4, 128)
    st = fresh(2, 4)
    assert G.decode(*rows(d, slice(0, 0)), d["A_log"], d["dt_bias"], st, ids([])).shape == (0, 4, 128)
    d1 = make([1], 2, 4, seed=27)
    st1, st2 = fresh(2, 4), fresh(2, 4)
    assert bitwise_equal(prefill(d1, state=st1, slot_in=ids([1])),
                         G.decode(*rows(d1, slice(0, 1)), d1["A_log"], d1["dt_bias"], st2, ids([1])))
    assert state_equal(st1, st2, [1])


@pytest.mark.parametrize("split", [None, 1, 4, 8])
def test_decode_cuda_graph_replay_is_bitwise_eager(split):
    """Captured steps (NULL lanes included, crossing a chunk boundary) replay bit for bit like eager."""
    hv, n, steps = 4, 8, 4
    d = make([steps] * n, 2, hv, seed=28)
    st = fresh(n + 1, hv)
    st.pos[1:] = 62   # the second step completes a chunk: the resync runs inside the graph
    st.set_S(slice(1, None), torch.randn(n, hv, 128, 128, device=DEV) * 0.05)
    st0 = st.data.clone()
    sl = ids(list(range(1, n - 1)) + [-1, -1])
    q, k, v, a, b = (x.view(steps, n, *x.shape[1:]) for x in rows(d, torch.arange(steps * n, device=DEV)))
    for x in (q, k, v, a, b):
        x[:, -2:] = float("nan")   # NULL lanes carry garbage
    outs = torch.empty(steps, n, hv, 128, device=DEV, dtype=BF)

    def run():
        for t in range(steps):
            G.decode(q[t], k[t], v[t], a[t], b[t], d["A_log"], d["dt_bias"], st, sl, out=outs[t], split=split)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            st.data.copy_(st0)
            run()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    st.data.copy_(st0)
    graph.replay()
    torch.cuda.synchronize()
    g_out, g_state = outs.clone(), st.data.clone()
    st.data.copy_(st0)
    outs.zero_()
    run()
    torch.cuda.synchronize()
    assert bitwise_equal(outs[:, : n - 2], g_out[:, : n - 2]) and bitwise_equal(st.data, g_state)
    assert torch.isfinite(g_out[:, : n - 2].float()).all()
