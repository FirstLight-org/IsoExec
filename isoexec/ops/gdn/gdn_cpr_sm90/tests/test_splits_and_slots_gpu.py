"""Chunked prefill split anywhere, slot moves, NULL slots, empty sequences, slot reuse and poisoned pools.

Every equality claim is compared on integer views (``ops._testing.bitwise_equal``). Claim map:
tests/README.md.
"""

from __future__ import annotations

import random

import pytest
import torch

from isoexec.ops._testing import bitwise_equal
from isoexec.ops.gdn.gdn_cpr_sm90 import ops as G
from isoexec.ops.gdn.gdn_cpr_sm90.tests._fixtures import (
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

HK, HV, TOTAL = 2, 6, 300


def _split_patterns():
    rng = random.Random(0)
    pats = [[64, 128], [1, 2, 3, 63, 65], [63], [65], [128, 192]]
    pats += [sorted(rng.sample(range(1, TOTAL), rng.randint(1, 8))) for _ in range(10)]
    return pats


@pytest.fixture(scope="module")
def whole_seq():
    d = make([TOTAL], HK, HV, seed=3)
    st = fresh(2, HV)
    one = ids([1])
    st.load_initial(one, torch.randn(1, HV, 128, 128, device=DEV) * 0.05)
    init = st.data.clone()
    o = prefill(d, state=st, slot_in=one)
    return d, init, o, st


@pytest.mark.parametrize("cuts", _split_patterns(), ids=lambda c: "-".join(map(str, c)))
def test_prefill_split_anywhere_is_unsplit(whole_seq, cuts):
    """Split points anywhere (mid-chunk included); one-token pieces alternately by decode."""
    d, init, o_full, st_full = whole_seq
    st = G.CprState(init.clone())
    one = ids([1])
    parts, pos = [], 0
    for n, L in enumerate(c - p for c, p in zip(cuts + [TOTAL], [0] + cuts)):
        sl = slice(pos, pos + L)
        if L == 1 and n % 2:
            parts.append(G.decode(*rows(d, sl), d["A_log"], d["dt_bias"], st, one))
        else:
            parts.append(prefill(d, sl, cu=[0, L], state=st, slot_in=one))
        pos += L
    assert bitwise_equal(torch.cat(parts), o_full)
    assert state_equal(st, st_full, [1])


def test_slot_moves_and_empty_sequences(whole_seq):
    """slot_in != slot_out moves a record (prefill and decode); zero-length sequences in a pack are inert
    and a zero-length sequence with distinct slots moves its record unchanged."""
    d, init, o_full, st_full = whole_seq
    st = fresh(6, HV)
    st.data[1] = init[1]
    o1 = prefill(d, slice(0, 100), cu=[0, 100], state=st, slot_in=ids([1]), slot_out=ids([2]))
    o2 = G.decode(*rows(d, slice(100, 101)), d["A_log"], d["dt_bias"], st, ids([2]), slot_out=ids([3]))
    # an empty sequence moving 3 -> 4, a NULL empty sequence, and the continuation 4 -> 5 next call
    prefill(d, slice(101, 106), cu=[0, 0, 0, 5], state=st, slot_in=ids([3, -1, -1]), slot_out=ids([4, -1, -1]))
    o3 = prefill(d, slice(101, TOTAL), cu=[0, 0, TOTAL - 101, TOTAL - 101], state=st,
                 slot_in=ids([-1, 4, -1]), slot_out=ids([-1, 5, -1]))
    assert bitwise_equal(torch.cat([o1, o2, o3]), o_full)
    assert state_equal(st, st_full, [5], [1])


def _null_pack(nan_null):
    lanes = [40, 70, 9, 66]
    d = make(lanes, HK, HV, seed=7)
    if nan_null:   # lanes 1 and 3 are NULL: poison their inputs
        for i in (1, 3):
            r = slice(d["cu"][i], d["cu"][i + 1])
            for n in ("q", "k", "v", "a", "b"):
                d[n][r] = float("nan")
    return lanes, d


@pytest.mark.parametrize("nan_null", [False, True])
def test_null_slots_are_inert_in_prefill(nan_null):
    lanes, d = _null_pack(nan_null)
    st = fresh(5, HV)
    st.data[3:5].random_(0, 256)   # records a NULL lane could touch hold arbitrary bytes
    before = st.data.clone()
    o = prefill(d, state=st, slot_in=ids([1, -1, 2, -1]), slot_out=ids([1, -1, 2, -1]))
    assert bitwise_equal(st.data[0], before[0]) and bitwise_equal(st.data[3:], before[3:])
    # the real lanes alone
    ref = fresh(3, HV)
    sel = torch.cat([torch.arange(d["cu"][i], d["cu"][i + 1], device=DEV) for i in (0, 2)])
    o_alone = prefill(d, sel, cu=[0, lanes[0], lanes[0] + lanes[2]], state=ref, slot_in=ids([1, 2]))
    assert bitwise_equal(o[sel], o_alone) and state_equal(st, ref, [1, 2])
    assert torch.isfinite(o[sel].float()).all()


@pytest.mark.parametrize("split", DECODE_PLANS)
def test_null_slots_are_inert_in_decode(split):
    lanes, d = _null_pack(True)
    st = fresh(5, HV)
    st.data[3:5].random_(0, 256)
    before = st.data.clone()
    n = len(lanes)
    ref = fresh(3, HV)
    for t in range(9):
        r = torch.tensor([d["cu"][i] + t for i in range(n)], device=DEV)
        o = G.decode(*rows(d, r), d["A_log"], d["dt_bias"], st, ids([1, -1, 2, -1]), split=split)
        o_ref = G.decode(*rows(d, r[[0, 2]]), d["A_log"], d["dt_bias"], ref, ids([1, 2]), split=split)
        assert bitwise_equal(o[[0, 2]], o_ref)
    assert bitwise_equal(st.data[0], before[0]) and bitwise_equal(st.data[3:], before[3:])
    assert state_equal(st, ref, [1, 2])


@pytest.mark.parametrize("restart", ["reset", "null_prefill", "null_prefill_64", "null_decode"])
@pytest.mark.parametrize("poison", ["nan_sequence", "garbage_pool"])
def test_slot_reuse_after_poison(restart, poison):
    """A slot that held a sequence with NaN inputs, or a never-initialized (0xFF) pool, restarted by
    reset() or by a NULL input slot, carries nothing into the new sequence -- under every decode plan."""
    L = 64 if restart == "null_prefill_64" else 77
    d = make([L + 70], HK, HV, seed=11)
    clean = fresh(2, HV)
    o_ref = decode_stream(d, [L + 70], clean) if restart == "null_decode" else None
    if o_ref is None:
        o_ref = torch.cat([prefill(d, slice(0, L), cu=[0, L], state=clean, slot_in=ids([1])),
                           decode_stream(d, [L + 70], clean, start=[L])[L:]])
    st = fresh(2, HV, poison=poison == "garbage_pool")
    if poison == "nan_sequence":
        bad = make([150], HK, HV, seed=12)
        bad["v"][5:140:7] = float("nan")
        bad["k"][100] = float("inf")
        prefill(bad, slice(0, 100), cu=[0, 100], state=st, slot_in=ids([1]))
        decode_stream(bad, [150], st, start=[100])
        assert not torch.isfinite(st.S[1]).all()
    if restart == "reset":
        st.reset(ids([1]))
        head = prefill(d, slice(0, L), cu=[0, L], state=st, slot_in=ids([1]))
    elif restart.startswith("null_prefill"):
        head = prefill(d, slice(0, L), cu=[0, L], state=st, slot_in=ids([-1]), slot_out=ids([1]))
    else:
        head = G.decode(*rows(d, slice(0, 1)), d["A_log"], d["dt_bias"], st, ids([-1]), slot_out=ids([1]))
        L = 1
    plans = DECODE_PLANS
    tail = decode_stream(d, [L + 70] if restart != "null_decode" else [d["q"].size(0)], st, start=[L],
                         plan=lambda t: plans[t % len(plans)])[L:]
    assert bitwise_equal(torch.cat([head, tail]), o_ref)
    assert state_equal(st, clean, [1])


def test_empty_pack_moves_slots():
    """A stateful prefill whose pack has no tokens still moves every record from slot_in to slot_out."""
    d = make([10, 70], HK, HV, seed=11)
    st = fresh(4, HV)
    prefill(d, state=st, slot_in=ids([1, 3]))   # slot 3 holds a different sequence
    before = G.CprState(st.data.clone())
    assert not state_equal(before, before, [1], [2]) and not state_equal(before, before, [1], [3])
    e = make([0, 0], HK, HV, seed=12)
    prefill(e, state=st, slot_in=ids([1, -1]), slot_out=ids([2, 3]))   # a move, and a fresh start
    assert state_equal(st, before, [2], [1]), "slot 2 must now hold slot 1's record"
    assert state_equal(st, before, [1], [1]), "the input slot is left as it was"
    fresh_ref = fresh(1, HV)
    assert state_equal(st, fresh_ref, [3], [0]), "a NULL input slot moved into slot 3 is the fresh record"
