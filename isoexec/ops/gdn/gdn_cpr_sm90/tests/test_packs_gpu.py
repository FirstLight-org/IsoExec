"""Packs: many sequences (past the output units' k-major limit, sweep roles grouped), prefill continuing from poisoned
records, strided a/b, hv/hk = 3, a fork (two sequences continuing one input slot), and the workspace
sized from the host copy of cu_seqlens -- each bitwise equal to an independent evaluation.

Claim map: tests/README.md.
"""

from __future__ import annotations

import random

import pytest
import torch

from isoexec.ops._testing import bitwise_equal
from isoexec.ops.gdn.gdn_cpr_sm90 import ops as G
from isoexec.ops.gdn.gdn_cpr_sm90.tests._fixtures import DEV, SM90, decode_stream, fresh, ids, make, prefill, state_equal
from isoexec.ops.gdn.gdn_cpr_sm90.tests.test_poisoned_records_gpu import _poison

pytestmark = SM90


def _alone(d, n, **kw):
    """Sequence n of pack d as a pack of its own."""
    lo, hi = d["cu"][n], d["cu"][n + 1]
    return prefill(d, torch.arange(lo, hi, device=DEV), cu=[0, hi - lo], **kw)


@pytest.mark.parametrize("mode", ["stateless", "continue", "fresh"])
@pytest.mark.parametrize("host_cu", [False, True])
@pytest.mark.parametrize("one_group", [False, True])
@pytest.mark.parametrize("nseq", [40, 70])
def test_many_sequences_equal_each_alone(mode, host_cu, one_group, nseq):
    """40 or 70 sequences (within and past the k-major output order's 64; empty ones included) --
    stateless, continuing from records at varied positions, or fresh (no input slot) into poisoned output
    slots: every output row and every record equals the sequence prefilled alone. The pack's sweep roles
    hold several sequences each (many short sequences are grouped), or with `one_group` all of them."""
    rng = random.Random(9)
    lanes = [rng.choice([0, 1, 13, 64, 65, 150]) for _ in range(nseq)]
    hk, hv = 2, 4
    d = make(lanes, hk, hv, seed=21)
    n = len(lanes)
    kw, solo = {}, None
    if mode != "stateless":
        st = fresh(2 * n + 1, hv)
        if mode == "continue":
            pre = make([rng.randrange(0, 130) for _ in range(n)], hk, hv, seed=22)
            prefill(pre, state=st, slot_in=ids(range(1, n + 1)))   # varied open positions
            slot_in = ids(range(1, n + 1))
        else:
            slot_in = ids([-1] * n)
        solo = G.CprState(st.data.clone())
        st.data[n + 1:].fill_(0xFF)   # the pack's output records start as NaN: every meaningful byte is written
        kw = dict(state=st, slot_in=slot_in, slot_out=ids(range(n + 1, 2 * n + 1)))
    if host_cu:
        kw["cu_seqlens_cpu"] = torch.tensor(d["cu"], dtype=torch.int32)
    with G._sweep_schedule(one_group=one_group):
        out = prefill(d, **kw)
    for i, L in enumerate(lanes):
        if solo is not None:
            o = _alone(d, i, state=solo, slot_in=ids([kw["slot_in"][i].item()]), slot_out=ids([n + 1 + i]))
            assert state_equal(st, solo, [n + 1 + i]), f"record of sequence {i}"
        else:
            o = _alone(d, i)
        assert bitwise_equal(o, out[d["cu"][i]:d["cu"][i + 1]]), f"sequence {i} (length {L})"


def test_prefill_continues_from_a_poisoned_record():
    """A continuation reads only its record's meaningful rows: with every other byte of the input record set
    to NaN, a prefill continuing it equals the unpoisoned continuation."""
    lanes, hk, hv = [100, 37], 2, 4
    d = make(lanes, hk, hv, seed=31)
    first = [40, 20]   # the first pieces' lengths (the rest follows)
    st = fresh(3, hv)
    idx1 = torch.cat([torch.arange(0, 40), torch.arange(100, 120)]).to(DEV)
    idx2 = torch.cat([torch.arange(40, 100), torch.arange(120, 137)]).to(DEV)
    prefill(d, idx1, cu=[0, 40, 60], state=st, slot_in=ids([1, 2]))
    clean = G.CprState(st.data.clone())
    for slot, done in ((1, first[0]), (2, first[1])):
        _poison(st, slot, done % 64)
    got = prefill(d, idx2, cu=[0, 60, 77], state=st, slot_in=ids([1, 2]))
    want = prefill(d, idx2, cu=[0, 60, 77], state=clean, slot_in=ids([1, 2]))
    assert bitwise_equal(got, want)
    assert state_equal(st, clean, [1, 2])


def test_strided_gates_equal_contiguous():
    """a and b may be strided views (e.g. columns of a fused projection); the bits do not change."""
    lanes, hk, hv = [70, 5, 130], 2, 4
    d = make(lanes, hk, hv, seed=41)
    T = sum(lanes)
    wide = torch.randn(T, 3 * hv, device=DEV).to(torch.bfloat16)
    wide[:, hv:2 * hv], wide[:, 2 * hv:] = d["a"], d["b"]
    a_view, b_view = wide[:, hv:2 * hv], wide[:, 2 * hv:]
    assert not a_view.is_contiguous()
    want = prefill(d)
    got = G.prefill(d["q"], d["k"], d["v"], a_view, b_view, d["A_log"], d["dt_bias"], d["cut"])
    assert bitwise_equal(got, want)


def test_three_value_heads_per_key_head():
    """hv / hk = 3 (Qwen3.6-27B): a prefill split mid-chunk and continued equals one prefill."""
    lanes, hk, hv = [150, 77], 2, 6
    d = make(lanes, hk, hv, seed=51)
    want = prefill(d)
    st = fresh(3, hv)
    idx1 = torch.cat([torch.arange(0, 70), torch.arange(150, 160)]).to(DEV)
    idx2 = torch.cat([torch.arange(70, 150), torch.arange(160, 227)]).to(DEV)
    o1 = prefill(d, idx1, cu=[0, 70, 80], state=st, slot_in=ids([-1, -1]), slot_out=ids([1, 2]))
    o2 = prefill(d, idx2, cu=[0, 80, 147], state=st, slot_in=ids([1, 2]))
    assert bitwise_equal(o1, want[idx1]) and bitwise_equal(o2, want[idx2])


def test_fork_two_sequences_continue_one_slot():
    """Two sequences of one pack may continue the same input slot (neither writes it): each equals its
    continuation computed alone."""
    hk, hv = 2, 4
    base = make([90], hk, hv, seed=61)
    st = fresh(4, hv)
    prefill(base, state=st, slot_in=ids([1]))
    d = make([50, 33], hk, hv, seed=62)
    solo = G.CprState(st.data.clone())
    out = prefill(d, state=st, slot_in=ids([1, 1]), slot_out=ids([2, 3]))
    for i, so in enumerate((2, 3)):
        o = _alone(d, i, state=solo, slot_in=ids([1]), slot_out=ids([so]))
        assert bitwise_equal(o, out[d["cu"][i]:d["cu"][i + 1]])
        assert state_equal(st, solo, [so])
    assert state_equal(st, solo, [1]), "the shared input slot is not written"


def test_host_cu_seqlens_must_describe_the_pack():
    d = make([10, 20], 2, 4, seed=71)
    with pytest.raises(ValueError, match="cu_seqlens_cpu"):
        prefill(d, cu_seqlens_cpu=torch.tensor([0, 10, 29]))


@pytest.mark.two_pass_only
@pytest.mark.parametrize("mode", ["stateless", "fresh", "continue", "trainer"])
@pytest.mark.parametrize("other_mode", ["fused", "prepared"])
def test_prefill_modes_agree(mode, other_mode):
    """The fused prefill (one CTA per (sequence, head), everything on chip) and the two passes give the same
    bits: outputs, every output record, and the trainer's saves (h, v_new, w, A, norms), on a pack of
    mixed lengths (empty, one token, mid-chunk, exact chunks, several chunks), hv / hk = 3."""
    rng = random.Random(17)
    lanes = [0, 1, 63, 64, 65, 200, 7, 300, 128, 13] + [rng.randrange(0, 260) for _ in range(14)]
    hk, hv = 2, 6
    d = make(lanes, hk, hv, seed=81)
    n = len(lanes)
    pre = make([rng.randrange(0, 130) for _ in range(n)], hk, hv, seed=82)
    runs = {}
    for m in ("two_pass", other_mode):
        with G._prefill_mode(m):
            if mode == "trainer":
                runs[m] = G.trainer_forward(d["q"], d["k"], d["v"], d["a"], d["b"], d["A_log"], d["dt_bias"], d["cut"],
                                             save_states=True)
                continue
            st = fresh(2 * n + 1, hv)
            if mode == "continue":
                prefill(pre, state=st, slot_in=ids(range(1, n + 1)))
                slot_in = ids(range(1, n + 1))
            else:
                slot_in = ids([-1] * n)
            kw = {} if mode == "stateless" else dict(state=st, slot_in=slot_in, slot_out=ids(range(n + 1, 2 * n + 1)))
            runs[m] = (prefill(d, **kw), st)
    if mode == "trainer":
        (o2, s2), (of, sf) = runs["two_pass"], runs[other_mode]
        assert bitwise_equal(o2, of)
        nt = sum((L + 63) // 64 for L in lanes)   # h holds the pack's chunks; the rest is allocation
        for name in s2:
            a, b = (s2[name][:nt], sf[name][:nt]) if name == "h" else (s2[name], sf[name])
            assert bitwise_equal(a, b), name
        return
    (o2, st2), (of, stf) = runs["two_pass"], runs[other_mode]
    assert bitwise_equal(o2, of)
    if mode != "stateless":
        assert state_equal(st2, stf, list(range(n + 1, 2 * n + 1)))


@pytest.mark.two_pass_only
def test_fused_many_sequences_with_slowly_decaying_state():
    """Several waves of sequence/head CTAs, mixed chunk tails, and state that survives many chunks.

    Both modes must agree on every trainer save. Decode independently checks the shared WY arithmetic,
    so changing both prefill implementations together cannot conceal a change to the identity.
    """
    lanes = [0, 1, 63, 64, 65, 160, 320, 1280] * 8
    d = make(lanes, 4, 8, seed=91)
    d["A_log"].fill_(-4.0)
    d["dt_bias"].fill_(-1.0)
    args = [d[name] for name in ("q", "k", "v", "a", "b", "A_log", "dt_bias", "cut")]
    runs = {}
    for mode in ("two_pass", "fused", "prepared"):
        with G._prefill_mode(mode):
            runs[mode] = G.trainer_forward(*args, save_states=True, cu_seqlens_cpu=torch.tensor(d["cu"]))
    o2, s2 = runs["two_pass"]
    nt = sum((length + 63) // 64 for length in lanes)
    for mode in ("fused", "prepared"):
        of, sf = runs[mode]
        assert bitwise_equal(o2, of), mode
        for name in s2:
            a, b = (s2[name][:nt], sf[name][:nt]) if name == "h" else (s2[name], sf[name])
            assert bitwise_equal(a, b), (mode, name)

    # Select two 320-token items, one late in the pack, and decode all five chunks independently.
    indices = torch.cat([torch.arange(d["cu"][n], d["cu"][n + 1], device=DEV) for n in (6, 62)])
    solo = {name: d[name][indices] for name in ("q", "k", "v", "a", "b")}
    solo.update(A_log=d["A_log"], dt_bias=d["dt_bias"], cu=[0, 320, 640])
    st = fresh(3, 8)
    od = decode_stream(solo, [320, 320], st, plan=lambda t: (1, 2, 4, 8)[t % 4])
    assert bitwise_equal(od, o2[indices])


@pytest.mark.two_pass_only
def test_full_chunk_specialization_matches_general_and_decode():
    """An exact host chunk bound selects the whole-chunk specialization; a loose bound
    evaluates the same pack through the general kernels, including every trainer save."""
    lanes = [0, 64, 128, 320, 1280] * 7
    d = make(lanes, 2, 6, seed=113)
    d["A_log"].fill_(-4.0)
    d["dt_bias"].fill_(-1.0)
    args = [d[name] for name in ("q", "k", "v", "a", "b", "A_log", "dt_bias", "cut")]
    host_cu = torch.tensor(d["cu"], dtype=torch.int32)
    nt = sum(lanes) // 64
    reference = None
    for mode in ("two_pass", "fused", "prepared"):
        with G._prefill_mode(mode):
            general, general_saves = G.trainer_forward(*args, save_states=True)
            whole, whole_saves = G.trainer_forward(*args, save_states=True, cu_seqlens_cpu=host_cu)
        assert bitwise_equal(general, whole), mode
        for name in general_saves:
            a, b = general_saves[name], whole_saves[name]
            if name == "h":
                a, b = a[:nt], b[:nt]
            assert bitwise_equal(a, b), (mode, name)
        if reference is None:
            reference = whole
        else:
            assert bitwise_equal(reference, whole), mode

    # A nonzero state survives all five chunks; decode independently checks the identity.
    n = 3
    lo, hi = d["cu"][n:n + 2]
    solo = {**d, **{name: d[name][lo:hi] for name in ("q", "k", "v", "a", "b")}, "cu": [0, hi - lo]}
    decoded = decode_stream(solo, [hi - lo], fresh(2, 6), plan=4)
    assert bitwise_equal(reference[lo:hi], decoded)
