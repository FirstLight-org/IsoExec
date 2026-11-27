"""Claim (d): batch composition, packing, slot placement, slot moves, NULL lanes, empty sequences and
poisoned pools do not enter the bits."""

from __future__ import annotations

import pytest
import torch

from isoexec.ops.gdn.gdn_cpr_sm100 import ops as G

from ._fixtures import SM100, biteq, decode_step, fresh, ids, make, prefill, rows, state_equal

pytestmark = SM100


def test_slot_placement_is_independent():
    d = make([37, 64, 130, 1], 4, 8, seed=13)
    base_slots = ids([1, 2, 3, 4])
    base_state = fresh(9, 8)
    want = prefill(d, state=base_state, slot_in=base_slots)
    for placement in ([5, 6, 7, 8], [8, 7, 6, 5], [4, 1, 3, 2]):
        st = fresh(9, 8)
        got = prefill(d, state=st, slot_in=ids(placement))
        assert biteq(want, got), placement
        assert state_equal(base_state, st, base_slots, ids(placement)), placement


def _pb_rows_past_their_block(st, slots):
    """The pb elements of the rows the record defines that lie at or past the row block's own columns."""
    parts = []
    for s in list(slots):
        j = int(st.pos[int(s), 0, 0]) % 64
        if j >= 16:
            parts.append(st.pb[int(s), :, : j % 16, 16 * (j // 16):].reshape(-1))
    return torch.cat(parts).view(torch.int32) if parts else torch.zeros(0, dtype=torch.int32, device="cuda")


@pytest.mark.parametrize("plan", [None, 3, G.PREPARED], ids=["rule", "one_launch", "prepared"])
def test_a_record_out_of_a_zeroed_pool_is_a_function_of_the_calls(plan):
    """EVERY byte of a pool, and so the 16 x 64 `pb` section too, repeats when the calls repeat.

    A P row of block B has 16 B columns and is stored 64 wide. The columns past 16 B used to be copied
    out of shared memory nobody had written, so `pb` differed between two runs on identical inputs while
    every output and every other section agreed. Nothing reads those columns (`state_equal` never
    compared them); they are stored as +0 now, which is asserted directly as well as through the repeat,
    after a prefill, after decode steps in place and after a decode step that moves every slot.
    """
    hk, hv = 4, 8
    lanes = [17, 37, 63, 95, 126, 191, 64, 5]          # open-chunk offsets in every row block, and none
    d = make(lanes, hk, hv, seed=71)
    step = make([1] * len(lanes), hk, hv, seed=72)
    slots, moved = ids(range(1, 9)), ids(range(9, 17))
    lane = list(range(len(lanes)))

    def run():
        st = fresh(17, hv)
        snaps, outs = [], [prefill(d, state=st, slot_in=ids([-1] * 8), slot_out=slots, plan=plan)]
        snaps.append((st.data.clone(), _pb_rows_past_their_block(st, slots)))
        for _ in range(3):
            outs.append(decode_step(step, lane, st, slots))
        snaps.append((st.data.clone(), _pb_rows_past_their_block(st, slots)))
        outs.append(decode_step(step, lane, st, slots, slot_out=moved))
        snaps.append((st.data.clone(), _pb_rows_past_their_block(st, moved)))
        return outs, snaps

    outs, snaps = run()
    for phase, (_, past) in zip(("prefill", "decode in place", "decode slot move"), snaps):
        assert past.numel() > 0, phase
        assert int((past != 0).sum()) == 0, f"{phase}: pb columns past the row block are not +0"
    for _ in range(3):
        prefill(make([200, 64, 13], 16, 32, seed=73))   # other work on the same SMs between the repeats
        outs2, snaps2 = run()
        assert all(biteq(a, b) for a, b in zip(outs, outs2))
        for phase, (a, _), (b, _) in zip(("prefill", "decode in place", "decode slot move"), snaps, snaps2):
            assert biteq(a, b), f"{phase}: the pool's bytes differ between two identical runs"


def test_a_sequence_alone_equals_its_rows_inside_the_pack():
    d = make([37, 64, 130, 1], 4, 8, seed=13)
    packed = prefill(d, state=fresh(9, 8), slot_in=ids([1, 2, 3, 4]))
    for i, L in enumerate(d["lanes"]):
        s, e = d["cu"][i], d["cu"][i + 1]
        alone = prefill(d, slice(s, e), cu=[0, L], state=fresh(2, 8), slot_in=ids([1]))
        assert biteq(packed[s:e], alone), i


def test_lane_order_within_a_decode_step_is_independent():
    d = make([1] * 6, 4, 8, seed=19)
    st = fresh(8, 8)
    want = decode_step(d, range(6), st, ids(range(1, 7)))
    perm = [5, 0, 3, 1, 4, 2]
    st2 = fresh(8, 8)
    got = decode_step(d, perm, st2, ids([i + 1 for i in perm]))
    for a, b in enumerate(perm):
        assert biteq(want[b], got[a])


def test_empty_sequences_and_a_zero_token_pack():
    d = make([0, 64, 0, 33], 2, 4, seed=17)
    o = prefill(d, state=fresh(6, 4), slot_in=ids([1, 2, 3, 4]))
    assert torch.isfinite(o).all()
    # a stateful T = 0 pack still moves its sequences' slots
    st = fresh(6, 4)
    d1 = make([64], 2, 4, seed=19)
    prefill(d1, state=st, slot_in=ids([1]))
    before_S, before_pos = st.S[1].clone(), st.pos[1].clone()
    empty = make([0], 2, 4, seed=19)
    prefill(empty, state=st, slot_in=ids([1]), slot_out=ids([5]))
    assert biteq(before_S, st.S[5]) and biteq(before_pos, st.pos[5])


def test_a_sequence_moves_between_slots():
    d = make([130], 4, 8, seed=23)
    one = fresh(4, 8)
    whole = prefill(d, state=one, slot_in=ids([1]))
    moving = fresh(4, 8)
    p1 = prefill(d, slice(0, 70), cu=[0, 70], state=moving, slot_in=ids([1]), slot_out=ids([2]))
    p2 = prefill(d, slice(70, 130), cu=[0, 60], state=moving, slot_in=ids([2]), slot_out=ids([3]))
    assert biteq(whole, torch.cat([p1, p2]))
    assert state_equal(one, moving, ids([1]), ids([3]))


def test_a_fork_two_sequences_read_one_slot():
    d0 = make([70], 4, 8, seed=31)
    st = fresh(6, 8)
    prefill(d0, state=st, slot_in=ids([1]))
    d = make([20, 20], 4, 8, seed=32)
    forked = prefill(d, state=st, slot_in=ids([1, 1]), slot_out=ids([2, 3]))
    # each branch alone from the same input slot must agree with its rows in the fork
    for i, (s, e) in enumerate(((0, 20), (20, 40))):
        st_i = fresh(6, 8)
        prefill(d0, state=st_i, slot_in=ids([1]))
        alone = prefill(d, slice(s, e), cu=[0, 20], state=st_i, slot_in=ids([1]), slot_out=ids([4]))
        assert biteq(forked[s:e], alone), i


def test_null_lanes_and_nan_neighbours_do_not_leak():
    d = make([1, 1, 1], 4, 8, seed=41)
    q, k, v, a, b = rows(d, slice(None))
    q, a = q.clone(), a.clone()
    q[1] = float("nan")
    a[1] = float("nan")
    mixed = G.decode(q, k, v, a, b, d["A_log"], d["dt_bias"], fresh(4, 8), ids([1, -1, 2]))
    keep = torch.tensor([0, 2], device="cuda")
    clean = G.decode(q[keep], k[keep], v[keep], a[keep], b[keep], d["A_log"], d["dt_bias"], fresh(4, 8),
                     ids([1, 2]))
    assert biteq(mixed[keep], clean)


def test_a_poisoned_pool_is_not_read_past_the_open_chunk():
    """Every byte 0xFF (NaN floats, pos = -1). A fresh NULL input slot must ignore all of it."""
    d = make([70, 33], 4, 8, seed=47)
    clean = prefill(d, state=fresh(4, 8), slot_in=ids([-1, -1]), slot_out=ids([1, 2]))
    poisoned = fresh(4, 8, poison=True)
    got = prefill(d, state=poisoned, slot_in=ids([-1, -1]), slot_out=ids([1, 2]))
    assert biteq(clean, got)
