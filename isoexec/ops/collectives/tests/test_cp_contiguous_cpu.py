"""The contiguous context-parallel layout: shards tile the stream, every message has a matching end.

Every rank derives its plan from the global boundaries alone, so the properties below are what makes
the point-to-point hand-offs matched by construction at any degree.
"""

from __future__ import annotations

import random

import pytest

from isoexec.ops.collectives.cp_contiguous import ContiguousPlan, CpPlanError, kv_transfers, state_link


def _packs(seed: int, cp: int, count: int = 40):
    rng = random.Random(seed)
    for _ in range(count):
        n = rng.randint(1, 7)
        lengths = [cp * rng.randint(1, 9) for _ in range(n)]
        if rng.random() < 0.3:  # one sequence spanning many ranks
            lengths[rng.randrange(n)] = cp * rng.randint(10, 40)
        cu = [0]
        for length in lengths:
            cu.append(cu[-1] + length)
        yield cu


def _sequence_of(cu, row):
    for n in range(len(cu) - 1):
        if cu[n] <= row < cu[n + 1]:
            return n
    raise AssertionError(row)


@pytest.mark.parametrize("cp", [1, 2, 3, 4, 8])
def test_pieces_tile_the_stream_with_in_sequence_positions(cp):
    for cu in _packs(cp, cp):
        rows = []
        for rank in range(cp):
            plan = ContiguousPlan.build(cu, cp, rank)
            assert plan.local * cp == cu[-1]
            assert plan.local_cu[0] == 0 and plan.local_cu[-1] == plan.local
            assert [p.length for p in plan.pieces] == [b - a for a, b in zip(plan.local_cu, plan.local_cu[1:])]
            for piece in plan.pieces:
                assert piece.start - cu[piece.sequence] == piece.offset
                rows.extend(range(piece.start, piece.stop))
            positions = plan.positions()
            assert positions == [r - cu[_sequence_of(cu, r)] for r in range(plan.lo, plan.lo + plan.local)]
            assert plan.max_q == max(p.length for p in plan.pieces)
        assert rows == list(range(cu[-1]))


@pytest.mark.parametrize("cp", [2, 3, 4, 8])
def test_kv_prefix_transfers_cover_the_first_sequence_in_key_order(cp):
    for cu in _packs(10 + cp, cp):
        transfers = kv_transfers(cu, cp)
        for rank in range(cp):
            plan = ContiguousPlan.build(cu, cp, rank)
            first = plan.pieces[0]
            # The prefix is exactly the earlier rows of the first local sequence, in ascending order.
            got = []
            for t in plan.kv_recv:
                src_lo = t.src * plan.local
                got.extend(range(src_lo + t.src_start, src_lo + t.src_start + t.rows))
                assert t.dst_start == len(got) - t.rows
            assert got == list(range(cu[first.sequence], plan.lo))
            assert plan.prefix == first.offset
            # Every send is somebody's receive, and vice versa.
            assert set(plan.kv_send) == {t for t in transfers if t.src == rank}
            assert set(plan.kv_recv) == {t for t in transfers if t.dst == rank}
            assert all(t.src_start + t.rows == plan.local for t in plan.kv_send)
            # Key boundaries: the first piece sees its prefix, the others start fresh.
            assert plan.key_cu[1] == first.offset + first.length
            assert [b - a for a, b in zip(plan.key_cu[1:], plan.key_cu[2:])] == [p.length for p in plan.pieces[1:]]
            assert plan.max_k == max(b - a for a, b in zip(plan.key_cu, plan.key_cu[1:]))


@pytest.mark.parametrize("cp", [2, 3, 4, 8])
@pytest.mark.parametrize("width", [1, 2, 4, 6])
def test_state_links_and_conv_halos_agree_across_each_boundary(cp, width):
    for cu in _packs(20 + cp, cp):
        plans = [ContiguousPlan.build(cu, cp, rank) for rank in range(cp)]
        assert not plans[0].recv_prev and not plans[-1].send_next
        for lower, upper in zip(plans, plans[1:]):
            boundary = upper.lo
            interior = any(cu[n] < boundary < cu[n + 1] for n in range(len(cu) - 1))
            assert lower.send_next == upper.recv_prev == interior
            assert (state_link(cu, cp, upper.rank) is not None) == interior
            h_out, h_in = lower.halo(width)[1], upper.halo(width)[0]
            assert h_out == h_in
            if interior:
                n = _sequence_of(cu, boundary)
                assert h_in == min(width - 1, boundary - cu[n])
                # The tail a rank sends lies inside what it holds of that sequence plus its own halo.
                assert h_out <= lower.halo(width)[0] + lower.pieces[-1].length
            else:
                assert h_in == 0
            assert upper.halo_cu(width)[1] == upper.local_cu[1] + h_in


def test_refusals():
    with pytest.raises(CpPlanError, match="multiple of cp"):
        ContiguousPlan.build([0, 6, 9], 2, 0)
    with pytest.raises(CpPlanError, match="rank"):
        ContiguousPlan.build([0, 8], 2, 2)
    with pytest.raises(CpPlanError, match="start at 0"):
        ContiguousPlan.build([1, 8], 1, 0)


def test_empty_sequences_are_skipped():
    plan = ContiguousPlan.build([0, 4, 4, 8], 2, 1)
    assert [p.sequence for p in plan.pieces] == [2]
    assert not plan.recv_prev and plan.prefix == 0
