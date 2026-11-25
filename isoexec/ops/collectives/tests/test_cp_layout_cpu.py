"""The zigzag context-parallel layout, checked against brute force and against SkyRL's cut."""

from __future__ import annotations

import pytest

from isoexec.ops.collectives.cp_layout import (
    CpLayoutError,
    gathered_to_global,
    local_positions,
    local_to_global,
    owner_and_local_index,
    prefix_entries,
)

CASES = [([0, 8], 2), ([0, 8, 24, 28], 2), ([0, 16, 32], 4), ([0, 4, 8, 12], 1), ([0, 24], 3)]


def _skyrl_cut(tokens: list[int], cp: int, rank: int) -> list[int]:
    """``preprocess_packed_seqs``'s rule for ONE sequence: chunk ``rank`` then chunk ``2cp-1-rank``."""
    half = len(tokens) // (2 * cp)
    return tokens[half * rank : half * (rank + 1)] + tokens[len(tokens) - half * (rank + 1) : len(tokens) - half * rank]


@pytest.mark.parametrize("cu,cp", CASES)
def test_the_local_stream_is_skyrls_zigzag_cut_and_the_ranks_tile_the_stream(cu, cp):
    seen: list[int] = []
    for rank in range(cp):
        want: list[int] = []
        for a, b in zip(cu, cu[1:]):
            want += _skyrl_cut(list(range(a, b)), cp, rank)
        assert local_to_global(cu, cp, rank) == want
        seen += want
    assert sorted(seen) == list(range(cu[-1]))


@pytest.mark.parametrize("cu,cp", CASES)
def test_positions_are_global_positions_inside_the_owning_sequence(cu, cp):
    starts = {g: a for a, b in zip(cu, cu[1:]) for g in range(a, b)}
    for rank in range(cp):
        globals_ = local_to_global(cu, cp, rank)
        assert local_positions(cu, cp, rank) == [g - starts[g] for g in globals_]


@pytest.mark.parametrize("cu,cp", CASES)
def test_the_gather_permutation_restores_the_packed_stream(cu, cp):
    gathered = [g for rank in range(cp) for g in local_to_global(cu, cp, rank)]  # what all-gather yields
    perm = gathered_to_global(cu, cp)
    assert [gathered[p] for p in perm] == list(range(cu[-1]))
    owner, index = owner_and_local_index(cu, cp)
    for g in range(cu[-1]):
        assert local_to_global(cu, cp, owner[g])[index[g]] == g


@pytest.mark.parametrize("cu,cp", CASES)
def test_every_query_sees_exactly_its_causal_prefix(cu, cp):
    """Bottom-right alignment: query ``i`` of an entry sees keys ``[kv_start, kv_start + kv_len - q_len + i]``,
    which must be the owning sequence from its first token up to and including the query itself."""
    starts = {g: a for a, b in zip(cu, cu[1:]) for g in range(a, b)}
    for rank in range(cp):
        globals_ = local_to_global(cu, cp, rank)
        covered = 0
        for entry in prefix_entries(cu, cp, rank):
            assert entry.q_start == covered
            covered += entry.q_len
            for i in range(entry.q_len):
                g = globals_[entry.q_start + i]
                last_key = entry.kv_start + entry.kv_len - entry.q_len + i
                assert entry.kv_start == starts[g] and last_key == g
        assert covered == len(globals_)


def test_a_sequence_that_cannot_be_cut_refuses():
    with pytest.raises(CpLayoutError, match="not a positive multiple of 2\\*cp=4"):
        local_to_global([0, 6], 2, 0)
    with pytest.raises(CpLayoutError, match="outside"):
        local_to_global([0, 8], 2, 2)
    with pytest.raises(CpLayoutError, match="start at 0"):
        local_to_global([4, 8], 2, 0)
