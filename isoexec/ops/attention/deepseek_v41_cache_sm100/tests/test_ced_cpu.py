"""CED decoder entry points, host side: the boundary rule and the query-row contract (``ced.QueryRows``)."""
import pytest
import torch

from isoexec.ops.attention.deepseek_v41_cache_sm100 import ced
from isoexec.ops.attention.deepseek_v41_cache_sm100.tests import _harness as h


def test_boundary_is_the_last_ratio1_kv_source():
    cfg = h.native_cfg()
    ced.check_boundary(cfg, 4)
    for i in (0, 1, 3, 5):
        with pytest.raises(ValueError):
            ced.check_boundary(cfg, i)
    assert ced._decoder_layer(cfg, 6) == 4
    with pytest.raises(ValueError):
        ced._decoder_layer(cfg, 3)


def test_query_rows_groups_and_window_contiguity():
    rows = ced.QueryRows.make([0, 0, 0, 1, 1], [5, 6, 7, 0, 1], floor=[5, 5, 5, 0, 0], device="cpu")
    assert rows.window_contiguous and len(rows) == 5
    assert rows.groups == ((0, slice(0, 3)), (1, slice(3, 5)))
    assert rows.window_end.tolist() == [0, 1, 2, 3, 4]
    # a segment row whose predecessor (position p - 1 >= floor) is not the previous row
    assert not ced.QueryRows.make([0, 0, 1, 0], [5, 6, 0, 7], floor=[5, 5, 0, 5], device="cpu").window_contiguous
    assert not ced.QueryRows.make([0, 0], [6, 7], floor=[5, 5], device="cpu").window_contiguous
    # two segments of one sequence (multi-turn): the second restarts at its own floor
    two = ced.QueryRows.make([0, 0, 0, 0], [3, 4, 2, 3], floor=[3, 3, 2, 2], device="cpu")
    assert two.window_contiguous and two.groups == ((0, slice(0, 4)),)
    with pytest.raises(ValueError):
        ced.QueryRows.make([0], [3], floor=[4], device="cpu")


def test_query_rows_from_decoder_rows_checks_the_window_index():
    from types import SimpleNamespace
    W = 4
    pos = torch.tensor([5, 6, 7, 0, 1])
    floor = torch.tensor([5, 5, 5, 0, 0])
    seq = torch.tensor([0, 0, 0, 1, 1])
    j = torch.arange(W)
    r = torch.arange(5)
    wi = torch.where(pos[:, None] - W + 1 + j >= floor[:, None], r[:, None] - (W - 1 - j), -1)
    rows = SimpleNamespace(positions=pos, sequence=seq, window_floor=floor, window_index=wi, index=None, cu_seqlens=None)
    q = ced.QueryRows.from_decoder_rows(rows, W)
    assert q.window_contiguous and q.window_end.tolist() == [0, 1, 2, 3, 4] and q.floor.tolist() == floor.tolist()
    rows.window_index = wi.flip(0)
    with pytest.raises(ValueError):
        ced.QueryRows.from_decoder_rows(rows, W)
    # self-consistent slots that reach into another sequence's rows
    rows.window_index, rows.positions = wi, torch.tensor([5, 6, 7, 2, 3])
    rows.window_index = torch.where(rows.positions[:, None] - W + 1 + j >= floor[:, None], r[:, None] - (W - 1 - j), -1)
    with pytest.raises(ValueError):
        ced.QueryRows.from_decoder_rows(rows, W)
