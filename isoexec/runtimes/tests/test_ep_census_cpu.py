"""CPU tests for optional expert-routing counters and their reported split vectors."""

import os
import re

import pytest

from isoexec.ops.moe import expert_parallel as P


@pytest.fixture(autouse=True)
def _reset_census(monkeypatch):
    monkeypatch.setitem(P._EP_CENSUS, "calls", 0)
    monkeypatch.setitem(P._EP_CENSUS, "routes_sent", None)
    monkeypatch.setitem(P._EP_CENSUS, "routes_received", None)
    monkeypatch.delenv(P._EP_CENSUS_ENV, raising=False)


LINE = re.compile(
    r"^\[ISOEXEC-MOE-EP\] CENSUS pid=(\d+) ep_rank=(\d+) ep=(\d+) calls=(\d+) "
    r"routes_sent=\[([0-9, ]*)\] routes_received=\[([0-9, ]*)\]$"
)


def test_census_accumulates_and_is_silent_without_the_evidence_sink(capsys):
    line = P.ep_routing_census([3, 5], [2, 6], ep_rank=0, ep=2)
    assert LINE.match(line)
    assert capsys.readouterr().out == ""
    P.ep_routing_census([1, 0], [0, 4], ep_rank=0, ep=2)
    state = P.ep_routing_census_state()
    assert state == {"calls": 2, "routes_sent": [4, 5], "routes_received": [2, 10]}
    assert P.ep_routing_census_state() is not state  # a copy


def test_census_prints_one_line_per_call_under_the_evidence_sink(capsys, monkeypatch, tmp_path):
    monkeypatch.setenv(P._EP_CENSUS_ENV, str(tmp_path / "evidence"))
    P.ep_routing_census([7, 0], [3, 3], ep_rank=1, ep=2)
    P.ep_routing_census([0, 2], [1, 0], ep_rank=1, ep=2)
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 2
    m = LINE.match(out[-1])
    assert m and int(m.group(1)) == os.getpid()
    assert (m.group(2), m.group(3), m.group(4)) == ("1", "2", "2")
    assert [int(v) for v in m.group(5).split(",")] == [7, 2]
    assert [int(v) for v in m.group(6).split(",")] == [4, 3]


def test_census_refuses_split_vectors_that_disagree_with_ep():
    with pytest.raises(P.ShardError, match="EP census"):
        P.ep_routing_census([1, 2, 3], [1, 2], ep_rank=0, ep=2)


def test_explicit_emit_overrides_the_environment(capsys):
    P.ep_routing_census([1, 1], [1, 1], ep_rank=0, ep=2, emit=True)
    assert "[ISOEXEC-MOE-EP] CENSUS" in capsys.readouterr().out
