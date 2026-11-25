"""Expert data parallelism: which EP degrees tile a TP x DP mesh, and what the live mesh must show."""

from __future__ import annotations

import pytest

from isoexec.ops.moe.expert_parallel import ep_routing_census
from isoexec.runtimes import parallel as P


def test_ep_equal_to_the_dense_mesh_is_one_replica_per_shard():
    assert P.expert_data_parallel_degree(2, 2, 4) == 1
    assert P.expert_data_parallel_degree(2, 1, 2) == 1
    assert P.expert_data_parallel_degree(1, 4, 4) == 1
    assert P.expert_data_parallel_degree(2, 2, 1) == 1  # no expert parallelism: nothing to replicate
    assert P.expert_data_parallel_degree(1, 2, 4, cp=2) == 1  # CP ranks belong to the expert mesh too


@pytest.mark.parametrize("tp, dp, ep, cp, edp", [
    (2, 2, 2, 1, 2), (2, 4, 4, 1, 2), (1, 4, 2, 1, 2),
    (1, 8, 2, 1, 4), (2, 4, 2, 1, 4), (1, 16, 2, 1, 8), (2, 8, 2, 1, 8),  # EDP 4 and 8
    (4, 1, 2, 1, 2), (8, 1, 4, 1, 2), (2, 3, 3, 1, 2), (4, 2, 8, 1, 1),  # EP not a multiple of TP
    (1, 2, 2, 2, 2), (2, 2, 4, 2, 2),  # CP x EP
])
def test_any_ep_dividing_the_dense_mesh_is_admitted(tp, dp, ep, cp, edp):
    assert P.expert_data_parallel_degree(tp, dp, ep, cp) == edp


@pytest.mark.parametrize("tp, dp, ep, cp", [(2, 2, 3, 1), (2, 3, 4, 1), (2, 2, 8, 1), (1, 2, 8, 2)])
def test_an_ep_that_does_not_divide_the_dense_mesh_refuses(tp, dp, ep, cp):
    with pytest.raises(P.ShardError, match="must divide TP x CP x DP"):
        P.expert_data_parallel_degree(tp, dp, ep, cp)


def test_the_live_mesh_must_agree():
    assert P.expert_data_parallel_refusal(4, 4) is None
    assert P.expert_data_parallel_refusal(8, 8) is None
    assert P.expert_data_parallel_refusal(1, 1) is None
    assert "different mesh" in P.expert_data_parallel_refusal(2, 1)


def test_the_census_names_its_ep_group():
    line = ep_routing_census([3, 5], [2, 6], ep_rank=0, ep=2, emit=False, group="g2")
    assert line.endswith("routes_received=[2, 6] group=g2") or " group=g2" in line
    assert " group=" not in ep_routing_census([1, 1], [1, 1], ep_rank=0, ep=2, emit=False)
