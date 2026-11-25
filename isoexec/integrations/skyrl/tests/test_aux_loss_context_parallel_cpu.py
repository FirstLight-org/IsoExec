"""Megatron's load-balancing objective under a CP=2 cut (two Gloo ranks on CPU).

Each CP rank holds the zigzag rows of every sequence. Its objective is its share of the global one:
expert counts and the token denominator are reduced over the CP group, probabilities stay local. So
every row's gradient equals the CP=1 row's bits, and the two shares add up to the CP=1 objective.
"""

from __future__ import annotations

import tempfile

import pytest
import torch

pytest.importorskip("megatron.core.transformer.moe.moe_utils")

CU = [0, 12, 20, 44]
EXPERTS, TOP_K, COEFFICIENT = 8, 2, 0.01
NAME = "lm.layers.0.mlp.router.scores_full"


def _inputs():
    g = torch.Generator().manual_seed(5)
    scores = torch.softmax(torch.randn(CU[-1], EXPERTS, generator=g), dim=-1)
    valid = torch.rand(CU[-1], generator=g) > 0.2
    return scores, valid


def _placement(cp: int, rank: int, group):
    from isoexec.schema import Axis, Placement

    one = Axis(1, 0, None)
    return Placement(tp=one, ep=one, dp=one, pp=one, etp=one, cp=Axis(cp, rank, group if cp > 1 else None),
                     side="trainer", leaves=8)


def _objective(placement, scores, valid):
    from isoexec.integrations.skyrl.aux_loss import _native_aux_loss

    loss = _native_aux_loss([(NAME, EXPERTS, TOP_K)], placement, COEFFICIENT)
    leaf = scores.clone().requires_grad_()
    value, metrics = loss({NAME: leaf}, valid.float(), torch.tensor(CU))
    value.backward()
    return value.detach(), leaf.grad, float(metrics["moe_aux_loss"])


def _body(rank, world, out, init_method):
    import torch.distributed as dist

    from isoexec.ops.collectives.cp_layout import local_to_global

    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=world)
    scores, valid = _inputs()
    full_value, full_grad, _ = _objective(_placement(1, 0, None), scores, valid)
    mine = torch.tensor(local_to_global(CU, world, rank))
    value, grad, logged = _objective(_placement(world, rank, dist.group.WORLD), scores[mine], valid[mine])
    shares = [torch.zeros(()) for _ in range(world)]
    dist.all_gather(shares, value)
    out[rank] = {
        "row_gradients_equal_cp1": torch.equal(grad, full_grad[mine]),
        "shares_sum_to_cp1": torch.allclose(sum(shares), full_value, rtol=1e-6, atol=0),
        "logged_is_global": abs(logged - float(full_value) / COEFFICIENT) <= 1e-5 * abs(float(full_value) / COEFFICIENT),
    }
    dist.destroy_process_group()


@pytest.fixture(scope="module")
def results():
    from isoexec.ops.collectives.tests._gloo_harness import run_world

    with tempfile.TemporaryDirectory() as tmp:
        return run_world(_body, world=2, rendezvous=f"{tmp}/rdzv", timeout_s=240)


@pytest.mark.parametrize("claim", ["row_gradients_equal_cp1", "shares_sum_to_cp1", "logged_is_global"])
def test_cp2_aux_objective_is_the_cp1_objective_cut_by_rows(results, claim):
    from isoexec.ops.collectives.tests._gloo_harness import assert_claim

    assert_claim(results, claim, 2)
