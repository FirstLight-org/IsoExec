"""Probability placement and training fallback for opt-in gathered FC1."""

from types import SimpleNamespace

import pytest
import torch

from isoexec.ops.moe.carriers import RoutingResult
from isoexec.ops.moe.moe_sm90 import integration as m


@pytest.mark.parametrize("after,defer", [(False, False), (True, False), (True, True)])
def test_gather_probability_placement(monkeypatch, after, defer):
    monkeypatch.setattr(m, "_AFTER_FC2_GATHER", True)
    x = torch.ones(2, 2048, dtype=torch.bfloat16)
    w = torch.empty(4, 128, 2048, dtype=x.dtype)
    d = torch.empty(4, 2048, 64, dtype=x.dtype)
    p = torch.tensor([[0.25], [0.5]])
    idx = torch.tensor([[1], [2]], dtype=torch.int32)
    routing = RoutingResult(idx, p, None)
    plan = object()
    sorted_idx = torch.tensor([0, 1])
    rows = torch.tensor([[0], [1]], dtype=torch.int32)
    calls = []
    monkeypatch.setattr(m.ops, "_can_gather_fc1", lambda *a: True)
    monkeypatch.setattr(m.ops, "small_route_plan", lambda *a: (plan, sorted_idx, p.flatten(), rows))

    def fc1(*a):
        calls.append(a[5])
        return torch.ones(2, 64, dtype=x.dtype)

    monkeypatch.setattr(m.ops, "_fc1_from_routes", fc1)
    root = torch.full((2, 2048), 3.125, dtype=torch.float32)
    monkeypatch.setattr(m.ops, "fc2", lambda *a, **k: root)
    monkeypatch.setattr(m, "_compact_root", lambda *a: root)
    with torch.no_grad():
        out = m.expert_mlp(
            x, routing, w, d, n_leaves=1, compact_root=True, weight_after_fc2=after, defer_weight_after_fc2=defer
        )
    assert (calls[0] is None) == after
    if after and not defer:
        expected = (root.to(x.dtype) * p.to(x.dtype)).float()
    else:
        expected = root
    assert torch.equal(out.rows, expected)


def test_training_cannot_select_gather(monkeypatch):
    monkeypatch.setattr(
        m.ops,
        "_can_gather_fc1",
        lambda *a: (_ for _ in ()).throw(AssertionError("gather predicate evaluated under grad")),
    )
    from isoexec.ops.moe.moe_sm90 import moe_fused_permute as perm

    x = torch.ones(2, 2048, dtype=torch.bfloat16, requires_grad=True)
    p = torch.ones(2, 1, requires_grad=True)
    idx = torch.zeros(2, 1, dtype=torch.int32)
    w = torch.ones(1, 128, 2048, dtype=x.dtype)
    d = torch.ones(1, 2048, 64, dtype=x.dtype)
    monkeypatch.setattr(
        perm, "route_to_permuted_rows", lambda *a: (x, p.flatten(), torch.tensor([2]), torch.arange(2), idx)
    )
    monkeypatch.setattr(m.ops, "make_plan_from_counts", lambda *a: None)
    monkeypatch.setattr(m, "_Experts", SimpleNamespace(apply=lambda *a: x.float()))
    for after in (False, True):
        out = m.expert_mlp(x, RoutingResult(idx, p, None), w, d, weight_after_fc2=after)
        assert out.rows.requires_grad


def test_opt_in_disabled_keeps_after_fc2_fallback(monkeypatch):
    from isoexec.ops.moe.moe_sm90 import moe_fused_permute as perm

    monkeypatch.setattr(m, "_AFTER_FC2_GATHER", False)
    monkeypatch.setattr(
        m.ops, "_can_gather_fc1", lambda *a: (_ for _ in ()).throw(AssertionError("disabled opt-in inspected gather"))
    )
    x = torch.ones(2, 2048, dtype=torch.bfloat16)
    p = torch.tensor([[0.25], [0.5]])
    idx = torch.zeros(2, 1, dtype=torch.int32)
    w = torch.empty(1, 128, 2048, dtype=x.dtype)
    d = torch.empty(1, 2048, 64, dtype=x.dtype)
    monkeypatch.setattr(
        perm, "route_to_permuted_rows", lambda *a: (x, p.flatten(), torch.tensor([2]), torch.arange(2), idx)
    )
    monkeypatch.setattr(m.ops, "make_plan_from_counts", lambda *a: None)

    def fc1(*a):
        assert a[3] is None
        return torch.ones(2, 64, dtype=x.dtype)

    monkeypatch.setattr(m.ops, "fc1_glu", fc1)
    monkeypatch.setattr(m.ops, "fc2", lambda *a, **k: torch.full((2, 2048), 2.0))
    with torch.no_grad():
        out = m.expert_mlp(x, RoutingResult(idx, p, None), w, d, weight_after_fc2=True)
    assert torch.equal(out.rows, 2 * p.expand(2, 2048))
