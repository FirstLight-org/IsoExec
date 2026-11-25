"""Check live placement traversal through containers, safe descendant pruning, re-entry refusal and
deferred engine attachment.
"""

from __future__ import annotations

import pytest
from torch import nn

from isoexec.nn import Component, PlacementError, RMSNorm, place, resolved_ops, verify_placed
from isoexec.ops.table import OpTable

from isoexec.nn.tests import _fixtures as F


class _Nothing(Component):
    """A component with no slots and nothing to resolve: the walk is what is under test."""

    def facts(self):
        return {"kind": "nothing"}

    def _place(self, placement) -> None:
        del placement


class _Pruner(Component):
    """A component that drops one of its own children while it is being placed.

    The mixture does exactly this at EP > 1, for the experts another rank owns; this is the same
    shape of act in eight lines, so the walk's laziness is tested without an expert-parallel
    placement standing in the way.
    """

    def __init__(self) -> None:
        super().__init__()
        self.kept = _Nothing()
        self.dropped = _Nothing()

    def facts(self):
        return {"kind": "pruner"}

    def _place(self, placement) -> None:
        del placement
        del self._modules["dropped"]


def test_every_component_is_placed_under_its_own_module_path():
    model = nn.Module()
    model.norm = RMSNorm(hidden=F.HIDDEN, role="final")
    model.rows = nn.ModuleList([_Nothing(), _Nothing()])
    sites = place(model, F.placement(), op_table=OpTable(mode="reference"))
    assert sites == ("norm", "rows.0", "rows.1")
    assert [m.site for m in model.modules() if isinstance(m, Component)] == list(sites)


def test_the_walk_descends_through_containers_that_are_not_components():
    """Traverse ordinary containers to place their component descendants."""
    row = nn.Module()
    row.mixer = _Nothing()
    model = nn.Module()
    model.rows = nn.ModuleList([row])
    assert place(model, F.placement(), op_table=OpTable(mode="reference")) == ("rows.0.mixer",)


def test_a_component_pruned_while_the_walk_runs_is_never_placed():
    """The walk consumes ``named_modules()`` lazily, so "placed" and "held" stay the same set."""
    model = nn.Module()
    model.row = _Pruner()
    assert place(model, F.placement(), op_table=OpTable(mode="reference")) == ("row", "row.kept")
    assert "dropped" not in dict(model.named_modules())


def test_a_bare_component_root_is_refused_rather_than_placed_at_the_empty_site():
    with pytest.raises(PlacementError, match=r"prefix="):
        place(RMSNorm(hidden=F.HIDDEN), F.placement(), op_table=OpTable(mode="reference"))


def test_a_prefix_places_a_subtree_under_the_fqn_it_will_occupy():
    norm = RMSNorm(hidden=F.HIDDEN, role="pre_attn")
    assert place(norm, F.placement(), prefix=f"{F.NS}.pre_attn_norm", op_table=OpTable(mode="reference")) == (f"{F.NS}.pre_attn_norm",)
    assert list(resolved_ops(norm)) == [f"{F.NS}.pre_attn_norm"]


def test_the_walk_refuses_a_second_pass_over_the_same_tree():
    """The base class's re-entry refusal, reached through the walk: a second placement would
    resolve over already-derived activation geometry and silently halve it again."""
    model = nn.Module()
    model.norm = RMSNorm(hidden=F.HIDDEN, role="final")
    place(model, F.placement(), op_table=OpTable(mode="reference"))
    with pytest.raises(PlacementError, match="already ran"):
        place(model, F.placement(), op_table=OpTable(mode="reference"))


def test_placing_an_engine_model_leaves_its_cores_waiting_and_does_not_refuse():
    """Placement succeeds with resource-dependent engine cores awaiting adapter attachment."""
    model = nn.Module()
    model.mixer = F.component(F.spec(), device="meta")
    assert place(model, F.placement(side="engine"), op_table=OpTable(mode="reference")) == ("mixer",)
    assert list(verify_placed(model)["mixer"]) == ["core"]
