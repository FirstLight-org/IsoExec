"""Check that held tensors remain in inventory and sharding while execution walks skip unexecuted
modules and subtrees.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from isoexec.nn import (
    Component,
    RMSNorm,
    Unexecuted,
    inventory,
    is_unexecuted_subtree,
    place,
    resolved_ops,
    verify_placed,
)
from isoexec.ops.table import OpTable
from isoexec.schema import col, replicated

from isoexec.nn.tests import _fixtures as F


def _tree() -> nn.Module:
    model = nn.Module()
    model.proj = Unexecuted((8, 16), col())
    model.enorm = Unexecuted((8,), replicated())
    return model


def test_the_tensor_is_in_the_inventory_exactly_as_any_other():
    rows = inventory(_tree(), prefix="mtp.0")
    assert list(rows) == ["mtp.0.proj.weight", "mtp.0.enorm.weight"]
    assert rows["mtp.0.proj.weight"].shape == (8, 16)
    assert rows["mtp.0.proj.weight"].dtype == "bf16"
    assert rows["mtp.0.proj.weight"].shard == col()
    assert rows["mtp.0.enorm.weight"].shard == replicated()
    assert {p.device.type for p in _tree().parameters()} == {"meta"}


def test_it_is_not_a_component_so_no_site_claims_an_op_that_never_runs():
    """Wrapping it in ``RMSNorm`` would resolve a ``norms.rms.*`` at a site that never executes,
    and a peer comparison would read an op neither host computes."""
    tree = _tree()
    assert not isinstance(tree.proj, Component)
    assert place(tree, F.placement(), op_table=OpTable(mode="reference")) == ()
    assert resolved_ops(tree) == {} and verify_placed(tree) == {}


def test_calling_it_refuses_by_name():
    with pytest.raises(TypeError, match="held and not executed"):
        _tree().proj(torch.zeros(4, 16))


def test_a_degenerate_shape_is_refused_at_construction():
    with pytest.raises(ValueError, match="positive dims"):
        Unexecuted((8, 0), col())


def test_the_held_dtype_is_the_fleet_policy_and_can_be_stated():
    assert Unexecuted((4,), replicated()).weight.dtype is torch.bfloat16
    assert Unexecuted((4,), replicated(), dtype="fp32").weight.dtype is torch.float32




class _HeldRow(nn.Module):
    """A held, unexecuted subtree built from ordinary components; its root declares that execution
    walks skip all children.
    """

    def __init__(self) -> None:
        super().__init__()
        self.proj = Unexecuted((8, 16), col())
        self.norm = RMSNorm(hidden=8, device="meta")

    def unexecuted_subtree(self) -> bool:
        return True


def _mixed() -> nn.Module:
    model = nn.Module()
    model.lm = RMSNorm(hidden=8, device="meta")
    model.mtp = _HeldRow()
    return model


def test_a_subtree_that_declares_itself_unexecuted_is_placed_and_reported_by_nothing():
    """All three execution walks skip it, and they skip the same set -- which is the property that
    makes "every site the walk placed is a site the report covers" hold at all."""
    tree = _mixed()
    assert place(tree, F.placement(), op_table=OpTable(mode="reference")) == ("lm",)
    assert list(resolved_ops(tree)) == ["lm"]
    assert list(verify_placed(tree)) == ["lm"]
    assert is_unexecuted_subtree(tree.mtp) and not is_unexecuted_subtree(tree.lm)


def test_its_tensors_are_still_loaded_sharded_and_synced():
    """The split the skip does not make: an unexecuted tensor is still in the checkpoint, still has
    a logical name and a shard rule, and still travels on every weight sync."""
    rows = inventory(_mixed())
    assert list(rows) == ["lm.weight", "mtp.proj.weight", "mtp.norm.weight"]
    assert rows["mtp.proj.weight"].shard == col()


def test_an_unplaced_component_under_it_is_not_a_refusal():
    """All execution walks skip components under an unexecuted root consistently."""
    tree = _mixed()
    place(tree, F.placement(), op_table=OpTable(mode="reference"))
    assert tree.mtp.norm.site is None and tree.lm.site == "lm"
    assert resolved_ops(tree)["lm"]["norm"].op == "norms.rms.pre_attn"


def test_a_module_that_never_heard_of_the_question_is_walked_as_before():
    """Duck-typed on the method, so the default is the safe one: a module that does not answer is
    placed and reported exactly as it always was."""
    plain = nn.Module()
    plain.lm = RMSNorm(hidden=8, device="meta")
    plain.mtp = nn.Module()
    plain.mtp.norm = RMSNorm(hidden=8, device="meta")
    assert not is_unexecuted_subtree(plain.mtp)
    assert set(place(plain, F.placement(), op_table=OpTable(mode="reference"))) == {"lm", "mtp.norm"}
