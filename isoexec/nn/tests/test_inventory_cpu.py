"""Ordered attention inventory parity against the schema vocabulary, covering names, shapes,
dtypes and shard axes.
"""

from __future__ import annotations

import pytest
import torch

from isoexec.nn import Component, RMSNorm, inventory
from isoexec.nn.inspect import InventoryError
from isoexec.schema import attention_state_specs, attention_vocabulary, replicated

from isoexec.nn.tests import _fixtures as F

CASES = [
    pytest.param({}, id="qwen_tiny"),
    pytest.param({"qk_norm": None}, id="no_qk_norm"),
    pytest.param({"out_gate": None}, id="no_out_gate"),
    pytest.param({"qkv_bias": True}, id="qkv_bias"),
    pytest.param({"qk_norm": None, "out_gate": None, "qkv_bias": True}, id="bare_plus_bias"),
    pytest.param({"heads": 8, "kv_heads": 1}, id="mqa_8x1"),
]


@pytest.mark.parametrize("kwargs", CASES)
def test_inventory_equals_the_schema_vocabulary(kwargs):
    row = F.spec(**kwargs)
    got = list(inventory(F.component(row, device="meta"), prefix=F.SITE).values())
    want = list(attention_vocabulary(F.NS, row, F.HIDDEN))
    assert got == want, f"\n got: {got}\nwant: {want}"


def test_the_comparison_is_ordered_and_field_for_field():
    """Guard the guard: the rows carry the axis, and an axis field difference has to show."""
    row = F.spec()
    rows = list(inventory(F.component(row, device="meta"), prefix=F.SITE).values())
    assert [t.name.rsplit(".", 1)[1] for t in rows] == ["q", "k", "v", "gate", "q_norm", "k_norm", "o"]
    by_role = {t.name.rsplit(".", 1)[1]: t for t in rows}
    assert by_role["k"].shard.units == F.KV_HEADS and by_role["q"].shard.units is None
    assert by_role["q_norm"].shard == replicated(reduce="sum_over_tp")
    assert by_role["o"].shard.kind == "row" and by_role["o"].shard.dim == 1
    assert {t.dtype for t in rows} == {"bf16"}


@pytest.mark.parametrize("kwargs", CASES)
def test_state_specs_reproduce_the_schema_declaration(kwargs):
    """Compare cache/state specifications field for field with the schema declaration."""
    row = F.spec(**kwargs)
    assert F.component(row).state_specs() == attention_state_specs(row)


def test_a_norm_declares_no_state():
    assert RMSNorm(hidden=F.HIDDEN).state_specs() == ()


def test_facts_record_the_effective_configuration():
    facts = F.component(F.spec(qkv_bias=True, out_gate=None, mrope=True)).facts()
    assert facts["rotary_dim"] == int(F.HEAD_DIM * F.ROTARY_FRACTION)  # derived, not restated
    assert (facts["qkv_bias"], facts["out_gate"], facts["mrope_interleaved"]) == (True, None, True)
    assert facts["held_dtype"] == "bf16"  # family policy, not an argument


def test_construction_is_global_shaped_on_meta_and_allocates_nothing():
    attn = F.component(F.spec())
    assert {p.device.type for p in attn.parameters()} == {"meta"}
    assert tuple(attn.q.shape) == (F.HEADS * F.HEAD_DIM, F.HIDDEN)
    assert tuple(attn.k.shape) == (F.KV_HEADS * F.HEAD_DIM, F.HIDDEN)
    assert attn.site is None and attn.placement is None


def test_a_norm_declares_its_one_replicated_gamma():
    norm = RMSNorm(hidden=F.HIDDEN, eps=F.EPS, role="pre_attn")
    rows = list(inventory(norm, prefix=f"{F.NS}.pre_attn_norm").values())
    assert [(t.name, t.shape, t.dtype, t.shard) for t in rows] == [
        (f"{F.NS}.pre_attn_norm.weight", (F.HIDDEN,), "bf16", replicated())
    ]


def test_a_module_with_parameters_and_no_axes_is_refused_by_name():
    """The failure this walk exists to catch: an undeclared tensor is planned, not refused."""

    class Undeclared(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.stray = torch.nn.Parameter(torch.empty(4, dtype=torch.bfloat16, device="meta"))

    tree = torch.nn.Module()
    tree.add_module("odd_one", Undeclared())
    with pytest.raises(InventoryError, match=r"odd_one: Undeclared holds parameters \['stray'\]"):
        inventory(tree)


def test_an_axis_with_no_parameter_is_refused_too():
    """Both directions. A declaration nothing answers to is how a vocabulary drifts away."""
    attn = F.component(F.spec())
    attn.tensor_axes = lambda: {**type(attn).tensor_axes(attn), "ghost": replicated()}
    with pytest.raises(InventoryError, match=r"declared with no parameter: \['ghost'\]"):
        inventory(attn)


def test_ties_come_from_parameter_identity():
    """Two names for one Parameter object is what a tie is; nothing has to declare it."""
    row = F.spec()
    tree = torch.nn.Module()
    tree.add_module("a", F.component(row))
    tree.add_module("b", F.component(row))
    tree.b.q = tree.a.q  # the shipped shape of a tie: one object, two logical names
    inv = inventory(tree)
    assert inv.aliases == ("b.q",)
    assert [(t.logical, t.aliases, t.owner) for t in inv.ties] == [("a.q", ("b.q",), "first")]
    assert "b.q" not in inv and "a.q" in inv


class _Forgetful(Component):
    """A component that registers a tensor and never overrides ``tensor_axes()``."""

    def __init__(self) -> None:
        super().__init__()
        self.stray = torch.nn.Parameter(torch.empty(4, dtype=torch.bfloat16, device="meta"))

    def facts(self):
        return {"kind": "forgetful"}

    def _place(self, placement) -> None:
        del placement


class _Silent(Component):
    """A component whose author never considered what it contributes to model identity."""

    def _place(self, placement) -> None:
        del placement


def test_a_composite_inherits_an_empty_tensor_axes_and_a_forgotten_tensor_is_still_refused():
    """An empty default axis mapping is valid only when the component registers no direct
    parameters.
    """
    assert dict(Component().tensor_axes()) == {}
    tree = torch.nn.Module()
    tree.add_module("row", _Forgetful())
    with pytest.raises(InventoryError, match=r"registered with no axis: \['stray'\]"):
        inventory(tree)


def test_facts_has_no_default_because_nothing_anywhere_can_check_it():
    """Require an explicit effective-configuration record, including an explicit empty record for
    composites.
    """
    with pytest.raises(NotImplementedError, match="contributes to model identity"):
        _Silent().facts()
