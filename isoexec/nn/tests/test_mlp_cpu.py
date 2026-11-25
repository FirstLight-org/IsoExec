"""Ordered dense-MLP inventory against the schema vocabulary. Gate/up remain separate projections
and down uses the fixed leaf tree.
"""

from __future__ import annotations

import pytest

from isoexec.nn import inventory
from isoexec.nn.mlp import MLP
from isoexec.schema import col, dense_vocabulary, row

from isoexec.nn.tests import _gdn_fixtures as F

WIDTHS = [pytest.param(w, id=f"intermediate_{w}") for w in (8, 128, 512, 1536)]


@pytest.mark.parametrize("width", WIDTHS)
def test_inventory_equals_the_schema_vocabulary(width):
    spec = F.mlp_spec(intermediate=width)
    got = list(inventory(F.mlp_component(spec, device="meta"), prefix=F.MLP_SITE).values())
    want = list(dense_vocabulary(F.NS, spec, F.HIDDEN))
    assert got == want, f"\n got: {got}\nwant: {want}"


def test_the_comparison_is_ordered_and_field_for_field():
    rows = list(inventory(F.mlp_component(F.mlp_spec()), prefix=F.MLP_SITE).values())
    assert [t.name.rsplit(".", 1)[1] for t in rows] == ["gate", "up", "down"]
    by_role = {t.name.rsplit(".", 1)[1]: t for t in rows}
    assert by_role["gate"].shard == col() and by_role["gate"].shard.units is None
    assert by_role["up"].shape == (F.INTERMEDIATE, F.HIDDEN)
    assert by_role["down"].shard == row() and by_role["down"].shape == (F.HIDDEN, F.INTERMEDIATE)
    assert {t.dtype for t in rows} == {"bf16"}


def test_a_dense_mlp_declares_no_state_and_records_its_configuration():
    mlp = F.mlp_component(F.mlp_spec())
    assert mlp.state_specs() == ()
    assert mlp.facts() == {
        "kind": "mlp",
        "hidden": F.HIDDEN,
        "intermediate": F.INTERMEDIATE,
        "activation": "silu_mul",
        "held_dtype": "bf16",
    }


def test_construction_is_global_shaped_on_meta_and_allocates_nothing():
    mlp = F.mlp_component(F.mlp_spec())
    assert {p.device.type for p in mlp.parameters()} == {"meta"}
    assert mlp.site is None and mlp.placement is None


def test_the_constructor_validates_its_own_domain():
    with pytest.raises(ValueError, match="must be >= 1"):
        MLP(hidden=F.HIDDEN, intermediate=0)
    with pytest.raises(ValueError, match="activation must be one of"):
        MLP(hidden=F.HIDDEN, intermediate=8, activation="gelu_mul")




