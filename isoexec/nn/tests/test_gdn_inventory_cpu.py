"""Ordered GDN tensor and state-spec parity against the schema declarations. Cases cover
segmented q/k/v convolution axes and TP-summed output-gamma gradients.
"""

from __future__ import annotations

import pytest

from isoexec.nn import inventory
from isoexec.nn.gdn import GDN
from isoexec.schema import col, gdn_state_specs, gdn_vocabulary, replicated, row

from isoexec.nn.tests import _gdn_fixtures as F

# Geometry tuples exercise unequal key/value head counts and widths,
# checking segmented convolution axes and nonsquare recurrent state.
GEOMETRIES = [
    pytest.param(4, 8, 32, 32, 4, id="qwen_tiny_hv_2x_hk"),
    pytest.param(4, 4, 32, 32, 4, id="hv_equals_hk"),
    pytest.param(1, 4, 64, 64, 4, id="one_key_head"),
    pytest.param(2, 8, 64, 128, 4, id="wide_value_head"),
    pytest.param(4, 8, 32, 64, 3, id="odd_kernel_wide_value"),
    pytest.param(16, 32, 128, 128, 4, id="production_shaped"),
]


def _spec(hk, hv, dk, dv, w):
    return F.spec(key_heads=hk, value_heads=hv, key_head_dim=dk, value_head_dim=dv, conv_kernel=w)


@pytest.mark.parametrize(("hk", "hv", "dk", "dv", "w"), GEOMETRIES)
def test_inventory_equals_the_schema_vocabulary(hk, hv, dk, dv, w):
    spec = _spec(hk, hv, dk, dv, w)
    got = list(inventory(F.component(spec, device="meta"), prefix=F.SITE).values())
    want = list(gdn_vocabulary(F.NS, spec, F.HIDDEN))
    assert got == want, f"\n got: {got}\nwant: {want}"


def test_the_comparison_is_ordered_and_field_for_field():
    """Guard the guard: the rows carry the axis, and an axis field difference has to show."""
    spec = F.spec()
    rows = list(inventory(F.component(spec, device="meta"), prefix=F.SITE).values())
    assert [t.name.rsplit(".", 1)[1] for t in rows] == [
        "q", "k", "v", "z", "beta", "alpha", "conv_w", "A_log", "dt_bias", "out_norm", "o",
    ]
    by_role = {t.name.rsplit(".", 1)[1]: t for t in rows}
    kd, vd = F.KEY_HEADS * F.KEY_HEAD_DIM, F.VALUE_HEADS * F.VALUE_HEAD_DIM
    # The segmented axis: q and k segments count key heads, the v segment counts value heads.
    assert by_role["conv_w"].shard == col(segments=((kd, F.KEY_HEADS), (kd, F.KEY_HEADS), (vd, F.VALUE_HEADS)))
    assert by_role["conv_w"].shape == (2 * kd + vd, 1, F.CONV_KERNEL)
    assert by_role["conv_w"].shard.units is None  # a unit count here would be the wrong split
    assert by_role["q"].shard == col(units=F.KEY_HEADS) and by_role["v"].shard == col(units=F.VALUE_HEADS)
    assert by_role["out_norm"].shard == replicated(reduce="sum_over_tp")
    assert by_role["o"].shard == row() and by_role["o"].shape == (F.HIDDEN, vd)
    assert {t.dtype for t in rows} == {"bf16"}


@pytest.mark.parametrize(("hk", "hv", "dk", "dv", "w"), GEOMETRIES)
def test_state_specs_reproduce_the_schema_declaration(hk, hv, dk, dv, w):
    """Compare cache/state specifications field for field with the schema declaration."""
    spec = _spec(hk, hv, dk, dv, w)
    got, want = F.component(spec).state_specs(), gdn_state_specs(spec)
    assert got == want, f"\n got: {got}\nwant: {want}"
    assert [s.name for s in got] == ["conv_q", "conv_k", "conv_v", "ssm"]
    assert [s.dtype for s in got] == ["bf16", "bf16", "bf16", "fp32"]
    assert {s.reuse for s in got} == {"align"} and {s.granularity for s in got} == {"per_request"}
    assert got[3].shape == (hv, dk, dv)  # the ssm state is not square when dk != dv


def test_facts_record_the_effective_configuration():
    facts = F.component(F.spec()).facts()
    assert facts["kind"] == "gdn"
    assert (facts["key_heads"], facts["value_heads"]) == (F.KEY_HEADS, F.VALUE_HEADS)
    assert facts["key_dim"] == F.KEY_HEADS * F.KEY_HEAD_DIM  # derived, not restated
    assert facts["value_dim"] == F.VALUE_HEADS * F.VALUE_HEAD_DIM
    assert facts["out_norm"] == "gated_rms_zero_centered"
    assert facts["held_dtype"] == "bf16"  # family policy, not an argument
    assert "chunk_size" not in facts  # the kernel's pin is not a model fact (see nn/gdn.py)


def test_the_constructor_takes_no_chunk_size():
    """The one config field this refactor deletes. A caller that passes it is refused, not ignored."""
    with pytest.raises(TypeError, match="chunk_size"):
        GDN(
            hidden=F.HIDDEN,
            key_heads=F.KEY_HEADS,
            value_heads=F.VALUE_HEADS,
            key_head_dim=F.KEY_HEAD_DIM,
            value_head_dim=F.VALUE_HEAD_DIM,
            conv_kernel=F.CONV_KERNEL,
            chunk_size=64,
        )


def test_construction_is_global_shaped_on_meta_and_allocates_nothing():
    gdn = F.component(F.spec())
    assert {p.device.type for p in gdn.parameters()} == {"meta"}
    assert tuple(gdn.q.shape) == (F.KEY_HEADS * F.KEY_HEAD_DIM, F.HIDDEN)
    assert tuple(gdn.v.shape) == (F.VALUE_HEADS * F.VALUE_HEAD_DIM, F.HIDDEN)
    assert tuple(gdn.out_norm.shape) == (F.VALUE_HEAD_DIM,)
    assert gdn.site is None and gdn.placement is None


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"value_heads": 6}, "not a whole multiple"),
        ({"key_heads": 0}, "must be >= 1"),
        ({"conv_kernel": 0}, "must be >= 1"),
        ({"out_norm": "rms"}, "out_norm must be one of"),
    ],
)
def test_the_constructor_validates_its_own_domain(kwargs, match):
    base = {
        "hidden": F.HIDDEN,
        "key_heads": F.KEY_HEADS,
        "value_heads": F.VALUE_HEADS,
        "key_head_dim": F.KEY_HEAD_DIM,
        "value_head_dim": F.VALUE_HEAD_DIM,
        "conv_kernel": F.CONV_KERNEL,
    }
    with pytest.raises(ValueError, match=match):
        GDN(**{**base, **kwargs})


def test_the_out_norm_convention_and_the_out_norm_gamma_are_different_attributes():
    """Keep the held gamma parameter separate from the convention used to select its kernel."""
    gdn = F.component(F.spec())
    assert gdn.out_norm_convention == "gated_rms_zero_centered"
    assert tuple(gdn.out_norm.shape) == (F.VALUE_HEAD_DIM,) and gdn.out_norm.requires_grad
