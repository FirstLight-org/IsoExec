"""Portable recurrent state, independent of convolution and host storage."""

from dataclasses import replace

import pytest

from isoexec.ops.gdn.state import infer_gdn_state_specs
from isoexec.schema import GDN, Attn, col
from isoexec.state.contract import StateSpec
from isoexec.state.validation import ContractError

SPEC = GDN(num_key_heads=4, num_value_heads=8, key_head_dim=16, value_head_dim=32, conv_kernel=4)


@pytest.mark.parametrize(
    "metadata,shape",
    [
        (SPEC, (8, 16, 32)),
        (replace(SPEC, num_key_heads=2, num_value_heads=2, key_head_dim=8, value_head_dim=16), (2, 8, 16)),
    ],
)
def test_gdn_core_derives_only_fp32_recurrent_state(metadata, shape):
    assert infer_gdn_state_specs(metadata) == (StateSpec("ssm", shape, "fp32", col(dim=0), "per_request"),)


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_key_heads", 0),
        ("num_key_heads", True),
        ("num_value_heads", -1),
        ("num_value_heads", 8.0),
        ("key_head_dim", 0),
        ("key_head_dim", False),
        ("value_head_dim", -1),
        ("value_head_dim", "32"),
    ],
)
def test_gdn_core_rejects_invalid_dimensions(field, value):
    with pytest.raises(ContractError, match=field):
        infer_gdn_state_specs(replace(SPEC, **{field: value}))


def test_gdn_core_requires_value_heads_to_group_over_key_heads():
    with pytest.raises(ContractError, match="num_value_heads.*divisible by num_key_heads"):
        infer_gdn_state_specs(replace(SPEC, num_key_heads=3))


def test_recurrent_state_shape_does_not_depend_on_convolution_or_chunk_size():
    changed = replace(SPEC, conv_kernel=7, chunk_size=128, num_key_heads=2)
    assert infer_gdn_state_specs(changed) == infer_gdn_state_specs(SPEC)


def test_gdn_core_rejects_wrong_metadata_type():
    metadata = Attn(num_q_heads=8, num_kv_heads=4, head_dim=32, rope_theta=10000.0)
    with pytest.raises(ContractError, match="gdn.core requires GDN metadata"):
        infer_gdn_state_specs(metadata)
