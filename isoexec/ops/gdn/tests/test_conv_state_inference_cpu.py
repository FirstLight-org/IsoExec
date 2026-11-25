"""Portable short-convolution state inferred from GDN metadata."""

from dataclasses import replace

import pytest

from isoexec.ops.gdn.conv_state import infer_conv_state_specs
from isoexec.schema import GDN, Attn, col
from isoexec.state.contract import StateSpec
from isoexec.state.validation import ContractError

SPEC = GDN(num_key_heads=4, num_value_heads=8, key_head_dim=16, value_head_dim=32, conv_kernel=4)


@pytest.mark.parametrize("width,taps", [(2, 1), (4, 3), (5, 4)])
def test_convolution_carries_only_previous_inputs(width, taps):
    specs = infer_conv_state_specs(replace(SPEC, conv_kernel=width))

    assert specs == (
        StateSpec("conv_q", (4, 16, taps), "bf16", col(dim=0), "per_request"),
        StateSpec("conv_k", (4, 16, taps), "bf16", col(dim=0), "per_request"),
        StateSpec("conv_v", (8, 32, taps), "bf16", col(dim=0), "per_request"),
    )


@pytest.mark.parametrize("width", [1, 0, -1, True, 4.0])
def test_conv_width_without_positive_integer_carried_taps_refuses(width):
    with pytest.raises(ContractError, match="conv_kernel"):
        infer_conv_state_specs(replace(SPEC, conv_kernel=width))


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
def test_convolution_rejects_invalid_dimensions(field, value):
    with pytest.raises(ContractError, match=field):
        infer_conv_state_specs(replace(SPEC, **{field: value}))


def test_convolution_state_does_not_depend_on_recurrence_chunk_size():
    assert infer_conv_state_specs(replace(SPEC, chunk_size=128)) == infer_conv_state_specs(SPEC)


def test_convolution_rejects_wrong_metadata_type():
    metadata = Attn(num_q_heads=8, num_kv_heads=4, head_dim=32, rope_theta=10000.0)
    with pytest.raises(ContractError, match="conv.short requires GDN metadata"):
        infer_conv_state_specs(metadata)
