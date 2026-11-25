"""Portable state inference for the GDN component's short convolution."""

from isoexec.schema import GDN, col
from isoexec.schema.op_metadata import GDNMetadata
from isoexec.state.contract import StateSpec
from isoexec.state.validation import ContractError


def infer_conv_state_specs(spec: GDN | GDNMetadata) -> tuple[StateSpec, ...]:
    """Describe the Q/K/V input histories carried between convolution calls."""
    if not isinstance(spec, (GDN, GDNMetadata)):
        raise ContractError("conv.short requires GDN metadata")
    for field in ("num_key_heads", "num_value_heads", "key_head_dim", "value_head_dim"):
        value = getattr(spec, field)
        if type(value) is not int or value <= 0:
            raise ContractError(f"conv.short requires {field} to be a positive integer")
    if type(spec.conv_kernel) is not int or spec.conv_kernel < 2:
        raise ContractError(
            "conv.short requires conv_kernel to be an integer >= 2: zero-sized carried state is unsupported"
        )
    taps = spec.conv_kernel - 1
    return (
        StateSpec("conv_q", (spec.num_key_heads, spec.key_head_dim, taps), "bf16", col(dim=0), "per_request"),
        StateSpec("conv_k", (spec.num_key_heads, spec.key_head_dim, taps), "bf16", col(dim=0), "per_request"),
        StateSpec("conv_v", (spec.num_value_heads, spec.value_head_dim, taps), "bf16", col(dim=0), "per_request"),
    )
