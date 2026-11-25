"""Portable state inference for the GDN recurrence."""

from isoexec.schema import GDN, col
from isoexec.schema.op_metadata import GDNMetadata
from isoexec.state.contract import StateSpec
from isoexec.state.validation import ContractError

from .conv_state import infer_conv_state_specs


def infer_gdn_state_specs(spec: GDN | GDNMetadata) -> tuple[StateSpec, ...]:
    """Describe legacy recurrence state or one production GDN component's aggregate state."""
    if not isinstance(spec, (GDN, GDNMetadata)):
        raise ContractError("gdn.core requires GDN metadata")
    for field in ("num_key_heads", "num_value_heads", "key_head_dim", "value_head_dim"):
        value = getattr(spec, field)
        if type(value) is not int or value <= 0:
            raise ContractError(f"gdn.core requires {field} to be a positive integer")
    if spec.num_value_heads % spec.num_key_heads:
        raise ContractError("gdn.core requires num_value_heads to be divisible by num_key_heads")
    recurrent = (
        StateSpec(
            "ssm", (spec.num_value_heads, spec.key_head_dim, spec.value_head_dim), "fp32", col(dim=0), "per_request"
        ),
    )
    if isinstance(spec, GDNMetadata):
        return (*infer_conv_state_specs(spec), *recurrent)
    return recurrent
