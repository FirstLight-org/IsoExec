"""Host-independent facts for one authored GDN invocation."""

from dataclasses import dataclass

from isoexec.state.validation import ContractError


@dataclass(frozen=True)
class GDNOpFacts:
    """Model-owned geometry required to project one aggregate GDN state."""

    num_key_heads: int
    num_value_heads: int
    key_head_dim: int
    value_head_dim: int
    conv_kernel: int


def infer_gdn_op_facts(spec: object) -> GDNOpFacts:
    """Snapshot the model-owned geometry required by GDN host projection."""
    for field in ("num_key_heads", "num_value_heads", "key_head_dim", "value_head_dim"):
        value = getattr(spec, field, None)
        if type(value) is not int or value <= 0:
            raise ContractError(f"gdn.core requires {field} to be a positive integer")
    if spec.num_value_heads % spec.num_key_heads:
        raise ContractError("gdn.core requires num_value_heads to be divisible by num_key_heads")
    conv_kernel = getattr(spec, "conv_kernel", None)
    if type(conv_kernel) is not int or conv_kernel < 2:
        raise ContractError("gdn.core requires conv_kernel to be an integer >= 2")
    return GDNOpFacts(
        num_key_heads=spec.num_key_heads,
        num_value_heads=spec.num_value_heads,
        key_head_dim=spec.key_head_dim,
        value_head_dim=spec.value_head_dim,
        conv_kernel=spec.conv_kernel,
    )


def validate_gdn_op_facts(facts: GDNOpFacts) -> None:
    if not isinstance(facts, GDNOpFacts):
        raise ContractError("gdn.core facts must be GDNOpFacts")
    for field in ("num_key_heads", "num_value_heads", "key_head_dim", "value_head_dim"):
        value = getattr(facts, field)
        if type(value) is not int or value <= 0:
            raise ContractError(f"gdn.core facts require {field} to be a positive integer")
    if facts.num_value_heads % facts.num_key_heads:
        raise ContractError("gdn.core facts require num_value_heads to be divisible by num_key_heads")
    if type(facts.conv_kernel) is not int or facts.conv_kernel < 2:
        raise ContractError("gdn.core facts require conv_kernel to be an integer >= 2")


__all__ = ["GDNOpFacts", "infer_gdn_op_facts", "validate_gdn_op_facts"]
