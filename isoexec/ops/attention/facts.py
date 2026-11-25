"""Host-independent facts for stateful attention invocations."""

from dataclasses import dataclass
from math import isfinite

from isoexec.state.validation import ContractError

from ._metadata import validate_gqa_metadata, validate_mla_metadata


@dataclass(frozen=True)
class GQAOpFacts:
    """Facts required to construct a native GQA attention operation."""

    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    v_head_dim: int
    scale: float


@dataclass(frozen=True)
class MLAOpFacts:
    """Facts required to construct a native MLA attention operation."""

    num_q_heads: int
    kv_lora_rank: int
    qk_rope_head_dim: int
    scale: float


def infer_gqa_op_facts(spec: object) -> GQAOpFacts:
    """Snapshot the authoring metadata needed by native GQA constructors."""
    validate_gqa_metadata(spec)
    return GQAOpFacts(
        num_q_heads=spec.num_q_heads,
        num_kv_heads=spec.num_kv_heads,
        head_dim=spec.head_dim,
        v_head_dim=spec.head_dim,
        scale=spec.scale,
    )


def infer_mla_op_facts(spec: object) -> MLAOpFacts:
    """Snapshot the authoring metadata needed by native MLA constructors."""
    validate_mla_metadata(spec)
    return MLAOpFacts(
        num_q_heads=spec.num_q_heads,
        kv_lora_rank=spec.kv_lora_rank,
        qk_rope_head_dim=spec.qk_rope_head_dim,
        scale=spec.scale,
    )


def validate_gqa_op_facts(facts: GQAOpFacts) -> None:
    """Validate a materialized GQA facts value."""
    if not isinstance(facts, GQAOpFacts):
        raise ContractError("attention.core.gqa facts must be GQAOpFacts")
    for field in ("num_q_heads", "num_kv_heads", "head_dim", "v_head_dim"):
        value = getattr(facts, field)
        if type(value) is not int or value <= 0:
            raise ContractError(f"attention.core.gqa facts require {field} to be a positive integer")
    if facts.num_q_heads % facts.num_kv_heads:
        raise ContractError("attention.core.gqa facts require num_q_heads to be divisible by num_kv_heads")
    if facts.v_head_dim != facts.head_dim:
        raise ContractError("attention.core.gqa facts require v_head_dim to equal head_dim")
    if type(facts.scale) is not float or not isfinite(facts.scale):
        raise ContractError("attention.core.gqa facts require scale to be a finite float")


def validate_mla_op_facts(facts: MLAOpFacts) -> None:
    """Validate a materialized MLA facts value."""
    if not isinstance(facts, MLAOpFacts):
        raise ContractError("attention.core.mla facts must be MLAOpFacts")
    for field in ("num_q_heads", "kv_lora_rank", "qk_rope_head_dim"):
        value = getattr(facts, field)
        if type(value) is not int or value <= 0:
            raise ContractError(f"attention.core.mla facts require {field} to be a positive integer")
    if facts.qk_rope_head_dim % 2:
        raise ContractError("attention.core.mla facts require qk_rope_head_dim to be even")
    if type(facts.scale) is not float or not isfinite(facts.scale):
        raise ContractError("attention.core.mla facts require scale to be a finite float")
