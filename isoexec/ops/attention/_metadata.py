"""Validation shared by op-owned attention inference functions."""

from math import isfinite

from isoexec.state.validation import ContractError


def validate_gqa_metadata(spec: object) -> None:
    for field in ("num_q_heads", "num_kv_heads", "head_dim"):
        field_value = getattr(spec, field, None)
        if type(field_value) is not int or field_value <= 0:
            raise ContractError(f"attention.core.gqa metadata: {field} must be a positive integer")
    if spec.num_q_heads % spec.num_kv_heads:
        raise ContractError("attention.core.gqa metadata: num_q_heads must be divisible by num_kv_heads")
    scale = getattr(spec, "scale", None)
    if type(scale) is not float or not isfinite(scale):
        raise ContractError("attention.core.gqa metadata: scale must be a finite float")


def validate_mla_metadata(spec: object) -> None:
    for field in ("num_q_heads", "kv_lora_rank", "qk_rope_head_dim"):
        field_value = getattr(spec, field, None)
        if type(field_value) is not int or field_value <= 0:
            raise ContractError(f"attention.core.mla metadata: {field} must be a positive integer")
    if spec.qk_rope_head_dim % 2:
        raise ContractError("attention.core.mla metadata: qk_rope_head_dim must be even")
    scale = getattr(spec, "scale", None)
    if type(scale) is not float or not isfinite(scale):
        raise ContractError("attention.core.mla metadata: scale must be a finite float")
