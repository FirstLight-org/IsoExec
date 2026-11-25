"""Portable state inference for GQA and MLA."""

from isoexec.schema import MLA, Attn, col, replicated
from isoexec.state.contract import StateSpec
from isoexec.state.validation import ContractError


def infer_attention_state_specs(spec: Attn) -> tuple[StateSpec, ...]:
    """Describe the separate K and V values carried for each token."""
    if not isinstance(spec, Attn):
        raise ContractError("attention.core.gqa requires Attn metadata")
    for field in ("num_q_heads", "num_kv_heads", "head_dim"):
        value = getattr(spec, field)
        if type(value) is not int or value <= 0:
            raise ContractError(f"attention.core.gqa requires {field} to be a positive integer")
    if spec.num_q_heads % spec.num_kv_heads:
        raise ContractError("attention.core.gqa requires num_q_heads to be divisible by num_kv_heads")
    return (
        StateSpec(
            name="k",
            shape=(spec.num_kv_heads, spec.head_dim),
            dtype="bf16",
            shard=col(dim=0),
            granularity="per_token",
            tokens_per_entry=1,
            retention="full_history",
        ),
        StateSpec(
            name="v",
            shape=(spec.num_kv_heads, spec.head_dim),
            dtype="bf16",
            shard=col(dim=0),
            granularity="per_token",
            tokens_per_entry=1,
            retention="full_history",
        ),
    )


def infer_mla_state_specs(spec: MLA) -> tuple[StateSpec, ...]:
    """Describe the latent KV and rotary key coordinates carried for each token."""
    if not isinstance(spec, MLA):
        raise ContractError("attention.core.mla requires MLA metadata")
    for field in ("kv_lora_rank", "qk_rope_head_dim"):
        value = getattr(spec, field)
        if type(value) is not int or value <= 0:
            raise ContractError(f"attention.core.mla requires {field} to be a positive integer")
    if spec.qk_rope_head_dim % 2:
        raise ContractError("attention.core.mla requires qk_rope_head_dim to be even")
    return (
        StateSpec(
            name="kv_latent",
            shape=(spec.kv_lora_rank + spec.qk_rope_head_dim,),
            dtype="bf16",
            shard=replicated(),
            granularity="per_token",
            tokens_per_entry=1,
            retention="full_history",
        ),
    )
