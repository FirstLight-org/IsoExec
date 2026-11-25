"""Derive concrete engine-cache requirements from model configuration before allocation. Prefix
reuse is unrestricted, chunk-aligned, or disabled according to each state declaration.
"""

from __future__ import annotations

from isoexec.schema.model import GDN as GdnSpec
from isoexec.schema.model import Attn
from isoexec.schema.params import StateSpec, col


def attention_state_specs(spec: Attn) -> tuple[StateSpec, ...]:
    """The paged KV of one softmax-GQA row. ONE spec: the key and the value entry have the same
    ``(kv heads, head dim)`` geometry on every shipped model, and vLLM's ``FullAttentionSpec`` is one
    page shape, so a second declaration could only ever restate the first."""
    return (
        StateSpec(
            "kv",
            (spec.num_kv_heads, spec.head_dim),
            "bf16",
            "per_token",
            "all",
            shard=col(units=spec.num_kv_heads),
        ),
    )


def gdn_state_specs(spec: GdnSpec) -> tuple[StateSpec, ...]:
    """The cpr per-request state of one GDN row: the conv tails of the q | k | v channel segments
    (``weights.py`` CONV_W_SEGMENTS) plus the fp32 recurrent state. ``reuse="align"``: a resume point
    must be a chunk boundary."""
    hk, hv, dk, dv, w = (
        spec.num_key_heads,
        spec.num_value_heads,
        spec.key_head_dim,
        spec.value_head_dim,
        spec.conv_kernel,
    )
    colk, colv = col(units=hk), col(units=hv)
    return (
        StateSpec("conv_q", (hk, dk, w), "bf16", "per_request", "align", shard=colk),
        StateSpec("conv_k", (hk, dk, w), "bf16", "per_request", "align", shard=colk),
        StateSpec("conv_v", (hv, dv, w), "bf16", "per_request", "align", shard=colv),
        StateSpec("ssm", (hv, dk, dv), "fp32", "per_request", "align", shard=colv),
    )


__all__ = [
    "attention_state_specs",
    "gdn_state_specs",
]
