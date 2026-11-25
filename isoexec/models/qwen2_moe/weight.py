"""Hugging Face checkpoint renamings for Qwen1.5-MoE."""

from transformers import Qwen2MoeConfig
from transformers.core_model_loading import WeightRenaming


def checkpoint_conversion_mapping(config: Qwen2MoeConfig) -> list[WeightRenaming]:
    """Qwen1.5-MoE only needs renames, so `config` is unused."""
    return [
        WeightRenaming(r"^model\.", ""),
        WeightRenaming(r"^embed_tokens\.weight$", "embed.weight"),
        WeightRenaming(r"^lm_head\.weight$", "head.weight"),
        WeightRenaming(r"\.input_layernorm\.weight$", ".attn_norm.weight"),
        WeightRenaming(r"\.post_attention_layernorm\.weight$", ".mlp_norm.weight"),
        *[WeightRenaming(rf"\.self_attn\.{role}_proj\.weight$", f".attn.{role}") for role in ("q", "k", "v", "o")],
        *[WeightRenaming(rf"\.self_attn\.{role}_proj\.bias$", f".attn.{role}_bias") for role in ("q", "k", "v")],
        WeightRenaming(r"\.mlp\.gate\.weight$", ".mlp.router.weight"),
        WeightRenaming(r"\.mlp\.shared_expert_gate\.weight$", ".mlp.shared.expert_gate"),
        WeightRenaming(r"\.mlp\.shared_expert\.", ".mlp.shared."),
        *[WeightRenaming(rf"\.{role}_proj\.weight$", f".{role}") for role in ("gate", "up", "down")],
    ]


__all__ = ["checkpoint_conversion_mapping"]
