"""Hugging Face checkpoint renamings for Qwen3 dense models."""

from transformers import Qwen3Config
from transformers.core_model_loading import WeightRenaming


def attention_renamings() -> list[WeightRenaming]:
    """Qwen3 attention and norm names, shared with qwen3_moe. The q/k norm gammas are held as is
    (``qk_norm="rms"``), so they are renames, never a ``gamma - 1`` conversion."""
    return [
        WeightRenaming(r"^model\.", ""),
        WeightRenaming(r"^embed_tokens\.weight$", "embed.weight"),
        WeightRenaming(r"\.input_layernorm\.weight$", ".attn_norm.weight"),
        WeightRenaming(r"\.post_attention_layernorm\.weight$", ".mlp_norm.weight"),
        *[WeightRenaming(rf"\.self_attn\.{role}_proj\.weight$", f".mixer.{role}") for role in ("q", "k", "v", "o")],
        WeightRenaming(r"\.self_attn\.q_norm\.weight$", ".mixer.q_norm"),
        WeightRenaming(r"\.self_attn\.k_norm\.weight$", ".mixer.k_norm"),
    ]


def checkpoint_conversion_mapping(config: Qwen3Config) -> list[WeightRenaming]:
    """Qwen3 dense checkpoints need renames only."""
    mapping = [
        *attention_renamings(),
        *[WeightRenaming(rf"\.mlp\.{role}_proj\.weight$", f".mlp.{role}") for role in ("gate", "up", "down")],
    ]
    if not config.tie_word_embeddings:
        mapping.append(WeightRenaming(r"^lm_head\.weight$", "head.weight"))
    return mapping


__all__ = ["attention_renamings", "checkpoint_conversion_mapping"]
