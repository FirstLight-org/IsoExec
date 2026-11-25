"""Hugging Face checkpoint renamings for Qwen3 MoE (per-expert checkpoint keys)."""

from transformers import Qwen3MoeConfig
from transformers.core_model_loading import WeightRenaming

from isoexec.models.qwen3.weight import attention_renamings


def checkpoint_conversion_mapping(config: Qwen3MoeConfig) -> list[WeightRenaming]:
    """Released Qwen3-MoE checkpoints store ``mlp.experts.{e}.{gate,up,down}_proj``: renames only."""
    mapping = [
        *attention_renamings(),
        WeightRenaming(r"\.mlp\.gate\.weight$", ".mlp.router.weight"),
        *[WeightRenaming(rf"\.{role}_proj\.weight$", f".{role}") for role in ("gate", "up", "down")],
    ]
    if not config.tie_word_embeddings:
        mapping.append(WeightRenaming(r"^lm_head\.weight$", "head.weight"))
    return mapping


__all__ = ["checkpoint_conversion_mapping"]
