"""Hugging Face checkpoint names and tensor conversions for Qwen3.5 dense text."""

import torch
from transformers import Qwen3_5Config, Qwen3_5TextConfig
from transformers.core_model_loading import Split, WeightConverter, WeightRenaming

from isoexec.weights import AddScalar, Deinterleave


def checkpoint_conversion_mapping(config: Qwen3_5Config) -> list[WeightConverter | WeightRenaming]:
    """These rules see whole tensors; IsoExec takes each rank's slice afterwards."""
    text_config = config.get_text_config()
    if not isinstance(text_config, Qwen3_5TextConfig):
        raise TypeError("Qwen3.5 checkpoint conversion requires Qwen3_5TextConfig")

    mapping: list[WeightConverter | WeightRenaming] = [
        WeightRenaming(r"^model\.language_model\.", ""),
        WeightRenaming(r"^embed_tokens\.weight$", "embed.weight"),
        WeightRenaming(r"\.input_layernorm\.weight$", ".attn_norm.weight"),
        WeightRenaming(r"\.post_attention_layernorm\.weight$", ".mlp_norm.weight"),
        WeightRenaming(r"\.mlp\.gate_proj\.weight$", ".mlp.gate"),
        WeightRenaming(r"\.mlp\.up_proj\.weight$", ".mlp.up"),
        WeightRenaming(r"\.mlp\.down_proj\.weight$", ".mlp.down"),
        WeightRenaming(r"\.self_attn\.k_proj\.weight$", ".mixer.k"),
        WeightRenaming(r"\.self_attn\.v_proj\.weight$", ".mixer.v"),
        WeightRenaming(r"\.self_attn\.o_proj\.weight$", ".mixer.o"),
        WeightRenaming(r"\.self_attn\.q_norm\.weight$", ".mixer.q_norm"),
        WeightRenaming(r"\.self_attn\.k_norm\.weight$", ".mixer.k_norm"),
        WeightRenaming(r"\.linear_attn\.in_proj_z\.weight$", ".mixer.z"),
        WeightRenaming(r"\.linear_attn\.in_proj_a\.weight$", ".mixer.alpha"),
        WeightRenaming(r"\.linear_attn\.in_proj_b\.weight$", ".mixer.beta"),
        WeightRenaming(r"\.linear_attn\.conv1d\.weight$", ".mixer.conv_w"),
        WeightRenaming(r"\.linear_attn\.A_log$", ".mixer.A_log"),
        WeightRenaming(r"\.linear_attn\.dt_bias$", ".mixer.dt_bias"),
        WeightRenaming(r"\.linear_attn\.out_proj\.weight$", ".mixer.o"),
        # q_proj packs q and gate per head: [q0, gate0, q1, gate1, ...].
        WeightConverter(
            source_patterns=r"\.self_attn\.q_proj\.weight$",
            target_patterns=[".mixer.q", ".mixer.gate"],
            operations=[Deinterleave(groups=text_config.num_attention_heads, parts=2, dim=0), Split(dim=0)],
        ),
        # in_proj_qkv packs [q, k, v] back to back.
        WeightConverter(
            source_patterns=r"\.linear_attn\.in_proj_qkv\.weight$",
            target_patterns=[".mixer.q", ".mixer.k", ".mixer.v"],
            operations=[Split(dim=0)],
        ),
        # The checkpoint stores the norm scale; out_norm stores (scale - 1).
        WeightConverter(
            source_patterns=r"\.linear_attn\.norm\.weight$",
            target_patterns=".mixer.out_norm",
            operations=[AddScalar(-1.0, dtype=torch.float32)],
        ),
    ]
    if not config.tie_word_embeddings:
        mapping.append(WeightRenaming(r"^lm_head\.weight$", "head.weight"))
    return mapping


__all__ = ["checkpoint_conversion_mapping"]
