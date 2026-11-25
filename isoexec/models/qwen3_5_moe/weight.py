"""Hugging Face checkpoint names and tensor conversions for Qwen3.5 MoE text."""

import torch
from transformers import Qwen3_5MoeConfig, Qwen3_5MoeTextConfig
from transformers.core_model_loading import ConversionOps, Split, WeightConverter, WeightRenaming

from isoexec.weights import AddScalar, Deinterleave


def checkpoint_conversion_mapping(config: Qwen3_5MoeConfig) -> list[WeightConverter | WeightRenaming]:
    """These rules see whole tensors; IsoExec takes each rank's slice afterwards."""
    text_config = config.get_text_config()
    if not isinstance(text_config, Qwen3_5MoeTextConfig):
        raise TypeError("Qwen3.5 checkpoint conversion requires Qwen3_5MoeTextConfig")

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
    mapping.extend([
        WeightRenaming(r"\.mlp\.gate\.weight$", ".mlp.router.weight"),
        WeightRenaming(r"\.mlp\.shared_expert_gate\.weight$", ".mlp.shared.expert_gate"),
        WeightRenaming(r"\.mlp\.shared_expert\.", ".mlp.shared."),
        *[WeightRenaming(rf"\.{role}_proj\.weight$", f".{role}") for role in ("gate", "up", "down")],
        WeightConverter(
            source_patterns=r"\.mlp\.experts\.gate_up_proj$",
            target_patterns=[f".mlp.experts.{i}.{role}" for i in range(text_config.num_experts)
                             for role in ("gate", "up")],
            operations=[_ExpertProjection(text_config.num_experts, packed=True)],
        ),
        WeightConverter(
            source_patterns=r"\.mlp\.experts\.down_proj$",
            target_patterns=[f".mlp.experts.{i}.down" for i in range(text_config.num_experts)],
            operations=[_ExpertProjection(text_config.num_experts, packed=False)],
        ),
    ])
    if not config.tie_word_embeddings:
        mapping.append(WeightRenaming(r"^lm_head\.weight$", "head.weight"))
    return mapping


__all__ = ["checkpoint_conversion_mapping"]


class _ExpertProjection(ConversionOps):
    """Unpack HF fused expert storage before the native loader takes rank slices."""

    def __init__(self, experts: int, *, packed: bool):
        self.experts, self.packed = experts, packed

    @torch.no_grad()
    def convert(self, input_dict, source_patterns, target_patterns, **kwargs):
        if len(input_dict) != 1:
            raise ValueError("Expert projection requires one fused checkpoint tensor")
        value = next(iter(input_dict.values()))
        if isinstance(value, list):
            if len(value) != 1:
                raise ValueError("Expert projection requires one source tensor")
            value = value[0]
        if value.ndim != 3 or value.shape[0] != self.experts:
            raise ValueError("Fused expert tensor must have [experts, rows, columns] shape")
        if self.packed and value.shape[1] % 2:
            raise ValueError("Fused gate/up projection requires even rows")
        parts = [part for expert in value.unbind(0)
                 for part in (expert.chunk(2, dim=0) if self.packed else (expert,))]
        if len(parts) != len(target_patterns):
            raise ValueError("Expert projection target count disagrees with checkpoint")
        return dict(zip(target_patterns, parts))
