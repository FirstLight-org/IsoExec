"""Qwen3.5 MoE text models with HF configuration and IsoExec components.

TODO: Add vision and MTP execution and checkpoint loading.
"""

from typing import Any, cast

import torch
from torch import nn
from transformers import Qwen3_5MoeConfig, Qwen3_5MoeTextConfig

import isoexec.nn as inn


class Qwen3_5MoeDecoderLayer(nn.Module):
    def __init__(
        self, config: Qwen3_5MoeTextConfig, layer_type: str, *, device: str | torch.device = "meta",
    ) -> None:
        super().__init__()
        self.attn_norm = inn.RMSNorm(
            hidden=config.hidden_size, eps=config.rms_norm_eps, device=device,
        )
        if layer_type == "linear_attention":
            self.mixer = inn.GDN(
                hidden=config.hidden_size,
                key_heads=config.linear_num_key_heads,
                value_heads=config.linear_num_value_heads,
                key_head_dim=config.linear_key_head_dim,
                value_head_dim=config.linear_value_head_dim,
                conv_kernel=config.linear_conv_kernel_dim,
                output_activation="silu", eps=config.rms_norm_eps, device=device,
            )
        elif layer_type == "full_attention":
            rope = cast(dict[str, Any], config.rope_parameters)
            if rope["rope_type"] != "default":
                raise ValueError("IsoExec Qwen3.5 supports default RoPE only")
            self.mixer = inn.Attention(
                hidden=config.hidden_size,
                heads=config.num_attention_heads,
                kv_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                rope_theta=float(rope["rope_theta"]),
                rotary_fraction=float(rope["partial_rotary_factor"]),
                mrope_section=tuple(rope["mrope_section"]),
                mrope_interleaved=rope["mrope_interleaved"],
                out_gate="sigmoid", eps=config.rms_norm_eps, device=device,
            )
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")
        self.mlp_norm = inn.RMSNorm(
            hidden=config.hidden_size, eps=config.rms_norm_eps, role="pre_mlp", device=device,
        )
        self.mlp = inn.MoE(
            hidden=config.hidden_size, num_experts=config.num_experts,
            top_k=config.num_experts_per_tok, intermediate=config.moe_intermediate_size,
            shared_intermediate=config.shared_expert_intermediate_size,
            routing="softmax_renorm", weight_after_fc2=True, device=device,
        )

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = hidden + self.mixer(self.attn_norm(hidden))
        value, scores = self.mlp(self.mlp_norm(hidden), return_aux=True)
        return hidden + value, scores


class Qwen3_5MoeForConditionalGeneration(nn.Module):
    """Text-only Qwen3.5 MoE definition, including 35B-A3B."""

    # TODO: Add vision and MTP execution before loading these namespaces.
    _keys_to_ignore_on_load_unexpected = (r"^model\.visual\.", r"^mtp\.")

    def __init__(self, hf_config: Qwen3_5MoeConfig, *, device: str | torch.device = "meta") -> None:
        super().__init__()
        config = hf_config.get_text_config()
        if not isinstance(config, Qwen3_5MoeTextConfig):
            raise TypeError("Expected Qwen3_5MoeTextConfig")
        if config.hidden_act != "silu" or config.attention_bias or config.attention_dropout:
            raise ValueError("IsoExec Qwen3.5 supports SiLU, bias-free attention and zero attention dropout")
        if not getattr(config, "attn_output_gate", True):
            raise ValueError("IsoExec Qwen3.5 implements gated attention")
        if getattr(config, "output_gate_type", "silu") not in ("silu", "swish"):
            raise ValueError("IsoExec Qwen3.5 implements a SiLU GDN output gate")
        if getattr(config, "mamba_ssm_dtype", "float32") != "float32":
            raise ValueError("IsoExec GDN uses float32 recurrent state")
        if config.layer_types is None or len(config.layer_types) != config.num_hidden_layers:
            raise ValueError("layer_types must match num_hidden_layers")
        if getattr(config, "mlp_only_layers", []):
            raise ValueError("IsoExec Qwen3.5 MoE requires an expert layer in every decoder block")
        self.config = config
        # Root HF config is authoritative for embedding tying.
        padded_vocab = (config.vocab_size + 127) // 128 * 128
        self.embed = inn.Embedding(vocab=padded_vocab, hidden=config.hidden_size, device=device)
        self.layers = nn.ModuleList(
            Qwen3_5MoeDecoderLayer(config, layer_type, device=device)
            for layer_type in config.layer_types
        )
        self.norm = inn.RMSNorm(
            hidden=config.hidden_size, eps=config.rms_norm_eps, role="final", device=device,
        )
        self.head = inn.LMHead(
            vocab=config.vocab_size, hidden=config.hidden_size, padded_vocab=padded_vocab,
            tied_to=self.embed.weight if hf_config.tie_word_embeddings else None, device=device,
        )
        if hf_config.tie_word_embeddings:
            self._keys_to_ignore_on_load_unexpected += (r"^lm_head\.weight$",)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        hidden = self.embed(input_ids)
        aux = {}
        for index, layer in enumerate(self.layers):
            hidden, scores = layer(hidden)
            aux[f"layers.{index}.mlp.router.scores_full"] = scores
        return self.head(self.norm(hidden)), aux
