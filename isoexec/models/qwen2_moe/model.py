"""Qwen1.5-MoE decoder with explicit per-layer router outputs."""

from typing import Any, cast

import torch
from torch import nn

from isoexec.nn import Attention, Embedding, LMHead, MoE, RMSNorm

from transformers import Qwen2MoeConfig



class Qwen2MoeDecoderLayer(nn.Module):
    def __init__(self, config: Qwen2MoeConfig) -> None:
        super().__init__()
        rope = cast(dict[str, Any], config.rope_parameters)
        if config.num_key_value_heads is None:
            raise ValueError("Qwen1.5-MoE requires num_key_value_heads")
        self.attn_norm = RMSNorm(hidden=config.hidden_size, eps=config.rms_norm_eps, convention="rms")
        self.attn = Attention(hidden=config.hidden_size, heads=config.num_attention_heads,
            kv_heads=config.num_key_value_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
            rope_theta=float(rope["rope_theta"]), qkv_bias=True, qk_norm=None, out_gate=None)
        self.mlp_norm = RMSNorm(hidden=config.hidden_size, eps=config.rms_norm_eps,
                                convention="rms", role="pre_mlp")
        self.mlp = MoE(hidden=config.hidden_size, num_experts=config.num_experts,
            top_k=config.num_experts_per_tok, intermediate=config.moe_intermediate_size,
            shared_intermediate=config.shared_expert_intermediate_size,
            routing="softmax", weight_after_fc2=True)

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = hidden + self.attn(self.attn_norm(hidden))
        value, scores = self.mlp(self.mlp_norm(hidden), return_aux=True)
        return hidden + value, scores


class Qwen2MoeForCausalLM(nn.Module):
    def __init__(self, config: Qwen2MoeConfig) -> None:
        super().__init__()
        for name, expected in (("norm_topk_prob", False), ("use_sliding_window", False),
                               ("tie_word_embeddings", False), ("decoder_sparse_step", 1),
                               ("hidden_act", "silu"), ("attention_dropout", 0.0), ("qkv_bias", True)):
            if getattr(config, name) != expected:
                raise ValueError(f"Qwen1.5-MoE requires {name}={expected!r}")
        if config.mlp_only_layers:
            raise ValueError("Qwen1.5-MoE requires an expert layer in every decoder block")
        if config.layer_types != ["full_attention"] * config.num_hidden_layers:
            raise ValueError("Qwen1.5-MoE requires full_attention in every decoder block")
        if not isinstance(config.rope_parameters, dict) or config.rope_parameters["rope_type"] != "default":
            raise ValueError("Qwen1.5-MoE supports default RoPE only")
        self.config = config
        self.embed = Embedding(vocab=config.vocab_size, hidden=config.hidden_size)
        self.layers = nn.ModuleList(Qwen2MoeDecoderLayer(config) for _ in range(config.num_hidden_layers))
        self.norm = RMSNorm(hidden=config.hidden_size, eps=config.rms_norm_eps, convention="rms", role="final")
        self.head = LMHead(vocab=config.vocab_size, hidden=config.hidden_size)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        hidden = self.embed(input_ids)
        aux = {}
        for index, layer in enumerate(self.layers):
            hidden, scores = layer(hidden)
            aux[f"layers.{index}.mlp.router.scores_full"] = scores
        return self.head(self.norm(hidden)), aux
