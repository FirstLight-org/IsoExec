"""Qwen3 dense decoder (e.g. Qwen3-8B) with HF configuration and IsoExec components."""

from typing import Any, cast

import torch
from torch import nn
from transformers import Qwen3Config

import isoexec.nn as inn


def check_attention_config(config: Any, family: str) -> None:
    """Refuse Qwen3 attention settings this definition does not implement (shared with qwen3_moe)."""
    for name, expected in (("hidden_act", "silu"), ("attention_bias", False), ("attention_dropout", 0.0),
                           ("use_sliding_window", False)):
        if getattr(config, name, expected) != expected:
            raise ValueError(f"IsoExec {family} requires {name}={expected!r}")
    layer_types = getattr(config, "layer_types", None)
    if layer_types is not None and list(layer_types) != ["full_attention"] * config.num_hidden_layers:
        raise ValueError(f"IsoExec {family} requires full_attention in every decoder block")
    rope = config.rope_parameters
    if not isinstance(rope, dict) or rope.get("rope_type") != "default":
        raise ValueError(f"IsoExec {family} supports default RoPE only")
    if config.num_key_value_heads is None:
        raise ValueError(f"IsoExec {family} requires num_key_value_heads")


def attention(config: Any, *, device: str | torch.device) -> inn.Attention:
    """Qwen3 attention: bias-free GQA, per-head plain-gamma q/k RMSNorm, full NeoX RoPE, no output gate."""
    rope = cast(dict[str, Any], config.rope_parameters)
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    return inn.Attention(
        hidden=config.hidden_size, heads=config.num_attention_heads, kv_heads=config.num_key_value_heads,
        head_dim=head_dim, rope_theta=float(rope["rope_theta"]),
        qkv_bias=False, qk_norm="rms", out_gate=None, eps=config.rms_norm_eps, device=device,
    )


def padded_vocab(config: Any) -> int:
    return (config.vocab_size + 127) // 128 * 128


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config, *, device: str | torch.device = "meta") -> None:
        super().__init__()
        self.attn_norm = inn.RMSNorm(hidden=config.hidden_size, eps=config.rms_norm_eps, convention="rms",
                                     device=device)
        self.mixer = attention(config, device=device)
        self.mlp_norm = inn.RMSNorm(hidden=config.hidden_size, eps=config.rms_norm_eps, convention="rms",
                                    role="pre_mlp", device=device)
        self.mlp = inn.MLP(hidden=config.hidden_size, intermediate=config.intermediate_size, device=device)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.mixer(self.attn_norm(hidden))
        return hidden + self.mlp(self.mlp_norm(hidden))


class Qwen3ForCausalLM(nn.Module):
    """One dense definition for the Qwen3 HF configurations (0.6B-32B)."""

    def __init__(self, config: Qwen3Config, *, device: str | torch.device = "meta") -> None:
        super().__init__()
        if not isinstance(config, Qwen3Config):
            raise TypeError("Expected Qwen3Config")
        check_attention_config(config, "Qwen3")
        self.config = config
        vocab = padded_vocab(config)
        self.embed = inn.Embedding(vocab=vocab, hidden=config.hidden_size, device=device)
        self.layers = nn.ModuleList(Qwen3DecoderLayer(config, device=device)
                                    for _ in range(config.num_hidden_layers))
        self.norm = inn.RMSNorm(hidden=config.hidden_size, eps=config.rms_norm_eps, convention="rms",
                                role="final", device=device)
        self.head = inn.LMHead(
            vocab=config.vocab_size, hidden=config.hidden_size, padded_vocab=vocab,
            tied_to=self.embed.weight if config.tie_word_embeddings else None, device=device,
        )
        if config.tie_word_embeddings:
            self._keys_to_ignore_on_load_unexpected = (r"^lm_head\.weight$",)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embed(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return self.head(self.norm(hidden))
