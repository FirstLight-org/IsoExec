"""Qwen3 MoE decoder (e.g. Qwen3-30B-A3B) with explicit per-layer router outputs."""

import torch
from torch import nn
from transformers import Qwen3MoeConfig

import isoexec.nn as inn
from isoexec.models.qwen3.model import attention, check_attention_config, padded_vocab


class Qwen3MoeDecoderLayer(nn.Module):
    def __init__(self, config: Qwen3MoeConfig, *, device: str | torch.device = "meta") -> None:
        super().__init__()
        self.attn_norm = inn.RMSNorm(hidden=config.hidden_size, eps=config.rms_norm_eps, convention="rms",
                                     device=device)
        self.mixer = attention(config, device=device)
        self.mlp_norm = inn.RMSNorm(hidden=config.hidden_size, eps=config.rms_norm_eps, convention="rms",
                                    role="pre_mlp", device=device)
        # HF: softmax over all experts in fp32, top-k, renormalize the selected (norm_topk_prob=True),
        # then weight each expert's fc2 output. Renormalized top-k of the full softmax is the softmax over
        # the selected logits, which is what ``softmax_renorm`` computes. No shared expert.
        self.mlp = inn.MoE(
            hidden=config.hidden_size, num_experts=config.num_experts, top_k=config.num_experts_per_tok,
            intermediate=config.moe_intermediate_size, routing="softmax_renorm", weight_after_fc2=True,
            device=device,
        )

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = hidden + self.mixer(self.attn_norm(hidden))
        value, scores = self.mlp(self.mlp_norm(hidden), return_aux=True)
        return hidden + value, scores


class Qwen3MoeForCausalLM(nn.Module):
    """Qwen3 MoE definition with an expert layer in every block (30B-A3B, 235B-A22B)."""

    def __init__(self, config: Qwen3MoeConfig, *, device: str | torch.device = "meta") -> None:
        super().__init__()
        if not isinstance(config, Qwen3MoeConfig):
            raise TypeError("Expected Qwen3MoeConfig")
        check_attention_config(config, "Qwen3-MoE")
        if config.norm_topk_prob is not True:
            raise ValueError("IsoExec Qwen3-MoE requires norm_topk_prob=True (renormalized top-k routing)")
        if config.decoder_sparse_step != 1 or config.mlp_only_layers:
            raise ValueError("IsoExec Qwen3-MoE requires an expert layer in every decoder block "
                             "(decoder_sparse_step=1, mlp_only_layers=[])")
        self.config = config
        vocab = padded_vocab(config)
        self.embed = inn.Embedding(vocab=vocab, hidden=config.hidden_size, device=device)
        self.layers = nn.ModuleList(Qwen3MoeDecoderLayer(config, device=device)
                                    for _ in range(config.num_hidden_layers))
        self.norm = inn.RMSNorm(hidden=config.hidden_size, eps=config.rms_norm_eps, convention="rms",
                                role="final", device=device)
        self.head = inn.LMHead(
            vocab=config.vocab_size, hidden=config.hidden_size, padded_vocab=vocab,
            tied_to=self.embed.weight if config.tie_word_embeddings else None, device=device,
        )
        if config.tie_word_embeddings:
            self._keys_to_ignore_on_load_unexpected = (r"^lm_head\.weight$",)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        hidden = self.embed(input_ids)
        aux = {}
        for index, layer in enumerate(self.layers):
            hidden, scores = layer(hidden)
            aux[f"layers.{index}.mlp.router.scores_full"] = scores
        return self.head(self.norm(hidden)), aux
