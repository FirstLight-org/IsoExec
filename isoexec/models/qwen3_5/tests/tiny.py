"""Tiny dense Qwen3.5 HF ``config.json`` for CPU tests.

Build it with ``ModelSource("isoexec.models.qwen3_5").build_meta(TINY_DENSE_HF)``.
Copy before mutating; the dict is shared by every importer.
"""

from __future__ import annotations

from typing import Any

#: Text fields shared by the tiny dense and MoE configs: 4 layers (3 GDN, 1 full attention).
TINY_TEXT_COMMON: dict[str, Any] = {
    "attention_bias": False,
    "attn_output_gate": True,
    "hidden_size": 256,
    "num_hidden_layers": 4,
    "full_attention_interval": 4,  # -> [linear, linear, linear, full]
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "linear_num_key_heads": 4,
    "linear_num_value_heads": 8,
    "linear_key_head_dim": 32,
    "linear_value_head_dim": 32,
    "linear_conv_kernel_dim": 4,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-6,
    "vocab_size": 2048,
    "mtp_num_hidden_layers": 1,
    "mtp_use_dedicated_embeddings": False,
    "mamba_ssm_dtype": "float32",
    "rope_parameters": {
        "rope_type": "default",
        "rope_theta": 10000000.0,
        "partial_rotary_factor": 0.25,
        "mrope_section": [3, 3, 2],  # sums to rotary_dim/2 = 8
        "mrope_interleaved": True,
    },
}

TINY_DENSE_HF: dict[str, Any] = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "model_type": "qwen3_5",
    "tie_word_embeddings": False,
    "text_config": {
        **TINY_TEXT_COMMON,
        "model_type": "qwen3_5_text",
        "intermediate_size": 512,
    },
}

__all__ = ["TINY_DENSE_HF", "TINY_TEXT_COMMON"]
