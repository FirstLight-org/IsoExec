"""Tiny Qwen3.5 MoE HF ``config.json`` for CPU tests.

Build it with ``ModelSource("isoexec.models.qwen3_5_moe").build_meta(TINY_MOE_HF)``.
Copy before mutating; the dict is shared by every importer.
"""

from __future__ import annotations

from typing import Any

from isoexec.models.qwen3_5.tests.tiny import TINY_TEXT_COMMON

TINY_MOE_HF: dict[str, Any] = {
    "architectures": ["Qwen3_5MoeForConditionalGeneration"],
    "model_type": "qwen3_5_moe",
    "tie_word_embeddings": False,
    "text_config": {
        **TINY_TEXT_COMMON,
        "model_type": "qwen3_5_moe_text",
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 128,
        "shared_expert_intermediate_size": 128,
    },
}

__all__ = ["TINY_MOE_HF"]
