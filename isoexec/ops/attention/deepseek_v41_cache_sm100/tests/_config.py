"""The published DeepSeek-V4.1-Flash text geometry the composition serves, as a plain config object.

The values are the attention-relevant fields of the released ``config.json`` ``text_config``
(deepseek-ai/DeepSeek-V4.1-Flash); tests derive smaller geometries from them with ``config(**overrides)``.
"""
from __future__ import annotations

from types import MappingProxyType

PUBLISHED = MappingProxyType({
    'vocab_size': 129280,
    'hidden_size': 5120,
    'num_hidden_layers': 40,
    'num_attention_heads': 64,
    'head_dim': 512,
    'qk_rope_head_dim': 64,
    'q_lora_rank': 1280,
    'o_lora_rank': 1024,
    'o_groups': 8,
    'rms_norm_eps': 1e-20,
    'max_position_embeddings': 1048576,
    'rope_theta': 10000,
    'rope_scaling': {'rope_type': 'yarn', 'factor': 16, 'beta_fast': 32, 'beta_slow': 1, 'original_max_position_embeddings': 65536},
    'sliding_window': 128,
    'compress_ratios': [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    'compress_rope_theta': 160000,
    'kv_source_layer_ids': [2, 8, 14, 20],
    'index_source_layer_ids': [2, 8, 14, 20, 24, 28, 32, 36],
    'index_n_heads': 32,
    'index_head_dim': 128,
    'index_topk': 512,
    'candidate_source_layer_id': 20,
    'candidate_topk_blocks': 2048,
    'candidate_block_size': 8,
    'hc_mult': 4,
    'engram_layer_ids': [1, 14],
    'engram_num_embeddings': [384006168, 384016682],
    'num_nextn_predict_layers': 3,
    'dspark_target_layer_ids': [37, 38, 39],
})


def _freeze(v):
    if isinstance(v, dict):
        return MappingProxyType({k: _freeze(x) for k, x in v.items()})
    if isinstance(v, list):
        return tuple(_freeze(x) for x in v)
    return v


class Config:
    """Attribute view of a text config; ``source_layer`` is the model's shared-producer rule."""

    def __init__(self, fields):
        self._fields = {k: _freeze(v) for k, v in fields.items()}

    def __getattr__(self, name):
        try:
            return self.__dict__["_fields"][name]
        except KeyError:
            raise AttributeError(name) from None

    def source_layer(self, index, kind):
        """The latest producer at or before ``index``; None for a window-only layer."""
        if kind not in ("kv", "index"):
            raise ValueError(f"unknown shared source kind {kind!r}")
        if self.compress_ratios[index] == 0:
            return None
        sources = self.kv_source_layer_ids if kind == "kv" else self.index_source_layer_ids
        candidates = [i for i in sources if i <= index]
        if not candidates:
            raise ValueError(f"layer {index} has no earlier {kind} source")
        return candidates[-1]


def published_config():
    return Config(dict(PUBLISHED))


def config(**overrides):
    fields = dict(PUBLISHED)
    unknown = set(overrides) - set(fields)
    if unknown:
        raise KeyError(f"not a served config field: {sorted(unknown)}")
    fields.update(overrides)
    return Config(fields)
