"""The configuration of a model: five frozen dataclasses and the layer-table layout helpers
(design 00 sections 2.1 and 6 decision 4).

A field exists when a shipped model reads it. There are no reserved fields, no reserved types and no
one-legal-value fields; a new architecture adds a field (and the code that reads it) in the same
change. The layer table is fully materialized -- one ``(mixer, mlp)`` row per layer, decoder rows
first and then the MTP rows -- so nothing derives a layer's kind from an interval, a frequency or a
pattern string.

Import-time rule (lint L2): this module imports nothing beyond the standard library.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, get_args, get_origin, get_type_hints


@lru_cache(maxsize=None)
def _literal_fields(cls):
    return tuple((name, get_args(hint)) for name, hint in get_type_hints(cls).items() if get_origin(hint) is Literal)


def _validate_literals(instance) -> None:
    for name, choices in _literal_fields(type(instance)):
        value = getattr(instance, name)
        if value not in choices:
            raise ValueError(f"{type(instance).__name__}.{name}: {value!r} is not one of {choices!r}")


@dataclass(frozen=True, kw_only=True)
class Attn:
    """A softmax GQA mixer row.

    ``partial_rotary`` is the fraction of ``head_dim`` that rotates; ``mrope_section`` is present iff
    the checkpoint declares multimodal rope (its entries sum to ``rotary_dim // 2``). ``out_gate`` is
    the gate function applied to the attention output, or ``None`` for an ungated mixer.
    """

    kind: Literal["attn"] = "attn"
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    rope_theta: float
    partial_rotary: float = 1.0
    rope_style: Literal["neox", "interleaved"] = "neox"
    mrope_section: tuple[int, ...] | None = None
    mrope_interleaved: bool = False
    qk_norm: Literal["rms_zero_centered", "rms", None] = "rms_zero_centered"
    qkv_bias: bool = False
    out_gate: Literal["sigmoid", "swish", None] = "sigmoid"

    def __post_init__(self) -> None:
        _validate_literals(self)

    @property
    def rotary_dim(self) -> int:
        """The number of head dims that rotate (``head_dim * partial_rotary``)."""
        return int(self.head_dim * self.partial_rotary)


@dataclass(frozen=True, kw_only=True)
class MLA:
    """Absorbed latent attention with low-rank query and shared KV projections."""

    kind: Literal["mla"] = "mla"
    num_q_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    rope_theta: float
    rope_style: Literal["neox", "interleaved"] = "neox"

    def __post_init__(self) -> None:
        _validate_literals(self)

    @property
    def rotary_dim(self) -> int:
        return self.qk_rope_head_dim

    @property
    def head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim


@dataclass(frozen=True, kw_only=True)
class GDN:
    """A gated delta-net (linear) mixer row. CPR is the only contracted algorithm, so ``chunk_size``
    is the chunk length its state machine anchors at per-request position 0."""

    kind: Literal["gdn"] = "gdn"
    num_key_heads: int
    num_value_heads: int
    key_head_dim: int
    value_head_dim: int
    conv_kernel: int
    chunk_size: int = 64
    out_norm: Literal["gated_rms_zero_centered"] = "gated_rms_zero_centered"
    output_activation: Literal["silu", "sigmoid"] = "silu"

    def __post_init__(self) -> None:
        _validate_literals(self)

    @property
    def key_dim(self) -> int:
        return self.num_key_heads * self.key_head_dim

    @property
    def value_dim(self) -> int:
        return self.num_value_heads * self.value_head_dim


@dataclass(frozen=True, kw_only=True)
class Mlp:
    """The MLP half of a row: dense when ``num_experts == 0``, a top-k mixture otherwise.

    ``intermediate_size`` is the dense width, or the per-expert width of a mixture;
    ``shared_intermediate`` is the always-on shared expert's width (0 = none).
    """

    intermediate_size: int
    num_experts: int = 0
    top_k: int = 0
    shared_intermediate: int = 0
    routing: Literal["softmax_renorm", "softmax", "sigmoid_noaux_bias_select"] = "softmax_renorm"
    routed_scale: float = 1.0
    shared_expert_gate: bool = True
    activation: Literal["silu_mul"] = "silu_mul"

    def __post_init__(self) -> None:
        _validate_literals(self)

    @property
    def is_moe(self) -> bool:
        return self.num_experts > 0


@dataclass(frozen=True, kw_only=True)
class Mtp:
    """The multi-token-prediction head geometry the MTP rows themselves cannot carry: how many rows
    are MTP rows, how each row's input is projected, and whether the depths share parameters."""

    depth: int
    proj: Literal["fused_eh"] = "fused_eh"
    share_weights: bool = False

    def __post_init__(self) -> None:
        _validate_literals(self)


@dataclass(frozen=True, kw_only=True)
class ModelConfig:
    """The whole model: the decoder-wide constants and the materialized layer table.

    ``layers`` holds the decoder rows first and then ``mtp.depth`` MTP rows, so a row's position in
    the tuple is its global index and ``layer_ns`` maps it to its parameter namespace.
    """

    model_id: str
    hidden: int
    vocab: int
    padded_vocab: int
    norm: Literal["rms_zero_centered", "rms"] = "rms_zero_centered"
    rms_eps: float = 1e-6
    tied_head: bool = False
    mtp: Mtp | None = None
    layers: tuple[tuple[Attn | GDN | MLA, Mlp], ...] = ()

    def __post_init__(self) -> None:
        _validate_literals(self)


Row = tuple[Attn | GDN | MLA, Mlp]

def n_decoder(cfg: ModelConfig) -> int:
    """How many of ``cfg.layers`` are decoder rows (the rest are the MTP rows)."""
    return len(cfg.layers) - (cfg.mtp.depth if cfg.mtp is not None else 0)


def layer_ns(cfg: ModelConfig, i: int) -> str:
    """The parameter namespace of row ``i``: ``lm.layers.{i}`` for a decoder row, ``mtp.{d}.layers``
    for the MTP row at depth ``d``."""
    n = n_decoder(cfg)
    if not (0 <= i < len(cfg.layers)):
        raise IndexError(f"row {i} is outside the {len(cfg.layers)}-row table")
    return f"lm.layers.{i}" if i < n else f"mtp.{i - n}.layers"


__all__ = [
    "Attn",
    "MLA",
    "GDN",
    "Mlp",
    "Mtp",
    "ModelConfig",
    "Row",
    "n_decoder",
    "layer_ns",
]
