"""HF configuration of the DeepSeek V4.1 Flash text tower and its Causal Encoder-Decoder geometry.

DeepSeek V4.1 Flash is a Causal Encoder-Decoder (CED, the tech report's Fig. 3): a causal encoder of
``L/2`` blocks runs on every token; the decoder's global KV is projected from the encoder output by the
decoder's first layer (the boundary ``b``); the decoder's ``L/2`` blocks run on decoder segments. The
config states the split only implicitly (``b = max(kv_source_layer_ids)``, the candidate source); this
module derives it, validates every CED invariant against the config (``validate_ced``; a layout that is
not an encoder-decoder is refused by name) and returns the frozen ``CedGeometry`` every component and
kernel reads (``cfg.num_attention_heads``, ``cfg.encoder``, ``cfg.decoder``, ``cfg.boundary``...).

A reduced-depth checkpoint declares itself with a root ``isoexec_subset`` mapping
``{"layer_sources": [production layer of each row, ...], "layout": "physical" | "renamed"}``; it is
accepted only if it is still a CED whose boundary row is production layer 20. A checkpoint without a
``quantization_config`` stores BF16 tensors; the released one stores FP8/FP4 blocks decoded exactly to
BF16 masters at load.

Excluded by declaration (text-only RL): the vision tower and DSpark (the "MTP" rows). Their config
fields are accepted and recorded in ``CedGeometry.excluded``; nothing of them is built.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Literal

from transformers import PretrainedConfig

MODEL_TYPE = "deepseek_v41"
ARCHITECTURE = "DeepseekV41ForCausalLM"
#: The one recipe this package ships: BF16 masters, rollout and trainer on one program.
RECIPE = "sm100_bf16_actor"
#: The forward program of the CED definition (bumped from the flat 40-row ``deepseek_v41_bf16_actor_v1``).
FORWARD_PROGRAM = "deepseek_v41_ced_bf16_actor_v2"
#: Learned-sink sparse attention recipe (``deepseek_v41_cache_sm100.ops.sink_forward``): dual rail.
SINK_RECIPE = 3
#: The one MoE rounding program (user decision 2026-11-17, = the official reference's): each routed expert's
#: weight multiplied into its activation before FC2, an FP32 add chain over the selected experts in ascending
#: expert id, the shared expert added last, one BF16 rounding.
MOE_RECIPE = "deepseek_v41_moe_reference_order_v1"

#: CED prefill programs (``DecoderSegments``). ``bounded_replay`` is the release program (user decision
#: 2026-11-17): the decoder runs on the last ``replay_window`` prompt tokens and on generated tokens, the rule
#: a function of the prompt length only. ``full_decoder`` runs the decoder on every token (the official
#: reference code's program); it is a declared mode for reference-parity gates only.
BOUNDED_REPLAY = "bounded_replay"
FULL_DECODER = "full_decoder"
CED_PREFILL_MODES = (BOUNDED_REPLAY, FULL_DECODER)
RELEASE_CED_PREFILL = BOUNDED_REPLAY

#: Engram hash layout of the released checkpoint. Hash primes and multipliers depend on every Engram layer
#: of the released tower, so a reduced-depth model hashes with this layout and reads its rows' columns.
RELEASED_ENGRAM_LAYERS = (1, 14)
RELEASED_ENGRAM_ROWS = (384006168, 384016682)
#: The released CED boundary (first decoder layer): production layer L/2 = 20.
RELEASED_BOUNDARY = 20

_PINNED = {"hidden_act": "silu", "attention_bias": False, "tie_word_embeddings": False,
           "scoring_func": "sqrtsoftplus", "topk_method": "noaux_tc", "norm_topk_prob": True,
           "num_key_value_heads": 1, "hc_mult": 4, "n_shared_experts": 1}
_QUANTIZATION = {"quant_method": "fp8", "activation_scheme": "dynamic", "weight_block_size": [32, 32],
                 "scale_fmt": "ue8m0", "expert_dtype": "fp4"}
_FIELDS = (
    "vocab_size", "hidden_size", "moe_intermediate_size", "num_hidden_layers", "num_attention_heads",
    "num_key_value_heads", "head_dim", "qk_rope_head_dim", "q_lora_rank", "o_lora_rank", "o_groups",
    "hidden_act", "swiglu_limit", "rms_norm_eps", "attention_bias", "tie_word_embeddings",
    "max_position_embeddings", "rope_theta", "rope_scaling", "n_routed_experts", "n_shared_experts",
    "num_experts_per_tok", "scoring_func", "topk_method", "norm_topk_prob", "routed_scaling_factor",
    "sliding_window", "compress_ratios", "compress_rope_theta", "kv_source_layer_ids",
    "index_source_layer_ids", "index_n_heads", "index_head_dim", "index_topk", "candidate_source_layer_id",
    "candidate_topk_blocks", "candidate_block_size", "hc_mult", "hc_sinkhorn_iters", "hc_eps",
    "engram_layer_ids", "engram_num_embeddings", "engram_max_ngram_size", "engram_vocab_size",
    "engram_n_heads", "engram_head_dim", "engram_pad_token_id", "engram_compressed_vocab_size",
    "num_nextn_predict_layers",
)
#: DSpark fields: accepted, recorded as excluded, never built.
_DSPARK_FIELDS = ("dspark_block_size", "dspark_noise_token_id", "dspark_target_layer_ids", "dspark_markov_rank",
                  "dspark_n_routed_experts", "dspark_num_experts_per_tok")
_SUBSET_FIELDS = {"layer_sources", "experts", "intermediate", "source", "qualification", "layout"}


class CedError(ValueError):
    """A config that is not the Causal Encoder-Decoder this package executes; ``code`` names the invariant."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


class DeepseekV41Config(PretrainedConfig):
    """Root config: ``text_config`` (the executed tower), ``vision_config`` and ``quantization_config``
    as stored; the vision tower and the DSpark rows are not built."""

    model_type = MODEL_TYPE

    def __init__(self, text_config: dict | None = None, vision_config: dict | None = None,
                 isoexec_subset: dict | None = None, **kwargs: Any) -> None:
        self.text_config = dict(text_config or {})
        self.vision_config = dict(vision_config or {})
        self.isoexec_subset = None if isoexec_subset is None else dict(isoexec_subset)
        super().__init__(**kwargs)


def register() -> None:
    """Make ``AutoConfig.for_model("deepseek_v41")`` resolve; a Transformers release that already ships
    the model type keeps its own class."""
    from transformers import AutoConfig
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    if MODEL_TYPE not in CONFIG_MAPPING:
        AutoConfig.register(MODEL_TYPE, DeepseekV41Config)


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, MappingProxyType):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


# ---------------------------------------------------------------------------------------------- layer kinds
class Csa2Mode(str, Enum):
    """Attention mode of a layer, in the report's terms (Fig. 3, Fig. 4)."""

    SWA = "swa"            # sliding window only (ratio 0)
    FULL = "full"          # SWA + CSA2: own compressor, own indexer, owns the index keys
    REUSE = "reuse"        # SWA + CSA2: the latest FULL's compressed KV, the latest index source's top-k
    REINDEX = "reindex"    # SWA + CSA2 (decoder): own indexer queries over the boundary's index keys, in its pool


@dataclass(frozen=True)
class LayerKind:
    """One backbone layer as the definition executes it."""

    layer: int                                       # row in this checkpoint
    checkpoint_layer: int                            # production layer id
    stack: Literal["encoder", "decoder"]
    mode: Csa2Mode
    ratio: int                                       # 0 / encoder ratio (2) / 1 (decoder)
    kv_source: int | None                            # layer whose compressed (global) KV this layer reads
    index_source: int | None                         # layer whose top-k this layer attends with
    candidate: Literal["none", "select", "mask"]     # Hierarchical Sparse Indexer role
    engram: int | None                               # Engram hash column (released Engram layer), encoder only
    rope: Literal["swa", "compressed"]               # theta rope_theta, no YaRN | compress_rope_theta + YaRN
    window_clamp: bool                               # decoder SWA truncated at the decoder segment start

    @property
    def is_entry(self) -> bool:
        """The decoder's FULL layer: its global-KV projection is ``decoder.entry`` (every token)."""
        return self.stack == "decoder" and self.mode is Csa2Mode.FULL


# ---------------------------------------------------------------------------------------------- geometry
@dataclass(frozen=True)
class Geometry:
    """The validated text tower. Attribute access reads ``text`` fields; the rest is derived."""

    text: Any
    layer_sources: tuple[int, ...]
    subset: bool

    def __getattr__(self, name: str) -> Any:
        text = self.__dict__.get("text")       # absent while a copy is being built
        if text is None or name.startswith("__") or name not in text:
            raise AttributeError(name)
        return text[name]

    def __deepcopy__(self, memo):
        return self           # immutable

    def __reduce__(self):
        return (Geometry, (_thaw(self.text), self.layer_sources, self.subset))

    def __post_init__(self):
        if not isinstance(self.text, MappingProxyType):
            object.__setattr__(self, "text", _freeze(dict(self.text)))

    def source_layer(self, index: int, kind: str) -> int | None:
        """The producer layer whose compressed KV (``"kv"``) or index keys (``"index"``) row ``index`` reads."""
        if kind not in ("kv", "index"):
            raise ValueError(f"unknown shared source kind {kind!r}")
        if self.compress_ratios[index] == 0:
            return None
        sources = self.kv_source_layer_ids if kind == "kv" else self.index_source_layer_ids
        candidates = [i for i in sources if i <= index]
        if not candidates:
            raise ValueError(f"layer {index} has no earlier {kind} source")
        return candidates[-1]

    @property
    def released_engram_layers(self) -> tuple[int, ...]:
        return RELEASED_ENGRAM_LAYERS if self.subset else tuple(self.engram_layer_ids)

    @property
    def released_engram_rows(self) -> tuple[int, ...]:
        return RELEASED_ENGRAM_ROWS if self.subset else tuple(self.engram_num_embeddings)

    def hash_column(self, index: int) -> int:
        """The hash column (released Engram layer) of Engram row ``index``."""
        return self.released_engram_layers.index(self.layer_sources[index])

    def engram_width(self) -> int:
        return (self.engram_max_ngram_size - 1) * self.engram_n_heads * self.engram_head_dim


@dataclass(frozen=True)
class CedGeometry(Geometry):
    """``Geometry`` plus the Causal Encoder-Decoder split (``validate_ced``)."""

    storage: Literal["fp8_fp4", "bf16"] = "fp8_fp4"
    layout: Literal["renamed", "physical"] = "renamed"
    encoder: tuple[LayerKind, ...] = ()
    decoder: tuple[LayerKind, ...] = ()
    boundary: int = 0
    replay_window: int = 0
    excluded: Any = field(default_factory=lambda: MappingProxyType({}))
    native: bool = True

    def __reduce__(self):
        return (_rebuild, (_thaw(self.text), self.layer_sources, self.subset, self.storage, self.layout,
                           self.native))

    @property
    def kinds(self) -> tuple[LayerKind, ...]:
        return self.encoder + self.decoder

    def kind(self, index: int) -> LayerKind:
        return self.kinds[index]

    def encoder_groups(self) -> tuple[tuple[int, ...], ...]:
        """Encoder CSA2 groups: each FULL layer with the REUSE layers reading it (report: groups of six)."""
        groups: list[list[int]] = []
        for k in self.encoder:
            if k.mode is Csa2Mode.FULL:
                groups.append([k.layer])
            elif k.mode is Csa2Mode.REUSE:
                groups[-1].append(k.layer)
        return tuple(tuple(g) for g in groups)

    def admissible_pipeline_cuts(self) -> tuple[int, ...]:
        """Layers at which a pipeline stage may start: only the residual carry crosses. Encoder: after the
        SWA-only layers (any of them) and at each encoder group start; the CED boundary (the handoff). No cut
        inside the decoder: every decoder layer reads the decoder global KV that ``decoder.entry`` owns."""
        cuts = [k.layer for k in self.encoder if k.layer > 0 and (k.mode is Csa2Mode.SWA or
                                                                  k.mode is Csa2Mode.FULL)]
        return tuple(sorted(set(cuts) | {self.boundary}))


def _rebuild(text, layer_sources, subset, storage, layout, native):
    return _derive(Geometry(MappingProxyType(_freeze(text)), tuple(layer_sources), subset), storage, layout,
                   native=native)


def _config_dict(config: Any) -> tuple[dict, dict | None, dict | None]:
    if isinstance(config, PretrainedConfig):
        text = getattr(config, "text_config", None)
        quant = getattr(config, "quantization_config", None)
        subset = getattr(config, "isoexec_subset", None)
        arch = getattr(config, "architectures", None)
    else:
        text, quant = config.get("text_config"), config.get("quantization_config")
        subset, arch = config.get("isoexec_subset"), config.get("architectures")
    if arch is not None and list(arch) != [ARCHITECTURE]:
        raise ValueError(f"DeepSeek V4.1 Flash expects architectures [{ARCHITECTURE!r}], got {arch!r}")
    if isinstance(text, PretrainedConfig):
        text = text.to_dict()
    if not isinstance(text, dict):
        raise ValueError("DeepSeek V4.1 Flash requires a text_config mapping")
    if quant is not None and not isinstance(quant, dict):
        quant = quant.to_dict() if hasattr(quant, "to_dict") else dict(quant)
    return dict(text), None if quant is None else dict(quant), None if subset is None else dict(subset)


def _base_geometry(config: Any) -> tuple[Geometry, str, str]:
    """Field, pin and encoding checks of the text tower (not yet the CED structure)."""
    text, quant, subset = _config_dict(config)
    missing = [name for name in _FIELDS if name not in text]
    if missing:
        raise ValueError(f"text_config lacks {missing[:6]}")
    for key, value in _PINNED.items():
        if text[key] != value:
            raise ValueError(f"text_config.{key}: this package implements {value!r}, got {text[key]!r}")
    if quant is None:
        storage = "bf16"
    else:
        for key, value in _QUANTIZATION.items():
            if quant.get(key) != value:
                raise ValueError(f"quantization_config.{key}: the decoded checkpoint encoding is {value!r}, "
                                 f"got {quant.get(key)!r}")
        storage = "fp8_fp4"
    n = text["num_hidden_layers"]
    if text["num_nextn_predict_layers"] and len(text["compress_ratios"]) == n:
        raise ValueError("compress_ratios must cover the declared DSpark rows")
    if len(text["compress_ratios"]) < n:
        raise ValueError("compress_ratios must name every backbone layer")
    t = {name: _freeze(text[name]) for name in _FIELDS}
    t.update({name: _freeze(text[name]) for name in _DSPARK_FIELDS if name in text})
    t["compress_ratios"] = tuple(t["compress_ratios"][:n])     # DSpark rows are not built
    sources = tuple(range(n)) if subset is None else tuple(int(i) for i in subset.get("layer_sources", ()))
    if len(sources) != n or list(sources) != sorted(set(sources)):
        raise ValueError("isoexec_subset.layer_sources must name one ascending checkpoint layer per row")
    unknown = set(subset or ()) - _SUBSET_FIELDS
    if unknown:
        raise ValueError(f"isoexec_subset has unknown fields {sorted(unknown)}")
    layout = (subset or {}).get("layout", "renamed")
    if layout not in ("renamed", "physical"):
        raise ValueError(f"isoexec_subset.layout must be 'renamed' or 'physical', got {layout!r}")
    g = Geometry(MappingProxyType(t), sources, subset is not None)
    positive = ("vocab_size", "hidden_size", "moe_intermediate_size", "num_hidden_layers", "num_attention_heads",
                "head_dim", "qk_rope_head_dim", "q_lora_rank", "o_lora_rank", "o_groups", "n_routed_experts",
                "num_experts_per_tok", "index_n_heads", "index_head_dim", "index_topk", "sliding_window",
                "hc_sinkhorn_iters", "candidate_topk_blocks", "candidate_block_size")
    if any(type(t[k]) is not int or t[k] < 1 for k in positive):
        raise ValueError("model dimensions must be positive integers")
    if g.num_experts_per_tok > g.n_routed_experts or g.num_attention_heads % g.o_groups:
        raise ValueError("invalid attention groups or expert top-k")
    if g.hidden_size % 32 or g.moe_intermediate_size % 32 or g.engram_head_dim % 32:
        raise ValueError("FP4/FP8 contracted dimensions must be multiples of 32")
    for kind, ids in (("kv", g.kv_source_layer_ids), ("index", g.index_source_layer_ids)):
        if tuple(sorted(set(ids))) != tuple(ids) or any(not 0 <= i < n or not g.compress_ratios[i] for i in ids):
            raise ValueError(f"{kind} sources must be unique ordered compressed layers")
        for i in range(n):
            src = g.source_layer(i, kind)
            if src is not None and g.compress_ratios[src] != g.compress_ratios[i]:
                raise ValueError(f"layer {i} cannot share a {kind} source with a different compression ratio")
    if not set(g.kv_source_layer_ids) <= set(g.index_source_layer_ids):
        raise ValueError("each KV source must produce index keys")
    if g.candidate_source_layer_id not in g.index_source_layer_ids:
        raise ValueError("candidate source must be an index-producing layer")
    if len(g.engram_layer_ids) != len(g.engram_num_embeddings):
        raise ValueError("Engram rows and table sizes must pair")
    for i, rows in zip(g.engram_layer_ids, g.engram_num_embeddings):
        if not 0 <= i < n or g.layer_sources[i] not in g.released_engram_layers:
            raise ValueError(f"Engram row {i} is not a released Engram layer")
        if rows != g.released_engram_rows[g.hash_column(i)]:
            raise ValueError(f"Engram row {i} table size differs from the released hash layout")
    if math.isqrt(g.index_head_dim * g.index_n_heads) ** 2 != g.index_head_dim * g.index_n_heads:
        raise ValueError("index weight scale must be an exact power of two")
    if g.qk_rope_head_dim > g.head_dim or g.qk_rope_head_dim > g.index_head_dim:
        raise ValueError("the RoPE tail must fit the attention and index heads")
    rope = dict(t["rope_scaling"])
    if rope.get("rope_type") != "yarn":
        raise ValueError("DeepSeek V4.1 uses YaRN rotary scaling")
    return g, storage, layout


def validate_ced(g: Geometry) -> tuple[tuple[LayerKind, ...], tuple[LayerKind, ...], int]:
    """The Causal Encoder-Decoder invariants of ``g``; returns ``(encoder kinds, decoder kinds, boundary)``.

    Raises ``CedError`` naming the first violated invariant (report pages in ``debug-model/ARCHITECTURE.md``):
    the boundary is the last KV source and the candidate source (p.11-12); at production depth it is ``L/2``
    (p.9, p.21); a subset maps its boundary to production layer 20 and keeps encoder rows below it; encoder
    layers are a non-empty SWA-only prefix then compressed layers of one ratio > 1 whose groups start at a FULL
    layer, with no Reindex and no candidate role (p.22, Fig. 3); every decoder layer has ratio 1 and reads the
    boundary's KV; Reindex layers are the index sources after the boundary; Engram only in the encoder (p.13).
    """
    n = g.num_hidden_layers
    kv, ix = tuple(g.kv_source_layer_ids), tuple(g.index_source_layer_ids)
    if not kv:
        raise CedError("CED_NO_KV_SOURCE", "a CED needs the decoder's KV-source layer")
    b = max(kv)
    if g.candidate_source_layer_id != b:
        raise CedError("CED_BOUNDARY_NOT_CANDIDATE_SOURCE",
                       f"the last KV source {b} (the decoder's FULL layer) must be the candidate source, got "
                       f"candidate_source_layer_id {g.candidate_source_layer_id}")
    if b < 1 or b >= n:
        raise CedError("CED_EMPTY_STACK", f"boundary {b} leaves an empty encoder or decoder in {n} layers")
    ratio_one = [i for i in kv if g.compress_ratios[i] == 1]
    if ratio_one != [b]:
        raise CedError("CED_DECODER_KV_SOURCES", f"the decoder has one KV source (its first layer, ratio 1), got "
                                                 f"ratio-1 KV sources {ratio_one}")
    if not g.subset and 2 * b != n:
        raise CedError("CED_BOUNDARY_NOT_HALF_DEPTH", f"at production depth the decoder starts at L/2 = {n / 2}, "
                                                      f"got boundary {b}")
    if g.subset:
        if g.layer_sources[b] != RELEASED_BOUNDARY:
            raise CedError("CED_SUBSET_BOUNDARY", f"subset boundary row {b} is production layer "
                                                  f"{g.layer_sources[b]}, not the released boundary "
                                                  f"{RELEASED_BOUNDARY}")
        if any(s >= RELEASED_BOUNDARY for s in g.layer_sources[:b]):
            raise CedError("CED_SUBSET_STACKS", "subset encoder rows must be production encoder layers")
    ratios = g.compress_ratios
    enc_ratios = sorted({ratios[i] for i in range(b) if ratios[i]})
    if ratios[0] != 0:
        raise CedError("CED_ENCODER_SWA_PREFIX", "the encoder starts with sliding-window-only layers")
    swa = [i for i in range(b) if ratios[i] == 0]
    if swa != list(range(len(swa))):
        raise CedError("CED_ENCODER_SWA_PREFIX", f"sliding-window-only layers must be an encoder prefix, got {swa}")
    if len(enc_ratios) > 1 or (enc_ratios and enc_ratios[0] < 2):
        raise CedError("CED_ENCODER_RATIO", f"encoder compressed layers have one ratio > 1, got {enc_ratios}")
    if any(ratios[i] != 1 for i in range(b, n)):
        raise CedError("CED_DECODER_RATIO", "every decoder layer has compression ratio 1 (no sequence compression)")
    enc_ix = tuple(i for i in ix if i < b)
    if enc_ix != tuple(i for i in kv if i < b):
        raise CedError("CED_ENCODER_REINDEX", "encoder index sources must be exactly its FULL (KV-source) layers")
    if swa and len(swa) < b and swa[-1] + 1 not in kv:
        raise CedError("CED_ENCODER_GROUP_START", "the first compressed encoder layer must be a FULL layer")
    if any(i >= b for i in g.engram_layer_ids):
        raise CedError("CED_ENGRAM_IN_DECODER", "Engram modules sit in the encoder only")

    columns = {i: g.hash_column(i) for i in g.engram_layer_ids}
    kinds = []
    for i in range(n):
        r = ratios[i]
        stack = "encoder" if i < b else "decoder"
        if r == 0:
            mode = Csa2Mode.SWA
        elif i in kv:
            mode = Csa2Mode.FULL
        elif i in ix:
            mode = Csa2Mode.REINDEX
        else:
            mode = Csa2Mode.REUSE
        candidate = "select" if i == b else "mask" if (i > b and i in ix) else "none"
        kinds.append(LayerKind(
            layer=i, checkpoint_layer=g.layer_sources[i], stack=stack, mode=mode, ratio=r,
            kv_source=g.source_layer(i, "kv"), index_source=g.source_layer(i, "index"), candidate=candidate,
            engram=columns.get(i), rope="compressed" if r else "swa", window_clamp=i > b))
    return tuple(kinds[:b]), tuple(kinds[b:]), b


def _derive(g: Geometry, storage: str, layout: str, *, native: bool) -> CedGeometry:
    encoder, decoder, b = validate_ced(g)
    excluded = {"vision": "text-only RL: no vision tower, aligner, image delimiters or bias_vl; image tokens "
                          "are refused at input",
                "dspark": "speculation only (the report's DSpark rows, config 'MTP'); no gradient into the "
                          "backbone [R p.14]; compress_ratios[L:] and dspark_* are recorded, never built"}
    ced = CedGeometry(g.text, g.layer_sources, g.subset, storage=storage, layout=layout, encoder=encoder,
                      decoder=decoder, boundary=b, replay_window=g.sliding_window,
                      excluded=MappingProxyType(excluded), native=native)
    if native:
        native_admission(ced)
    return ced


def native_admission(cfg) -> None:
    """Every SM100 op package's own admission of this geometry, before anything is built (each raises naming
    what it does not serve): the attention composition (``supported_config``), the DeepSeek mHC (its declared
    ``hidden`` / ``hc_mult`` domains), BF16 MegaMoE (``check_geometry``), ``mm_sm100``'s leaf for every dense
    projection it runs, the FP32 log-prob head's vocabulary tiling and the head's vocab shard."""
    from isoexec.ops.attention.deepseek_v41_cache_sm100.ops import supported_config
    from isoexec.ops.logprobs.logprobs_sm100 import spec as logprobs_spec
    from isoexec.ops.mm.mm_sm100 import segments
    from isoexec.ops.moe.deepseek_v41_sm100.mega import check_geometry
    from isoexec.ops.residual.deepseek_mhc_sm100 import spec as mhc_spec

    from .providers import vocab_shard_is_bit_neutral

    supported_config(cfg)
    for impl in mhc_spec.IMPLS:
        for key, value in (("hidden", cfg.hidden_size), ("hc_mult", cfg.hc_mult)):
            if value not in impl.inputs[key]:
                raise ValueError(f"{impl.op} ({impl.impl_id}) serves {key} in {impl.inputs[key]!r}, got {value}")
    check_geometry(cfg.n_routed_experts, cfg.hidden_size, cfg.moe_intermediate_size)
    H, D, heads = cfg.hidden_size, cfg.head_dim, cfg.num_attention_heads
    shapes = {"attn.wq_a": (H, cfg.q_lora_rank), "attn.wq_b": (cfg.q_lora_rank, heads * D), "attn.wkv": (H, D),
              "attn.wo_b": (cfg.o_groups * cfg.o_lora_rank, H), "attn.compressor.wkv": (H, D),
              "attn.indexer.wq_b": (cfg.q_lora_rank, cfg.index_n_heads * cfg.index_head_dim),
              "attn.indexer.wk": (D, cfg.index_head_dim), "head": (H, cfg.vocab_size)}
    if cfg.engram_layer_ids:
        shapes["engram.wkv"] = (cfg.engram_width(), H * (cfg.hc_mult + 1))
    for name, (k, n) in shapes.items():
        if not segments.supported(k, n):
            raise ValueError(f"mm_sm100 does not serve {name} (K = {k}, N = {n})")
    tile = logprobs_spec.LEAVES * logprobs_spec.STRIP
    if cfg.vocab_size % tile:
        raise ValueError(f"logprobs_sm100 needs a vocabulary multiple of {tile}, got {cfg.vocab_size}")
    if not vocab_shard_is_bit_neutral(cfg, 1):
        raise ValueError("the FP32 head's vocabulary is not a weight mm_sm100 serves")


def ced_geometry(config: Any, *, native: bool = True) -> CedGeometry:
    """Validate the HF config (root config object or raw mapping) and return the executed CED geometry.

    ``native=False`` skips the SM100 kernels' geometry check (``supported_config``): only the CPU reference
    program (``reference.py``) builds such a geometry (tiny synthetic models)."""
    g, storage, layout = _base_geometry(config)
    return _derive(g, storage, layout, native=native)


def text_geometry(config: Any) -> CedGeometry:
    """The executed geometry (kept name for the op packages and runtime bindings)."""
    return ced_geometry(config)


def program_identity(cfg: CedGeometry, ced_prefill: str) -> dict:
    """Everything that defines the computed FORWARD function besides the weights. The trainer streams it with
    every weight update and the engine refuses a mismatch (``isoexec.sync.identity``).

    ``index_scores`` is the index-score program in effect (``spec.INDEX_FOLD``: ``sm100_dsa_index_logits_dg_v1``,
    or ``sm100_dsa_indexer_tree_v1`` under the selectable FP32 tree), ``attention_forward`` the forward
    sub-recipes. Not part of the identity: the backward provider tags (cuDNN deterministic / the IsoExec kernel:
    the backward does not enter engine == trainer forward logprobs) and the transport fallbacks
    (``ISOEXEC_DS_TP_REDUCER``, ``ISOEXEC_DS_EP_EXCHANGE``: same bits either way); ``program_recorded`` reports
    them beside the identity."""
    from isoexec.ops.attention.deepseek_v41_cache_sm100 import spec

    if ced_prefill not in CED_PREFILL_MODES:
        raise ValueError(f"ced_prefill must be one of {CED_PREFILL_MODES}, got {ced_prefill!r}")
    index_scores = spec.INDEXER_TREE_RECIPE if spec.INDEX_FOLD == "fp32_tree" else spec.SUB_RECIPES["indexer"]
    forward = {name: tag for name, tag in spec.SUB_RECIPES.items() if not name.endswith("_bwd")}
    forward["indexer"] = index_scores
    return {"program": FORWARD_PROGRAM, "recipe": RECIPE, "ced_prefill": ced_prefill,
            "index_scores": index_scores, "attention_forward": forward,
            "replay_window": cfg.replay_window, "boundary": cfg.boundary,
            "encoder_layers": len(cfg.encoder), "decoder_layers": len(cfg.decoder),
            "layer_sources": tuple(cfg.layer_sources), "moe": MOE_RECIPE, "attention": spec.RECIPE,
            "sink_recipe": SINK_RECIPE, "index_query_format": spec.INDEX_QUERY_FORMAT,
            "cache_records": {"kv": spec.KV_RECORD_BYTES, "index_key": spec.INDEX_RECORD_BYTES},
            "excluded": tuple(sorted(cfg.excluded))}


def program_recorded() -> dict:
    """Switches reported beside the program identity but not compared: the requested transport fallbacks (same
    bits), the trainer's attention-backward provider (not a forward program) and the trained set (``actor``, or
    ``actor_routed_experts_frozen`` under the debug-layout switch ``ISOEXEC_DS_FREEZE_ROUTED_EXPERTS``)."""
    import os

    from .providers import EP_EXCHANGE_ENV, TP_REDUCER_ENV
    from .training import trained_set
    return {"tp_reducer": os.environ.get(TP_REDUCER_ENV, "tree"), "ep_exchange": os.environ.get(EP_EXCHANGE_ENV, "mega"),
            "attention_backward": os.environ.get("ISOEXEC_DSA_BWD", "cudnn_det"), "trained_set": trained_set()}


__all__ = [
    "ARCHITECTURE",
    "BOUNDED_REPLAY",
    "CED_PREFILL_MODES",
    "FORWARD_PROGRAM",
    "FULL_DECODER",
    "MODEL_TYPE",
    "MOE_RECIPE",
    "RECIPE",
    "RELEASE_CED_PREFILL",
    "SINK_RECIPE",
    "CedError",
    "CedGeometry",
    "Csa2Mode",
    "DeepseekV41Config",
    "Geometry",
    "LayerKind",
    "ced_geometry",
    "program_identity",
    "program_recorded",
    "register",
    "text_geometry",
    "validate_ced",
]
