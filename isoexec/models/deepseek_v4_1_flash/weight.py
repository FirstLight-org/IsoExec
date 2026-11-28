"""Released DeepSeek V4.1 Flash checkpoint names and the exact FP8/FP4 -> BF16 master decode.

Component parameters keep the checkpoint's names (``layers.{i}.attn.wq_a.weight``,
``layers.{i}.ffn.experts.{e}.w1.weight``...). Each FP8 (E4M3, 32 x 32 UE8M0 blocks) or FP4 (packed E2M1,
1 x 32 UE8M0 blocks) projection is one many-to-one converter: its ``.weight`` codes and ``.scale`` decode to
the BF16 master (``isoexec.core.checkpoint_quant.decode_checkpoint_weight``: exact, every decoded value is
representable in BF16). Engram tables load their FP8 rows and UE8M0 scales byte for byte. Rules see whole
tensors; the loader takes this rank's TP slice afterwards, and experts other EP ranks own are remote names.

A reduced-depth checkpoint (``isoexec_subset``) renames its rows' checkpoint layers (``layout: "renamed"``; a
``"physical"`` checkpoint already stores its rows as ``layers.{row}``) and keeps a prefix of the routed experts,
of every expert's channels and of the router rows, exactly as stored. A checkpoint without a
``quantization_config`` stores BF16 tensors and loads them as they are.

Module paths follow the CED (``stack_name``): ``layers.{i}.*`` of an encoder layer lands in
``encoder.blocks.{i}.*``; the decoder FULL layer's attention boundary and token-wise projection
(``ENTRY_PARTS``) in ``decoder.entry.*``; every other decoder parameter of layer ``i`` in
``decoder.blocks.{i - b}.*``. ``checkpoint_name`` is the inverse (weight streaming keeps checkpoint names).
"""
from __future__ import annotations

import torch
from transformers.core_model_loading import ConversionOps, WeightConverter, WeightRenaming

from .config import text_geometry

#: FP8 projections of the decoder rows, matched on the checkpoint key suffix (order matters: the indexer
#: and Engram projections first, so ``attn.wq_b`` and ``attn.wkv`` never claim them).
_FP8 = ("attn.indexer.wq_b", "engram.wkv", "attn.wq_a", "attn.wq_b", "attn.wkv", "attn.wo_a", "attn.wo_b",
        "shared_experts.w1", "shared_experts.w2", "shared_experts.w3")
_FP4 = ("w1", "w2", "w3")     # routed experts (ffn.experts.{e}.w*)
#: Towers and rows of the released checkpoint this package does not build (vision and DSpark, declared).
_NOT_BUILT = (r"^mtp\.", r"^vision\.", r"^aligner\.", r"^image_(start|end|newline)$", r"\.ffn\.gate\.bias_vl$")
#: The decoder FULL layer's parameters owned by ``decoder.entry`` (relative to ``layers.{b}.``).
ENTRY_PARTS = ("hc_attn_fn", "hc_attn_base", "hc_attn_scale", "attn_norm.", "attn.wkv.", "attn.kv_norm.",
               "attn.compressor.", "attn.indexer.wk.", "attn.indexer.k_norm.")


def _entry_part(rest: str) -> bool:
    return any(rest == p or rest.startswith(p) for p in ENTRY_PARTS)


def stack_name(name: str, cfg) -> str:
    """Checkpoint name (``layers.{row}.*``, rows of this checkpoint) -> module path of the CED definition."""
    if not name.startswith("layers."):
        return name
    _, i, rest = name.split(".", 2)
    i, b = int(i), cfg.boundary
    if i < b:
        return f"encoder.blocks.{i}.{rest}"
    if i == b and _entry_part(rest):
        return f"decoder.entry.{rest}"
    return f"decoder.blocks.{i - b}.{rest}"


def checkpoint_name(path: str, cfg) -> str:
    """Module path -> checkpoint name (``layers.{row}.*``), the inverse of ``stack_name``."""
    b = cfg.boundary
    for prefix, offset in (("encoder.blocks.", 0), ("decoder.blocks.", b)):
        if path.startswith(prefix):
            j, rest = path[len(prefix):].split(".", 1)
            return f"layers.{int(j) + offset}.{rest}"
    if path.startswith("decoder.entry."):
        return f"layers.{b}.{path[len('decoder.entry.'):]}"
    return path


def _escape(name: str) -> str:
    return name.replace(".", r"\.")


def _leading(value: torch.Tensor, shape: tuple[int, ...], allow: bool, name: str) -> torch.Tensor:
    if tuple(value.shape) == tuple(shape):
        return value
    if not allow or len(value.shape) != len(shape) or any(a < b for a, b in zip(value.shape, shape)):
        raise ValueError(f"{name}: checkpoint shape {tuple(value.shape)} != {tuple(shape)}")
    return value[tuple(slice(0, n) for n in shape)]


class DecodeBlocks(ConversionOps):
    """``{weight codes, block scales}`` -> the BF16 master. A reduced checkpoint keeps the leading rows/channels
    of its experts (codes and their 32-aligned scale blocks), sliced before the decode."""

    def __init__(self, *, reduced: bool = False) -> None:
        self.reduced = reduced

    @torch.no_grad()
    def convert(self, input_dict, source_patterns, target_patterns, full_layer_name=None, model=None, **kwargs):
        from isoexec.core.checkpoint_quant import decode_checkpoint_weight
        weight, scale = (input_dict[pattern][0] for pattern in source_patterns)
        n, k = model.get_parameter(full_layer_name).shape
        if weight.dtype in (torch.int8, torch.uint8):
            storage, packed = "deepseek_fp4", (n, k // 2)
            blocks = (n, -(-k // 32))
        elif weight.dtype == torch.float8_e4m3fn:
            storage, packed = "deepseek_fp8", (n, k)
            blocks = (-(-n // 32), -(-k // 32))
        else:
            raise TypeError(f"{full_layer_name}: checkpoint codes of dtype {weight.dtype} are not FP8/FP4")
        weight = _leading(weight, packed, self.reduced, full_layer_name).contiguous()
        scale = _leading(scale, blocks, self.reduced, full_layer_name).contiguous()
        value = decode_checkpoint_weight(weight, scale, storage).to(torch.bfloat16)
        return {target_patterns[0]: value}


class LeadingRows(ConversionOps):
    """A reduced checkpoint's router keeps the rows of its leading experts."""

    @torch.no_grad()
    def convert(self, input_dict, source_patterns, target_patterns, full_layer_name=None, model=None, **kwargs):
        (value,) = (item for values in input_dict.values() for item in values)
        shape = model.get_parameter(full_layer_name).shape
        return {target_patterns[0]: _leading(value, tuple(shape), True, full_layer_name).contiguous()}


def _decode(name: str, reduced: bool) -> WeightConverter:
    stem = _escape(name)
    return WeightConverter(source_patterns=[rf"\.{stem}\.weight$", rf"\.{stem}\.scale$"],
                           target_patterns=f".{name}.weight", operations=[DecodeBlocks(reduced=reduced)])


def checkpoint_conversion_mapping(config) -> list[WeightConverter | WeightRenaming]:
    """Checkpoint key -> component parameter, with the FP8/FP4 decode; rules see whole tensors."""
    cfg = text_geometry(config)
    mapping: list[WeightConverter | WeightRenaming] = []
    if cfg.subset and cfg.layout == "renamed":
        # Checkpoint layer s of row i: every checkpoint layer moves aside first, then each kept one lands on its
        # row, so a dropped layer can never alias a kept row's name.
        mapping.append(WeightRenaming(r"^layers\.(\d+)\.", r"checkpoint_layers.\1."))
        mapping += [WeightRenaming(rf"^checkpoint_layers\.{source}\.", f"layers.{row}.")
                    for row, source in enumerate(cfg.layer_sources)]
    # Rows -> the CED stacks: the entry's parts of the boundary layer first, then one rule per layer.
    b = cfg.boundary
    parts = "|".join(_escape(p.rstrip(".")) + (r"\." if p.endswith(".") else "") for p in ENTRY_PARTS)
    mapping.append(WeightRenaming(rf"^layers\.{b}\.({parts})", r"decoder.entry.\1"))
    mapping += [WeightRenaming(rf"^layers\.{i}\.", f"encoder.blocks.{i}.") for i in range(b)]
    mapping += [WeightRenaming(rf"^layers\.{i}\.", f"decoder.blocks.{i - b}.")
                for i in range(b, cfg.num_hidden_layers)]
    if cfg.storage == "fp8_fp4":
        # Only experts are reduced; the routed ones follow the shared expert so it is never claimed as routed.
        mapping += [_decode(name, cfg.subset and name.startswith("shared_experts.")) for name in _FP8]
        mapping += [_decode(name, cfg.subset) for name in _FP4]
    if cfg.subset and cfg.layout == "renamed":
        mapping += [WeightConverter(source_patterns=rf"\.ffn\.gate\.{leaf}$", target_patterns=f".ffn.gate.{leaf}",
                                    operations=[LeadingRows()]) for leaf in ("weight", "bias")]
    return mapping


def ignored_checkpoint_patterns(config) -> tuple[str, ...]:
    """Checkpoint keys no component receives: the towers this package does not build and, for a reduced
    checkpoint, the dropped layers and routed experts."""
    cfg = text_geometry(config)
    patterns = list(_NOT_BUILT)
    if cfg.subset:
        patterns.append(r"^checkpoint_layers\.")
        E = cfg.n_routed_experts
        kept = "|".join(map(str, range(E)))
        patterns.append(rf"^(?:encoder|decoder)\.blocks\.\d+\.ffn\.experts\.(?!(?:{kept})\.)\d+\.")
    return tuple(patterns)


__all__ = [
    "ENTRY_PARTS",
    "DecodeBlocks",
    "LeadingRows",
    "checkpoint_conversion_mapping",
    "checkpoint_name",
    "ignored_checkpoint_patterns",
    "stack_name",
]
