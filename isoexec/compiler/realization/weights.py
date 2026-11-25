"""Primitives for converting and loading HF weights."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Collection, Iterable, Mapping
from itertools import chain
from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch
from torch import nn
from transformers.core_model_loading import WeightConverter, WeightRenaming, rename_source_key

from isoexec.compiler.source import ModelSource, ModelSourceError

from .tp import global_shape, slice_weight

if TYPE_CHECKING:
    from transformers import PretrainedConfig

VALID_DTYPES = (torch.bfloat16, torch.float16, torch.float32, torch.float64)


def ignored_patterns(hf_config: PretrainedConfig) -> tuple[str, ...]:
    """Return unexpected weight names excluded from native loading."""
    # Match the existing Qwen3_5ForConditionalGeneration declarations during migration.
    # TODO(jiangjiawei1103): Better handling for ignored patterns.
    patterns: tuple[str, ...] = ()
    if hf_config.model_type in ("qwen3_5", "qwen3_5_moe"):
        patterns = (r"^model\.visual\.", r"^mtp\.")
    # A tied head is the embedding table; some tied checkpoints (Qwen3-0.6B..4B) still ship lm_head.
    if hf_config.model_type in ("qwen3_5", "qwen3_5_moe", "qwen3", "qwen3_moe") and hf_config.tie_word_embeddings:
        patterns += (r"^lm_head\.weight$",)
    return patterns


def _named_tensors(module: nn.Module) -> dict[str, torch.Tensor]:
    """Return all parameters and buffers of ``module``.

    Tensors are keyed by fully qualified name. Shared tensors are kept under
    every name they appear at, e.g., ``embed.weight`` and ``lm_head.weight``.
    """
    params = module.named_parameters(remove_duplicate=False)
    buffers = module.named_buffers(remove_duplicate=False)
    return dict(chain(params, buffers))


def _is_allocated_floating(tensor: object) -> bool:
    return isinstance(tensor, torch.Tensor) and not tensor.is_meta and tensor.dtype in VALID_DTYPES


def _conversion_mapping(
    source: ModelSource, hf_config: PretrainedConfig,
) -> tuple[list[WeightRenaming], list[WeightConverter]]:
    """Read checkpoint conversions from the selected model package."""
    factory = getattr(source.weight_module(), "checkpoint_conversion_mapping", None)
    if not callable(factory):
        raise ModelSourceError(f"{source.package}.weight must define checkpoint_conversion_mapping(config)")
    transforms = factory(hf_config)
    if not isinstance(transforms, Iterable):
        raise TypeError("checkpoint_conversion_mapping(config) must return an iterable of HF conversion entries")
    renamings, converters = [], []
    for transform in transforms:
        if isinstance(transform, WeightConverter):
            if len(transform.source_patterns) != 1:
                raise ValueError("Weight loading currently requires one source pattern per converter")
            converters.append(transform)
        elif isinstance(transform, WeightRenaming):
            renamings.append(transform)
        else:
            raise TypeError("checkpoint_conversion_mapping must return HF WeightRenaming/WeightConverter entries")
    return renamings, converters


def _find_mapping(
    hf_weight_name: str,
    renamings: list[WeightRenaming],
    converters: list[WeightConverter],
) -> tuple[WeightConverter | None, str | None, tuple[str, ...]]:
    """Return the matched converter, its source pattern, and the logical names.

    As in HF ``rename_source_key``, every renaming applies, then at most one
    converter. A converter renames the key to its first target pattern; the
    other logical names share that prefix and suffix, as in
    ``WeightConverter.convert``.
    """
    renamed_key, _ = rename_source_key(hf_weight_name, renamings, [])
    for converter in converters:
        logical_name, source_pattern = converter.rename_source_key(renamed_key)
        if source_pattern is None:
            continue
        first_pattern = converter.target_patterns[0]
        prefix, found, suffix = logical_name.partition(first_pattern)
        if not found:
            raise ValueError(f"{hf_weight_name}: renamed key {logical_name!r} lacks target pattern {first_pattern!r}")
        return converter, source_pattern, tuple(prefix + target + suffix for target in converter.target_patterns)
    return None, None, (renamed_key,)


def _write(
    dst: torch.Tensor,
    src: torch.Tensor,
    logical_name: str,
    loaded: defaultdict[torch.Tensor, set[str]],
) -> None:
    """Copy ``src`` into the registered ``dst`` and record ``logical_name``.

    A shared tensor is copied once; later logical names must match it bitwise.
    Note that ``loaded`` has side effect.
    """
    if not _is_allocated_floating(dst):
        raise ValueError(f"{logical_name}: expected an allocated floating registered tensor")
    if not _is_allocated_floating(src):
        raise ValueError(f"{logical_name}: conversion must return an allocated floating tensor")
    if src.shape != dst.shape:
        raise ValueError(f"{logical_name}: loaded shape {tuple(src.shape)} != registered shape {tuple(dst.shape)}")
    if dst in loaded:
        # Bitwise parity for shared tensor; make byte views valid for strided inputs.
        src = src.to(dtype=dst.dtype).contiguous()
        current = dst.detach().to(device=src.device).contiguous()
        if not torch.equal(current.reshape(-1).view(torch.uint8), src.reshape(-1).view(torch.uint8)):
            raise ValueError(f"{logical_name}: conflicting values for a shared tensor in one load_weights call")
    else:
        # Like vLLM's default_weight_loader, copy_ casts to dst.dtype automatically.
        # https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/model_loader/weight_utils.py#L1244-L1258
        dst.copy_(src)
    loaded[dst].add(logical_name)


def _global_shapes(module: nn.Module, tp_size: int, side: str, etp_size: int | None) -> SimpleNamespace:
    """Expose ``get_parameter`` at global TP shapes for HF conversion operations."""

    def get_parameter(name: str) -> torch.Tensor:
        parameter = module.get_parameter(name)
        shape = global_shape(module, name, parameter, tp_size=tp_size, side=side, etp_size=etp_size)
        return torch.empty(shape, dtype=parameter.dtype, device="meta")

    return SimpleNamespace(get_parameter=get_parameter)


@torch.no_grad()
def load_weights(
    module: nn.Module,
    weights: Iterable[tuple[str, torch.Tensor]],
    *,
    source: ModelSource,
    hf_config: PretrainedConfig,
    tp_size: int = 1,
    tp_rank: int = 0,
    side: str | None = None,
    remote: Collection[str] = (),
    aliases: Mapping[str, str] | None = None,
    etp_size: int | None = None,
) -> set[str]:
    """Load HF weights using ``source``'s weight.py and return the loaded logical names.

    An HF weight name is a checkpoint key. A logical name is the fully qualified
    name of a parameter or buffer registered in ``module``; a shared tensor has
    several, e.g., ``embed.weight`` and ``head.weight``. Each HF weight is mapped
    to its logical names, converted, and written into the registered tensors.
    With TP, conversion sees global tensors and global target shapes; each
    registered Parameter then receives this rank's slice of its declared axis.
    ``side`` ("trainer" or "engine") selects how heads below TP are held.
    Expert tensors split over ``etp_size`` ranks, which defaults to TP.
    With PP or EP, ``remote`` names belong to other stages or EP ranks and are
    skipped, and ``aliases`` resolve tied names to this stage's registered
    names.
    Copies completed before an error are not rolled back.
    """
    if tp_size > 1 and side not in ("trainer", "engine"):
        raise ValueError(f"TP={tp_size} loading requires side='trainer' or side='engine', got {side!r}")
    # TODO(jiangjiawei1103): Add tests.
    renamings, converters = _conversion_mapping(source, hf_config)
    ignored = ignored_patterns(hf_config)

    registered = _named_tensors(module)
    aliases = aliases or {}
    # HF operations such as Split size targets from registered shapes, which are TP-local.
    shapes = module if tp_size == 1 else _global_shapes(module, tp_size, side, etp_size)
    loaded: defaultdict[torch.Tensor, set[str]] = defaultdict(set)
    for hf_weight_name, tensor in weights:
        # TODO(jiangjiawei1103): Refactor find mapping.
        converter, source_pattern, logical_names = _find_mapping(hf_weight_name, renamings, converters)
        if remote and all(name in remote for name in logical_names):
            continue
        missing = [name for name in logical_names if name not in registered and name not in aliases and name not in remote]
        if len(missing) == len(logical_names) and all(any(re.search(p, name) for p in ignored) for name in missing):
            continue
        if missing:
            if logical_names == (hf_weight_name,):
                raise ValueError(f"Unknown HF weight name: {hf_weight_name}")
            raise ValueError(f"{hf_weight_name}: mapped logical name is not registered: {missing}")
        if not _is_allocated_floating(tensor):
            raise ValueError(f"{hf_weight_name}: expected an allocated floating tensor")

        converted = {logical_names[0]: tensor}
        if converter is not None:
            # Clone so an in-place ConversionOps cannot modify the supplied tensor.
            converter.collected_tensors[source_pattern] = [tensor.clone()]
            converted = converter.convert(logical_names[0], model=shapes, config=hf_config)
            if set(converted) != set(logical_names):
                raise ValueError(f"{hf_weight_name}: converted logical names do not match {logical_names}")
        for name, src in converted.items():
            if name in remote:
                continue
            logical_name = aliases.get(name, name)
            dst = registered[logical_name]
            if tp_size > 1 and isinstance(dst, nn.Parameter):
                expected = global_shape(module, logical_name, dst, tp_size=tp_size, side=side, etp_size=etp_size)
                if tuple(src.shape) != expected:
                    raise ValueError(f"{logical_name}: loaded shape {tuple(src.shape)} != global shape {expected}")
                src = slice_weight(module, logical_name, src, tp_size=tp_size, tp_rank=tp_rank, side=side,
                                   etp_size=etp_size)
            _write(dst, src, logical_name, loaded)
    return set().union(*loaded.values())
