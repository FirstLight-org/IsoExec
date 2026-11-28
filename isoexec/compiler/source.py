"""A place to build meta device model.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from torch import nn
    from transformers import PretrainedConfig


class ModelSourceError(ValueError):
    """An author package does not satisfy the model package construction contract."""


@dataclass(frozen=True)
class ModelSource:
    """Identify an author package without importing it.

    Packages provide HF configuration and an ``nn.Module`` in ``model.py``
    named by ``architectures``. ``weight.py`` declares Hugging Face checkpoint conversions.
    The package must be importable in each worker; this class does not deploy it.
    """

    package: str

    def __post_init__(self) -> None:
        _ = _package_name(self.package)

    def to_dict(self) -> dict[str, str]:
        """Return the explicit JSON-compatible source fields for worker transport."""
        return {"package": self.package}

    @classmethod
    def from_dict(cls, data: object) -> ModelSource:
        """Validate source fields without importing or reconstructing author code."""
        if not isinstance(data, Mapping):
            raise ModelSourceError("source data must contain exactly 'package'")
        fields = cast(Mapping[object, object], data)
        if set(fields) != {"package"}:
            raise ModelSourceError("source data must contain exactly 'package'")
        return cls(package=_package_name(fields["package"]))

    def _module(self, role: str) -> ModuleType:
        name = f"{self.package}.{role}"
        try:
            return import_module(name)
        except ModuleNotFoundError as exc:
            if exc.name and (name == exc.name or name.startswith(exc.name + ".")):
                raise ModelSourceError(
                    f"cannot import {name!r}; model packages provide model.py and weight.py "
                    + "on the worker's Python path"
                ) from exc
            raise  # Preserve a missing dependency inside an existing author module.

    def build_meta(self, raw_config: Mapping[str, object]) -> nn.Module:
        """Reconstruct the author's module with registered tensors on meta.

        Config parsing receives a private copy. Placement, checkpoint loading,
        graph validation and runtime installation belong to subsequent stages.
        """
        module = self._module("model")
        import torch
        from torch import nn

        from isoexec.compiler.capture.capture_context import capture_defaults

        with capture_defaults():
            config = self.build_config(raw_config)
            architectures = config.architectures
            assert architectures is not None
            candidates = [getattr(module, name, None) for name in architectures]
            factory = next((candidate for candidate in candidates
                            if isinstance(candidate, type) and issubclass(candidate, nn.Module)), None)
            if factory is None:
                raise ModelSourceError(f"{module.__name__} must define an nn.Module from architectures={architectures}")
            with torch.device("meta"):
                model = factory(config)
        if not isinstance(model, nn.Module):
            name = getattr(factory, "__name__", "model factory")
            raise ModelSourceError(f"{self.package}.model.{name}(config) must return an nn.Module")
        for kind, tensors in (("parameter", model.named_parameters()), ("buffer", model.named_buffers())):
            for name, tensor in tensors:
                if not tensor.is_meta:
                    raise ModelSourceError(
                        f"{self.package}.model: {kind} {name!r} is on {tensor.device}; "
                        + "model construction requires meta tensors before placement/loading"
                    )
        return model

    def build_config(self, raw_config: Mapping[str, object]) -> PretrainedConfig:
        """Parse the same private HF config for model construction and weight conversion."""
        from transformers import AutoConfig

        data = _json_config(deepcopy(dict(_raw_config_mapping(raw_config))))
        model_type = data.pop("model_type", None)
        if not isinstance(model_type, str) or not model_type:
            raise ModelSourceError("HF config.json requires model_type")
        architectures = data.get("architectures")
        if (not isinstance(architectures, list) or not architectures
                or any(not isinstance(name, str) or not name.isidentifier() for name in architectures)):
            raise ModelSourceError("HF config.json requires a nonempty architectures list of class names")
        # A package that ships its own config class names it (``CONFIG_CLASS`` in model.py): the AutoConfig
        # mapping is process-wide, and a runtime may have registered another class for the same model_type.
        declared = getattr(self._module("model"), "CONFIG_CLASS", None)
        if declared is not None:
            if getattr(declared, "model_type", None) != model_type:
                raise ModelSourceError(f"CONFIG_CLASS parses model_type {declared.model_type!r}, not {model_type!r}")
            return declared(**data)
        return AutoConfig.for_model(model_type, **data)

    def apply_overrides(self, base: Mapping[str, object], patch: Mapping[str, object]) -> dict[str, Any]:
        """Resolve effective JSON before worker transport, without constructing a model.

        Nested mappings merge recursively; other values replace the base value.
        Both inputs are private JSON copies; no author code is imported here.
        """
        if not isinstance(base, Mapping) or not isinstance(patch, Mapping):
            raise ModelSourceError("Model configuration overrides require JSON mappings")
        base_data, patch_data = _json_config(dict(base)), _json_config(dict(patch))
        return _merge_config(base_data, patch_data)

    def weight_module(self) -> ModuleType:
        """Resolve weight.py without imposing a checkpoint-loading API here."""
        return self._module("weight")


def _merge_config(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """Apply explicit nested HF config overrides without discarding sibling fields."""
    result = dict(base)
    for key, value in patch.items():
        previous = result.get(key)
        result[key] = _merge_config(previous, value) if isinstance(previous, dict) and isinstance(value, dict) else value
    return result


def _package_name(value: object) -> str:
    if (
        not isinstance(value, str)
        or not all(part.isidentifier() for part in value.split("."))
        or value.split(".")[0] == "__main__"
    ):
        raise ModelSourceError(
            "model package must be an importable dotted Python package, not a "
            + "checkpoint path or __main__; e.g. 'my_models.decoder'"
        )
    return value


def _raw_config_mapping(raw_config: object) -> Mapping[object, object]:
    if not isinstance(raw_config, Mapping):
        raise ModelSourceError("build_meta() requires a mapping of raw model configuration")
    return cast(Mapping[object, object], raw_config)


def _json_config(value: object) -> dict[str, Any]:
    """Copy ordinary JSON without coercing key collisions or custom Python values."""
    def validate(item: object) -> None:
        if type(item) is dict:
            if any(type(key) is not str for key in item):
                raise ModelSourceError("Model configuration overrides require string keys at every JSON mapping")
            for child in item.values():
                validate(child)
        elif type(item) is list:
            for child in item:
                validate(child)
        elif type(item) not in (str, int, float, bool, type(None)):
            raise ModelSourceError("Model configuration overrides require ordinary JSON-compatible values")

    if type(value) is not dict:
        raise ModelSourceError("Model configuration overrides require an ordinary JSON mapping")
    try:
        encoded = json.dumps(value, allow_nan=False)
    except (TypeError, ValueError, RecursionError) as exc:
        raise ModelSourceError("Model configuration overrides require finite JSON-compatible values") from exc
    try:
        validate(value)  # Reject key coercion/custom types before consuming encoded data.
    except RecursionError as exc:
        raise ModelSourceError("Model configuration overrides require finite JSON-compatible values") from exc
    return cast(dict[str, Any], json.loads(encoded))
