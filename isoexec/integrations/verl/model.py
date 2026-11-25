"""Resolve the IsoExec model package and source configuration for a local checkpoint."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from isoexec.compiler.source import ModelSource, _json_config

#: Built-in packages by Hugging Face ``model_type``. Another package must be named explicitly.
PACKAGES = {
    "qwen3_5": "isoexec.models.qwen3_5",
    "qwen3_5_moe": "isoexec.models.qwen3_5_moe",
    "qwen2_moe": "isoexec.models.qwen2_moe",
    "qwen3": "isoexec.models.qwen3",
    "qwen3_moe": "isoexec.models.qwen3_moe",
}


def checkpoint_directory(path: str) -> Path:
    """A local checkpoint directory; verl workers and vLLM read the same snapshot."""
    from isoexec.compiler.weights import local_checkpoint_path

    resolved = local_checkpoint_path(path)
    if not resolved.is_dir() or not (resolved / "config.json").is_file():
        raise ValueError(f"IsoExec verl integration requires a local checkpoint directory with config.json: {path}")
    return resolved


def source_config(path: str, package: str | None = None) -> tuple[ModelSource, dict[str, Any]]:
    """Return the model source and the checkpoint's unmodified ``config.json``."""
    directory = checkpoint_directory(path)
    raw = _json_config(json.loads((directory / "config.json").read_text()))
    if package is None:
        model_type = raw.get("model_type")
        package = PACKAGES.get(model_type) if isinstance(model_type, str) else None
        if package is None:
            raise ValueError(f"No built-in IsoExec model package for model_type={model_type!r}; "
                             f"built-in packages: {sorted(PACKAGES)}")
    return ModelSource(package=package), raw


def describe(path: str, package: str | None = None) -> Mapping[str, Any]:
    source, raw = source_config(path, package)
    return {"package": source.package, "model_type": raw.get("model_type"), "path": str(checkpoint_directory(path))}


__all__ = ["PACKAGES", "checkpoint_directory", "describe", "source_config"]
