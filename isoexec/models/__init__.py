"""Built-in model packages and the source descriptor that selects one.

Each architecture package owns its model and weight mapping. The selected local
directory or Hugging Face snapshot supplies ``config.json`` and safetensors.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from isoexec.compiler.source import ModelSource, ModelSourceError

if TYPE_CHECKING:
    from torch import nn


def build_component_model(source: ModelSource, hf_config: Mapping[str, object]) -> nn.Module:
    """Construct an unplaced model from its source package and HF config."""
    if not isinstance(source, ModelSource):
        raise TypeError("build_component_model requires a ModelSource")
    return source.build_meta(hf_config)


__all__ = ["ModelSource", "ModelSourceError", "build_component_model"]
