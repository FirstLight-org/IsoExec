"""Shipped model-package geometries for component tests, built on meta through ``ModelSource``.

The configs live in ``fixtures/gemm_census_snapshot.json`` beside the census each one produces.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from torch import nn

from isoexec.compiler.source import ModelSource

_SNAPSHOT = Path(__file__).parent / "fixtures" / "gemm_census_snapshot.json"


def cases() -> dict[str, dict]:
    """``{name: {"package", "config", "sites"}}``, a fresh copy per call."""
    return json.loads(_SNAPSHOT.read_text(encoding="utf-8"))


def config(name: str) -> tuple[str, dict]:
    """``(package, raw HF config)`` of one named geometry."""
    case = cases()[name]
    return case["package"], copy.deepcopy(case["config"])


def build_meta(name: str) -> nn.Module:
    package, raw = config(name)
    return ModelSource(package).build_meta(raw)
