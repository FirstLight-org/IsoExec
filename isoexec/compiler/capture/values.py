"""Resolve pipeline scalar metadata from root paths, never tensor data."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import operator
from typing import Any, Literal, cast

import torch
from torch import fx


@dataclass(frozen=True)
class ValueSource:
    root: str
    path: tuple[tuple[Literal["attr", "item"], str | int], ...] = ()
    dimension: int | None = None

    def read(self, roots: Mapping[str, Any], tensors: Mapping[str, torch.Tensor]) -> Any:
        if self.root in roots:
            value = roots[self.root]
        elif self.root in tensors:
            value = tensors[self.root]
        else:
            raise ValueError(f"Pipeline metadata needs root {self.root!r}")
        for kind, key in self.path:
            value = getattr(value, str(key)) if kind == "attr" else operator.getitem(cast(Any, value), key)
        if self.dimension is not None:
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"Pipeline metadata root {self.root!r} must resolve to a Tensor")
            return value.shape[self.dimension]
        if type(value) not in (int, float, bool):
            raise ValueError(f"Pipeline metadata root {self.root!r} must resolve to a Python scalar")
        return value


def root_source(node: fx.Node) -> ValueSource | None:
    if node.op == "placeholder":
        return ValueSource(node.name)
    if node.op != "call_function" or not node.args or not isinstance(node.args[0], fx.Node):
        return None
    parent = root_source(node.args[0])
    if parent is None or len(node.args) != 2:
        return None
    key = node.args[1]
    if node.target is getattr and isinstance(key, str):
        return ValueSource(parent.root, (*parent.path, ("attr", key)))
    if node.target is operator.getitem and isinstance(key, (int, str)):
        return ValueSource(parent.root, (*parent.path, ("item", key)))
    return None


__all__ = ["ValueSource", "root_source"]
