"""Native input ownership, separate from authored graph and TP/SP movement."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
from torch import fx
from torch.utils import _pytree as pytree

from .values import ValueSource, root_source

if TYPE_CHECKING:
    from .export_boundary import CaptureSession


@dataclass(frozen=True)
class RootTensorLayout:
    """One native tensor leaf: global metadata or CP-local tokens on an axis.

    The runtime input provider owns these declarations. Ordinary model authors
    using its standard text contract need no extra file or model annotation.
    """

    root: str
    locality: Literal["global", "cp_local"]
    token_axis: int | None = None
    path: tuple[tuple[Literal["attr", "item"], str | int], ...] = ()

    def __post_init__(self) -> None:
        if (not isinstance(self.root, str) or not self.root or self.locality not in ("global", "cp_local")
                or (self.locality == "global" and self.token_axis is not None)
                or (self.locality == "cp_local" and type(self.token_axis) is not int)
                or any(kind not in ("attr", "item") or type(key) not in (str, int)
                       or (kind == "attr" and not isinstance(key, str)) for kind, key in self.path)):
            raise ValueError("Native root layout requires a tensor path and global or cp_local with a token axis")


def _source(session: CaptureSession, path: pytree.KeyPath) -> ValueSource | None:
    if not path or not isinstance(path[0], pytree.SequenceKey):
        return None
    selectors: list[tuple[Literal["attr", "item"], str | int]] = []
    for key in path[1:]:
        if isinstance(key, pytree.GetAttrKey):
            selectors.append(("attr", key.name))
        elif isinstance(key, pytree.SequenceKey):
            selectors.append(("item", key.idx))
        elif isinstance(key, pytree.MappingKey) and type(key.key) in (str, int):
            selectors.append(("item", key.key))
        else:
            return None
    return ValueSource(session.names[path[0].idx], tuple(selectors))


def bind_root_layouts(program: fx.GraphModule, session: CaptureSession,
                      declarations: Sequence[RootTensorLayout]) -> None:
    """Bind guarded Tensor identity aliases without retaining any input tensors."""
    leaves = [(path, value, _source(session, path)) for path, value in
              zip(session.leaf_paths, session.export_values, strict=True) if isinstance(value, torch.Tensor)]
    lookup = {source: value for _, value, source in leaves if source is not None}
    by_identity: dict[int, int | None] = {}
    declared: set[ValueSource] = set()
    for declaration in declarations:
        if not isinstance(declaration, RootTensorLayout):
            raise ValueError("Native root layouts must be RootTensorLayout records")
        source = ValueSource(declaration.root, declaration.path)
        if source in declared or source not in lookup:
            raise ValueError(f"Native root layout {source} is duplicate or does not name a tensor input leaf")
        declared.add(source)
        tensor = lookup[source]
        axis = declaration.token_axis
        if axis is not None:
            if tensor.ndim == 0 or not -tensor.ndim <= axis < tensor.ndim:
                raise ValueError(f"Native root layout {source} token axis {axis} is outside tensor rank {tensor.ndim}")
            axis %= tensor.ndim
        if id(tensor) in by_identity and by_identity[id(tensor)] != axis:
            raise ValueError(f"Native root layout {source} conflicts with a guarded Tensor alias locality/token axis")
        by_identity[id(tensor)] = axis
    layouts = {source: by_identity[id(value)] for _, value, source in leaves
               if source is not None and id(value) in by_identity}
    program.meta["isoexec_root_layout_missing"] = tuple(pytree.keystr(path) for path, value, source in leaves
                                                       if source is None or id(value) not in by_identity)
    program.meta["isoexec_root_layouts"] = tuple(layouts.items())
    for node in program.graph.nodes:
        source = root_source(node)
        if source in layouts:
            node.meta["isoexec_root_context_axis"] = layouts[source]


__all__ = ["RootTensorLayout", "bind_root_layouts"]
