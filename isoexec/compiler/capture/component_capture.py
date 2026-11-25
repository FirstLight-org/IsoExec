"""Explicit exact-class registration for opaque Component capture boundaries."""

import inspect
from typing import cast

from torch import nn

import isoexec.nn as inn


_BUILTINS = frozenset(cast(type[inn.Component], getattr(inn, name)) for name in inn.COMPONENTS)
_SEQUENCE_BUILTINS = frozenset({inn.Attention, inn.GDN, inn.LatentAttention, inn.MLP, inn.MoE, inn.DecoderBlock})
_REGISTERED: dict[type[inn.Component], tuple[object, bool]] = {
    kind: (kind.forward, kind in _SEQUENCE_BUILTINS) for kind in _BUILTINS
}


def register_capture_component(
    kind: type[inn.Component], *, sequence_context: bool | None = None,
) -> type[inn.Component]:
    """Register the exact definition in each worker before capture.

    This is an explicit author contract: abstract output, facts, tensor axes,
    state, layout and OpTable placement still belong to the Component. It does
    not register subclasses or authorize changing the definition afterward.
    sequence_context=True explicitly allows capture to supply an omitted seq
    argument from its internal input; explicitly passed values are preserved.
    """
    if (not isinstance(kind, type) or not issubclass(kind, inn.Component)
            or not callable(kind.forward) or kind.forward is nn.Module.forward):
        raise TypeError("Capture registration requires a concrete Component class with forward()")
    previous = _REGISTERED.get(kind)
    if previous is not None and previous[0] is not kind.forward:
        raise ValueError("A registered Component forward changed; define and register a new exact class")
    supplied = previous[1] if previous is not None else False
    if sequence_context is not None:
        if type(sequence_context) is not bool:
            raise TypeError("sequence_context must be a bool")
        supplied = sequence_context
    if supplied:
        parameter = inspect.signature(kind.forward).parameters.get("seq")
        if parameter is None or parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            raise ValueError("Sequence capture registration requires an explicit seq parameter")
    if previous is not None and supplied != previous[1]:
        raise ValueError("A registered Component sequence contract changed; register a new exact class")
    _REGISTERED[kind] = (kind.forward, supplied)
    return kind


def is_capture_component(module: nn.Module) -> bool:
    """Require exact-class registration and its unchanged forward definition."""
    if not isinstance(module, inn.Component):
        return False
    kind = type(module)
    registered = _REGISTERED.get(kind)
    return registered is not None and registered[0] is kind.forward


def supplies_sequence_context(module: nn.Module) -> bool:
    return isinstance(module, inn.Component) and is_capture_component(module) and _REGISTERED[type(module)][1]


def is_builtin_capture_component(module: nn.Module) -> bool:
    return type(module) in _BUILTINS and is_capture_component(module)


__all__ = ["register_capture_component"]
