"""Portable declarations assembled for one authored model."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from isoexec.state.contract import StateDecl, StateRef


def _empty_op_facts() -> Mapping[str, object]:
    return MappingProxyType({})


@dataclass(frozen=True)
class ModelContract:
    """The model's portable state declarations, references, and op facts."""

    state_decls: tuple[StateDecl, ...]
    state_refs: tuple[StateRef, ...]
    op_facts: Mapping[str, object] = field(default_factory=_empty_op_facts)


__all__ = ["ModelContract"]
