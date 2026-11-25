"""Assemble family-owned reference bindings without loading accelerator kernels.
Duplicate operation claims refuse, and reference IDs remain outside production equivalence classes."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from isoexec.core.bindings import BoundOp
from isoexec.ops._families import FAMILIES

REFERENCE_FAMILIES = tuple(family.package for family in FAMILIES)


@dataclass(frozen=True)
class ReferencePlan:
    """Caller choices shared across family reference builders.
    conventions selects norm gamma storage, leaves/leaf_dtype specify the reduction plan,
    and selectors supplies per-op facts; None leaf fields use collectives-family defaults."""

    conventions: Mapping[str, str] = field(default_factory=dict)
    leaves: int | None = None
    leaf_dtype: str | None = None
    selectors: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


def family_bindings(family: str, plan: ReferencePlan) -> Mapping[str, BoundOp]:
    """The op ids ``ops/<family>/reference.py`` states, bound for ``plan``.

    Imported through ``__package__`` rather than spelled ``isoexec.ops.``, for the reason
    ``ops/_resolve.py::package_module`` gives: a second statement of where this tree is would be
    wrong first in exactly the case that matters, a vendored or renamed copy.
    """
    module = importlib.import_module(f"{__package__}.{family}.reference")
    return module.bindings(plan)


def reference_bindings(plan: ReferencePlan) -> dict[str, BoundOp]:
    """Every family's contribution, merged, refusing an op id two families both claim.

    A duplicate is not a merge conflict to resolve by import order -- it is two references for one
    op id, and which of them a table ends up holding would depend on the order of a tuple above.
    """
    out: dict[str, BoundOp] = {}
    owner: dict[str, str] = {}
    for family in REFERENCE_FAMILIES:
        for op, binding in family_bindings(family, plan).items():
            if op in out:
                raise ValueError(
                    f"reference_table: op id {op!r} is stated by both ops/{owner[op]}/reference.py "
                    f"({out[op].impl}) and ops/{family}/reference.py ({binding.impl}); one op "
                    f"id has one reference, and which of two a table held would depend on import order"
                )
            out[op], owner[op] = binding, family
    return out


__all__ = [
    "REFERENCE_FAMILIES",
    "ReferencePlan",
    "family_bindings",
    "reference_bindings",
]
