"""Read the facts each component of a constructed model states about itself, as canonical JSON
data. A fact that has no plain-data form is refused, naming the component that holds it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from isoexec.checks import Refusal


class UnhashableFact(Refusal):
    """A component's ``facts()`` holds a value that has no canonical JSON form."""


def _plain(value: Any, *, where: str) -> Any:
    """Convert a fact to canonical JSON data; tuples become lists."""
    if isinstance(value, bool) or value is None or isinstance(value, (int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_plain(v, where=where) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _plain(v, where=where) for k, v in sorted(value.items())}
    raise UnhashableFact(
        f"config.facts: {where} holds {value!r} of type {type(value).__name__}, which has no canonical "
        f"JSON form; a fact must be plain data"
    )


def component_facts(model: Any) -> dict[str, dict]:
    """Return component facts by module path, including unexecuted subtrees whose weights are still
    loaded and synchronized.
    """
    from isoexec.nn.component import Component

    out: dict[str, dict] = {}
    for path, module in model.named_modules():
        if isinstance(module, Component):
            site = path or "<root>"
            out[site] = {
                str(key): _plain(value, where=f"{site}.facts().{key}") for key, value in sorted(module.facts().items())
            }
    return out


__all__ = ["UnhashableFact", "component_facts"]
