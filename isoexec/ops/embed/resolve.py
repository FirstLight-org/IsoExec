"""Resolve the embedding row gather; tensor-parallel callers provide vocabulary movement.
Reserved PLE lookup remains unregistered and refuses resolution."""

from __future__ import annotations

from typing import Any

from isoexec.ops._resolve import refuse, resolution

#: The impl id ``spec.py`` declares for the row gather.
LOOKUP_IMPL = "eager"


def resolve(op: str, *, site: str, **unused: Any):
    """The vocabulary row gather on the live arch."""
    if unused:
        raise refuse(op, site=site, asked=unused, on_offer=(),
                     why="the lookup takes no caller-supplied fact; the TP degree is recorded, not selected on.")
    if op != "embed.lookup":
        raise refuse(op, site=site, asked={}, on_offer=("embed.lookup",),
                     why="embed.ngram_hash_ple is reserved and deliberately unregistered.")
    from isoexec.ops.embed.eager import embed_lookup

    return resolution(op, LOOKUP_IMPL, embed_lookup, provider="ops.embed", site=site)


__all__ = ["LOOKUP_IMPL", "resolve"]
