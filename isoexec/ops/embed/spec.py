"""Declarations for embedding lookup, a pure row gather."""

from __future__ import annotations

from isoexec.core.registry import ImplSpec

_SM90 = frozenset({"sm90", "sm100"})

IMPLS: tuple[ImplSpec, ...] = (ImplSpec(op="embed.lookup", impl_id="eager", version=1, supported_archs=_SM90),)
