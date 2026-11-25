"""Assemble the implementation registry from family spec modules without importing kernels."""

from __future__ import annotations

import importlib

from isoexec.core.registry import Registry, RegistryError

# Keep the catalog import independent of the installed package name.
_ISOEXEC_ROOT = __name__.rsplit(".core.", 1)[0]
_FAMILIES = tuple(f.package for f in importlib.import_module(f"{_ISOEXEC_ROOT}.ops._families").FAMILIES)


def declare_into(reg: Registry, mod) -> Registry:
    """Read a family's IMPLS and add them to the registry. Direct attribute access preserves errors
    from lazy architecture dispatch.
    """
    reg.add(*mod.IMPLS)
    return reg


def build_registry() -> Registry:
    """Build and validate the complete registry, refusing missing family declarations."""
    reg = Registry()
    missing = []
    for fam in _FAMILIES:
        try:
            mod = importlib.import_module(f"{_ISOEXEC_ROOT}.ops.{fam}.spec")
        except ModuleNotFoundError as exc:
            if exc.name not in (f"{_ISOEXEC_ROOT}.ops.{fam}", f"{_ISOEXEC_ROOT}.ops.{fam}.spec"):
                raise
            missing.append(fam)
            continue
        declare_into(reg, mod)
    if missing:
        raise RegistryError(f"op families missing a spec module: {missing}")
    problems = reg.lint()
    if problems:
        raise RegistryError("registry declaration lint failed:\n  " + "\n  ".join(problems))
    return reg
