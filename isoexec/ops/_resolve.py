"""Resolve production implementations through their owning families when bindings are requested.
BoundOp carries the selected implementation and registry version; unsupported requests refuse."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from functools import cache
from types import MappingProxyType
from typing import Any

from isoexec.core.bindings import BoundOp


class ResolutionError(RuntimeError):
    """No implementation of an op id serves the facts it was asked for, on the live arch."""


from isoexec.ops._families import FAMILIES

FAMILY_PACKAGES: Mapping[str, str] = MappingProxyType(
    {prefix: family.package for family in FAMILIES if not family.placement_owned for prefix in family.prefixes}
)


@cache
def _registry(arch: str, packages: tuple[tuple[str, str], ...] = ()):
    """The built registry, per ARCH and selection. Keyed on both because that is what it is a function
    of: every family's ``IMPLS`` is the SELECTED package's, so a registry cached across a change of arch
    or of a flag-selected package would answer for declarations the live process never selected."""
    del arch, packages  # the key, not arguments: `build_registry` reads them through `_dispatch`
    from isoexec.core.registry_build import build_registry

    return build_registry()


def registry():
    """The declarations of the live arch."""
    from isoexec.ops import _dispatch

    return _registry(_dispatch.ARCH, tuple(sorted(_dispatch.resolved().items())))


def declared_version(op: str, impl: str, *, site: str) -> int:
    """Return the recipe version only when the declaration admits the active architecture."""
    from isoexec.ops import _dispatch

    reg = registry()
    if reg.has_op(op):
        spec = reg.get_op(op).impls.get(impl)
        if spec is not None:
            if _dispatch.ARCH not in spec.supported_archs:
                raise ResolutionError(
                    f"{site}: {op}:{impl} is not proven on architecture {_dispatch.ARCH!r} "
                    f"(supported_archs: {sorted(spec.supported_archs)}); a declaration must admit "
                    "the active architecture before a production binding can be built"
                )
            return spec.version
    on_offer = sorted(reg.get_op(op).impls) if reg.has_op(op) else []
    raise ResolutionError(
        f"{site}: no ops/*/spec.py declares {op}:{impl} on this arch (declared for {op!r}: {on_offer}); "
        f"an impl id with no declaration behind it has no twin class and no battery, so it would reach "
        f"the peer digest as a difference rather than as a refusal"
    )


def kernel_package(seam: str, *, op: str, site: str) -> str:
    """The ops package ``_dispatch`` selects for ``seam`` on the live arch.

    Refuses the legacy arm (``selected() is None``) rather than answering for it: a seam whose arch
    row names no package has no kernel here, and the caller that reached this module did so because
    the arch says one exists.
    """
    from isoexec.ops import _dispatch

    package = _dispatch.selected(seam)
    if package is None:
        raise ResolutionError(
            f"{site}: seam {seam!r} selects no kernel package on arch {_dispatch.ARCH!r}, so {op!r} has no "
            f"accelerator implementation to resolve; the plain-torch statement of it is "
            f"ops/reference.py and requires an explicitly selected OpTable(mode='reference')"
        )
    return package


def package_module(package: str, module: str):
    """Import a module from the selected kernel package relative to this ops root."""
    return importlib.import_module(f"{__package__}.{package}.{module}")


def integration(package: str):
    """The selected package's callable factories and implementation identities."""
    return package_module(package, "integration")


def provider_of(package: str) -> str:
    """``ops.<package>`` -- the same spelling ``ops/table.accelerator_provider`` answers with."""
    return f"ops.{package}"


def resolution(
    op: str,
    impl: str,
    fn: Callable[..., Any],
    *,
    provider: str,
    site: str,
    owns: Mapping[str, str] | None = None,
    inputs: Mapping[str, Any] | None = None,
) -> BoundOp:
    """An :class:`BoundOp` with its version read off the declaration. The one constructor."""
    return BoundOp(
        op=op,
        impl=impl,
        version=declared_version(op, impl, site=site),
        provider=provider,
        fn=fn,
        owns=owns or {},
        inputs=inputs or {},
    )


def refuse(op: str, *, site: str, asked: Mapping[str, Any], on_offer: Any, why: str = "") -> ResolutionError:
    """Raise the shared resolution error naming the op, requested facts and supported choices."""
    detail = f" {why}" if why else ""
    return ResolutionError(
        f"{site}: op id {op!r} has no implementation for {dict(asked)!r} on this arch (on offer: {on_offer!r}).{detail}"
    )


def resolve(op: str, *, site: str, **facts: Any) -> BoundOp:
    """Resolve op on the live architecture using family-specific selection facts.
    Selection facts and recorded BoundOp.inputs have separate roles; some values appear in both."""
    family = op.split(".", 1)[0]
    package = FAMILY_PACKAGES.get(family)
    if package is None:
        raise ResolutionError(
            f"{site}: no ops family resolves op id {op!r} (families that do: {sorted(FAMILY_PACKAGES)}); "
            f"a collective is resolved by nn/_seams.py from the placement, not from an arch table"
        )
    module = importlib.import_module(f"{__package__}.{package}.resolve")
    return module.resolve(op, site=site, **facts)


__all__ = [
    "FAMILY_PACKAGES",
    "BoundOp",
    "ResolutionError",
    "declared_version",
    "integration",
    "kernel_package",
    "package_module",
    "provider_of",
    "refuse",
    "registry",
    "resolution",
    "resolve",
]
