"""Private per-build implementation selection used during component setup."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from isoexec.core.bindings import BoundOp

REFERENCE_PROVIDER = "ops.reference"


class OpResolutionError(RuntimeError):
    """No implementation serves an operation's requested configuration."""


def live_arch() -> str:
    """Read the same architecture used by the family dispatchers."""
    from isoexec.ops import _dispatch

    return _dispatch.ARCH


def accelerator_provider(op: str) -> str | None:
    """Identify a production provider without constructing its kernel."""
    from isoexec.core.arch import is_accelerator_arch
    from isoexec.ops import _dispatch

    family = op.split(".", 1)[0]
    if family in _dispatch.FAMILIES:
        package = _dispatch.selected(family)
        return None if package is None else f"ops.{package}"
    return f"ops.{family}" if is_accelerator_arch(live_arch()) else None


def reference_binding(
    op: str, *, site: str, convention: str | None = None, facts: Mapping[str, Any] | None = None
) -> BoundOp:
    """Ask the owning family to select a reference for the component's facts."""
    from isoexec.ops._families import FAMILIES
    from isoexec.ops.reference import ReferencePlan, family_bindings

    prefix = op.split(".", 1)[0]
    family = next((family for family in FAMILIES if prefix in family.prefixes), None)
    if family is None:
        raise OpResolutionError(f"{site}: no reference binding for op id {op!r}")
    selectors = dict(facts or {})
    selectors.pop("convention", None)  # the dedicated convention field is validated by the norm family
    plan = ReferencePlan(
        conventions={} if convention is None else {op: convention},
        selectors={op: selectors},
    )
    try:
        bound = family_bindings(family.package, plan)
    except ValueError as exc:
        raise OpResolutionError(f"{site}: {exc}") from None
    if op not in bound:
        raise OpResolutionError(f"{site}: no reference binding for op id {op!r} (bound: {tuple(sorted(bound))})")
    return bound[op]


def _declaration_admits_convention(op: str, impl: str, convention: str, *, site: str) -> None:
    """Check that the selected kernel's declaration admits the requested convention."""
    from isoexec.core.registry import OneOf
    from isoexec.ops._resolve import registry

    reg = registry()
    if not reg.has_op(op):
        return
    spec = reg.get_op(op).impls.get(impl)
    declared = None if spec is None else spec.inputs.get("convention")
    if isinstance(declared, OneOf) and convention not in declared:
        raise OpResolutionError(
            f"{site}: {op}:{impl} was selected for norm convention {convention!r} but its declaration "
            f"states {declared!r}; the family resolver and spec.py disagree"
        )


@dataclass(frozen=True, init=False)
class OpTable:
    """Per-build selection context; resolved bindings belong to component sites.

    Production selection never falls back to reference arithmetic. Tests and
    reference execution opt in with ``mode="reference"``. The architecture is
    captured at construction and must remain unchanged while production bindings
    are built. No runtime resources or site-specific callables are cached here.
    """

    mode: str
    arch: str

    def __init__(self, *, mode: str = "production", arch: str | None = None) -> None:
        if mode not in ("production", "reference"):
            raise ValueError(f"unknown OpTable mode {mode!r}; expected production or reference")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "arch", live_arch() if arch is None else arch)

    def require_production(self, op: str, *, site: str) -> None:
        """Check production selection before a resource-dependent factory runs."""
        if self.mode != "production":
            raise OpResolutionError(f"{site}: a reference OpTable cannot select a production implementation")
        if live_arch() != self.arch:
            raise OpResolutionError(
                f"{site}: OpTable was created for {self.arch!r}, but the active architecture is {live_arch()!r}"
            )
        if accelerator_provider(op) is None:
            raise OpResolutionError(
                f"{site}: no production implementation for {op!r} on architecture {self.arch!r}; "
                "reference execution requires OpTable(mode='reference')"
            )

    def bind(
        self,
        op: str,
        *,
        site: str,
        inputs: Mapping[str, Any] | None = None,
        convention: str | None = None,
        facts: Mapping[str, Any] | None = None,
    ) -> BoundOp:
        """Bind a site's operation with its selection facts and numerical settings.

        ``facts`` select the implementation (for example, normalized width).
        ``inputs`` record settings compared by the numerical contract. A fact
        recorded under both names must agree. Family-fixed inputs take precedence
        because the implementation, rather than its caller, determines their value.
        """
        if not site:
            raise ValueError("OpTable.bind requires a non-empty component site")
        recorded = dict(inputs or {})
        selectors = dict(facts or {})
        for key in recorded.keys() & selectors.keys():
            if recorded[key] != selectors[key]:
                raise OpResolutionError(f"{site}: selection fact {key!r} disagrees with its recorded input")
        if convention is None:
            convention = selectors.get("convention")
        if convention is not None:
            for values in (recorded, selectors):
                if "convention" in values and values["convention"] != convention:
                    raise OpResolutionError(f"{site}: conflicting norm conventions in the binding request")
            recorded["convention"] = selectors["convention"] = convention
        if self.mode == "production":
            from isoexec.ops import _resolve

            self.require_production(op, site=site)
            try:
                binding = _resolve.resolve(op, site=site, **selectors)
            except _resolve.ResolutionError as exc:
                raise OpResolutionError(str(exc)) from None
            if convention is not None:
                _declaration_admits_convention(binding.op, binding.impl, convention, site=site)
        else:
            binding = reference_binding(op, site=site, convention=convention, facts=selectors)
            declared = binding.inputs.get("convention")
            if convention is not None and declared is not None and declared != convention:
                raise OpResolutionError(
                    f"{site}: {op}: binding convention {declared!r} != model convention {convention!r}"
                )
        recorded.update(binding.inputs)
        return replace(binding, inputs=recorded)


__all__ = ["OpResolutionError", "OpTable"]
