"""Select architecture-proven kernel packages for each operation group, refusing unsupported accelerators."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cache

def __getattr__(name):
    if name == "ARCH":
        from isoexec.core.arch import live_device_arch
        return live_device_arch()
    raise AttributeError(name)


def current_arch() -> str:
    """Read the current device during placement; bound forwards keep their provider."""
    import sys
    return getattr(sys.modules[__name__], "ARCH")


class ArchSelectionError(RuntimeError):
    """No implementation of a family is proven on the live arch."""


@dataclass(frozen=True)
class Family:
    """One seam's arch table.

    ``archs`` names each arch's default package. ``alternatives`` names, per arch, packages a deployment
    selects instead of that default through ``choice`` (a flag read at call time; ``None`` keeps the
    default). The flag is a declared FUNCTION flag both runtimes must agree on, so they select the same
    package; no flag can deselect a row's package for the legacy arm.
    """

    verbs: tuple[str, ...]
    archs: dict[str, str | None]
    reference_package: str
    alternatives: dict[str, dict[str, str]] = field(default_factory=dict)
    choice: Callable[[], str | None] | None = None


def _requested_gdn_kernel() -> str | None:
    from isoexec.core.gdn_kernel_env import requested_gdn_kernel

    return requested_gdn_kernel()


FAMILIES: dict[str, Family] = {
    "conv": Family(("install",), {"sm90": "gdn.conv_sm90", "sm100": "gdn.conv_sm100", "cpu": None}, "gdn.conv_sm90"),
    # SM90 GDN runs the native recurrent composition (ops/gdn/gdn_recurrent_sm90, fla_native_recurrent_v1) on the
    # trainer and the engine, with the engine state in vLLM's mamba pages; ISOEXEC_GDN_KERNEL=cpr selects the
    # chunk-parallel core with a private state pool (gdn.gdn_cpr_sm90, cuda_cpr_v1). SM100 defaults to the
    # chunk-parallel core (gdn.gdn_cpr_sm100); ISOEXEC_GDN_KERNEL=recurrent selects the native recurrent
    # composition (gdn.gdn_recurrent_sm100). docs/gdn-modes.md compares the modes.
    "gdn": Family((), {"sm90": "gdn.gdn_recurrent_sm90", "sm100": "gdn.gdn_cpr_sm100", "cpu": None}, "gdn.gdn_cpr_sm90",
                  alternatives={"sm90": {"cpr": "gdn.gdn_cpr_sm90"}, "sm100": {"recurrent": "gdn.gdn_recurrent_sm100"}},
                  choice=_requested_gdn_kernel),
    "gqa": Family((), {"sm90": "attention.gqa_sm90", "sm100": "attention.gqa_sm100", "cpu": None}, "attention.gqa_sm90"),
    "logprobs": Family((), {"sm90": "logprobs.logprobs_sm90", "sm100": "logprobs.logprobs_sm100", "cpu": None}, "logprobs.logprobs_sm90"),
    "moe": Family((), {"sm90": "moe.moe_sm90", "sm100": "moe.moe_sm100", "cpu": None}, "moe.moe_sm90"),
    "norms": Family((), {"sm90": "norms.norms_sm90", "sm100": "norms.norms_sm100", "cpu": None}, "norms.norms_sm90"),
    "rope": Family((), {"sm90": "attention.rope_sm90", "sm100": "attention.rope_sm100", "cpu": None}, "attention.rope_sm90"),
}


@cache
def _assert_proven(pkg: str, arch: str) -> None:
    """Refuse a package whose own declarations do not list ``arch``, or declare nothing at all."""
    from isoexec.core.registry import Registry
    from isoexec.core.registry_build import declare_into

    reg = declare_into(Registry(), importlib.import_module(f"{__package__}.{pkg}.spec"))
    impls = [(op, impl) for op, spec in reg.ops.items() for impl in spec.impls.values()]
    if not impls:
        raise ArchSelectionError(
            f"ops/{pkg} is selected on arch {arch!r} but its spec.py declares no impl, so nothing "
            f"states which arch its kernels were proved on; declare the impls this package owns"
        )
    unproven = sorted(f"{op}:{impl.impl_id}" for op, impl in impls if arch not in impl.supported_archs)
    if unproven:
        raise ArchSelectionError(
            f"ops/{pkg} is selected on arch {arch!r} but {unproven} declare supported_archs without it; "
            f"rerun the battery on {arch} and widen the declaration -- a proof does not travel between archs"
        )


def selected(family: str) -> str | None:
    """The ops package serving seam ``family`` on the live arch, or ``None`` without a production kernel."""
    arch = current_arch()
    fam = FAMILIES[family]
    archs = fam.archs
    if arch not in archs:
        raise ArchSelectionError(
            f"seam {family!r} has no implementation proven on arch {current_arch()!r} (rows: {sorted(archs)}); "
            f"IsoExec refuses rather than running a kernel proved on another arch"
        )
    pkg = archs[arch]
    alternatives = fam.alternatives.get(arch)
    if alternatives and fam.choice is not None:
        pkg = alternatives.get(fam.choice(), pkg)
    if pkg is not None:
        _assert_proven(pkg, arch)
    return pkg


def declarations(family: str):
    """Import declaration metadata without claiming that this CPU/GPU can execute it."""
    pkg = selected(family) or FAMILIES[family].reference_package
    return importlib.import_module(f"{__package__}.{pkg}.spec")


def backend_module(family: str, module: str, *, reference: bool = False):
    """Resolve a seam's module through the architecture table."""
    pkg = selected(family)
    if pkg is None:
        if not reference:
            raise ArchSelectionError(f"seam {family!r} has no accelerator implementation on arch {current_arch()!r}")
        pkg = FAMILIES[family].reference_package
    suffix = f".{module}" if module else ""
    return importlib.import_module(f"{__package__}.{pkg}{suffix}")


def exported(family: str, name: str, exports: dict):
    """Resolve an explicitly declared facade name without importing unrelated kernel modules."""
    if name not in exports:
        raise AttributeError(f"seam {family!r} has no public export {name!r}")
    module, symbol = exports[name]
    try:
        provider = backend_module(family, module, reference=True)
    except ModuleNotFoundError as e:
        pkg = selected(family) or FAMILIES[family].reference_package
        if e.name != f"{__package__}.{pkg}.{module}":
            raise
        raise ArchSelectionError(
            f"seam {family!r} export {name!r} is not provided by ops/{pkg}, which arch {current_arch()!r} "
            f"selects; it is another arch's recipe, so this arch binds its own implementation instead"
        ) from None
    return provider if symbol is None else getattr(provider, symbol)


def facade(family: str, exports: dict, namespace: dict):
    """Return lazy (__getattr__, __dir__) closures for a facade's declared exports."""

    def __getattr__(name):
        return exported(family, name, exports)

    def __dir__():
        return sorted(set(namespace) | set(exports))

    return __getattr__, __dir__


def family_impls(module: str, *seams: str, own: tuple = ()):
    """Build a lazy IMPLS reader combining family declarations with the listed kernel packages in order."""

    def __getattr__(name):
        if name == "IMPLS":
            impls = own
            for seam in seams:
                impls += declarations(seam).IMPLS
            return impls
        raise AttributeError(f"module {module!r} has no attribute {name!r}")

    return __getattr__


def _integration(pkg: str):
    return importlib.import_module(f"{__package__}.{pkg}.integration")


def install_arch_impls() -> dict[str, bool]:
    """``{package: active}`` for the seams applied as a module-attribute rebind, not an entry."""
    out: dict[str, bool] = {}
    for family, fam in sorted(FAMILIES.items()):
        pkg = selected(family)
        if pkg is not None and "install" in fam.verbs:
            out[pkg] = bool(_integration(pkg).install())
    return out


def resolved() -> dict[str, str]:
    """``{seam: package or "legacy"}`` -- what this process resolved, COMPARED by check 3."""
    return {family: selected(family) or "legacy" for family in sorted(FAMILIES)}


__all__ = [
    "ARCH",
    "FAMILIES",
    "ArchSelectionError",
    "Family",
    "backend_module",
    "declarations",
    "exported",
    "facade",
    "family_impls",
    "install_arch_impls",
    "resolved",
    "selected",
]
