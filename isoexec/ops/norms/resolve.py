"""Resolve norms by gamma convention and supported width without eager fallback on unsupported accelerator shapes."""

from __future__ import annotations

from typing import Any

from isoexec.core.bindings import BoundOp
from isoexec.ops._resolve import integration, kernel_package, package_module, provider_of, refuse, resolution

RMS_STAGES: tuple[str, ...] = ("norms.rms.pre_attn", "norms.rms.pre_mlp", "norms.rms.final")
RMS_CONVENTIONS: tuple[str, ...] = ("rms", "rms_zero_centered")
MLA_CONVENTIONS: tuple[str, ...] = ("rms",)
GATED_CONVENTIONS: tuple[str, ...] = ("gated_rms_zero_centered",)

#: Every q/k norm convention the family knows. Which of them a fused q/k-norm+rope package serves is
#: what that package's ``spec.py`` declares on ``norms.qk`` (``_declared_qk_conventions``).
QK_CONVENTIONS: tuple[str, ...] = ("rms_zero_centered", "rms")

#: The residual joins a fused residual + RMSNorm kernel may absorb (``residual=`` selection fact).
RESIDUAL_JOINS: tuple[str, ...] = ("residual.add.attn", "residual.add.mlp")


def _supported_width(package: str, width: int, *, op: str, site: str, recipe: str) -> None:
    """Refuse a width recipe v1 has no forward for, naming the op, the width and the recipe."""
    ops = package_module(package, "ops")
    if not ops.supported_width(int(width)):
        raise refuse(
            op,
            site=site,
            asked={"width": int(width)},
            on_offer="N % 256 == 0, N == 128, N == 384",
            why=(
                f"{recipe} has no forward for width {int(width)}; this model has no SM90 norm at this "
                f"site, and the eager torch statement is a different rounding schedule rather than a "
                f"fallback. The selected package's supported_width defines this domain."
            ),
        )


def _declared_qk_conventions(package: str, impl: str) -> tuple[str, ...]:
    """The conventions ``package``'s spec declares for its fused ``norms.qk`` impl, in spec order."""
    from isoexec.core.registry import OneOf

    for impl_spec in package_module(package, "spec").IMPLS:
        if impl_spec.op == "norms.qk" and impl_spec.impl_id == impl:
            domain = impl_spec.inputs.get("convention")
            return tuple(domain.values) if isinstance(domain, OneOf) else ()
    return ()


def _fused_qk(site: str, *, convention: str | None) -> BoundOp | None:
    """The rope seam's fused q/k-norm-plus-rope kernel, or ``None`` if this arch's seam has none."""
    from isoexec.ops._dispatch import selected

    package = selected("rope")
    if package is None:
        return None
    integ = integration(package)
    build = getattr(integ, "qk_norm_rope", None)
    impl = getattr(integ, "QK_IMPL", None)
    if build is None or impl is None:
        return None
    admitted = tuple(c for c in _declared_qk_conventions(package, impl) if c in QK_CONVENTIONS)
    if convention not in admitted:
        raise refuse(
            "norms.qk",
            site=site,
            asked={"convention": convention},
            on_offer=admitted,
            why=(
                f"ops/{package} declares its fused q/k norm for {admitted} only, and no other package on "
                f"this arch states a q/k norm -- reading gamma under a convention the kernel was not "
                f"declared (and verified) for is the +-1 checkpoint-mapping bug class, so it refuses "
                f"instead of falling back."
            ),
        )
    return resolution(
        "norms.qk",
        impl,
        build(zero_centered=convention == "rms_zero_centered"),
        provider=provider_of(package),
        site=site,
        owns={"norms.qk": impl, "rope.apply": impl},
    )


def resolve(op: str, *, site: str, convention: str | None = None, width: int | None = None,
            activation: str = "silu", residual: str | None = None, **unused: Any):
    """The norm kernel serving ``op`` at ``convention`` and ``width`` on the live arch."""
    if unused:
        raise refuse(op, site=site, asked=unused, on_offer=("convention", "width"),
                     why="the norms resolver selects on the gamma convention and the normalized width.")
    if op == "norms.qk":
        fused = _fused_qk(site, convention=convention)
        if fused is not None:
            return fused
        raise refuse(
            op, site=site, asked={"convention": convention}, on_offer=(),
            why="The selected architecture has no fused q/k norm; reference execution requires OpTable(mode='reference').",
        )

    package = kernel_package("norms", op=op, site=site)
    integ = integration(package)

    if op in RMS_STAGES or op == "norms.mla":
        admitted = RMS_CONVENTIONS if op in RMS_STAGES else MLA_CONVENTIONS
        if convention not in admitted:
            raise refuse(op, site=site, asked={"convention": convention}, on_offer=admitted)
        if width is None:
            raise refuse(op, site=site, asked={"width": None}, on_offer="an integer width",
                         why="recipe v1 admits a closed set of widths, so it cannot be selected without one.")
        _supported_width(package, width, op=op, site=site, recipe=f"ops/{package} RMS recipe v1")
        if residual is not None:
            # The fused residual join + RMSNorm: the same RMS recipe on the bf16-rounded sum, which the
            # kernel also stores. It owns the residual join it absorbs, by name.
            build = getattr(integ, "add_rms", None)
            impl = (integ.add_rms_impl(width) if hasattr(integ, "add_rms_impl")
                    else getattr(integ, "ADD_RMS_IMPL", None))
            if op not in RMS_STAGES or residual not in RESIDUAL_JOINS or build is None or impl is None:
                raise refuse(op, site=site, asked={"residual": residual},
                             on_offer=RESIDUAL_JOINS if build is not None else (),
                             why=f"ops/{package} states no fused residual join + RMSNorm for this stage; "
                                 "the caller keeps the separate add and norm (the same program).")
            fn = (build(convention == "rms_zero_centered", width=width) if hasattr(integ, "add_rms_impl")
                  else build(convention == "rms_zero_centered"))
            return resolution(
                op, impl, fn,
                provider=provider_of(package), site=site, owns={residual: impl},
                inputs={"residual": residual},
            )
        return resolution(
            op, integ.rms_impl(width) if hasattr(integ, "rms_impl") else integ.RMS_IMPL,
            (integ.rms(convention == "rms_zero_centered", width=width) if hasattr(integ, "rms_impl")
             else integ.rms(convention == "rms_zero_centered")),
            provider=provider_of(package), site=site,
        )

    if op == "norms.gated_out":
        if activation != "silu":
            raise refuse(op, site=site, asked={"activation": activation}, on_offer=("silu",),
                         why="the selected fused gated RMSNorm implements SiLU only; no silent gate substitution")
        if convention not in GATED_CONVENTIONS:
            raise refuse(op, site=site, asked={"convention": convention}, on_offer=GATED_CONVENTIONS)
        if width is None:
            raise refuse(op, site=site, asked={"width": None}, on_offer="an integer width",
                         why="recipe v1 admits a closed set of widths, so it cannot be selected without one.")
        _supported_width(package, width, op=op, site=site, recipe=f"ops/{package} gated recipe v1")
        return resolution(
            op, integ.GATED_IMPL, integ.gated(True), provider=provider_of(package), site=site,
            inputs={"activation": activation},
        )

    raise refuse(op, site=site, asked={}, on_offer=(*RMS_STAGES, "norms.qk", "norms.mla", "norms.gated_out"),
                 why="the norms family states these op ids and no others.")


__all__ = [
    "GATED_CONVENTIONS",
    "MLA_CONVENTIONS",
    "QK_CONVENTIONS",
    "RESIDUAL_JOINS",
    "RMS_CONVENTIONS",
    "RMS_STAGES",
    "resolve",
]
