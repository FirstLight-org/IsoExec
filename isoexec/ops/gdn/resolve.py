"""Resolve the GDN core and short convolution from their architecture-specific packages.
Fused ownership is site-specific: both cores own preparation, and the engine also owns convolution state.
The trainer wrapper checks the installed conv kernel at call time against the resolved identity."""

from __future__ import annotations

from typing import Any

from isoexec.ops._resolve import integration, kernel_package, provider_of, refuse, resolution


def resolve(op: str, *, site: str, **unused: Any):
    """The GDN scan or the short convolution, on the live arch."""
    if unused:
        raise refuse(op, site=site, asked=unused, on_offer=(),
                     why="neither the recurrence nor the convolution takes a caller-supplied fact.")

    if op == "gdn.core":
        package = kernel_package("gdn", op=op, site=site)
        integ = integration(package)
        from isoexec.ops.gdn.trainer import gdn_core_native_cpr

        impl = integ.IMPL
        # `_OWNED` is the package's own list of the ops its one kernel also computes. Read rather
        # than restated: a second copy of it is how a stage and its declaration part company.
        owns = {op: impl, **{owned: impl for owned in integ._OWNED}}
        return resolution(op, impl, gdn_core_native_cpr, provider=provider_of(package), site=site, owns=owns)

    if op == "conv.short":
        package = kernel_package("conv", op=op, site=site)
        integ = integration(package)
        from isoexec.ops.gdn.trainer import conv_short_native

        return resolution(op, integ.IMPL, conv_short_native, provider=provider_of(package), site=site)

    raise refuse(op, site=site, asked={}, on_offer=("gdn.core", "conv.short"),
                 why="the gdn family states these op ids and no others; l2norm and gating are OWNED by the core.")


__all__ = ["resolve"]
