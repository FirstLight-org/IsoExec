"""Trainer GDN and short-convolution wrappers over the selected family providers.
Convolution resolves the installed module attribute at call time and verifies its identity before execution."""

from __future__ import annotations

from functools import cache

import torch


def conv_short_native(x: torch.Tensor, w: torch.Tensor, cu_seqlens) -> torch.Tensor:
    """Dispatch packed SiLU convolution after verifying the installed kernel identity.
    The selected impl is already recorded in the host handshake, so a missing install must refuse
    rather than change the kernel or retag the binding during forward."""
    from isoexec.ops.gdn import gdn_native_conv

    _refuse_uninstalled_conv(gdn_native_conv)
    # A kernel that reads a row window in place (``reads_row_windows``) takes the q|k|v columns of the
    # in_proj output as they are; any other gets a packed copy.
    if not getattr(gdn_native_conv, "reads_row_windows", False):
        x = x.contiguous()
    return gdn_native_conv(x, w.squeeze(1), None, cu_seqlens=cu_seqlens.to(torch.int32), activation="silu")


@cache
def _conv_kernel(arch: str) -> tuple[str, object] | None:
    """Return (selected package, expected conv kernel), or None without a conv package.
    Cache by architecture; the caller re-reads the installed module attribute on every invocation."""
    from isoexec.ops._dispatch import selected
    from isoexec.ops._resolve import integration

    package = selected("conv")
    if package is None:
        return None
    kernel = getattr(integration(package), "gdn_native_conv", None)
    return None if kernel is None else (package, kernel)


def _refuse_uninstalled_conv(live) -> None:
    """Refuse when ``live`` is not the kernel the selected conv package's impl id names."""
    from isoexec.ops._dispatch import ARCH

    named = _conv_kernel(ARCH)
    if named is None:
        raise RuntimeError("conv.short has no selected production convolution kernel; "
                           "reference execution requires OpTable(mode='reference')")
    if live is named[1]:
        return
    package = named[0]
    raise RuntimeError(
        f"conv.short is bound under {package}'s impl id and this process has not run "
        f"{package}.integration.install(), so the call would reach {getattr(live, '__module__', '?')}."
        f"{getattr(live, '__qualname__', live)!r} instead -- a different recipe under the kernel's "
        f"impl id, which a peer comparison reads as agreement. Call install_trainer_kernels() "
        f"(or ops/_dispatch.install_arch_impls()) before running a forward."
    )


def gdn_core_native_cpr(q, k, v, alpha, beta, A_log, dt_bias, cu_seqlens,
                        *, cu_seqlens_cpu=None) -> torch.Tensor:
    """Delegate the trainer recurrence and input-layout contract to the GDN family."""
    from isoexec.ops.gdn import gdn_core

    metadata = {} if cu_seqlens_cpu is None else {"cu_seqlens_cpu": cu_seqlens_cpu}
    return gdn_core(q, k, v, alpha, beta, A_log, dt_bias, cu_seqlens, **metadata)


__all__ = ["conv_short_native", "gdn_core_native_cpr"]
