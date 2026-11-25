"""Arch-neutral entry to the installed segment-tree ``mm`` provider (runtimes import this, never an
``*_sm90`` / ``*_sm100`` package directly)."""

from __future__ import annotations


def install_mm_segtree() -> bool:
    """Install this architecture's segment-tree GEMM as the outermost ``matmul_persistent`` provider.

    sm90: ``mm_sm90.integration.install_mm_sm90`` (gated by its recipe flag). Other architectures install
    their providers elsewhere (sm100 resolves ``mm`` to ``mm_sm100`` directly); False there.
    """
    from isoexec.ops._dispatch import current_arch

    if current_arch() != "sm90":
        return False
    from isoexec.ops.mm.mm_sm90.integration import install_mm_sm90

    return install_mm_sm90()


__all__ = ["install_mm_segtree"]
