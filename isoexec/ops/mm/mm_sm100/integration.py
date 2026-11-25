"""Install this package as the process-wide ``mm`` provider on a Blackwell device."""

from __future__ import annotations

import os

from isoexec.ops.mm import mm_batch_invariant as V
from isoexec.ops.mm import mm_runtime_seam as seam

from . import ops as mm_ops
from . import segments as seg
from .spec import IMPL_ID, RECIPE_VERSION

__all__ = [
    "install_mm_sm100",
    "mm_sm100_enabled",
    "record",
    "stats",
]

_LOG_ONCE = False
#: Stable identity of OUR installed wrapper, so installation is idempotent by object rather than by
#: a boolean a later rebind could invalidate.
_WRAPPER = None
#: An impl id proves binding; ``served > 0`` proves execution. Unsupported shapes refuse.
#: ``served_general`` stays zero; the key is retained for existing statistics readers.
_STATS = {"served": 0, "served_general": 0, "declined": 0}


def mm_sm100_enabled() -> bool:
    return os.environ.get("ISOEXEC_MM_SM100", "0") == "1"


def record() -> dict:
    """What a peer must compare to claim two processes ran the same ``mm``."""
    return {
        "impl_id": IMPL_ID,
        "recipe_version": RECIPE_VERSION,
        "segment_table_digest": seg.table_digest(),
        "segment_policy_version": seg.POLICY_VERSION,

    }


def stats() -> dict:
    return dict(_STATS)


def install_mm_sm100() -> bool:
    """Wrap the current vendored ``matmul_persistent`` with this package."""
    global _LOG_ONCE, _WRAPPER

    if not mm_sm100_enabled():
        return False

    # Idempotency by stable identity: a no-op when our exact wrapper is already outermost, and a
    # re-assert on top if something has since wrapped over us. Only ever wraps the current top, so
    # the chain cannot grow unboundedly.
    if _WRAPPER is not None and V.matmul_persistent is _WRAPPER:
        return True

    # Both of these refuse loudly rather than falling back: an enabled provider that silently did
    # not install is the split-brain the manifest handshake exists to catch.
    mm_ops.require_sm100()
    # Builds the extension and cross-checks every configuration name, A-fetch height,
    # segments-per-CTA and publish mode against segments.py in both directions, so a stale build
    # cannot be installed.
    mm_ops.config_ids()

    stock = getattr(V, "_isoexec_stock_matmul_persistent", V.matmul_persistent)

    def matmul_persistent_sm100(a, b, bias=None):
        """Serve only the recorded segment-tree recipe; unsupported operands refuse."""
        if bias is not None or not mm_ops.supported_shape(a, b):
            _STATS["declined"] += 1
            raise mm_ops.UnsupportedShape("installed SM100 segment-tree GEMM declined the operands")
        _STATS["served"] += 1
        return mm_ops.mm_sm100(a, b)

    matmul_persistent_sm100._isoexec_sm100 = True
    V._isoexec_stock_matmul_persistent = stock
    V.matmul_persistent = matmul_persistent_sm100
    _WRAPPER = matmul_persistent_sm100
    # Rebind the engine's entry points to the installed chain, so an unscoped call does not retain
    # upstream routing. The seam's proxy resolves through this module's globals at call time, so it
    # needs no knowledge of this provider.
    seam.install_vendored_mm()

    if not _LOG_ONCE:
        _LOG_ONCE = True
        rows = seg.published_table().get("rows", ())
        print(
            f"[ISOEXEC-MM-SM100] {IMPL_ID} installed: recipe {RECIPE_VERSION}, segment table "
            f"{seg.table_digest()} (policy v{seg.POLICY_VERSION}, {len(rows)} K values, "
            f"{sum(1 for r in rows if int(r[2]) > 1)} split). bf16 in and out, K % 256 == 0, "
            f"N >= 64 and a multiple of 8, no fused bias, 2-D only; unsupported shapes refuse. "
            f"Flag ISOEXEC_MM_SM100=1.",
            flush=True,
        )
    return True
