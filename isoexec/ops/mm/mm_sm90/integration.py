"""Install this package as the process-wide ``mm`` provider on a Hopper device.

The wrapper sits outermost on ``mm_batch_invariant.matmul_persistent`` -- the callable the trainer's
``mm`` binding (``ops/mm/trainer.mm_trainer``), the ``aten::mm`` forward scope and the engine's rebound
batch-invariant entry points all resolve at call time -- so trainer and engine run the same kernel.
A shape the kernel cannot express (bias, non-bf16, K % 8, strides) falls through to the chain
beneath, decided by the shape alone and therefore identically on both sides; ``stats()`` counts it.
"""

from __future__ import annotations

import os

import torch

from isoexec.ops.mm import mm_batch_invariant as V
from isoexec.ops.mm import mm_runtime_seam as seam

from . import ops as mm_ops
from . import segments as seg
from .spec import FLAG, IMPL_ID, RECIPE_VERSION

__all__ = ["install_mm_sm90", "mm_sm90_enabled", "record", "stats"]

_LOG_ONCE = False
_WRAPPER = None
_STATS = {"served": 0, "declined": 0}


def mm_sm90_enabled() -> bool:
    """The segment-tree GEMM is THE sm90 ``mm`` provider; ``ISOEXEC_MM_SM90_SEGTREE=0`` exists only to
    take it out for a diagnostic A/B (both sides must then agree, like any recipe selector)."""
    return os.environ.get(FLAG, "1") == "1"


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


def installed() -> bool:
    return _WRAPPER is not None and V.matmul_persistent is _WRAPPER


def install_mm_sm90(*, force: bool = False) -> bool:
    """Wrap the current ``matmul_persistent`` with this package. Idempotent; True when installed."""
    global _LOG_ONCE, _WRAPPER

    if not (force or mm_sm90_enabled()):
        return False
    if _WRAPPER is not None and V.matmul_persistent is _WRAPPER:
        return True

    mm_ops.require_sm90()
    mm_ops.config_ids()  # builds the extension and cross-checks the compiled table both ways

    below = V.matmul_persistent
    stock = getattr(V, "_isoexec_stock_matmul_persistent", below)

    def matmul_persistent_sm90(a, b, bias=None):
        """The recorded segment-tree recipe for every shape it can express; the chain below otherwise."""
        if bias is None and mm_ops.supported_shape(a, b):
            _STATS["served"] += 1
            return mm_ops.mm_sm90(a, b)
        _STATS["declined"] += 1
        return below(a, b, bias=bias)

    matmul_persistent_sm90._isoexec_sm90_segtree = True
    V._isoexec_stock_matmul_persistent = stock
    V.matmul_persistent = matmul_persistent_sm90
    _WRAPPER = matmul_persistent_sm90
    seam.install_vendored_mm()

    from isoexec.ops.mm import mm_shapes

    mm_shapes.report_install_coverage(
        "MM_SM90_SEGTREE", lambda key: key[2] is torch.bfloat16 and seg.supported(key[0], key[1]), flag=FLAG)
    if not _LOG_ONCE:
        _LOG_ONCE = True
        print(
            f"[ISOEXEC-MM-SM90] {IMPL_ID} installed: recipe {RECIPE_VERSION}, segment table "
            f"{seg.table_digest()} (policy v{seg.POLICY_VERSION}: S = 1 for every K, K % "
            f"{seg.K_ALIGN} == 0, any N). bf16 in and out, no fused bias; "
            f"other shapes fall through to {getattr(below, '__name__', below)!s}. Flag {FLAG} (default 1).",
            flush=True,
        )
    return True
