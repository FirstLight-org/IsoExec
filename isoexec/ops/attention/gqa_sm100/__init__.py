"""SM100 GQA attention: the ``sm100_gqa_segtree_v2`` forward (``csrc/``) and FlashAttention-4's SM100
backward (vendored under ``isoexec/ops/_vendor/flash_attn_cute``).

The trainer's packed varlen core and the engine's paged core run the same forward kernel; spec.py
states the recipe and the twin class, README.md the numerical program. Exports load lazily and
importing this package builds nothing; ``available()`` probes without raising.
"""

from __future__ import annotations

import importlib

_EXPORTS = {
    "gqa_sm100_fwd": "ops",
    "varlen_fwd": "ops",
    "varlen_bwd": "ops",
    "paged_fwd": "ops",
    "varlen_attention": "ops",
    "extension": "_extension",
}

__all__ = sorted(_EXPORTS) + ["available"]


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        mod = importlib.import_module(f"{__name__}.{module}")
    except ImportError as exc:
        raise ImportError(f"{__name__}.{name} is unavailable: {module}.py failed to import ({exc})") from exc
    value = getattr(mod, name)
    globals()[name] = value
    return value


def available() -> bool:
    """True iff the extension builds/loads on this process's current CUDA device (an SM100 GPU)."""
    try:
        import torch

        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
            return False
        __getattr__("extension")()
        return True
    except Exception:
        return False
