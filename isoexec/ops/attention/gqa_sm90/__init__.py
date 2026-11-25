"""SM90 GQA attention: vendored FlashAttention-3 (hopper, ``fa3/``) built for one KV split.

The trainer's packed varlen core (forward + FA3's own backward) and the engine's paged core run the
same kernels; spec.py states the build profile and the twin class. Exports load lazily and importing
this package builds nothing; ``available()`` probes without raising.
"""

from __future__ import annotations

import importlib

_EXPORTS = {
    "forward_kernel": "ops",
    "varlen_fwd": "ops",
    "varlen_bwd": "ops",
    "paged_fwd": "ops",
    "varlen_attention": "ops",
    "flash_attn_paged": "ops",
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
    """True iff the library builds/loads on this process's current CUDA device (an SM90 GPU)."""
    try:
        import torch

        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
            return False
        __getattr__("extension")()
        return True
    except Exception:
        return False
