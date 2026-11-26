"""Deterministic absorbed MLA on SM100 (Blackwell); see README.md.

Exports load lazily and nothing here builds a kernel on import; use ``available()`` to probe.
"""

from __future__ import annotations

import importlib

_EXPORTS = {
    "mla_sm100_fwd": "fwd",
    "mla_sm100_plan": "fwd",
    "mla_sm100_fwd_selfcheck": "fwd",
    "kernel_fwd": "ops",
    "kernel_plan": "ops",
    "seg_tokens_for": "fwd",
    "segment_span": "fwd",
    "profile_digest": "fwd",
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
    try:
        value = getattr(mod, name)
    except AttributeError as exc:
        raise ImportError(f"{__name__}.{module} does not define {name}") from exc
    globals()[name] = value
    return value


def available(name: str) -> bool:
    """True iff the named export imports (the kernel module exists and loads)."""
    try:
        __getattr__(name)
        return True
    except Exception:
        return False
