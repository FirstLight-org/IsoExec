"""Deterministic absorbed MLA on SM90, derived from DeepSeek FlashMLA (see DESIGN.md).
Exports load lazily; use available() to probe without raising an import error."""

from __future__ import annotations

import importlib

_EXPORTS = {
    "mla_sm90_fwd": "fwd",
    "mla_sm90_plan": "fwd",
    "mla_sm90_attention_grads": "bwd",
    "mla_sm90_attention": "ops",
    "mla_reference_fwd": "reference",
    "reference_attention": "reference",
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
