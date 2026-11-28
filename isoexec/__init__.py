"""Exact trainer and rollout execution, with explicit host installation.

Importing IsoExec initializes no CUDA context and imports no serving framework.
"""
from __future__ import annotations

from importlib import import_module

__version__ = "0.1.0"
__all__ = ["IsoExecConfig", "ModelSource", "ModelSourceError"]

_HOMES = {"IsoExecConfig": ".config", "ModelSource": ".compiler.source", "ModelSourceError": ".compiler.source"}

# A prebuilt cache (ISOEXEC_PREBUILT_DIR) or prebuilt-only mode (ISOEXEC_PREBUILT_ONLY=1) places the DeepGEMM JIT and
# Triton caches and installs their refusals before any of them can compile (environment only; ops/prebuilt.py).
import os as _os

if _os.environ.get("ISOEXEC_PREBUILT_DIR") or _os.environ.get("ISOEXEC_PREBUILT_ONLY") == "1":
    from .ops.prebuilt_env import activate as _activate_prebuilt

    _activate_prebuilt()


def __getattr__(name):
    home = _HOMES.get(name)
    if home is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(home, __name__), name)
    globals()[name] = value
    return value
