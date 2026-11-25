"""Exact trainer and rollout execution, with explicit host installation.

Importing IsoExec initializes no CUDA context and imports no serving framework.
"""
from __future__ import annotations

from importlib import import_module

__version__ = "0.1.0"
__all__ = ["IsoExecConfig", "ModelSource", "ModelSourceError"]

_HOMES = {"IsoExecConfig": ".config", "ModelSource": ".compiler.source", "ModelSourceError": ".compiler.source"}


def __getattr__(name):
    home = _HOMES.get(name)
    if home is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(home, __name__), name)
    globals()[name] = value
    return value
