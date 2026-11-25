"""Pinned, source-vendored DeepGEMM; no installed ``deep_gemm`` is imported.

IsoExec adaptation: lazy namespaced loading only. The kernel implementation is
DeepSeek's, identified per file in ``upstream_manifest.json``; see ``licenses``.
Importing this package performs no CUDA initialization, compilation, or download.
Call ``load()`` during warmup, before CUDA graph capture.
"""
from __future__ import annotations

import importlib
import sys
import threading
import types

__version__ = "2.8.2+isoexec.5081f85"
UPSTREAM_REVISION = "5081f8513af57e61696de4f4f04b0e707ffb503e"
_LOCK = threading.RLock()
_API = None
_INITIALIZING = False


def load():
    """Build/load the owned extension and return the upstream-compatible API."""
    global _API, _C, _INITIALIZING
    with _LOCK:
        if _API is None:
            from ._extension import extension

            _C = extension()
            # Relative upstream imports bind only to this private namespace.
            sys.modules[__name__ + "._C"] = _C
            for name, value in vars(_C).items():
                if isinstance(value, types.ModuleType):
                    sys.modules[__name__ + "._C." + name] = value
            _INITIALIZING = True
            try:
                _API = importlib.import_module("._api", __name__)
            finally:
                _INITIALIZING = False
        return _API


def __getattr__(name):
    # Let Python resolve relative submodule imports while _api is initializing.
    if name.startswith("_") or _INITIALIZING:
        raise AttributeError(name)
    return getattr(load(), name)
