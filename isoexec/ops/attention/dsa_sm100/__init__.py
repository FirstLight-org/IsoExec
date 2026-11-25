"""Deterministic DeepSeek Sparse Attention (DSA) on SM100 (Blackwell).

Three sub-ops -- the lightning indexer, the top-k selection and the sparse attention -- each with its
own recipe id (``spec.IMPL_IDS``).  Exports load lazily and nothing here builds a kernel on import;
use ``available()`` to probe.
"""

from __future__ import annotations

import importlib

_EXPORTS = {
    "dsa_topk": "topk",
    "dsa_indexer_decode": "indexer",
    "dsa_indexer_prefill": "indexer",
    "dsa_sparse_attn": "fwd",
    "dsa_sparse_attn_decode": "fwd",
    "dsa_sparse_attn_prefill": "fwd",
    "sparse_attention_sink": "sink",
    "sparse_attention_sink_forward": "sink",
    "profile_digest": "_extension",
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
