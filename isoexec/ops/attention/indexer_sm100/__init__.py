"""The lightning indexer and the top-k selection on SM100 (Blackwell): a separate operator.

Their output -- the selected index list of a query row -- is a logical value that the sparse attention
(``attention/sparse_sm100``) reads as the ``selected`` part of its domain; it is shared by SWA + CSA (DeepSeek V4.1)
and DSA (V3.2 / GLM-5). Sub-ops, each with its own recipe id (``sparse_sm100.spec.IMPL_IDS`` and
``deepseek_v41_cache_sm100.spec.SUB_RECIPES``): index scores (``indexer``: the FP32 tree; ``index_logits``:
DeepGEMM's SM100 MQA logits), the top-k (``topk``), and the indexer backward (``indexer_bwd``).

Moved out of ``sparse_sm100`` without any arithmetic change. Still shared with it: the constants (``sparse_sm100.spec``),
the numpy references (``sparse_sm100.reference``), the test inputs (``sparse_sm100.tests._inputs``) and one kernel,
``reduce`` of the sparse attention's backward extension, which ``indexer_bwd`` calls.
Exports load lazily and nothing here builds a kernel on import.
"""

from __future__ import annotations

import importlib

_EXPORTS = {
    "dsa_topk": "topk",
    "dsa_indexer_decode": "indexer",
    "dsa_indexer_prefill": "indexer",
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
