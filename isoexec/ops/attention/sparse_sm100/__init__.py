"""Deterministic sparse attention on SM100 (Blackwell).

ONE operator: a softmax-weighted sum over an ordered key list, in blocks of 64 list positions folded by one
continued chain (the tile reducer has no merge); an empty position is the reducer's neutral term. The domain
of the list is a parameter of the expression, not a different kernel:

* window only -- sliding-window attention (SWA): the last W keys (DeepSeek V4.1 ratio-0 layers);
* window ++ selected -- SWA + compressed sparse attention (CSA): the window followed by the k selected
  compressed keys, with the learned sink in the normaliser (DeepSeek V4.1 compressed layers);
* selected only -- DeepSeek sparse attention proper (DSA: V3.2 / GLM-5): the k selected keys, no sink.

Window capacity W (0 allowed), selected capacity k (0 allowed) and the sink (present or not) are launch
parameters of the leaves (``dualrail.dualrail_attn``: ``lists`` / ``extra_lists`` widths and ``sink``;
``fwd.sparse_attn`` and ``sink.sparse_attention_sink_forward``: the list width and the sink).

The selection itself (the lightning indexer's scores and the top-k) is a separate operator whose output, the
selected index list, is a logical value this package reads. Each recipe has its own id (``spec.IMPL_IDS``).
Exports load lazily and nothing here builds a kernel on import; use ``available()`` to probe.
"""

from __future__ import annotations

import importlib

_EXPORTS = {
    "sparse_attn": "fwd",
    "sparse_attn_decode": "fwd",
    "sparse_attn_prefill": "fwd",
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
