"""Persist autotuned Triton configs to disk and bucket M in the tuning key.

Triton's autotuner cache is per-process, so without persistence every rank re-tunes from scratch at startup.
Above ``PIK_AUTOTUNE_M_BUCKET_FLOOR`` (default 1) M is rounded up to a power of two for config selection, so
that changing token counts share tuning results in both prefill and decode. The actual M, launch grid, and
CUDA-graph shapes remain exact; only the selected tile configuration is shared. Neither is a correctness knob:
block sizes, warps, and stages cannot perturb the K-order, so a cache miss or a mis-bucketed config costs time
at most, never a bit.
"""

from __future__ import annotations

import atexit
import json
import os
import threading

import triton

from isoexec.core.paths import pik_cache_dir

_CACHE_DIR = pik_cache_dir()
_CACHE_FILE = _CACHE_DIR / "autotune.json"

_REGISTRY: dict[str, triton.runtime.Autotuner] = {}
_LOCK = threading.Lock()
_LOADED: dict = {}
_dirty = False

# M <= floor keeps exact keys; above it M is bucketed. The default covers every positive M, with M=1
# already its own power-of-two bucket. 0 disables bucketing; an explicit older floor remains supported.
_BUCKET_FLOOR = int(os.environ.get("PIK_AUTOTUNE_M_BUCKET_FLOOR", "1"))


def _canon(key):
    """Canonicalize an Autotuner cache key by bucketing M above the floor.

    ``_AUTOTUNE_KEY`` in gemm.py fixes element 0 as M; non-conforming keys pass through untouched.
    """
    if (
        _BUCKET_FLOOR > 0
        and isinstance(key, tuple)
        and key
        and isinstance(key[0], int)
        and not isinstance(key[0], bool)
        and key[0] > _BUCKET_FLOOR
    ):
        return (1 << (key[0] - 1).bit_length(),) + key[1:]
    return key


class _BucketedCache(dict):
    """A dict that canonicalizes keys so every M in a bucket shares one tuned config.

    ``items()`` yields the canonical keys, so the disk cache converges to a finite key set instead of growing
    one entry per batch size ever seen.
    """

    __slots__ = ()

    def __contains__(self, key):
        return dict.__contains__(self, _canon(key))

    def __getitem__(self, key):
        return dict.__getitem__(self, _canon(key))

    def __setitem__(self, key, value):
        dict.__setitem__(self, _canon(key), value)


def _cfg_to_dict(c: triton.Config) -> dict:
    return {"kwargs": c.kwargs, "num_warps": c.num_warps, "num_stages": c.num_stages}


def _dict_to_cfg(d: dict) -> triton.Config:
    return triton.Config(d["kwargs"], num_warps=d["num_warps"], num_stages=d["num_stages"])


# Mirror of ``gemm.MIN_BLOCK_K``: the smallest BLOCK_K a ``tl.dot`` will take on bf16 operands, below
# which no leaf-tree kernel can be emitted at all. Duplicated rather than imported because gemm.py
# imports THIS module at its own import time (``from .autotune_cache import register``) and its
# autotune decorators call ``register`` while it is still half-initialized. A CPU test asserts the two
# constants agree, so they cannot drift.
_MIN_BLOCK_K = 16
_dropped_on_read = 0


def _cfg_legal(key, cfg: dict) -> bool:
    """Is this cached (key, config) pair one the CURRENT pruner would ever have produced?

    A tuning cache is a perf record, not a licence: it is keyed on the tuning key and replayed without
    re-running ``early_config_prune``, so a config written by an older, buggier pruner is served forever.
    ``_AUTOTUNE_KEY`` puts LEAF_K at index 2 for every leaf-tree kernel, and the kernels walk a leaf with
    ``range(LEAF_K // BLOCK_K)``; an entry whose BLOCK_K does not divide LEAF_K is therefore a zero-trip
    k-loop -- a GEMM that returns exact zeros. Drop it and let the shape re-tune. Keys of another shape
    (no int at index 2, no BLOCK_K) pass through untouched.

    The predicate is the pruner's WHOLE admission, not just the divisibility half: ``gemm.triton_leaf_ok``
    also requires the leaf to be at least ``MIN_BLOCK_K`` wide, and ``_fallback_config`` will not go below
    it. A disk entry at (LEAF_K=8, BLOCK_K=8) divides exactly and is still a config no pruner in this tree
    would emit, because ``tl.dot`` cannot take it -- so it is dropped here too rather than left to be
    caught, or not caught, further downstream.
    """
    try:
        leaf_k = key[2]
        bk = cfg["kwargs"]["BLOCK_K"]
    except Exception:  # noqa: BLE001 -- not a leaf-tree key/config; nothing to check
        return True
    if not isinstance(leaf_k, int) or isinstance(leaf_k, bool) or not isinstance(bk, int):
        return True
    return leaf_k >= _MIN_BLOCK_K and bk >= _MIN_BLOCK_K and leaf_k % bk == 0


def _legal_only(raw: dict) -> tuple[dict, int]:
    """``raw`` with every entry the current pruner would refuse removed, and how many that was."""
    out, dropped = {}, 0
    for name, entry in raw.items():
        if not isinstance(entry, dict):
            out[name] = entry
            continue
        keep = {}
        for key_s, cfg in entry.items():
            try:
                ok = _cfg_legal(tuple(json.loads(key_s)), cfg)
            except Exception:  # noqa: BLE001 -- unparseable key; leave it for the seeding loop to skip
                ok = True
            if ok:
                keep[key_s] = cfg
            else:
                dropped += 1
        out[name] = keep
    return out, dropped


def _read() -> dict:
    """The on-disk cache, with poisoned entries dropped ON READ.

    The filter lives here, at the single point the file enters the process, rather than at each consumer:
    a config the current pruner would refuse must not reach ANY reader, including one added later that
    forgets to ask. ``_dropped_on_read`` remembers that the file needed repairing so ``save`` writes the
    repair back even when this process tuned nothing new.
    """
    global _LOADED, _dropped_on_read
    if not _LOADED:
        try:
            raw = json.loads(_CACHE_FILE.read_text())
        except Exception:  # noqa: BLE001 -- a corrupt cache must never be fatal
            raw = {"_": {}}
        _LOADED, _dropped_on_read = _legal_only(raw)
        if _dropped_on_read:
            print(
                f"[ISOEXEC-PIK] autotune cache: dropped {_dropped_on_read} entry(ies) the current pruner "
                f"would refuse (BLOCK_K does not tile LEAF_K, or either is below {_MIN_BLOCK_K}) while "
                f"reading {_CACHE_FILE}. Those shapes will re-tune; the file is repaired at exit.",
                flush=True,
            )
    return _LOADED


def register(name: str, kernel):
    """Attach an autotuner to the on-disk cache, seeding it with any known winners.

    The config cache is swapped for the M-bucketing one before seeding, so exact-M disk entries collapse into
    their buckets; the first entry seen for a bucket wins, which is a perf detail rather than a correctness one.
    """
    if not isinstance(kernel, triton.runtime.Autotuner):
        return kernel
    with _LOCK:
        _REGISTRY[name] = kernel
        if not isinstance(kernel.cache, _BucketedCache):
            fresh = _BucketedCache()
            for k, v in kernel.cache.items():
                fresh[k] = v
            kernel.cache = fresh
        for key_s, cfg in _read().get(name, {}).items():
            # `_read` has already dropped everything `_cfg_legal` refuses; see there.
            try:
                key = tuple(json.loads(key_s))
                if key not in kernel.cache:  # never clobber an in-process winner
                    kernel.cache[key] = _dict_to_cfg(cfg)
            except Exception:  # noqa: BLE001
                continue
    return kernel


def save() -> None:
    """Dump every autotuner's winners. Cheap; called at exit."""
    # `_read` already dropped what the current pruner refuses, and recorded that it had to: writing this
    # back is how a file poisoned by an older pruner is repaired rather than carried forward forever.
    out = dict(_read())
    changed = _dropped_on_read > 0
    with _LOCK:
        for name, kern in _REGISTRY.items():
            entry = dict(out.get(name, {}))
            for key, cfg in kern.cache.items():
                key_s = json.dumps(list(key))
                if key_s not in entry:
                    entry[key_s] = _cfg_to_dict(cfg)
                    changed = True
            out[name] = entry
    if not changed:
        return
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = _CACHE_FILE.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(out, indent=1))
        tmp.replace(_CACHE_FILE)  # atomic: ranks tune concurrently and race here
    except Exception:  # noqa: BLE001
        pass


atexit.register(save)
