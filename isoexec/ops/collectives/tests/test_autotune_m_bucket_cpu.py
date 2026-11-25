"""CPU tests for deterministic autotuning buckets and cache reuse."""

from __future__ import annotations

import importlib
import pathlib
import sys

import pytest

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[3]))  # repo root

triton = pytest.importorskip("triton")


def _fresh_cache_module(monkeypatch, floor: str | None):
    if floor is not None:
        monkeypatch.setenv("PIK_AUTOTUNE_M_BUCKET_FLOOR", floor)
    else:
        monkeypatch.delenv("PIK_AUTOTUNE_M_BUCKET_FLOOR", raising=False)
    # a throwaway PIK_CACHE so _read() never sees a real autotune.json
    import tempfile

    monkeypatch.setenv("PIK_CACHE", tempfile.mkdtemp(prefix="pik_test_"))
    from isoexec.ops.collectives.pik import autotune_cache

    return importlib.reload(autotune_cache)


_TAIL = (2048, 640, 2, "torch.bfloat16", "torch.bfloat16", "torch.float32")


def test_exact_below_floor_bucketed_above(monkeypatch):
    ac = _fresh_cache_module(monkeypatch, "384")
    c = ac._BucketedCache()

    # 1) an explicit legacy floor still preserves exact keys below it
    c[(96,) + _TAIL] = "cfg96"
    assert (96,) + _TAIL in c
    assert (104,) + _TAIL not in c
    assert (384,) + _TAIL not in c  # floor itself is still exact

    # 2) eager-prefill territory: every M in (1024, 2048] shares one entry
    c[(1046,) + _TAIL] = "cfg2048"
    for m in (1025, 1174, 1603, 2048):
        assert (m,) + _TAIL in c, m
        assert c[(m,) + _TAIL] == "cfg2048"
    assert (1024,) + _TAIL not in c  # own bucket (2^10 maps to itself)
    assert (2049,) + _TAIL not in c  # next bucket

    # store canonicalizes too: items() yields the bucket key, not the exact M
    assert list(c.keys()) == [(96,) + _TAIL, (2048,) + _TAIL]


def test_default_buckets_small_and_large_token_counts(monkeypatch):
    ac = _fresh_cache_module(monkeypatch, None)
    c = ac._BucketedCache()
    for m in (1, 2, 3, 4, 64, 65, 66, 76, 82, 93, 99, 127, 128, 129, 384, 385, 1046):
        c[(m,) + _TAIL] = f"cfg-{1 << (m - 1).bit_length()}"
    for m in (65, 66, 76, 82, 93, 99, 127, 128):
        assert c[(m,) + _TAIL] == "cfg-128"
    assert c[(64,) + _TAIL] == "cfg-64"
    assert c[(129,) + _TAIL] == "cfg-256"
    assert list(c) == [(m,) + _TAIL for m in (1, 2, 4, 64, 128, 256, 512, 2048)]


def test_default_keeps_shape_and_dtype_parts_distinct(monkeypatch):
    ac = _fresh_cache_module(monkeypatch, None)
    c = ac._BucketedCache()
    c[(65,) + _TAIL] = "cfg"
    # N, leaf width, leaf count, and every tensor dtype remain part of the key.
    for index, value in enumerate(_TAIL):
        changed = list(_TAIL)
        changed[index] = value + 1 if isinstance(value, int) else "torch.float64"
        assert (99,) + tuple(changed) not in c


def test_real_autotuner_reuses_bucket_without_rounding_launch_arguments(monkeypatch):
    ac = _fresh_cache_module(monkeypatch, None)

    @triton.autotune(
        configs=[triton.Config({"BLOCK": 32}), triton.Config({"BLOCK": 64})],
        key=["M"],
    )
    @triton.jit
    def k(M, BLOCK: triton.language.constexpr):  # pragma: no cover - never launched
        pass

    # Exercise Triton's actual cache lookup/store and config selection on CPU;
    # replace only GPU benchmarking and the final GPU launch.
    benches, launches = [], []

    def bench(*args, config, **kwargs):
        benches.append((args[0], config.kwargs["BLOCK"]))
        return [float(config.kwargs["BLOCK"])] * 3

    monkeypatch.setattr(k, "_bench", bench)
    # Disk-cache keys query the active GPU backend; this test exercises the
    # in-memory cache even when the launcher's environment enables disk caching.
    monkeypatch.setattr(k, "cache_results", False)
    monkeypatch.setattr(k.fn, "run", lambda *args, **kwargs: launches.append((args[0], kwargs["BLOCK"])))
    ac.register("cpu_bucket_launch", k)
    try:
        shapes = (65, 66, 76, 82, 93, 99, 127, 128, 64, 129)
        for m in shapes:
            k.run(m)
        assert benches == [(m, block) for m in (65, 64, 129) for block in (32, 64)]
        assert launches == [(m, 32) for m in shapes]
        assert list(k.cache) == [(128,), (64,), (256,)]
    finally:
        ac._REGISTRY.clear()


def test_floor_zero_disables(monkeypatch):
    ac = _fresh_cache_module(monkeypatch, "0")
    c = ac._BucketedCache()
    c[(1046,) + _TAIL] = "cfg"
    assert (1046,) + _TAIL in c
    assert (1047,) + _TAIL not in c


def test_non_conforming_keys_pass_through(monkeypatch):
    ac = _fresh_cache_module(monkeypatch, "384")
    c = ac._BucketedCache()
    # a key that does not start with an int M must not be rewritten
    c[("torch.float32", 4096)] = "x"
    assert ("torch.float32", 4096) in c
    assert list(c.keys()) == [("torch.float32", 4096)]


def test_register_installs_and_seeds(monkeypatch):
    ac = _fresh_cache_module(monkeypatch, "384")

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK": 64}, num_warps=4, num_stages=2),
        ],
        key=["M"],
    )
    @triton.jit
    def k(x_ptr, M, BLOCK: triton.language.constexpr):  # pragma: no cover - never launched
        pass

    assert isinstance(k, triton.runtime.Autotuner)

    # Seed a real cached winner above the bucket floor.
    inproc_winner = triton.Config({"BLOCK": 96}, num_warps=8, num_stages=3)
    k.cache[(1046,) + _TAIL] = inproc_winner
    # ...and a legacy exact-M disk entry landing in the SAME bucket
    import json

    ac._LOADED = {
        "k": {
            json.dumps([1174, *_TAIL]): {
                "kwargs": {"BLOCK": 64},
                "num_warps": 4,
                "num_stages": 2,
            },
            json.dumps([100, *_TAIL]): {
                "kwargs": {"BLOCK": 32},
                "num_warps": 4,
                "num_stages": 2,
            },
        }
    }
    ac.register("k", k)

    assert isinstance(k.cache, ac._BucketedCache)
    # the in-process winner survived the seeding (never clobbered)
    assert k.cache[(2048,) + _TAIL] is inproc_winner
    assert k.cache[(1603,) + _TAIL] is inproc_winner
    # the sub-floor disk entry seeded at its exact key
    assert (100,) + _TAIL in k.cache
    assert (101,) + _TAIL not in k.cache
    # idempotent: re-register must not rebuild the cache object
    cache_obj = k.cache
    ac.register("k", k)
    assert k.cache is cache_obj
    # keep the module-level registry clean for the atexit save()
    ac._REGISTRY.clear()
