"""Symmetric allocations survive recorded CUDA pointers without retaining eager growth."""

from types import SimpleNamespace

import pytest

from isoexec.ops.collectives.pik import allreduce


def test_capture_growth_retains_only_recorded_pools_with_geometric_bound(monkeypatch):
    capturing = [False]
    monkeypatch.setattr(allreduce, "_capturing", lambda: capturing[0])
    monkeypatch.setattr(allreduce, "_CAPTURED_POOLS", [])
    registry = {}

    def allocate(n, device, group, dtype):
        return SimpleNamespace(cap=max(n, 16))

    def get(n):
        return allreduce._get_sym_pool(registry, "group", n, "cpu", None, None, allocate)

    eager = get(1)
    recorded = get(17)
    assert allreduce._CAPTURED_POOLS == []
    assert recorded is not eager
    capturing[0] = True
    assert get(17) is recorded
    with pytest.raises(RuntimeError, match="symmetric_pool.cap"):
        get(33)
    assert registry["group"] is recorded
    capturing[0] = False
    larger = get(33)
    assert allreduce._CAPTURED_POOLS == [recorded]
    assert larger.cap == 64
    # Repeated eager growth never accumulates retired eager-only buffers.
    for n in (65, 129, 257, 513):
        larger = get(n)
    assert allreduce._CAPTURED_POOLS == [recorded]
    capturing[0] = True
    assert get(513) is larger
    capturing[0] = False
    largest = get(1025)
    assert allreduce._CAPTURED_POOLS == [recorded, larger]
    assert sum(p.cap for p in allreduce._CAPTURED_POOLS) < largest.cap
