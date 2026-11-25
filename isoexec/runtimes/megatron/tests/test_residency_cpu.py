"""The colocated offload returns the driver's grown local-memory reservation."""

import pytest

from isoexec.runtimes.megatron import residency


@pytest.fixture
def limits(monkeypatch):
    state = {"stack": 1024, "released": []}
    monkeypatch.setattr(residency, "stack_limit", lambda: state["stack"])

    def release(limit, reason=""):
        state["released"].append((state["stack"], limit))
        state["stack"] = min(state["stack"], limit)
        return 0

    monkeypatch.setattr(residency, "release_local_memory", release)
    return state


def test_offload_restores_the_build_time_stack_limit(limits):
    baseline = residency.baseline()
    assert baseline == 1024
    limits["stack"] = 27936  # an autotune sweep launched a spilling config
    residency.release(baseline)
    assert limits["released"] == [(27936, 1024)]
    assert limits["stack"] == 1024


def test_release_is_skipped_without_a_cuda_runtime(monkeypatch):
    def unavailable():
        raise RuntimeError("no CUDA runtime library is loaded in this process")

    monkeypatch.setattr(residency, "stack_limit", unavailable)
    monkeypatch.setattr(residency, "release_local_memory", lambda *a, **k: pytest.fail("nothing to release to"))
    baseline = residency.baseline()
    assert baseline is None
    residency.release(baseline)
