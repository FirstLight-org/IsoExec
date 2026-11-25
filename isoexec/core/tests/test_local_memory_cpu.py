"""Autotune sweeps hand back the local memory their spilling candidates reserved."""

import pytest

from isoexec.core import local_memory


class FakeAutotuner:
    """The two Triton Autotuner entry points the guard wraps."""

    def __init__(self, state, candidates):
        self.state, self.candidates, self.fn = state, candidates, self.run
        self.cached = False

    def _bench(self, *args, config, **kwargs):
        self.state["stack"] = max(self.state["stack"], config)  # a candidate's launch grows the limit
        return [float(config)]

    def run(self, *args, **kwargs):
        if not self.cached:
            timings = {c: self._bench(config=c) for c in self.candidates}
            self.best = min(timings, key=timings.get)
            self.cached = True
        self.state["stack"] = max(self.state["stack"], self.best)  # the selected launch
        return self.best


@pytest.fixture
def guarded(monkeypatch):
    state = {"stack": 1024, "released": []}
    monkeypatch.setattr(local_memory, "stack_limit", lambda: state["stack"])

    def release(limit, reason=""):
        state["released"].append((state["stack"], limit))
        state["stack"] = min(state["stack"], limit)
        return 0

    monkeypatch.setattr(local_memory, "release_local_memory", release)
    monkeypatch.setattr(local_memory, "_GUARD_INSTALLED", False)
    fake_autotuner = type("Autotuner", (FakeAutotuner,), {})
    import sys
    import types

    module = types.ModuleType("triton.runtime.autotuner")
    module.Autotuner = fake_autotuner
    monkeypatch.setitem(sys.modules, "triton.runtime.autotuner", module)
    assert local_memory.install_autotune_guard()
    assert local_memory.install_autotune_guard()  # idempotent
    return state, fake_autotuner


def test_sweep_restores_the_pre_sweep_limit(guarded):
    state, autotuner = guarded
    tuner = autotuner(state, candidates=[600, 27936])  # the spilling candidate loses
    assert tuner.run() == 600
    assert state["released"] == [(27936, 1024)] and state["stack"] == 1024


def test_cached_selection_does_not_touch_the_limit(guarded):
    state, autotuner = guarded
    tuner = autotuner(state, candidates=[600, 27936])
    tuner.run()
    state["released"].clear()
    tuner.run()
    assert state["released"] == []


def test_subclasses_are_guarded(guarded):
    state, autotuner = guarded
    sub = type("CachedAutotuner", (autotuner,), {})
    sub(state, candidates=[27936, 40000]).run()
    assert state["released"] == [(40000, 1024)]
