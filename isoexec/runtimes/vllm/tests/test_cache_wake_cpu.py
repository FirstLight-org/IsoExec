"""KV-cache wake: an IsoExec model zeroes its engine cache pages right after the kv_cache pool is mapped again
(vLLM maps new physical memory there and clears it only on ROCm); and the execute guard refuses generation after
a failed weight update until a complete update succeeds."""
from types import SimpleNamespace as NS

import pytest

gpu_worker = pytest.importorskip("vllm.v1.worker.gpu_worker")

from isoexec.runtimes.vllm import weights_sleep  # noqa: E402


class _Model:
    def __init__(self):
        self.zeroed = 0

    def zero_engine_caches(self):
        self.zeroed += 1
        return 3


@pytest.fixture
def worker_cls(monkeypatch):
    calls = []

    class Worker:
        def wake_up(self, tags=None):
            calls.append(tags)

        def sleep(self, level=1):
            pass

        def execute_model(self, scheduler_output, *args, **kwargs):
            return "ran"

    monkeypatch.setattr(gpu_worker, "Worker", Worker)
    import torch
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    return Worker, calls


@pytest.mark.parametrize("tags,zeroed", [(None, 1), (["kv_cache"], 1), (["weights"], 0), (["weights", "kv_cache"], 1)])
def test_kv_wake_zeroes_the_engine_caches_once(worker_cls, monkeypatch, tags, zeroed):
    Worker, calls = worker_cls
    model = _Model()
    monkeypatch.setattr(weights_sleep, "_isoexec_model", lambda worker: model)
    assert weights_sleep.install_cache_wake_hook()
    assert weights_sleep.install_cache_wake_hook()          # idempotent: one wrapper
    Worker().wake_up(tags)
    assert calls == [tags]                                   # vLLM's own wake ran first, once
    assert model.zeroed == zeroed


def test_stock_models_are_untouched(worker_cls, monkeypatch):
    Worker, calls = worker_cls
    monkeypatch.setattr(weights_sleep, "_isoexec_model", lambda worker: None)
    weights_sleep.install_cache_wake_hook()
    Worker().wake_up(None)
    assert calls == [None]


def test_vllm_model_failure_flag_feeds_the_execute_guard():
    from isoexec.runtimes.vllm.model import VllmModel
    m = VllmModel.__new__(VllmModel)
    m._weight_update_failed = True
    assert id(m) in weights_sleep._failed and m._weight_update_failed
    m._weight_update_failed = False
    assert id(m) not in weights_sleep._failed and not m._weight_update_failed


def test_v41_rows_zero_only_mapped_cache_pages(monkeypatch):
    import torch
    from isoexec.models.deepseek_v4_1_flash import block, engine
    pages = [torch.ones(4), torch.ones(4)]
    monkeypatch.setattr(engine, "cache_tensors", lambda state: pages)
    monkeypatch.setattr(weights_sleep, "cumem_asleep", lambda t: t is pages[1])
    assert block._zero_engine_caches(object()) == 1
    assert pages[0].eq(0).all() and pages[1].eq(1).all()
    assert block._zero_engine_caches(None) == 0
