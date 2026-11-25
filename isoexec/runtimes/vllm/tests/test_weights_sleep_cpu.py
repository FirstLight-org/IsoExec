"""CPU tests for the exact level-2 sleep: what is backed up, what is discarded, and the forward guard."""

import sys
import types

import pytest
import torch

from isoexec.runtimes.vllm import weights_sleep as ws


# ------------------------------------------------------------------------------------------ ranges
def test_merge_ranges_unions_and_drops_empty():
    assert ws.merge_ranges([(10, 20), (0, 5), (15, 30), (40, 40), (30, 35)]) == [(0, 5), (10, 35)]


def test_backup_is_the_exact_complement_of_covered_bytes():
    # Allocation [0, 100): Parameters cover [10, 30) and [50, 60); the rest must be backed up.
    got = ws.backup_ranges([(0, 100)], [(10, 30), (50, 60)], coalesce=0)
    assert got == [(0, 10), (30, 50), (60, 100)]


def test_fully_covered_allocation_backs_up_nothing_and_uncovered_one_everything():
    got = ws.backup_ranges([(0, 64), (1000, 64)], [(0, 64)], coalesce=0)
    assert got == [(1000, 1064)]


def test_ranges_never_cross_allocations_and_cover_spills_are_clipped():
    # A covered range that starts before and ends inside the second allocation; one spanning two allocations.
    got = ws.backup_ranges([(100, 100), (300, 100)], [(50, 120), (180, 320)], coalesce=0)
    assert got == [(120, 180), (320, 400)]


def test_coalescing_copies_short_covered_spans_along():
    got = ws.backup_ranges([(0, 100)], [(10, 12), (50, 90)], coalesce=8)
    # [0,10) and [12,50) merge across the 2-byte covered span; [90,100) stays separate (40-byte span).
    assert got == [(0, 50), (90, 100)]


def test_every_uncovered_byte_is_in_some_range_property():
    import random

    rng = random.Random(0)
    for _ in range(200):
        allocs, base = [], 0
        for _ in range(rng.randint(1, 5)):
            base += rng.randint(0, 50)
            size = rng.randint(1, 200)
            allocs.append((base, size))
            base += size
        covered = [(a, a + rng.randint(0, 80)) for a in (rng.randint(0, base) for _ in range(rng.randint(0, 8)))]
        coalesce = rng.choice([0, 4, 64])
        got = ws.backup_ranges(allocs, covered, coalesce=coalesce)
        cov = set()
        for lo, hi in covered:
            cov.update(range(lo, hi))
        inside = set()
        for a, s in allocs:
            inside.update(range(a, a + s))
        backed = set()
        for lo, hi in got:
            assert any(a <= lo and hi <= a + s for a, s in allocs), "a range crosses an allocation"
            backed.update(range(lo, hi))
        assert (inside - cov) <= backed
        assert backed <= inside
        if coalesce == 0:
            assert backed == inside - cov


class _FakeParam:
    def __init__(self, ptr, numel, esize=2, contiguous=True, cuda=True):
        self._ptr, self._n, self._e, self._c, self.is_cuda = ptr, numel, esize, contiguous, cuda

    def data_ptr(self):
        return self._ptr

    def numel(self):
        return self._n

    def element_size(self):
        return self._e

    def is_contiguous(self):
        return self._c


def test_parameter_ranges_count_only_contiguous_cuda_parameters():
    ps = [_FakeParam(0, 8), _FakeParam(16, 4, esize=4), _FakeParam(100, 8, contiguous=False),
          _FakeParam(200, 8, cuda=False), _FakeParam(300, 0)]
    assert ws.parameter_ranges(ps) == [(0, 32)]


# --------------------------------------------------------------------------------- buffer narrowing
def test_narrowed_named_buffers_hides_and_restores():
    m = torch.nn.Module()
    m.register_buffer("a", torch.zeros(1))
    m.register_buffer("b", torch.ones(1))
    with ws.narrowed_named_buffers(m, lambda b: False):
        assert list(m.named_buffers()) == []
        with ws.narrowed_named_buffers(m, lambda b: True):
            assert list(m.named_buffers()) == []  # composes with the enclosing narrowing
        assert list(m.named_buffers()) == []
    assert "named_buffers" not in m.__dict__
    assert [n for n, _ in m.named_buffers()] == ["a", "b"]


# ------------------------------------------------------------------------------------------- guard
def test_guard_refuses_until_restored():
    model = types.SimpleNamespace()
    ws.check_weights_live(model)
    ws.mark_discarded(model)
    try:
        assert ws.weights_discarded(model)
        with pytest.raises(ws.WeightsDiscardedError):
            ws.check_weights_live(model)
    finally:
        ws.mark_restored(model)
    assert not ws.weights_discarded(model)
    ws.check_weights_live(model)


def test_flag_selects_the_verl_sleep_level(monkeypatch):
    monkeypatch.delenv(ws.FLAG, raising=False)
    assert ws.exact_sleep_enabled()
    monkeypatch.setenv(ws.FLAG, "0")
    assert not ws.exact_sleep_enabled()


def test_flag_is_declared():
    from isoexec.core.flags import FLAGS

    flag = next(f for f in FLAGS if f.name == ws.FLAG)
    assert flag.default == "1" and flag.sides == ("engine",)


# ----------------------------------------------------------------------------- hooks on a fake vLLM
@pytest.fixture
def fake_vllm(monkeypatch):
    for name in ("vllm", "vllm.device_allocator", "vllm.v1", "vllm.v1.worker"):
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    cumem = types.ModuleType("vllm.device_allocator.cumem")

    class CuMemAllocator:
        instance = None

        def use_memory_pool(self, tag=None):  # pragma: no cover - surface probe only
            raise NotImplementedError

    cumem.CuMemAllocator = CuMemAllocator
    gw = types.ModuleType("vllm.v1.worker.gpu_worker")
    calls = []

    class Worker:
        def __init__(self, model):
            self.model_runner = types.SimpleNamespace(model=model, get_model=lambda: model)

        def sleep(self, level=1):
            calls.append(("sleep", level, [n for n, _ in self.model_runner.model.named_buffers()]))

        def wake_up(self, tags=None):
            calls.append(("wake", tags))

        def execute_model(self, scheduler_output):
            calls.append(("exec", scheduler_output.total_num_scheduled_tokens))
            return "ran"

    gw.Worker = Worker
    monkeypatch.setitem(sys.modules, "vllm.device_allocator.cumem", cumem)
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.gpu_worker", gw)
    monkeypatch.setattr(sys.modules["vllm.v1.worker"], "gpu_worker", gw, raising=False)
    monkeypatch.setattr(ws, "_installed", False)
    monkeypatch.setattr(ws, "_discarded", set())
    return types.SimpleNamespace(Worker=Worker, CuMemAllocator=CuMemAllocator, calls=calls)


def test_stock_models_keep_stock_sleep(fake_vllm, monkeypatch):
    monkeypatch.delenv(ws.FLAG, raising=False)
    assert ws.install_exact_sleep_hooks()
    m = torch.nn.Module()
    m.register_buffer("x", torch.zeros(1))
    w = fake_vllm.Worker(m)  # not a VllmModel
    w.sleep(2)
    w.sleep(1)
    w.wake_up(None)
    assert w.execute_model(types.SimpleNamespace(total_num_scheduled_tokens=3)) == "ran"
    assert fake_vllm.calls == [("sleep", 2, ["x"]), ("sleep", 1, ["x"]), ("wake", None), ("exec", 3)]


def test_disabled_flag_installs_only_the_level2_guard(fake_vllm, monkeypatch):
    monkeypatch.setenv(ws.FLAG, "0")
    assert not ws.install_exact_sleep_hooks()
    assert getattr(fake_vllm.Worker, "_ix_level2_guard", False)
    assert not getattr(fake_vllm.Worker, "_ix_exact_sleep", False)
    stock = torch.nn.Module()
    stock.register_buffer("x", torch.zeros(1))
    ws.install_exact_sleep_hooks()  # idempotent: one guard
    fake_vllm.Worker(stock).sleep(2)  # a stock model keeps stock level 2
    assert fake_vllm.calls == [("sleep", 2, ["x"])]


def test_flag_off_worker_never_runs_stock_level2_on_an_isoexec_model(fake_vllm, monkeypatch):
    # The rollout server (flag 1) asks for level 2; this worker's flag is 0: it must sleep at level 1.
    monkeypatch.setenv(ws.FLAG, "0")
    ws.install_exact_sleep_hooks()
    model = torch.nn.Module()
    model.register_buffer("x", torch.zeros(1))
    monkeypatch.setattr(ws, "_isoexec_model", lambda worker: model)
    w = fake_vllm.Worker(model)
    w.sleep(2)
    w.sleep(level=2)
    w.sleep(1)
    assert [c[:2] for c in fake_vllm.calls] == [("sleep", 1), ("sleep", 1), ("sleep", 1)]
    assert not ws.weights_discarded(model)


def test_reload_from_disk_after_level2_sleep_restores_once_every_parameter_is_written():
    from isoexec.runtimes.vllm.model import VllmModel

    m = torch.nn.Module()
    m.body = torch.nn.Linear(2, 2)
    m.head = torch.nn.Linear(2, 1, bias=False)
    note = VllmModel._note_reload_after_discard
    ws.mark_discarded(m)
    try:
        note(m, {"body.weight"})
        assert ws.weights_discarded(m)  # partial reload: still refused
        note(m, {"body.bias", "head.weight", "body.not_a_param"})
        assert not ws.weights_discarded(m)
        assert "_reload_written" not in m.__dict__
    finally:
        ws.mark_restored(m)


def test_execute_refuses_scheduled_tokens_on_discarded_weights(fake_vllm, monkeypatch):
    monkeypatch.delenv(ws.FLAG, raising=False)
    ws.install_exact_sleep_hooks()
    model = torch.nn.Module()
    monkeypatch.setattr(ws, "_isoexec_model", lambda worker: model)
    w = fake_vllm.Worker(model)
    ws.mark_discarded(model)
    # An empty step (no scheduled tokens) still runs: it reads no weights.
    assert w.execute_model(types.SimpleNamespace(total_num_scheduled_tokens=0)) == "ran"
    with pytest.raises(ws.WeightsDiscardedError):
        w.execute_model(types.SimpleNamespace(total_num_scheduled_tokens=5))
    ws.mark_restored(model)
    assert w.execute_model(types.SimpleNamespace(total_num_scheduled_tokens=5)) == "ran"


def test_surface_change_refuses_loudly(fake_vllm, monkeypatch):
    monkeypatch.delenv(ws.FLAG, raising=False)

    def sleep(self, level=1, extra=None):  # a vLLM that changed the signature
        pass

    fake_vllm.Worker.sleep = sleep
    with pytest.raises(RuntimeError, match=ws.FLAG):
        ws.install_exact_sleep_hooks()
