"""Engine hooks for placement-bound components: host-resident buffers, derived-weight refresh and the
graph-capturable expert exchange check."""
import mmap

import pytest
import torch
from torch import nn

from isoexec.nn.component import Component
from isoexec.runtimes.vllm import _binding
from isoexec.compiler.realization.storage import take_host_resident_buffers
from isoexec.runtimes.vllm.model import _place_host_resident_buffers


class Table(nn.Module):
    HOST_RESIDENT_BUFFERS = ("weight",)

    def __init__(self):
        super().__init__()
        self.register_buffer("weight", torch.empty((3, 5), dtype=torch.float8_e4m3fn, device="meta"))
        self.register_buffer("other", torch.empty((2,), device="meta"))


def test_host_resident_buffers_are_page_aligned_host_memory_outside_allocation(tmp_path, monkeypatch):
    from isoexec.runtimes.vllm import host_tables
    monkeypatch.setattr(host_tables, "SHM_DIR", str(tmp_path))
    monkeypatch.setattr(host_tables, "SHM_MARGIN", 0)
    root = nn.Module()
    root.table = Table()
    taken = take_host_resident_buffers(root)
    assert [(m, n) for m, n, _ in taken] == [(root.table, "weight")]
    assert root.table._buffers["weight"] is None and root.table.other.is_meta
    assert _place_host_resident_buffers([]) is None
    shared = _place_host_resident_buffers(taken)
    assert shared.leader and list(tmp_path.iterdir()) == []         # one segment per node, unlinked once mapped
    weight = root.table.weight
    assert weight.device.type == "cpu" and weight.dtype == torch.float8_e4m3fn and weight.shape == (3, 5)
    assert weight.data_ptr() % mmap.PAGESIZE == 0 and weight.is_contiguous()
    weight.view(torch.uint8).fill_(7)
    assert int(weight.view(torch.uint8).sum()) == 7 * 15
    assert dict(root.named_buffers())["table.weight"] is weight


class Row(Component):
    PLACEMENT_BOUND = True

    def __init__(self, exchange=None, stale=()):
        super().__init__()
        self.exchange, self.stale, self.refreshed = exchange, list(stale), 0

    def forward(self, x):
        return x

    def engine_expert_exchange(self):
        return self.exchange

    def refresh_derived_weights(self):
        self.refreshed += 1
        self.stale = []
        return 3

    def stale_derived_weights(self):
        return list(self.stale)


def _body(monkeypatch, rows):
    monkeypatch.setattr(_binding, "executed_modules", lambda body, prefix="": [(f"body.layers.{i}", r) for i, r in
                                                                              enumerate(rows)])
    return object()


def test_refresh_and_stale_derived_weights_visit_every_placement_bound_component(monkeypatch):
    rows = [Row(stale=["wcat"]), Row()]
    body = _body(monkeypatch, rows)
    assert _binding.stale_derived_weights(body) == ["body.layers.0: wcat"]
    assert _binding.refresh_derived_weights(body) == 6 and [r.refreshed for r in rows] == [1, 1]
    assert _binding.stale_derived_weights(body) == []


def test_graphs_refuse_a_host_sized_expert_exchange_of_a_placement_bound_component(monkeypatch):
    def host_sized(block, rows):
        return rows

    def device_sized(block, rows):
        return rows
    device_sized.graph_capturable = True
    body = _body(monkeypatch, [Row(), Row(device_sized)])
    _binding.refuse_uncapturable_expert_parallel(body, "FULL_DECODE_ONLY")
    body = _body(monkeypatch, [Row(host_sized)])
    _binding.refuse_uncapturable_expert_parallel(body, "NONE")
    with pytest.raises(ValueError, match="capturable"):
        _binding.refuse_uncapturable_expert_parallel(body, "FULL_DECODE_ONLY")


class Owner:
    """A runtime-serialized collective's execution scope (records entries)."""

    def __init__(self, log, name):
        self.log, self.name = log, name

    def execution_scope(self, *, capture=False):
        from contextlib import contextmanager

        @contextmanager
        def scope():
            self.log.append((self.name, "enter", capture))
            yield
            self.log.append((self.name, "exit", capture))
        return scope()


class ScopedRow(Row):
    def __init__(self, owners, log):
        super().__init__()
        self.owners, self.log = owners, log

    def engine_execution_scopes(self):
        return list(self.owners)

    def prepare_engine_execution(self):
        self.log.append(("prepare", id(self)))


def test_load_hook_prepares_rows_in_order_and_scopes_are_shared(monkeypatch):
    log = []
    tree = Owner(log, "tree")
    rows = [ScopedRow([tree], log), Row(), ScopedRow([tree], log)]
    body = _body(monkeypatch, rows)
    assert _binding.finish_engine_load(body) == ["body.layers.0", "body.layers.2"]
    assert log == [("prepare", id(rows[0])), ("prepare", id(rows[2]))]
    assert _binding.engine_execution_scopes(body) == [tree]


def test_model_holds_every_declared_scope():
    from isoexec.runtimes.vllm.model import VllmModel
    log = []
    model = object.__new__(VllmModel)
    model.__dict__["_execution_scopes"] = [Owner(log, "a"), Owner(log, "b")]
    with model.execution_scope(capture=True):
        log.append("body")
    assert log == [("a", "enter", True), ("b", "enter", True), "body", ("b", "exit", True), ("a", "exit", True)]
    bare = object.__new__(VllmModel)
    with bare.execution_scope():
        pass


def test_runner_holds_the_scope_around_capture_replay_and_eager_forwards():
    """capture_model: the capture lifecycle, plus a runtime scope entered right after vLLM's graph_capture
    switches streams (before the first torch.cuda.graph); execute_model: the runtime scope."""
    from contextlib import contextmanager
    from types import SimpleNamespace

    from isoexec.runtimes.vllm.execution_scope import wrap_capture, wrap_execute
    log = []

    class Model:
        def execution_scope(self, *, capture=False):
            return Owner(log, "model").execution_scope(capture=capture)

    @contextmanager
    def graph_capture(device=None):
        log.append("graph_capture")
        yield "ctx"

    capture_module = SimpleNamespace(graph_capture=graph_capture)

    def capture_model(runner):
        with capture_module.graph_capture(device=0) as ctx:
            log.append(("capture", ctx))
        return 7

    def execute_model(runner, step):
        log.append(("execute", step))
        return step

    runner = SimpleNamespace(model=Model())
    assert wrap_capture(capture_model, capture_module)(runner) == 7
    assert capture_module.graph_capture is graph_capture          # restored
    assert log == [("model", "enter", True), "graph_capture", ("model", "enter", False), ("capture", "ctx"),
                   ("model", "exit", False), ("model", "exit", True)]
    log.clear()
    assert wrap_execute(execute_model)(runner, 3) == 3
    assert log == [("model", "enter", False), ("execute", 3), ("model", "exit", False)]
    log.clear()
    assert wrap_execute(execute_model)(SimpleNamespace(model=object()), 4) == 4 and log == [("execute", 4)]


def test_tree_transport_ownership_admits_rows_inside_a_held_capture_lifecycle():
    """The real ownership rules of the tree transport: a runtime's capture lifecycle admits the rows' nested
    scopes and adopts the capture stream before capture; a scope entered during capture is refused."""
    from isoexec.ops.collectives.detreduce_sm100.phased import ExecutionOwnership
    state = {"stream": "main", "capturing": False}
    own = ExecutionOwnership(lambda: state["stream"], lambda: state["capturing"], lambda: None)
    with own.scope(capture=True):
        state["stream"] = "capture"                     # vLLM's graph_capture stream
        with own.scope():                               # entered before torch.cuda.graph begins
            assert own.capture_stream == "capture"
        state["capturing"] = True
        own.validate()                                  # a reduction recorded into the graph
        state["capturing"] = False
    state["stream"] = "main"
    with own.scope():
        own.validate()
    state["capturing"] = True
    with pytest.raises(RuntimeError, match="before CUDA capture"):
        with own.scope():
            pass


class GuardedRow(Row):
    """A placement-bound row whose per-step check is a guard of derived copies keyed on source versions."""

    def __init__(self, source):
        super().__init__()
        self.source = source
        self.guards_made = 0

    def derived_weight_guard(self):
        self.guards_made += 1
        source, recorded = self.source, [self.source._version]

        class Guard:
            def stale(self):
                return [] if source._version == recorded[0] else ["copy"]

            def refresh(self):
                recorded[0] = source._version
        guard = Guard()
        self._guard = guard
        return guard


def _counting_body(monkeypatch, rows):
    walks = []

    def executed_modules(body, prefix=""):
        walks.append(1)
        return [(f"body.layers.{i}", r) for i, r in enumerate(rows)]
    monkeypatch.setattr(_binding, "executed_modules", executed_modules)
    return object(), walks


def test_derived_weight_check_resolves_once_and_fails_closed_on_a_stale_write(monkeypatch):
    source = torch.zeros(4)
    rows = [GuardedRow(source), Row(stale=[])]
    body, walks = _counting_body(monkeypatch, rows)
    check = _binding.DerivedWeightCheck(body)
    assert len(walks) == 1
    for _ in range(5):
        assert check.stale() == []
    assert len(walks) == 1                                      # no module walk per step
    source.add_(1)                                              # a write that bypassed the commit
    assert check.stale() == ["copy"]
    rows[1].stale = ["wcat"]                                    # a row without a guard keeps its own check
    assert check.stale() == ["copy", "body.layers.1: wcat"]


def test_derived_weight_check_is_rebuilt_by_a_commit_and_after_a_placement(monkeypatch):
    from isoexec.nn import component
    source = torch.zeros(4)
    rows = [GuardedRow(source)]
    body, walks = _counting_body(monkeypatch, rows)
    check = _binding.DerivedWeightCheck(body)
    source.add_(1)
    rows[0]._guard.refresh()                                    # the commit re-derives the copies...
    check.rebuild()                                             # ...and resolves the components again
    assert check.stale() == [] and len(walks) == 2 and rows[0].guards_made == 2
    late = torch.zeros(2)
    rows.append(GuardedRow(late))                               # a component placed after the check resolved
    monkeypatch.setattr(component, "_PLACEMENTS", [component.placement_generation() + 1])
    assert check.stale() == [] and len(walks) == 3 and rows[1].guards_made == 1
    late.add_(1)
    assert check.stale() == ["copy"]                            # the new component is guarded too


def test_model_resolves_the_check_after_the_load_and_every_commit(monkeypatch):
    from isoexec.runtimes.vllm.model import VllmModel
    source = torch.zeros(4)
    rows = [GuardedRow(source)]
    body, walks = _counting_body(monkeypatch, rows)
    model = object.__new__(VllmModel)
    model.__dict__.update(body=body, _derived_check=None, _loaded_weights=None, _weight_update_failed=False)
    monkeypatch.setattr("isoexec.runtimes.vllm.model.refresh_packed_storage", lambda body: None)
    refreshed = []
    monkeypatch.setattr("isoexec.runtimes.vllm.model.refresh_derived_weights",
                        lambda body: refreshed.append(rows[0]._guard.refresh()))
    assert model.stale_derived_weights() == [] and model.stale_derived_weights() == [] and len(walks) == 1
    source.add_(1)
    assert model.stale_derived_weights() == ["copy"]            # fails closed until the commit
    model.__dict__["_loaded_weights"] = set()
    model.__dict__["named_parameters"] = lambda: iter(())
    model.finish_weight_update()
    assert refreshed and model.stale_derived_weights() == [] and len(walks) == 2


def test_moved_sample_walks_the_graph_read_parameters_round_robin(monkeypatch):
    from isoexec.models.deepseek_v4_1_flash.block import V41Layer

    class Moving(Row):
        def __init__(self, n):
            super().__init__()
            owner = nn.Module()
            for i in range(n):
                owner.register_parameter(f"p{i}", nn.Parameter(torch.zeros(2)))
            self.owner = owner
            self._graph_slots = tuple((f"owner.p{i}", owner._parameters, f"p{i}", owner._parameters[f"p{i}"],
                                       owner._parameters[f"p{i}"].data_ptr()) for i in range(n))
        moved_parameter_sample = V41Layer.moved_parameter_sample
        graph_parameter_count = V41Layer.graph_parameter_count

    rows = [Moving(5), Moving(3)]
    body, _ = _counting_body(monkeypatch, rows)
    check = _binding.DerivedWeightCheck(body)
    rows[1].owner.p2 = nn.Parameter(torch.zeros(2))                 # replaced in its owner
    with torch.no_grad():
        rows[0].owner.p4.data = torch.ones(2)                        # rebound to another storage
    seen = [check.moved_sample(2) for _ in range(5)]
    assert seen == [[], [], ["body.layers.0: owner.p4"], [], ["body.layers.1: owner.p2"]]
